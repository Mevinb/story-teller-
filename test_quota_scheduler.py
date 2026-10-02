import concurrent.futures
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI

from models.quota_scheduler import QuotaScheduler, QuotaDeferred
from models.groq_model import GroqModel
from models.resilient_model import ResilientModel
from models.base import LLMResponse
from pipeline.errors import PipelineCancelledError


class Clock:
    now = 100000.0
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()
        self.path = str(Path(self.tmp.name, 'quota.sqlite3'))
        self.q = QuotaScheduler(self.path, self.clock, limits={'rpm': 30, 'tpm': 1000, 'rpd': 100, 'tpd': 10000}, safety=0, interval=0)

    def reserve(self, input=100, output=200, **kwargs):
        return self.q.try_reserve('workload', 'qwen', input, output, **kwargs)

    def test_atomic_concurrent_requests_share_budget_across_instances(self):
        schedulers = [QuotaScheduler(self.path, self.clock, limits=self.q.limits, safety=0, interval=0) for _ in range(7)]
        with concurrent.futures.ThreadPoolExecutor(7) as pool:
            results = list(pool.map(lambda q: q.try_reserve('workload', 'qwen', 100, 200)[0], schedulers))
        self.assertEqual(sum(r is not None for r in results), 3)

    def test_settle_actual_frees_unused_reservation(self):
        r, _, _ = self.reserve()
        self.q.settle(r, {'prompt_tokens': 50, 'completion_tokens': 20})
        self.assertIsNotNone(self.reserve(input=400, output=400)[0])

    def test_missing_usage_and_cancellation_retain_charge(self):
        r, _, _ = self.reserve(input=400, output=400)
        self.q.settle(r)
        self.assertIsNone(self.reserve()[0])
        self.assertEqual(self.q.status()['groups'][0]['uncertain_requests'], 1)

    def test_restart_does_not_reset_cooldown(self):
        r, _, _ = self.reserve()
        self.q.settle(r, rejected=True)
        self.q.observe(r, {}, retry_after=53)
        new = QuotaScheduler(self.path, self.clock, limits=self.q.limits, safety=0, interval=0)
        _, wait, reason = new.try_reserve('workload', 'qwen', 100, 200)
        self.assertEqual((wait, reason), (53, 'provider cooldown'))

    def test_long_active_stream_keeps_full_charge(self):
        self.reserve(input=400, output=400)
        self.clock.now += 61
        self.assertIsNone(self.reserve()[0])

    def test_success_headers_debit_and_stale_response_cannot_restore(self):
        a, _, _ = self.reserve()
        self.q.observe(a, {'x-ratelimit-remaining-tokens': '500', 'x-ratelimit-reset-tokens': '1m'})
        b, _, _ = self.reserve()
        self.q.observe(a, {'x-ratelimit-remaining-tokens': '900', 'x-ratelimit-reset-tokens': '1m'})
        self.assertIsNotNone(b)
        self.assertIsNone(self.reserve()[0])
        self.assertLessEqual(self.q.status()['groups'][0]['tokens_remaining'], 200)

    def test_headers_requests_are_daily_not_per_minute(self):
        r, _, _ = self.reserve()
        self.q.observe(r, {'x-ratelimit-limit-requests': '1000', 'x-ratelimit-remaining-requests': '0', 'x-ratelimit-reset-requests': '2h4m'})
        _, wait, _ = self.reserve()
        self.assertEqual(wait, 7440)

    def test_malformed_headers_are_ignored(self):
        r, _, _ = self.reserve()
        self.q.observe(r, {'x-ratelimit-remaining-tokens': 'oops', 'x-ratelimit-reset-tokens': 'NaN'})
        self.assertIsNotNone(self.reserve()[0])

    def test_input_and_output_dimensions(self):
        self.q.limits['itpm'] = 200
        self.reserve(input=150, output=10)
        self.assertIsNone(self.reserve(input=100, output=10)[0])

    def test_oversized_request_is_explicit(self):
        with self.assertRaises(QuotaDeferred): self.reserve(input=800, output=800)

    def test_fifo_waiter_and_cancel_cleanup(self):
        a, _, _ = self.reserve(ticket='first')
        self.assertIsNotNone(a)
        self.reserve(input=800, output=100, ticket='second')
        self.assertIsNone(self.reserve(input=1, output=1, ticket='third')[0])
        def cancel(): raise PipelineCancelledError('stop')
        with self.assertRaises(PipelineCancelledError):
            self.q.acquire('workload', 'qwen', 100, 100, cancel, self.clock.sleep)
        with self.q.db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM waiters').fetchone()[0], 2)

    def test_daily_window_and_interruptible_wait(self):
        self.q.limits['rpd'] = 1
        r, _, _ = self.reserve()
        self.q.settle(r, {'prompt_tokens': 1, 'completion_tokens': 1})
        _, wait, _ = self.reserve()
        self.assertEqual(wait, 86400)
        self.clock.now += 86401
        self.assertIsNotNone(self.reserve()[0])


class GroqWireTests(SchedulerTests):
    def model(self, handler):
        q = QuotaScheduler(self.path, self.clock, limits={'tpm': 10000, 'rpm': 100, 'tpd': 100000}, safety=0, interval=0)
        model = GroqModel(api_key='test-only-key', scheduler=q)
        model._client = OpenAI(api_key='test-only-key', base_url='https://unit.invalid/v1', max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        model._client_key = 'test-only-key'
        model._sleep_interruptible = self.clock.sleep
        return model

    def test_normal_usage_and_schema_in_reservation(self):
        bodies = []
        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={'id': 'c1', 'object': 'chat.completion', 'created': 0, 'model': 'qwen', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '{"ok":true}'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 123, 'completion_tokens': 7, 'total_tokens': 130}})
        m = self.model(handler)
        result = m.generate(prompt='hello', schema={'type': 'object'}, max_tokens=100)
        self.assertEqual(result.usage, {'prompt_tokens': 123, 'completion_tokens': 7})
        self.assertIn('Schema:', str(bodies[0]['messages']))
        self.assertIn('/no_think', str(bodies[0]['messages']))
        self.assertEqual(m.scheduler.status()['groups'][0]['tokens_24h'], 130)

    def test_stream_usage_metadata_without_choices(self):
        def handler(request):
            self.assertTrue(json.loads(request.content)['stream_options']['include_usage'])
            chunks = [{'id': 's1', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'qwen', 'choices': [{'index': 0, 'delta': {'content': 'Hello world.'}, 'finish_reason': None}]},
                      {'id': 's1', 'object': 'chat.completion.chunk', 'created': 0, 'model': 'qwen', 'choices': [], 'x_groq': {'usage': {'prompt_tokens': 91, 'completion_tokens': 3, 'total_tokens': 94}}}]
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, text=''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks) + 'data: [DONE]\n\n')
        m = self.model(handler)
        self.assertEqual(''.join(m.generate_streaming(prompt='hello', max_tokens=100)), 'Hello world.')
        self.assertEqual(m.last_usage, {'prompt_tokens': 91, 'completion_tokens': 3})
        self.assertEqual(m.scheduler.status()['groups'][0]['tokens_24h'], 94)

    def test_429_one_retry_no_account_rotation(self):
        attempts = []
        def handler(request):
            attempts.append(request.headers['authorization'])
            return httpx.Response(429, headers={'retry-after': '53'}, json={'error': {'message': 'TPM rate limit', 'type': 'rate_limit_error'}})
        m = self.model(handler)
        with patch('config.GROQ_RATE_LIMIT_BUFFER', 0):
            with self.assertRaises(QuotaDeferred): m.generate(prompt='hello', max_tokens=100)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(set(attempts)), 1)
        self.assertEqual(self.clock.now, 100053)

    def test_strict_json_retry_is_admitted_and_accounted(self):
        calls = []
        def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            if 'response_format' in body:
                return httpx.Response(400, json={'error': {'message': 'json_validate_failed'}})
            return httpx.Response(200, json={'id': 'c1', 'object': 'chat.completion', 'created': 0, 'model': 'test', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '{}'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 20, 'completion_tokens': 2, 'total_tokens': 22}})
        m = self.model(handler)
        m.model = 'openai/gpt-oss-20b'
        m.generate(prompt='json', schema={'type': 'object'}, max_tokens=100)
        status = m.scheduler.status()['groups'][0]
        self.assertEqual(status['requests_24h'], 2)
        self.assertEqual(status['uncertain_requests'], 1)
        self.assertEqual(len(calls), 2)


class FakeModel:
    model = 'fake'
    provider = 'fake'
    _should_cancel = None
    last_usage = {}
    def __init__(self): self.calls = 0
    def generate(self, **kwargs):
        self.calls += 1
        return LLMResponse('saved prose', self.model, self.provider, {'prompt_tokens': 10, 'completion_tokens': 5})
    def generate_streaming(self, **kwargs):
        self.calls += 1
        yield 'saved prose'
    def get_name(self): return 'Fake (fake)'
    def generate_with_retry(self, **kwargs):
        kwargs.pop('max_retries', None)
        return self.generate(**kwargs)


class CheckpointTests(unittest.TestCase):
    def test_exact_input_replays_across_restart_and_state_change_invalidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = QuotaScheduler(str(Path(tmp, 'q.db')))
            first = FakeModel()
            a = ResilientModel(first, storage=q)
            a._cache_scope = 'state-A'
            a.generate(prompt='write scene', max_tokens=100)
            second = FakeModel()
            b = ResilientModel(second, storage=q)
            b._cache_scope = 'state-A'
            self.assertEqual(b.generate(prompt='write scene', max_tokens=100).content, 'saved prose')
            self.assertEqual(second.calls, 0)
            b._cache_scope = 'state-B'
            b.generate(prompt='write scene', max_tokens=100)
            self.assertEqual(second.calls, 1)

    def test_partial_stream_is_checkpointed_and_continued(self):
        class Broken(FakeModel):
            def generate_streaming(self, **kwargs):
                yield 'Earlier prose. '
                raise RuntimeError('connection lost')
        class Continuation(FakeModel):
            def generate_streaming(self, **kwargs):
                assert 'Earlier prose.' in kwargs['prompt']
                yield 'Later prose.'
        with tempfile.TemporaryDirectory() as tmp:
            q = QuotaScheduler(str(Path(tmp, 'q.db')))
            a = ResilientModel(Broken(), storage=q)
            a._cache_scope = 'chapter-1'
            with self.assertRaises(RuntimeError): list(a.generate_streaming(prompt='write scene'))
            b = ResilientModel(Continuation(), storage=q)
            b._cache_scope = 'chapter-1'
            self.assertEqual(''.join(b.generate_streaming(prompt='write scene')), 'Earlier prose. Later prose.')

    def test_waiting_primary_uses_backup(self):
        class Limited(FakeModel):
            def generate(self, **kwargs): raise QuotaDeferred('daily tokens', 3600)
        backup = FakeModel()
        model = ResilientModel(Limited(), backup)
        self.assertEqual(model.generate(prompt='write').content, 'saved prose')
        self.assertEqual(backup.calls, 1)

class DurableJobTests(unittest.TestCase):
    def test_lease_restart_recovers_only_uncommitted_chapters(self):
        from pipeline.job_store import JobStore
        with tempfile.TemporaryDirectory() as tmp:
            clock = Clock()
            q = QuotaScheduler(str(Path(tmp, 'q.db')), clock=clock)
            a, b = JobStore(q, clock), JobStore(q, clock)
            job = a.claim('isolated', {'chapter_count': 3, 'pacing': 'moderate'}, 1)
            self.assertEqual(job['remaining'], 3)
            self.assertIsNone(b.claim('isolated', {}, 1))
            clock.now += 41
            recovered = b.claim('isolated', {}, 2)
            self.assertEqual(recovered['remaining'], 2)
            self.assertFalse(a.heartbeat('isolated', job['owner']))
            self.assertTrue(b.heartbeat('isolated', recovered['owner']))
            clock.now += 41
            self.assertIsNone(a.claim('isolated', {}, 4))
            self.assertEqual(a.status()[0]['status'], 'done')

    def test_cancelled_job_is_never_automatically_resurrected(self):
        from pipeline.job_store import JobStore
        with tempfile.TemporaryDirectory() as tmp:
            clock = Clock()
            store = JobStore(QuotaScheduler(str(Path(tmp, 'q.db')), clock=clock), clock)
            store.claim('isolated', {'chapter_count': 1}, 0)
            store.cancel('isolated')
            clock.now += 100
            self.assertEqual(store.recoverable(), [])

    def test_retry_wait_persists(self):
        from pipeline.job_store import JobStore
        with tempfile.TemporaryDirectory() as tmp:
            clock = Clock()
            q = QuotaScheduler(str(Path(tmp, 'q.db')), clock=clock)
            store = JobStore(q, clock)
            job = store.claim('isolated', {'chapter_count': 1}, 0)
            store.finish('isolated', job['owner'], 'retry')
            self.assertEqual(JobStore(q, clock).recoverable(), [])
            clock.now += 31
            self.assertEqual(len(JobStore(q, clock).recoverable()), 1)


class RepairBudgetTests(unittest.TestCase):
    def test_writer_expansion_patch_editor_and_retry_share_one_budget(self):
        from agents.writer import SceneWriter, SceneRepairBudgetExceeded
        fake = FakeModel()
        writer = SceneWriter(fake, fake)
        with patch('config.SCENE_REPAIR_CALLS', 2):
            writer.begin_scene((1, 1))
            writer._generate_with_fallback('draft', 'write')
            writer._generate_with_fallback('expand', 'write')
            self.assertTrue(writer.reserve_edit('complete draft saved for review'))
            writer.begin_scene((1, 1))  # Graph retry must not refill allowance.
            with self.assertRaises(SceneRepairBudgetExceeded) as err:
                writer._generate_with_fallback('patch', 'write')
            self.assertEqual(err.exception.text, 'complete draft saved for review')
            self.assertEqual(fake.calls, 2)
            self.assertTrue(writer.needs_review)
            writer.begin_scene((1, 2))
            self.assertEqual(writer._call_budget, 3)

    def test_missing_sentinel_beat_and_future_kavya_arc_are_repaired_without_model(self):
        from agents.architect import StoryArchitect
        fake = FakeModel()
        state = type('State', (), {'project_dir': None})()
        architect = StoryArchitect(fake, state)
        steps = ['A second tear releases Echoes in the financial district.',
                 'Vishunu fails to save the prominent journalist.',
                 'The Sentinel secretly hunts rogue superhumans using the energy signature.']
        plan = {'chapter_title': 'Echoes', 'key_events': steps[:2],
                'character_arcs': {'Vishunu': 'loses confidence', 'Kavya': 'future revelation'}}
        repaired = architect._repair_known_violations(plan, steps, {'vishunu'}, [], 3)
        self.assertIn(steps[2], repaired['key_events'])
        self.assertNotIn('Kavya', repaired['character_arcs'])
        self.assertEqual(fake.calls, 0)

    def test_kavya_introduction_evidence_recovered_from_saved_chapter(self):
        from agents.architect import StoryArchitect
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'chapters').mkdir()
            Path(tmp, 'chapters/chapter_002.md').write_text('Kavya introduced herself as a physicist.')
            state = type('State', (), {'project_dir': tmp})()
            architect = StoryArchitect(FakeModel(), state)
            plan = {'key_events': [], 'character_arcs': {'Kavya': 'helps the wounded hero'}}
            repaired = architect._repair_known_violations(plan, [], {'vishunu'}, [], 3)
            self.assertIn('Kavya', repaired['character_arcs'])

class CircuitBreakerTests(unittest.TestCase):
    def test_failing_support_is_skipped_for_following_tasks(self):
        class Unavailable(Exception): code = 503
        class Support(FakeModel):
            def generate(self, **kwargs):
                self.calls += 1
                raise Unavailable()
            def get_name(self): return 'Gemini (fake)'
        with tempfile.TemporaryDirectory() as tmp:
            q = QuotaScheduler(str(Path(tmp, 'q.db')))
            support, backup = Support(), FakeModel()
            m = ResilientModel(support, backup, storage=q)
            m.generate(prompt='planning')
            m.generate(prompt='review')
            self.assertEqual(support.calls, 1)
            self.assertEqual(backup.calls, 2)

    def test_google_support_has_one_sdk_attempt_and_finite_timeout(self):
        from models.resilient_model import _support_instance
        with patch('models.gemini_model.genai.Client') as constructor:
            model = _support_instance('gemini-2.5-flash')
            model._custom_api_key = 'unit-only'
            _ = model.client
            options = constructor.call_args.kwargs['http_options']
            self.assertEqual(options.timeout, 15000)
            self.assertEqual(options.retry_options.attempts, 1)

    def test_underestimated_input_calibrates_next_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = QuotaScheduler(str(Path(tmp, 'q.db')), limits={'tpm': 10000}, safety=0, interval=0)
            messages = [{'role': 'user', 'content': 'long content'}]
            before = q.estimate_for('group', 'model', messages)
            r, _, _ = q.try_reserve('group', 'model', before, 100)
            q.settle(r, {'prompt_tokens': before * 2, 'completion_tokens': 10})
            self.assertGreater(q.estimate_for('group', 'model', messages), before * 2)

    def test_never_dispatched_reservation_is_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = QuotaScheduler(str(Path(tmp, 'q.db')), limits={'tpm': 1000}, safety=0, interval=0)
            r, _, _ = q.try_reserve('group', 'model', 500, 400)
            q.release_unsent(r)
            self.assertIsNotNone(q.try_reserve('group', 'model', 500, 400)[0])
