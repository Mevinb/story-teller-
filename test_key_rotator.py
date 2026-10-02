"""
Unit and integration tests for the multi-account key rotation system.
Covers AccountKey, KeyRotator, GroqKeyManager, GeminiKeyManager, OpenRouterKeyManager,
and GroqModel shared workload quotas across credential rotations.
"""
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from openai import RateLimitError
import httpx

import config
from models.base import QuotaExhaustedError
from models.key_rotator import AccountKey, KeyRotator, KeyStatus
from models.groq_model import GroqKeyManager, GroqModel
from models.gemini_model import GeminiKeyManager, GeminiModel
from models.openrouter_model import OpenRouterKeyManager, OpenRouterModel


class AccountKeyTests(unittest.TestCase):
    def test_account_key_initialization_and_masking(self):
        acc = AccountKey(key="gsk_1234567890abcdef", index=0, provider="groq")
        self.assertEqual(acc.key, "gsk_1234567890abcdef")
        self.assertEqual(acc.index, 0)
        self.assertEqual(acc.status, KeyStatus.HEALTHY)
        self.assertTrue(acc.masked_key.startswith("gsk_1"))
        self.assertTrue(acc.masked_key.endswith("cdef"))
        self.assertIn("Groq Account #1", acc.account_id)

    def test_short_key_masking(self):
        acc = AccountKey(key="abc", index=1, provider="gemini")
        self.assertEqual(acc.masked_key, "***abc")

    def test_rate_limit_and_cooldown_expiry(self):
        acc = AccountKey(key="key_1", index=0, provider="groq")
        now = 1000.0
        acc.record_rate_limit(retry_after=10.0, is_daily=False, now=now)
        self.assertEqual(acc.status, KeyStatus.COOLDOWN)
        self.assertFalse(acc.is_available(now=1005.0))
        self.assertAlmostEqual(acc.seconds_until_available(now=1005.0), 5.0)

        # After cooldown expiry
        self.assertTrue(acc.is_available(now=1011.0))
        self.assertEqual(acc.status, KeyStatus.HEALTHY)
        self.assertEqual(acc.seconds_until_available(now=1011.0), 0.0)

    def test_daily_exhaustion_expiry(self):
        acc = AccountKey(key="key_1", index=0, provider="groq")
        now = 1000.0
        acc.record_rate_limit(retry_after=3600.0, is_daily=True, now=now)
        self.assertEqual(acc.status, KeyStatus.EXHAUSTED)
        self.assertFalse(acc.is_available(now=2000.0))

        # After daily reset
        self.assertTrue(acc.is_available(now=5000.0))
        self.assertEqual(acc.status, KeyStatus.HEALTHY)

    def test_auth_error_permanently_disables_key(self):
        acc = AccountKey(key="bad_key", index=0, provider="groq")
        acc.record_error(is_auth_error=True)
        self.assertEqual(acc.status, KeyStatus.DISABLED)
        self.assertFalse(acc.is_available(now=999999.0))
        self.assertEqual(acc.seconds_until_available(now=999999.0), float("inf"))

    def test_tpm_budget_tracking(self):
        acc = AccountKey(key="key_1", index=0, provider="groq")
        now = 100.0
        # Check budget for 2000 tokens against 6000 limit with 10% safety margin (budget = 5400)
        has_budget, wait, res = acc.check_tpm_budget(
            model="test-model", estimated_tokens=2000, limit=6000, window_seconds=60.0, safety_margin=0.1, now=now
        )
        self.assertTrue(has_budget)
        self.assertEqual(wait, 0.0)
        self.assertIsNotNone(res)

        # Another 2000 tokens fits
        has_budget2, _, res2 = acc.check_tpm_budget(
            model="test-model", estimated_tokens=2000, limit=6000, window_seconds=60.0, safety_margin=0.1, now=now
        )
        self.assertTrue(has_budget2)

        # Another 2000 tokens exceeds 5400 budget (total 6000 > 5400)
        has_budget3, wait3, res3 = acc.check_tpm_budget(
            model="test-model", estimated_tokens=2000, limit=6000, window_seconds=60.0, safety_margin=0.1, now=now
        )
        self.assertFalse(has_budget3)
        self.assertGreater(wait3, 0.0)
        self.assertIsNone(res3)

        # Settle first reservation with actual tokens (e.g. 500)
        acc.settle_tpm(res, 500)
        # Now current sum is 500 + 2000 = 2500, so another 2000 will fit!
        has_budget4, _, _ = acc.check_tpm_budget(
            model="test-model", estimated_tokens=2000, limit=6000, window_seconds=60.0, safety_margin=0.1, now=now
        )
        self.assertTrue(has_budget4)


class KeyRotatorTests(unittest.TestCase):
    def test_rotator_initialization_and_deduplication(self):
        rotator = KeyRotator("groq", "key1, key2, key1, key3 , key2")
        self.assertEqual(rotator.get_keys(), ["key1", "key2", "key3"])
        self.assertEqual(len(rotator._accounts), 3)

    def test_round_robin_rotation(self):
        rotator = KeyRotator("groq", ["acc_1", "acc_2", "acc_3"], strategy="round_robin")
        self.assertEqual(rotator.get_key(), "acc_1")
        self.assertTrue(rotator.rotate())
        self.assertEqual(rotator.get_key(), "acc_2")
        self.assertTrue(rotator.rotate())
        self.assertEqual(rotator.get_key(), "acc_3")
        self.assertTrue(rotator.rotate())
        self.assertEqual(rotator.get_key(), "acc_1")

    def test_least_recently_used_strategy(self):
        rotator = KeyRotator("groq", ["acc_1", "acc_2", "acc_3"], strategy="least_recently_used")
        acc1 = rotator.get_account_by_key("acc_1")
        acc2 = rotator.get_account_by_key("acc_2")
        acc3 = rotator.get_account_by_key("acc_3")

        acc1.last_used = 100.0
        acc2.last_used = 50.0   # Rested longest
        acc3.last_used = 120.0

        self.assertEqual(rotator.get_key(), "acc_2")

    def test_least_loaded_strategy(self):
        rotator = KeyRotator("groq", ["acc_1", "acc_2", "acc_3"], strategy="least_loaded")
        acc1 = rotator.get_account_by_key("acc_1")
        acc2 = rotator.get_account_by_key("acc_2")
        acc3 = rotator.get_account_by_key("acc_3")

        acc1.tokens_used = 5000
        acc2.tokens_used = 12000
        acc3.tokens_used = 1500  # Lowest usage

        self.assertEqual(rotator.get_key(), "acc_3")

    def test_immediate_failover_on_rate_limit(self):
        rotator = KeyRotator("groq", ["acc_1", "acc_2", "acc_3"])
        self.assertEqual(rotator.get_key(), "acc_1")

        # acc_1 hits 429 rate limit
        has_alternate, min_wait, next_acc = rotator.mark_rate_limited("acc_1", retry_after=30.0)
        self.assertTrue(has_alternate)
        self.assertEqual(min_wait, 0.0)
        self.assertIsNotNone(next_acc)
        self.assertEqual(next_acc.key, "acc_2")
        self.assertEqual(rotator.get_key(), "acc_2")

    def test_all_accounts_cooling_down_reports_min_wait(self):
        rotator = KeyRotator("groq", ["acc_1", "acc_2"])
        # acc_1 gets 60s cooldown
        rotator.mark_rate_limited("acc_1", retry_after=60.0)
        # acc_2 gets 15s cooldown
        has_alternate, min_wait, next_acc = rotator.mark_rate_limited("acc_2", retry_after=15.0)

        self.assertFalse(has_alternate)
        self.assertIsNone(next_acc)
        self.assertLessEqual(min_wait, 15.0)
        self.assertGreater(min_wait, 0.0)

    def test_all_accounts_daily_exhausted(self):
        rotator = KeyRotator("groq", ["acc_1", "acc_2"])
        rotator.mark_rate_limited("acc_1", retry_after=3600.0, is_daily=True)
        rotator.mark_rate_limited("acc_2", retry_after=1800.0, is_daily=True)

        self.assertTrue(rotator.are_all_daily_exhausted())

    def test_set_keys_preserves_metrics(self):
        rotator = KeyRotator("groq", ["acc_1", "acc_2"])
        rotator.mark_success("acc_1", tokens=2500)
        acc1 = rotator.get_account_by_key("acc_1")
        self.assertEqual(acc1.tokens_used, 2500)
        self.assertEqual(acc1.successful_requests, 1)

        # Update key pool with a new key and preserved old key
        rotator.set_keys(["acc_1", "acc_3"])
        self.assertEqual(rotator.get_keys(), ["acc_1", "acc_3"])
        acc1_after = rotator.get_account_by_key("acc_1")
        self.assertEqual(acc1_after.tokens_used, 2500)
        self.assertEqual(acc1_after.successful_requests, 1)

    def test_thread_safety_concurrent_access(self):
        rotator = KeyRotator("groq", [f"key_{i}" for i in range(10)])
        errors = []

        def worker(w_id):
            try:
                for _ in range(50):
                    k = rotator.get_key(proactive_rotate=True)
                    rotator.mark_success(k, tokens=50)
                    if w_id % 3 == 0:
                        rotator.rotate()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        summary = rotator.status_summary()
        self.assertEqual(summary["total_accounts"], 10)
        total_success = sum(acc["successful_requests"] for acc in summary["accounts"])
        self.assertEqual(total_success, 400)


class ManagerIntegrationTests(unittest.TestCase):
    def test_groq_key_manager_methods(self):
        GroqKeyManager.sync_keys(["gsk_test1", "gsk_test2"])
        self.assertIn("gsk_test1", GroqKeyManager.get_keys())
        self.assertEqual(GroqKeyManager.get_key(), "gsk_test1")
        self.assertTrue(GroqKeyManager.rotate())
        self.assertEqual(GroqKeyManager.get_key(), "gsk_test2")

        summary = GroqKeyManager.status_summary()
        self.assertEqual(summary["provider"], "groq")
        self.assertEqual(summary["total_accounts"], 2)

    def test_gemini_key_manager_methods(self):
        GeminiKeyManager.sync_keys(["AIza_test1", "AIza_test2"])
        self.assertEqual(GeminiKeyManager.get_key(), "AIza_test1")
        self.assertTrue(GeminiKeyManager.rotate())
        self.assertEqual(GeminiKeyManager.get_key(), "AIza_test2")

    def test_openrouter_key_manager_methods(self):
        OpenRouterKeyManager.sync_keys(["sk-or-test1", "sk-or-test2"])
        self.assertEqual(OpenRouterKeyManager.get_key(), "sk-or-test1")
        self.assertTrue(OpenRouterKeyManager.rotate())
        self.assertEqual(OpenRouterKeyManager.get_key(), "sk-or-test2")


class GroqSharedWorkloadTests(unittest.TestCase):
    def test_manual_credential_rotation_does_not_change_quota_group(self):
        import tempfile
        from models.quota_scheduler import QuotaScheduler
        old_rotator = GroqKeyManager.rotator
        old_keys, old_key = config.GROQ_API_KEYS, config.GROQ_API_KEY
        try:
            GroqKeyManager.rotator = KeyRotator("groq", ["test-A", "test-B"])
            GroqKeyManager.sync_keys(["test-A", "test-B"])
            with tempfile.TemporaryDirectory() as tmp:
                q = QuotaScheduler(tmp + "/q.db", limits={"tpm": 1000}, safety=0, interval=0)
                model = GroqModel(scheduler=q)
                self.assertEqual(model.api_key, "test-A")
                q.try_reserve(config.GROQ_QUOTA_GROUP, model.model, 400, 400)
                GroqKeyManager.rotate()
                self.assertEqual(model.api_key, "test-B")
                self.assertIsNone(q.try_reserve(config.GROQ_QUOTA_GROUP, model.model, 400, 400)[0])
        finally:
            GroqKeyManager.rotator = old_rotator
            config.GROQ_API_KEYS, config.GROQ_API_KEY = old_keys, old_key


class FlaskKeyEndpointTests(unittest.TestCase):
    def setUp(self):
        from app import create_app
        self.app = create_app()
        self.client = self.app.test_client()

    def test_get_keys_status_endpoint(self):
        GroqKeyManager.sync_keys(["gsk_flask_1", "gsk_flask_2"])
        GeminiKeyManager.sync_keys(["AIza_flask_1"])
        OpenRouterKeyManager.sync_keys(["sk-or-flask_1"])

        res = self.client.get("/api/keys/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("groq", data)
        self.assertIn("gemini", data)
        self.assertIn("openrouter", data)
        self.assertEqual(data["groq"]["total_accounts"], 2)
        self.assertEqual(data["gemini"]["total_accounts"], 1)
        self.assertEqual(data["openrouter"]["total_accounts"], 1)

    def test_post_rotate_endpoint(self):
        GroqKeyManager.sync_keys(["gsk_flask_1", "gsk_flask_2"])
        initial_key = GroqKeyManager.get_key()

        res = self.client.post("/api/keys/rotate/groq")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["rotated"])
        self.assertNotEqual(GroqKeyManager.get_key(), initial_key)

    def test_post_rotate_invalid_provider_returns_400(self):
        res = self.client.post("/api/keys/rotate/unknown_provider")
        self.assertEqual(res.status_code, 400)


class IntelligentSwappingTests(unittest.TestCase):
    def test_least_loaded_headroom_prioritization(self):
        rotator = KeyRotator("groq", ["key_1", "key_2", "key_3"], strategy="least_loaded")
        acc1 = rotator.get_account_by_key("key_1")
        acc2 = rotator.get_account_by_key("key_2")
        acc3 = rotator.get_account_by_key("key_3")

        now = time.time()
        # Acc1 used 4000 tokens recently, Acc2 used 1000 tokens, Acc3 used 0
        acc1.check_tpm_budget("qwen", 4000, 6000, 60.0, 0.0, now)
        acc2.check_tpm_budget("qwen", 1000, 6000, 60.0, 0.0, now)

        # Acc3 has most headroom (0 window tokens)
        picked = rotator.get_active_account(model="qwen")
        self.assertEqual(picked.key, "key_3")

        # After Acc3 uses 2500 tokens, Acc2 has the most headroom (1000 < 2500 < 4000)
        acc3.check_tpm_budget("qwen", 2500, 6000, 60.0, 0.0, now)
        picked2 = rotator.get_active_account(model="qwen")
        self.assertEqual(picked2.key, "key_2")

    def test_headroom_estimation_swapping(self):
        rotator = KeyRotator("groq", ["key_low_headroom", "key_high_headroom"], strategy="least_loaded")
        acc1 = rotator.get_account_by_key("key_low_headroom")
        acc2 = rotator.get_account_by_key("key_high_headroom")

        now = time.time()
        # Limit is 6000, 20% safety margin -> budget is 4800
        # Acc1 already used 4000 tokens -> remaining headroom is 800 tokens
        acc1.check_tpm_budget("qwen", 4000, 6000, 60.0, 0.20, now)
        # Acc2 used 0 -> remaining headroom is 4800 tokens

        # If incoming request needs 1500 tokens, Acc1 does not have enough headroom, so Acc2 must be picked
        picked = rotator.get_active_account(model="qwen", estimated_tokens=1500)
        self.assertEqual(picked.key, "key_high_headroom")

    def test_groq_model_immediate_429_failover_to_alternate_key(self):
        import tempfile
        import json
        from models.quota_scheduler import QuotaScheduler
        from openai import OpenAI

        old_rotator = GroqKeyManager.rotator
        old_keys, old_key = config.GROQ_API_KEYS, config.GROQ_API_KEY
        try:
            GroqKeyManager.rotator = KeyRotator("groq", ["gsk_fail", "gsk_success"])
            GroqKeyManager.sync_keys(["gsk_fail", "gsk_success"])

            used_keys = []
            def handler(request):
                auth = request.headers.get("authorization", "")
                used_keys.append(auth)
                if "gsk_fail" in auth:
                    return httpx.Response(
                        429,
                        headers={"retry-after": "30"},
                        json={"error": {"message": "Rate limit reached", "type": "rate_limit_error"}},
                    )
                return httpx.Response(
                    200,
                    json={
                        "id": "c1",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "qwen",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "swapped successfully"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                )

            with tempfile.TemporaryDirectory() as tmp:
                q = QuotaScheduler(tmp + "/q.db", limits={"tpm": 10000, "rpm": 100}, safety=0, interval=0)
                model = GroqModel(scheduler=q)
                # Client factory will use mock http client
                client_fail = OpenAI(api_key="gsk_fail", base_url="https://mock.groq/v1", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
                client_succ = OpenAI(api_key="gsk_success", base_url="https://mock.groq/v1", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
                def mock_client_for(k):
                    return client_fail if k == "gsk_fail" else client_succ
                model._client_for = mock_client_for

                res = model.generate(prompt="test prompt", max_tokens=50)
                self.assertEqual(res.content, "swapped successfully")
                # Both keys were tried: fail first, then immediately success
                self.assertTrue(any("gsk_fail" in k for k in used_keys))
                self.assertTrue(any("gsk_success" in k for k in used_keys))
                # gsk_fail should now be in cooldown
                fail_acc = GroqKeyManager.rotator.get_account_by_key("gsk_fail")
                self.assertEqual(fail_acc.status, KeyStatus.COOLDOWN)
        finally:
            GroqKeyManager.rotator = old_rotator
            config.GROQ_API_KEYS, config.GROQ_API_KEY = old_keys, old_key


if __name__ == "__main__":
    unittest.main()
