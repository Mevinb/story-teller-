"""Exact-input checkpoints and bounded, visible provider fallback."""
import hashlib
import json
import logging
import threading
import time

from pipeline.errors import PipelineCancelledError
from .base import LLMInterface, LLMResponse, QuotaExhaustedError, ContentBlockedError
from .quota_scheduler import QuotaScheduler, QuotaDeferred
import config

logger = logging.getLogger(__name__)

_probe_lock = threading.Lock()
_support = None
_support_checked = 0

def _support_instance(name):
    from .gemini_model import GeminiModel
    model = GeminiModel(model=name)
    model._fast_support = True
    return model


def verified_support():
    global _support, _support_checked
    if config.GROQ_SUPPORT_BACKEND != 'gemini' or not config.GEMINI_API_KEY:
        return None
    with _probe_lock:
        if time.time() - _support_checked < 300:
            return _support_instance(_support) if _support else None
        model = _support_instance(config.GEMINI_MODEL)
        try:
            response = model.generate(prompt='Return a JSON object with ok set to true.',
                                      schema={'type': 'object', 'properties': {'ok': {'type': 'boolean'}}, 'required': ['ok']},
                                      max_tokens=160, temperature=0)
            _support = model.model if response.as_json() == {'ok': True} else None
        except Exception:
            _support = None
        _support_checked = time.time()
        # Use a separate model instance per project so cancellation hooks do not collide.
        return _support_instance(_support) if _support else None


class ResilientModel(LLMInterface):
    def __init__(self, primary, backup=None, storage=None):
        self.primary, self.backup = primary, backup
        self.model = primary.model
        self.provider = getattr(primary, 'provider', primary.get_name().split(' ')[0].lower())
        self.storage = storage
        self._cache_scope = None
        self._quota_callback = None
        self.last_usage = {}

    def _hooks(self):
        for model in (self.primary, self.backup):
            if model is not None:
                model._should_cancel = self._should_cancel
                model._quota_callback = self._quota_callback
        self.primary._max_quota_wait = config.GROQ_BACKUP_WAIT_SECONDS if self.backup else None

    def _key(self, kwargs, stream):
        if not self._cache_scope: return None
        raw = json.dumps([self._cache_scope, self.primary.get_name(), kwargs, stream], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _db(self):
        if self.storage is None: self.storage = QuotaScheduler()
        return self.storage.db()

    def _read(self, key):
        if not key: return None
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS task_results (id TEXT PRIMARY KEY, data TEXT, updated REAL)')
            row = db.execute('SELECT data FROM task_results WHERE id=?', (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def _write(self, key, content, provider, usage, complete):
        if not key: return
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS task_results (id TEXT PRIMARY KEY, data TEXT, updated REAL)')
            db.execute('INSERT OR REPLACE INTO task_results VALUES (?,?,?)', (key, json.dumps(
                {'content': content, 'provider': provider, 'usage': usage, 'complete': complete}), time.time()))
            db.execute('DELETE FROM task_results WHERE updated<?', (time.time() - 7 * 86400,))

    def _cooldown_key(self, model):
        return model.get_name()

    def _check_health(self, model):
        # QuotaScheduler already owns Groq cooldowns. This circuit protects
        # the supporting provider from repeated 503/429 probes by every agent.
        if 'Groq' in model.get_name(): return
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS provider_health (id TEXT PRIMARY KEY, until REAL)')
            row = db.execute('SELECT until FROM provider_health WHERE id=?', (self._cooldown_key(model),)).fetchone()
        if row and row[0] > time.time():
            raise QuotaDeferred('support provider cooldown', row[0] - time.time())

    def _record_failure(self, model, error):
        code = getattr(error, 'status_code', None) or getattr(error, 'code', None)
        if isinstance(error, QuotaExhaustedError) or 'Groq' in model.get_name(): return
        if code not in (429, 500, 502, 503, 504) and not isinstance(error, (TimeoutError, ConnectionError)):
            return
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS provider_health (id TEXT PRIMARY KEY, until REAL)')
            db.execute('INSERT OR REPLACE INTO provider_health VALUES (?,?)', (self._cooldown_key(model), time.time() + 60))

    def _notify(self, phase, model, error=None):
        if self._quota_callback:
            details = {'model': model.model, 'provider': model.get_name()}
            if error:
                details['error_type'] = type(error).__name__
                details['status_code'] = getattr(error, 'status_code', None) or getattr(error, 'code', None)
            self._quota_callback(phase, details)

    def generate(self, **kwargs):
        self._hooks()
        self._raise_if_cancelled()
        key = self._key(kwargs, False)
        saved = self._read(key)
        if saved and saved['complete']:
            self.provider = saved['provider']
            self._notify('checkpoint', self.primary)
            return LLMResponse(saved['content'], self.model, saved['provider'], saved['usage'])
        try:
            self._check_health(self.primary)
            result = self.primary.generate(**kwargs)
            if not result.content.strip():
                raise RuntimeError("Empty model response")
        except (PipelineCancelledError, ContentBlockedError):
            raise
        except Exception as error:
            self._record_failure(self.primary, error)
            if not self.backup:
                logger.error("Primary model %s failed and no backup configured: %s", getattr(self.primary, "model", "unknown"), error)
                raise
            logger.warning(
                "Primary model %s failed (%s). Attempting backup model %s...",
                getattr(self.primary, "model", "unknown"),
                error,
                getattr(self.backup, "model", "unknown"),
            )
            self._raise_if_cancelled()
            self._notify('backup', self.backup, error)
            try:
                self._check_health(self.backup)
                result = self.backup.generate(**kwargs)
            except PipelineCancelledError:
                raise
            except Exception as backup_error:
                logger.error(
                    "Backup model %s also failed: %s",
                    getattr(self.backup, "model", "unknown"),
                    backup_error,
                )
                self._record_failure(self.backup, backup_error)
                self._notify('backup unavailable', self.backup, backup_error)
                # Both unavailable: stay on the queued primary, retaining the job.
                old = self.primary._max_quota_wait
                self.primary._max_quota_wait = None
                try: result = self.primary.generate(**kwargs)
                finally: self.primary._max_quota_wait = old
        self.provider, self.last_usage = result.provider, result.usage
        if self._quota_callback and result.provider != 'groq':
            self._quota_callback('usage', {'model': result.model, 'provider': result.provider, **result.usage})
        self._write(key, result.content, result.provider, result.usage, True)
        return result

    def generate_with_retry(self, **kwargs):
        kwargs.pop('max_retries', None)
        return self.generate(**kwargs)

    def generate_streaming(self, **kwargs):
        self._hooks()
        self._raise_if_cancelled()
        key = self._key(kwargs, True)
        saved = self._read(key)
        if saved and saved['complete']:
            self.provider = saved['provider']
            self._notify('checkpoint', self.primary)
            yield saved['content']
            return
        content = saved['content'] if saved else ''
        if content:
            yield content
            kwargs = dict(kwargs)
            kwargs['prompt'] += '\n\nA draft was interrupted. Continue from its last sentence without repeating any text. Finish the remaining scene only.\nDRAFT:\n' + content
            self._notify('resuming draft', self.primary)
        active = self.primary
        self.provider = 'groq' if 'Groq' in active.get_name() else active.get_name().split(' ')[0].lower()
        last_saved = time.monotonic()
        emitted = False
        complete = False
        try:
            try:
                self._check_health(active)
                for chunk in active.generate_streaming(**kwargs):
                    self._raise_if_cancelled()
                    content += chunk
                    emitted = True
                    yield chunk
                    if time.monotonic() - last_saved > 1:
                        self._write(key, content, self.provider, getattr(active, 'last_usage', {}), False)
                        last_saved = time.monotonic()
                complete = True
            except (PipelineCancelledError, ContentBlockedError):
                raise
            except Exception as error:
                self._record_failure(active, error)
                if not self.backup: raise
                self._raise_if_cancelled()
                active = self.backup
                self.provider = active.get_name().split(' ')[0].lower()
                self._notify('backup', active, error)
                if emitted:
                    kwargs = dict(kwargs)
                    kwargs['prompt'] += '\n\nContinue this interrupted draft without repeating it. Finish only what remains:\n' + content
                try:
                    self._check_health(active)
                    for chunk in active.generate_streaming(**kwargs):
                        self._raise_if_cancelled()
                        content += chunk
                        yield chunk
                    complete = True
                except PipelineCancelledError:
                    raise
                except Exception as backup_error:
                    self._record_failure(active, backup_error)
                    if content: raise  # Saved partial draft is resumed, never replayed blindly.
                    self.primary._max_quota_wait = None
                    try:
                        for chunk in self.primary.generate_streaming(**kwargs):
                            content += chunk
                            yield chunk
                        complete = True
                    finally:
                        self.primary._max_quota_wait = config.GROQ_BACKUP_WAIT_SECONDS
        finally:
            self.last_usage = getattr(active, 'last_usage', {})
            self._write(key, content, self.provider, self.last_usage, complete)

    def is_available(self):
        return self.primary.is_available()

    def get_name(self):
        return self.primary.get_name()
