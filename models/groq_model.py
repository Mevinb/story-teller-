"""
Groq Cloud wrapper using OpenAI-compatible SDK.
Primary backend for cloud prose generation.
"""
import json
import logging
import threading
import time
from typing import Optional

from openai import OpenAI, APIError, RateLimitError

import config
from .base import (
    ContentBlockedError,
    LLMInterface,
    LLMResponse,
    ReasoningStreamFilter,
    parse_retry_after_seconds,
    ModelUnavailableError,
    RateLimitExhaustedError,
    QuotaExhaustedError,
)
from pipeline.errors import PipelineCancelledError
from .key_rotator import KeyRotator
from .quota_scheduler import QuotaScheduler, QuotaDeferred, reset_seconds
from logger import get_logger, log_llm_call

logger = get_logger("groq")


_groq_rotator = KeyRotator(
    provider="groq",
    keys=config.GROQ_API_KEYS,
    strategy=getattr(config, "KEY_ROTATION_STRATEGY", "least_loaded"),
)


def quota_group_for_key(key: str) -> str:
    """Returns an isolated quota group per account so multiple accounts do not stall each other."""
    if not key:
        return config.GROQ_QUOTA_GROUP
    keys = GroqKeyManager.get_keys()
    if len(keys) > 1 and key in keys:
        return f"{config.GROQ_QUOTA_GROUP}:acc_{keys.index(key)}"
    return config.GROQ_QUOTA_GROUP


class GroqKeyManager:
    """Manages rotation, failover, and health tracking across multiple Groq accounts."""
    rotator = _groq_rotator

    @classmethod
    def sync_keys(cls, keys=None):
        if keys is None:
            keys = config.GROQ_API_KEYS
        else:
            config.GROQ_API_KEYS = [k.strip() for k in keys if k.strip()]
            if config.GROQ_API_KEYS:
                config.GROQ_API_KEY = config.GROQ_API_KEYS[0]
        cls.rotator.set_keys(keys)

    @classmethod
    def get_keys(cls) -> list:
        keys = cls.rotator.get_keys()
        if not keys and config.GROQ_API_KEY:
            return [config.GROQ_API_KEY]
        return keys

    @classmethod
    def get_key(cls, proactive_rotate: bool = False, model: Optional[str] = None, estimated_tokens: int = 0) -> str:
        key = cls.rotator.get_key(proactive_rotate=proactive_rotate, model=model, estimated_tokens=estimated_tokens)
        if not key:
            return config.GROQ_API_KEY
        return key

    @classmethod
    def get_active_account(cls, proactive_rotate: bool = False, model: Optional[str] = None, estimated_tokens: int = 0):
        return cls.rotator.get_active_account(proactive_rotate=proactive_rotate, model=model, estimated_tokens=estimated_tokens)

    @classmethod
    def rotate(cls) -> bool:
        return cls.rotator.rotate()

    @classmethod
    def mark_rate_limited(
        cls,
        key: str,
        retry_after: float,
        is_daily: bool = False,
        header_limit: Optional[int] = None,
        model: Optional[str] = None,
    ):
        return cls.rotator.mark_rate_limited(key, retry_after, is_daily, header_limit, model)

    @classmethod
    def mark_success(cls, key: str, tokens: int = 0):
        cls.rotator.mark_success(key, tokens)

    @classmethod
    def mark_error(cls, key: str, error: Exception):
        cls.rotator.mark_error(key, error)

    @classmethod
    def are_all_daily_exhausted(cls) -> bool:
        return cls.rotator.are_all_daily_exhausted()

    @classmethod
    def status_summary(cls):
        return cls.rotator.status_summary()


class GroqModel(ReasoningStreamFilter, LLMInterface):
    """One admission/usage path for normal and streaming Groq requests."""

    def __init__(self, model=None, api_key=None, scheduler=None):
        self.model = model or config.GROQ_MODEL
        self._custom_api_key = api_key
        self._client = None
        self._client_lock = threading.Lock()
        self._content_blocked = False
        self._scheduler = scheduler
        self.last_usage = {}
        self.provider = "groq"

    @property
    def scheduler(self):
        if self._scheduler is None:
            self._scheduler = QuotaScheduler()
        return self._scheduler

    @property
    def api_key(self):
        # Never switch accounts because a workload budget ran out. Rotation is
        # a credential operation; the ledger remains shared across all keys.
        return self._custom_api_key or GroqKeyManager.get_key()

    @property
    def client(self):
        return self._client_for(self.api_key)

    def _client_for(self, key):
        with self._client_lock:
            if self._client is not None and getattr(self, '_client_key', key) == key:
                return self._client
            if not key:
                raise RuntimeError("GROQ_API_KEY not set. Add it to .env.")
            client = OpenAI(base_url=config.GROQ_BASE_URL, api_key=key,
                            timeout=60.0, max_retries=0)
            self._client, self._client_key = client, key
            return client

    @property
    def was_content_blocked(self):
        return self._content_blocked

    def _is_qwen_model(self):
        return 'qwen' in self.model.lower()

    def _use_strict_json_mode(self):
        return not self._is_qwen_model()

    def _qwen_reasoning_params(self):
        return {'reasoning_effort': 'none', 'reasoning_format': 'hidden'} if self._is_qwen_model() else {}

    def _prepare_messages(self, prompt, system=''):
        if self._is_qwen_model():
            system = (system.rstrip() + "\n\nReasoning mode is disabled. Do not output hidden reasoning, "
                      "analysis, <think> tags, or planning. Start directly with the final requested content.").strip()
            prompt = prompt.rstrip() + "\n\n/no_think"
        messages = [{'role': 'system', 'content': system}] if system else []
        return messages + [{'role': 'user', 'content': prompt}]

    def _kwargs(self, prompt, system, schema, temperature, max_tokens, stream=False):
        messages = self._prepare_messages(prompt, system)
        if schema:
            instruction = "\n\nReturn ONLY minified valid JSON matching this schema. No markdown or reasoning.\nSchema: " + json.dumps(schema, separators=(',', ':'))
            if messages[0]['role'] == 'system': messages[0]['content'] += instruction
            else: messages.insert(0, {'role': 'system', 'content': instruction})
        kwargs = dict(model=self.model, messages=messages,
                      temperature=config.CLOUD_MODEL_PARAMS['temperature'] if temperature is None else temperature,
                      top_p=config.CLOUD_MODEL_PARAMS['top_p'],
                      max_tokens=max_tokens or config.CLOUD_MODEL_PARAMS['max_tokens'])
        if self._is_qwen_model(): kwargs['extra_body'] = self._qwen_reasoning_params()
        if schema and self._use_strict_json_mode(): kwargs['response_format'] = {'type': 'json_object'}
        if stream:
            kwargs.update(stream=True, stream_options={'include_usage': True})
        return kwargs

    def _progress(self, phase, details):
        callback = getattr(self, '_quota_callback', None)
        if callback: callback(phase, details)

    @staticmethod
    def _usage(response):
        usage = getattr(response, 'usage', None)
        if not usage:
            extra = getattr(response, 'x_groq', None)
            usage = extra.get('usage') if isinstance(extra, dict) else getattr(extra, 'usage', None)
        if not usage: return None
        if hasattr(usage, 'model_dump'): usage = usage.model_dump()
        if isinstance(usage, dict):
            return {'prompt_tokens': usage.get('prompt_tokens'), 'completion_tokens': usage.get('completion_tokens')}
        return {'prompt_tokens': usage.prompt_tokens, 'completion_tokens': usage.completion_tokens}

    def _send(self, kwargs):
        input_tokens = self.scheduler.estimate_for(config.GROQ_QUOTA_GROUP, self.model, kwargs['messages'])
        max_output = kwargs.get('max_tokens') or config.CLOUD_MODEL_PARAMS.get('max_tokens', 2048)
        estimated_tokens = input_tokens + max_output

        # Select initial key/account
        if self._custom_api_key:
            key = self._custom_api_key
            account = None
            group = config.GROQ_QUOTA_GROUP
        else:
            proactive = getattr(config, "GROQ_PROACTIVE_ROTATION", True)
            account = GroqKeyManager.get_active_account(
                proactive_rotate=proactive,
                model=self.model,
                estimated_tokens=estimated_tokens,
            )
            key = account.key if account else GroqKeyManager.get_key()
            group = quota_group_for_key(key)

        all_keys = GroqKeyManager.get_keys()
        max_attempts = max(len(all_keys), 1) * 3 if not self._custom_api_key else 2
        attempt = 0

        while attempt < max_attempts:
            self._raise_if_cancelled()
            client = self._client_for(key)
            try:
                reservation = self.scheduler.acquire(
                    group, self.model, input_tokens, kwargs['max_tokens'],
                    self._raise_if_cancelled, self._sleep_interruptible, self._progress,
                    getattr(self, '_max_quota_wait', None))
            except (QuotaDeferred, QuotaExhaustedError) as q_err:
                wait = getattr(q_err, 'wait_seconds', 15.0)
                if not self._custom_api_key and len(all_keys) > 1:
                    has_alt, min_wait, next_acc = GroqKeyManager.mark_rate_limited(
                        key=key, retry_after=wait, model=self.model
                    )
                    if has_alt and next_acc and next_acc.key != key:
                        logger.warning(
                            "[Groq] Admission quota/TPM limit on %s (%s). Swapping to %s.",
                            account.account_id if account else (key[:8] + "..."),
                            q_err,
                            next_acc.account_id
                        )
                        account = next_acc
                        key = next_acc.key
                        group = quota_group_for_key(key)
                        attempt += 1
                        continue
                raise RateLimitExhaustedError(
                    f"Groq token quota or TPM limit exceeded for model '{self.model}'. {q_err}",
                    wait_seconds=wait,
                ) from q_err

            dispatched = False
            try:
                self._raise_if_cancelled()
                if account:
                    account.record_request()
                dispatched = True
                raw = client.chat.completions.with_raw_response.create(**kwargs)
                self.scheduler.observe(reservation, raw.headers)
                response = raw.parse()
                return response, reservation, key
            except RateLimitError as error:
                self.scheduler.settle(reservation, rejected=True)
                headers = getattr(error.response, 'headers', {})
                wait = parse_retry_after_seconds(error) + config.GROQ_RATE_LIMIT_BUFFER
                message = str(error).lower()
                is_daily = any(term in message for term in ('tokens per day', 'requests per day', 'tpd', 'rpd'))
                if is_daily:
                    wait = max(wait, reset_seconds(headers.get('x-ratelimit-reset-requests', '')) or 86400)
                self.scheduler.observe(reservation, headers, retry_after=wait)

                # Check if we can rotate to an alternate healthy key
                if not self._custom_api_key and len(all_keys) > 1:
                    has_alt, min_wait, next_acc = GroqKeyManager.mark_rate_limited(
                        key=key, retry_after=wait, is_daily=is_daily, model=self.model
                    )
                    if has_alt and next_acc:
                        masked_old = account.masked_key if account else (key[:8] + "...")
                        logger.warning(
                            "[Groq] Rate limited on %s. Intelligently swapping to %s (attempt %d/%d).",
                            masked_old, next_acc.account_id, attempt + 1, max_attempts
                        )
                        self._progress('waiting', {
                            'model': self.model,
                            'reason': f"swapping to {next_acc.account_id}",
                            'wait_seconds': 0
                        })
                        account = next_acc
                        key = next_acc.key
                        group = quota_group_for_key(key)
                        attempt += 1
                        continue

                # If no alternate key or single key:
                self._progress('waiting', {'model': self.model, 'reason': 'provider cooldown', 'wait_seconds': wait})
                attempt += 1
                if attempt >= max_attempts or (self._custom_api_key and attempt >= 2):
                    raise RateLimitExhaustedError(
                        f"Groq rate limit exceeded for model '{self.model}' across all accounts. Please retry in {int(wait)}s or switch to Gemini.",
                        wait_seconds=wait,
                        is_daily=is_daily,
                    ) from error
                self._sleep_interruptible(wait)
            except APIError as error:
                self.scheduler.settle(reservation)
                if error.status_code in (401, 403):
                    GroqKeyManager.mark_error(key, error)
                    if not self._custom_api_key and len(all_keys) > 1:
                        next_acc = GroqKeyManager.get_active_account(model=self.model)
                        if next_acc and next_acc.key != key:
                            logger.error("[Groq] Auth failed on key %s. Swapping to %s.", key[:8], next_acc.account_id)
                            account = next_acc
                            key = next_acc.key
                            group = quota_group_for_key(key)
                            attempt += 1
                            continue
                # Handle 404 Not Found / Model Discontinued
                if error.status_code == 404:
                    raise ModelUnavailableError(
                        f"Groq model '{self.model}' not found or discontinued (404 NOT FOUND). Please switch to an active model.",
                        is_temporary=False,
                        status_code=404,
                    ) from error
                # Handle 503 Service Unavailable / Server Overload
                if error.status_code == 503 or any(term in str(error).lower() for term in ('unavailable', 'overloaded', 'high demand')):
                    raise ModelUnavailableError(
                        f"Groq model '{self.model}' is temporarily unavailable due to high demand (503 UNAVAILABLE). Please try again in a few moments or switch to Gemini.",
                        is_temporary=True,
                        status_code=503,
                    ) from error
                if error.status_code == 400 and 'response_format' in kwargs and any(
                        term in str(error).lower() for term in ('json_validate_failed', 'failed to generate json')):
                    kwargs = dict(kwargs)
                    kwargs.pop('response_format')
                    continue  # The repair is a fresh, fully accounted request.
                raise
            except BaseException:
                if dispatched:
                    self.scheduler.settle(reservation)
                else:
                    self.scheduler.release_unsent(reservation)
                raise

    def generate(self, prompt, system='', schema=None, temperature=None, max_tokens=None,
                 stream=False, _retry_count=0):
        if stream:
            content = ''.join(self.generate_streaming(prompt, system, temperature, max_tokens))
            return LLMResponse(content, self.model, 'groq', dict(self.last_usage))
        self._content_blocked = False
        start_time = time.monotonic()
        try:
            response, reservation, key = self._send(self._kwargs(prompt, system, schema, temperature, max_tokens))
            usage = self._usage(response)
            self.scheduler.settle(reservation, usage)
            self.last_usage = usage or {}
            self._progress('usage', {'model': self.model, 'provider': 'groq', **self.last_usage})
            GroqKeyManager.mark_success(key, sum(v or 0 for v in self.last_usage.values()))
            choice = response.choices[0]
            if choice.finish_reason == 'content_filter':
                self._content_blocked = True
                raise ContentBlockedError('Groq content moderation blocked this request.')
            result_text = self._strip_reasoning(choice.message.content or '')
            latency = time.monotonic() - start_time
            log_llm_call(
                provider="groq",
                model=self.model,
                prompt=prompt,
                system=system,
                response=result_text,
                latency=latency,
                usage=self.last_usage,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return LLMResponse(result_text, self.model, 'groq', self.last_usage, response)
        except Exception as e:
            latency = time.monotonic() - start_time
            log_llm_call(
                provider="groq",
                model=self.model,
                prompt=prompt,
                system=system,
                error=e,
                latency=latency,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            raise

    def generate_with_retry(self, **kwargs):
        # Admission owns rate retries; agent-level retries must not multiply them.
        kwargs.pop('max_retries', None)
        return self.generate(**kwargs)

    def generate_streaming(self, prompt, system='', temperature=None, max_tokens=None, _retry_count=0):
        self._content_blocked = False
        stream, reservation, key = self._send(self._kwargs(prompt, system, None, temperature, max_tokens, True))
        usage = None
        finish_reason = None
        last_touch = time.monotonic()
        def chunks():
            nonlocal usage, last_touch, finish_reason
            for chunk in stream:
                self._raise_if_cancelled()
                if time.monotonic() - last_touch > 10:
                    self.scheduler.touch(reservation)
                    last_touch = time.monotonic()
                usage = self._usage(chunk) or usage  # Metadata-only terminal chunks count.
                if not chunk.choices: continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                if choice.finish_reason == 'content_filter':
                    self._content_blocked = True
                    raise ContentBlockedError('Groq content moderation blocked this request.')
                if choice.delta and choice.delta.content: yield choice.delta.content
        completed = False
        try:
            yield from self._filter_reasoning_stream(chunks())
            if finish_reason is None and usage is None:
                raise RuntimeError("Groq stream ended before completion metadata; draft saved for continuation")
            completed = True
        finally:
            try:
                stream.close()
            finally:
                self.scheduler.settle(reservation, usage)
            self.last_usage = usage or {}
            self._progress('usage', {'model': self.model, 'provider': 'groq', 'complete': completed, **self.last_usage})
            if completed:
                GroqKeyManager.mark_success(key, sum(v or 0 for v in self.last_usage.values()))

    def is_available(self):
        if not self.api_key: return False
        try:
            return any(m.id == self.model and getattr(m, 'active', True) is not False
                       for m in self.client.models.list().data)
        except Exception as error:
            logger.warning('[Groq] Availability check failed: %s', type(error).__name__)
            return False

    def get_name(self):
        return f"Groq ({self.model})"
