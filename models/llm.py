"""
llama.cpp (local GGUF) wrapper.
Provides structured and streaming generation through llama-cpp-python.
"""
import json
import logging
import os
import threading
from typing import Optional

import config
from .base import LLMInterface, LLMResponse

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - import can fail before dependency install
    Llama = None


def resolve_model_path(model: str) -> str:
    candidate = (model or "").strip()
    if not candidate:
        return ""
    if os.path.isabs(candidate):
        return os.path.abspath(candidate)

    root_candidate = os.path.abspath(os.path.join(config.PROJECT_ROOT, candidate))
    if os.path.exists(root_candidate):
        return root_candidate

    return os.path.abspath(os.path.join(config.LLAMA_MODELS_DIR, candidate))


def to_model_id(model_path: str) -> str:
    resolved = resolve_model_path(model_path)
    if not resolved:
        return ""

    base = os.path.abspath(config.LLAMA_MODELS_DIR)
    try:
        rel = os.path.relpath(resolved, base)
    except ValueError:
        return resolved

    if rel == ".." or rel.startswith(f"..{os.sep}"):
        return resolved
    return rel.replace(os.sep, "/")


def list_gguf_models() -> list[str]:
    models = []
    base = os.path.abspath(config.LLAMA_MODELS_DIR)
    if os.path.isdir(base):
        for root, _, files in os.walk(base):
            for name in files:
                if name.lower().endswith(".gguf"):
                    models.append(to_model_id(os.path.join(root, name)))

    configured = resolve_model_path(config.LLAMA_MODEL_PATH)
    if configured.lower().endswith(".gguf") and os.path.isfile(configured):
        models.append(to_model_id(configured))

    return sorted(dict.fromkeys(models))


class LlamaCPP(LLMInterface):
    """Local LLM via llama.cpp with GGUF models."""
    _MODEL_CACHE = {}
    _MODEL_LOCK = threading.Lock()
    _INFERENCE_LOCKS = {}

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_gpu_layers: Optional[int] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_batch: Optional[int] = None,
        f16_kv: Optional[bool] = None,
    ):
        self.model_path = resolve_model_path(model_path or config.LLAMA_MODEL_PATH)
        self.model = to_model_id(self.model_path) or self.model_path
        self.n_gpu_layers = config.LLAMA_CPP_PARAMS["n_gpu_layers"] if n_gpu_layers is None else n_gpu_layers
        self.n_ctx = config.LOCAL_MODEL_PARAMS["num_ctx"] if n_ctx is None else n_ctx
        self.n_threads = config.LLAMA_CPP_PARAMS["n_threads"] if n_threads is None else n_threads
        self.n_batch = config.LLAMA_CPP_PARAMS["n_batch"] if n_batch is None else n_batch
        self.f16_kv = config.LLAMA_CPP_PARAMS["f16_kv"] if f16_kv is None else f16_kv
        self.prompt_template = config.LLAMA_PROMPT_TEMPLATE
        self._cache_key = (
            self.model_path,
            int(self.n_gpu_layers),
            int(self.n_ctx),
            int(self.n_threads),
            int(self.n_batch),
            bool(self.f16_kv),
            self.prompt_template,
        )

    @property
    def llm(self):
        with self._MODEL_LOCK:
            cached = self._MODEL_CACHE.get(self._cache_key)
            if cached is not None:
                return cached

            if Llama is None:
                raise RuntimeError(
                    "llama-cpp-python is not installed. Run: pip install llama-cpp-python"
                )
            if not os.path.isfile(self.model_path):
                raise RuntimeError(
                    f"GGUF model file not found: {self.model_path}. "
                    "Set LLAMA_MODEL_PATH or LLAMA_MODELS_DIR correctly."
                )

            logger.info(
                "Loading GGUF model",
                extra={
                    "model": self.model,
                    "n_gpu_layers": self.n_gpu_layers,
                    "n_ctx": self.n_ctx,
                    "n_batch": self.n_batch,
                    "f16_kv": self.f16_kv,
                },
            )
            llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                f16_kv=self.f16_kv,
                verbose=False,
            )
            self._MODEL_CACHE[self._cache_key] = llm
            self._INFERENCE_LOCKS[self._cache_key] = threading.Lock()
            return llm

    @property
    def inference_lock(self):
        lock = self._INFERENCE_LOCKS.get(self._cache_key)
        if lock is not None:
            return lock
        _ = self.llm
        return self._INFERENCE_LOCKS[self._cache_key]

    @staticmethod
    def _schema_instruction(schema: dict) -> str:
        return (
            "You MUST respond with ONLY valid JSON matching this schema. "
            "No additional text and no markdown fences.\n"
            f"Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
        )

    def _format_prompt(
        self,
        prompt: str,
        system: str = "",
        schema: Optional[dict] = None,
    ) -> str:
        system_text = (system or "").strip()
        if schema:
            schema_text = self._schema_instruction(schema)
            system_text = f"{system_text}\n\n{schema_text}".strip() if system_text else schema_text
        user_text = (prompt or "").strip()

        if self.prompt_template == "llama3":
            parts = ["<|begin_of_text|>"]
            if system_text:
                parts.append(
                    "<|start_header_id|>system<|end_header_id|>\n\n"
                    f"{system_text}<|eot_id|>"
                )
            parts.append(
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_text}<|eot_id|>"
            )
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(parts)

        if self.prompt_template == "chatml":
            parts = []
            if system_text:
                parts.append(f"<|im_start|>system\n{system_text}<|im_end|>\n")
            parts.append(f"<|im_start|>user\n{user_text}<|im_end|>\n")
            parts.append("<|im_start|>assistant\n")
            return "".join(parts)

        merged = user_text
        if system_text:
            merged = f"{system_text}\n\n{user_text}".strip()
        return f"<s>[INST] {merged} [/INST]"

    def _build_generation_kwargs(
        self,
        prompt: str,
        system: str = "",
        schema: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> dict:
        prompt_text = self._format_prompt(prompt=prompt, system=system, schema=schema)
        max_out = max_tokens
        if max_out is None:
            max_out = (
                config.LOCAL_MODEL_PARAMS["structured_max_tokens"]
                if schema
                else config.LOCAL_MODEL_PARAMS["max_tokens"]
            )

        kwargs = {
            "prompt": prompt_text,
            "max_tokens": max(1, int(max_out)),
            "temperature": (
                config.LOCAL_MODEL_PARAMS["temperature"] if temperature is None else temperature
            ),
            "top_p": config.LOCAL_MODEL_PARAMS["top_p"],
            "stream": stream,
        }
        if self.prompt_template == "llama3":
            kwargs["stop"] = ["<|eot_id|>"]
        elif self.prompt_template == "chatml":
            kwargs["stop"] = ["<|im_end|>"]
        return kwargs

    def generate(
        self,
        prompt: str,
        system: str = "",
        schema: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> LLMResponse:
        if stream:
            content = "".join(
                self.generate_streaming(
                    prompt=prompt,
                    system=system,
                    schema=schema,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            return LLMResponse(
                content=content,
                model=self.model,
                provider="llama.cpp",
                usage={},
                raw=None,
            )

        kwargs = self._build_generation_kwargs(
            prompt=prompt,
            system=system,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        with self.inference_lock:
            response = self.llm(**kwargs)
        choice = (response.get("choices") or [{}])[0]
        content = choice.get("text", "")
        if not content.strip():
            raise RuntimeError(
                f"llama.cpp returned empty output for '{self.model}'. "
                "Try a lower LOCAL_NUM_CTX or a different GGUF quantization."
            )
        return LLMResponse(
            content=content,
            model=self.model,
            provider="llama.cpp",
            usage=response.get("usage", {}),
            raw=response,
        )

    def generate_streaming(
        self,
        prompt: str,
        system: str = "",
        schema: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        kwargs = self._build_generation_kwargs(
            prompt=prompt,
            system=system,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        with self.inference_lock:
            stream = self.llm(**kwargs)
            for chunk in stream:
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text", "")
                if text:
                    yield text

    def is_available(self) -> bool:
        try:
            _ = self.llm
            return True
        except Exception as e:
            logger.error(f"[llama.cpp] Availability check failed: {e}")
            return False

    def get_name(self) -> str:
        return f"llama.cpp ({self.model})"
