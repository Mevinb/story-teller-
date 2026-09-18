"""
Story Teller — Web UI Server
Flask application with REST API and Server-Sent Events for live generation.
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["DISABLE_TQDM"] = "1"
os.environ["TQDM_DISABLE"] = "1"
import sys
import json
import threading
import queue
import logging
import re
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify, Response
import config
from pipeline.orchestrator import PipelineOrchestrator, normalize_project_name
from pipeline.gemini_combiner import combine_chapters, analyze_and_polish, generate_whole_story
from pipeline.errors import PipelineCancelledError
from models.groq_model import GroqModel
from models.openrouter_model import OpenRouterModel
from models.llm import LlamaCPP, list_gguf_models, resolve_model_path, to_model_id
from vision.router import vision_bp

logger = logging.getLogger(__name__)

# Global state for active generation
_active_pipelines = {}
_event_queues = {}
_cancel_requests = set()
_generation_lock = threading.Lock()
_selected_local_model_path = resolve_model_path(config.LLAMA_MODEL_PATH)
_initial_backend_mode = os.getenv(
    "BACKEND_MODE",
    "hybrid" if config.USE_CLOUD_MODEL else ("groq" if config.GROQ_API_KEY else "local"),
).strip().lower()
if _initial_backend_mode not in {"local", "hybrid", "groq", "gemini", "openrouter"}:
    _initial_backend_mode = "groq" if config.GROQ_API_KEY else "local"
config.USE_CLOUD_MODEL = _initial_backend_mode == "hybrid"
if _initial_backend_mode == "gemini":
    _selected_backend = "gemini"
elif _initial_backend_mode == "groq":
    _selected_backend = "groq"
elif _initial_backend_mode == "openrouter":
    _selected_backend = "openrouter"
else:
    _selected_backend = "local"
_model_health_cache = {}
_combine_queues = {}
_active_combines = {}
_combine_results = {}
_combine_cancel_requests = set()
_manual_sessions = {}  # project_name -> {"pipeline": ..., "session": ..., "eq": ...}
_CHAPTER_FILE_RE = re.compile(r"^chapter_(\d{3,})\.md$")
_GROQ_MODEL_OPTION = "__groq_api__"
_GEMINI_MODEL_OPTION = "__gemini_api__"
_OPENROUTER_MODEL_OPTION = "__openrouter_api__"
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
_SENSITIVE_SETTING_KEYS = {"GROQ_API_KEY", "GROQ_API_KEYS", "GEMINI_API_KEY", "OPENROUTER_API_KEY"}
_REDACTED_VALUE = "********"

# Editable settings exposed by /api/settings. Defaults are sourced from
# `config` so the UI can never drift from the actual runtime configuration.
_SETTING_DEFS = {
    "BACKEND_MODE": {"type": "str", "default": "local"},
    "GROQ_API_KEY": {"type": "str", "default": ""},
    "GROQ_API_KEYS": {"type": "str", "default": ""},
    "GROQ_MODEL": {"type": "str", "default": config.GROQ_MODEL},
    "GEMINI_API_KEY": {"type": "str", "default": ""},
    "GEMINI_MODEL": {"type": "str", "default": config.GEMINI_MODEL},
    "OPENROUTER_API_KEY": {"type": "str", "default": ""},
    "OPENROUTER_MODEL": {"type": "str", "default": config.OPENROUTER_MODEL},
    "OPENROUTER_MIN_REQUEST_INTERVAL": {
        "type": "float", "default": config.OPENROUTER_MIN_REQUEST_INTERVAL,
    },
    "USE_CLOUD_MODEL": {"type": "bool", "default": False},
    "LLAMA_MODELS_DIR": {"type": "str", "default": config.LLAMA_MODELS_DIR},
    "LLAMA_MODEL_PATH": {"type": "str", "default": config.LLAMA_MODEL_PATH},
    "LLAMA_PROMPT_TEMPLATE": {"type": "str", "default": config.LLAMA_PROMPT_TEMPLATE},
    "LLAMA_N_GPU_LAYERS": {"type": "int", "default": config.LLAMA_CPP_PARAMS["n_gpu_layers"]},
    "LLAMA_N_BATCH": {"type": "int", "default": config.LLAMA_CPP_PARAMS["n_batch"]},
    "LLAMA_N_THREADS": {"type": "int", "default": config.LLAMA_CPP_PARAMS["n_threads"]},
    "LLAMA_F16_KV": {"type": "bool", "default": config.LLAMA_CPP_PARAMS["f16_kv"]},
    "LOCAL_NUM_CTX": {"type": "int", "default": config.LOCAL_MODEL_PARAMS["num_ctx"]},
    "LOCAL_TEMPERATURE": {"type": "float", "default": config.LOCAL_MODEL_PARAMS["temperature"]},
    "LOCAL_TOP_P": {"type": "float", "default": config.LOCAL_MODEL_PARAMS["top_p"]},
    "LOCAL_MAX_TOKENS": {"type": "int", "default": config.LOCAL_MODEL_PARAMS["max_tokens"]},
    "LOCAL_STRUCTURED_MAX_TOKENS": {
        "type": "int", "default": config.LOCAL_MODEL_PARAMS["structured_max_tokens"],
    },
    "CLOUD_TEMPERATURE": {"type": "float", "default": config.CLOUD_MODEL_PARAMS["temperature"]},
    "CLOUD_TOP_P": {"type": "float", "default": config.CLOUD_MODEL_PARAMS["top_p"]},
    "CLOUD_MAX_TOKENS": {"type": "int", "default": config.CLOUD_MODEL_PARAMS["max_tokens"]},
    "GROQ_MIN_REQUEST_INTERVAL": {"type": "float", "default": config.GROQ_MIN_REQUEST_INTERVAL},
    "GROQ_RATE_LIMIT_BUFFER": {"type": "float", "default": config.GROQ_RATE_LIMIT_BUFFER},
    "GROQ_ROTATE_ON_RATE_LIMIT": {"type": "bool", "default": config.GROQ_ROTATE_ON_RATE_LIMIT},
    "GROQ_CONTINUATION_ATTEMPTS": {"type": "int", "default": config.GROQ_CONTINUATION_ATTEMPTS},
    "GROQ_TPM_LIMIT": {"type": "int", "default": config.GROQ_TPM_LIMIT},
    "GROQ_TPM_WINDOW_SECONDS": {"type": "float", "default": config.GROQ_TPM_WINDOW_SECONDS},
    "GROQ_TPM_SAFETY_MARGIN": {"type": "float", "default": config.GROQ_TPM_SAFETY_MARGIN},
    "GROQ_TPM_RESERVE_DEFAULT": {"type": "int", "default": config.GROQ_TPM_RESERVE_DEFAULT},
    "GROQ_TPM_PACING": {"type": "bool", "default": config.GROQ_TPM_PACING},
    "GROQ_PROACTIVE_ROTATION": {"type": "bool", "default": config.GROQ_PROACTIVE_ROTATION},
    "HYBRID_ROUTING": {"type": "bool", "default": config.HYBRID_ROUTING},
    "RETRY_BASE_DELAY": {"type": "float", "default": config.RETRY_BASE_DELAY},
    "RETRY_MAX_DELAY": {"type": "float", "default": config.RETRY_MAX_DELAY},
    "RETRY_BACKOFF_FACTOR": {"type": "float", "default": config.RETRY_BACKOFF_FACTOR},
    "WORDS_PER_SCENE_MIN": {"type": "int", "default": config.WORDS_PER_SCENE_MIN},
    "WORDS_PER_SCENE_MAX": {"type": "int", "default": config.WORDS_PER_SCENE_MAX},
    "MIN_SCENE_WORDS": {"type": "int", "default": config.MIN_SCENE_WORDS},
    "MAX_SCENE_ITERATIONS": {"type": "int", "default": config.MAX_SCENE_ITERATIONS},
    "BEST_OF_N_KEY_SCENES": {"type": "int", "default": config.BEST_OF_N_KEY_SCENES},
    "BEST_OF_N_NORMAL_SCENES": {"type": "int", "default": config.BEST_OF_N_NORMAL_SCENES},
    "MAX_PIPELINE_STEPS": {"type": "int", "default": config.MAX_PIPELINE_STEPS},
    "MAX_TOKEN_BUDGET": {"type": "int", "default": config.MAX_TOKEN_BUDGET},
    "TOP_K_RETRIEVAL": {"type": "int", "default": config.TOP_K_RETRIEVAL},
    "CONTEXT_TOKEN_BUDGET": {"type": "int", "default": config.CONTEXT_TOKEN_BUDGET},
    "SSE_QUEUE_MAXSIZE": {"type": "int", "default": config.SSE_QUEUE_MAXSIZE},
    "FLASK_HOST": {"type": "str", "default": config.FLASK_HOST},
    "FLASK_PORT": {"type": "int", "default": config.FLASK_PORT},
    "FLASK_DEBUG": {"type": "bool", "default": config.FLASK_DEBUG},
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_event(event_type: str, data=None) -> dict:
    data = data if data is not None else {}
    agent = data.get("agent") if isinstance(data, dict) else None
    content = (
        data
        if isinstance(data, str)
        else (
            data.get("content")
            or data.get("step")
            or data.get("error")
            or ""
        )
        if isinstance(data, dict)
        else ""
    )
    payload = {
        "type": event_type,
        "agent": agent,
        "content": content,
        "timestamp": _now_iso(),
        "payload": data,
        # Compatibility for current frontend
        "event": event_type,
        "data": data,
    }
    return payload


def _list_local_models() -> list:
    return list_gguf_models()


def _probe_local_model(model: str, force: bool = False) -> tuple[bool, str]:
    """Quick runtime probe to detect unavailable GGUF models."""
    resolved = resolve_model_path(model)
    model_key = to_model_id(resolved)
    cached = _model_health_cache.get(model_key)
    if (not force) and cached is not None:
        return cached

    try:
        probe = LlamaCPP(model_path=resolved)
        if not probe.is_available():
            result = (
                False,
                f"Model '{model_key}' could not be loaded by llama.cpp. "
                "Check that the GGUF file is valid and your llama-cpp-python build supports your hardware.",
            )
            _model_health_cache[model_key] = result
            return result
    except Exception as e:
        result = (False, f"llama.cpp probe failed for '{model_key}': {e}")
        _model_health_cache[model_key] = result
        return result

    result = (True, "")
    _model_health_cache[model_key] = result
    return result


def _queue_event(eq: queue.Queue, message: dict, force: bool = False) -> bool:
    try:
        eq.put_nowait(message)
        return True
    except queue.Full:
        if not force:
            return False
        try:
            eq.get_nowait()
        except queue.Empty:
            pass
        try:
            eq.put_nowait(message)
            return True
        except queue.Full:
            return False


def _is_groq_selection(selected: str) -> bool:
    """Return True if the model selection string refers to Groq."""
    return selected == _GROQ_MODEL_OPTION or selected.startswith("groq:")


def _groq_model_from_selection(selected: str) -> str:
    """Extract the Groq model ID from a selection string."""
    if selected.startswith("groq:"):
        return selected.split(":", 1)[1]
    return config.GROQ_MODEL


def _is_gemini_selection(selected: str) -> bool:
    """Return True if the model selection string refers to Gemini."""
    return selected == _GEMINI_MODEL_OPTION or selected.startswith("gemini:")


def _gemini_model_from_selection(selected: str) -> str:
    """Extract the Gemini model ID from a selection string."""
    if selected.startswith("gemini:"):
        return selected.split(":", 1)[1]
    return config.GEMINI_MODEL


def _is_openrouter_selection(selected: str) -> bool:
    """Return True if the model selection string refers to OpenRouter."""
    return selected == _OPENROUTER_MODEL_OPTION or selected.startswith("openrouter:")


def _openrouter_model_from_selection(selected: str) -> str:
    """Extract the OpenRouter model ID from a selection string."""
    if selected.startswith("openrouter:"):
        return selected.split(":", 1)[1]
    return config.OPENROUTER_MODEL


def _active_model_selection() -> str:
    if _selected_backend == "gemini":
        return f"gemini:{config.GEMINI_MODEL}"
    if _selected_backend == "openrouter":
        return f"openrouter:{config.OPENROUTER_MODEL}"
    if _selected_backend == "groq":
        return f"groq:{config.GROQ_MODEL}"
    if config.USE_CLOUD_MODEL:
        return "hybrid"
    return to_model_id(_selected_local_model_path)


def _get_llm_for_premise(requested_model=None):
    selected = (requested_model or "").strip() or _active_model_selection()
    if _is_groq_selection(selected):
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set. Add it to .env before using Groq.")
        return GroqModel(model=_groq_model_from_selection(selected))
    if _is_gemini_selection(selected):
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not set. Add it to .env before using Gemini.")
        from models.gemini_model import GeminiModel
        return GeminiModel(model=_gemini_model_from_selection(selected))
    if _is_openrouter_selection(selected):
        if not config.OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY not set. Add it to .env before using OpenRouter.")
        return OpenRouterModel(model=_openrouter_model_from_selection(selected))
    if selected == "hybrid":
        if config.GROQ_API_KEY:
            return GroqModel(model=config.GROQ_MODEL)
    
    requested_model_path = resolve_model_path(selected)
    if not os.path.isfile(requested_model_path):
        raise RuntimeError(f"Local model not found at: {requested_model_path}")
    return LlamaCPP(model_path=requested_model_path)


def _active_generation_mode() -> str:
    if _selected_backend == "groq":
        return "groq"
    if _selected_backend == "gemini":
        return "gemini"
    if _selected_backend == "openrouter":
        return "openrouter"
    return "hybrid" if config.USE_CLOUD_MODEL else "local"


def _current_pipeline_kwargs() -> dict:
    kwargs = {"local_model": _selected_local_model_path}
    if _selected_backend == "groq":
        kwargs["backend"] = "groq"
        kwargs["groq_model"] = config.GROQ_MODEL
    elif _selected_backend == "gemini":
        kwargs["backend"] = "gemini"
        kwargs["gemini_model"] = config.GEMINI_MODEL
    elif _selected_backend == "openrouter":
        kwargs["backend"] = "openrouter"
        kwargs["openrouter_model"] = config.OPENROUTER_MODEL
    elif config.USE_CLOUD_MODEL:
        kwargs["groq_model"] = config.GROQ_MODEL
    return kwargs


def _setting_to_env(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _coerce_setting(key: str, value):
    spec = _SETTING_DEFS[key]
    kind = spec["type"]
    if value is None:
        value = ""
    if kind == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    return str(value).strip()


def _read_env_values() -> dict:
    values = {}
    if not os.path.exists(_ENV_PATH):
        return values
    with open(_ENV_PATH, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            values[key] = value
    return values


def _write_env_values(updates: dict) -> None:
    lines = []
    seen = set()
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH, encoding="utf-8") as f:
            lines = f.readlines()

    out = []
    for raw in lines:
        stripped = raw.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                out.append(f"{key}={_setting_to_env(updates[key])}\n")
                seen.add(key)
                continue
        out.append(raw)

    missing = [key for key in updates if key not in seen]
    if missing:
        if out and out[-1].strip():
            out.append("\n")
        out.append("# Story Teller UI settings\n")
        for key in missing:
            out.append(f"{key}={_setting_to_env(updates[key])}\n")

    # Atomic replace: this file holds API keys, so a crash mid-write must
    # never leave a truncated .env behind.
    tmp_path = _ENV_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(out)
    os.replace(tmp_path, _ENV_PATH)


def _apply_runtime_settings(settings: dict) -> None:
    global _selected_local_model_path, _selected_backend

    for key, value in settings.items():
        os.environ[key] = _setting_to_env(value)

    config.GROQ_API_KEY = settings.get("GROQ_API_KEY", config.GROQ_API_KEY)
    raw_keys = settings.get("GROQ_API_KEYS", "")
    groq_keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if config.GROQ_API_KEY and config.GROQ_API_KEY not in groq_keys:
        groq_keys.insert(0, config.GROQ_API_KEY)
    config.GROQ_API_KEYS = groq_keys
    config.GROQ_MODEL = settings.get("GROQ_MODEL", config.GROQ_MODEL)
    config.GEMINI_API_KEY = settings.get("GEMINI_API_KEY", config.GEMINI_API_KEY)
    config.GEMINI_MODEL = settings.get("GEMINI_MODEL", config.GEMINI_MODEL)
    config.OPENROUTER_API_KEY = settings.get("OPENROUTER_API_KEY", config.OPENROUTER_API_KEY)
    config.OPENROUTER_MODEL = settings.get("OPENROUTER_MODEL", config.OPENROUTER_MODEL)
    config.OPENROUTER_MIN_REQUEST_INTERVAL = settings.get(
        "OPENROUTER_MIN_REQUEST_INTERVAL",
        config.OPENROUTER_MIN_REQUEST_INTERVAL,
    )
    config.USE_CLOUD_MODEL = bool(settings.get("USE_CLOUD_MODEL", config.USE_CLOUD_MODEL))

    config.LLAMA_MODELS_DIR = os.path.abspath(settings.get("LLAMA_MODELS_DIR", config.LLAMA_MODELS_DIR))
    config.LLAMA_MODEL_PATH = settings.get("LLAMA_MODEL_PATH", config.LLAMA_MODEL_PATH)
    config.LLAMA_PROMPT_TEMPLATE = settings.get("LLAMA_PROMPT_TEMPLATE", config.LLAMA_PROMPT_TEMPLATE).lower()
    config.LLAMA_CPP_PARAMS.update({
        "n_gpu_layers": settings.get("LLAMA_N_GPU_LAYERS", config.LLAMA_CPP_PARAMS["n_gpu_layers"]),
        "n_batch": settings.get("LLAMA_N_BATCH", config.LLAMA_CPP_PARAMS["n_batch"]),
        "n_threads": settings.get("LLAMA_N_THREADS", config.LLAMA_CPP_PARAMS["n_threads"]),
        "f16_kv": settings.get("LLAMA_F16_KV", config.LLAMA_CPP_PARAMS["f16_kv"]),
    })
    config.LOCAL_MODEL_PARAMS.update({
        "num_ctx": settings.get("LOCAL_NUM_CTX", config.LOCAL_MODEL_PARAMS["num_ctx"]),
        "temperature": settings.get("LOCAL_TEMPERATURE", config.LOCAL_MODEL_PARAMS["temperature"]),
        "top_p": settings.get("LOCAL_TOP_P", config.LOCAL_MODEL_PARAMS["top_p"]),
        "max_tokens": settings.get("LOCAL_MAX_TOKENS", config.LOCAL_MODEL_PARAMS["max_tokens"]),
        "structured_max_tokens": settings.get(
            "LOCAL_STRUCTURED_MAX_TOKENS",
            config.LOCAL_MODEL_PARAMS["structured_max_tokens"],
        ),
    })
    config.CLOUD_MODEL_PARAMS.update({
        "temperature": settings.get("CLOUD_TEMPERATURE", config.CLOUD_MODEL_PARAMS["temperature"]),
        "top_p": settings.get("CLOUD_TOP_P", config.CLOUD_MODEL_PARAMS["top_p"]),
        "max_tokens": settings.get("CLOUD_MAX_TOKENS", config.CLOUD_MODEL_PARAMS["max_tokens"]),
    })

    for key in (
        "GROQ_MIN_REQUEST_INTERVAL",
        "GROQ_RATE_LIMIT_BUFFER",
        "GROQ_ROTATE_ON_RATE_LIMIT",
        "GROQ_CONTINUATION_ATTEMPTS",
        "RETRY_BASE_DELAY",
        "RETRY_MAX_DELAY",
        "RETRY_BACKOFF_FACTOR",
        "WORDS_PER_SCENE_MIN",
        "WORDS_PER_SCENE_MAX",
        "MIN_SCENE_WORDS",
        "MAX_SCENE_ITERATIONS",
        "BEST_OF_N_KEY_SCENES",
        "BEST_OF_N_NORMAL_SCENES",
        "MAX_PIPELINE_STEPS",
        "MAX_TOKEN_BUDGET",
        "TOP_K_RETRIEVAL",
        "CONTEXT_TOKEN_BUDGET",
        "SSE_QUEUE_MAXSIZE",
        "FLASK_HOST",
        "FLASK_PORT",
        "FLASK_DEBUG",
    ):
        if key in settings:
            setattr(config, key, settings[key])

    _selected_local_model_path = resolve_model_path(config.LLAMA_MODEL_PATH)
    mode = settings.get("BACKEND_MODE")
    if mode:
        if mode == "gemini":
            _selected_backend = "gemini"
        elif mode == "groq":
            _selected_backend = "groq"
        elif mode == "openrouter":
            _selected_backend = "openrouter"
        else:
            _selected_backend = "local"
        config.USE_CLOUD_MODEL = mode == "hybrid"


def _settings_payload() -> dict:
    env_values = _read_env_values()
    values = {}
    for key, spec in _SETTING_DEFS.items():
        if key in env_values:
            try:
                values[key] = _coerce_setting(key, env_values[key])
            except (TypeError, ValueError):
                values[key] = env_values[key]
        else:
            values[key] = getattr(config, key, spec["default"])

    values["BACKEND_MODE"] = _active_generation_mode()
    values["ACTIVE_MODEL"] = _active_model_selection()
    public_values = {
        key: (_REDACTED_VALUE if key in _SENSITIVE_SETTING_KEYS and value else value)
        for key, value in values.items()
    }
    return {
        "settings": public_values,
        "local_models": _list_local_models(),
        "groq_models": config.GROQ_MODELS,
        "cloud_options": [_GROQ_MODEL_OPTION, _GEMINI_MODEL_OPTION, _OPENROUTER_MODEL_OPTION],
        "active_model": _active_model_selection(),
        "backend_mode": _active_generation_mode(),
        "env_path": _ENV_PATH,
    }


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", os.urandom(24).hex())
    app.register_blueprint(vision_bp)

    def _resolve_generation_pipeline_kwargs(requested_model: str):
        global _selected_local_model_path, _selected_backend
        pipeline_kwargs = {"local_model": _selected_local_model_path}
        selected = (requested_model or "").strip() or _active_model_selection()

        if _is_groq_selection(selected):
            if not config.GROQ_API_KEY:
                return None, jsonify({"error": "GROQ_API_KEY not set. Add it to .env before using Groq."}), 400
            groq_m = _groq_model_from_selection(selected)
            config.GROQ_MODEL = groq_m
            pipeline_kwargs.update({"backend": "groq", "groq_model": groq_m})
            _selected_backend = "groq"
            return pipeline_kwargs, None, None

        if _is_gemini_selection(selected):
            if not config.GEMINI_API_KEY:
                return None, jsonify({"error": "GEMINI_API_KEY not set. Add it to .env before using Gemini."}), 400
            gemini_m = _gemini_model_from_selection(selected)
            config.GEMINI_MODEL = gemini_m
            pipeline_kwargs.update({"backend": "gemini", "gemini_model": gemini_m})
            _selected_backend = "gemini"
            return pipeline_kwargs, None, None

        if _is_openrouter_selection(selected):
            if not config.OPENROUTER_API_KEY:
                return None, jsonify({"error": "OPENROUTER_API_KEY not set. Add it to .env before using OpenRouter."}), 400
            openrouter_m = _openrouter_model_from_selection(selected)
            config.OPENROUTER_MODEL = openrouter_m
            pipeline_kwargs.update({"backend": "openrouter", "openrouter_model": openrouter_m})
            _selected_backend = "openrouter"
            return pipeline_kwargs, None, None

        if selected == "hybrid":
            if not config.GROQ_API_KEY:
                return None, jsonify({"error": "GROQ_API_KEY not set. Add it to .env for Hybrid mode."}), 400
            config.USE_CLOUD_MODEL = True
            _selected_backend = "local"
            pipeline_kwargs.update({"local_model": _selected_local_model_path, "groq_model": config.GROQ_MODEL})
            return pipeline_kwargs, None, None

        requested_model_path = resolve_model_path(selected)
        if not requested_model_path.lower().endswith(".gguf"):
            return None, jsonify({"error": "Model must be a .gguf file"}), 400
        if not os.path.isfile(requested_model_path):
            return None, jsonify({"error": f"Model file not found: {requested_model_path}"}), 400

        ok, probe_error = _probe_local_model(requested_model_path)
        if not ok:
            return None, jsonify({
                "error": (
                    f"{probe_error} Try a different GGUF file, lower LLAMA_N_GPU_LAYERS, "
                    "or choose a smaller quantized model (e.g., 7B Q4_K_M)."
                )
            }), 400

        _selected_backend = "local"
        _selected_local_model_path = requested_model_path
        config.LLAMA_MODEL_PATH = requested_model_path
        pipeline_kwargs["local_model"] = requested_model_path
        return pipeline_kwargs, None, None

    # ─── Pages ────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    # ─── API: Model Management ────────────────────────────────────
    @app.route("/api/settings", methods=["GET"])
    def get_settings():
        """Return runtime settings and editable .env-backed values."""
        return jsonify(_settings_payload())

    @app.route("/api/settings", methods=["PUT"])
    def update_settings():
        """Persist settings to .env and apply them to new pipelines immediately."""
        data = request.json or {}
        raw_settings = data.get("settings", data)
        if not isinstance(raw_settings, dict):
            return jsonify({"error": "settings must be an object"}), 400

        mode = str(raw_settings.get("BACKEND_MODE", _active_generation_mode())).strip().lower()
        if mode not in {"local", "hybrid", "groq", "gemini", "openrouter"}:
            return jsonify({"error": "BACKEND_MODE must be local, hybrid, groq, gemini, or openrouter"}), 400

        updates = {}
        errors = {}
        for key in _SETTING_DEFS:
            if key not in raw_settings:
                continue
            if key in _SENSITIVE_SETTING_KEYS and raw_settings[key] == _REDACTED_VALUE:
                continue
            try:
                updates[key] = _coerce_setting(key, raw_settings[key])
            except (TypeError, ValueError) as e:
                errors[key] = str(e)

        if errors:
            return jsonify({"error": "Invalid setting value", "details": errors}), 400

        selected_model = str(raw_settings.get("ACTIVE_MODEL", "") or "").strip()
        _cloud_ids = {_GROQ_MODEL_OPTION, _GEMINI_MODEL_OPTION, _OPENROUTER_MODEL_OPTION}
        is_cloud = selected_model in _cloud_ids or selected_model.startswith("groq:") or selected_model.startswith("gemini:") or selected_model.startswith("openrouter:")
        if selected_model and not is_cloud and mode not in {"groq", "gemini", "openrouter"}:
            resolved = resolve_model_path(selected_model)
            if not resolved.lower().endswith(".gguf"):
                return jsonify({"error": "ACTIVE_MODEL must be a .gguf local model, Groq API, Gemini API, or OpenRouter API"}), 400
            if not os.path.isfile(resolved):
                return jsonify({"error": f"Model file not found: {resolved}"}), 400
            updates["LLAMA_MODEL_PATH"] = resolved
        elif selected_model and selected_model.startswith("groq:"):
            updates["GROQ_MODEL"] = _groq_model_from_selection(selected_model)
        elif selected_model and selected_model.startswith("gemini:"):
            updates["GEMINI_MODEL"] = _gemini_model_from_selection(selected_model)
        elif selected_model and selected_model.startswith("openrouter:"):
            updates["OPENROUTER_MODEL"] = _openrouter_model_from_selection(selected_model)
        elif mode not in {"groq", "gemini", "openrouter"} and "LLAMA_MODEL_PATH" in updates and updates["LLAMA_MODEL_PATH"]:
            resolved = resolve_model_path(updates["LLAMA_MODEL_PATH"])
            if not resolved.lower().endswith(".gguf"):
                return jsonify({"error": "LLAMA_MODEL_PATH must point to a .gguf file"}), 400
            if not os.path.isfile(resolved):
                return jsonify({"error": f"Model file not found: {resolved}"}), 400
            updates["LLAMA_MODEL_PATH"] = resolved

        primary_key = updates.get("GROQ_API_KEY", config.GROQ_API_KEY)
        raw_keys = updates.get("GROQ_API_KEYS", ",".join(config.GROQ_API_KEYS))
        has_groq_key = bool(primary_key or raw_keys.strip())
        if mode in {"hybrid", "groq"} and not has_groq_key:
            return jsonify({"error": "Groq API key is required for hybrid or full Groq mode"}), 400

        gemini_key = updates.get("GEMINI_API_KEY", config.GEMINI_API_KEY)
        if mode == "gemini" and not gemini_key:
            return jsonify({"error": "Gemini API key is required for Gemini mode"}), 400

        openrouter_key = updates.get("OPENROUTER_API_KEY", config.OPENROUTER_API_KEY)
        if mode == "openrouter" and not openrouter_key:
            return jsonify({"error": "OpenRouter API key is required for OpenRouter mode"}), 400

        updates["BACKEND_MODE"] = mode
        updates["USE_CLOUD_MODEL"] = mode == "hybrid"
        runtime_updates = {**updates, "BACKEND_MODE": mode}
        _write_env_values(updates)
        _apply_runtime_settings(runtime_updates)
        _model_health_cache.clear()

        return jsonify({
            "status": "ok",
            **_settings_payload(),
        })

    @app.route("/api/models", methods=["GET"])
    def list_models():
        """List available local GGUF models plus cloud API options."""
        global _selected_local_model_path, _selected_backend
        models = _list_local_models()
        if models:
            active_local = to_model_id(_selected_local_model_path)
            if active_local not in models:
                _selected_local_model_path = resolve_model_path(models[0])
        elif _selected_backend not in {"groq", "gemini", "openrouter"}:
            _selected_backend = "local"

        groq_model_ids = [f"groq:{m['id']}" for m in config.GROQ_MODELS]
        gemini_model_ids = [f"gemini:{m['id']}" for m in getattr(config, "GEMINI_MODELS", [])]
        openrouter_model_ids = [f"openrouter:{m['id']}" for m in getattr(config, "OPENROUTER_MODELS", [])]

        available_models = [
            *models,
            *groq_model_ids,
            *gemini_model_ids,
            *openrouter_model_ids,
        ]

        active = _active_model_selection()

        return jsonify({
            "models": available_models,
            "active": active,
            "groq_model": config.GROQ_MODEL,
            "groq_models": config.GROQ_MODELS,
            "gemini_model": config.GEMINI_MODEL,
            "gemini_models": getattr(config, "GEMINI_MODELS", []),
            "openrouter_model": config.OPENROUTER_MODEL,
            "openrouter_models": getattr(config, "OPENROUTER_MODELS", []),
            "local_models": models,
            "local_model": to_model_id(_selected_local_model_path) if _selected_local_model_path else "",
            "backend_mode": _active_generation_mode(),
        })

    @app.route("/api/models/switch", methods=["POST"])
    def switch_model():
        """Switch active model between local GGUF, Groq API, Gemini API, and OpenRouter API."""
        global _selected_local_model_path, _selected_backend
        data = request.json or {}
        model = (data.get("model", "") or "").strip()
        if not model:
            return jsonify({"error": "No model specified"}), 400

        if model == "hybrid" or model.startswith("hybrid:"):
            if not config.GROQ_API_KEY:
                return jsonify({
                    "error": "GROQ_API_KEY not set. Hybrid mode requires a Groq key for the Writer agent.",
                }), 400
            _selected_backend = "local"
            config.USE_CLOUD_MODEL = True
            if model.startswith("hybrid:groq:"):
                config.GROQ_MODEL = model.split("hybrid:groq:", 1)[1]
            return jsonify({
                "status": "ok",
                "active": "hybrid",
                "provider": "hybrid",
                "groq_model": config.GROQ_MODEL,
                "local_model": to_model_id(_selected_local_model_path),
            })

        if model == _GROQ_MODEL_OPTION or model.startswith("groq:"):
            if not config.GROQ_API_KEY:
                return jsonify({
                    "error": "GROQ_API_KEY not set. Add it to .env before selecting Groq.",
                }), 400
            groq_model = _groq_model_from_selection(model)
            _selected_backend = "groq"
            config.GROQ_MODEL = groq_model
            config.USE_CLOUD_MODEL = False
            logger.info("Switched backend to Groq API (%s)", groq_model)
            return jsonify({
                "status": "ok",
                "active": f"groq:{groq_model}",
                "provider": "groq",
                "groq_model": groq_model,
            })

        if model == _GEMINI_MODEL_OPTION or model.startswith("gemini:"):
            if not config.GEMINI_API_KEY:
                return jsonify({
                    "error": "GEMINI_API_KEY not set. Add it to .env before selecting Gemini.",
                }), 400
            gemini_model = _gemini_model_from_selection(model)
            _selected_backend = "gemini"
            config.GEMINI_MODEL = gemini_model
            config.USE_CLOUD_MODEL = False
            logger.info("Switched backend to Gemini API (%s)", gemini_model)
            return jsonify({
                "status": "ok",
                "active": f"gemini:{gemini_model}",
                "provider": "gemini",
                "gemini_model": gemini_model,
            })

        if model == _OPENROUTER_MODEL_OPTION or model.startswith("openrouter:"):
            if not config.OPENROUTER_API_KEY:
                return jsonify({
                    "error": "OPENROUTER_API_KEY not set. Add it to .env before selecting OpenRouter.",
                }), 400
            openrouter_model = _openrouter_model_from_selection(model)
            _selected_backend = "openrouter"
            config.OPENROUTER_MODEL = openrouter_model
            config.USE_CLOUD_MODEL = False
            logger.info("Switched backend to OpenRouter API (%s)", openrouter_model)
            return jsonify({
                "status": "ok",
                "active": f"openrouter:{openrouter_model}",
                "provider": "openrouter",
                "openrouter_model": openrouter_model,
            })

        resolved = resolve_model_path(model)
        if not resolved.lower().endswith(".gguf"):
            return jsonify({"error": "Model must be a .gguf file"}), 400
        if not os.path.isfile(resolved):
            return jsonify({"error": f"Model file not found: {resolved}"}), 400

        ok, probe_error = _probe_local_model(resolved, force=True)
        if not ok:
            return jsonify({"error": probe_error}), 400

        _selected_backend = "local"
        config.USE_CLOUD_MODEL = False
        _selected_local_model_path = resolved
        config.LLAMA_MODEL_PATH = resolved
        active = to_model_id(resolved)
        logger.info(f"Switched local model to: {active}")
        return jsonify({"status": "ok", "active": active, "provider": "local"})

    # ─── API: Projects ────────────────────────────────────────────
    @app.route("/api/projects", methods=["GET"])
    def list_projects():
        projects = []
        if os.path.exists(config.PROJECTS_DIR):
            for name in sorted(os.listdir(config.PROJECTS_DIR)):
                state_path = os.path.join(config.PROJECTS_DIR, name, "state.json")
                if os.path.exists(state_path):
                    try:
                        with open(state_path, encoding="utf-8") as f:
                            state = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        logger.warning("Skipping unreadable project state: %s", state_path)
                        continue
                    meta = state.get("metadata", {})
                    # Calculate total chapters and word count from disk and state
                    chap_dir = os.path.join(config.PROJECTS_DIR, name, "chapters")
                    chap_files = []
                    if os.path.exists(chap_dir):
                        chap_files = [f for f in os.listdir(chap_dir) if f.startswith("chapter_") and (f.endswith(".md") or f.endswith(".txt"))]

                    total_chapters = len(chap_files) or meta.get("current_chapter", 0)
                    word_cnt = meta.get("word_count", 0)
                    if not word_cnt and os.path.exists(chap_dir):
                        for f in chap_files:
                            try:
                                with open(os.path.join(chap_dir, f), encoding="utf-8") as cf:
                                    word_cnt += len(cf.read().split())
                            except Exception:
                                pass

                    projects.append({
                        "name": name,
                        "title": meta.get("title") or name.replace("_", " ").title(),
                        "genre": meta.get("genre", "Fiction"),
                        "premise": meta.get("premise", ""),
                        "setting": meta.get("setting", ""),
                        "themes": meta.get("themes", []),
                        "current_chapter": meta.get("current_chapter", total_chapters),
                        "total_chapters": total_chapters,
                        "total_scenes": meta.get("total_scenes_written", 0),
                        "characters": list(state.get("characters", {}).keys()),
                        "word_count": word_cnt,
                        "created_at": meta.get("created_at", ""),
                    })
        return jsonify({"projects": projects, "total": len(projects)})

    @app.route("/api/project/create", methods=["POST"])
    def create_project():
        data = request.json or {}
        title = (data.get("title", "Untitled") or "Untitled").strip()[:120]
        genre = data.get("genre", "fiction")
        premise = (data.get("premise", "") or "").strip()[:8000]
        characters = data.get("characters", {})
        themes = data.get("themes", [])
        setting = (data.get("setting", "") or "").strip()[:1000]

        project_name = normalize_project_name(title)
        try:
            pipeline = PipelineOrchestrator(project_name, **_current_pipeline_kwargs())
            state = pipeline.create_project(
                title=title, genre=genre, premise=premise,
                characters=characters, themes=themes, setting=setting,
            )
            return jsonify({"status": "ok", "project": project_name, "state": state})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/project/<name>", methods=["GET"])
    def get_project(name):
        name = normalize_project_name(name)
        project_dir = os.path.join(config.PROJECTS_DIR, name)
        state_path = os.path.join(project_dir, "state.json")

        if not os.path.exists(state_path):
            return jsonify({"error": f"Project '{name}' not found"}), 404

        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            return jsonify({"error": f"Cannot read project state: {e}"}), 500

        meta = state.get("metadata", {})
        info = {
            "name": name,
            "title": meta.get("title", name),
            "genre": meta.get("genre", ""),
            "current_chapter": meta.get("current_chapter", 0),
            "total_scenes": meta.get("total_scenes_written", 0),
        }

        combine_data = None
        polished_path = os.path.join(project_dir, "combined_polished.md")
        analysis_path = os.path.join(project_dir, "story_analysis.md")
        original_path = os.path.join(project_dir, "combined_original.md")

        if os.path.exists(polished_path) and os.path.exists(analysis_path):
            try:
                with open(polished_path, "r", encoding="utf-8") as f:
                    polished_text = f.read()
                with open(analysis_path, "r", encoding="utf-8") as f:
                    analysis_text = f.read()
                original_chars = os.path.getsize(original_path) if os.path.exists(original_path) else 0

                combine_data = {
                    "analysis": analysis_text,
                    "revised": polished_text,
                    "original_chars": original_chars,
                    "revised_chars": len(polished_text),
                    "model": "Previously Combined"
                }
            except Exception as e:
                logger.warning("Could not read combine data for %s: %s", name, e)

        return jsonify({"info": info, "state": state, "combine_data": combine_data})

    @app.route("/api/project/<name>/state", methods=["GET"])
    def get_project_state(name):
        name = normalize_project_name(name)
        state_path = os.path.join(config.PROJECTS_DIR, name, "state.json")
        if not os.path.exists(state_path):
            return jsonify({"error": f"Project '{name}' not found"}), 404
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            return jsonify({"state": state})
        except (OSError, json.JSONDecodeError) as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/state", methods=["PUT"])
    def update_project_state(name):
        name = normalize_project_name(name)
        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            pipeline.load_project()
            updates = request.json
            pipeline.state_manager.apply_state_update(updates)
            return jsonify({"status": "ok", "state": pipeline.get_state()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/premise/generate", methods=["POST"])
    def generate_premise(name):
        name = normalize_project_name(name)
        try:
            data = request.json or {}
            characters = data.get("characters", {})
            setting = data.get("setting", "")
            themes = data.get("themes", "")
            idea = data.get("idea", "")
            requested_model = data.get("model")

            llm = _get_llm_for_premise(requested_model)

            # Build characters context
            chars_txt = ""
            if isinstance(characters, dict):
                for cname, cinfo in characters.items():
                    desc = cinfo.get("description", "")
                    traits = ", ".join(cinfo.get("traits", []))
                    chars_txt += f"- {cname}: {desc} (Traits: {traits})\n"
            elif isinstance(characters, list):
                for c in characters:
                    chars_txt += f"- {c.get('name')}: {c.get('description')} (Traits: {', '.join(c.get('traits', []))})\n"

            prompt = (
                "You are a master story architect. Your task is to transform a rough story idea/timeline and character/setting details "
                "into a clean, highly structured, sequential story premise outline.\n\n"
                "=== INPUT DETAILS ===\n"
                f"CHARACTERS:\n{chars_txt}\n"
                f"SETTING: {setting}\n"
                f"THEMES: {themes}\n\n"
                f"ROUGH STORY IDEA / TIMELINE:\n{idea}\n\n"
                "=== REQUIREMENTS ===\n"
                "1. Output the premise as a clean list of sequential story steps (one step per line).\n"
                "2. Ensure the steps flow in strict chronological order, starting from the introduction and building logically to the climax and resolution.\n"
                "3. Do NOT include act divisions, chapter headings, metadata, commentary, or short lines with colons.\n"
                "4. Each step should be a clear, concrete event or scene (about 1-3 sentences describing who does what and the consequence).\n"
                "5. Aim for 8 to 20 steps, depending on the complexity of the story.\n"
                "6. Output ONLY the story steps list. Do not write any introduction (like 'Here is the premise') or wrap the output in markdown code blocks."
            )

            response = llm.generate(prompt=prompt, system="You are an expert developer and novelist planning a structured outline.")
            content = response.content or ""
            
            # Parse content into steps
            steps = PipelineOrchestrator._premise_steps(content)
            return jsonify({"status": "ok", "steps": steps})
        except Exception as e:
            logger.exception("Error in generate_premise")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/premise/refine", methods=["POST"])
    def refine_premise(name):
        name = normalize_project_name(name)
        try:
            data = request.json or {}
            steps = data.get("steps", [])
            characters = data.get("characters", {})
            setting = data.get("setting", "")
            themes = data.get("themes", "")
            requested_model = data.get("model")

            llm = _get_llm_for_premise(requested_model)

            # Build characters and current steps context
            chars_txt = ""
            if isinstance(characters, dict):
                for cname, cinfo in characters.items():
                    desc = cinfo.get("description", "")
                    traits = ", ".join(cinfo.get("traits", []))
                    chars_txt += f"- {cname}: {desc} (Traits: {traits})\n"
            
            steps_txt = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))

            prompt = (
                "You are a developmental editor. Your task is to refine, correct, and optimize the chronological flow of a story premise.\n\n"
                "=== INPUT DETAILS ===\n"
                f"CHARACTERS:\n{chars_txt}\n"
                f"SETTING: {setting}\n"
                f"THEMES: {themes}\n\n"
                f"CURRENT STORY PREMISE STEPS:\n{steps_txt}\n\n"
                "=== REQUIREMENTS ===\n"
                "1. Analyze the steps for narrative drift, logical gaps, pacing issues, or premature conflict resolutions.\n"
                "2. Correct any character inconsistencies, rename errors, or spelling issues based on the character profiles.\n"
                "3. Ensure the events are in the absolute correct chronological order of how they must be generated.\n"
                "4. Output the refined premise as a clean list of sequential story steps (one step per line).\n"
                "5. Do NOT include headings, act divisions, metadata, or commentary.\n"
                "6. Output ONLY the story steps list. Do not write any introductory/concluding text or wrap in code blocks."
            )

            response = llm.generate(prompt=prompt, system="You are an expert developmental editor focusing on pacing and continuity.")
            content = response.content or ""
            
            refined_steps = PipelineOrchestrator._premise_steps(content)
            return jsonify({"status": "ok", "steps": refined_steps})
        except Exception as e:
            logger.exception("Error in refine_premise")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/premise/expand", methods=["POST"])
    def expand_premise(name):
        name = normalize_project_name(name)
        try:
            data = request.json or {}
            steps = data.get("steps", [])
            characters = data.get("characters", {})
            setting = data.get("setting", "")
            themes = data.get("themes", "")
            requested_model = data.get("model")

            llm = _get_llm_for_premise(requested_model)

            # Build characters and current steps context
            chars_txt = ""
            if isinstance(characters, dict):
                for cname, cinfo in characters.items():
                    desc = cinfo.get("description", "")
                    traits = ", ".join(cinfo.get("traits", []))
                    chars_txt += f"- {cname}: {desc} (Traits: {traits})\n"
            
            steps_txt = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))

            prompt = (
                "You are a creative writer. Your task is to expand an existing story premise by introducing new intermediate events, scenes, or beats.\n\n"
                "=== INPUT DETAILS ===\n"
                f"CHARACTERS:\n{chars_txt}\n"
                f"SETTING: {setting}\n"
                f"THEMES: {themes}\n\n"
                f"CURRENT STORY PREMISE STEPS:\n{steps_txt}\n\n"
                "=== REQUIREMENTS ===\n"
                "1. Creatively invent and insert new intermediate steps/scenes between the existing steps to flesh out the narrative arc.\n"
                "2. Focus on character relationships, build suspense/conflict gradually, and add detailed pacing beats.\n"
                "3. Maintain absolute continuity with the existing steps — do NOT alter their key outcomes, but insert setup or reaction scenes between them.\n"
                "4. Output the expanded premise as a clean list of sequential story steps (one step per line).\n"
                "5. Do NOT include headings, act divisions, metadata, or commentary.\n"
                "6. Output ONLY the story steps list. Do not write any introductory/concluding text or wrap in code blocks."
            )

            response = llm.generate(prompt=prompt, system="You are an expert creative novelist fleshing out outlines.")
            content = response.content or ""
            
            expanded_steps = PipelineOrchestrator._premise_steps(content)
            return jsonify({"status": "ok", "steps": expanded_steps})
        except Exception as e:
            logger.exception("Error in expand_premise")
            return jsonify({"error": str(e)}), 500

    # ─── API: Chapters ────────────────────────────────────────────
    @app.route("/api/project/<name>/chapters", methods=["GET"])
    def list_chapters(name):
        name = normalize_project_name(name)
        chapters_dir = os.path.join(config.PROJECTS_DIR, name, "chapters")
        chapters = []
        
        # Track which chapters we've already seen as completed
        seen_nums = set()
        
        if os.path.exists(chapters_dir):
            # 1. Completed chapters
            for f in sorted(os.listdir(chapters_dir)):
                match = _CHAPTER_FILE_RE.match(f)
                if not match:
                    continue
                num = int(match.group(1))
                seen_nums.add(num)
                path = os.path.join(chapters_dir, f)
                with open(path, encoding="utf-8") as fh:
                    content = fh.read()
                words = len(content.split())
                chapters.append({
                    "number": num,
                    "num": num,
                    "title": f"Chapter {num}",
                    "filename": f,
                    "words": words,
                    "status": "completed"
                })
            
            # 2. WIP chapters
            wip_re = re.compile(r"^\.wip_chapter_(\d{3,})\.json$")
            for f in sorted(os.listdir(chapters_dir)):
                match = wip_re.match(f)
                if not match:
                    continue
                num = int(match.group(1))
                if num in seen_nums:
                    continue
                
                path = os.path.join(chapters_dir, f)
                try:
                    with open(path, encoding="utf-8") as fh:
                        wip_data = json.load(fh)
                    scenes = wip_data.get("completed_scenes", [])
                    words = sum(len(s.get("text", "").split()) for s in scenes)
                    chapters.append({
                        "number": num,
                        "num": num,
                        "title": f"Chapter {num} (In Progress)",
                        "filename": f,
                        "words": words,
                        "status": "writing",
                        "scenes_completed": len(scenes),
                        "scenes_total": len(wip_data.get("scene_plans", []))
                    })
                except Exception:
                    continue

        # Sort by number
        chapters.sort(key=lambda x: x["number"])
        return jsonify({"chapters": chapters})

    @app.route("/api/project/<name>/chapter/<int:num>", methods=["GET"])
    def read_chapter(name, num):
        name = normalize_project_name(name)
        chapters_dir = os.path.join(config.PROJECTS_DIR, name, "chapters")
        chapter_path = os.path.join(chapters_dir, f"chapter_{num:03d}.md")
        if not os.path.exists(chapter_path):
            chapter_path = os.path.join(chapters_dir, f"chapter_{num:03d}.txt")
        if not os.path.exists(chapter_path):
            return jsonify({"error": "Chapter not found"}), 404
        try:
            with open(chapter_path, encoding="utf-8") as f:
                text = f.read()
            return jsonify({"chapter": num, "content": text, "title": f"Chapter {num}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/chapter/<int:num>", methods=["PUT"])
    def update_chapter(name, num):
        """Update the content of a chapter (direct text editing from reader)."""
        name = normalize_project_name(name)
        data = request.json or {}
        content = (data.get("content", "") or "").strip()
        if not content:
            return jsonify({"error": "content is required and cannot be empty"}), 400
        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            pipeline.load_project()
            result = pipeline.update_chapter_content(num, content)
            return jsonify({"status": "ok", **result})
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/chapters/delete", methods=["POST"])
    def delete_chapters(name):
        """
        Delete chapter N and all later chapters, then sync state/WIP/vector memory.
        Body: {"from_chapter": <int>} (alias: "chapter")
        """
        name = normalize_project_name(name)
        data = request.json or {}
        from_chapter = data.get("from_chapter", data.get("chapter"))
        try:
            from_chapter = int(from_chapter)
        except (TypeError, ValueError):
            return jsonify({"error": "from_chapter must be an integer >= 1"}), 400
        if from_chapter < 1:
            return jsonify({"error": "from_chapter must be >= 1"}), 400

        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            pipeline.load_project()
            result = pipeline.delete_chapters_from(from_chapter)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ─── API: Generation (SSE) ────────────────────────────────────
    @app.route("/api/project/<name>/generate", methods=["POST"])
    def start_generation(name):
        name = normalize_project_name(name)
        data = request.json or {}
        pacing = data.get("pacing", "moderate")
        try:
            chapter_count = int(data.get("chapter_count", 1))
        except (TypeError, ValueError):
            chapter_count = 1
        if chapter_count not in (-1, -2):
            chapter_count = max(1, min(chapter_count, 20))  # Clamp 1-20
        requested_model = (data.get("model", "") or "").strip()
        pipeline_kwargs, err_resp, err_code = _resolve_generation_pipeline_kwargs(requested_model)
        if err_resp is not None:
            return err_resp, err_code

        with _generation_lock:
            if name in _active_pipelines:
                return jsonify({"error": "Generation already active"}), 409
            # Reserve the slot synchronously. The real pipeline is built in a
            # worker thread; without a placeholder two rapid requests could
            # both pass the check above and generate concurrently.
            _active_pipelines[name] = None
            _cancel_requests.discard(name)
            eq = queue.Queue(maxsize=config.SSE_QUEUE_MAXSIZE)
            _event_queues[name] = eq

        def progress_cb(event, data=None, **kwargs):
            msg = _normalize_event(event, data)
            if not _queue_event(eq, msg):
                logger.warning(f"SSE queue full for project '{name}'. Dropping event '{event}'.")

        def run_pipeline():
            try:
                pipeline = PipelineOrchestrator(
                    name,
                    progress_cb,
                    **pipeline_kwargs,
                )
                pipeline.load_project()
                with _generation_lock:
                    cancel_requested = name in _cancel_requests
                    if not cancel_requested:
                        _active_pipelines[name] = pipeline
                if cancel_requested:
                    _queue_event(eq, _normalize_event("done", {
                        "status": "cancelled",
                        "message": "Generation cancelled before it started.",
                    }), force=True)
                    return
                if chapter_count > 1 or chapter_count in (-1, -2):
                    results = pipeline.generate_chapters(count=chapter_count, pacing=pacing)
                    _queue_event(eq, _normalize_event("done", results[-1] if results else {}), force=True)
                else:
                    result = pipeline.generate_chapter(pacing=pacing)
                    _queue_event(eq, _normalize_event("done", result), force=True)
            except PipelineCancelledError:
                _queue_event(eq, _normalize_event("done", {
                    "status": "cancelled",
                    "message": "Generation cancelled by user.",
                }), force=True)
            except Exception as e:
                _queue_event(eq, _normalize_event("error", {"error": str(e)}), force=True)
            finally:
                with _generation_lock:
                    _active_pipelines.pop(name, None)

        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()

        return jsonify({"status": "started", "chapters": chapter_count})

    @app.route("/api/project/<name>/generate/manual", methods=["POST"])
    def start_manual_generation(name):
        name = normalize_project_name(name)
        data = request.json or {}
        pacing = data.get("pacing", "moderate")
        chapter_title = (data.get("chapter_title", "") or "").strip()[:220]
        try:
            scene_count = int(data.get("scene_count", 1))
        except (TypeError, ValueError):
            scene_count = 1
        scene_count = max(1, min(scene_count, 20))

        raw_scenes = data.get("scenes", [])
        if not isinstance(raw_scenes, list):
            return jsonify({"error": "scenes must be a list of scene briefs"}), 400
        scenes = [str(scene).strip() for scene in raw_scenes if str(scene).strip()]

        if not chapter_title:
            return jsonify({"error": "chapter_title is required"}), 400
        if len(scenes) < scene_count:
            return jsonify({
                "error": f"Provide {scene_count} non-empty scene briefs before generating.",
            }), 400

        requested_model = (data.get("model", "") or "").strip()
        pipeline_kwargs, err_resp, err_code = _resolve_generation_pipeline_kwargs(requested_model)
        if err_resp is not None:
            return err_resp, err_code

        with _generation_lock:
            if name in _active_pipelines:
                return jsonify({"error": "Generation already active"}), 409
            # Reserve the slot synchronously (same rationale as start_generation).
            _active_pipelines[name] = None
            _cancel_requests.discard(name)
            eq = queue.Queue(maxsize=config.SSE_QUEUE_MAXSIZE)
            _event_queues[name] = eq

        final_scenes = scenes[:scene_count]

        def progress_cb(event, data=None, **kwargs):
            msg = _normalize_event(event, data)
            if not _queue_event(eq, msg):
                logger.warning(f"SSE queue full for project '{name}'. Dropping event '{event}'.")

        def run_pipeline():
            try:
                pipeline = PipelineOrchestrator(
                    name,
                    progress_cb,
                    **pipeline_kwargs,
                )
                pipeline.load_project()
                with _generation_lock:
                    cancel_requested = name in _cancel_requests
                    if not cancel_requested:
                        _active_pipelines[name] = pipeline
                if cancel_requested:
                    _queue_event(eq, _normalize_event("done", {
                        "status": "cancelled",
                        "message": "Generation cancelled before it started.",
                    }), force=True)
                    return
                result = pipeline.generate_chapter_manual(
                    chapter_title=chapter_title,
                    scene_briefs=final_scenes,
                    pacing=pacing,
                )
                _queue_event(eq, _normalize_event("done", result), force=True)
            except PipelineCancelledError:
                _queue_event(eq, _normalize_event("done", {
                    "status": "cancelled",
                    "message": "Generation cancelled by user.",
                }), force=True)
            except Exception as e:
                _queue_event(eq, _normalize_event("error", {"error": str(e)}), force=True)
            finally:
                with _generation_lock:
                    _active_pipelines.pop(name, None)

        thread = threading.Thread(target=run_pipeline, daemon=True)
        thread.start()

        return jsonify({"status": "started", "chapters": 1, "scenes": len(final_scenes)})

    # ─── API: Interactive Manual Generation (Scene-by-Scene) ──────
    @app.route("/api/project/<name>/generate/manual/start", methods=["POST"])
    def start_manual_session_api(name):
        """Start an interactive manual chapter session."""
        name = normalize_project_name(name)
        data = request.json or {}
        chapter_title = (data.get("chapter_title", "") or "").strip()[:220]
        pacing = data.get("pacing", "moderate")

        if not chapter_title:
            return jsonify({"error": "chapter_title is required"}), 400

        requested_model = (data.get("model", "") or "").strip()
        pipeline_kwargs, err_resp, err_code = _resolve_generation_pipeline_kwargs(requested_model)
        if err_resp is not None:
            return err_resp, err_code

        with _generation_lock:
            if name in _manual_sessions:
                return jsonify({"error": "A manual session is already active for this project"}), 409
            if name in _active_pipelines:
                return jsonify({"error": "Generation already active"}), 409
            # Reserve synchronously: pipeline construction below is slow and
            # must not let a second request slip past this check.
            _manual_sessions[name] = None

        try:
            eq = queue.Queue(maxsize=config.SSE_QUEUE_MAXSIZE)

            def progress_cb(event, data=None, **kwargs):
                msg = _normalize_event(event, data)
                if not _queue_event(eq, msg):
                    logger.warning(f"SSE queue full for manual session '{name}'.")

            pipeline = PipelineOrchestrator(name, progress_cb, **pipeline_kwargs)
            pipeline.load_project()
            session = pipeline.start_manual_session(
                chapter_title=chapter_title,
                pacing=pacing,
            )

            with _generation_lock:
                _manual_sessions[name] = {
                    "pipeline": pipeline,
                    "session": session,
                    "eq": eq,
                    "progress_cb": progress_cb,
                }
                _event_queues[name] = eq

            return jsonify({
                "status": "started",
                "chapter_num": session["chapter_num"],
                "chapter_title": session["chapter_title"],
            })
        except Exception as e:
            with _generation_lock:
                # Release the reservation; keep a fully registered session.
                if _manual_sessions.get(name) is None:
                    _manual_sessions.pop(name, None)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/generate/manual/scene", methods=["POST"])
    def generate_manual_scene_api(name):
        """Generate the next scene in the interactive manual session."""
        name = normalize_project_name(name)
        data = request.json or {}
        scene_brief = (data.get("scene_brief", "") or "").strip()

        if not scene_brief:
            return jsonify({"error": "scene_brief is required"}), 400

        with _generation_lock:
            ms = _manual_sessions.get(name)
            if not ms:
                return jsonify({"error": "No active manual session. Call /manual/start first."}), 404
            if name in _active_pipelines:
                return jsonify({"error": "A scene is currently being generated. Wait for it to finish."}), 409

        pipeline = ms["pipeline"]
        session = ms["session"]
        eq = ms["eq"]

        with _generation_lock:
            _cancel_requests.discard(name)
            _active_pipelines[name] = pipeline

        def run_scene():
            try:
                result = pipeline.generate_manual_scene(
                    session=session,
                    scene_brief=scene_brief,
                )
                _queue_event(eq, _normalize_event("manual_scene_done", result), force=True)
            except PipelineCancelledError:
                _queue_event(eq, _normalize_event("manual_scene_done", {
                    "status": "cancelled",
                    "scene_number": session.get("scene_counter", 0) + 1,
                    "scenes_completed": session.get("scene_counter", 0),
                    "completed_scenes": session.get("completed_scenes", []),
                }), force=True)
            except Exception as e:
                _queue_event(eq, _normalize_event("manual_scene_done", {
                    "status": "error",
                    "error": str(e),
                    "scene_number": session.get("scene_counter", 0) + 1,
                    "scenes_completed": session.get("scene_counter", 0),
                    "completed_scenes": session.get("completed_scenes", []),
                }), force=True)
            finally:
                with _generation_lock:
                    _active_pipelines.pop(name, None)

        thread = threading.Thread(target=run_scene, daemon=True)
        thread.start()

        scene_number = session["scene_counter"] + 1
        return jsonify({"status": "started", "scene_number": scene_number})

    @app.route("/api/project/<name>/generate/manual/scene/typed", methods=["POST"])
    def add_typed_scene_api(name):
        """Add a user-typed scene directly to the manual session (no AI generation)."""
        name = normalize_project_name(name)
        data = request.json or {}
        text = (data.get("text", "") or "").strip()

        if not text:
            return jsonify({"error": "text is required and cannot be empty"}), 400

        with _generation_lock:
            ms = _manual_sessions.get(name)
            if not ms:
                return jsonify({"error": "No active manual session. Call /manual/start first."}), 404
            if name in _active_pipelines:
                return jsonify({"error": "A scene is currently being generated. Wait for it to finish."}), 409

        pipeline = ms["pipeline"]
        session = ms["session"]

        try:
            pipeline.add_typed_scene(session, text)
            words = len(text.split())
            return jsonify({
                "status": "added",
                "scene_number": session["scene_counter"],
                "words": words,
                "scenes_completed": session["scene_counter"],
                "completed_scenes": session["completed_scenes"],
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 400

    @app.route("/api/project/<name>/generate/manual/cancel_scene", methods=["POST"])
    def cancel_manual_scene_api(name):
        """Cancel only the currently generating scene."""
        name = normalize_project_name(name)
        with _generation_lock:
            pipeline = _active_pipelines.get(name)
            if pipeline:
                pipeline.cancel_scene()
                _cancel_requests.add(name)
                return jsonify({"status": "cancelling"})
            ms = _manual_sessions.get(name)
            if ms and ms.get("pipeline"):
                ms["pipeline"].cancel_scene()
                _cancel_requests.add(name)
                return jsonify({"status": "cancel_requested"})
            return jsonify({"status": "not_generating"})

    @app.route("/api/project/<name>/chapter/<int:num>/resume", methods=["POST"])
    def resume_manual_chapter_api(name, num):
        """Resume a saved chapter by turning it back into an active manual session."""
        name = normalize_project_name(name)
        with _generation_lock:
            if name in _active_pipelines:
                return jsonify({"error": "Generation in progress"}), 409
            if name in _manual_sessions:
                return jsonify({"error": "A manual session is already active for this project"}), 409
            # Reserve synchronously; construction happens outside the lock so
            # model init doesn't stall every other generation endpoint.
            _manual_sessions[name] = None

        try:
            eq = queue.Queue(maxsize=config.SSE_QUEUE_MAXSIZE)

            def progress_cb(event, data=None, **kwargs):
                msg = _normalize_event(event, data)
                if not _queue_event(eq, msg):
                    logger.warning(f"SSE queue full for manual session '{name}'.")

            pipeline = PipelineOrchestrator(name, progress_cb, **_current_pipeline_kwargs())
            session = pipeline.resume_manual_chapter(num)

            with _generation_lock:
                _manual_sessions[name] = {
                    "pipeline": pipeline,
                    "session": session,
                    "eq": eq,
                    "progress_cb": progress_cb,
                }
                _event_queues[name] = eq

            return jsonify({
                "status": "resumed",
                "chapter_num": session["chapter_num"],
                "chapter_title": session["chapter_title"],
                "scenes_completed": session["scene_counter"],
                "completed_scenes": session["completed_scenes"]
            })
        except Exception as e:
            with _generation_lock:
                # Release the reservation; keep a fully registered session.
                if _manual_sessions.get(name) is None:
                    _manual_sessions.pop(name, None)
            logger.error("Failed to resume chapter %s for '%s'", num, name, exc_info=True)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/generate/manual/finish", methods=["POST"])
    def finish_manual_chapter_api(name):
        """Finalize the interactive manual chapter."""
        name = normalize_project_name(name)

        with _generation_lock:
            ms = _manual_sessions.get(name)
            if not ms:
                return jsonify({"error": "No active manual session."}), 404
            if name in _active_pipelines:
                return jsonify({"error": "A scene is currently being generated. Wait for it to finish."}), 409

        pipeline = ms["pipeline"]
        session = ms["session"]
        eq = ms["eq"]

        with _generation_lock:
            _active_pipelines[name] = pipeline

        def run_finish():
            try:
                result = pipeline.finish_manual_chapter(session=session)
                _queue_event(eq, _normalize_event("done", result), force=True)
            except Exception as e:
                _queue_event(eq, _normalize_event("error", {"error": str(e)}), force=True)
            finally:
                with _generation_lock:
                    _active_pipelines.pop(name, None)
                    _manual_sessions.pop(name, None)

        thread = threading.Thread(target=run_finish, daemon=True)
        thread.start()

        return jsonify({"status": "finishing", "scenes_count": len(session["completed_scenes"])})

    @app.route("/api/project/<name>/generate/manual/scene/<int:index>", methods=["DELETE"])
    def delete_manual_scene_api(name, index):
        """Delete a generated scene from the active manual session."""
        name = normalize_project_name(name)
        with _generation_lock:
            ms = _manual_sessions.get(name)
            if not ms:
                return jsonify({"error": "No active manual session."}), 404
            if name in _active_pipelines:
                return jsonify({"error": "A scene is currently being generated. Wait for it to finish."}), 409
            
            pipeline = ms["pipeline"]
            session = ms["session"]
            try:
                pipeline.delete_manual_scene(session, index)
                return jsonify({
                    "status": "deleted",
                    "scenes_completed": session["scene_counter"],
                    "completed_scenes": session["completed_scenes"]
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(e)}), 400

    @app.route("/api/project/<name>/generate/manual/scene/<int:index>", methods=["PUT"])
    def update_manual_scene_api(name, index):
        """Update the text of a scene in the active manual session."""
        name = normalize_project_name(name)
        data = request.json or {}
        new_text = (data.get("text", "") or "").strip()
        if not new_text:
            return jsonify({"error": "text is required and cannot be empty"}), 400

        with _generation_lock:
            ms = _manual_sessions.get(name)
            if not ms:
                return jsonify({"error": "No active manual session."}), 404
            if name in _active_pipelines:
                return jsonify({"error": "A scene is currently being generated. Wait for it to finish."}), 409

            pipeline = ms["pipeline"]
            session = ms["session"]
            try:
                pipeline.update_manual_scene(session, index, new_text)
                return jsonify({
                    "status": "updated",
                    "scene_index": index,
                    "words": len(new_text.split()),
                    "scenes_completed": session["scene_counter"],
                    "completed_scenes": session["completed_scenes"],
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(e)}), 400

    @app.route("/api/project/<name>/generate/manual/status", methods=["GET"])
    def manual_session_status(name):
        """Get the current state of the manual session."""
        name = normalize_project_name(name)
        ms = _manual_sessions.get(name)
        if not ms:
            return jsonify({"active": False})
        session = ms["session"]
        return jsonify({
            "active": True,
            "chapter_num": session["chapter_num"],
            "chapter_title": session["chapter_title"],
            "scenes_completed": session["scene_counter"],
            "completed_scenes": session["completed_scenes"],
            "is_generating": name in _active_pipelines,
        })

    @app.route("/api/project/<name>/generate/stream", methods=["GET"])
    def stream_generation(name):
        name = normalize_project_name(name)
        def event_stream():
            eq = _event_queues.get(name)
            if not eq:
                yield f"data: {json.dumps(_normalize_event('error', {'error': 'No active generation'}))}\n\n"
                return
            try:
                while True:
                    try:
                        msg = eq.get(timeout=120)
                        yield f"data: {json.dumps(msg)}\n\n"
                        if msg.get("type") in ("done", "error"):
                            break
                    except queue.Empty:
                        yield f"data: {json.dumps(_normalize_event('heartbeat', {}))}\n\n"
            finally:
                with _generation_lock:
                    _event_queues.pop(name, None)

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/project/<name>/generate/cancel", methods=["POST"])
    def cancel_generation(name):
        name = normalize_project_name(name)
        with _generation_lock:
            _cancel_requests.add(name)
            pipeline = _active_pipelines.get(name)
            ms = _manual_sessions.get(name)
            if pipeline:
                pipeline.cancel()
                pipeline.cancel_scene()
            elif ms and ms.get("pipeline"):
                ms["pipeline"].cancel()
                ms["pipeline"].cancel_scene()
            eq = _event_queues.get(name)
            if eq:
                _queue_event(eq, _normalize_event("status", "Cancellation requested"))
            status = "cancelling" if pipeline or ms or eq else "cancel_requested"
        return jsonify({"status": status})

    # ─── API: Combine & Polish (Gemini) ─────────────────────────────
    @app.route("/api/project/<name>/combine", methods=["POST"])
    def start_combine(name):
        """Combine all chapters and send to Gemini for continuity polish."""
        name = normalize_project_name(name)

        if not config.GEMINI_API_KEY:
            return jsonify({
                "error": "GEMINI_API_KEY is not set. Add it to your .env file.",
            }), 400

        data = request.json or {}
        requested_model = data.get("model")
        mode = data.get("mode", "polish")

        with _generation_lock:
            if name in _active_combines:
                return jsonify({"error": "Combine already in progress"}), 409
            # Claim synchronously so concurrent requests can't double-start.
            _active_combines[name] = True
            eq = queue.Queue(maxsize=100)
            _combine_queues[name] = eq
            _combine_cancel_requests.discard(name)

        def progress_cb(event, data=None, **kwargs):
            msg = _normalize_event(event, data)
            _queue_event(eq, msg)

        def run_combine():
            try:
                pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
                pipeline.load_project()
                state = pipeline.get_state()
                metadata = state.get("metadata", {})
                chapters_dir = os.path.join(config.PROJECTS_DIR, name, "chapters")
                output_dir = os.path.join(config.PROJECTS_DIR, name)

                # Archive old files
                try:
                    timestamp = int(time.time())
                    for fname in ["combined_original.md", "combined_polished.md", "story_analysis.md"]:
                        path = os.path.join(output_dir, fname)
                        if os.path.exists(path):
                            base, ext = os.path.splitext(fname)
                            archive_path = os.path.join(output_dir, f"{base}_{timestamp}{ext}")
                            os.rename(path, archive_path)
                except Exception as e:
                    logger.warning("Could not archive old combine files: %s", e)

                # Step 1: Combine chapters
                _queue_event(eq, _normalize_event("combine_status", {
                    "step": "Combining chapters into a single document...",
                }))
                combined = ""
                try:
                    combined = combine_chapters(chapters_dir, metadata)
                except RuntimeError:
                    if mode != "whole":
                        raise
                    combined = ""

                # Save original combined file
                original_path = os.path.join(output_dir, "combined_original.md")
                with open(original_path, "w", encoding="utf-8") as f:
                    f.write(combined)

                _queue_event(eq, _normalize_event("combine_status", {
                    "step": f"Combined {len(combined)} characters. Sending to Gemini...",
                }))

                # Step 2: Analyze and polish with Gemini, or generate the whole
                # story in a single call.
                # Pass the premise and full state so Gemini can use canonical
                # character/world data as ground truth.
                story_premise = metadata.get("premise", "")
                if mode == "whole":
                    result = generate_whole_story(
                        combined, progress_cb, premise=story_premise, model_name=requested_model,
                        is_cancelled=lambda: name in _combine_cancel_requests,
                        state=state,
                    )
                else:
                    result = analyze_and_polish(
                        combined, progress_cb, premise=story_premise, model_name=requested_model,
                        is_cancelled=lambda: name in _combine_cancel_requests,
                        state=state,
                    )

                # Save polished file
                polished_path = os.path.join(output_dir, "combined_polished.md")
                with open(polished_path, "w", encoding="utf-8") as f:
                    f.write(result["final_story"])

                # Save analysis
                analysis_path = os.path.join(output_dir, "story_analysis.md")
                with open(analysis_path, "w", encoding="utf-8") as f:
                    f.write(result["analysis"])

                _combine_results[name] = {
                    "analysis": result["analysis"],
                    "model": result["model"],
                    "original_chars": len(combined),
                    "revised_chars": len(result["final_story"]),
                    "coverage_gaps": result.get("coverage_gaps", []),
                    "inserted_sections": result.get("inserted_sections", []),
                }

                _queue_event(eq, _normalize_event("combine_done", {
                    "analysis": result["analysis"],
                    "revised": result["final_story"],
                    "model": result["model"],
                    "original_chars": len(combined),
                    "revised_chars": len(result["final_story"]),
                    "coverage_gaps": result.get("coverage_gaps", []),
                    "inserted_sections": result.get("inserted_sections", []),
                }), force=True)

            except Exception as e:
                logger.error("Combine failed for %s: %s", name, e, exc_info=True)
                _queue_event(eq, _normalize_event("combine_error", {
                    "error": str(e),
                }), force=True)
            finally:
                with _generation_lock:
                    _active_combines.pop(name, None)
                    _combine_cancel_requests.discard(name)

        thread = threading.Thread(target=run_combine, daemon=True)
        thread.start()
        return jsonify({"status": "started"})

    @app.route("/api/project/<name>/combine/cancel", methods=["POST"])
    def cancel_combine(name):
        """Cancel the active story combination process."""
        name = normalize_project_name(name)
        _combine_cancel_requests.add(name)
        eq = _combine_queues.get(name)
        if eq:
            _queue_event(eq, _normalize_event("combine_error", {
                "error": "Cancellation requested by user",
            }), force=True)
        return jsonify({"status": "cancel_requested"})

    @app.route("/api/project/<name>/combine/stream", methods=["GET"])
    def stream_combine(name):
        """SSE stream for combine progress events."""
        name = normalize_project_name(name)

        def event_stream():
            eq = _combine_queues.get(name)
            if not eq:
                yield f"data: {json.dumps(_normalize_event('combine_error', {'error': 'No active combine'}))}\n\n"
                return
            try:
                while True:
                    try:
                        msg = eq.get(timeout=300)
                        yield f"data: {json.dumps(msg)}\n\n"
                        if msg.get("type") in ("combine_done", "combine_error"):
                            break
                    except queue.Empty:
                        yield f"data: {json.dumps(_normalize_event('heartbeat', {}))}\n\n"
            finally:
                with _generation_lock:
                    _combine_queues.pop(name, None)

        return Response(
            event_stream(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    def _list_combine_versions(name):
        project_dir = os.path.join(config.PROJECTS_DIR, name)
        if not os.path.exists(project_dir):
            return []
        
        versions = []
        for fname in os.listdir(project_dir):
            if fname.startswith("combined_polished") and fname.endswith(".md"):
                if fname == "combined_polished.md":
                    suffix = "latest"
                    label = "Latest (Active)"
                else:
                    suffix = fname[len("combined_polished_"):-len(".md")]
                    try:
                        timestamp_val = int(suffix)
                        from datetime import datetime
                        dt = datetime.fromtimestamp(timestamp_val)
                        label = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        label = suffix.replace("_", " ")

                polished_path = os.path.join(project_dir, fname)
                mtime = os.path.getmtime(polished_path)
                revised_chars = os.path.getsize(polished_path)
                
                s_part = "" if suffix == "latest" else f"_{suffix}"
                original_path = os.path.join(project_dir, f"combined_original{s_part}.md")
                analysis_path = os.path.join(project_dir, f"story_analysis{s_part}.md")
                
                original_chars = os.path.getsize(original_path) if os.path.exists(original_path) else 0
                has_analysis = os.path.exists(analysis_path)
                
                versions.append({
                    "suffix": suffix,
                    "label": label,
                    "timestamp": mtime,
                    "original_chars": original_chars,
                    "revised_chars": revised_chars,
                    "has_analysis": has_analysis
                })
        
        versions.sort(key=lambda x: x["timestamp"])
        return versions

    @app.route("/api/project/<name>/combine/versions", methods=["GET"])
    def list_combine_versions(name):
        name = normalize_project_name(name)
        try:
            versions = _list_combine_versions(name)
            return jsonify({"versions": versions})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/combine/version/<suffix>", methods=["GET"])
    def get_combine_version(name, suffix):
        name = normalize_project_name(name)
        project_dir = os.path.join(config.PROJECTS_DIR, name)
        
        s_part = "" if suffix == "latest" else f"_{suffix}"
        polished_path = os.path.join(project_dir, f"combined_polished{s_part}.md")
        analysis_path = os.path.join(project_dir, f"story_analysis{s_part}.md")
        original_path = os.path.join(project_dir, f"combined_original{s_part}.md")
        
        if not os.path.exists(polished_path):
            return jsonify({"error": f"Version {suffix} not found"}), 404
            
        try:
            with open(polished_path, "r", encoding="utf-8") as f:
                revised = f.read()
            
            analysis = ""
            if os.path.exists(analysis_path):
                with open(analysis_path, "r", encoding="utf-8") as f:
                    analysis = f.read()
                    
            original_chars = os.path.getsize(original_path) if os.path.exists(original_path) else 0
            
            if suffix == "latest":
                label = "Latest (Active)"
            else:
                try:
                    timestamp_val = int(suffix)
                    from datetime import datetime
                    dt = datetime.fromtimestamp(timestamp_val)
                    label = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    label = suffix.replace("_", " ")
                    
            return jsonify({
                "suffix": suffix,
                "label": label,
                "analysis": analysis,
                "revised": revised,
                "original_chars": original_chars,
                "revised_chars": len(revised),
                "model": "Gemini Combined"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/combine/rename", methods=["POST"])
    def rename_combine_version(name):
        name = normalize_project_name(name)
        data = request.json or {}
        suffix = data.get("suffix")
        new_name = data.get("new_name")
        
        if not suffix or not new_name:
            return jsonify({"error": "suffix and new_name are required"}), 400
            
        import re as re_mod
        new_suffix = re_mod.sub(r"[^\w\s-]", "", new_name).strip()
        new_suffix = re_mod.sub(r"[\s]+", "_", new_suffix)
        
        if not new_suffix:
            return jsonify({"error": "Invalid new name"}), 400
            
        project_dir = os.path.join(config.PROJECTS_DIR, name)
        s_part = "" if suffix == "latest" else f"_{suffix}"
        
        src_original = os.path.join(project_dir, f"combined_original{s_part}.md")
        src_polished = os.path.join(project_dir, f"combined_polished{s_part}.md")
        src_analysis = os.path.join(project_dir, f"story_analysis{s_part}.md")
        
        if not os.path.exists(src_polished):
            return jsonify({"error": f"Version {suffix} not found"}), 404
            
        tgt_original = os.path.join(project_dir, f"combined_original_{new_suffix}.md")
        tgt_polished = os.path.join(project_dir, f"combined_polished_{new_suffix}.md")
        tgt_analysis = os.path.join(project_dir, f"story_analysis_{new_suffix}.md")
        
        if os.path.exists(tgt_polished) and new_suffix != suffix:
            return jsonify({"error": f"Version with name '{new_name}' already exists"}), 400
            
        try:
            if os.path.exists(src_original):
                os.rename(src_original, tgt_original)
            if os.path.exists(src_polished):
                os.rename(src_polished, tgt_polished)
            if os.path.exists(src_analysis):
                os.rename(src_analysis, tgt_analysis)
                
            return jsonify({"status": "ok", "new_suffix": new_suffix, "new_label": new_name})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/combine/delete_version", methods=["POST"])
    def delete_combine_version(name):
        name = normalize_project_name(name)
        data = request.json or {}
        suffix = data.get("suffix")
        
        if not suffix:
            return jsonify({"error": "suffix is required"}), 400
            
        project_dir = os.path.join(config.PROJECTS_DIR, name)
        s_part = "" if suffix == "latest" else f"_{suffix}"
        
        files = [
            os.path.join(project_dir, f"combined_original{s_part}.md"),
            os.path.join(project_dir, f"combined_polished{s_part}.md"),
            os.path.join(project_dir, f"story_analysis{s_part}.md")
        ]
        
        deleted = False
        try:
            for f in files:
                if os.path.exists(f):
                    os.remove(f)
                    deleted = True
            return jsonify({"status": "ok", "deleted": deleted})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/combine/download/<file_type>", methods=["GET"])
    def download_combined(name, file_type):
        """Download the combined story file (original or polished)."""
        name = normalize_project_name(name)
        project_dir = os.path.join(config.PROJECTS_DIR, name)
        suffix = request.args.get("suffix", "latest")

        s_part = "" if suffix == "latest" else f"_{suffix}"

        if file_type == "original":
            path = os.path.join(project_dir, f"combined_original{s_part}.md")
            filename = f"{name}_combined_original{s_part}.md"
        elif file_type == "polished":
            path = os.path.join(project_dir, f"combined_polished{s_part}.md")
            filename = f"{name}_combined_polished{s_part}.md"
        elif file_type == "analysis":
            path = os.path.join(project_dir, f"story_analysis{s_part}.md")
            filename = f"{name}_story_analysis{s_part}.md"
        else:
            return jsonify({"error": "Invalid file type"}), 400

        if not os.path.exists(path):
            return jsonify({"error": "File not found. Run Combine & Polish first."}), 404

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return Response(
            content,
            mimetype="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )

    @app.route("/api/project/<name>/combine/delete", methods=["POST"])
    def delete_combined(name):
        """Delete the combined story files (original, polished, analysis) including all old archived versions."""
        name = normalize_project_name(name)
        project_dir = os.path.join(config.PROJECTS_DIR, name)
        
        prefixes = [
            "combined_original",
            "combined_polished",
            "story_analysis"
        ]
        
        deleted = False
        try:
            if os.path.exists(project_dir):
                for fname in os.listdir(project_dir):
                    if any(fname.startswith(prefix) for prefix in prefixes) and fname.endswith(".md"):
                        path = os.path.join(project_dir, fname)
                        os.remove(path)
                        deleted = True
            
            return jsonify({"status": "ok", "deleted": deleted})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ─── API: Project Management ──────────────────────────────────
    @app.route("/api/project/<name>/delete", methods=["POST"])
    def delete_project(name):
        name = normalize_project_name(name)
        try:
            deleted = PipelineOrchestrator.delete_project(name)
            if deleted:
                return jsonify({"status": "ok"})
            return jsonify({"error": "Project not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/reset", methods=["POST"])
    def reset_project(name):
        """Reset generated content (chapters, plot, vectors) while keeping user data."""
        name = normalize_project_name(name)
        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            pipeline.load_project()
            pipeline.state_manager.reset_generated()
            return jsonify({"status": "ok", "message": "Generated content cleared"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/export", methods=["GET"])
    def export_project(name):
        name = normalize_project_name(name)
        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            pipeline.load_project()
            content = pipeline.export_full_story()
            return Response(
                content,
                mimetype="text/markdown",
                headers={
                    "Content-Disposition": f"attachment; filename={name}_full_story.md",
                },
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 404

    # ─── Phase 5: World Bible API ──────────────────────────────────────
    @app.route("/api/project/<name>/world-bible", methods=["POST"])
    def generate_world_bible(name):
        """Generate or regenerate the World Bible for a project."""
        name = normalize_project_name(name)
        data = request.json or {}
        use_llm = bool(data.get("use_llm", False))
        max_chapters = int(data.get("max_chapters", 999))
        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            pipeline.load_project()
            result = pipeline.generate_world_bible(
                use_llm=use_llm,
                max_chapters=max_chapters,
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/world-bible", methods=["GET"])
    def get_world_bible(name):
        """Read an existing World Bible for a project (JSON index)."""
        name = normalize_project_name(name)
        project_dir = os.path.join(config.PROJECTS_DIR, name)
        json_path = os.path.join(project_dir, "world_bible.json")
        md_path = os.path.join(project_dir, "world_bible.md")
        if not os.path.exists(json_path):
            return jsonify({"error": "World Bible not generated yet. POST to /world-bible first."}), 404
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            md_content = ""
            if os.path.exists(md_path):
                with open(md_path, encoding="utf-8") as f:
                    md_content = f.read()
            return jsonify({"world_bible": data, "markdown": md_content})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ─── Phase 5: Story Export API ─────────────────────────────────────
    @app.route("/api/project/<name>/export-story", methods=["POST"])
    def export_story(name):
        """Export the story to a distributable format (md, txt, epub, all)."""
        name = normalize_project_name(name)
        data = request.json or {}
        fmt = str(data.get("format", "md")).lower()
        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            pipeline.load_project()
            result = pipeline.export_story(fmt=fmt)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/export/download", methods=["GET"])
    def download_export(name):
        """
        Download a file produced by /export-story.
        Query: ?file=<basename> — restricted to .md/.txt/.epub inside the project dir.
        """
        name = normalize_project_name(name)
        file_name = request.args.get("file", "")
        # Only bare filenames, no traversal
        if not file_name or os.path.basename(file_name) != file_name:
            return jsonify({"error": "Invalid file name"}), 400
        if not file_name.lower().endswith((".md", ".txt", ".epub")):
            return jsonify({"error": "Invalid file type"}), 400

        project_dir = os.path.join(config.PROJECTS_DIR, name)
        path = os.path.join(project_dir, file_name)
        if not os.path.isfile(path):
            return jsonify({"error": "File not found. Run the export first."}), 404

        mimetypes = {
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".epub": "application/epub+zip",
        }
        ext = file_name.lower().rsplit(".", 1)[-1]
        with open(path, "rb") as f:
            content = f.read()
        return Response(
            content,
            mimetype=mimetypes[f".{ext}"],
            headers={"Content-Disposition": f"attachment; filename={file_name}"},
        )

    # ─── Phase 3: Scene Quality Score API ──────────────────────────────
    @app.route("/api/project/<name>/score-scene", methods=["POST"])
    def score_scene(name):
        """
        Score a scene text across coherence, pacing, voice, and temporal dimensions.
        Body: {"scene_text": "...", "scene": {...}, "previous_ending": "..."}
        """
        name = normalize_project_name(name)
        data = request.json or {}
        scene_text = str(data.get("scene_text", ""))
        scene = data.get("scene", {})
        previous_ending = str(data.get("previous_ending", ""))
        if not scene_text:
            return jsonify({"error": "scene_text is required"}), 400
        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            pipeline.load_project()
            result = pipeline.score_scene(
                scene_text=scene_text,
                scene=scene,
                previous_ending=previous_ending,
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ─── Phase C: Branch Options API ───────────────────────────────────
    @app.route("/api/project/<name>/branch-options", methods=["POST"])
    def branch_options(name):
        """
        Propose distinct directions for the NEXT chapter.
        Body: {"count": 3, "direction_hint": "..."} (both optional)
        """
        name = normalize_project_name(name)
        data = request.json or {}
        count = int(data.get("count", 3))
        direction_hint = str(data.get("direction_hint", "") or "")
        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            pipeline.load_project()
            result = pipeline.generate_branch_options(
                count=count,
                direction_hint=direction_hint,
            )
            code = 200 if result.get("status") == "ok" else 502
            return jsonify(result), code
        except Exception as e:
            logger.exception("Error in branch_options")
            return jsonify({"error": str(e)}), 500

    # ─── Phase C: Continuity Report API ────────────────────────────────
    @app.route("/api/project/<name>/continuity", methods=["GET"])
    def continuity_report(name):
        """
        Deterministic continuity overview: threads, seeds, characters,
        motifs, and warnings. No LLM involved.
        """
        name = normalize_project_name(name)
        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            pipeline.load_project()
            return jsonify(pipeline.continuity_report())
        except Exception as e:
            logger.exception("Error in continuity_report")
            return jsonify({"error": str(e)}), 500

    # ─── Phase C: Manual Scene Regeneration API ────────────────────────
    @app.route("/api/project/<name>/generate/manual/scene/<int:index>/regenerate", methods=["POST"])
    def regenerate_manual_scene_api(name, index):
        """
        Delete the last scene of the active manual session and regenerate it.
        Body: {"scene_brief": "..."} (optional — falls back to the original plan)
        Streams results via the manual_scene_done SSE event, like /manual/scene.
        """
        name = normalize_project_name(name)
        data = request.json or {}
        scene_brief = str(data.get("scene_brief", "") or "").strip()

        with _generation_lock:
            ms = _manual_sessions.get(name)
            if not ms:
                return jsonify({"error": "No active manual session. Call /manual/start first."}), 404
            if name in _active_pipelines:
                return jsonify({"error": "A scene is currently being generated. Wait for it to finish."}), 409

            pipeline = ms["pipeline"]
            session = ms["session"]
            eq = ms["eq"]

            try:
                recovered_brief = pipeline.prepare_manual_scene_regeneration(
                    session, index, scene_brief=scene_brief,
                )
            except (IndexError, ValueError) as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500

            _cancel_requests.discard(name)
            _active_pipelines[name] = pipeline

        def run_scene():
            try:
                result = pipeline.generate_manual_scene(
                    session=session,
                    scene_brief=recovered_brief,
                )
                _queue_event(eq, _normalize_event("manual_scene_done", result), force=True)
            except PipelineCancelledError:
                _queue_event(eq, _normalize_event("manual_scene_done", {
                    "status": "cancelled",
                    "scene_number": session.get("scene_counter", 0) + 1,
                    "scenes_completed": session.get("scene_counter", 0),
                    "completed_scenes": session.get("completed_scenes", []),
                }), force=True)
            except Exception as e:
                _queue_event(eq, _normalize_event("manual_scene_done", {
                    "status": "error",
                    "error": str(e),
                    "scene_number": session.get("scene_counter", 0) + 1,
                    "scenes_completed": session.get("scene_counter", 0),
                    "completed_scenes": session.get("completed_scenes", []),
                }), force=True)
            finally:
                with _generation_lock:
                    _active_pipelines.pop(name, None)

        thread = threading.Thread(target=run_scene, daemon=True)
        thread.start()

        scene_number = session["scene_counter"] + 1
        return jsonify({"status": "started", "scene_number": scene_number})

    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
