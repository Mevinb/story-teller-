"""
Centralized logging system for Story Teller.
Provides unified, thread-safe, rotating file logging and formatted console output
across all application systems: LLM interfaces, Key Rotators, Premise Architect,
Story Pipeline, Memory & Evolution, Vision Studio, and Flask/Gunicorn HTTP APIs.
"""
import os
import sys
import time
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

# Base logs directory in workspace
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

APP_LOG_PATH = os.path.join(LOGS_DIR, "app.log")
LLM_LOG_PATH = os.path.join(LOGS_DIR, "llm.log")
ERROR_LOG_PATH = os.path.join(LOGS_DIR, "error.log")

_INITIALIZED = False


class PlainFormatter(logging.Formatter):
    """Clean, consistent log formatter with millisecond timestamps and component names."""

    def format(self, record: logging.LogRecord) -> str:
        record.asctime = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        return f"[{record.asctime}] [{record.levelname:<7}] [{record.name}] {record.getMessage()}"


def init_logging(level: int = logging.INFO) -> None:
    """Initialize root logger and attach rotating file handlers for all project subsystems."""
    global _INITIALIZED
    if _INITIALIZED:
        return

    root = logging.getLogger()
    root.setLevel(level)

    formatter = PlainFormatter()

    # 1. Console Stream Handler (stdout)
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root.handlers)
    if not has_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    # 2. Main Application Log: logs/app.log (10 MB, up to 5 backups)
    app_handler = RotatingFileHandler(
        APP_LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    app_handler.setLevel(level)
    app_handler.setFormatter(formatter)
    root.addHandler(app_handler)

    # 3. Dedicated Error Log: logs/error.log (WARNING and ERROR only, with tracebacks)
    error_handler = RotatingFileHandler(
        ERROR_LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    root.addHandler(error_handler)

    # 4. Dedicated LLM Interaction Logger: logs/llm.log
    llm_logger = logging.getLogger("storyteller.llm")
    llm_logger.setLevel(logging.INFO)
    llm_logger.propagate = True  # also route to app.log

    llm_handler = RotatingFileHandler(
        LLM_LOG_PATH, maxBytes=15 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    llm_handler.setLevel(logging.INFO)
    llm_handler.setFormatter(formatter)
    llm_logger.addHandler(llm_handler)

    # Suppress overly chatty external libraries while preserving internal logs
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _INITIALIZED = True
    root.info("Story Teller unified logging system initialized. Log dir: %s", LOGS_DIR)


def get_logger(component_name: str) -> logging.Logger:
    """
    Get a standard named logger for any component or subsystem.
    Example: get_logger("premise"), get_logger("gemini"), get_logger("rotator")
    """
    if not _INITIALIZED:
        init_logging()

    if component_name.startswith("storyteller."):
        name = component_name
    elif component_name in ("root", "storyteller"):
        name = "storyteller"
    else:
        name = f"storyteller.{component_name}"

    return logging.getLogger(name)


def log_llm_call(
    provider: str,
    model: str,
    prompt: str,
    system: str = "",
    response: Optional[str] = None,
    error: Optional[Exception] = None,
    latency: float = 0.0,
    usage: Optional[Dict[str, Any]] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> None:
    """
    Structured logger for every LLM interaction across Gemini, Groq, OpenRouter, and Local GGUF.
    Writes rich diagnostic entries to logs/llm.log and logs/app.log.
    """
    llm_logger = logging.getLogger("storyteller.llm")
    if not _INITIALIZED:
        init_logging()

    prompt_preview = (prompt[:250] + "...") if len(prompt) > 250 else prompt
    prompt_len = len(prompt)

    if error:
        llm_logger.error(
            "[%s:%s] CALL FAILED after %.2fs | prompt_chars=%d | err=%s | prompt_preview=%r",
            provider.upper(), model, latency, prompt_len, str(error), prompt_preview,
            exc_info=True,
        )
    else:
        resp_str = response or ""
        resp_len = len(resp_str)
        resp_preview = (resp_str[:300] + "...") if resp_len > 300 else resp_str
        tok_info = ""
        if usage:
            p_tok = usage.get("prompt_tokens", 0)
            c_tok = usage.get("completion_tokens", 0)
            tok_info = f" | tokens=(p:{p_tok}, c:{c_tok})"

        llm_logger.info(
            "[%s:%s] CALL OK in %.2fs | prompt_chars=%d | resp_chars=%d%s | preview=%r",
            provider.upper(), model, latency, prompt_len, resp_len, tok_info, resp_preview.replace("\n", " "),
        )


def read_recent_logs(log_type: str = "app", max_lines: int = 200, filter_query: str = "") -> list[str]:
    """Read the tail of a specific log file for UI viewing or API inspection."""
    file_map = {
        "app": APP_LOG_PATH,
        "llm": LLM_LOG_PATH,
        "error": ERROR_LOG_PATH,
    }
    path = file_map.get(log_type.lower(), APP_LOG_PATH)
    if not os.path.isfile(path):
        return [f"Log file not created yet: {path}"]

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        if filter_query:
            q = filter_query.lower()
            lines = [line for line in lines if q in line.lower()]

        return [line.rstrip() for line in lines[-max_lines:]]
    except Exception as e:
        return [f"Failed to read log {path}: {e}"]
