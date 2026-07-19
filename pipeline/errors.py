"""
Pipeline Error Types — Structured exception hierarchy.

Replaces bare RuntimeError throughout the pipeline so callers can
catch specific error categories without string-parsing error messages.

Hierarchy:
    PipelineError (base)
    ├── ModelError
    │   ├── RateLimitError       — API quota / 429
    │   ├── ModelUnavailableError — model offline or missing
    │   └── TokenBudgetExceeded  — context/token limit hit
    ├── ValidationError
    │   ├── SceneIncompleteError  — scene below word threshold
    │   ├── SceneTruncatedError   — scene ends mid-sentence
    │   └── PremiseViolationError — future premise step leaked
    ├── StateError
    │   ├── WIPCorruptError      — WIP checkpoint unreadable
    │   └── StateSchemaMismatch  — state.json schema drift
    └── PipelineCancelledError   — user-initiated cancellation
"""

from __future__ import annotations


class PipelineError(RuntimeError):
    """Base class for all Story Teller pipeline errors."""

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.context: dict = context or {}

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} [{ctx_str}]"
        return base


# ─── Model Errors ─────────────────────────────────────────────────────────────

class ModelError(PipelineError):
    """An LLM backend returned an error or is unavailable."""


class RateLimitError(ModelError):
    """API rate limit / quota exceeded (HTTP 429)."""

    def __init__(self, message: str = "Rate limit exceeded", *, backend: str = "", retry_after: float | None = None):
        super().__init__(message, context={"backend": backend, "retry_after": retry_after})
        self.backend = backend
        self.retry_after = retry_after


class ModelUnavailableError(ModelError):
    """Model is offline, missing, or its API key is invalid."""

    def __init__(self, message: str = "Model unavailable", *, model_id: str = ""):
        super().__init__(message, context={"model_id": model_id})
        self.model_id = model_id


class TokenBudgetExceeded(ModelError):
    """Generation paused because the session token budget was exhausted."""

    def __init__(
        self,
        message: str = "Token budget exceeded",
        *,
        chapter_num: int = 0,
        scene_num: int = 0,
        tokens_used: int = 0,
        budget: int = 0,
    ):
        super().__init__(
            message,
            context={
                "chapter": chapter_num,
                "scene": scene_num,
                "tokens_used": tokens_used,
                "budget": budget,
            },
        )
        self.chapter_num = chapter_num
        self.scene_num = scene_num
        self.tokens_used = tokens_used
        self.budget = budget


# ─── Validation Errors ────────────────────────────────────────────────────────

class ValidationError(PipelineError):
    """A generated artifact failed a quality or completeness check."""


class SceneIncompleteError(ValidationError):
    """Scene word count is below the minimum threshold."""

    def __init__(
        self,
        message: str = "Scene is incomplete",
        *,
        scene_num: int | str = "?",
        chapter_num: int = 0,
        words: int = 0,
        minimum: int = 0,
    ):
        super().__init__(
            message,
            context={
                "scene": scene_num,
                "chapter": chapter_num,
                "words": words,
                "minimum": minimum,
            },
        )
        self.scene_num = scene_num
        self.chapter_num = chapter_num
        self.words = words
        self.minimum = minimum


class SceneTruncatedError(ValidationError):
    """Scene text ends mid-sentence (generation was cut off)."""

    def __init__(
        self,
        message: str = "Scene appears truncated",
        *,
        scene_num: int | str = "?",
        chapter_num: int = 0,
        words: int = 0,
    ):
        super().__init__(
            message,
            context={"scene": scene_num, "chapter": chapter_num, "words": words},
        )
        self.scene_num = scene_num
        self.chapter_num = chapter_num
        self.words = words


class PremiseViolationError(ValidationError):
    """A chapter plan or scene references future premise steps."""

    def __init__(
        self,
        message: str = "Premise order violated",
        *,
        violations: list[str] | None = None,
        chapter_num: int = 0,
    ):
        super().__init__(
            message,
            context={"chapter": chapter_num, "violations": violations or []},
        )
        self.violations = violations or []
        self.chapter_num = chapter_num


# ─── State Errors ─────────────────────────────────────────────────────────────

class StateError(PipelineError):
    """Persistent state is corrupt or incompatible with the current schema."""


class WIPCorruptError(StateError):
    """WIP checkpoint file is missing, partially written, or contains corrupt JSON."""

    def __init__(
        self,
        message: str = "WIP checkpoint is corrupt",
        *,
        chapter_num: int = 0,
        path: str = "",
    ):
        super().__init__(message, context={"chapter": chapter_num, "path": path})
        self.chapter_num = chapter_num
        self.path = path


class StateSchemaMismatch(StateError):
    """state.json is missing expected fields or has incompatible types."""

    def __init__(self, message: str = "State schema mismatch", *, missing_fields: list[str] | None = None):
        super().__init__(message, context={"missing_fields": missing_fields or []})
        self.missing_fields = missing_fields or []


# ─── Cancellation ─────────────────────────────────────────────────────────────

class PipelineCancelledError(PipelineError):
    """User requested cancellation of the active generation."""

    def __init__(self, message: str = "Generation cancelled by user"):
        super().__init__(message)
