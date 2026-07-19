"""
Agent Registry — Phase 2 Architectural Refactoring.

Decouples agent instantiation from PipelineOrchestrator.
Supports lazy initialization, compact-mode toggling, and easy swapping
of agent implementations without touching the orchestrator.

Usage:
    registry = AgentRegistry(model_provider)
    registry.build(backend="groq", cloud_model=groq_model, local_model=None)
    writer = registry.get("writer")
    registry.set_compact_mode(True)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ─── Agent Protocol ───────────────────────────────────────────────────────────

@runtime_checkable
class Agent(Protocol):
    """Minimal interface every agent must satisfy."""

    def run(self, inputs: dict) -> dict:
        ...


# ─── Registry ─────────────────────────────────────────────────────────────────

class AgentRegistry:
    """
    Central registry for all story-generation agents.

    Agents are registered lazily: call ``build()`` once, then ``get(name)``
    to retrieve individual agents.  All agents can be put into compact mode
    via a single ``set_compact_mode()`` call.

    Registered agent names (matches orchestrator usage):
        architect, planner, writer, critic, editor, pacing, voice
    """

    # Default factories use keyword arguments that match agent __init__ sigs.
    _COMPACT_CAPABLE = frozenset({"architect", "planner", "writer", "critic", "editor"})

    def __init__(self) -> None:
        self._agents: Dict[str, Any] = {}
        self._compact_mode: bool = False
        self._built: bool = False

    # ─── Build ────────────────────────────────────────────────────────────────

    def build(
        self,
        backend: str,
        *,
        cloud_model=None,
        local_model=None,
        state_manager=None,
        cloud_available: bool = False,
    ) -> None:
        """Instantiate all agents for the given backend configuration."""
        from agents.architect import StoryArchitect
        from agents.planner import ScenePlanner
        from agents.writer import SceneWriter
        from agents.consistency import ConsistencyEngine
        from agents.editor import Editor
        from agents.pacing import PacingAgent
        from agents.voice import VoiceAgent

        is_cloud_backend = backend in {"groq", "gemini", "openrouter"}

        if is_cloud_backend:
            if cloud_model is None:
                raise ValueError(f"{backend} backend requires a cloud_model")
            self._agents["architect"] = StoryArchitect(cloud_model, state_manager)
            self._agents["planner"] = ScenePlanner(cloud_model)
            self._agents["writer"] = SceneWriter(cloud_model, cloud_model)
            self._agents["critic"] = ConsistencyEngine(cloud_model)
            self._agents["editor"] = Editor(cloud_model)
        else:
            # Local or hybrid
            primary_model = local_model
            if primary_model is None:
                raise ValueError("local backend requires a local_model")
            self._agents["architect"] = StoryArchitect(primary_model, state_manager)
            self._agents["planner"] = ScenePlanner(primary_model)
            writer_model = cloud_model if cloud_available else primary_model
            self._agents["writer"] = SceneWriter(writer_model, primary_model)
            self._agents["critic"] = ConsistencyEngine(primary_model)
            self._agents["editor"] = Editor(primary_model)

        self._agents["pacing"] = PacingAgent()
        self._agents["voice"] = VoiceAgent()

        self._built = True
        logger.info(
            "AgentRegistry built for backend=%s (cloud=%s, compact=%s)",
            backend,
            is_cloud_backend,
            self._compact_mode,
        )

        # Apply compact mode if it was set before build()
        if self._compact_mode:
            self._apply_compact_mode()

    def rebuild_for_genre(self, genre: str) -> None:
        """Propagate genre to agents that care (writer, editor)."""
        for name in ("writer", "editor"):
            agent = self._agents.get(name)
            if agent and hasattr(agent, "set_genre"):
                agent.set_genre(genre)

    # ─── Access ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> Any:
        """Return agent by name; raises KeyError if not registered."""
        if not self._built and name not in self._agents:
            raise RuntimeError("AgentRegistry.build() must be called before get()")
        if name not in self._agents:
            raise KeyError(f"Unknown agent: {name!r}. Available: {list(self._agents)}")
        return self._agents[name]

    def register(self, name: str, agent: Any) -> None:
        """Manually register or replace an agent (useful for testing with mocks)."""
        self._agents[name] = agent
        logger.debug("AgentRegistry: registered custom agent %r", name)

    def all_agents(self) -> Dict[str, Any]:
        """Return a copy of the full agent dict."""
        return dict(self._agents)

    # ─── Compact Mode ─────────────────────────────────────────────────────────

    def set_compact_mode(self, enabled: bool) -> None:
        """Toggle compact prompts on all capable agents."""
        self._compact_mode = enabled
        if self._built:
            self._apply_compact_mode()

    def _apply_compact_mode(self) -> None:
        touched = []
        for name in self._COMPACT_CAPABLE:
            agent = self._agents.get(name)
            if agent and hasattr(agent, "set_compact_mode"):
                agent.set_compact_mode(self._compact_mode)
                touched.append(name)
        logger.debug(
            "AgentRegistry: compact_mode=%s applied to %s",
            self._compact_mode,
            touched,
        )

    # ─── Properties (convenience aliases) ─────────────────────────────────────

    @property
    def architect(self):
        return self.get("architect")

    @property
    def planner(self):
        return self.get("planner")

    @property
    def writer(self):
        return self.get("writer")

    @property
    def critic(self):
        return self.get("critic")

    @property
    def editor(self):
        return self.get("editor")

    @property
    def pacing(self):
        return self.get("pacing")

    @property
    def voice(self):
        return self.get("voice")

    # ─── Diagnostics ──────────────────────────────────────────────────────────

    def describe(self) -> Dict[str, str]:
        """Return {agent_name: class_name} for logging/debugging."""
        return {
            name: type(agent).__name__
            for name, agent in self._agents.items()
        }

    def __repr__(self) -> str:
        status = "built" if self._built else "empty"
        return f"<AgentRegistry {status} agents={list(self._agents)}>"
