"""
Pipeline Orchestrator — Main controller for the story generation pipeline.
Manages the full chapter generation flow from planning to storage.
"""
import os
import json
import logging
import shutil
import time
import re
from datetime import datetime
from typing import Optional, Callable

import config
from models.llm import LlamaCPP, to_model_id
from models.groq_model import GroqModel
from models.gemini_model import GeminiModel
from models.openrouter_model import OpenRouterModel
from memory.state_manager import StateManager
from memory.vector_store import VectorStore
from memory.retriever import Retriever
from memory.tension_tracker import TensionTracker
from memory.motif_tracker import MotifTracker
from agents.architect import StoryArchitect
from agents.planner import ScenePlanner
from agents.writer import SceneWriter
from agents.consistency import ConsistencyEngine
from agents.editor import Editor
from agents.pacing import PacingAgent
from agents.voice import VoiceAgent
from memory.evolution_engine import evolve_after_scene
from pipeline.quality_controller import QualityController

logger = logging.getLogger(__name__)

_REASONING_BLOCK_RE = re.compile(
    r"<(?:think|analysis|reasoning)>.*?</(?:think|analysis|reasoning)>",
    flags=re.IGNORECASE | re.DOTALL,
)
_REASONING_TAG_RE = re.compile(r"</?(?:think|analysis|reasoning)>", flags=re.IGNORECASE)
_PROJECT_SLUG_RE = re.compile(r"[^a-z0-9_-]+")
_CHAPTER_FILE_RE = re.compile(r"^chapter_(\d{3,})\.md$")
_META_OUTPUT_RE = re.compile(
    r"(?im)^\s*(?:thinking process|analysis|step[- ]by[- ]step|analyze the request|"
    r"professional fiction editor|task:|goals:|constraints:|output:|"
    r"\d+\.\s+\*\*analyze|\*\s+\*\*role:)"
)
_GENERIC_CHARACTER_LABELS = {
    "class", "classmates", "students", "group", "crowd", "passengers",
    "people", "onlookers", "everyone", "men", "women",
}
_FUTURE_MARKER_TOKEN_STOPWORDS = {
    # Conjunctions / prepositions
    "with", "while", "from", "that", "this", "then", "their", "also", "when",
    "after", "into", "over", "upon", "about", "through", "during",
    # Common narrative verbs — appear in virtually every premise step
    "becomes", "starts", "continues", "begins", "tries", "tries", "finds",
    "feels", "knows", "goes", "gets", "lets", "puts", "uses", "sees",
    "come", "comes", "make", "makes", "take", "takes", "give", "gives",
    "thinking", "feeling", "noticing", "realizing", "realises", "realizes",
    "discovering", "discovering", "just", "only", "more", "suddenly",
    "scene", "scenes", "chapter", "story", "event",
    # Legacy project-specific names kept for backwards compat
    "sherin", "jomy", "riya",
}

PIPELINE_GRAPH = {
    "start": ["planner_context"],
    "planner_context": ["planner"],
    "planner": ["scene_planner"],
    "scene_planner": ["scene_loop"],
    "scene_loop": ["persist"],
    "persist": ["end"],
}

SCENE_GRAPH = {
    "start": ["writer"],
    "writer": ["critic"],
    "critic": ["decision"],
    "decision": ["writer", "editor"],
    "editor": ["end"],
}

MANUAL_SCENE_ENHANCER_SYSTEM = (
    "You are a prompt enhancer for fiction scene plans. "
    "Expand a user's brief into concrete, production-ready scene instructions. "
    "CRITICAL RULE: You must ONLY describe events the user explicitly mentioned in their brief. "
    "Do NOT invent new story events, new scenes, or story progression beyond what the user wrote. "
    "Your job is to add HOW-TO-WRITE detail (mood, sensory cues, tone), NOT to advance the story further. "
    "The scene ends exactly where the user's brief ends — no continuation, no aftermath, no next steps. "
    "ANTI-REPETITION: Check the 'Previously completed scenes' section carefully. "
    "Do NOT produce a summary or key_events that overlap with events already written. "
    "Your scene must start where the previous scene ended and cover NEW ground only. "
    "Return JSON only."
)

MANUAL_SCENE_ENHANCER_PROMPT = """Convert this user scene brief into a detailed scene plan.

Chapter title: {chapter_title}
Scene number: {scene_number}
User brief: {scene_brief}
Pacing: {pacing}
Known character names: {character_names}
Previously completed scenes in this chapter:
{previous_scenes_summary}
Story context:
{context}

Requirements:
- SCOPE LOCK: Only include events and beats that the user EXPLICITLY described in their brief.
  Do NOT add new events, future scenes, or story beats not present in the brief.
  The scene ends exactly where the brief ends — do not extend it further.
- PRESERVE the user's original theme and intent. Do not overdrift or change what happens.
- Preserve every distinct user-provided beat in order; do not merge away introductions, arrivals, departures, or first meetings.
- Keep the user's wording in the summary where possible — refine, don't replace.
- Add only HOW-TO-WRITE detail: mood cues, sensory notes, character actions, dialogue hints.
- Use exact known character names when applicable.
- ANTI-REPETITION RULE: Read the 'Previously completed scenes' section VERY carefully.
  If a scene ending is listed, the new scene MUST start from that exact moment.
  Do NOT produce key_events that describe events already covered in previous scenes.
  Do NOT re-introduce characters, locations, or situations that were already established.
- If previous scenes are listed, ensure strict continuity: this scene picks up exactly
  where the last scene's ending left off. No recap, no re-introduction.
- Keep the user's core action on-page in this scene. Do NOT convert the brief into aftermath-only framing.
- Do not add meta text or commentary.
- key_events must map 1:1 to user-provided beats. Do not invent extra key events.
"""


def normalize_project_name(name: str) -> str:
    """Convert user-facing titles/names into safe project directory names."""
    slug = (name or "").strip().lower().replace(" ", "_").replace("'", "")
    slug = _PROJECT_SLUG_RE.sub("_", slug)
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug[:80] or "untitled"


def _safe_project_dir(project_name: str) -> tuple[str, str]:
    normalized = normalize_project_name(project_name)
    base = os.path.abspath(config.PROJECTS_DIR)
    project_dir = os.path.abspath(os.path.join(base, normalized))
    if os.path.commonpath([base, project_dir]) != base:
        raise ValueError(f"Unsafe project name: {project_name!r}")
    return normalized, project_dir


class PipelineOrchestrator:
    """
    Orchestrates the full story generation pipeline.

    Flow per chapter (explicit graph):
    start -> planner_context -> planner -> scene_planner -> scene_loop -> persist -> end

    Scene loop graph:
    start -> writer -> critic -> decision -> (writer | editor) -> end
    """

    def __init__(
        self,
        project_name: str,
        progress_callback: Callable = None,
        local_model: Optional[str] = None,
        backend: str = "local",
        groq_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
        openrouter_model: Optional[str] = None,
    ):
        self.project_name, self.project_dir = _safe_project_dir(project_name)
        self.chapters_dir = os.path.join(self.project_dir, "chapters")
        self.logs_dir = os.path.join(self.project_dir, "logs")
        self._progress_cb = progress_callback or (lambda *a, **kw: None)
        self._cancelled = False
        self._scene_cancelled = False
        self._backend = (backend or "local").strip().lower()
        if self._backend not in {"local", "groq", "gemini", "openrouter"}:
            raise ValueError(f"Unsupported backend: {backend}")
        selected_local = (local_model or config.LLAMA_MODEL_PATH).strip()
        self._local_model_ref = selected_local
        self._local_model_id = to_model_id(selected_local)
        self._groq_model_id = (groq_model or config.GROQ_MODEL).strip()
        self._gemini_model_id = (gemini_model or config.GEMINI_MODEL).strip()
        self._openrouter_model_id = (openrouter_model or config.OPENROUTER_MODEL).strip()
        self._active_backend = "local"
        self._active_model_id = self._local_model_id

        # Create directories
        for d in [self.project_dir, self.chapters_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)

        # Initialize components
        self._init_models()
        self._init_memory()
        self._init_agents()

    def _init_models(self):
        self._emit("status", "Initializing models...")

        self.cloud_model = None
        self._cloud_available = False
        self._local_model = None  # Lazy-loaded only when needed
        self._active_backend = "local"
        self._active_model_id = self._local_model_id

        if self._backend == "groq":
            if not config.GROQ_API_KEY:
                raise RuntimeError(
                    "GROQ_API_KEY not set. Add it to your .env file before using Groq mode."
                )
            self.cloud_model = GroqModel(model=self._groq_model_id)
            self._log("Checking Groq API connectivity...", details={"model": self._groq_model_id})
            if not self._has_internet():
                raise RuntimeError("Internet unavailable. Groq mode requires network access.")
            if not self.cloud_model.is_available():
                raise RuntimeError(
                    f"Groq model '{self._groq_model_id}' is unavailable. Verify API key and model name."
                )
            self._cloud_available = True
            self._active_backend = "groq"
            self._active_model_id = self._groq_model_id
            self._emit("status", f"Groq model ready: {self.cloud_model.get_name()}")
            self._log(f"Groq model ready: {self.cloud_model.get_name()}", level="success")
            return

        if self._backend == "gemini":
            if not config.GEMINI_API_KEY:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. Add it to your .env file before using Gemini mode."
                )
            self.cloud_model = GeminiModel(model=self._gemini_model_id)
            self._log("Checking Gemini API connectivity...", details={"model": self._gemini_model_id})
            if not self._has_internet():
                raise RuntimeError("Internet unavailable. Gemini mode requires network access.")
            if not self.cloud_model.is_available():
                raise RuntimeError(
                    f"Gemini model '{self._gemini_model_id}' is unavailable. Verify API key and model name."
                )
            self._cloud_available = True
            self._active_backend = "gemini"
            self._active_model_id = self._gemini_model_id
            self._emit("status", f"Gemini model ready: {self.cloud_model.get_name()}")
            self._log(f"Gemini model ready: {self.cloud_model.get_name()}", level="success")
            return

        if self._backend == "openrouter":
            if not config.OPENROUTER_API_KEY:
                raise RuntimeError(
                    "OPENROUTER_API_KEY not set. Add it to your .env file before using OpenRouter mode."
                )
            self.cloud_model = OpenRouterModel(model=self._openrouter_model_id)
            self._log("Checking OpenRouter API connectivity...", details={"model": self._openrouter_model_id})
            if not self._has_internet():
                raise RuntimeError("Internet unavailable. OpenRouter mode requires network access.")
            if not self.cloud_model.is_available():
                raise RuntimeError(
                    f"OpenRouter model '{self._openrouter_model_id}' is unavailable. Verify API key and model name."
                )
            self._cloud_available = True
            self._active_backend = "openrouter"
            self._active_model_id = self._openrouter_model_id
            self._emit("status", f"OpenRouter model ready: {self.cloud_model.get_name()}")
            self._log(f"OpenRouter model ready: {self.cloud_model.get_name()}", level="success")
            return

        # Local-only by default. Cloud can be enabled explicitly with USE_CLOUD_MODEL=true.
        if config.USE_CLOUD_MODEL and config.GROQ_API_KEY:
            self.cloud_model = GroqModel()
            self._log("Checking internet connectivity...", details={"model": config.GROQ_MODEL})
            if self._has_internet():
                try:
                    self._cloud_available = self.cloud_model.is_available()
                except Exception:
                    pass

        if self._cloud_available:
            self._emit("status", f"Cloud model ready: {self.cloud_model.get_name()}")
            self._log(f"Cloud model ready: {self.cloud_model.get_name()}", level="success")
        else:
            if config.USE_CLOUD_MODEL:
                self._emit("status", "Cloud unavailable — using local model")
                self._log("Cloud model unavailable — local model will load on first generation", level="warn")
            else:
                self._emit("status", "Local-only mode enabled")
                self._log("Local-only mode enabled — skipping cloud model")
            self._ensure_local_model()

    @property
    def local_model(self):
        """Lazy-load local model only when actually needed (saves GPU/heat)."""
        if self._local_model is None:
            self._ensure_local_model()
        return self._local_model

    def _ensure_local_model(self):
        """Initialize the local GGUF model on demand."""
        if self._local_model is not None:
            return
        self._local_model = LlamaCPP(model_path=self._local_model_ref)
        if not os.path.isfile(self._local_model.model_path):
            raise RuntimeError(
                f"Local GGUF model '{self._local_model_id}' is not available. "
                "Set LLAMA_MODEL_PATH to a valid .gguf file."
            )
        self._log("Local model configured; weights will load on first inference",
                  details={"model": self._local_model_id})

    @staticmethod
    def _has_internet():
        """Quick check if we can reach the internet."""
        import socket
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            return True
        except OSError:
            return False

    def _init_memory(self):
        self.state_manager = StateManager(self.project_dir)
        self.vector_store = VectorStore(self.project_dir)
        self.retriever = Retriever(self.state_manager, self.vector_store)

    def _init_agents(self):
        if self._backend in {"groq", "gemini", "openrouter"}:
            if not self.cloud_model:
                raise RuntimeError(f"{self._backend.title()} backend selected but cloud model is not initialized.")
            self.architect = StoryArchitect(self.cloud_model, self.state_manager)
            self.planner = ScenePlanner(self.cloud_model)
            self.writer = SceneWriter(self.cloud_model, self.cloud_model)
            self.consistency = ConsistencyEngine(self.cloud_model)
            self.editor = Editor(self.cloud_model)
            self.pacing = PacingAgent()
            self.voice = VoiceAgent()
            self._enable_compact_agent_prompts()
            # Phase 2/3: Quality controller (stateless, always available)
            self.quality_ctrl = QualityController()
            return

        # All local by default; cloud can optionally be writer-primary.
        self.architect = StoryArchitect(self.local_model, self.state_manager)
        self.planner = ScenePlanner(self.local_model)
        primary_writer = self.cloud_model if self._cloud_available else self.local_model
        self.writer = SceneWriter(primary_writer, self.local_model)
        self.consistency = ConsistencyEngine(self.local_model)
        self.editor = Editor(self.local_model)
        self.pacing = PacingAgent()
        self.voice = VoiceAgent()
        self._enable_compact_local_planning_prompts()
        # Phase 2/3: Quality controller (stateless, always available)
        self.quality_ctrl = QualityController()

    def _enable_compact_agent_prompts(self):
        """Use shorter prompts/context for Groq so daily token quotas last longer."""
        for agent in (
            self.architect,
            self.planner,
            self.writer,
            self.consistency,
            self.editor,
        ):
            if hasattr(agent, "set_compact_mode"):
                agent.set_compact_mode(True)
        self._log("Groq compact prompt mode enabled", details={"backend": self._backend})

    def _enable_compact_local_planning_prompts(self):
        """Keep local planning/review calls responsive while preserving full scene prose prompts."""
        for agent in (
            self.architect,
            self.planner,
            self.consistency,
            self.editor,
        ):
            if hasattr(agent, "set_compact_mode"):
                agent.set_compact_mode(True)
        self._log("Local compact planning mode enabled", details={"backend": self._backend})

    def _emit(self, event: str, data=None, **kwargs):
        self._progress_cb(event=event, data=data, **kwargs)

    def _log(self, message: str, level: str = "info", details: dict = None):
        """Emit a detailed log entry for the live logs view."""
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "level": level,
            "message": message,
        }
        if details:
            entry["details"] = details
        self._emit("log", entry)

    @staticmethod
    def _truncate(value, limit: int = 220) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _record_trace(
        self,
        trace_entries: list,
        chapter_num: int,
        scene_num: int,
        step_idx: int,
        agent: str,
        agent_input: dict,
        agent_output: dict,
        latency_ms: float,
        next_action: str,
    ) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "chapter": chapter_num,
            "scene": scene_num,
            "step": step_idx,
            "agent": agent,
            "input": agent_input,
            "output": agent_output,
            "latency_ms": round(latency_ms, 2),
            "next_action": next_action,
        }
        trace_entries.append(entry)
        self._emit("trace", entry)

    def _save_trace(self, chapter_num: int, trace_entries: list) -> None:
        trace_path = os.path.join(
            self.logs_dir, f"chapter_{chapter_num:03d}_trace.jsonl"
        )
        with open(trace_path, "w", encoding="utf-8") as f:
            for entry in trace_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _run_agent_step(
        self,
        chapter_num: int,
        scene_num: int,
        trace_entries: list,
        step_counter: dict,
        agent_name: str,
        call_fn,
        agent_input: dict,
    ) -> dict:
        if step_counter["steps"] >= config.MAX_PIPELINE_STEPS:
            raise RuntimeError(
                f"Pipeline step budget exceeded ({config.MAX_PIPELINE_STEPS})"
            )
        start = time.time()
        try:
            result = call_fn()
        except Exception as e:
            error_payload = {"error": str(e), "agent": agent_name}
            self._emit("error", error_payload)
            raise RuntimeError(json.dumps(error_payload))
        latency_ms = (time.time() - start) * 1000
        step_counter["steps"] += 1
        raw_output = result.get("output")
        if isinstance(raw_output, dict):
            output_summary = {k: self._truncate(v, 140) for k, v in raw_output.items()}
        else:
            output_summary = self._truncate(raw_output, 140)
        self._record_trace(
            trace_entries=trace_entries,
            chapter_num=chapter_num,
            scene_num=scene_num,
            step_idx=step_counter["steps"],
            agent=agent_name,
            agent_input=agent_input,
            agent_output={
                "output": output_summary,
                "confidence": result.get("confidence"),
                "next_action": result.get("next_action"),
            },
            latency_ms=latency_ms,
            next_action=result.get("next_action", ""),
        )
        return result

    def _run_scene_graph(
        self,
        chapter_num: int,
        scene: dict,
        scene_context: str,
        previous_ending: str,
        trace_entries: list,
        step_counter: dict,
    ) -> dict:
        scene_num = scene["scene_number"]
        graph_node = "start"
        iteration = 0
        state = {
            "scene_text": "",
            "report": {"is_consistent": True, "issues": [], "state_updates": {}},
            "consistency_context": "",
            "blocking_issues": [],
            "non_blocking_issues": [],
            "last_blocking_fingerprint": "",
            "blocking_repeat_count": 0,
        }

        while graph_node != "end":
            if self._cancelled or self._scene_cancelled:
                return {"status": "cancelled"}

            if graph_node == "start":
                graph_node = SCENE_GRAPH["start"][0]
                continue

            if graph_node == "writer":
                self._emit("agent_active", {
                    "agent": "Scene Writer",
                    "step": f"scene {scene_num} iteration {iteration + 1}",
                })
                self._log(
                    f"Scene Writer: scene {scene_num}, iteration {iteration + 1}",
                    details={"graph_node": graph_node},
                )
                iteration += 1
                def stream_scene_token(chunk):
                    if self._cancelled or self._scene_cancelled:
                        return False
                    self._emit(
                        "token",
                        {
                            "scene": scene_num,
                            "content": chunk,
                            "provider": self.writer.last_provider,
                        },
                    )
                    return True

                mode = "write"
                if state.get("blocking_issues"):
                    if iteration >= config.MAX_SCENE_ITERATIONS:
                        mode = "final_patch"
                    elif state.get("blocking_repeat_count", 0) >= 2:
                        mode = "patch"
                    else:
                        mode = "rewrite"

                writer_input = {
                    "mode": mode,
                    "iteration": iteration,
                    "scene_plan": scene,
                    "chapter_num": chapter_num,
                    "context": scene_context,
                    "previous_ending": previous_ending,
                    "original_text": state["scene_text"],
                    "issues": state.get("blocking_issues", []),
                    "state_context": state.get("consistency_context", ""),
                    "stream_callback": stream_scene_token,
                }
                writer_result = self._run_agent_step(
                    chapter_num=chapter_num,
                    scene_num=scene_num,
                    trace_entries=trace_entries,
                    step_counter=step_counter,
                    agent_name="writer",
                    call_fn=lambda: self.writer.run(writer_input),
                    agent_input={
                        "mode": mode,
                        "summary": self._truncate(scene.get("summary", "")),
                        "issues": len(state.get("blocking_issues", [])),
                    },
                )
                if self._cancelled or self._scene_cancelled:
                    return {"status": "cancelled"}
                state["scene_text"] = writer_result["output"]["scene_text"]
                words = len(state["scene_text"].split())
                self._emit("scene_written", {
                    "scene": scene_num,
                    "words": words,
                    "provider": writer_result["output"].get("provider", self.writer.last_provider),
                })
                graph_node = SCENE_GRAPH["writer"][0]
                continue

            if graph_node == "critic":
                self._emit("agent_active", {
                    "agent": "Consistency Engine",
                    "step": f"scene {scene_num}",
                })
                consistency_ctx = self.retriever.retrieve_for_consistency(
                    scene_text=state["scene_text"],
                    characters=scene.get("characters_present"),
                )
                critic_input = {
                    "scene_text": state["scene_text"],
                    "state_context": consistency_ctx,
                    "scene_plan": scene,
                }
                critic_result = self._run_agent_step(
                    chapter_num=chapter_num,
                    scene_num=scene_num,
                    trace_entries=trace_entries,
                    step_counter=step_counter,
                    agent_name="critic",
                    call_fn=lambda: self.consistency.run(critic_input),
                    agent_input={"scene_words": len(state["scene_text"].split())},
                )
                if self._cancelled or self._scene_cancelled:
                    return {"status": "cancelled"}
                state["consistency_context"] = consistency_ctx
                state["report"] = critic_result["output"]["report"]
                state["blocking_issues"] = self.consistency.get_blocking_issues(state["report"])
                state["non_blocking_issues"] = self.consistency.get_non_blocking_issues(state["report"])
                blocking_count = len(state["blocking_issues"])
                transition_count = len(state["non_blocking_issues"])
                if blocking_count == 0 and transition_count == 0:
                    self._log("Consistency check passed (0 issues)", level="success")
                elif blocking_count == 0:
                    self._log(
                        f"Consistency check: {transition_count} non-blocking state updates detected",
                        level="info",
                        details={"updates": [i.get("detail", "") for i in state["non_blocking_issues"][:3]]},
                    )
                else:
                    self._log(
                        f"Consistency check: {blocking_count} blocking, {transition_count} non-blocking issues",
                        level="warn",
                        details={"issues": [i.get("detail", "") for i in state["blocking_issues"][:3]]},
                    )
                graph_node = SCENE_GRAPH["critic"][0]
                continue

            if graph_node == "decision":
                blocking = bool(state.get("blocking_issues"))
                reached_max_iter = iteration >= config.MAX_SCENE_ITERATIONS
                fingerprint = "|".join(
                    sorted(
                        self.consistency.issue_signature(issue)
                        for issue in state.get("blocking_issues", [])
                    )
                )
                if blocking:
                    if fingerprint and fingerprint == state.get("last_blocking_fingerprint", ""):
                        state["blocking_repeat_count"] += 1
                    else:
                        state["blocking_repeat_count"] = 0
                    state["last_blocking_fingerprint"] = fingerprint
                else:
                    state["blocking_repeat_count"] = 0
                    state["last_blocking_fingerprint"] = ""

                stuck_on_same_issues = blocking and state["blocking_repeat_count"] >= 2

                # ── Oscillation breaker ──────────────────────────────
                # If the exact same set of blocking issues has appeared
                # 3+ times, the writer cannot fix them. Downgrade to
                # non-blocking so the pipeline can proceed.
                if blocking and state["blocking_repeat_count"] >= 3:
                    self._log(
                        f"Scene {scene_num}: oscillation detected — same issues "
                        f"repeated {state['blocking_repeat_count']} times. "
                        "Downgrading remaining blockers to non-blocking.",
                        level="warn",
                        details={
                            "issues": [
                                i.get("detail", "")
                                for i in state.get("blocking_issues", [])[:3]
                            ]
                        },
                    )
                    # Move all remaining blocking issues into non-blocking
                    for issue in state.get("blocking_issues", []):
                        issue["blocking"] = False
                        issue["severity"] = "low"
                        issue["detail"] = (
                            f"[auto-downgraded after {state['blocking_repeat_count']} "
                            f"rewrite attempts] {issue.get('detail', '')}"
                        )
                    state["non_blocking_issues"].extend(state["blocking_issues"])
                    state["blocking_issues"] = []
                    blocking = False

                if step_counter["steps"] >= config.MAX_PIPELINE_STEPS:
                    raise RuntimeError(
                        f"Pipeline step budget exceeded ({config.MAX_PIPELINE_STEPS})"
                    )
                step_counter["steps"] += 1
                
                # Send to editor if no issues, or if we already did the final patch
                if not blocking or (blocking and iteration > config.MAX_SCENE_ITERATIONS):
                    next_action = "editor"
                else:
                    next_action = "writer"

                self._record_trace(
                    trace_entries=trace_entries,
                    chapter_num=chapter_num,
                    scene_num=scene_num,
                    step_idx=step_counter["steps"],
                    agent="decision",
                    agent_input={
                        "blocking": blocking,
                        "iteration": iteration,
                        "blocking_repeat_count": state["blocking_repeat_count"],
                    },
                    agent_output={
                        "blocking_issues": len(state.get("blocking_issues", [])),
                        "non_blocking_issues": len(state.get("non_blocking_issues", [])),
                        "max_iteration_reached": reached_max_iter,
                        "stuck_on_same_issues": stuck_on_same_issues,
                        "next_action": next_action
                    },
                    latency_ms=0.0,
                    next_action=next_action,
                )
                
                if next_action == "writer":
                    if reached_max_iter:
                        self._emit("status", f"Scene {scene_num} reached max iterations; running final targeted patch...")
                        self._log(f"Scene {scene_num} reached max iterations; running final targeted patch", level="warn")
                    elif stuck_on_same_issues:
                        self._emit("status", f"Scene {scene_num} stuck on same issues; running targeted patch...")
                        self._log(f"Scene {scene_num} hit repeated blocker loop; running targeted patch", level="warn")
                    else:
                        self._emit("status", f"Consistency issues in scene {scene_num}. Rewriting (attempt {iteration})...")
                    graph_node = SCENE_GRAPH["decision"][0]
                else:
                    state_updates = state["report"].get("state_updates", {})
                    if state_updates and not blocking:
                        self.state_manager.apply_state_update(state_updates)
                    elif state_updates and blocking:
                        self._log(
                            f"Skipping state updates for scene {scene_num} until blocking issues are resolved",
                            level="warn",
                        )
                    graph_node = SCENE_GRAPH["decision"][1]
                continue

            if graph_node == "editor":
                self._emit("agent_active", {"agent": "Editor", "step": f"scene {scene_num}"})
                pre_edit_text = state["scene_text"]
                editor_input = {"scene_text": state["scene_text"]}
                editor_result = self._run_agent_step(
                    chapter_num=chapter_num,
                    scene_num=scene_num,
                    trace_entries=trace_entries,
                    step_counter=step_counter,
                    agent_name="editor",
                    call_fn=lambda: self.editor.run(editor_input),
                    agent_input={"scene_words": len(state["scene_text"].split())},
                )
                if self._cancelled or self._scene_cancelled:
                    return {"status": "cancelled"}
                state["scene_text"] = self._sanitize_generated_text(
                    editor_result["output"]["scene_text"]
                )
                state["scene_text"] = self._remove_repeated_reference_sentences(
                    state["scene_text"],
                    previous_ending,
                )
                try:
                    self._validate_scene_completion(
                        scene_text=state["scene_text"],
                        scene=scene,
                        chapter_num=chapter_num,
                    )
                except RuntimeError as e:
                    self._log(
                        f"Editor output rejected for scene {scene_num}; keeping writer draft",
                        level="warn",
                        details={"reason": str(e)},
                    )
                    # Try to fix the editor output's truncation before giving up
                    fixed_editor = self._fix_truncated_ending(state["scene_text"], scene)
                    try:
                        self._validate_scene_completion(
                            scene_text=fixed_editor, scene=scene, chapter_num=chapter_num,
                        )
                        state["scene_text"] = fixed_editor
                    except RuntimeError:
                        # Editor output unfixable — fall back to writer draft
                        state["scene_text"] = pre_edit_text
                        # Try to fix truncation on the writer draft too
                        state["scene_text"] = self._fix_truncated_ending(state["scene_text"], scene)
                        # Validate the fallback draft but only WARN — never abort a
                        # chapter because the writer produced a short best-effort draft.
                        try:
                            self._validate_scene_completion(
                                scene_text=state["scene_text"],
                                scene=scene,
                                chapter_num=chapter_num,
                            )
                        except RuntimeError as fallback_err:
                            self._log(
                                f"Scene {scene_num} fallback draft also below minimum; "
                                "continuing with best available text",
                                level="warn",
                                details={"reason": str(fallback_err)},
                            )
                graph_node = SCENE_GRAPH["editor"][0]

        return {
            "status": "complete",
            "scene_text": self._sanitize_generated_text(state["scene_text"]),
            "iterations": iteration,
        }

    @staticmethod
    def _sanitize_generated_text(text: str) -> str:
        cleaned = _REASONING_BLOCK_RE.sub("", text or "")
        cleaned = _REASONING_TAG_RE.sub("", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    @staticmethod
    def _looks_like_meta_output(text: str) -> bool:
        return bool(_META_OUTPUT_RE.search(text or ""))

    @classmethod
    def _remove_repeated_reference_sentences(cls, text: str, reference: str) -> str:
        """Drop long sentences copied from the prior scene/chapter ending."""
        if not text or not reference:
            return text

        reference_keys = {
            cls._sentence_key(sentence)
            for sentence in cls._split_sentences(reference)
            if len(sentence.split()) >= 8
        }
        if not reference_keys:
            return text

        # Work paragraph-by-paragraph to preserve prose structure
        paragraphs = re.split(r"\n{2,}", text)
        result_paragraphs = []
        total_removed = 0
        for para in paragraphs:
            kept = []
            for sentence in cls._split_sentences(para):
                key = cls._sentence_key(sentence)
                if len(sentence.split()) >= 8 and key in reference_keys:
                    total_removed += 1
                    continue
                kept.append(sentence)
            if kept:
                result_paragraphs.append(" ".join(kept))

        if not total_removed:
            return text

        logger.warning(
            "Removed %s sentence(s) copied from the previous scene ending",
            total_removed,
        )
        return "\n\n".join(result_paragraphs).strip() or text

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text or "")
            if sentence.strip()
        ]

    @staticmethod
    def _sentence_key(sentence: str) -> str:
        return re.sub(r"[^a-z0-9']+", " ", sentence.lower()).strip()

    def _validate_scene_completion(self, scene_text: str, scene: dict, chapter_num: int) -> None:
        """Delegate scene validation to QualityController.

        Raises RuntimeError (for backward compatibility) if the scene fails.
        """
        result = self.quality_ctrl.validate(scene_text, scene, chapter_num)
        if result.ok:
            return
        scene_num = scene.get("scene_number", "?")
        if result.truncated:
            self._log(
                f"Scene {scene_num} rejected because it ends mid-sentence",
                level="warn",
                details={"chapter": chapter_num, "words": result.words},
            )
            raise RuntimeError(
                f"Scene {scene_num} in chapter {chapter_num} appears truncated. "
                "WIP checkpoint preserved; retry later to resume."
            )
        # Too short
        self._log(
            f"Scene {scene_num} rejected as incomplete",
            level="warn",
            details={
                "words": result.words,
                "minimum_words": result.minimum,
                "chapter": chapter_num,
            },
        )
        raise RuntimeError(
            f"Scene {scene_num} in chapter {chapter_num} is incomplete "
            f"({result.words} words; expected at least {result.minimum}). "
            "Generation likely stopped early due to rate limits or truncation."
        )

    def _fix_truncated_ending(self, scene_text: str, scene: dict) -> str:
        """Attempt to complete a scene that ends mid-sentence by generating
        a short continuation.  Returns the fixed text, or the original if
        the fix fails."""
        if not scene_text or re.search(r'[.!?]["\']?\s*$', scene_text):
            return scene_text  # Not truncated

        scene_num = scene.get("scene_number", "?")
        self._log(
            f"Scene {scene_num}: attempting to fix truncated ending...",
            level="info",
        )

        # Take the last ~300 chars as context for the completion
        tail = scene_text[-300:].strip()
        fix_prompt = (
            "The following prose scene was cut off mid-sentence. "
            "Write ONLY the remaining 1-3 sentences to complete the final thought naturally. "
            "Do NOT repeat any of the existing text. Output ONLY the missing ending.\n\n"
            f"=== SCENE ENDING (cut off) ===\n{tail}"
        )
        try:
            completion = self.writer._generate_with_fallback(
                fix_prompt,
                "Complete the scene's final sentence naturally. Output only the missing text.",
                max_tokens=200,
            )
            completion = self._sanitize_generated_text(completion or "")
            if completion and len(completion.split()) < 80:
                fixed = f"{scene_text.rstrip()} {completion.lstrip()}".strip()
                if re.search(r'[.!?]["\']?\s*$', fixed):
                    self._log(
                        f"Scene {scene_num}: truncation fixed successfully",
                        level="success",
                    )
                    return fixed
        except Exception as e:
            self._log(
                f"Scene {scene_num}: truncation fix failed: {e}",
                level="warn",
            )
        return scene_text

    _MAX_SCENE_RETRIES = 2

    def _resilient_scene_generation(
        self,
        scene_text: str,
        scene: dict,
        chapter_num: int,
        scene_context: str,
        previous_ending: str,
        trace_entries: list,
        step_counter: dict,
    ) -> str:
        """Validate a scene and automatically retry or fix if it is
        truncated/incomplete, instead of crashing the pipeline."""
        scene_num = scene.get("scene_number", "?")

        for attempt in range(self._MAX_SCENE_RETRIES + 1):
            # Try to fix truncated endings before full validation
            scene_text = self._fix_truncated_ending(scene_text, scene)

            try:
                self._validate_scene_completion(
                    scene_text=scene_text,
                    scene=scene,
                    chapter_num=chapter_num,
                )
                return scene_text  # Passed validation
            except RuntimeError as e:
                remaining = self._MAX_SCENE_RETRIES - attempt
                if remaining > 0 and not (self._cancelled or self._scene_cancelled):
                    self._log(
                        f"Scene {scene_num} validation failed: {e}. "
                        f"Auto-retrying ({remaining} attempt(s) left)...",
                        level="warn",
                    )
                    time.sleep(3)  # Brief pause before retry
                    try:
                        retry_result = self._run_scene_graph(
                            chapter_num=chapter_num,
                            scene=scene,
                            scene_context=scene_context,
                            previous_ending=previous_ending,
                            trace_entries=trace_entries,
                            step_counter=step_counter,
                        )
                        if retry_result.get("status") == "cancelled":
                            break
                        scene_text = self._sanitize_generated_text(
                            retry_result["scene_text"]
                        )
                        scene_text = self._remove_repeated_reference_sentences(
                            scene_text, previous_ending
                        )
                        continue  # Re-validate the new text
                    except Exception as retry_err:
                        self._log(
                            f"Scene {scene_num} retry generation failed: {retry_err}",
                            level="warn",
                        )
                else:
                    # All retries exhausted — accept the best text with a warning
                    self._log(
                        f"Scene {scene_num} could not pass validation after "
                        f"{self._MAX_SCENE_RETRIES} retries. "
                        "Continuing with best available text.",
                        level="warn",
                        details={"last_error": str(e)},
                    )
                    break

        return scene_text  # Return whatever we have

    def _previous_chapter_ending(self, chapter_num: int) -> str:
        if chapter_num <= 1:
            return ""
        prev_chapter_path = os.path.join(
            self.chapters_dir, f"chapter_{(chapter_num - 1):03d}.md"
        )
        if not os.path.exists(prev_chapter_path):
            return ""
        with open(prev_chapter_path, "r", encoding="utf-8") as f:
            return self._extract_ending(f.read())

    @staticmethod
    def _append_continuity_anchor(context: str, previous_ending: str) -> str:
        if not previous_ending:
            return context
        return (
            f"{context.rstrip()}\n\n"
            "=== LAST WRITTEN ENDING (HARD CONTINUITY ANCHOR) ===\n"
            f"{previous_ending}\n"
            "The next chapter must continue from this actual ending. "
            "Do not assume skipped premise events happened off-page."
        )

    @staticmethod
    def _premise_steps(premise: str) -> list[str]:
        """
        Parse premise text into an ordered list of story steps.

        Act/section headings (e.g. "Act One: The Arrival", "Act Two: Marcus") are
        treated as section dividers only — they are NOT added to the steps list.
        This prevents the premise-order validator from treating heading keywords
        as future-violation markers.
        """
        text = premise or ""
        marker = re.search(r"story steps in order\s*:\s*", text, flags=re.IGNORECASE)
        if marker:
            text = text[marker.end():]

        # Match structural section headings regardless of trailing colon
        _HEADING_RE = re.compile(
            r"^(?:act|chapter|part|section|phase)\s+\w",
            flags=re.IGNORECASE,
        )

        steps = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Skip section headings — they are structure, not story steps
            if _HEADING_RE.match(line):
                continue
            # Also skip any short line (<=5 words) that ends with a colon
            if line.endswith(":") and len(line.split()) <= 5:
                continue
            # Strip bullet/numbering prefix
            line = re.sub(r"^(?:[-*]|\d+[\).\:-])\s*", "", line).strip()
            if not line:
                continue
            steps.append(line)
        return steps

    @staticmethod
    def _premise_exhausted(chapter_num: int, steps: list[str]) -> bool:
        if not steps:
            return False
        # A chapter is only truly exhausted when its *start* index is past the
        # end of the steps list (i.e. all steps have been fully covered).
        start_idx = (chapter_num - 1) * config.PREMISE_STEPS_PER_CHAPTER
        return start_idx >= len(steps)

    @staticmethod
    def _premise_exhausted_from_state(steps: list[str], state: dict) -> bool:
        """Return True if all premise steps have already been covered.

        Uses the persisted ``premise_step_completed`` cursor when available so
        that the test stays accurate even when chapters have covered varying
        numbers of steps.  Falls back to the arithmetic estimate for projects
        that pre-date the cursor field (premise_step_completed == -1 means
        'never set; use arithmetic fallback via the chapter number').

        Note: uses strict ``>`` (not ``>=``) so the step at ``completed_idx``
        itself is NOT treated as exhausted — it may still be the *current*
        chapter's last step that hasn't been written yet.  The cursor is only
        advanced *after* a chapter is successfully persisted.
        """
        if not steps:
            return False
        completed_idx = int(
            (state.get("metadata") or {}).get("premise_step_completed", -1)
        )
        if completed_idx < 0:
            # Legacy / fresh project — no cursor yet.  Not exhausted.
            return False
        # All steps covered once we have passed the last valid index.
        # Using strict > means step at completed_idx is still "in window" for
        # the chapter that set it; the NEXT chapter will have completed_idx+1
        # which will then be > len(steps)-1 and trigger post-premise mode.
        return completed_idx > len(steps) - 1

    @staticmethod
    def _allowed_premise_step_index(chapter_num: int, steps: list[str]) -> int:
        """Return the LAST (inclusive) premise-step index this chapter may cover.

        Each chapter is allocated a window of ``config.PREMISE_STEPS_PER_CHAPTER``
        consecutive steps.  Returning the *last* index of that window keeps the
        existing call-sites correct: anything beyond this index is "future" and
        anything at-or-before it is "allowed or completed".

        This is the *arithmetic* fallback used only by legacy / zero-state code
        paths.  Prefer ``_allowed_premise_step_index_from_state`` in the main
        pipeline so the cursor is driven by what was actually written.
        """
        if not steps:
            return -1
        n = config.PREMISE_STEPS_PER_CHAPTER
        # last index in this chapter's window (clamped to list length)
        end_idx = chapter_num * n - 1
        return max(0, min(end_idx, len(steps) - 1))

    @staticmethod
    def _allowed_premise_step_index_from_state(
        chapter_num: int, steps: list[str], state: dict
    ) -> int:
        """Return the LAST (inclusive) premise-step index the current chapter may cover.

        Reads the persisted ``premise_step_completed`` cursor so the window
        always starts exactly where the previous chapter left off, regardless
        of how many steps that chapter actually covered.

        Fallback: if the cursor has never been set (value == -1, i.e. a fresh
        project or one that pre-dates this field), the arithmetic estimate is
        used so old projects keep working.
        """
        if not steps:
            return -1
        n = config.PREMISE_STEPS_PER_CHAPTER
        completed_idx = int(
            (state.get("metadata") or {}).get("premise_step_completed", -1)
        )
        if completed_idx < 0:
            # No cursor yet — use arithmetic estimate (chapter 1 window: steps 0..n-1)
            end_idx = chapter_num * n - 1
        else:
            # Cursor is the LAST step that was covered in the PREVIOUS chapter.
            # This chapter's window starts right after and covers `n` steps.
            end_idx = completed_idx + n
        return max(0, min(end_idx, len(steps) - 1))


    def _append_premise_step_anchor(self, context: str, chapter_num: int, state: dict) -> str:
        steps = self._premise_steps(state.get("metadata", {}).get("premise", ""))
        if not steps:
            return context
        if self._premise_exhausted_from_state(steps, state):
            # Build a specific list of the last few completed beats as forbidden repetition ground
            last_beats = steps[-3:] if len(steps) >= 3 else steps
            last_beats_text = "\n".join(f"  - {s}" for s in last_beats)
            return (
                f"{context.rstrip()}\n\n"
                "=== PREMISE COMPLETE — POST-PREMISE ESCALATION REQUIRED ===\n"
                "All original premise steps have been fully covered in prior chapters.\n"
                "FORBIDDEN: Do NOT repeat, re-describe, or revisit any of the following completed beats:\n"
                f"{last_beats_text}\n"
                "REQUIRED: This chapter must introduce NEW consequences, complications, or escalations that\n"
                "arise AFTER the premise events. Develop the emotional aftermath, new conflicts, or resolution\n"
                "threads that were set in motion. Each chapter must move the story meaningfully forward — \n"
                "never restart from the same emotional beat or scene setup as a prior chapter.\n"
                "The story is entering its final arc. Raise the stakes."
            )
        n = config.PREMISE_STEPS_PER_CHAPTER
        allowed_idx = self._allowed_premise_step_index_from_state(chapter_num, steps, state)
        start_idx = max(0, allowed_idx - n + 1)  # first step in this chapter's window
        completed = steps[:start_idx]
        current_steps = steps[start_idx: allowed_idx + 1]  # window for this chapter
        future = steps[allowed_idx + 1: allowed_idx + 4]
        current_block = "\n".join(f"  - {s}" for s in current_steps)
        return (
            f"{context.rstrip()}\n\n"
            "=== PREMISE ORDER CURSOR ===\n"
            f"Completed premise steps: {' | '.join(completed) if completed else 'None yet.'}\n"
            f"This chapter must cover ALL of these premise steps (in order):\n{current_block}\n"
            f"Future steps forbidden for this chapter: {' | '.join(future) if future else 'None.'}\n"
            "Hard rule: do not introduce events, locations, relationships, private contact, or characters "
            "whose first premise appearance belongs to a future step."
        )


    def _append_used_titles_anchor(self, context: str, state: dict) -> str:
        """Inject previously used chapter titles so the Architect cannot repeat them.

        Titles are read from chapter files (``# Chapter N: Title`` heading) because
        ``state.json`` chapter_summaries only store chapter number + summary text.
        Falls back to extracting the title from the summary's first line if the file
        cannot be read.
        """
        _TITLE_RE = re.compile(r"^#+\s+Chapter\s+\d+\s*:\s*(.+)$", re.IGNORECASE)
        used_titles = []

        summaries = state.get("plot", {}).get("chapter_summaries", [])
        for summary_entry in summaries:
            if not isinstance(summary_entry, dict):
                continue
            ch_num = summary_entry.get("chapter")
            if not ch_num:
                continue
            # Try to read the title from the chapter file heading
            chapter_path = os.path.join(self.chapters_dir, f"chapter_{int(ch_num):03d}.md")
            title = ""
            if os.path.exists(chapter_path):
                try:
                    with open(chapter_path, "r", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                    m = _TITLE_RE.match(first_line)
                    if m:
                        title = m.group(1).strip()
                except OSError:
                    pass
            # Fallback: try first sentence of summary
            if not title:
                raw_summary = str(summary_entry.get("summary", ""))
                first_part = re.split(r"[|.\n]", raw_summary)[0].strip()
                if len(first_part) <= 60:
                    title = first_part
            if title:
                used_titles.append((int(ch_num), title))

        if not used_titles:
            return context

        titles_block = "\n".join(f"  - Chapter {n}: {t}" for n, t in sorted(used_titles))
        return (
            f"{context.rstrip()}\n\n"
            "=== USED CHAPTER TITLES (MUST NOT REPEAT) ===\n"
            f"{titles_block}\n"
            "Hard rule: the new chapter_title you choose MUST be completely different from every title above.\n"
            "Do not reuse any title from this list, even partially (e.g. if 'Breaking Point' is used, "
            "do not use 'Breaking Point Again' or any variant)."
        )

    @staticmethod
    def _character_first_premise_steps(character_names: list[str], steps: list[str]) -> dict[str, int]:
        first = {}
        for name in character_names:
            raw = str(name or "").strip()
            cleaned = re.sub(r"\([^)]*\)", "", raw).strip()
            variants = {raw.lower(), cleaned.lower()}
            canonical = re.sub(r"[^a-z0-9]+", "", cleaned.lower())
            if canonical:
                variants.add(canonical)
            for idx, step in enumerate(steps):
                step_lower = (step or "").lower()
                step_canonical = re.sub(r"[^a-z0-9]+", "", step_lower)
                matched = False
                for variant in variants:
                    if not variant:
                        continue
                    if variant == canonical:
                        if canonical and canonical in step_canonical:
                            matched = True
                            break
                    else:
                        if re.search(r"\b" + re.escape(variant) + r"\b", step_lower):
                            matched = True
                            break
                if matched:
                    first[name] = idx
                    break
        return first

    @staticmethod
    def _distinctive_future_markers(step: str) -> set[str]:
        """Extract distinctive n-gram markers from a future premise step.

        Rules to avoid false positives:
        - 3-grams: require total phrase length >= 14 chars.
        - 2-grams: require BOTH tokens to be >= 6 chars (no short connector words)
          AND total phrase length >= 16 chars.
        - Single tokens are never used as markers alone (too generic).
        """
        tokens = PipelineOrchestrator._marker_tokens(step)
        markers = set()
        # 3-word phrases — more specific, lower bar
        for i in range(0, max(0, len(tokens) - 2)):
            phrase = " ".join(tokens[i:i + 3])
            if len(phrase) >= 14:
                markers.add(phrase)
        # 2-word phrases — only when BOTH tokens are substantial (>=6 chars each)
        for i in range(0, max(0, len(tokens) - 1)):
            t1, t2 = tokens[i], tokens[i + 1]
            if len(t1) >= 6 and len(t2) >= 6:
                phrase = f"{t1} {t2}"
                if len(phrase) >= 16:
                    markers.add(phrase)
        return markers

    @staticmethod
    def _marker_tokens(text: str) -> list[str]:
        lowered = re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower())
        return [
            token for token in lowered.split()
            if len(token) > 4 and token not in _FUTURE_MARKER_TOKEN_STOPWORDS
        ]

    @staticmethod
    def _normalized_text_for_marker_checks(text: str) -> str:
        # Normalize with the same token filtering used by marker extraction.
        # This avoids false positives from punctuation/joiner differences like
        # "Arjun/Marcus" vs "Arjun and Marcus".
        return " ".join(PipelineOrchestrator._marker_tokens(text))

    def _future_premise_violations(
        self,
        text: str,
        chapter_num: int,
        state: dict,
        check_character_names: bool = True,
    ) -> list[str]:
        steps = self._premise_steps(state.get("metadata", {}).get("premise", ""))
        if not steps:
            return []
        if self._premise_exhausted_from_state(steps, state):
            return []
        allowed_idx = self._allowed_premise_step_index_from_state(chapter_num, steps, state)
        lowered = self._normalized_text_for_marker_checks(text)
        allowed_and_prior = self._normalized_text_for_marker_checks(" ".join(steps[:allowed_idx + 1]))
        violations = []

        if check_character_names:
            char_first = self._character_first_premise_steps(
                list(state.get("characters", {}).keys()),
                steps,
            )
            text_lower_for_names = (text or "").lower()
            for name, first_idx in char_first.items():
                if first_idx <= allowed_idx:
                    continue
                # Use strict word-boundary match to prevent "Kai" matching inside
                # "Lukas" or other names that share a substring.
                name_pattern = r"\b" + re.escape(str(name).strip().lower()) + r"\b"
                if re.search(name_pattern, text_lower_for_names):
                    violations.append(f"{name} belongs to future premise step {first_idx + 1}")

        for idx, step in enumerate(steps[allowed_idx + 1:], start=allowed_idx + 1):
            for marker in self._distinctive_future_markers(step):
                if marker in allowed_and_prior:
                    continue
                if marker in lowered:
                    violations.append(f"'{marker}' belongs to future premise step {idx + 1}")
                    break
        return violations

    def _names_in_text(self, text: str, state: dict) -> list[str]:
        found = []
        for name in state.get("characters", {}).keys():
            if re.search(r"\b" + re.escape(str(name).lower()) + r"\b", (text or "").lower()):
                found.append(name)
        return found

    @staticmethod
    def _sentences_from_step(step: str) -> list[str]:
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", step or "")
            if len(s.strip()) > 8
        ]
        return sentences or [step.strip()]

    @staticmethod
    def _primary_character_name(state: dict) -> str:
        names = list(state.get("characters", {}).keys())
        return names[0] if names else ""

    @staticmethod
    def _infer_location_from_text(text: str) -> str:
        """
        Scan the step/summary text for recognisable location keywords and return
        a human-readable location string.  Falls back to a neutral placeholder so
        the writer is never locked to a wrong hardcoded setting.
        """
        text_lower = (text or "").lower()
        # Ordered longest/most-specific first so "shoreditch" beats "club", etc.
        LOCATION_KEYWORDS = [
            ("shoreditch", "Nightclub in Shoreditch"),
            ("dance floor", "Nightclub dance floor"),
            ("nightclub", "Nightclub"),
            ("living room", "Living room"),
            ("master bedroom", "Master bedroom"),
            ("marcus's apartment", "Marcus's apartment"),
            ("lukas's apartment", "Lukas's apartment"),
            ("dining table", "Dining room"),
            ("moves in", "Family home"),
            ("boutique", "Boutique / shopping area"),
            ("shopping", "Shopping area"),
            ("dress", "Shopping area"),
            ("apartment", "Apartment"),
            ("bedroom", "Bedroom"),
            ("kitchen", "Kitchen"),
            ("shower", "Bathroom / shower"),
            ("bathroom", "Bathroom / shower"),
            ("club", "Nightclub"),
            ("paris", "Paris"),
            ("restaurant", "Restaurant"),
            ("hotel", "Hotel room"),
            ("hospital", "Hospital"),
            ("dorm", "University dormitory"),
            ("campus", "University campus"),
            ("university", "University campus"),
            ("college", "University campus"),
            ("london", "London"),
            ("house", "Family home"),
            ("home", "Family home"),
        ]
        for keyword, label in LOCATION_KEYWORDS:
            if keyword in text_lower:
                return label
        return "As established in the story"

    @staticmethod
    def _infer_intimacy_from_text(text: str) -> str:
        """
        Scan step/summary text and return the appropriate intimacy_level string.
        Mirrors the keyword sets used by agents/planner.py so repair-fallback scenes
        get the correct writer mode (explicit/sensual/romantic/none).
        """
        text_lower = (text or "").lower()
        # Explicit sexual content markers
        EXPLICIT_KEYWORDS = (
            "sex", "fuck", "fucking", "blowjob", "deepthroat", "penetrat",
            "intercourse", "oral", "handjob", "fingering", "creampie",
            "orgasm", "cum", "ejaculat", "anal", "masturbat", "cock", "pussy",
            "clit", "rides", "riding", "makes love", "sleeps together",
            "spend the night together", "intense and overwhelming",
            "explicit", "erotic", "raw sex", "passionate sex",
        )
        # Sensual/foreplay markers
        SENSUAL_KEYWORDS = (
            "kiss", "kissing", "make out", "undress", "caress", "touch",
            "tease", "foreplay", "arousal", "desire", "seduce", "heated",
            "steamy", "passionate kiss", "bodies press", "dancing close",
            "pulls her closer",
        )
        # Romantic/tender markers
        ROMANTIC_KEYWORDS = (
            "romantic", "tender", "date", "confession", "cuddle",
            "intimate", "dinner", "candle",
        )
        if any(kw in text_lower for kw in EXPLICIT_KEYWORDS):
            return "explicit"
        if any(kw in text_lower for kw in SENSUAL_KEYWORDS):
            return "sensual"
        if any(kw in text_lower for kw in ROMANTIC_KEYWORDS):
            return "romantic"
        return "none"


    def _inject_missing_character_introductions(
        self,
        chapter_plan: dict,
        chapter_num: int,
        state: dict,
    ) -> dict:

        if chapter_num <= 1 or not isinstance(chapter_plan, dict):
            return chapter_plan

        introduced = self._introduced_character_names(chapter_num, state)
        all_names = list(state.get("characters", {}).keys())
        not_introduced = {name for name in all_names if name not in introduced}
        if not not_introduced:
            return chapter_plan

        events = self._normalize_text_list(chapter_plan.get("key_events"))
        if not events:
            fallback_event = str(chapter_plan.get("plot_direction", "")).strip()
            if fallback_event:
                events = [fallback_event]

        plan_blob = " ".join(
            [
                str(chapter_plan.get("chapter_title", "")),
                str(chapter_plan.get("plot_direction", "")),
                " ".join(events),
                json.dumps(chapter_plan.get("character_arcs", {}), ensure_ascii=False),
            ]
        ).lower()

        lead_name = self._primary_character_name(state) or "The protagonist"
        intro_events = []
        for name in all_names:
            if name not in not_introduced:
                continue
            if not re.search(r"\b" + re.escape(str(name).lower()) + r"\b", plan_blob):
                continue
            if any(self._event_introduces_character(event, name) for event in events):
                continue
            if str(name) == lead_name:
                intro_events.append(f"The chapter opens by introducing {name} on-page.")
            else:
                intro_events.append(f"{lead_name} meets {name} for the first time.")

        if not intro_events:
            return chapter_plan

        chapter_plan["key_events"] = intro_events + events
        chapter_plan["estimated_scenes"] = max(
            config.SCENES_PER_CHAPTER_MIN,
            min(len(chapter_plan["key_events"]), config.SCENES_PER_CHAPTER_MAX),
        )
        constraints = self._normalize_text_list(chapter_plan.get("constraints"))
        marker = "Ensure newly referenced characters are explicitly introduced on-page before private contact."
        if marker not in constraints:
            constraints.append(marker)
        chapter_plan["constraints"] = constraints
        return chapter_plan

    def _clear_pending_chapter_artifacts(self, chapter_num: int) -> None:
        chapter_path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
        wip_path = self._wip_path(chapter_num)
        log_paths = [
            os.path.join(self.logs_dir, f"chapter_{chapter_num:03d}.json"),
            os.path.join(self.logs_dir, f"chapter_{chapter_num:03d}_trace.jsonl"),
        ]

        removed = []
        for path in [chapter_path, wip_path, *log_paths]:
            if os.path.exists(path):
                os.remove(path)
                removed.append(os.path.basename(path))

        if removed:
            self._log(
                "Cleared stale files before chapter auto-retry",
                level="warn",
                details={"chapter": chapter_num, "files": removed},
            )

    def _repair_chapter_plan_to_premise_cursor(
        self,
        chapter_plan: dict,
        chapter_num: int,
        pacing: str,
        state: dict,
    ) -> dict:
        steps = self._premise_steps(state.get("metadata", {}).get("premise", ""))
        if not steps or self._premise_exhausted_from_state(steps, state):
            return chapter_plan
        n = config.PREMISE_STEPS_PER_CHAPTER
        allowed_idx = self._allowed_premise_step_index_from_state(chapter_num, steps, state)
        start_idx = max(0, allowed_idx - n + 1)
        # Collect ALL steps in this chapter's window
        window_steps = steps[start_idx: allowed_idx + 1]
        step = steps[allowed_idx]  # last step in window (for compat checks)
        # Build key_events from every step in the window
        key_events = []
        for ws in window_steps:
            key_events.extend(self._sentences_from_step(ws))
        if not key_events:
            key_events = self._sentences_from_step(step)
        names = self._names_in_text(" ".join(window_steps), state)
        if not names:
            all_names = list(state.get("characters", {}).keys())
            char_first = self._character_first_premise_steps(all_names, steps)
            allowed_names = [
                name for name in all_names
                if char_first.get(name, allowed_idx + 1) <= allowed_idx
            ]
            if not allowed_names and all_names:
                allowed_names = [all_names[0]]
            lead = allowed_names[0] if allowed_names else ""
            names = [lead] if lead else []
        arcs = {
            name: f"{name} develops through the allowed premise steps for this chapter."
            for name in names
        }
        window_desc = " → ".join(window_steps)
        repaired = {
            "chapter_number": chapter_num,
            "chapter_title": (
                chapter_plan.get("chapter_title")
                if chapter_plan and not self._future_premise_violations(
                    str(chapter_plan.get("chapter_title", "")),
                    chapter_num,
                    state,
                )
                else f"Chapter {chapter_num}"
            ),
            "plot_direction": window_desc,
            "character_arcs": arcs,
            "key_events": key_events,
            "constraints": [
                f"Cover premise steps {start_idx + 1}–{allowed_idx + 1} in order.",
                "Do not include future premise steps or characters before their first appearance.",
            ],
            "tone": chapter_plan.get("tone", pacing) if isinstance(chapter_plan, dict) else pacing,
            "estimated_scenes": max(config.SCENES_PER_CHAPTER_MIN, min(len(key_events), config.SCENES_PER_CHAPTER_MAX)),
            "unresolved_threads_to_address": [],
            "new_threads_to_introduce": [],
        }
        return self._inject_missing_character_introductions(
            repaired,
            chapter_num,
            state,
        )

    def _repair_scene_plan_to_chapter_plan(self, chapter_plan: dict, chapter_num: int, state: dict) -> list[dict]:
        key_events = self._normalize_text_list(chapter_plan.get("key_events"))
        if not key_events:
            key_events = [str(chapter_plan.get("plot_direction", "Advance the story."))]

        ordered_names = list(state.get("characters", {}).keys())
        all_names = set(ordered_names)
        steps = self._premise_steps(state.get("metadata", {}).get("premise", ""))
        premise_active = bool(steps) and not self._premise_exhausted_from_state(steps, state)
        allowed_idx = None
        safe_step = ""
        allowed_names = set(all_names)
        if premise_active:
            allowed_idx = self._allowed_premise_step_index_from_state(chapter_num, steps, state)
            safe_step = steps[allowed_idx]
            char_first = self._character_first_premise_steps(ordered_names, steps)
            allowed_names = {
                name for name in ordered_names
                if char_first.get(name, allowed_idx + 1) <= allowed_idx
            }
            if not allowed_names and ordered_names:
                allowed_names = {ordered_names[0]}

        introduced = set(self._introduced_character_names(chapter_num, state))
        allowed_lead = ""
        for name in ordered_names:
            if name in allowed_names:
                allowed_lead = name
                break

        fallback_text = safe_step or "Advance the story."

        def safe_text(text: str, fallback: str) -> str:
            candidate = str(text).strip() if text is not None else ""
            if not candidate:
                candidate = fallback
            if premise_active and self._future_premise_violations(candidate, chapter_num, state):
                return fallback
            return candidate

        safe_plot_direction = safe_text(chapter_plan.get("plot_direction", ""), fallback_text)

        scenes = []
        for idx, event in enumerate(key_events):
            event_text = safe_text(event, safe_plot_direction)
            event_names = self._names_in_text(event_text, state)
            event_names = [name for name in event_names if name in allowed_names]
            if not event_names and allowed_lead:
                event_names = [allowed_lead]

            summary_text = event_text
            key_event_text = event_text
            intro_lead = allowed_lead or "The protagonist"
            for name in event_names:
                if name not in (all_names - introduced):
                    continue
                if not self._event_introduces_character(f"{summary_text} {key_event_text}", name):
                    if name == allowed_lead and allowed_lead:
                        intro = f"The chapter opens by introducing {name} on-page."
                    else:
                        intro = f"{intro_lead} meets {name} for the first time."
                    summary_text = f"{intro} {event_text}".strip()
                    key_event_text = summary_text
                introduced.add(name)

            # Infer location from the step/event text rather than hardcoding a campus
            scene_location = self._infer_location_from_text(summary_text)
            scene_intimacy = self._infer_intimacy_from_text(summary_text)
            scene = {
                "scene_number": idx + 1,
                "type": "setup" if idx == 0 else "build_tension" if idx < len(key_events) - 1 else "resolution",
                "summary": summary_text,
                "characters_present": event_names,
                "location": scene_location,
                "mood": chapter_plan.get("tone", "intimate"),
                "key_events": [key_event_text],
                "dialogue_notes": "Keep the scene focused on this event only.",
                "sensory_details": "Use vivid, grounded sensory details appropriate to the setting.",
                "intimacy_level": scene_intimacy,
                "word_target": max(config.WORDS_PER_SCENE_MIN, min(600, config.WORDS_PER_SCENE_MAX)),
            }
            if premise_active:
                scene_blob = " ".join(
                    [
                        str(scene.get("summary", "")),
                        str(scene.get("location", "")),
                        " ".join(str(e) for e in scene.get("key_events", []) or []),
                        str(scene.get("dialogue_notes", "")),
                        " ".join(str(c) for c in scene.get("characters_present", []) or []),
                    ]
                )
                if self._future_premise_violations(scene_blob, chapter_num, state):
                    scene["summary"] = safe_plot_direction
                    scene["key_events"] = [safe_plot_direction]
                    scene["characters_present"] = [allowed_lead] if allowed_lead else []
                    scene_blob = " ".join(
                        [
                            str(scene.get("summary", "")),
                            str(scene.get("location", "")),
                            " ".join(str(e) for e in scene.get("key_events", []) or []),
                            str(scene.get("dialogue_notes", "")),
                            " ".join(str(c) for c in scene.get("characters_present", []) or []),
                        ]
                    )
                    if self._future_premise_violations(scene_blob, chapter_num, state):
                        scene["summary"] = "Advance the story."
                        scene["key_events"] = ["Advance the story."]
                        scene["characters_present"] = []
            scenes.append(scene)
        while len(scenes) < config.SCENES_PER_CHAPTER_MIN:
            filler_summary = safe_plot_direction or "Resolve the chapter beat without advancing to future premise steps."
            filler_event = safe_plot_direction or "Resolve the chapter beat."
            filler_location = self._infer_location_from_text(filler_summary)
            filler_intimacy = self._infer_intimacy_from_text(filler_summary)
            scenes.append({
                "scene_number": len(scenes) + 1,
                "type": "resolution",
                "summary": filler_summary,
                "characters_present": [allowed_lead] if allowed_lead else [],
                "location": filler_location,
                "mood": chapter_plan.get("tone", "intimate"),
                "key_events": [filler_event],
                "dialogue_notes": "Do not introduce future premise material.",
                "sensory_details": "Focus on emotional aftermath.",
                "intimacy_level": filler_intimacy,
                "word_target": config.WORDS_PER_SCENE_MIN,
            })
        return scenes

    def _validate_plan_premise_order(self, chapter_plan: dict, chapter_num: int, state: dict) -> None:
        plan_blob = " ".join(
            [
                str(chapter_plan.get("chapter_title", "")),
                str(chapter_plan.get("plot_direction", "")),
                " ".join(str(e) for e in chapter_plan.get("key_events", []) or []),
                " ".join(str(c) for c in chapter_plan.get("constraints", []) or []),
            ]
        )
        violations = self._future_premise_violations(plan_blob, chapter_num, state)
        if violations:
            raise RuntimeError(
                "Chapter plan skipped ahead in the premise order: "
                + "; ".join(violations[:4])
                + ". Planner must follow the premise cursor."
            )

        # Check if it missed the required premise steps
        steps = self._premise_steps(state.get("metadata", {}).get("premise", ""))
        premise_active = bool(steps) and not self._premise_exhausted_from_state(steps, state)
        if premise_active:
            n = config.PREMISE_STEPS_PER_CHAPTER
            allowed_idx = self._allowed_premise_step_index_from_state(chapter_num, steps, state)
            start_idx = max(0, allowed_idx - n + 1)
            window_steps = steps[start_idx: allowed_idx + 1]

            key_events = chapter_plan.get("key_events", [])
            if not isinstance(key_events, list):
                key_events = [key_events]
            key_events_lower = [str(e).lower() for e in key_events]

            # Build character name set to filter out of content words
            char_names_lower = {str(name).strip().lower() for name in list(state.get("characters", {}).keys())}
            split_char_names = set()
            for name in char_names_lower:
                split_char_names.update(name.split())
            char_names_lower.update(split_char_names)

            stopwords = {
                'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 
                'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
                'cant', 'cannot', 'could', 'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 
                'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 
                'havent', 'having', 'he', 'hed', 'hell', 'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him', 
                'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is', 'isnt', 
                'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my', 'myself', 'no', 'nor', 'not', 
                'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 
                'own', 'same', 'shant', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt', 'so', 'some', 'such', 
                'than', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 
                'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 
                'under', 'until', 'up', 'very', 'was', 'wasnt', 'we', 'wed', 'well', 'were', 'weve', 'werent', 
                'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which', 'while', 'who', 'whos', 'whom', 
                'why', 'whys', 'with', 'wont', 'would', 'wouldnt', 'you', 'youd', 'youll', 'youre', 'youve', 
                'your', 'yours', 'yourself', 'yourselves', 'story', 'chapter', 'event', 'character', 'characters'
            }

            for step_idx, step in enumerate(window_steps):
                step_lower = step.strip().lower()
                step_words = [
                    re.sub(r'[^a-z0-9]', '', w)
                    for w in step_lower.split()
                ]
                step_content_words = {
                    w for w in step_words
                    if len(w) > 4 and w not in stopwords and w not in char_names_lower
                }

                # Find a matching key event
                matched = False
                for event in key_events_lower:
                    event_words = [
                        re.sub(r'[^a-z0-9]', '', w)
                        for w in event.split()
                    ]
                    event_content_words = {
                        w for w in event_words
                        if len(w) > 4 and w not in stopwords and w not in char_names_lower
                    }

                    # Match if they share at least one non-trivial content word
                    if step_content_words & event_content_words:
                        matched = True
                        break

                if not matched:
                    raise RuntimeError(
                        f"Chapter plan key_events missed mandatory premise step {start_idx + step_idx + 1}: '{step[:60]}...'"
                    )

    def _validate_scene_premise_order(self, scenes: list, chapter_num: int, state: dict) -> None:
        for scene in scenes or []:
            scene_blob = " ".join(
                [
                    str(scene.get("summary", "")),
                    str(scene.get("location", "")),
                    " ".join(str(e) for e in scene.get("key_events", []) or []),
                    str(scene.get("dialogue_notes", "")),
                    " ".join(str(c) for c in scene.get("characters_present", []) or []),
                ]
            )
            violations = self._future_premise_violations(scene_blob, chapter_num, state)
            if violations:
                raise RuntimeError(
                    f"Scene {scene.get('scene_number', '?')} skipped ahead in premise order: "
                    + "; ".join(violations[:4])
                    + ". Regenerate chapter scenes from the allowed premise step."
                )

        # Check if the scenes missed the required steps
        steps = self._premise_steps(state.get("metadata", {}).get("premise", ""))
        premise_active = bool(steps) and not self._premise_exhausted_from_state(steps, state)
        if premise_active:
            n = config.PREMISE_STEPS_PER_CHAPTER
            allowed_idx = self._allowed_premise_step_index_from_state(chapter_num, steps, state)
            start_idx = max(0, allowed_idx - n + 1)
            window_steps = steps[start_idx: allowed_idx + 1]

            # Build character name set to filter out of content words
            char_names_lower = {str(name).strip().lower() for name in list(state.get("characters", {}).keys())}
            split_char_names = set()
            for name in char_names_lower:
                split_char_names.update(name.split())
            char_names_lower.update(split_char_names)

            stopwords = {
                'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 
                'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
                'cant', 'cannot', 'could', 'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 
                'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 
                'havent', 'having', 'he', 'hed', 'hell', 'hes', 'her', 'here', 'heres', 'hers', 'herself', 'him', 
                'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is', 'isnt', 
                'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my', 'myself', 'no', 'nor', 'not', 
                'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 
                'own', 'same', 'shant', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt', 'so', 'some', 'such', 
                'than', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 
                'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 
                'under', 'until', 'up', 'very', 'was', 'wasnt', 'we', 'wed', 'well', 'were', 'weve', 'werent', 
                'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which', 'while', 'who', 'whos', 'whom', 
                'why', 'whys', 'with', 'wont', 'would', 'wouldnt', 'you', 'youd', 'youll', 'youre', 'youve', 
                'your', 'yours', 'yourself', 'yourselves', 'story', 'chapter', 'event', 'character', 'characters'
            }

            for step_idx, step in enumerate(window_steps):
                step_lower = step.strip().lower()
                step_words = [
                    re.sub(r'[^a-z0-9]', '', w)
                    for w in step_lower.split()
                ]
                step_content_words = {
                    w for w in step_words
                    if len(w) > 4 and w not in stopwords and w not in char_names_lower
                }

                matched = False
                for scene in scenes or []:
                    scene_blob = " ".join([
                        str(scene.get("summary", "")),
                        " ".join(str(e) for e in scene.get("key_events", []) or []),
                    ]).lower()
                    
                    scene_words = [
                        re.sub(r'[^a-z0-9]', '', w)
                        for w in scene_blob.split()
                    ]
                    scene_content_words = {
                        w for w in scene_words
                        if len(w) > 4 and w not in stopwords and w not in char_names_lower
                    }

                    if step_content_words & scene_content_words:
                        matched = True
                        break

                if not matched:
                    raise RuntimeError(
                        f"Scene plan missed required premise step {start_idx + step_idx + 1}: '{step[:60]}...'"
                    )

    def _introduced_character_names(self, chapter_num: int, state: dict = None) -> set[str]:
        """Characters that have appeared in completed chapters or persisted plot history."""
        if chapter_num <= 1:
            return set()

        state = state or self.state_manager.state
        characters = list(state.get("characters", {}).keys())
        canonical_to_name = {
            re.sub(r"[^a-z0-9]+", "", re.sub(r"\([^)]*\)", "", name).strip().lower()): name
            for name in characters
        }
        evidence_parts = []

        plot = state.get("plot", {})
        for summary in plot.get("chapter_summaries", []):
            try:
                if int(summary.get("chapter", 0)) < chapter_num:
                    evidence_parts.append(str(summary.get("summary", "")))
            except (TypeError, ValueError):
                continue
        for event in plot.get("major_events", []):
            try:
                if int(event.get("chapter", 0)) < chapter_num:
                    evidence_parts.append(str(event.get("event", "")))
            except (TypeError, ValueError):
                continue

        for num in range(1, chapter_num):
            chapter_path = os.path.join(self.chapters_dir, f"chapter_{num:03d}.md")
            if os.path.exists(chapter_path):
                try:
                    with open(chapter_path, "r", encoding="utf-8") as f:
                        evidence_parts.append(f.read())
                except OSError:
                    pass

        evidence = "\n".join(evidence_parts).lower()
        introduced = set()
        for canonical, exact in canonical_to_name.items():
            if not canonical:
                continue
            exact_lower = str(exact).lower()
            exact_pattern = r"\b" + re.escape(exact_lower) + r"\b"
            short_name = re.sub(r"\([^)]*\)", "", str(exact)).strip().lower()
            short_pattern = r"\b" + re.escape(short_name) + r"\b" if short_name else ""
            canonical_pattern = r"\b" + re.escape(canonical) + r"\b"
            if (
                re.search(exact_pattern, evidence)
                or (short_pattern and re.search(short_pattern, evidence))
                or re.search(canonical_pattern, evidence)
            ):
                introduced.add(exact)
        return introduced

    def _append_character_introduction_anchor(self, context: str, chapter_num: int, state: dict) -> str:
        all_names = list(state.get("characters", {}).keys())
        introduced = self._introduced_character_names(chapter_num, state)
        if chapter_num <= 1:
            introduced_text = "None yet; chapter 1 must introduce characters in premise order."
        else:
            introduced_text = ", ".join(sorted(introduced)) or "None detected."
        not_introduced = [name for name in all_names if name not in introduced]
        not_introduced_text = ", ".join(not_introduced) or "None."
        return (
            f"{context.rstrip()}\n\n"
            "=== CHARACTER INTRODUCTION CONTINUITY ===\n"
            f"Already introduced on-page: {introduced_text}\n"
            f"Not introduced yet: {not_introduced_text}\n"
            "Hard rule: a not-yet-introduced character cannot text, call, flirt, coordinate, "
            "meet privately, or affect the plot until this chapter first shows their actual introduction/meeting."
        )

    @staticmethod
    def _event_introduces_character(text: str, name: str) -> bool:
        lowered = (text or "").lower()
        name_lower = str(name or "").lower()
        if name_lower not in lowered:
            return False
        intro_markers = (
            "meet", "meets", "meeting", "first encounter", "introduced",
            "introduces", "encounters", "runs into", "sees for the first time",
            "arrives", "arrival", "appears", "enters", "shows up", "named",
            "first sees", "first time seeing",
        )
        return any(marker in lowered for marker in intro_markers)

    def _validate_plan_character_introductions(self, chapter_plan: dict, chapter_num: int, state: dict) -> None:
        if chapter_num <= 1:
            return
        introduced = self._introduced_character_names(chapter_num, state)
        all_names = set(state.get("characters", {}).keys())
        not_introduced = all_names - introduced
        if not not_introduced:
            return

        events = chapter_plan.get("key_events", []) or []
        narrative_blob = " ".join(
            [
                str(chapter_plan.get("chapter_title", "")),
                str(chapter_plan.get("plot_direction", "")),
                " ".join(str(e) for e in events),
                " ".join(str(c) for c in chapter_plan.get("constraints", []) or []),
            ]
        ).lower()

        invalid = []
        for name in sorted(not_introduced):
            if str(name).lower() not in narrative_blob:
                continue
            if any(self._event_introduces_character(str(event), name) for event in events):
                continue
            if self._event_introduces_character(
                str(chapter_plan.get("plot_direction", "")),
                name,
            ):
                continue
            invalid.append(name)

        if invalid:
            raise RuntimeError(
                "Chapter plan violates character introduction continuity: "
                f"{', '.join(invalid)} used before being introduced. "
                "Planner must follow the next premise step in order."
            )

    def _validate_scene_character_introductions(self, scenes: list, chapter_num: int, state: dict) -> None:
        if chapter_num <= 1:
            return

        introduced = set(self._introduced_character_names(chapter_num, state))
        all_names = set(state.get("characters", {}).keys())
        private_markers = (
            "phone", "message", "messages", "text", "texts", "chat", "call",
            "private", "secretly talking", "coordinate", "coordination",
        )

        for scene in scenes or []:
            scene_num = scene.get("scene_number", "?")
            scene_blob = " ".join(
                [
                    str(scene.get("summary", "")),
                    str(scene.get("location", "")),
                    " ".join(str(e) for e in scene.get("key_events", []) or []),
                    str(scene.get("dialogue_notes", "")),
                ]
            )
            scene_blob_lower = scene_blob.lower()
            names_here = set(scene.get("characters_present", []) or [])
            names_here.update(
                name for name in all_names
                if re.search(r"\b" + re.escape(str(name).lower()) + r"\b", scene_blob_lower)
            )

            for name in sorted(names_here & (all_names - introduced)):
                introduces_now = self._event_introduces_character(scene_blob, name)
                if not introduces_now:
                    raise RuntimeError(
                        f"Scene {scene_num} uses {name} before they are introduced on-page. "
                        "Regenerate the chapter plan/scenes in premise order."
                    )
                if any(marker in scene_blob_lower for marker in private_markers):
                    raise RuntimeError(
                        f"Scene {scene_num} starts {name} in private contact before their first on-page meeting. "
                        "The scene must show the actual introduction first."
                    )
                introduced.add(name)

    @staticmethod
    def _looks_like_named_character(name: str) -> bool:
        cleaned = str(name or "").strip()
        if not cleaned:
            return False
        if cleaned.lower() in _GENERIC_CHARACTER_LABELS:
            return False
        if "," in cleaned or len(cleaned.split()) > 4:
            return False
        return cleaned[0].isupper() or "(" in cleaned

    # ── Character lifecycle / age-awareness ──────────────────────────
    # Characters born during the story (children, babies) must be written
    # age-appropriately.  This method infers a lifecycle stage from the
    # premise text + the chapter we're currently generating.

    # Two patterns to catch both "gives birth to Kai" and "Leo is born":
    _BIRTH_PATTERN_AFTER = re.compile(
        r"gives?\s+birth\s+to\s+(?:a\s+)?(?P<name>[A-Z][a-z]{2,})\b",
        re.IGNORECASE,
    )
    _BIRTH_PATTERN_BEFORE = re.compile(
        r"\b(?P<name>[A-Z][a-z]{2,})\s+is\s+born\b",
        re.IGNORECASE,
    )
    # Words that look like names but aren't (common false positives)
    _NOT_NAMES = {
        "fair", "fourth", "second", "third", "first", "another",
        "the", "dark", "light", "baby", "child", "boy", "girl",
    }

    def _build_character_profiles(
        self,
        scene: dict,
        chapter_num: int,
    ) -> str:
        """Return a formatted character-profiles block for the writer prompt.

        For each character in scene['characters_present'], include:
          - description (from state)
          - lifecycle stage (newborn / baby / toddler / child / teen / adult)
          - any behavioural constraints (e.g. 'cannot speak or act autonomously')

        The lifecycle stage is determined by checking the premise for birth
        events and comparing against the current chapter number.
        """
        state = self.state_manager.load()
        characters = state.get("characters", {})
        premise = state.get("metadata", {}).get("premise", "")

        # Build a mapping: character_name_lower -> chapter of birth (estimated)
        birth_chapter: dict[str, int] = {}
        if premise:
            # Parse premise to find birth events and map them to approximate
            # chapter numbers based on act ordering.
            acts = re.split(r"Act\s+\w+", premise, flags=re.IGNORECASE)
            total_acts = max(len(acts) - 1, 1)
            total_chapters = max(chapter_num, state.get("metadata", {}).get("total_chapters", 15))
            chapters_per_act = max(total_chapters / total_acts, 1)

            for act_idx, act_text in enumerate(acts):
                for pattern in (self._BIRTH_PATTERN_AFTER, self._BIRTH_PATTERN_BEFORE):
                    for m in pattern.finditer(act_text):
                        name = m.group("name").strip()
                        if name.lower() in self._NOT_NAMES:
                            continue
                        estimated_ch = int(act_idx * chapters_per_act) + 1
                        birth_chapter[name.lower()] = estimated_ch

        scene_chars = scene.get("characters_present", [])
        if not scene_chars:
            return ""

        lines = ["CHARACTER PROFILES (respect ages and lifecycle stages):"]
        for name in scene_chars:
            name_str = str(name).strip()
            char_info = characters.get(name_str, {})
            desc = char_info.get("description", "").strip()
            role = char_info.get("role", "supporting")

            # Determine lifecycle stage
            name_lower = name_str.lower()
            # Also check first name only
            first_name_lower = name_lower.split()[0] if name_lower else name_lower

            born_at = birth_chapter.get(name_lower) or birth_chapter.get(first_name_lower)
            if born_at is not None:
                chapters_since_birth = max(0, chapter_num - born_at)
                if chapters_since_birth <= 0:
                    stage = "newborn (just born this chapter)"
                    constraint = "A newborn baby. Cannot speak, walk, or act. Can only cry, sleep, or be held."
                elif chapters_since_birth <= 1:
                    stage = "baby (infant)"
                    constraint = "A baby. Cannot speak sentences, walk independently, or have conversations."
                elif chapters_since_birth <= 3:
                    stage = "toddler"
                    constraint = "A toddler. May babble, toddle, reach for things. Cannot have adult conversations."
                elif chapters_since_birth <= 6:
                    stage = "young child"
                    constraint = "A young child. Can speak simple sentences, play, and express basic emotions."
                else:
                    stage = "child / growing up"
                    constraint = "A child growing up. Can speak and interact but is still a minor, not an adult."
            else:
                stage = "adult"
                constraint = ""

            profile = f"  - {name_str} ({role}): {desc}" if desc else f"  - {name_str} ({role})"
            profile += f" [Age/stage: {stage}]"
            if constraint:
                profile += f"\n    ⚠ CONSTRAINT: {constraint}"
            lines.append(profile)

        return "\n".join(lines)

    def _ensure_scene_characters_known(self, scene: dict) -> None:
        known_keys = {
            re.sub(r"[^a-z0-9]+", "", re.sub(r"\([^)]*\)", "", name).strip().lower())
            for name in self.state_manager.get_characters().keys()
        }
        new_chars = {}
        for name in scene.get("characters_present", []):
            canonical = re.sub(
                r"[^a-z0-9]+", "",
                re.sub(r"\([^)]*\)", "", str(name)).strip().lower(),
            )
            if canonical not in known_keys and self._looks_like_named_character(name):
                new_chars[str(name).strip()] = {
                    "description": "Supporting character introduced by the story.",
                    "role": "supporting",
                    "generated": True,
                }
        if new_chars:
            self.state_manager.apply_state_update({"characters": new_chars})

    @staticmethod
    def _normalize_text_list(value) -> list[str]:
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [chunk.strip() for chunk in re.split(r"[\n;,]", value) if chunk.strip()]
        return []

    @staticmethod
    def _event_tokens(text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9']+", str(text or "").lower())
            if len(token) > 3 and token not in {
                "then", "after", "before", "during", "while", "with", "from", "into",
                "they", "them", "this", "that", "their", "scene", "event", "chapter",
            }
        }

    @classmethod
    def _event_matches_brief(cls, event_text: str, brief_text: str) -> bool:
        event_norm = re.sub(r"\s+", " ", str(event_text or "").strip().lower())
        brief_norm = re.sub(r"\s+", " ", str(brief_text or "").strip().lower())
        if not event_norm or not brief_norm:
            return False
        if event_norm in brief_norm or brief_norm in event_norm:
            return True
        event_tokens = cls._event_tokens(event_norm)
        brief_tokens = cls._event_tokens(brief_norm)
        if not event_tokens or not brief_tokens:
            return False
        return len(event_tokens & brief_tokens) >= 2

    @classmethod
    def _manual_brief_beats(cls, scene_brief: str) -> list[str]:
        """Split a manual scene brief into user-authored beats without inventing structure."""
        brief = str(scene_brief or "").strip()
        if not brief:
            return []

        # Users commonly enter manual scene beats as comma/newline/semicolon-separated phrases.
        # Avoid splitting on plain "and" because character pairs like "Rosna and Royce" matter.
        raw_beats = re.split(r"[\n;,]+|\s+(?:then|after that|afterwards|next)\s+", brief, flags=re.IGNORECASE)
        beats = [re.sub(r"\s+", " ", beat).strip(" .") for beat in raw_beats]
        beats = [beat for beat in beats if beat]
        return beats or [brief]

    @staticmethod
    def _coerce_manual_scene_type(raw_type: str, idx: int, total: int) -> str:
        allowed = {"setup", "build_tension", "peak", "resolution", "transition"}
        candidate = str(raw_type or "").strip().lower().replace(" ", "_")
        if candidate in allowed:
            return candidate
        if idx == 0:
            return "setup"
        if idx >= total - 1:
            return "resolution"
        return "build_tension"

    def _active_planning_model(self):
        if self._backend in {"groq", "gemini"} and self.cloud_model:
            return self.cloud_model
        return self.local_model

    def _enhance_manual_scene(
        self,
        chapter_title: str,
        scene_number: int,
        scene_brief: str,
        pacing: str,
        context: str,
        character_names: list[str],
        previous_scenes_summary: str = "",
    ) -> dict:
        model = self._active_planning_model()
        prompt = MANUAL_SCENE_ENHANCER_PROMPT.format(
            chapter_title=chapter_title,
            scene_number=scene_number,
            scene_brief=scene_brief,
            pacing=pacing,
            character_names=", ".join(character_names) if character_names else "None specified",
            previous_scenes_summary=previous_scenes_summary or "None (this is the first scene).",
            context=context[:10000],
        )
        schema = {
            "type": "object",
            "properties": {
                "type": {"type": "string"},
                "summary": {"type": "string"},
                "characters_present": {"type": "array", "items": {"type": "string"}},
                "location": {"type": "string"},
                "mood": {"type": "string"},
                "key_events": {"type": "array", "items": {"type": "string"}},
                "dialogue_notes": {"type": "string"},
                "sensory_details": {"type": "string"},
                "word_target": {"type": "integer"},
            },
            "required": ["summary", "key_events"],
        }
        response = model.generate_with_retry(
            prompt=prompt,
            system=MANUAL_SCENE_ENHANCER_SYSTEM,
            schema=schema,
            temperature=config.AGENT_TEMPERATURES["planner"],
            max_tokens=1200 if self._backend in {"groq", "gemini"} else None,
        )
        parsed = response.as_json() if response else None
        if not isinstance(parsed, dict):
            parsed = {}

        summary = str(parsed.get("summary", "")).strip() or scene_brief
        key_events = self._normalize_text_list(parsed.get("key_events"))
        if not key_events:
            key_events = [summary]
        brief = str(scene_brief or "").strip()
        required_beats = self._manual_brief_beats(brief)
        missing_beats = [
            beat for beat in required_beats
            if not any(self._event_matches_brief(event, beat) for event in key_events)
        ]
        if missing_beats:
            key_events = missing_beats + key_events
        if required_beats:
            # Keep the user's exact beat order at the front so later planning prose cannot
            # accidentally turn an introduction into off-page backstory.
            ordered_events = []
            seen_event_keys = set()
            for event in required_beats + key_events:
                event_key = re.sub(r"[^a-z0-9']+", " ", str(event).lower()).strip()
                if event_key and event_key not in seen_event_keys:
                    ordered_events.append(event)
                    seen_event_keys.add(event_key)
            key_events = ordered_events
        characters_present = self._normalize_text_list(parsed.get("characters_present"))
        if not characters_present and character_names:
            characters_present = character_names[:1]
        location = str(parsed.get("location", "")).strip() or "Primary story location"
        mood = str(parsed.get("mood", "")).strip() or pacing
        dialogue_notes = str(parsed.get("dialogue_notes", "")).strip()
        sensory_details = str(parsed.get("sensory_details", "")).strip()
        try:
            word_target = int(parsed.get("word_target", 0))
        except (TypeError, ValueError):
            word_target = 0
        if word_target <= 0:
            word_target = 600
        word_target = max(config.WORDS_PER_SCENE_MIN, min(word_target, config.WORDS_PER_SCENE_MAX))

        return {
            "scene_number": scene_number,
            "type": str(parsed.get("type", "")),
            "summary": summary,
            "characters_present": characters_present,
            "location": location,
            "mood": mood,
            "key_events": key_events,
            "original_user_brief": brief,
            "required_beats": required_beats,
            "dialogue_notes": dialogue_notes,
            "sensory_details": sensory_details,
            "word_target": word_target,
        }

    def _build_manual_chapter_spec(
        self,
        chapter_num: int,
        chapter_title: str,
        scene_briefs: list[str],
        pacing: str,
        state: dict,
    ) -> dict:
        previous_chapter_ending = self._previous_chapter_ending(chapter_num)
        context = self._append_continuity_anchor(
            self.state_manager.get_context_window(include_premise=False),
            previous_chapter_ending,
        )
        context = self._append_character_introduction_anchor(context, chapter_num, state)
        context = self._append_premise_step_anchor(context, chapter_num, state)

        seen_names = set()
        character_names = []
        for name in state.get("characters", {}).keys():
            key = str(name).lower()
            if key not in seen_names:
                seen_names.add(key)
                character_names.append(name)

        scenes = []
        for idx, brief in enumerate(scene_briefs):
            self._emit("status", f"Enhancing manual scene {idx + 1}/{len(scene_briefs)}...")
            # Build summary of previously enhanced scenes for continuity
            prev_summaries = "\n".join(
                f"Scene {i + 1}: {s.get('summary', '')[:120]}"
                for i, s in enumerate(scenes)
            ) if scenes else ""
            scene = self._enhance_manual_scene(
                chapter_title=chapter_title,
                scene_number=idx + 1,
                scene_brief=brief,
                pacing=pacing,
                context=context,
                character_names=character_names,
                previous_scenes_summary=prev_summaries,
            )
            scene["type"] = self._coerce_manual_scene_type(
                scene.get("type", ""),
                idx,
                len(scene_briefs),
            )
            scenes.append(scene)

        chapter_key_events = [scene.get("summary", "") for scene in scenes if scene.get("summary")]
        chapter_plan = {
            "chapter_number": chapter_num,
            "chapter_title": chapter_title,
            "plot_direction": "Manual chapter plan provided by user scene briefs.",
            "character_arcs": {
                name: f"{name} progresses through user-directed chapter beats."
                for name in character_names
            },
            "key_events": chapter_key_events,
            "constraints": [
                "Manual mode: preserve user-provided scene intent and sequence.",
            ],
            "tone": pacing,
            "estimated_scenes": len(scenes),
            "unresolved_threads_to_address": [],
            "new_threads_to_introduce": [],
        }
        return {"chapter_plan": chapter_plan, "scenes": scenes}

    def generate_chapter_manual(
        self,
        chapter_title: str,
        scene_briefs: list[str],
        pacing: str = "moderate",
    ) -> dict:
        cleaned_title = str(chapter_title or "").strip()
        briefs = [str(scene).strip() for scene in (scene_briefs or []) if str(scene).strip()]
        if not cleaned_title:
            raise ValueError("chapter_title is required for manual generation")
        if not briefs:
            raise ValueError("At least one non-empty scene brief is required")

        state = self.state_manager.load()
        chapter_num = state["metadata"]["current_chapter"] + 1
        manual_spec = self._build_manual_chapter_spec(
            chapter_num=chapter_num,
            chapter_title=cleaned_title,
            scene_briefs=briefs,
            pacing=pacing,
            state=state,
        )
        return self.generate_chapter(pacing=pacing, manual_spec=manual_spec)

    # ─── Interactive Scene-by-Scene Manual Generation ─────────────────

    def start_manual_session(
        self,
        chapter_title: str,
        pacing: str = "moderate",
    ) -> dict:
        """
        Start an interactive manual chapter session.
        Returns a session dict that must be passed to generate_manual_scene().
        """
        cleaned_title = str(chapter_title or "").strip()
        if not cleaned_title:
            raise ValueError("chapter_title is required")

        self.state_manager.normalize_character_traits()
        state = self.state_manager.load()
        chapter_num = state["metadata"]["current_chapter"] + 1

        # Build context for the chapter
        previous_chapter_ending = self._previous_chapter_ending(chapter_num)
        context = self._append_continuity_anchor(
            self.state_manager.get_context_window(include_premise=False),
            previous_chapter_ending,
        )
        context = self._append_character_introduction_anchor(context, chapter_num, state)
        context = self._append_premise_step_anchor(context, chapter_num, state)

        # Gather character names
        seen_names = set()
        character_names = []
        for name in state.get("characters", {}).keys():
            key = str(name).lower()
            if key not in seen_names:
                seen_names.add(key)
                character_names.append(name)

        # Set genre on writer/editor
        genre = state.get("metadata", {}).get("genre", "")
        self.writer.set_genre(genre)
        self.editor.set_genre(genre)

        session = {
            "chapter_num": chapter_num,
            "chapter_title": cleaned_title,
            "pacing": pacing,
            "context": context,
            "character_names": character_names,
            "previous_ending": previous_chapter_ending or "",
            "completed_scenes": [],
            "completed_scene_plans": [],
            "scene_counter": 0,
            "trace_entries": [],
            "log_entries": [],
            "step_counter": {"steps": 0},
            "estimated_tokens_used": 0,
        }

        self._emit("chapter_start", {"chapter": chapter_num})
        self._log(
            f"═══ Starting Manual Chapter {chapter_num}: \"{cleaned_title}\" ═══",
            level="header",
            details={"pacing": pacing, "project": self.project_name, "mode": "interactive_manual"},
        )

        return session

    def resume_manual_chapter(self, chapter_num: int) -> dict:
        """
        Resume an existing chapter by converting it back into an active manual session.
        This rolls back the story state and vector store to before this chapter was finalized.
        """
        chapter_num = int(chapter_num)
        if chapter_num < 1:
            raise ValueError("Chapter number must be >= 1")

        state = self.state_manager.load()
        
        # Calculate total scenes written up to chapter_num - 1
        total_scenes_written = 0
        separator = "\n\n* * *\n\n"
        for i in range(1, chapter_num):
            path = os.path.join(self.chapters_dir, f"chapter_{i:03d}.md")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    total_scenes_written += len(content.split(separator))

        # Roll back the state and vector store
        self.state_manager.prune_after_chapter(chapter_num - 1, total_scenes_written)
        self.vector_store.prune_chapters(chapter_num - 1)

        # Read the chapter to resume
        chapter_path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
        if not os.path.exists(chapter_path):
            raise FileNotFoundError(f"Chapter {chapter_num} file not found.")

        with open(chapter_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Try to extract the title from the heading
        chapter_title = f"Chapter {chapter_num}"
        if content.startswith("# Chapter"):
            heading = content.split("\n", 1)[0]
            if ":" in heading:
                chapter_title = heading.split(":", 1)[1].strip()

        # Remove the chapter heading from the first scene text
        parts = content.split(separator)
        if parts:
            first_part = parts[0]
            if first_part.startswith("# Chapter"):
                first_part_lines = first_part.split("\n", 2)
                if len(first_part_lines) >= 3 and first_part_lines[0].startswith("# Chapter"):
                    parts[0] = first_part_lines[-1].strip()

        completed_scenes = [p.strip() for p in parts if p.strip()]
        completed_scene_plans = [{"summary": f"Recovered scene {i+1}", "type": "continuation"} for i in range(len(completed_scenes))]

        # Refresh state after prune
        self.state_manager.normalize_character_traits()
        state = self.state_manager.load()

        # Build context
        previous_chapter_ending = self._previous_chapter_ending(chapter_num)
        context = self._append_continuity_anchor(
            self.state_manager.get_context_window(include_premise=False),
            previous_chapter_ending,
        )
        context = self._append_character_introduction_anchor(context, chapter_num, state)
        context = self._append_premise_step_anchor(context, chapter_num, state)

        # Gather character names
        seen_names = set()
        character_names = []
        for name in state.get("characters", {}).keys():
            key = str(name).lower()
            if key not in seen_names:
                seen_names.add(key)
                character_names.append(name)

        # Set genre
        genre = state.get("metadata", {}).get("genre", "")
        self.writer.set_genre(genre)
        self.editor.set_genre(genre)

        session = {
            "chapter_num": chapter_num,
            "chapter_title": chapter_title,
            "pacing": "moderate",
            "context": context,
            "character_names": character_names,
            "previous_ending": self._extract_ending(completed_scenes[-1]) if completed_scenes else (previous_chapter_ending or ""),
            "completed_scenes": completed_scenes,
            "completed_scene_plans": completed_scene_plans,
            "scene_counter": len(completed_scenes),
            "trace_entries": [],
            "log_entries": [],
            "step_counter": {"steps": 0},
            "estimated_tokens_used": 0,
        }

        self._emit("chapter_start", {"chapter": chapter_num})
        self._log(
            f"═══ Resuming Manual Chapter {chapter_num}: \"{chapter_title}\" ═══",
            level="header",
            details={"project": self.project_name, "mode": "resume"},
        )

        return session

    def generate_manual_scene(
        self,
        session: dict,
        scene_brief: str,
    ) -> dict:
        """
        Generate a single scene in an interactive manual session.
        Returns the scene result with text, and mutates the session in-place.
        """
        self._scene_cancelled = False
        brief = str(scene_brief or "").strip()
        if not brief:
            raise ValueError("scene_brief cannot be empty")

        chapter_num = session["chapter_num"]
        chapter_title = session["chapter_title"]
        pacing = session["pacing"]
        base_context = session["context"]
        character_names = session["character_names"]
        previous_ending = session["previous_ending"]
        trace_entries = session["trace_entries"]
        step_counter = session["step_counter"]
        scene_idx = session["scene_counter"]
        scene_number = scene_idx + 1

        self._log(
            f"Manual scene {scene_number}: enhancing user brief...",
            details={"brief": brief[:120]},
        )

        # ── Build rich previous-scenes summary for the enhancer ──────────
        # Include both the plan summary AND the actual ending of each
        # completed scene so the enhancer knows exactly what was written
        # and can avoid generating repeated content.
        prev_summaries = []
        completed_scenes = session.get("completed_scenes", [])
        completed_plans = session.get("completed_scene_plans", [])
        for i in range(max(len(completed_plans), len(completed_scenes))):
            plan_summary = ""
            plan_location = ""
            if i < len(completed_plans):
                plan_summary = completed_plans[i].get("summary", "")[:120]
                plan_location = completed_plans[i].get("location", "")
            loc_str = f" (Location: {plan_location})" if plan_location else ""
            scene_ending = ""
            if i < len(completed_scenes):
                scene_ending = self._extract_ending(completed_scenes[i], max_chars=200)
            if plan_summary and scene_ending:
                prev_summaries.append(
                    f"Scene {i + 1}{loc_str}: {plan_summary}\n  [Ended with]: \"{scene_ending}\""
                )
            elif plan_summary:
                prev_summaries.append(f"Scene {i + 1}{loc_str}: {plan_summary}")
            elif scene_ending:
                prev_summaries.append(f"Scene {i + 1}{loc_str}: [Ended with]: \"{scene_ending}\"")
        previous_scenes_summary = "\n".join(prev_summaries) if prev_summaries else "None (this is the first scene)."

        # ── Enrich context with completed-scene recap ────────────────────
        # The base session context was captured once at session start. We
        # append a recap of what was actually written so the enhancer and
        # writer don't re-generate already-covered events.
        context = base_context
        if completed_scenes:
            recap_parts = []
            for i, scene_text in enumerate(completed_scenes):
                ending = self._extract_ending(scene_text, max_chars=180)
                plan_sum = ""
                plan_loc = ""
                if i < len(completed_plans):
                    plan_sum = completed_plans[i].get("summary", "")[:100]
                    plan_loc = completed_plans[i].get("location", "")
                label = plan_sum or f"Scene {i + 1}"
                loc_label = f" [at {plan_loc}]" if plan_loc else ""
                recap_parts.append(f"  Scene {i + 1}{loc_label} — {label}: ...{ending}")
            recap_block = "\n".join(recap_parts)
            context = (
                f"{context.rstrip()}\n\n"
                "=== ALREADY WRITTEN IN THIS CHAPTER (do NOT repeat these events) ===\n"
                f"{recap_block}\n"
                "=== END OF ALREADY WRITTEN SCENES ===\n"
                "The next scene must continue FROM where the last scene ended. "
                "Do NOT re-introduce, recap, or re-describe events from the scenes above."
            )

        # Enhance the scene brief (lightly — preserving user intent)
        scene = self._enhance_manual_scene(
            chapter_title=chapter_title,
            scene_number=scene_number,
            scene_brief=brief,
            pacing=pacing,
            context=context,
            character_names=character_names,
            previous_scenes_summary=previous_scenes_summary,
        )
        total_known = scene_idx + 1  # Only know about current scene count
        scene["type"] = self._coerce_manual_scene_type(
            scene.get("type", ""),
            scene_idx,
            max(total_known, 2),  # Avoid treating scene 1 as 'resolution'
        )

        self._emit("scenes_planned", {
            "count": scene_number,
            "types": [sp.get("type", "?") for sp in session["completed_scene_plans"]] + [scene["type"]],
        })
        self._log(
            f"Scene {scene_number} enhanced: \"{scene.get('summary', '')[:100]}\"",
            level="success",
            details={
                "type": scene["type"],
                "characters": scene.get("characters_present", []),
                "word_target": scene.get("word_target", 600),
            },
        )

        # Inject continuity metadata so the writer can detect location changes
        # Use the actual plan location of the previous scene if available, otherwise fall back to parsing text.
        prev_plans = session.get("completed_scene_plans", [])
        if prev_plans:
            scene["prev_location"] = prev_plans[-1].get("location", "")
        else:
            scene["prev_location"] = self._extract_scene_location(previous_ending)

        if not scene.get("narrative_bridge"):
            if prev_plans:
                prev_s = prev_plans[-1].get("summary", "")
                if prev_s:
                    scene["narrative_bridge"] = (
                        f"This scene follows directly after: {prev_s[:120]}"
                    )

        # Inject character lifecycle profiles so the writer knows ages
        scene["character_profiles"] = self._build_character_profiles(
            scene, chapter_num
        )

        # Retrieve scene-level context
        self._emit("agent_active", {"agent": "Retriever", "step": f"scene {scene_number}"})
        scene_context = self.retriever.retrieve_context(
            scene_plan=scene,
            chapter_num=chapter_num,
            previous_ending=previous_ending,
            top_k=2 if self._backend in {"groq", "gemini"} else None,
            max_chars=2200 if self._backend in {"groq", "gemini"} else None,
        )

        # ── Enrich writer context with completed-scene recap ─────────────
        # The retriever returns generic vector-store context. We append a
        # summary of scenes already written in THIS chapter so the writer
        # can avoid repeating them.
        if completed_scenes:
            scene_recap_parts = []
            for i, st in enumerate(completed_scenes):
                plan_label = ""
                plan_loc = ""
                if i < len(completed_plans):
                    plan_label = completed_plans[i].get("summary", "")[:80]
                    plan_loc = completed_plans[i].get("location", "")
                loc_str = f" [at {plan_loc}]" if plan_loc else ""
                ending_snip = self._extract_ending(st, max_chars=150)
                scene_recap_parts.append(
                    f"  Scene {i + 1}{loc_str} ({plan_label}): ...{ending_snip}"
                )
            scene_context = (
                f"{scene_context.rstrip()}\n\n"
                "=== SCENES ALREADY WRITTEN IN THIS CHAPTER ===\n"
                + "\n".join(scene_recap_parts)
                + "\n=== END ===\n"
                "CRITICAL: Do NOT repeat, recap, or re-describe any of the events above. "
                "Continue the story from where Scene " + str(len(completed_scenes)) + " ended."
            )

        # For scene 2+, use a richer previous_ending from the actual last
        # completed scene text (the session cache may be a shorter extract)
        if completed_scenes:
            previous_ending = self._extract_ending(completed_scenes[-1])

        # Run the full scene graph (writer → critic → decision → editor)
        self._emit("scene_start", {
            "scene": scene_number,
            "total": scene_number,  # Dynamic total in interactive mode
            "type": scene["type"],
        })
        self._log(
            f"─── Scene {scene_number} ({scene['type']}) ───",
            level="header",
            details={
                "summary": scene.get("summary", "")[:120],
                "characters": scene.get("characters_present", []),
            },
        )

        scene_result = self._run_scene_graph(
            chapter_num=chapter_num,
            scene=scene,
            scene_context=scene_context,
            previous_ending=previous_ending,
            trace_entries=trace_entries,
            step_counter=step_counter,
        )

        if scene_result.get("status") == "cancelled":
            return {"status": "cancelled", "scene_number": scene_number}
        if self._cancelled or self._scene_cancelled:
            return {"status": "cancelled", "scene_number": scene_number}

        scene_text = self._sanitize_generated_text(scene_result["scene_text"])
        scene_text = self._remove_repeated_reference_sentences(scene_text, previous_ending)
        scene_text = self._resilient_scene_generation(
            scene_text=scene_text,
            scene=scene,
            chapter_num=chapter_num,
            scene_context=scene_context,
            previous_ending=previous_ending,
            trace_entries=trace_entries,
            step_counter=step_counter,
        )

        post_edit_words = len(scene_text.split())
        session["estimated_tokens_used"] += int(post_edit_words * 1.35)

        # Update session state
        session["completed_scenes"].append(scene_text)
        session["completed_scene_plans"].append(scene)
        session["previous_ending"] = self._extract_ending(scene_text)
        session["scene_counter"] = scene_idx + 1
        self._ensure_scene_characters_known(scene)

        # Post-scene narrative evolution
        try:
            self._emit("status", "Analyzing narrative evolution...")
            evolve_after_scene(
                scene_text=scene_text,
                scene_plan=scene,
                chapter_num=chapter_num,
                scene_num=scene_number,
                state_manager=self.state_manager,
                llm=self._active_planning_model(),
                compact_mode=(self._backend in {"groq", "gemini"}),
                progress_callback=lambda msg: self._log(msg, level="info"),
            )
            self._log("Narrative evolution complete", level="success")
        except Exception as e:
            self._log(
                f"Evolution engine error (non-fatal): {e}",
                level="warn",
            )

        self._emit("scene_complete", {
            "scene": scene_number,
            "words": post_edit_words,
            "text": scene_text,
        })
        self._log(
            f"Scene {scene_number} complete: {post_edit_words} words",
            level="success",
        )

        # Save WIP checkpoint
        chapter_plan = self._build_manual_chapter_plan_stub(
            chapter_num, chapter_title, session["completed_scene_plans"], character_names, pacing,
        )
        completed_wip = [{"scene": s + 1, "text": t} for s, t in enumerate(session["completed_scenes"])]
        self._save_wip(chapter_num, completed_wip, chapter_plan, session["completed_scene_plans"])

        # Save chapter file progressively so Reader can preview
        separator = "\n\n* * *\n\n"
        chapter_text = separator.join(session["completed_scenes"])
        full_chapter = self._sanitize_generated_text(
            f"# Chapter {chapter_num}: {chapter_title}\n\n{chapter_text}"
        )
        chapter_path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
        with open(chapter_path, "w", encoding="utf-8") as f:
            f.write(full_chapter)

        return {
            "status": "complete",
            "scene_number": scene_number,
            "words": post_edit_words,
            "scene_text": scene_text,
            "enhanced_summary": scene.get("summary", ""),
        }

    def delete_manual_scene(self, session: dict, index: int) -> dict:
        """
        Delete a scene from the active manual session and recalculate previous_ending.
        """
        if index < 0 or index >= len(session["completed_scenes"]):
            raise IndexError("Scene index out of range.")
        
        session["completed_scenes"].pop(index)
        if index < len(session["completed_scene_plans"]):
            session["completed_scene_plans"].pop(index)
            
        session["scene_counter"] -= 1

        # Recalculate previous_ending for context
        if session["completed_scenes"]:
            session["previous_ending"] = self._extract_ending(session["completed_scenes"][-1])
        else:
            session["previous_ending"] = self._previous_chapter_ending(session["chapter_num"]) or ""

        # Update WIP file since we removed a scene
        chapter_num = session["chapter_num"]
        chapter_title = session["chapter_title"]
        character_names = session["character_names"]
        pacing = session["pacing"]
        
        chapter_plan = self._build_manual_chapter_plan_stub(
            chapter_num, chapter_title, session["completed_scene_plans"], character_names, pacing,
        )
        completed_wip = [{"scene": s + 1, "text": t} for s, t in enumerate(session["completed_scenes"])]
        self._save_wip(chapter_num, completed_wip, chapter_plan, session["completed_scene_plans"])

        # Update the chapter text file as well
        separator = "\n\n* * *\n\n"
        chapter_text = separator.join(session["completed_scenes"])
        full_chapter = self._sanitize_generated_text(
            f"# Chapter {chapter_num}: {chapter_title}\n\n{chapter_text}"
        )
        chapter_path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
        with open(chapter_path, "w", encoding="utf-8") as f:
            f.write(full_chapter)

        self._log(f"Deleted scene {index + 1} from manual session.", level="info")
        return session

    def update_manual_scene(self, session: dict, index: int, new_text: str) -> dict:
        """
        Replace the text of a scene in the active manual session.
        Updates WIP and chapter file on disk.
        """
        if index < 0 or index >= len(session["completed_scenes"]):
            raise IndexError("Scene index out of range.")

        new_text = (new_text or "").strip()
        if not new_text:
            raise ValueError("Scene text cannot be empty.")

        session["completed_scenes"][index] = new_text

        # Recalculate previous_ending if this is the last scene
        if index == len(session["completed_scenes"]) - 1:
            session["previous_ending"] = self._extract_ending(new_text)

        # Update WIP file
        chapter_num = session["chapter_num"]
        chapter_title = session["chapter_title"]
        character_names = session["character_names"]
        pacing = session["pacing"]

        chapter_plan = self._build_manual_chapter_plan_stub(
            chapter_num, chapter_title, session["completed_scene_plans"], character_names, pacing,
        )
        completed_wip = [{"scene": s + 1, "text": t} for s, t in enumerate(session["completed_scenes"])]
        self._save_wip(chapter_num, completed_wip, chapter_plan, session["completed_scene_plans"])

        # Update the chapter text file
        separator = "\n\n* * *\n\n"
        chapter_text = separator.join(session["completed_scenes"])
        full_chapter = self._sanitize_generated_text(
            f"# Chapter {chapter_num}: {chapter_title}\n\n{chapter_text}"
        )
        chapter_path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
        with open(chapter_path, "w", encoding="utf-8") as f:
            f.write(full_chapter)

        self._log(f"Updated scene {index + 1} text in manual session ({len(new_text.split())} words).", level="info")
        return session

    def add_typed_scene(self, session: dict, text: str) -> dict:
        """
        Add a user-typed scene to the manual session without any AI generation.
        Creates a minimal scene plan stub and updates WIP + chapter file.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("Scene text cannot be empty.")

        chapter_num = session["chapter_num"]
        chapter_title = session["chapter_title"]
        character_names = session["character_names"]
        pacing = session["pacing"]
        scene_number = session["scene_counter"] + 1

        # Create a minimal scene plan for the typed scene
        words = len(text.split())
        scene_plan = {
            "scene_number": scene_number,
            "type": "custom",
            "summary": f"[User-typed scene] {text[:120]}...",
            "characters_present": list(character_names),
            "key_events": ["User-provided content"],
            "word_target": words,
            "mood": "custom",
        }

        # Update session state
        session["completed_scenes"].append(text)
        session["completed_scene_plans"].append(scene_plan)
        session["previous_ending"] = self._extract_ending(text)
        session["scene_counter"] = scene_number

        # Save WIP checkpoint
        chapter_plan = self._build_manual_chapter_plan_stub(
            chapter_num, chapter_title, session["completed_scene_plans"], character_names, pacing,
        )
        completed_wip = [{"scene": s + 1, "text": t} for s, t in enumerate(session["completed_scenes"])]
        self._save_wip(chapter_num, completed_wip, chapter_plan, session["completed_scene_plans"])

        # Save chapter file progressively
        separator = "\n\n* * *\n\n"
        chapter_text = separator.join(session["completed_scenes"])
        full_chapter = self._sanitize_generated_text(
            f"# Chapter {chapter_num}: {chapter_title}\n\n{chapter_text}"
        )
        chapter_path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
        with open(chapter_path, "w", encoding="utf-8") as f:
            f.write(full_chapter)

        self._log(
            f"Added typed scene {scene_number} to manual session ({words} words).",
            level="success",
        )
        return session

    def finish_manual_chapter(self, session: dict) -> dict:
        """
        Finalize an interactive manual chapter: persist state, index vectors, clean up WIP.
        """
        chapter_num = session["chapter_num"]
        chapter_title = session["chapter_title"]
        pacing = session["pacing"]
        character_names = session["character_names"]
        chapter_text_parts = session["completed_scenes"]
        scenes = session["completed_scene_plans"]
        trace_entries = session["trace_entries"]
        step_counter = session["step_counter"]

        if not chapter_text_parts:
            raise RuntimeError("No scenes generated. Generate at least one scene before finishing.")

        # Build the chapter plan from what was actually written
        chapter_plan = self._build_manual_chapter_plan_stub(
            chapter_num, chapter_title, scenes, character_names, pacing,
        )

        # Assemble final chapter text
        separator = "\n\n* * *\n\n"
        chapter_text = separator.join(chapter_text_parts)
        full_chapter = self._sanitize_generated_text(
            f"# Chapter {chapter_num}: {chapter_title}\n\n{chapter_text}"
        )

        # Save chapter file
        self._emit("status", "Saving chapter...")
        chapter_path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
        with open(chapter_path, "w", encoding="utf-8") as f:
            f.write(full_chapter)

        # Generate chapter summary from actual written content
        completed_scene_count = len(chapter_text_parts)
        actual_summary = " ".join(
            part.strip()[:150] + "..." for part in chapter_text_parts[:3]
        )
        plan_summary = chapter_plan.get("plot_direction", "")
        summary = f"{plan_summary} | Written content: {actual_summary[:400]}"
        title = chapter_plan.get("chapter_title", "")
        self.state_manager.add_chapter_summary(chapter_num, summary, title)
        self.state_manager.increment_scene_count(completed_scene_count)

        # ── Persist the premise step cursor ──────────────────────────────
        # Advance the cursor so the NEXT auto-generated chapter starts from
        # where this manual chapter left off.
        _manual_state = self.state_manager.load()
        _manual_steps = self._premise_steps(
            _manual_state.get("metadata", {}).get("premise", "")
        )
        if _manual_steps:
            _manual_cursor = self._allowed_premise_step_index_from_state(
                chapter_num, _manual_steps, _manual_state
            )
            self.state_manager.set_premise_step_completed(_manual_cursor)
            self._log(
                f"Premise cursor (manual) advanced to step {_manual_cursor + 1}/{len(_manual_steps)}",
                level="info",
            )

        # Add events
        for event in chapter_plan.get("key_events", []):
            self.state_manager.add_event(event, chapter_num)

        # Update character arcs
        for name, arc in chapter_plan.get("character_arcs", {}).items():
            self.state_manager.update_character(name, {"arc_progression": arc})

        # Track threads
        for thread in chapter_plan.get("new_threads_to_introduce", []):
            self.state_manager.add_unresolved_thread(thread)

        # Embed into vector store
        self._emit("status", "Embedding chapter into vector memory...")
        self._log("Embedding chapter into FAISS vector memory...")
        total_chunks = 0
        for i, part in enumerate(chapter_text_parts):
            chars = scenes[i].get("characters_present", []) if i < len(scenes) else []
            loc = scenes[i].get("location", "") if i < len(scenes) else ""
            n = self.vector_store.add_text(
                text=part, chapter=chapter_num,
                scene=i + 1, characters=chars, location=loc,
            )
            total_chunks += n
        self._log(f"Embedded {total_chunks} chunks into vector store", level="success")

        # Save logs and clean up
        self._save_trace(chapter_num, trace_entries)
        self._clear_wip(chapter_num)

        total_words = sum(len(p.split()) for p in chapter_text_parts)
        result = {
            "chapter_number": chapter_num,
            "chapter_title": chapter_title,
            "total_words": total_words,
            "scenes_count": completed_scene_count,
            "file": chapter_path,
            "status": "complete",
            "steps_used": step_counter["steps"],
            "estimated_tokens_used": session["estimated_tokens_used"],
        }

        self._emit("chapter_complete", result)
        self._log(
            f"═══ Chapter {chapter_num} finalized: \"{chapter_title}\" ═══",
            level="success",
            details={
                "total_words": total_words,
                "scenes": completed_scene_count,
                "file": chapter_path,
            },
        )
        return result

    @staticmethod
    def _build_manual_chapter_plan_stub(
        chapter_num: int,
        chapter_title: str,
        scene_plans: list,
        character_names: list,
        pacing: str,
    ) -> dict:
        """Build a chapter plan dict from the accumulated manual scene plans."""
        chapter_key_events = [
            sp.get("summary", "") for sp in scene_plans if sp.get("summary")
        ]
        return {
            "chapter_number": chapter_num,
            "chapter_title": chapter_title,
            "plot_direction": "Manual chapter plan built interactively from user scene briefs.",
            "character_arcs": {
                name: f"{name} progresses through user-directed chapter beats."
                for name in character_names
            },
            "key_events": chapter_key_events,
            "constraints": [
                "Manual mode: preserve user-provided scene intent and sequence.",
            ],
            "tone": pacing,
            "estimated_scenes": len(scene_plans),
            "unresolved_threads_to_address": [],
            "new_threads_to_introduce": [],
        }

    @classmethod
    def _wip_has_corrupt_scenes(cls, wip: dict) -> bool:
        if not isinstance(wip, dict):
            return False
        for scene in wip.get("completed_scenes", []):
            if cls._looks_like_meta_output(scene.get("text", "")):
                return True
        return False

    # ─── Project Setup ────────────────────────────────────────────────

    def create_project(
        self, title, genre, premise, characters=None, themes=None, setting=""
    ) -> dict:
        state = self.state_manager.initialize(
            title=title, genre=genre, premise=premise,
            characters=characters or {}, themes=themes or [], setting=setting,
        )
        self._emit("project_created", {"title": title, "genre": genre})
        logger.info(f"Project '{title}' created at {self.project_dir}")
        return state

    def load_project(self) -> dict:
        state = self.state_manager.load()
        trait_norm = self.state_manager.normalize_character_traits()
        if trait_norm.get("traits_removed", 0):
            logger.info(
                "Normalized character traits on load: %s traits removed across %s characters",
                trait_norm.get("traits_removed", 0),
                trait_norm.get("characters_touched", 0),
            )
            state = self.state_manager.load()
        # Set genre on writer and editor so they know routing and tone
        genre = state.get("metadata", {}).get("genre", "")
        self.writer.set_genre(genre)
        self.editor.set_genre(genre)
        return state

    def get_project_info(self) -> dict:
        state = self.state_manager.state
        stats = self.vector_store.get_stats()
        chapters = self._list_chapters()
        return {
            "name": self.project_name,
            "title": state["metadata"]["title"],
            "genre": state["metadata"]["genre"],
            "current_chapter": state["metadata"]["current_chapter"],
            "total_scenes": state["metadata"]["total_scenes_written"],
            "chapters_written": len(chapters),
            "vector_chunks": stats["total_vectors"],
            "characters": list(state["characters"].keys()),
        }

    # ─── Chapter Generation ──────────────────────────────────────────

    def generate_chapter(self, pacing: str = "moderate", manual_spec: Optional[dict] = None) -> dict:
        """Generate the next chapter through the full pipeline."""
        # Rotate API key at the start of each chapter
        from models.groq_model import GroqKeyManager
        GroqKeyManager.rotate()
        self._cancelled = False
        self.state_manager.normalize_character_traits()
        state = self.state_manager.load()

        # ── Self-healing state sync ───────────────────────────────────────
        # If chapter files exist on disk but state.current_chapter is behind
        # (e.g. after a reset or mid-chapter crash), fast-forward the state
        # pointer so we never regenerate already-written content.
        state_chapter = int(state["metadata"].get("current_chapter", 0))
        disk_chapter = state_chapter
        while True:
            candidate = disk_chapter + 1
            candidate_path = os.path.join(self.chapters_dir, f"chapter_{candidate:03d}.md")
            if os.path.exists(candidate_path):
                disk_chapter = candidate
            else:
                break
        if disk_chapter > state_chapter:
            self._log(
                f"State pointer was behind disk: advancing current_chapter "
                f"from {state_chapter} → {disk_chapter}",
                level="warn",
            )
            # Patch the in-memory state; a full prune_after_chapter is not
            # needed here — we just need the chapter counter to be correct.
            self.state_manager._transition(
                "auto_advance_chapter_pointer",
                {"from": state_chapter, "to": disk_chapter},
                lambda s: {**s, "metadata": {**s["metadata"], "current_chapter": disk_chapter}},
            )
            state = self.state_manager.load()
        # ─────────────────────────────────────────────────────────────────

        chapter_num = state["metadata"]["current_chapter"] + 1
        chapter_start_time = time.time()

        self._emit("chapter_start", {"chapter": chapter_num})
        self._log(f"═══ Starting Chapter {chapter_num} generation ═══", level="header",
                  details={"pacing": pacing, "project": self.project_name})
        log_entries = []
        trace_entries = []
        step_counter = {"steps": 0}
        estimated_tokens_used = 0
        graph_node = "start"
        manual_mode = bool(manual_spec)
        continuity_recovery_applied = False


        try:
            # Step 0: Re-anchor if needed
            if (chapter_num > 1 and
                    (chapter_num - 1) % config.REANCHOR_EVERY_N_CHAPTERS == 0):
                self._emit("status", "Re-anchoring narrative state...")
                self._log("Re-anchoring narrative state to prevent drift...", level="info")
                t0 = time.time()
                self.architect.reanchor()
                self._log(f"Re-anchoring complete in {time.time() - t0:.1f}s", level="success")
                log_entries.append({"step": "reanchor", "status": "done"})

            # Step 1: Retrieve context for planning
            graph_node = PIPELINE_GRAPH["start"][0]
            self._emit("agent_active", {"agent": "Retriever", "step": "context"})
            self._log("Retriever: assembling story context for planning...",
                      details={"vector_chunks": self.vector_store.get_stats()["total_vectors"]})
            previous_chapter_ending = self._previous_chapter_ending(chapter_num)
            plan_context = self._append_continuity_anchor(
                self.state_manager.get_context_window(include_premise=False),
                previous_chapter_ending,
            )
            plan_context = self._append_character_introduction_anchor(
                plan_context,
                chapter_num,
                state,
            )
            plan_context = self._append_premise_step_anchor(
                plan_context,
                chapter_num,
                state,
            )
            plan_context = self._append_used_titles_anchor(
                plan_context,
                state,
            )

            # Inject tension curve and motif summaries for architect awareness
            tension_summary = TensionTracker.get_tension_summary(state)
            if tension_summary:
                plan_context = f"{plan_context.rstrip()}\n\n{tension_summary}\n"
            motif_summary = MotifTracker.get_motif_summary(state)
            if motif_summary:
                plan_context = f"{plan_context.rstrip()}\n\n{motif_summary}\n"

            self._log(f"Context assembled: {len(plan_context)} chars",
                      details={"characters": list(state.get('characters', {}).keys())})

            # Step 2: Architect plans chapter
            self._emit("agent_active", {"agent": "Story Architect", "step": "planning"})
            self._log(f"Story Architect: planning chapter {chapter_num}...",
                      details={"model": self._active_model_id, "backend": self._active_backend})
            if self._cancelled:
                return self._cancel_result(chapter_num)

            graph_node = PIPELINE_GRAPH[graph_node][0]
            if manual_mode:
                chapter_plan = manual_spec["chapter_plan"]
                self._log("Manual chapter plan accepted", level="success")
                arch_time = 0.0
            else:
                steps = self._premise_steps(state.get("metadata", {}).get("premise", ""))
                architect_input = {
                    "chapter_num": chapter_num,
                    "context": plan_context,
                    "pacing": pacing,
                    "premise_steps": steps,
                    "current_step": self._allowed_premise_step_index_from_state(chapter_num, steps, state) + 1,
                    "used_titles": [s.get("chapter_title") for s in state.get("plot", {}).get("chapter_summaries", []) if s.get("chapter_title")],
                }
                architect_result = self._run_agent_step(
                    chapter_num=chapter_num,
                    scene_num=0,
                    trace_entries=trace_entries,
                    step_counter=step_counter,
                    agent_name="planner",
                    call_fn=lambda: self.architect.run(architect_input),
                    agent_input={
                        "chapter_num": chapter_num,
                        "context_chars": len(plan_context),
                        "pacing": pacing,
                    },
                )
                chapter_plan = architect_result["output"]
                try:
                    self._validate_plan_premise_order(chapter_plan, chapter_num, state)
                    self._validate_plan_character_introductions(chapter_plan, chapter_num, state)
                except RuntimeError as e:
                    original_error = str(e)
                    continuity_recovery_applied = True
                    chapter_plan = self._repair_chapter_plan_to_premise_cursor(
                        chapter_plan,
                        chapter_num,
                        pacing,
                        state,
                    )
                    chapter_plan = self._inject_missing_character_introductions(
                        chapter_plan,
                        chapter_num,
                        state,
                    )
                    try:
                        self._validate_plan_premise_order(chapter_plan, chapter_num, state)
                        self._validate_plan_character_introductions(chapter_plan, chapter_num, state)
                    except RuntimeError as re_err:
                        # The repaired plan itself still trips the validator — this can
                        # happen when the window steps share tokens with later steps.
                        # Trust the repair and continue; the anchor prompt will guide
                        # the writer away from future content.
                        self._log(
                            "Repaired chapter plan has residual premise markers; continuing with best repair",
                            level="warn",
                            details={"reason": str(re_err)},
                        )
                    self._log(
                        "Architect continuity issue auto-repaired",
                        level="warn",
                        details={
                            "reason": original_error,
                            "events": chapter_plan.get("key_events", []),
                        },
                    )
                arch_time = trace_entries[-1]["latency_ms"] / 1000.0
            self._emit("chapter_planned", {"plan": chapter_plan})
            self._log(f"Chapter planned: \"{chapter_plan.get('chapter_title', '?')}\"",
                      level="success",
                      details={
                          "time": f"{arch_time:.1f}s",
                          "tone": chapter_plan.get("tone"),
                          "events": chapter_plan.get("key_events", []),
                          "arcs": chapter_plan.get("character_arcs", {}),
                      })
            log_entries.append({
                "step": "architect", "status": "done",
                "title": chapter_plan.get("chapter_title"),
            })

            # Step 3: Planner decomposes into scenes
            self._emit("agent_active", {"agent": "Scene Planner", "step": "decomposing"})
            self._log("Scene Planner: decomposing chapter into scenes...",
                      details={"model": self._active_model_id, "backend": self._active_backend})
            if self._cancelled:
                return self._cancel_result(chapter_num)

            graph_node = PIPELINE_GRAPH[graph_node][0]
            if manual_mode:
                scenes = list(manual_spec.get("scenes", []))
                scene_plan = {"scenes": scenes}
                plan_time = 0.0
                self._log("Manual scene sequence accepted", level="success")
            else:
                seen_names = set()
                char_names = []
                for name in state.get("characters", {}).keys():
                    key = name.lower()
                    if key not in seen_names:
                        seen_names.add(key)
                        char_names.append(name)
                planner_input = {
                    "chapter_plan": chapter_plan,
                    "context": plan_context,
                    "character_names": char_names,
                }
                planner_result = self._run_agent_step(
                    chapter_num=chapter_num,
                    scene_num=0,
                    trace_entries=trace_entries,
                    step_counter=step_counter,
                    agent_name="scene_planner",
                    call_fn=lambda: self.planner.run(planner_input),
                    agent_input={
                        "chapter_title": chapter_plan.get("chapter_title", ""),
                        "characters": char_names,
                    },
                )
                scene_plan = planner_result["output"]
                plan_time = trace_entries[-1]["latency_ms"] / 1000.0
                scenes = scene_plan.get("scenes", [])
                try:
                    self._validate_scene_premise_order(scenes, chapter_num, state)
                    self._validate_scene_character_introductions(scenes, chapter_num, state)
                except RuntimeError as e:
                    original_error = str(e)
                    continuity_recovery_applied = True
                    scenes = self._repair_scene_plan_to_chapter_plan(chapter_plan, chapter_num, state)
                    scene_plan["scenes"] = scenes
                    try:
                        self._validate_scene_premise_order(scenes, chapter_num, state)
                        self._validate_scene_character_introductions(scenes, chapter_num, state)
                    except RuntimeError as re_err:
                        # Repair is best-effort; if residual markers remain, log and
                        # continue — the scene writer will be anchored by the prompt.
                        self._log(
                            "Repaired scene plan has residual premise markers; continuing with best repair",
                            level="warn",
                            details={"reason": str(re_err)},
                        )
                    self._log(
                        "Scene planner continuity issue auto-repaired",
                        level="warn",
                        details={
                            "reason": original_error,
                            "scenes": [scene.get("summary", "") for scene in scenes],
                        },
                    )
            self._emit("scenes_planned", {
                "count": len(scenes),
                "types": [s["type"] for s in scenes],
            })
            self._log(f"Scenes planned: {len(scenes)} scenes in {plan_time:.1f}s",
                      level="success",
                      details={
                          "types": [s["type"] for s in scenes],
                          "word_targets": [s.get("word_target", 600) for s in scenes],
                      })
            log_entries.append({
                "step": "planner", "status": "done", "scenes": len(scenes),
            })

            # Step 4: Generate each scene
            chapter_text_parts = []
            previous_ending = ""

            # For chapter 2+, carry over the ending from the previous chapter
            if previous_chapter_ending:
                previous_ending = previous_chapter_ending
                self._log(f"Loaded ending from chapter {chapter_num - 1} for continuity")

            # Check for WIP checkpoint (resume after crash)
            if continuity_recovery_applied:
                self._clear_pending_chapter_artifacts(chapter_num)
            wip = None if continuity_recovery_applied else self._load_wip(chapter_num)
            start_scene_idx = 0
            if wip and self._wip_has_corrupt_scenes(wip):
                self._log(
                    f"Ignoring WIP checkpoint for chapter {chapter_num}: corrupt meta-output scene detected",
                    level="warn",
                )
                wip = None
            if wip and wip.get("completed_scenes"):
                start_scene_idx = len(wip["completed_scenes"])
                chapter_text_parts = [s["text"] for s in wip["completed_scenes"]]
                if chapter_text_parts:
                    previous_ending = self._extract_ending(chapter_text_parts[-1])
                self._log(f"Resuming from WIP checkpoint: {start_scene_idx}/{len(scenes)} scenes done",
                          level="warn")
                self._emit("status", f"Resuming from scene {start_scene_idx + 1}")

            graph_node = PIPELINE_GRAPH[graph_node][0]
            for i, scene in enumerate(scenes):
                if i < start_scene_idx:
                    continue  # Skip already-completed scenes
                if self._cancelled:
                    return self._cancel_result(chapter_num)

                scene_num = scene["scene_number"]
                scene_start_time = time.time()
                self._emit("scene_start", {
                    "scene": scene_num, "total": len(scenes),
                    "type": scene["type"],
                })
                self._log(f"─── Scene {scene_num}/{len(scenes)} ({scene['type']}) ───",
                          level="header",
                          details={
                              "summary": scene.get("summary", "")[:120],
                              "characters": scene.get("characters_present", []),
                              "location": scene.get("location", ""),
                              "word_target": scene.get("word_target", 600),
                          })

                # 4a: Inject continuity metadata into scene plan
                # prev_location enables the writer prompt to detect location changes
                # and mandate a transition paragraph automatically.
                # Use the actual plan location of the previous scene if available, otherwise fall back to parsing text.
                if i > 0 and i - 1 < len(scenes):
                    scene["prev_location"] = scenes[i - 1].get("location", "")
                else:
                    scene["prev_location"] = self._extract_scene_location(previous_ending)

                # narrative_bridge explains WHY this scene follows the previous one.
                # Pull it from the planner-generated field or build a minimal default.
                if not scene.get("narrative_bridge"):
                    prev_summary = ""
                    if i > 0:
                        prev_summary = scenes[i - 1].get("summary", "")
                    if prev_summary:
                        scene["narrative_bridge"] = (
                            f"This scene follows directly after: {prev_summary[:120]}"
                        )

                # Inject character lifecycle profiles so the writer knows ages
                # (e.g. Leo is a baby born in Act 4, not an adult).
                scene["character_profiles"] = self._build_character_profiles(
                    scene, chapter_num
                )

                # 4a: Retrieve context for this scene
                self._emit("agent_active", {
                    "agent": "Retriever", "step": f"scene {scene_num}",
                })
                self._log(f"Retriever: fetching context for scene {scene_num}...")
                scene_context = self.retriever.retrieve_context(
                    scene_plan=scene,
                    chapter_num=chapter_num,
                    previous_ending=previous_ending,
                    top_k=2 if self._backend in {"groq", "gemini"} else None,
                    max_chars=2200 if self._backend in {"groq", "gemini"} else None,
                )

                # ── Enrich writer context with completed-scene recap ─────────────
                # The retriever returns generic vector-store context. We append a
                # summary of scenes already written in THIS chapter so the writer
                # can avoid repeating them.
                if chapter_text_parts:
                    scene_recap_parts = []
                    for idx, st in enumerate(chapter_text_parts):
                        plan_label = ""
                        plan_loc = ""
                        if idx < len(scenes):
                            plan_label = scenes[idx].get("summary", "")[:80]
                            plan_loc = scenes[idx].get("location", "")
                        loc_str = f" [at {plan_loc}]" if plan_loc else ""
                        ending_snip = self._extract_ending(st, max_chars=150)
                        scene_recap_parts.append(
                            f"  Scene {idx + 1}{loc_str} ({plan_label}): ...{ending_snip}"
                        )
                    scene_context = (
                        f"{scene_context.rstrip()}\n\n"
                        "=== SCENES ALREADY WRITTEN IN THIS CHAPTER ===\n"
                        + "\n".join(scene_recap_parts)
                        + "\n=== END ===\n"
                        "CRITICAL: Do NOT repeat, recap, or re-describe any of the events above. "
                        "Continue the story from where Scene " + str(len(chapter_text_parts)) + " ended."
                    )

                self._log(f"Context retrieved: {len(scene_context)} chars",
                          details={"top_k": config.TOP_K_RETRIEVAL})

                # Inject character voice guidance into scene context
                voice_guidance = self.voice.build_voice_guidance(
                    scene.get("characters_present", []),
                    state.get("characters", {}),
                )
                if voice_guidance:
                    scene_context = f"{scene_context.rstrip()}\n\n{voice_guidance}\n"

                # 4b-4f: Explicit scene execution graph
                scene_result = self._run_scene_graph(
                    chapter_num=chapter_num,
                    scene=scene,
                    scene_context=scene_context,
                    previous_ending=previous_ending,
                    trace_entries=trace_entries,
                    step_counter=step_counter,
                )
                if scene_result.get("status") == "cancelled":
                    return self._cancel_result(chapter_num)
                if self._cancelled:
                    return self._cancel_result(chapter_num)

                scene_text = self._sanitize_generated_text(scene_result["scene_text"])
                scene_text = self._remove_repeated_reference_sentences(
                    scene_text,
                    previous_ending,
                )
                retries = max(0, scene_result["iterations"] - 1)
                scene_text = self._resilient_scene_generation(
                    scene_text=scene_text,
                    scene=scene,
                    chapter_num=chapter_num,
                    scene_context=scene_context,
                    previous_ending=previous_ending,
                    trace_entries=trace_entries,
                    step_counter=step_counter,
                )
                post_edit_words = len(scene_text.split())
                estimated_tokens_used += int(post_edit_words * 1.35)

                chapter_text_parts.append(scene_text)
                previous_ending = self._extract_ending(scene_text)
                self._ensure_scene_characters_known(scene)

                # Post-scene pacing analysis (deterministic, no LLM)
                pacing_report = self.pacing.analyze(scene_text, scene)
                pacing_issues = pacing_report.get("issues", [])
                if pacing_issues:
                    self._log(
                        f"Pacing analysis: {pacing_report['overall_pace']} pace, "
                        f"{len(pacing_issues)} issue(s)",
                        level="info",
                        details={
                            "issues": [i.get("type", "") for i in pacing_issues],
                            "word_count": pacing_report.get("word_count", 0),
                        },
                    )
                else:
                    self._log(
                        f"Pacing analysis: {pacing_report['overall_pace']} pace ✓",
                        level="success",
                    )

                # Post-scene narrative evolution
                try:
                    self._emit("status", "Analyzing narrative evolution...")
                    evolve_after_scene(
                        scene_text=scene_text,
                        scene_plan=scene,
                        chapter_num=chapter_num,
                        scene_num=scene_num,
                        state_manager=self.state_manager,
                        llm=self._active_planning_model(),
                        compact_mode=(self._backend in {"groq", "gemini"}),
                        progress_callback=lambda msg: self._log(msg, level="info"),
                    )
                    self._log("Narrative evolution complete", level="success")
                except Exception as e:
                    self._log(
                        f"Evolution engine error (non-fatal): {e}",
                        level="warn",
                    )

                scene_elapsed = time.time() - scene_start_time
                self._emit("scene_complete", {
                    "scene": scene_num,
                    "words": post_edit_words,
                    "text": scene_text,
                })
                self._log(f"Scene {scene_num} complete: {post_edit_words} words, {scene_elapsed:.1f}s total",
                          level="success")

                log_entries.append({
                    "step": f"scene_{scene_num}", "status": "done",
                    "words": post_edit_words,
                    "provider": self.writer.last_provider,
                    "consistency_retries": retries,
                    "time_seconds": round(scene_elapsed, 1),
                })

                # Save WIP checkpoint after each scene
                completed = [{"scene": s + 1, "text": t} for s, t in enumerate(chapter_text_parts)]
                self._save_wip(chapter_num, completed, chapter_plan, scenes)

                if estimated_tokens_used > config.MAX_TOKEN_BUDGET:
                    self._emit("status", "Token budget reached; chapter saved as WIP.")
                    self._log(
                        f"Chapter paused after scene {scene_num}: token budget exceeded",
                        level="warn",
                        details={
                            "estimated_tokens": estimated_tokens_used,
                            "budget": config.MAX_TOKEN_BUDGET,
                            "completed_scenes": len(chapter_text_parts),
                            "planned_scenes": len(scenes),
                        },
                    )
                    raise RuntimeError(
                        f"Chapter {chapter_num} paused after scene {scene_num}: "
                        "token budget reached before all scenes completed. "
                        "WIP checkpoint preserved; retry later to resume."
                    )

                # Brief pause between scenes to avoid Groq rate limits
                if i < len(scenes) - 1:
                    time.sleep(5)

                # Save chapter file progressively so Reader shows it building up
                chapter_title = chapter_plan.get("chapter_title", f"Chapter {chapter_num}")
                separator = "\n\n* * *\n\n"
                chapter_text = separator.join(chapter_text_parts)
                full_chapter = self._sanitize_generated_text(
                    f"# Chapter {chapter_num}: {chapter_title}\n\n{chapter_text}"
                )
                chapter_path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
                with open(chapter_path, "w", encoding="utf-8") as f:
                    f.write(full_chapter)

            # Step 5: Assemble chapter
            graph_node = PIPELINE_GRAPH[graph_node][0]
            if not chapter_text_parts:
                raise RuntimeError(
                    "Chapter generation terminated before any scene text was produced."
                )
            if len(chapter_text_parts) < len(scenes):
                self._save_wip(
                    chapter_num,
                    [{"scene": s + 1, "text": t} for s, t in enumerate(chapter_text_parts)],
                    chapter_plan,
                    scenes,
                )
                raise RuntimeError(
                    f"Chapter {chapter_num} incomplete: generated "
                    f"{len(chapter_text_parts)}/{len(scenes)} scenes. "
                    "WIP checkpoint preserved; retry later to resume."
                )
            chapter_title = chapter_plan.get("chapter_title", f"Chapter {chapter_num}")
            separator = "\n\n* * *\n\n"
            chapter_text = separator.join(chapter_text_parts)
            full_chapter = self._sanitize_generated_text(
                f"# Chapter {chapter_num}: {chapter_title}\n\n{chapter_text}"
            )

            # Step 6: Store
            graph_node = PIPELINE_GRAPH[graph_node][0]
            self._emit("status", "Saving chapter...")
            chapter_path = os.path.join(
                self.chapters_dir, f"chapter_{chapter_num:03d}.md"
            )
            with open(chapter_path, "w", encoding="utf-8") as f:
                f.write(full_chapter)

            # Generate chapter summary from what was ACTUALLY written (not just the plan)
            # Use the first 500 chars of each scene to build a real summary
            completed_scene_count = len(chapter_text_parts)
            actual_summary = " ".join(
                part.strip()[:150] + "..." for part in chapter_text_parts[:3]
            )
            plan_summary = chapter_plan.get("plot_direction", "")
            summary = f"{plan_summary} | Written content: {actual_summary[:400]}"
            title = chapter_plan.get("chapter_title", "")
            self.state_manager.add_chapter_summary(chapter_num, summary, title)
            self.state_manager.increment_scene_count(completed_scene_count)

            # ── Persist the premise step cursor ──────────────────────────────
            # Record which premise step index was covered last so the NEXT
            # chapter's planning starts exactly from the correct position.
            _steps_for_cursor = self._premise_steps(
                state.get("metadata", {}).get("premise", "")
            )
            if _steps_for_cursor and not manual_mode:
                _cursor = self._allowed_premise_step_index_from_state(
                    chapter_num, _steps_for_cursor, state
                )
                self.state_manager.set_premise_step_completed(_cursor)
                self._log(
                    f"Premise cursor advanced to step {_cursor + 1}/{len(_steps_for_cursor)}",
                    level="info",
                    details={"step": _steps_for_cursor[_cursor][:80] if _cursor < len(_steps_for_cursor) else "premise complete"},
                )

            # Add events
            for event in chapter_plan.get("key_events", []):
                self.state_manager.add_event(event, chapter_num)

            # Update character arcs
            for name, arc in chapter_plan.get("character_arcs", {}).items():
                self.state_manager.update_character(name, {
                    "arc_progression": arc,
                })

            # Track threads
            for thread in chapter_plan.get("new_threads_to_introduce", []):
                self.state_manager.add_unresolved_thread(thread)
            for thread in chapter_plan.get("unresolved_threads_to_address", []):
                pass  # Will be resolved when actually paid off

            # Embed into vector store
            self._emit("status", "Embedding chapter into vector memory...")
            self._log("Embedding chapter into FAISS vector memory...",
                      details={"model": config.EMBEDDING_MODEL, "chunk_size": config.CHUNK_SIZE})
            total_chunks = 0
            for i, part in enumerate(chapter_text_parts):
                chars = scenes[i].get("characters_present", []) if i < len(scenes) else []
                loc = scenes[i].get("location", "") if i < len(scenes) else ""
                n = self.vector_store.add_text(
                    text=part, chapter=chapter_num,
                    scene=i + 1, characters=chars, location=loc,
                )
                total_chunks += n
            self._log(f"Embedded {total_chunks} chunks into vector store", level="success",
                      details={"total_vectors": self.vector_store.get_stats()["total_vectors"]})

            # Save log
            self._save_log(chapter_num, log_entries)
            self._save_trace(chapter_num, trace_entries)
            self._clear_wip(chapter_num)  # Remove WIP on success

            total_words = sum(len(p.split()) for p in chapter_text_parts)
            chapter_elapsed = time.time() - chapter_start_time
            result = {
                "chapter_number": chapter_num,
                "chapter_title": chapter_title,
                "total_words": total_words,
                "scenes_count": completed_scene_count,
                "file": chapter_path,
                "status": "complete",
                "steps_used": step_counter["steps"],
                "estimated_tokens_used": estimated_tokens_used,
            }

            self._emit("chapter_complete", result)
            self._log(f"═══ Chapter {chapter_num} complete: \"{chapter_title}\" ═══",
                      level="success",
                      details={
                          "total_words": total_words,
                          "scenes": completed_scene_count,
                          "time": f"{chapter_elapsed:.1f}s",
                          "file": chapter_path,
                          "steps_used": step_counter["steps"],
                          "estimated_tokens": estimated_tokens_used,
                      })
            logger.info(
                f"Chapter {chapter_num} complete: '{chapter_title}' "
                f"({total_words} words, {completed_scene_count} scenes)"
            )
            return result

        except Exception as e:
            logger.error(f"Pipeline failed at chapter {chapter_num}: {e}", exc_info=True)
            if trace_entries:
                self._save_trace(chapter_num, trace_entries)
            self._emit("error", {"chapter": chapter_num, "error": str(e)})
            raise

    def cancel(self):
        """Request pipeline cancellation."""
        self._cancelled = True

    def cancel_scene(self):
        """Request cancellation of only the current scene."""
        self._scene_cancelled = True

    def _cancel_result(self, chapter_num):
        return {"chapter_number": chapter_num, "status": "cancelled"}

    # ─── Utilities ────────────────────────────────────────────────────

    def _list_chapters(self) -> list:
        if not os.path.exists(self.chapters_dir):
            return []
        completed_chapter = int(
            self.state_manager.state.get("metadata", {}).get("current_chapter", 0)
        )
        files = sorted(
            f for f in os.listdir(self.chapters_dir)
            if _CHAPTER_FILE_RE.match(f)
            and int(_CHAPTER_FILE_RE.match(f).group(1)) <= completed_chapter
        )
        return files

    def read_chapter(self, chapter_num: int) -> Optional[str]:
        # Try reading completed chapter first
        path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        
        # If not completed, try reading WIP
        wip = self._load_wip(chapter_num)
        if wip and wip.get("completed_scenes"):
            title = wip.get("chapter_plan", {}).get("chapter_title", f"Chapter {chapter_num}")
            text_parts = [f"# {title}\n"]
            for s in wip["completed_scenes"]:
                text_parts.append(s.get("text", ""))
            return "\n\n".join(text_parts)
            
        return None

    def update_chapter_content(self, chapter_num: int, content: str) -> dict:
        """
        Overwrite the text of a completed chapter file on disk.
        Returns a summary with word count.
        """
        content = (content or "").strip()
        if not content:
            raise ValueError("Chapter content cannot be empty.")

        path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chapter {chapter_num} does not exist.")

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        words = len(content.split())
        self._log(
            f"Chapter {chapter_num} updated via editor ({words} words).",
            level="info",
        )
        return {"chapter": chapter_num, "words": words}

    def delete_chapters_from(self, from_chapter: int) -> dict:
        """
        Delete chapter files from `from_chapter` onward and synchronize state, WIP, logs, and vectors.
        Returns a summary payload.
        """
        from_chapter = int(from_chapter)
        if from_chapter < 1:
            raise ValueError("from_chapter must be >= 1")

        existing_numbers = self._list_chapter_numbers_on_disk()
        to_delete = [n for n in existing_numbers if n >= from_chapter]
        state_current_chapter = int(
            self.state_manager.state.get("metadata", {}).get("current_chapter", 0)
        )
        if not to_delete:
            # Files may already be gone while state still points past the requested
            # chapter. Still prune state/memory so deleted chapters do not linger in
            # characters, summaries, events, or vectors.
            if state_current_chapter >= from_chapter:
                remaining_numbers = [n for n in existing_numbers if n < from_chapter]
                current_chapter = max(remaining_numbers) if remaining_numbers else from_chapter - 1
                total_scenes = self._count_total_scenes_from_files(remaining_numbers)
                prune_summary = self.state_manager.prune_after_chapter(
                    max_chapter=current_chapter,
                    total_scenes_written=total_scenes,
                )
                deleted_vector_chunks = self.vector_store.prune_chapters(current_chapter)
                self._log(
                    "Synchronized stale state after chapter deletion",
                    level="warn",
                    details={
                        "from_chapter": from_chapter,
                        "current_chapter": current_chapter,
                        **prune_summary,
                    },
                )
                return {
                    "status": "ok",
                    "deleted_chapters": [],
                    "remaining_chapter_count": len(remaining_numbers),
                    "current_chapter": current_chapter,
                    "total_scenes_written": total_scenes,
                    "deleted_wip": 0,
                    "deleted_logs": 0,
                    "deleted_vector_chunks": deleted_vector_chunks,
                    **prune_summary,
                }
            return {
                "status": "ok",
                "deleted_chapters": [],
                "remaining_chapter_count": len(existing_numbers),
                "current_chapter": self.state_manager.get_current_chapter(),
                "total_scenes_written": self.state_manager.state.get("metadata", {}).get("total_scenes_written", 0),
                "deleted_wip": 0,
                "deleted_logs": 0,
                "deleted_vector_chunks": 0,
            }

        deleted_logs = 0
        deleted_wip = 0

        for chapter_num in to_delete:
            chapter_path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
            if os.path.exists(chapter_path):
                os.remove(chapter_path)

            for log_name in (
                f"chapter_{chapter_num:03d}.json",
                f"chapter_{chapter_num:03d}_trace.jsonl",
            ):
                log_path = os.path.join(self.logs_dir, log_name)
                if os.path.exists(log_path):
                    os.remove(log_path)
                    deleted_logs += 1

            wip_path = self._wip_path(chapter_num)
            if os.path.exists(wip_path):
                os.remove(wip_path)
                deleted_wip += 1

        # Extra safety: remove any stale WIP file >= from_chapter, even if chapter file was missing.
        if os.path.exists(self.chapters_dir):
            wip_re = re.compile(r"^\.wip_chapter_(\d{3,})\.json$")
            for fname in os.listdir(self.chapters_dir):
                match = wip_re.match(fname)
                if not match:
                    continue
                num = int(match.group(1))
                if num >= from_chapter:
                    os.remove(os.path.join(self.chapters_dir, fname))
                    deleted_wip += 1

        remaining_numbers = self._list_chapter_numbers_on_disk()
        current_chapter = max(remaining_numbers) if remaining_numbers else 0
        total_scenes = self._count_total_scenes_from_files(remaining_numbers)

        prune_summary = self.state_manager.prune_after_chapter(
            max_chapter=current_chapter,
            total_scenes_written=total_scenes,
        )

        deleted_vector_chunks = self.vector_store.prune_chapters(current_chapter)
        self._emit(
            "status",
            f"Deleted chapters {from_chapter}+ and synced state/memory.",
        )
        self._log(
            "Deleted chapters and synchronized state",
            level="warn",
            details={
                "from_chapter": from_chapter,
                "deleted_chapters": to_delete,
                "current_chapter": current_chapter,
                "total_scenes_written": total_scenes,
                "deleted_logs": deleted_logs,
                "deleted_wip": deleted_wip,
                "deleted_vector_chunks": deleted_vector_chunks,
                **prune_summary,
            },
        )

        return {
            "status": "ok",
            "deleted_chapters": to_delete,
            "remaining_chapter_count": len(remaining_numbers),
            "current_chapter": current_chapter,
            "total_scenes_written": total_scenes,
            "deleted_wip": deleted_wip,
            "deleted_logs": deleted_logs,
            "deleted_vector_chunks": deleted_vector_chunks,
            **prune_summary,
        }

    def _save_log(self, chapter_num: int, entries: list):
        log_path = os.path.join(
            self.logs_dir, f"chapter_{chapter_num:03d}.json"
        )
        log_data = {
            "chapter": chapter_num,
            "timestamp": datetime.now().isoformat(),
            "entries": entries,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

    def _list_chapter_numbers_on_disk(self) -> list[int]:
        if not os.path.exists(self.chapters_dir):
            return []
        nums = []
        for fname in os.listdir(self.chapters_dir):
            match = _CHAPTER_FILE_RE.match(fname)
            if match:
                nums.append(int(match.group(1)))
        return sorted(nums)

    def _count_total_scenes_from_files(self, chapter_numbers: list[int]) -> int:
        total = 0
        for num in chapter_numbers:
            path = os.path.join(self.chapters_dir, f"chapter_{num:03d}.md")
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
            except OSError:
                continue
            if not content:
                continue

            body = content
            first_blank = body.find("\n\n")
            if body.startswith("# Chapter ") and first_blank != -1:
                body = body[first_blank + 2:].strip()
            scenes = [chunk.strip() for chunk in body.split("\n\n* * *\n\n") if chunk.strip()]
            total += len(scenes) if scenes else 1
        return total

    def get_state(self) -> dict:
        return self.state_manager.state

    def get_state_json(self) -> str:
        return self.state_manager.to_json()

    # ─── Scene Checkpointing (WIP) ────────────────────────────────────

    def _wip_path(self, chapter_num: int) -> str:
        return os.path.join(self.chapters_dir, f".wip_chapter_{chapter_num:03d}.json")

    def _save_wip(self, chapter_num: int, completed_scenes: list, chapter_plan: dict, scene_plans: list):
        """Save work-in-progress after each completed scene.

        Uses an atomic write (temp file + rename) identical to
        ``StateManager._save_locked`` so a crash mid-write never leaves a
        corrupt WIP checkpoint on disk.
        """
        wip = {
            "chapter_num": chapter_num,
            "chapter_plan": chapter_plan,
            "scene_plans": scene_plans,
            "completed_scenes": completed_scenes,
            "timestamp": datetime.now().isoformat(),
        }
        wip_path = self._wip_path(chapter_num)
        tmp_path = wip_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(wip, f, indent=2, ensure_ascii=False)
            shutil.move(tmp_path, wip_path)
        except Exception:
            # Clean up partial temp file if something went wrong
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    def _load_wip(self, chapter_num: int) -> Optional[dict]:
        """Load a WIP checkpoint if it exists.

        Returns ``None`` if the file is missing OR if it is unreadable /
        contains corrupt JSON — so the caller can always start a fresh
        chapter without crashing the pipeline.
        """
        path = self._wip_path(chapter_num)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            self._log(
                f"WIP checkpoint for chapter {chapter_num} is corrupt and will be ignored: {exc}",
                level="warn",
                details={"path": path},
            )
            # Rename rather than delete so the user can inspect the file
            corrupt_path = path + ".corrupt"
            try:
                shutil.move(path, corrupt_path)
            except OSError:
                pass
            return None

    def _clear_wip(self, chapter_num: int):
        """Remove WIP file after successful chapter completion."""
        path = self._wip_path(chapter_num)
        if os.path.exists(path):
            os.remove(path)

    # ─── Sentence-Boundary Ending ─────────────────────────────────────

    @staticmethod
    def _extract_ending(text: str, max_chars: int = 420) -> str:
        """Extract a short complete-sentence tail for continuity prompts.

        Increased to 4 sentences / 420 chars so the next writer call
        has enough context about where the story is, not just the last line.
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return cleaned[-max_chars:]

        tail = " ".join(sentences[-4:]).strip()
        if len(tail) <= max_chars:
            return tail
        return tail[-max_chars:].lstrip()

    @staticmethod
    def _extract_scene_location(text: str) -> str:
        """Infer the dominant location from the tail of a scene.

        Populates ``prev_location`` in the next scene plan so the writer
        prompt detects location changes and mandates a transition paragraph.
        Returns a short string like 'Living room' or '' if unclear.
        """
        if not text:
            return ""
        tail = (text or "")[-700:].lower()
        location_patterns = [
            r"(?:in|at|inside|outside|back at|into|through)\s+(?:the\s+)?([a-z][a-z \'\-]{3,35}?)(?:[,.\n]|$)",
            r"(?:her|his|their|the)\s+([a-z]{3,20}(?:\s+[a-z]{2,15})?)\s+(?:room|apartment|office|house|flat|car|bus|bed|kitchen|bathroom|hallway)",
        ]
        for pat in location_patterns:
            m = re.search(pat, tail)
            if m:
                loc = m.group(1).strip().rstrip(".,;")
                if 3 < len(loc) < 40:
                    return loc.title()
        return ""
    # ─── Multi-Chapter Batch Generation ───────────────────────────────

    def generate_chapters(self, count: int = 1, pacing: str = "moderate") -> list:
        """Generate multiple chapters in sequence."""
        results = []
        state = self.state_manager.load()
        steps = self._premise_steps(state.get("metadata", {}).get("premise", ""))
        
        is_entire = (count in (-1, -2))
        limit = 50 if is_entire else count
        
        for i in range(limit):
            if self._cancelled:
                break
                
            # Reload state to get latest cursors and current chapter count
            state = self.state_manager.load()
            current_ch = self.state_manager.get_current_chapter()
            
            if is_entire:
                # 1. State-based completed cursor check
                if self._premise_exhausted_from_state(steps, state):
                    self._log("Entire story premise has been successfully covered (state check). Stopping batch generation.", level="success")
                    break
                # 2. Arithmetic fallback estimate check
                if self._premise_exhausted(current_ch + 1, steps):
                    self._log("Entire story premise has been successfully covered (arithmetic estimate). Stopping batch generation.", level="success")
                    break
                
            display_count = " (Entire Story)" if is_entire else f" {i + 1}/{count}"
            self._log(f"═══ Batch: chapter{display_count} ═══", level="header")
            try:
                result = self.generate_chapter(pacing=pacing)
            except Exception as e:
                self._log(
                    f"Chapter generation failed; skipping to next chapter",
                    level="error",
                    details={"error": str(e)},
                )
                results.append({"status": "failed", "error": str(e)})
                continue
            results.append(result)
            if result.get("status") not in ("complete", "failed"):
                break
        return results

    # ─── Full Story Export ────────────────────────────────────────────

    def export_full_story(self) -> str:
        """Concatenate all chapters into a single story document."""
        chapters = self._list_chapters()
        parts = []
        meta = self.state_manager.get_metadata()
        parts.append(f"# {meta.get('title', 'Untitled')}\n")
        parts.append(f"*{meta.get('genre', '')}*\n")
        if meta.get("premise"):
            parts.append(f"> {meta['premise']}\n")
        parts.append("---\n")

        for ch_file in chapters:
            path = os.path.join(self.chapters_dir, ch_file)
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read())
            parts.append("\n---\n")

        return "\n".join(parts)

    # ─── Delete Project ───────────────────────────────────────────────

    @staticmethod
    def delete_project(project_name: str):
        """Delete a project and all its data."""
        import shutil
        _, project_dir = _safe_project_dir(project_name)
        base = os.path.abspath(config.PROJECTS_DIR)
        if project_dir == base:
            raise ValueError("Refusing to delete the projects root")
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
            return True
        return False

    # ─── Phase 3: Scene Quality Scoring ──────────────────────────────

    def score_scene(
        self,
        scene_text: str,
        scene: dict,
        previous_ending: str = "",
        context: str = "",
    ) -> dict:
        """
        Run multi-dimensional quality scoring on a generated scene.

        Returns a dict with keys: coherence, pacing, voice, temporal,
        overall, passes, word_count, issues.
        """
        if not hasattr(self, "quality_ctrl"):
            self.quality_ctrl = QualityController()
        result = self.quality_ctrl.score(
            scene_text=scene_text,
            scene=scene,
            previous_ending=previous_ending,
            context=context,
        )
        summary = self.quality_ctrl.summarise(result)
        self._log(f"Scene quality: {summary}", level="info" if result.passes else "warn")
        return result.as_dict()

    # ─── Phase 5: World Bible ─────────────────────────────────────────

    def generate_world_bible(
        self,
        use_llm: bool = False,
        max_chapters: int = 999,
        progress_callback=None,
    ) -> dict:
        """
        Extract world-building facts from all completed chapters.

        Parameters
        ----------
        use_llm : bool
            If True and a model is available, use LLM-assisted extraction.
            If False (default), uses fast regex-only extraction.
        max_chapters : int
            Only scan chapters up to this number.
        progress_callback : callable | None
            Called with status strings during extraction.

        Returns a summary dict with entry counts and output paths.
        """
        from pipeline.world_bible import WorldBibleGenerator

        def _cb(msg: str) -> None:
            self._log(msg, level="info")
            if progress_callback:
                progress_callback(msg)
            self._emit("status", msg)

        llm = None
        if use_llm:
            llm = self._active_planning_model()

        gen = WorldBibleGenerator(
            project_dir=self.project_dir,
            llm=llm,
            progress_callback=_cb,
        )
        bible = gen.generate(max_chapters=max_chapters)

        out_md = os.path.join(self.project_dir, "world_bible.md")
        out_json = os.path.join(self.project_dir, "world_bible.json")

        return {
            "status": "ok",
            "locations": len(bible.locations),
            "factions": len(bible.factions),
            "objects": len(bible.objects),
            "rules": len(bible.rules),
            "events": len(bible.events),
            "relationships": len(bible.relationships),
            "markdown_path": out_md,
            "json_path": out_json,
        }

    # ─── Phase 5: Story Export ────────────────────────────────────────

    def export_story(
        self,
        fmt: str = "md",
        output_dir: Optional[str] = None,
    ) -> dict:
        """
        Export the story to a distributable file format.

        Parameters
        ----------
        fmt : str
            "md" | "txt" | "epub" | "all"
        output_dir : str | None
            Output directory. Defaults to the project directory.

        Returns a dict with status and output file path(s).
        """
        from pipeline.exporter import StoryExporter, epub_available

        exporter = StoryExporter(
            project_dir=self.project_dir,
            output_dir=output_dir or self.project_dir,
        )
        fmt = (fmt or "md").lower()
        results: dict[str, str] = {}

        if fmt in ("md", "markdown"):
            results["markdown"] = exporter.export_markdown()
        elif fmt == "txt":
            results["txt"] = exporter.export_txt()
        elif fmt == "epub":
            if not epub_available():
                return {
                    "status": "error",
                    "error": "ebooklib not installed. Run: pip install ebooklib",
                }
            results["epub"] = exporter.export_epub()
        elif fmt == "all":
            results = exporter.export_all(include_epub=epub_available())
        else:
            return {"status": "error", "error": f"Unknown format: {fmt!r}"}

        self._log(
            f"Story exported ({fmt}): {list(results.values())}",
            level="success",
        )
        return {"status": "ok", "format": fmt, "files": results}

