"""
Pipeline Orchestrator — Main controller for the story generation pipeline.
Manages the full chapter generation flow from planning to storage.
"""
import os
import json
import logging
import time
import re
from datetime import datetime
from typing import Optional, Callable

import config
from models.llm import LlamaCPP, to_model_id
from models.groq_model import GroqModel
from memory.state_manager import StateManager
from memory.vector_store import VectorStore
from memory.retriever import Retriever
from agents.architect import StoryArchitect
from agents.planner import ScenePlanner
from agents.writer import SceneWriter
from agents.consistency import ConsistencyEngine
from agents.editor import Editor
from memory.evolution_engine import evolve_after_scene

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
    "IMPORTANT: Preserve the user's original theme and intent exactly. "
    "Do NOT change what happens — only add detail about HOW to write it. "
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
- PRESERVE the user's original theme and intent. Do not overdrift or change what happens.
- Keep the user's wording in the summary where possible — refine, don't replace.
- Add actionable writing details: mood cues, sensory notes, character actions, dialogue hints.
- Use exact known character names when applicable.
- If previous scenes are listed, ensure continuity with them (do not repeat events).
- Keep the user's core action on-page in this scene. Do NOT convert the brief into aftermath-only framing.
- Do not add meta text or commentary.
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
    ):
        self.project_name, self.project_dir = _safe_project_dir(project_name)
        self.chapters_dir = os.path.join(self.project_dir, "chapters")
        self.logs_dir = os.path.join(self.project_dir, "logs")
        self._progress_cb = progress_callback or (lambda *a, **kw: None)
        self._cancelled = False
        self._scene_cancelled = False
        self._backend = (backend or "local").strip().lower()
        if self._backend not in {"local", "groq"}:
            raise ValueError(f"Unsupported backend: {backend}")
        selected_local = (local_model or config.LLAMA_MODEL_PATH).strip()
        self._local_model_ref = selected_local
        self._local_model_id = to_model_id(selected_local)
        self._groq_model_id = (groq_model or config.GROQ_MODEL).strip()
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
        if self._backend == "groq":
            if not self.cloud_model:
                raise RuntimeError("Groq backend selected but cloud model is not initialized.")
            self.architect = StoryArchitect(self.cloud_model, self.state_manager)
            self.planner = ScenePlanner(self.cloud_model)
            self.writer = SceneWriter(self.cloud_model, self.cloud_model)
            self.consistency = ConsistencyEngine(self.cloud_model)
            self.editor = Editor(self.cloud_model)
            self._enable_compact_agent_prompts()
            return

        # All local by default; cloud can optionally be writer-primary.
        self.architect = StoryArchitect(self.local_model, self.state_manager)
        self.planner = ScenePlanner(self.local_model)
        primary_writer = self.cloud_model if self._cloud_available else self.local_model
        self.writer = SceneWriter(primary_writer, self.local_model)
        self.consistency = ConsistencyEngine(self.local_model)
        self.editor = Editor(self.local_model)

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
                mode = "rewrite" if state.get("blocking_issues") else "write"
                writer_input = {
                    "mode": mode,
                    "scene_plan": scene,
                    "chapter_num": chapter_num,
                    "context": scene_context,
                    "previous_ending": previous_ending,
                    "original_text": state["scene_text"],
                    "issues": state.get("blocking_issues", []),
                    "state_context": state.get("consistency_context", ""),
                    "stream_callback": lambda chunk: self._emit(
                        "token",
                        {
                            "scene": scene_num,
                            "content": chunk,
                            "provider": self.writer.last_provider,
                        },
                    ),
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
                if step_counter["steps"] >= config.MAX_PIPELINE_STEPS:
                    raise RuntimeError(
                        f"Pipeline step budget exceeded ({config.MAX_PIPELINE_STEPS})"
                    )
                step_counter["steps"] += 1
                should_rewrite = blocking and not reached_max_iter and not stuck_on_same_issues
                next_action = "writer" if should_rewrite else "editor"
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
                    },
                    latency_ms=0.0,
                    next_action=next_action,
                )
                if should_rewrite:
                    self._emit(
                        "status",
                        f"Consistency issues in scene {scene_num}. Rewriting (attempt {iteration})...",
                    )
                    graph_node = SCENE_GRAPH["decision"][0]
                else:
                    if blocking and reached_max_iter:
                        self._log(
                            f"Scene {scene_num} reached max iterations ({config.MAX_SCENE_ITERATIONS}); continuing with best draft",
                            level="warn",
                        )
                    elif stuck_on_same_issues:
                        self._log(
                            f"Scene {scene_num} hit repeated blocker loop; continuing with current best draft",
                            level="warn",
                        )
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
                    state["scene_text"] = pre_edit_text
                    self._validate_scene_completion(
                        scene_text=state["scene_text"],
                        scene=scene,
                        chapter_num=chapter_num,
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

        kept = []
        removed = 0
        for sentence in cls._split_sentences(text):
            key = cls._sentence_key(sentence)
            if len(sentence.split()) >= 8 and key in reference_keys:
                removed += 1
                continue
            kept.append(sentence)

        if not removed:
            return text

        if removed:
            logger.warning(
                "Removed %s sentence(s) copied from the previous scene ending",
                removed,
            )
        return " ".join(kept).strip() or text

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
        words = len((scene_text or "").split())
        try:
            target = int(scene.get("word_target", config.MIN_SCENE_WORDS))
        except (TypeError, ValueError):
            target = config.MIN_SCENE_WORDS

        minimum = max(120, min(config.MIN_SCENE_WORDS, int(target * 0.45)))
        if words < minimum:
            scene_num = scene.get("scene_number", "?")
            self._log(
                f"Scene {scene_num} rejected as incomplete",
                level="warn",
                details={
                    "words": words,
                    "minimum_words": minimum,
                    "chapter": chapter_num,
                },
            )
            raise RuntimeError(
                f"Scene {scene_num} in chapter {chapter_num} is incomplete "
                f"({words} words; expected at least {minimum}). "
                "Generation likely stopped early due to rate limits or truncation."
            )

        if not re.search(r'[.!?]["\']?\s*$', scene_text or ""):
            scene_num = scene.get("scene_number", "?")
            self._log(
                f"Scene {scene_num} rejected because it ends mid-sentence",
                level="warn",
                details={"chapter": chapter_num, "words": words},
            )
            raise RuntimeError(
                f"Scene {scene_num} in chapter {chapter_num} appears truncated. "
                "WIP checkpoint preserved; retry later to resume."
            )

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
        text = premise or ""
        marker = re.search(r"story steps in order\s*:\s*", text, flags=re.IGNORECASE)
        if marker:
            text = text[marker.end():]

        steps = []
        current_heading = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.endswith(":") and len(line.split()) <= 8:
                current_heading = line.rstrip(":")
                continue
            line = re.sub(r"^(?:[-*]|\d+[\).\:-])\s*", "", line).strip()
            if not line:
                continue
            if current_heading:
                line = f"{current_heading}: {line}"
            steps.append(line)
        return steps

    @staticmethod
    def _allowed_premise_step_index(chapter_num: int, steps: list[str]) -> int:
        if not steps:
            return -1
        # Conservative default: one major premise step per chapter. This avoids
        # skipping ahead when the model gets excited by later high-drama beats.
        return max(0, min(chapter_num - 1, len(steps) - 1))

    def _append_premise_step_anchor(self, context: str, chapter_num: int, state: dict) -> str:
        steps = self._premise_steps(state.get("metadata", {}).get("premise", ""))
        if not steps:
            return context
        allowed_idx = self._allowed_premise_step_index(chapter_num, steps)
        completed = steps[:allowed_idx]
        current = steps[allowed_idx]
        future = steps[allowed_idx + 1: allowed_idx + 4]
        return (
            f"{context.rstrip()}\n\n"
            "=== PREMISE ORDER CURSOR ===\n"
            f"Completed premise steps: {' | '.join(completed) if completed else 'None yet.'}\n"
            f"This chapter may cover ONLY this next premise step: {current}\n"
            f"Future steps forbidden for this chapter: {' | '.join(future) if future else 'None.'}\n"
            "Hard rule: do not introduce events, locations, relationships, private contact, or characters "
            "whose first premise appearance belongs to a future step."
        )

    @staticmethod
    def _character_first_premise_steps(character_names: list[str], steps: list[str]) -> dict[str, int]:
        first = {}
        for name in character_names:
            lowered_name = str(name).lower()
            for idx, step in enumerate(steps):
                if re.search(r"\b" + re.escape(lowered_name) + r"\b", step.lower()):
                    first[name] = idx
                    break
        return first

    @staticmethod
    def _distinctive_future_markers(step: str) -> set[str]:
        lowered = re.sub(r"[^a-z0-9\s]+", " ", (step or "").lower())
        tokens = [
            token for token in lowered.split()
            if len(token) > 3 and token not in {
                "with", "while", "from", "that", "this", "then", "their",
                "becomes", "starts", "continues", "scene", "scenes",
                "sherin", "jomy", "riya",
            }
        ]
        markers = set()
        for size in (3, 2):
            for i in range(0, max(0, len(tokens) - size + 1)):
                phrase = " ".join(tokens[i:i + size])
                if len(phrase) >= 10:
                    markers.add(phrase)
        return markers

    @staticmethod
    def _normalized_text_for_marker_checks(text: str) -> str:
        return re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower())

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
        allowed_idx = self._allowed_premise_step_index(chapter_num, steps)
        lowered = self._normalized_text_for_marker_checks(text)
        allowed_and_prior = self._normalized_text_for_marker_checks(" ".join(steps[:allowed_idx + 1]))
        violations = []

        if check_character_names:
            char_first = self._character_first_premise_steps(
                list(state.get("characters", {}).keys()),
                steps,
            )
            for name, first_idx in char_first.items():
                if first_idx > allowed_idx and re.search(r"\b" + re.escape(str(name).lower()) + r"\b", text.lower()):
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
        if not steps:
            return chapter_plan
        allowed_idx = self._allowed_premise_step_index(chapter_num, steps)
        step = steps[allowed_idx]
        key_events = self._sentences_from_step(step)
        names = self._names_in_text(step, state)
        if not names:
            lead = self._primary_character_name(state)
            names = [lead] if lead else []
        arcs = {
            name: f"{name} develops through the allowed premise step for this chapter."
            for name in names
        }
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
            "plot_direction": step,
            "character_arcs": arcs,
            "key_events": key_events,
            "constraints": [
                f"Cover only premise step {allowed_idx + 1}.",
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

        lead_name = self._primary_character_name(state)
        introduced = set(self._introduced_character_names(chapter_num, state))
        all_names = set(state.get("characters", {}).keys())

        scenes = []
        for idx, event in enumerate(key_events):
            event_text = str(event).strip() or "Advance the chapter beat."
            event_names = self._names_in_text(event_text, state)
            if not event_names and lead_name:
                event_names = [lead_name]

            summary_text = event_text
            key_event_text = event_text
            for name in event_names:
                if name not in (all_names - introduced):
                    continue
                if not self._event_introduces_character(f"{summary_text} {key_event_text}", name):
                    if name == lead_name:
                        intro = f"The chapter opens by introducing {name} on-page."
                    else:
                        intro = f"{lead_name or 'The protagonist'} meets {name} for the first time."
                    summary_text = f"{intro} {event_text}".strip()
                    key_event_text = summary_text
                introduced.add(name)

            scenes.append({
                "scene_number": idx + 1,
                "type": "setup" if idx == 0 else "build_tension" if idx < len(key_events) - 1 else "resolution",
                "summary": summary_text,
                "characters_present": event_names,
                "location": "College campus",
                "mood": chapter_plan.get("tone", "intimate"),
                "key_events": [key_event_text],
                "dialogue_notes": "Keep the scene focused on this event only.",
                "sensory_details": "Use grounded campus details and character reactions.",
                "word_target": max(config.WORDS_PER_SCENE_MIN, min(600, config.WORDS_PER_SCENE_MAX)),
            })
        while len(scenes) < config.SCENES_PER_CHAPTER_MIN:
            scenes.append({
                "scene_number": len(scenes) + 1,
                "type": "resolution",
                "summary": chapter_plan.get("plot_direction", "Resolve the chapter beat without advancing to future premise steps."),
                "characters_present": [lead_name] if lead_name else [],
                "location": "College campus",
                "mood": chapter_plan.get("tone", "intimate"),
                "key_events": [chapter_plan.get("plot_direction", "Resolve the chapter beat.")],
                "dialogue_notes": "Do not introduce future premise material.",
                "sensory_details": "Focus on emotional aftermath.",
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
        if self._backend == "groq" and self.cloud_model:
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
            context=context[:1400],
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
            max_tokens=1200 if self._backend == "groq" else None,
        )
        parsed = response.as_json() if response else None
        if not isinstance(parsed, dict):
            parsed = {}

        summary = str(parsed.get("summary", "")).strip() or scene_brief
        key_events = self._normalize_text_list(parsed.get("key_events"))
        if not key_events:
            key_events = [summary]
        brief = str(scene_brief or "").strip()
        if brief and not any(self._event_matches_brief(event, brief) for event in key_events):
            key_events = [brief] + key_events
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
        context = session["context"]
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

        # Build summary of previously completed scenes for continuity
        prev_summaries = []
        for i, sp in enumerate(session["completed_scene_plans"]):
            prev_summaries.append(f"Scene {i + 1}: {sp.get('summary', '')[:120]}")
        previous_scenes_summary = "\n".join(prev_summaries) if prev_summaries else "None (this is the first scene)."

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

        # Retrieve scene-level context
        self._emit("agent_active", {"agent": "Retriever", "step": f"scene {scene_number}"})
        scene_context = self.retriever.retrieve_context(
            scene_plan=scene,
            chapter_num=chapter_num,
            previous_ending=previous_ending,
            top_k=2 if self._backend == "groq" else None,
            max_chars=2200 if self._backend == "groq" else None,
        )

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

        scene_text = self._sanitize_generated_text(scene_result["scene_text"])
        scene_text = self._remove_repeated_reference_sentences(scene_text, previous_ending)
        self._validate_scene_completion(
            scene_text=scene_text,
            scene=scene,
            chapter_num=chapter_num,
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
                compact_mode=(self._backend == "groq"),
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
        self.state_manager.add_chapter_summary(chapter_num, summary)
        self.state_manager.increment_scene_count(completed_scene_count)

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
        self._cancelled = False
        self.state_manager.normalize_character_traits()
        state = self.state_manager.load()
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
                architect_input = {
                    "chapter_num": chapter_num,
                    "context": plan_context,
                    "pacing": pacing,
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
                    self._validate_plan_premise_order(chapter_plan, chapter_num, state)
                    self._validate_plan_character_introductions(chapter_plan, chapter_num, state)
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
                    self._validate_scene_premise_order(scenes, chapter_num, state)
                    self._validate_scene_character_introductions(scenes, chapter_num, state)
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

                # 4a: Retrieve context for this scene
                self._emit("agent_active", {
                    "agent": "Retriever", "step": f"scene {scene_num}",
                })
                self._log(f"Retriever: fetching context for scene {scene_num}...")
                scene_context = self.retriever.retrieve_context(
                    scene_plan=scene,
                    chapter_num=chapter_num,
                    previous_ending=previous_ending,
                    top_k=2 if self._backend == "groq" else None,
                    max_chars=2200 if self._backend == "groq" else None,
                )
                self._log(f"Context retrieved: {len(scene_context)} chars",
                          details={"top_k": config.TOP_K_RETRIEVAL})

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

                scene_text = self._sanitize_generated_text(scene_result["scene_text"])
                scene_text = self._remove_repeated_reference_sentences(
                    scene_text,
                    previous_ending,
                )
                retries = max(0, scene_result["iterations"] - 1)
                post_edit_words = len(scene_text.split())
                self._validate_scene_completion(
                    scene_text=scene_text,
                    scene=scene,
                    chapter_num=chapter_num,
                )
                estimated_tokens_used += int(post_edit_words * 1.35)

                chapter_text_parts.append(scene_text)
                previous_ending = self._extract_ending(scene_text)
                self._ensure_scene_characters_known(scene)

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
                        compact_mode=(self._backend == "groq"),
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
            self.state_manager.add_chapter_summary(chapter_num, summary)
            self.state_manager.increment_scene_count(completed_scene_count)

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
        completed_chapter = int(
            self.state_manager.state.get("metadata", {}).get("current_chapter", 0)
        )
        if chapter_num > completed_chapter:
            return None
        path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return None

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
        """Save work-in-progress after each completed scene."""
        wip = {
            "chapter_num": chapter_num,
            "chapter_plan": chapter_plan,
            "scene_plans": scene_plans,
            "completed_scenes": completed_scenes,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self._wip_path(chapter_num), "w", encoding="utf-8") as f:
            json.dump(wip, f, indent=2, ensure_ascii=False)

    def _load_wip(self, chapter_num: int) -> Optional[dict]:
        """Load a WIP checkpoint if it exists."""
        path = self._wip_path(chapter_num)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _clear_wip(self, chapter_num: int):
        """Remove WIP file after successful chapter completion."""
        path = self._wip_path(chapter_num)
        if os.path.exists(path):
            os.remove(path)

    # ─── Sentence-Boundary Ending ─────────────────────────────────────

    @staticmethod
    def _extract_ending(text: str, max_chars: int = 320) -> str:
        """Extract a short complete-sentence tail for continuity prompts."""
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return cleaned[-max_chars:]

        tail = " ".join(sentences[-3:]).strip()
        if len(tail) <= max_chars:
            return tail
        return tail[-max_chars:].lstrip()

    # ─── Multi-Chapter Batch Generation ───────────────────────────────

    def generate_chapters(self, count: int = 1, pacing: str = "moderate") -> list:
        """Generate multiple chapters in sequence."""
        results = []
        for i in range(count):
            if self._cancelled:
                break
            self._log(f"═══ Batch: chapter {i + 1}/{count} ═══", level="header")
            result = self.generate_chapter(pacing=pacing)
            results.append(result)
            if result.get("status") != "complete":
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
