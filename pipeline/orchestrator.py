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

logger = logging.getLogger(__name__)

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
    ):
        self.project_name = project_name
        self.project_dir = os.path.join(config.PROJECTS_DIR, project_name)
        self.chapters_dir = os.path.join(self.project_dir, "chapters")
        self.logs_dir = os.path.join(self.project_dir, "logs")
        self._progress_cb = progress_callback or (lambda *a, **kw: None)
        self._cancelled = False
        selected_local = (local_model or config.LLAMA_MODEL_PATH).strip()
        self._local_model_ref = selected_local
        self._local_model_id = to_model_id(selected_local)

        # Create directories
        for d in [self.project_dir, self.chapters_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)

        # Initialize components
        self._init_models()
        self._init_memory()
        self._init_agents()

    def _init_models(self):
        self._emit("status", "Initializing models...")

        # Local-only by default. Cloud can be enabled explicitly with USE_CLOUD_MODEL=true.
        self.cloud_model = None
        self._cloud_available = False
        self._local_model = None  # Lazy-loaded only when needed

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
                self._log("Cloud model unavailable — loading local model", level="warn")
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
        self._log("Loading local model...", details={"model": self._local_model_id})
        self._local_model = LlamaCPP(model_path=self._local_model_ref)
        if not self._local_model.is_available():
            raise RuntimeError(
                f"Local GGUF model '{self._local_model_id}' is not available. "
                "Set LLAMA_MODEL_PATH to a valid .gguf file."
            )
        self._emit("status", f"Local model loaded: {self._local_model.get_name()}")
        self._log(f"Local model ready: {self._local_model.get_name()}", level="success")

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
        # All local — Groq only as bonus when available
        self.architect = StoryArchitect(self.local_model, self.state_manager)
        self.planner = ScenePlanner(self.local_model)
        primary_writer = self.cloud_model if self._cloud_available else self.local_model
        self.writer = SceneWriter(primary_writer, self.local_model)
        self.consistency = ConsistencyEngine(self.local_model)
        self.editor = Editor(self.local_model)

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
            if self._cancelled:
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
                state["scene_text"] = editor_result["output"]["scene_text"]
                graph_node = SCENE_GRAPH["editor"][0]

        return {
            "status": "complete",
            "scene_text": state["scene_text"],
            "iterations": iteration,
        }

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

    def generate_chapter(self, pacing: str = "moderate") -> dict:
        """Generate the next chapter through the full pipeline."""
        self._cancelled = False
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

        try:
            # Step 0: Re-anchor if needed
            if (chapter_num > 1 and
                    (chapter_num - 1) % config.REANCHOR_EVERY_N_CHAPTERS == 0):
                self._emit("status", "Re-anchoring narrative state...")
                self._log("Re-anchoring narrative state to prevent drift...", level="info")
                t0 = time.time()
                self.architect.reanchor()
                self._log(f"Re-anchoring complete in {time.time()-t0:.1f}s", level="success")
                log_entries.append({"step": "reanchor", "status": "done"})

            # Step 1: Retrieve context for planning
            graph_node = PIPELINE_GRAPH["start"][0]
            self._emit("agent_active", {"agent": "Retriever", "step": "context"})
            self._log("Retriever: assembling story context for planning...",
                      details={"vector_chunks": self.vector_store.get_stats()["total_vectors"]})
            plan_context = self.state_manager.get_context_window(include_premise=False)
            self._log(f"Context assembled: {len(plan_context)} chars",
                      details={"characters": list(state.get('characters', {}).keys())})

            # Step 2: Architect plans chapter
            self._emit("agent_active", {"agent": "Story Architect", "step": "planning"})
            self._log(f"Story Architect: planning chapter {chapter_num}...",
                      details={"model": self._local_model_id, "backend": "local"})
            if self._cancelled:
                return self._cancel_result(chapter_num)

            graph_node = PIPELINE_GRAPH[graph_node][0]
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
            self._log(f"Scene Planner: decomposing chapter into scenes...",
                      details={"model": self._local_model_id, "backend": "local"})
            if self._cancelled:
                return self._cancel_result(chapter_num)

            graph_node = PIPELINE_GRAPH[graph_node][0]
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
            if chapter_num > 1:
                prev_chapter_path = os.path.join(
                    self.chapters_dir, f"chapter_{(chapter_num - 1):03d}.md"
                )
                if os.path.exists(prev_chapter_path):
                    with open(prev_chapter_path, "r", encoding="utf-8") as f:
                        prev_text = f.read()
                    previous_ending = self._extract_ending(prev_text)
                    self._log(f"Loaded ending from chapter {chapter_num - 1} for continuity")

            # Check for WIP checkpoint (resume after crash)
            wip = self._load_wip(chapter_num)
            start_scene_idx = 0
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
                    scene_plan=scene, chapter_num=chapter_num,
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

                scene_text = scene_result["scene_text"]
                retries = max(0, scene_result["iterations"] - 1)
                post_edit_words = len(scene_text.split())
                estimated_tokens_used += int(post_edit_words * 1.35)

                if estimated_tokens_used > config.MAX_TOKEN_BUDGET:
                    self._emit("status", "Token budget reached; stopping early.")
                    self._log(
                        f"Chapter halted at scene {scene_num}: token budget exceeded",
                        level="warn",
                        details={
                            "estimated_tokens": estimated_tokens_used,
                            "budget": config.MAX_TOKEN_BUDGET,
                        },
                    )
                    break

                chapter_text_parts.append(scene_text)
                previous_ending = self._extract_ending(scene_text)

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
                completed = [{"scene": s+1, "text": t} for s, t in enumerate(chapter_text_parts)]
                self._save_wip(chapter_num, completed, chapter_plan, scenes)

                # Brief pause between scenes to avoid Groq rate limits
                if i < len(scenes) - 1:
                    time.sleep(5)

                # Save chapter file progressively so Reader shows it building up
                chapter_title = chapter_plan.get("chapter_title", f"Chapter {chapter_num}")
                separator = "\n\n* * *\n\n"
                chapter_text = separator.join(chapter_text_parts)
                full_chapter = f"# Chapter {chapter_num}: {chapter_title}\n\n{chapter_text}"
                chapter_path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
                with open(chapter_path, "w", encoding="utf-8") as f:
                    f.write(full_chapter)

            # Step 5: Assemble chapter
            graph_node = PIPELINE_GRAPH[graph_node][0]
            if not chapter_text_parts:
                raise RuntimeError(
                    "Chapter generation terminated before any scene text was produced."
                )
            chapter_title = chapter_plan.get("chapter_title", f"Chapter {chapter_num}")
            separator = "\n\n* * *\n\n"
            chapter_text = separator.join(chapter_text_parts)
            full_chapter = f"# Chapter {chapter_num}: {chapter_title}\n\n{chapter_text}"

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

    def _cancel_result(self, chapter_num):
        return {"chapter_number": chapter_num, "status": "cancelled"}

    # ─── Utilities ────────────────────────────────────────────────────

    def _list_chapters(self) -> list:
        if not os.path.exists(self.chapters_dir):
            return []
        files = sorted(f for f in os.listdir(self.chapters_dir) if f.endswith(".md"))
        return files

    def read_chapter(self, chapter_num: int) -> Optional[str]:
        path = os.path.join(self.chapters_dir, f"chapter_{chapter_num:03d}.md")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return None

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

        sentences = re.split(r'(?<=[.!?]["\']?)\s+', cleaned)
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
            self._log(f"═══ Batch: chapter {i+1}/{count} ═══", level="header")
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
        project_dir = os.path.join(config.PROJECTS_DIR, project_name)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
            return True
        return False
