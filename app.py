"""
Story Teller — Web UI Server
Flask application with REST API and Server-Sent Events for live generation.
"""
import os
import sys
import json
import threading
import queue
import logging
import re
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify, Response
import config
from pipeline.orchestrator import PipelineOrchestrator, normalize_project_name
from pipeline.gemini_combiner import combine_chapters, analyze_and_polish
from models.groq_model import GroqModel
from models.llm import LlamaCPP, list_gguf_models, resolve_model_path, to_model_id

logger = logging.getLogger(__name__)

# Global state for active generation
_active_pipelines = {}
_event_queues = {}
_generation_lock = threading.Lock()
_selected_local_model_path = resolve_model_path(config.LLAMA_MODEL_PATH)
_selected_backend = "groq" if config.GROQ_API_KEY else "local"
_model_health_cache = {}
_combine_queues = {}
_active_combines = {}
_combine_results = {}
_manual_sessions = {}  # project_name -> {"pipeline": ..., "session": ..., "eq": ...}
_CHAPTER_FILE_RE = re.compile(r"^chapter_(\d{3,})\.md$")
_GROQ_MODEL_OPTION = "__groq_api__"


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


def _active_model_selection() -> str:
    return _GROQ_MODEL_OPTION if _selected_backend == "groq" else to_model_id(_selected_local_model_path)


def _current_pipeline_kwargs() -> dict:
    kwargs = {"local_model": _selected_local_model_path}
    if _selected_backend == "groq":
        kwargs["backend"] = "groq"
        kwargs["groq_model"] = config.GROQ_MODEL
    return kwargs


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "storyteller-dev-secret-key")

    def _resolve_generation_pipeline_kwargs(requested_model: str):
        global _selected_local_model_path, _selected_backend
        pipeline_kwargs = {"local_model": _selected_local_model_path}
        selected = (requested_model or "").strip() or _active_model_selection()

        if selected == _GROQ_MODEL_OPTION:
            if not config.GROQ_API_KEY:
                return None, jsonify({"error": "GROQ_API_KEY not set. Add it to .env before using Groq."}), 400
            pipeline_kwargs.update({"backend": "groq", "groq_model": config.GROQ_MODEL})
            _selected_backend = "groq"
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
    @app.route("/api/models", methods=["GET"])
    def list_models():
        """List available local GGUF models plus Groq API option."""
        global _selected_local_model_path, _selected_backend
        models = _list_local_models()
        if models:
            active_local = to_model_id(_selected_local_model_path)
            if active_local not in models:
                _selected_local_model_path = resolve_model_path(models[0])
        elif _selected_backend != "groq":
            _selected_backend = "local"

        available_models = [*models, _GROQ_MODEL_OPTION]
        active = _active_model_selection()
        if active not in available_models:
            if models:
                _selected_backend = "local"
                _selected_local_model_path = resolve_model_path(models[0])
                active = to_model_id(_selected_local_model_path)
            else:
                _selected_backend = "groq"
                active = _GROQ_MODEL_OPTION

        return jsonify({
            "models": available_models,
            "active": active,
            "groq_model": config.GROQ_MODEL,
        })

    @app.route("/api/models/switch", methods=["POST"])
    def switch_model():
        """Switch active model between local GGUF and Groq API."""
        global _selected_local_model_path, _selected_backend
        data = request.json or {}
        model = (data.get("model", "") or "").strip()
        if not model:
            return jsonify({"error": "No model specified"}), 400

        if model == _GROQ_MODEL_OPTION:
            if not config.GROQ_API_KEY:
                return jsonify({
                    "error": "GROQ_API_KEY not set. Add it to .env before selecting Groq.",
                }), 400
            groq = GroqModel(model=config.GROQ_MODEL)
            if not groq.is_available():
                return jsonify({
                    "error": "Groq API is unavailable right now. Check internet and API key.",
                }), 400
            _selected_backend = "groq"
            logger.info("Switched backend to Groq API (%s)", config.GROQ_MODEL)
            return jsonify({
                "status": "ok",
                "active": _GROQ_MODEL_OPTION,
                "provider": "groq",
                "groq_model": config.GROQ_MODEL,
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
        _selected_local_model_path = resolved
        config.LLAMA_MODEL_PATH = resolved
        active = to_model_id(resolved)
        logger.info(f"Switched local model to: {active}")
        return jsonify({"status": "ok", "active": active})

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
                    projects.append({
                        "name": name,
                        "title": meta.get("title", ""),
                        "genre": meta.get("genre", ""),
                        "current_chapter": meta.get("current_chapter", 0),
                        "total_scenes": meta.get("total_scenes_written", 0),
                        "characters": list(state.get("characters", {}).keys()),
                    })
        return jsonify({"projects": projects})

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
        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            pipeline.load_project()
            info = pipeline.get_project_info()
            state = pipeline.get_state()
            return jsonify({"info": info, "state": state})
        except Exception as e:
            return jsonify({"error": str(e)}), 404

    @app.route("/api/project/<name>/state", methods=["GET"])
    def get_project_state(name):
        name = normalize_project_name(name)
        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            pipeline.load_project()
            return jsonify({"state": pipeline.get_state()})
        except Exception as e:
            return jsonify({"error": str(e)}), 404

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

    # ─── API: Chapters ────────────────────────────────────────────
    @app.route("/api/project/<name>/chapters", methods=["GET"])
    def list_chapters(name):
        name = normalize_project_name(name)
        chapters_dir = os.path.join(config.PROJECTS_DIR, name, "chapters")
        chapters = []
        if os.path.exists(chapters_dir):
            for f in sorted(os.listdir(chapters_dir)):
                match = _CHAPTER_FILE_RE.match(f)
                if not match:
                    continue
                num = int(match.group(1))
                path = os.path.join(chapters_dir, f)
                with open(path, encoding="utf-8") as fh:
                    content = fh.read()
                words = len(content.split())
                chapters.append({
                    "number": num, "filename": f,
                    "words": words,
                })
        return jsonify({"chapters": chapters})

    @app.route("/api/project/<name>/chapter/<int:num>", methods=["GET"])
    def read_chapter(name, num):
        name = normalize_project_name(name)
        try:
            pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
            text = pipeline.read_chapter(num)
            if text:
                return jsonify({"chapter": num, "content": text})
            return jsonify({"error": "Chapter not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 404

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
        chapter_count = max(1, min(chapter_count, 20))  # Clamp 1-20
        requested_model = (data.get("model", "") or "").strip()
        pipeline_kwargs, err_resp, err_code = _resolve_generation_pipeline_kwargs(requested_model)
        if err_resp is not None:
            return err_resp, err_code

        with _generation_lock:
            if name in _active_pipelines:
                return jsonify({"error": "Generation already active"}), 409
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
                    _active_pipelines[name] = pipeline
                if chapter_count > 1:
                    results = pipeline.generate_chapters(count=chapter_count, pacing=pacing)
                    _queue_event(eq, _normalize_event("done", results[-1] if results else {}), force=True)
                else:
                    result = pipeline.generate_chapter(pacing=pacing)
                    _queue_event(eq, _normalize_event("done", result), force=True)
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
                    _active_pipelines[name] = pipeline
                result = pipeline.generate_chapter_manual(
                    chapter_title=chapter_title,
                    scene_briefs=final_scenes,
                    pacing=pacing,
                )
                _queue_event(eq, _normalize_event("done", result), force=True)
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
            _active_pipelines[name] = pipeline

        def run_scene():
            try:
                result = pipeline.generate_manual_scene(
                    session=session,
                    scene_brief=scene_brief,
                )
                _queue_event(eq, _normalize_event("manual_scene_done", result), force=True)
            except Exception as e:
                _queue_event(eq, _normalize_event("error", {"error": str(e)}), force=True)
            finally:
                with _generation_lock:
                    _active_pipelines.pop(name, None)

        thread = threading.Thread(target=run_scene, daemon=True)
        thread.start()

        scene_number = session["scene_counter"] + 1
        return jsonify({"status": "started", "scene_number": scene_number})

    @app.route("/api/project/<name>/generate/manual/cancel_scene", methods=["POST"])
    def cancel_manual_scene_api(name):
        """Cancel only the currently generating scene."""
        name = normalize_project_name(name)
        with _generation_lock:
            pipeline = _active_pipelines.get(name)
            if pipeline:
                pipeline.cancel_scene()
                return jsonify({"status": "cancelling"})
            return jsonify({"status": "not_generating"})

    @app.route("/api/project/<name>/chapter/<int:num>/resume", methods=["POST"])
    def resume_manual_chapter_api(name, num):
        """Resume a saved chapter by turning it back into an active manual session."""
        name = normalize_project_name(name)
        with _generation_lock:
            if name in _active_pipelines:
                return jsonify({"error": "Generation in progress"}), 409
            
            try:
                eq = queue.Queue(maxsize=config.SSE_QUEUE_MAXSIZE)

                def progress_cb(event, data=None, **kwargs):
                    msg = _normalize_event(event, data)
                    if not _queue_event(eq, msg):
                        logger.warning(f"SSE queue full for manual session '{name}'.")

                pipeline = PipelineOrchestrator(name, progress_cb, **_current_pipeline_kwargs())
                session = pipeline.resume_manual_chapter(num)
                
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
                import traceback
                traceback.print_exc()
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
        pipeline = _active_pipelines.get(name)
        if pipeline:
            pipeline.cancel()
            eq = _event_queues.get(name)
            if eq:
                _queue_event(eq, _normalize_event("status", "Cancellation requested"))
            return jsonify({"status": "cancelling"})
        return jsonify({"status": "no_active_generation"}), 404

    # ─── API: Combine & Polish (Gemini) ─────────────────────────────
    @app.route("/api/project/<name>/combine", methods=["POST"])
    def start_combine(name):
        """Combine all chapters and send to Gemini for continuity polish."""
        name = normalize_project_name(name)

        if not config.GEMINI_API_KEY:
            return jsonify({
                "error": "GEMINI_API_KEY is not set. Add it to your .env file.",
            }), 400

        with _generation_lock:
            if name in _active_combines:
                return jsonify({"error": "Combine already in progress"}), 409
            eq = queue.Queue(maxsize=100)
            _combine_queues[name] = eq

        def progress_cb(event, data=None, **kwargs):
            msg = _normalize_event(event, data)
            _queue_event(eq, msg)

        def run_combine():
            try:
                with _generation_lock:
                    _active_combines[name] = True

                pipeline = PipelineOrchestrator(name, **_current_pipeline_kwargs())
                pipeline.load_project()
                state = pipeline.get_state()
                metadata = state.get("metadata", {})
                chapters_dir = os.path.join(config.PROJECTS_DIR, name, "chapters")

                # Step 1: Combine chapters
                _queue_event(eq, _normalize_event("combine_status", {
                    "step": "Combining chapters into a single document...",
                }))
                combined = combine_chapters(chapters_dir, metadata)

                # Save original combined file
                output_dir = os.path.join(config.PROJECTS_DIR, name)
                original_path = os.path.join(output_dir, "combined_original.md")
                with open(original_path, "w", encoding="utf-8") as f:
                    f.write(combined)

                _queue_event(eq, _normalize_event("combine_status", {
                    "step": f"Combined {len(combined)} characters. Sending to Gemini...",
                }))

                # Step 2: Analyze and polish with Gemini
                result = analyze_and_polish(combined, progress_cb)

                # Save polished file
                polished_path = os.path.join(output_dir, "combined_polished.md")
                with open(polished_path, "w", encoding="utf-8") as f:
                    f.write(result["revised"])

                # Save analysis
                analysis_path = os.path.join(output_dir, "story_analysis.md")
                with open(analysis_path, "w", encoding="utf-8") as f:
                    f.write(result["analysis"])

                _combine_results[name] = {
                    "analysis": result["analysis"],
                    "model": result["model"],
                    "original_chars": len(combined),
                    "revised_chars": len(result["revised"]),
                }

                _queue_event(eq, _normalize_event("combine_done", {
                    "analysis": result["analysis"],
                    "model": result["model"],
                    "original_chars": len(combined),
                    "revised_chars": len(result["revised"]),
                }), force=True)

            except Exception as e:
                logger.error("Combine failed for %s: %s", name, e, exc_info=True)
                _queue_event(eq, _normalize_event("combine_error", {
                    "error": str(e),
                }), force=True)
            finally:
                with _generation_lock:
                    _active_combines.pop(name, None)

        thread = threading.Thread(target=run_combine, daemon=True)
        thread.start()
        return jsonify({"status": "started"})

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

    @app.route("/api/project/<name>/combine/download/<file_type>", methods=["GET"])
    def download_combined(name, file_type):
        """Download the combined story file (original or polished)."""
        name = normalize_project_name(name)
        project_dir = os.path.join(config.PROJECTS_DIR, name)

        if file_type == "original":
            path = os.path.join(project_dir, "combined_original.md")
            filename = f"{name}_combined_original.md"
        elif file_type == "polished":
            path = os.path.join(project_dir, "combined_polished.md")
            filename = f"{name}_combined_polished.md"
        elif file_type == "analysis":
            path = os.path.join(project_dir, "story_analysis.md")
            filename = f"{name}_story_analysis.md"
        else:
            return jsonify({"error": "Invalid file type"}), 400

        if not os.path.exists(path):
            return jsonify({"error": f"File not found. Run Combine & Polish first."}), 404

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return Response(
            content,
            mimetype="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
            },
        )

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

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
