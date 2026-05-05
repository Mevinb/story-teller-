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
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import config
from pipeline.orchestrator import PipelineOrchestrator
from models.llm import LlamaCPP, list_gguf_models, resolve_model_path, to_model_id

logger = logging.getLogger(__name__)

# Global state for active generation
_active_pipelines = {}
_event_queues = {}
_generation_lock = threading.Lock()
_selected_local_model_path = resolve_model_path(config.LLAMA_MODEL_PATH)
_model_health_cache = {}


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


def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["SECRET_KEY"] = "storyteller-secret-key"

    # ─── Pages ────────────────────────────────────────────────────
    @app.route("/")
    def index():
        return render_template("index.html")

    # ─── API: Model Management ────────────────────────────────────
    @app.route("/api/models", methods=["GET"])
    def list_models():
        """List available local GGUF models and the currently active one."""
        global _selected_local_model_path
        models = _list_local_models()
        if not models:
            return jsonify({"models": [], "active": None})

        active = to_model_id(_selected_local_model_path)
        if active not in models:
            _selected_local_model_path = resolve_model_path(models[0])
            active = to_model_id(_selected_local_model_path)

        return jsonify({
            "models": models,
            "active": active,
        })

    @app.route("/api/models/switch", methods=["POST"])
    def switch_model():
        """Switch the active local GGUF model."""
        global _selected_local_model_path
        data = request.json or {}
        model = (data.get("model", "") or "").strip()
        if not model:
            return jsonify({"error": "No model specified"}), 400

        resolved = resolve_model_path(model)
        if not resolved.lower().endswith(".gguf"):
            return jsonify({"error": "Model must be a .gguf file"}), 400
        if not os.path.isfile(resolved):
            return jsonify({"error": f"Model file not found: {resolved}"}), 400

        ok, probe_error = _probe_local_model(resolved, force=True)
        if not ok:
            return jsonify({"error": probe_error}), 400

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
                    with open(state_path) as f:
                        state = json.load(f)
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

        project_name = title.lower().replace(" ", "_").replace("'", "")
        try:
            pipeline = PipelineOrchestrator(project_name, local_model=_selected_local_model_path)
            state = pipeline.create_project(
                title=title, genre=genre, premise=premise,
                characters=characters, themes=themes, setting=setting,
            )
            return jsonify({"status": "ok", "project": project_name, "state": state})
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/project/<name>", methods=["GET"])
    def get_project(name):
        try:
            pipeline = PipelineOrchestrator(name, local_model=_selected_local_model_path)
            pipeline.load_project()
            info = pipeline.get_project_info()
            state = pipeline.get_state()
            return jsonify({"info": info, "state": state})
        except Exception as e:
            return jsonify({"error": str(e)}), 404

    @app.route("/api/project/<name>/state", methods=["GET"])
    def get_project_state(name):
        try:
            pipeline = PipelineOrchestrator(name, local_model=_selected_local_model_path)
            pipeline.load_project()
            return jsonify({"state": pipeline.get_state()})
        except Exception as e:
            return jsonify({"error": str(e)}), 404

    @app.route("/api/project/<name>/state", methods=["PUT"])
    def update_project_state(name):
        try:
            pipeline = PipelineOrchestrator(name, local_model=_selected_local_model_path)
            pipeline.load_project()
            updates = request.json
            pipeline.state_manager.apply_state_update(updates)
            return jsonify({"status": "ok", "state": pipeline.get_state()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ─── API: Chapters ────────────────────────────────────────────
    @app.route("/api/project/<name>/chapters", methods=["GET"])
    def list_chapters(name):
        chapters_dir = os.path.join(config.PROJECTS_DIR, name, "chapters")
        chapters = []
        if os.path.exists(chapters_dir):
            for f in sorted(os.listdir(chapters_dir)):
                if f.endswith(".md"):
                    num = int(f.replace("chapter_", "").replace(".md", ""))
                    path = os.path.join(chapters_dir, f)
                    with open(path) as fh:
                        content = fh.read()
                    words = len(content.split())
                    chapters.append({
                        "number": num, "filename": f,
                        "words": words, "content": content,
                    })
        return jsonify({"chapters": chapters})

    @app.route("/api/project/<name>/chapter/<int:num>", methods=["GET"])
    def read_chapter(name, num):
        try:
            pipeline = PipelineOrchestrator(name, local_model=_selected_local_model_path)
            text = pipeline.read_chapter(num)
            if text:
                return jsonify({"chapter": num, "content": text})
            return jsonify({"error": "Chapter not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 404

    # ─── API: Generation (SSE) ────────────────────────────────────
    @app.route("/api/project/<name>/generate", methods=["POST"])
    def start_generation(name):
        data = request.json or {}
        pacing = data.get("pacing", "moderate")
        try:
            chapter_count = int(data.get("chapter_count", 1))
        except (TypeError, ValueError):
            chapter_count = 1
        chapter_count = max(1, min(chapter_count, 20))  # Clamp 1-20
        requested_model = (data.get("model", "") or "").strip() or to_model_id(_selected_local_model_path)
        requested_model_path = resolve_model_path(requested_model)

        if not requested_model_path.lower().endswith(".gguf"):
            return jsonify({"error": "Model must be a .gguf file"}), 400
        if not os.path.isfile(requested_model_path):
            return jsonify({"error": f"Model file not found: {requested_model_path}"}), 400

        ok, probe_error = _probe_local_model(requested_model_path)
        if not ok:
            return jsonify({
                "error": (
                    f"{probe_error} Try a different GGUF file, lower LLAMA_N_GPU_LAYERS, "
                    "or choose a smaller quantized model (e.g., 7B Q4_K_M)."
                )
            }), 400

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
                    local_model=requested_model_path,
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

    @app.route("/api/project/<name>/generate/stream", methods=["GET"])
    def stream_generation(name):
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
        pipeline = _active_pipelines.get(name)
        if pipeline:
            pipeline.cancel()
            eq = _event_queues.get(name)
            if eq:
                _queue_event(eq, _normalize_event("status", "Cancellation requested"))
            return jsonify({"status": "cancelling"})
        return jsonify({"status": "no_active_generation"}), 404

    # ─── API: Project Management ──────────────────────────────────
    @app.route("/api/project/<name>/delete", methods=["POST"])
    def delete_project(name):
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
        try:
            pipeline = PipelineOrchestrator(name, local_model=_selected_local_model_path)
            pipeline.load_project()
            pipeline.state_manager.reset_generated()
            return jsonify({"status": "ok", "message": "Generated content cleared"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/project/<name>/export", methods=["GET"])
    def export_project(name):
        try:
            pipeline = PipelineOrchestrator(name, local_model=_selected_local_model_path)
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
