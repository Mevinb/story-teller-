"""
Flask Router Blueprint for Standalone Vision Recognition System.
Completely isolated under /api/vision/*
"""

import logging
from flask import Blueprint, request, jsonify

from .analyzer import VisionAnalyzer
from .schema import PersonAnalysisResult
from .chat_provider import VisionChatProvider

logger = logging.getLogger(__name__)

vision_bp = Blueprint("vision_bp", __name__, url_prefix="/api/vision")
_analyzer = VisionAnalyzer()
_chat_provider = VisionChatProvider()


@vision_bp.route("/status", methods=["GET"])
def get_vision_status():
    """Get status of local and cloud vision backends."""
    try:
        status = _analyzer.get_available_backends()
        return jsonify({"success": True, "data": status}), 200
    except Exception as e:
        logger.error("Failed to get vision status: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@vision_bp.route("/analyze", methods=["POST"])
def analyze_person_image():
    """
    Endpoint for uploading a person image and receiving visual recognition & body description.
    Expects multipart/form-data with 'image' file OR JSON body with 'image_base64'.
    Optional parameters:
      - mode: 'local' | 'cloud' | 'auto'
      - prompt_extra: additional guidance for analysis
    """
    image_bytes = None
    mime_type = "image/jpeg"
    mode = None
    prompt_extra = ""

    if request.is_json:
        data = request.get_json() or {}
        mode = data.get("mode")
        prompt_extra = data.get("prompt_extra", "")
        base64_str = data.get("image_base64", "")
        if base64_str:
            import base64
            if "," in base64_str:
                header, base64_str = base64_str.split(",", 1)
                if "png" in header:
                    mime_type = "image/png"
                elif "webp" in header:
                    mime_type = "image/webp"
            image_bytes = base64.b64decode(base64_str)
    else:
        mode = request.form.get("mode")
        prompt_extra = request.form.get("prompt_extra", "")
        if "image" in request.files:
            file = request.files["image"]
            filename = file.filename.lower()
            if filename.endswith(".png"):
                mime_type = "image/png"
            elif filename.endswith(".webp"):
                mime_type = "image/webp"
            image_bytes = file.read()

    if not image_bytes:
        return jsonify({
            "success": False,
            "error": "No image provided. Upload a file field named 'image' or supply 'image_base64' in JSON body."
        }), 400

    try:
        result: PersonAnalysisResult = _analyzer.analyze(
            image_bytes=image_bytes,
            mime_type=mime_type,
            mode=mode,
            prompt_extra=prompt_extra
        )
        return jsonify({
            "success": True,
            "data": result.to_dict()
        }), 200
    except Exception as e:
        logger.error("Vision analysis endpoint error: %s", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@vision_bp.route("/pull", methods=["POST"])
def pull_local_vision_model():
    """Trigger automatic pull of local Ollama vision model."""
    try:
        success = _analyzer.local_provider.pull_model()
        if success:
            return jsonify({"success": True, "message": f"Successfully pulled local vision model '{_analyzer.local_provider.model}'."}), 200
        else:
            return jsonify({"success": False, "error": f"Failed to pull model '{_analyzer.local_provider.model}'. Ensure Ollama is running (`ollama serve`)."}), 500
    except Exception as e:
        logger.error("Pull endpoint error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500
@vision_bp.route("/chat/start", methods=["POST"])
def start_vision_chat():
    """Starts a new vision chat session with an uploaded image."""
    image_bytes = None
    
    if request.is_json:
        data = request.get_json() or {}
        base64_str = data.get("image_base64", "")
        if base64_str:
            import base64
            if "," in base64_str:
                header, base64_str = base64_str.split(",", 1)
            image_bytes = base64.b64decode(base64_str)
    else:
        if "image" in request.files:
            file = request.files["image"]
            image_bytes = file.read()
            
    if not image_bytes:
        return jsonify({"success": False, "error": "No image provided."}), 400
        
    try:
        result = _chat_provider.start_session(image_bytes)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        logger.error("Vision chat start error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@vision_bp.route("/chat/message", methods=["POST"])
def send_vision_chat_message():
    """Sends a message to an existing vision chat session."""
    data = request.get_json() or {}
    session_id = data.get("session_id")
    message = data.get("message")
    
    if not session_id or not message:
        return jsonify({"success": False, "error": "Missing session_id or message."}), 400
        
    try:
        response = _chat_provider.chat(session_id, message)
        return jsonify({"success": True, "data": {"response": response}}), 200
    except Exception as e:
        logger.error("Vision chat message error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500
