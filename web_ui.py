"""Serve the compiled React studio, with the classic UI kept for rollback."""
from pathlib import Path

from flask import render_template, send_from_directory

import config


def register_ui(app):
    @app.get("/")
    def index():
        build_dir = Path(app.static_folder) / "app"
        if (build_dir / "index.html").is_file():
            response = send_from_directory(build_dir, "index.html", max_age=0)
            response.headers["Cache-Control"] = "no-cache"
            return response
        # A Python-only checkout still has a functional UI before npm build.
        return render_template("index.html", groq_models=config.GROQ_MODELS)

    @app.get("/classic")
    def classic():
        return render_template("index.html", groq_models=config.GROQ_MODELS)
