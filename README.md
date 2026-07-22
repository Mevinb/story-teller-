# Story Teller

**Story Teller** is a multi-agent AI fiction generation system for writing long-form stories chapter by chapter — with a live-streaming web UI, a CLI, semantic story memory, and pluggable LLM backends.

## Features

- **Multi-agent pipeline** — Architect, Planner, Writer, Consistency Engine, Editor agents collaborate on each chapter
- **Live streaming** — Generation events stream over SSE to the web UI in real time
- **Story memory** — Semantic retrieval (FAISS + sentence-transformers) keeps characters and plot consistent across chapters
- **Multiple backends** — Local GGUF inference (`llama.cpp`), Groq, Gemini, or OpenRouter
- **Manual mode** — Write scenes interactively, scene by scene, with AI assistance
- **Combine & Polish** — Gemini-powered full-story combine, analysis, and polish pass
- **State management** — Editable `state.json` tracks all characters, premise steps, and chapter history
- **CLI** — Full-featured command-line interface for headless generation

## Project Structure

```text
story-teller/
├── app.py                  # Flask app + REST API + SSE endpoints
├── main.py                 # CLI entry point
├── config.py               # Central configuration and environment defaults
├── start.sh                # Gunicorn production launch script
├── agents/                 # Architect, Planner, Writer, Consistency, Editor, Voice, Pacing
├── pipeline/               # Orchestrator, Gemini combiner, exporter, world bible
├── memory/                 # State manager, retriever, vector store, evolution engine
├── models/                 # LlamaCPP, Groq, Gemini, OpenRouter wrappers
├── static/                 # Web UI JavaScript and CSS
├── templates/              # Jinja2 HTML templates
└── projects/               # Generated story projects (gitignored)
```

## Requirements

- Python 3.9+
- `pip` / `venv`
- For local generation: a `.gguf` model file (e.g. Mistral, LLaMA 3)
- Optional: `GROQ_API_KEY` for Groq cloud backend
- Optional: `GEMINI_API_KEY` for Combine & Polish

## Install

```bash
git clone <repo-url>
cd story-teller
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
# Edit .env and fill in your API keys and model paths
```

Key environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `BACKEND_MODE` | `local` | Active backend: `local` \| `groq` \| `gemini` \| `openrouter` \| `hybrid` |
| `LLAMA_MODEL_PATH` | — | Path to your `.gguf` model file |
| `LLAMA_PROMPT_TEMPLATE` | `mistral` | Prompt format: `mistral` \| `llama3` \| `chatml` |
| `LLAMA_N_GPU_LAYERS` | `20` | GPU offload layers for llama.cpp |
| `GROQ_API_KEY` | — | Enables Groq API backend |
| `GROQ_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq model ID |
| `GEMINI_API_KEY` | — | Enables Combine & Polish |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for polishing |
| `OPENROUTER_API_KEY` | — | Enables OpenRouter backend |
| `EMBEDDING_LOCAL_FILES_ONLY` | `true` | Keep embedding model offline; set `false` for first-run download |
| `FLASK_DEBUG` | `false` | Never set `true` in production |

See [`.env.example`](.env.example) for the full list of options.

## Local Model Setup (llama.cpp)

Place a `.gguf` file under `models/` and point `LLAMA_MODEL_PATH` to it.

CPU install:

```bash
pip install llama-cpp-python
```

CUDA (GPU) build:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --force-reinstall llama-cpp-python
```

## Running

**Web UI** (default at `http://127.0.0.1:5000`):

```bash
# Development
python app.py

# Production (Gunicorn)
./start.sh
```

**CLI**:

```bash
python main.py new                                          # Create a project
python main.py generate <project_name> --chapters 1        # Generate chapters
python main.py list                                         # List all projects
python main.py status <project_name>                        # Project status
python main.py read <project_name> [chapter_number]        # Read a chapter
python main.py delete-chapters <project_name> <from> --yes # Delete from chapter N
python main.py serve                                        # Start web server
```

## Web UI Workflow

1. Create a project — set title, genre, premise, setting, themes, and characters
2. Select a backend/model from the header model selector
3. Generate chapters (1–10 at a time) and watch live token/agent output stream
4. Use **Manual Page** for interactive scene-by-scene chapter writing
5. Read, edit, or resume chapters; delete from chapter N to rewind state
6. Inspect and edit story state in the **State** panel
7. Run **Combine & Polish** (requires Gemini API key) and download the polished story

## API Overview

| Group | Routes |
|---|---|
| Models | `GET /api/models` · `POST /api/models/switch` |
| Projects | `GET /api/projects` · `POST /api/project/create` · `GET/DELETE /api/project/<name>` |
| State | `GET/PUT /api/project/<name>/state` |
| Chapters | List · Read · Update · Delete |
| Generation | Auto · Manual · Interactive manual session · SSE stream · Cancel |
| Combine | Start · SSE progress · Download artifacts |

Notable SSE endpoints:
- `GET /api/project/<name>/generate/stream`
- `GET /api/project/<name>/combine/stream`

## Project Data Layout

Each project lives under `projects/<name>/` (gitignored):

```text
projects/<name>/
├── state.json                    # Authoritative story state
├── state_history.jsonl           # Full state change history
├── chapters/
│   └── chapter_NNN.md
├── logs/
│   ├── chapter_NNN.json
│   └── chapter_NNN_trace.jsonl
├── vector_index/                 # FAISS semantic memory
├── chapter_NNN_wip.json          # In-progress checkpoints
├── combined_original.md          # Raw combined story
├── combined_polished.md          # Gemini-polished story
└── story_analysis.md             # Gemini story analysis
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `GGUF model file not found` | Set `LLAMA_MODEL_PATH` or place model in `LLAMA_MODELS_DIR` |
| Groq errors | Check internet, `GROQ_API_KEY`, and `GROQ_MODEL` |
| Gemini combine fails | Set `GEMINI_API_KEY` |
| Port 5000 in use | Run `./start.sh` (auto-kills existing) or change `FLASK_PORT` |
| Slow local inference | Lower `LLAMA_N_GPU_LAYERS`, reduce batch/context size, or use smaller model |
| Semantic memory offline | Set `EMBEDDING_LOCAL_FILES_ONLY=false` for a first-run model download |

## License

MIT
