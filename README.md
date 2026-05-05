# Story Teller

Story Teller is a multi-agent story generation pipeline with both a web UI and a CLI. It orchestrates a set of specialized agents (architect, planner, writer, consistency checker, editor) to generate long-form fiction with structured memory and semantic retrieval.

## Features

- Multi-agent pipeline: Story Architect, Scene Planner, Scene Writer, Consistency Engine, Editor.
- Transition-aware consistency checks that capture valid character evolution as state updates.
- Explicit execution graphs for deterministic orchestration (chapter graph + scene graph).
- Local-first generation via llama.cpp GGUF models with optional Groq cloud acceleration.
- Structured memory (JSON state) plus semantic retrieval (FAISS vector index).
- Immutable state transitions with versioned state snapshots and change history.
- Web UI with live generation updates (SSE), logs, reader, and state inspector.
- CLI for project creation, generation, status, and reading.
- Project export, reset, and delete tools.

## Requirements

- Python 3.x with pip
- A local `.gguf` model file
- `llama-cpp-python` installed with CUDA support for GPU acceleration
- Optional: Groq API key for cloud generation

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure environment variables:

   ```bash
   cp .env.example .env
   # edit .env
   ```

3. Install llama.cpp support and place a GGUF model under `models/`:

   ```bash
   # CPU install:
   pip install llama-cpp-python

   # CUDA build for RTX GPUs:
   CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --force-reinstall llama-cpp-python
   ```

   Recommended starting point for an RTX 4050 with ~6GB VRAM: a 7B or 8B `Q4_K_M` GGUF model, such as Mistral 7B Instruct or LLaMA 3 8B Instruct.

## Running

### Web UI

Run the Flask server and open the UI:

```bash
python app.py
```

Or via the CLI entry point:

```bash
python main.py serve
```

The UI runs at http://127.0.0.1:5000 by default.

If you use the provided script (expects a venv at ./venv):

```bash
./start.sh
```

### CLI

Create a project:

```bash
python main.py new
```

Generate chapters:

```bash
python main.py generate <project_name> --chapters 1 --pacing moderate
```

Check status:

```bash
python main.py status <project_name>
```

Read chapters:

```bash
python main.py read <project_name> [chapter_number]
```

List projects:

```bash
python main.py list
```

## Configuration

Environment variables are loaded from .env via python-dotenv. Key options:

- GROQ_API_KEY: enables Groq cloud model usage when set
- USE_CLOUD_MODEL: set to true to prefer Groq (fallback to local on failure)
- GROQ_MODEL: default llama-3.3-70b-versatile
- LLAMA_MODELS_DIR: directory scanned for `.gguf` files
- LLAMA_MODEL_PATH: active `.gguf` model path
- LLAMA_PROMPT_TEMPLATE: `mistral` or `llama3`
- LLAMA_N_GPU_LAYERS: default 20; raise gradually for more GPU offload
- LLAMA_N_BATCH: default 512
- LLAMA_F16_KV: default true
- LOCAL_NUM_CTX: default 4096

Generation parameters (scene counts, word targets, retries, etc.) are defined in config.py.
Determinism controls are also in config.py, including:

- `AGENT_TEMPERATURES` (Planner 0.2, Writer 0.7, Critic 0.1, Editor 0.2)
- `MAX_SCENE_ITERATIONS`
- `MAX_PIPELINE_STEPS`
- `MAX_TOKEN_BUDGET`
- `SSE_QUEUE_MAXSIZE`

## Project Data Layout

Projects are stored under the projects/ directory:

- projects/<name>/state.json: structured story state
- projects/<name>/state_history.jsonl: change log
- projects/<name>/chapters/chapter_XXX.md: generated chapters
- projects/<name>/logs/: per-chapter generation logs
- projects/<name>/vector_index/: FAISS index and metadata

## Web API Endpoints

- GET /api/models: list available GGUF models
- POST /api/models/switch: switch active GGUF model
- GET /api/projects: list projects
- POST /api/project/create: create project
- GET /api/project/<name>: project info and state
- GET /api/project/<name>/state: get state
- PUT /api/project/<name>/state: update state
- GET /api/project/<name>/chapters: list chapters
- GET /api/project/<name>/chapter/<num>: read a chapter
- POST /api/project/<name>/generate: start generation
- GET /api/project/<name>/generate/stream: SSE stream for live updates
- POST /api/project/<name>/generate/cancel: cancel generation
- POST /api/project/<name>/delete: delete project
- POST /api/project/<name>/reset: reset generated content
- GET /api/project/<name>/export: download full story

## Notes

- Local-only mode is the default. To use Groq, set GROQ_API_KEY and USE_CLOUD_MODEL=true.
- If a Groq request is blocked by moderation or rate limits, the pipeline falls back to the local model.
- Resetting a project clears generated content (chapters, logs, vectors) but keeps user-entered metadata.
- SSE payloads now include a normalized schema: `type`, `agent`, `content`, `timestamp`, `payload` (and compatibility keys `event`, `data`).
