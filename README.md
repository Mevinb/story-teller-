# Story Teller

Story Teller is a multi-agent fiction generation system with:

- a Flask web app (live streaming generation, project manager, reader, manual scene flow)
- a Rich-powered CLI
- structured story memory (`state.json`) plus semantic retrieval (FAISS + sentence-transformers)

It is designed for long-form chapter writing with consistency checks, editable project state, and reproducible project files.

## What it does

- Orchestrates specialized agents: architect, planner, writer, consistency engine, editor
- Generates chapters automatically or scene-by-scene in interactive manual mode
- Streams generation events over SSE to the web UI
- Tracks story state and history on disk
- Stores and retrieves semantic memory chunks for continuity
- Supports local GGUF inference (`llama.cpp`) and optional Groq backend
- Supports Gemini-powered "Combine & Polish" across all chapters

## Project structure

```text
story teller/
├── app.py                  # Flask app + REST API + SSE
├── main.py                 # CLI entry point
├── config.py               # Central configuration and defaults
├── agents/                 # Architect, planner, writer, consistency, editor
├── pipeline/               # Orchestrator + Gemini combiner
├── memory/                 # State manager, retriever, vector store, evolution tools
├── models/                 # LlamaCPP + Groq wrappers
├── static/ templates/      # Web UI assets
└── projects/               # Generated story projects
```

## Requirements

- Python 3.9+
- `pip`
- For local generation: at least one `.gguf` model file
- Optional: `GROQ_API_KEY` for Groq backend
- Optional: `GEMINI_API_KEY` for Combine & Polish

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configure

Create a `.env` file (or copy from `.env.example`) and set what you need:

```bash
cp .env.example .env
```

Common variables:

| Variable | Purpose |
| --- | --- |
| `LLAMA_MODELS_DIR` | Directory scanned for `.gguf` models (default: `./models`) |
| `LLAMA_MODEL_PATH` | Active local `.gguf` model path |
| `LLAMA_PROMPT_TEMPLATE` | Prompt format (`mistral`, `llama3`, `chatml`) |
| `LLAMA_N_GPU_LAYERS` | GPU offload layers for llama.cpp |
| `LLAMA_N_BATCH` | llama.cpp batch size |
| `LLAMA_N_THREADS` | CPU threads for local inference |
| `LOCAL_NUM_CTX` | Local context window |
| `USE_CLOUD_MODEL` | `true` to prefer Groq in local/cloud hybrid mode |
| `GROQ_API_KEY` | Enables Groq API backend |
| `GROQ_MODEL` | Groq model id |
| `GEMINI_API_KEY` | Enables Combine & Polish |
| `GEMINI_MODEL` | Gemini model for combining/polishing |

## Local model setup (llama.cpp)

Install `llama-cpp-python` and provide a GGUF file under `models/` (or point `LLAMA_MODEL_PATH` elsewhere).

CPU install:

```bash
pip install llama-cpp-python
```

CUDA build example:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --force-reinstall llama-cpp-python
```

## Run the app

Web UI (default `http://127.0.0.1:5000`):

```bash
python app.py
```

or:

```bash
python main.py serve
```

Helper script:

```bash
./start.sh
```

`start.sh` expects `./venv/bin/python` and attempts to stop anything already bound to port 5000.

## CLI usage

Create a project:

```bash
python main.py new
```

Generate chapters:

```bash
python main.py generate <project_name> --chapters 1 --pacing moderate
```

Other commands:

```bash
python main.py list
python main.py status <project_name>
python main.py read <project_name> [chapter_number]
python main.py delete-chapters <project_name> <from_chapter> --yes
python main.py serve
```

## Web UI workflow

1. Create a project (title, genre, premise, setting, themes, characters)
2. Pick a backend/model from the header model selector
3. Generate chapters (1-10 at a time), watch live token/agent output
4. Use **Manual Page** for interactive scene-by-scene chapter writing
5. Read chapters, edit/resume a chapter, delete from chapter N onward
6. Inspect and edit story state in **State**
7. Optionally run **Combine & Polish** (Gemini) and download outputs

## API overview

Main route groups:

- Models: `/api/models`, `/api/models/switch`
- Projects: `/api/projects`, `/api/project/create`, `/api/project/<name>`
- State: `/api/project/<name>/state` (GET, PUT)
- Chapters: list/read/delete/export
- Generation: auto, manual, interactive manual session, SSE stream, cancel
- Combine: start, SSE progress, download artifacts

Notable SSE endpoints:

- `/api/project/<name>/generate/stream`
- `/api/project/<name>/combine/stream`

## Project data layout

Each project is stored under `projects/<project_name>/`:

```text
projects/<name>/
├── state.json
├── state_history.jsonl
├── chapter_###_wip.json          # checkpoints during generation
├── chapters/
│   └── chapter_###.md
├── logs/
│   ├── chapter_###.json
│   └── chapter_###_trace.jsonl
├── vector_index/
│   ├── index.faiss
│   ├── metadata.json
│   └── texts.json
├── combined_original.md          # after Combine & Polish
├── combined_polished.md          # after Combine & Polish
└── story_analysis.md             # after Combine & Polish
```

## Troubleshooting

- **`GGUF model file not found`**: set `LLAMA_MODEL_PATH` correctly or place model in `LLAMA_MODELS_DIR`.
- **Groq unavailable**: verify internet, API key, and `GROQ_MODEL`.
- **Gemini combine fails**: ensure `GEMINI_API_KEY` is set.
- **Port 5000 already in use**: stop the process using it, or use `start.sh` to auto-stop it.
- **Slow local inference**: reduce context/batch, lower model size, or tune `LLAMA_N_GPU_LAYERS`.

## Notes

- Default mode is local generation unless Groq is explicitly selected/enabled.
- Project reset keeps user metadata and characters, and clears generated chapters/memory artifacts.
- Deleting chapters from N rewinds state, vector memory, WIP checkpoints, and related logs.
