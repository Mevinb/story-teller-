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
├── app.py                  # Flask REST API + SSE endpoints
├── web_ui.py               # React build serving + classic fallback
├── frontend/               # React + TypeScript views, Vite build, browser tests
├── main.py                 # CLI entry point
├── config.py               # Central configuration and environment defaults
├── start.sh                # Gunicorn production launch script
├── agents/                 # Architect, Planner, Writer, Consistency, Editor, Voice, Pacing
├── pipeline/               # Orchestrator, Gemini combiner, exporter, world bible
├── memory/                 # State manager, retriever, vector store, evolution engine
├── models/                 # LlamaCPP, Groq, Gemini, OpenRouter wrappers
├── static/                 # Compiled React assets + classic JavaScript/CSS
├── templates/              # Classic Jinja2 interface (/classic)
└── projects/               # Generated story projects (gitignored)
```

## Requirements

- Python 3.9+
- `pip` / `venv`
- Node.js 22.12+ and npm to build the React interface
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
bash scripts/build_frontend.sh
```

The main interface now uses **React + TypeScript + Vite**, with separate components
for the library, story details, premise, generation, manual scenes, reader,
polishing, story bible, activity, vision, and settings. Flask still serves the
APIs and existing Python story engine. Existing project files need no migration.

`./start.sh` rebuilds changed frontend sources and serves everything at
`http://localhost:5000`. The build is also available explicitly through
`bash scripts/build_frontend.sh`. Node is not a second production server.
The previous interface remains available at `/classic`; a Python-only checkout
without compiled assets falls back to it automatically.

For frontend development, run the Flask app, then `npm --prefix frontend run dev`
and open `http://localhost:5173/static/app/`. API requests are proxied to Flask.
Set `STORY_API_URL` when using another backend port.

See [frontend architecture and validation](docs/FRONTEND_MIGRATION.md) for the
component map, test commands, and migration boundaries.

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
| `GROQ_MODEL` | `qwen/qwen3.8-27b` | Groq model ID |
| `GROQ_TPM_LIMIT` | `6000` | Set to the selected model's actual organization TPM limit in Groq Console |
| `GEMINI_API_KEY` | — | Enables Combine & Polish |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model for polishing |
| `OPENROUTER_API_KEY` | — | Enables OpenRouter backend |
| `EMBEDDING_LOCAL_FILES_ONLY` | `true` | Keep embedding model offline; set `false` for first-run download |
| `FLASK_DEBUG` | `false` | Never set `true` in production |

See [`.env.example`](.env.example) for the full list of options.

### Quota-aware Groq generation

Qwen remains the prose writer. When the configured Gemini model passes a structured
health check, it handles architecture, scene planning, continuity, editing and
narrative evolution. A working local GGUF takes precedence when hybrid routing is
enabled. Support-provider outages open a short circuit breaker; necessary work
falls back through the same Groq admission path.

Every Groq request, including streaming and JSON repairs, reserves complete input
and output allowance atomically in `runtime/quotas.sqlite3`. Successful and error
headers update the ledger. Normal and terminal streamed usage reconcile charges;
missing usage and possibly dispatched failures retain conservative charges.
Credentials share one workload budget. There is no account quota stacking.

Set `GROQ_RPM_LIMIT`, `GROQ_TPM_LIMIT`, `GROQ_RPD_LIMIT`, `GROQ_TPD_LIMIT` to the
actual Console ceilings; optional `GROQ_ITPM_LIMIT`/`GROQ_OTPM_LIMIT` use zero when
not applicable. Header ceilings can lower configured limits. Defaults keep 20%
headroom and learn a more conservative input estimate after underestimation.
The terminal reports waiting, provider switches and token usage. Capacity and
external usage can still cause waits; this system cannot guarantee uninterrupted
cloud inference. `/api/keys/status` exposes safe quota telemetry.

Scene generation gets one draft plus `SCENE_REPAIR_CALLS=2` shared repair/polish
calls. Exact cited passages are patched first. Exhaustion preserves the latest
full draft and marks it for review. Known missing premise beats and character
introduction evidence are repaired before prose without a corrective model call.

Exact input/state checkpoints save completed model responses and interrupted
stream drafts. Normal and batch-manual jobs persist their target chapter with a
lease; `./start.sh` enables recovery after a restart and bounded retries after
outages. Completed chapters are never repeated. Interactive manual sessions keep
their existing WIP resume flow. Stop cancels persisted jobs. `/api/jobs` reports
job status. Private runtime data is excluded from version control. Restart the
running server and refresh the browser after upgrading.

Validation and observed limitations are recorded in
[the optimization plan](docs/RATE_LIMIT_OPTIMIZATION_PLAN.md).

### Optional rotating proxy for OpenCode

This repository includes a small localhost-only HTTP proxy that rotates through
upstream HTTP proxies once per client connection (including HTTPS `CONNECT`).
It is independent of the story application and can be used by OpenCode or any
client that honors standard proxy environment variables:

```bash
export ROTATING_PROXY_URLS='http://user:password@proxy1.example:3128,http://proxy2.example:3128'
python proxy_rotator.py --listen-port 8080
```

In the shell that launches OpenCode:

```bash
export HTTP_PROXY=http://127.0.0.1:8080
export HTTPS_PROXY=http://127.0.0.1:8080
opencode
```

Use `NO_PROXY=127.0.0.1,localhost` so local services are not sent through the
upstream pool. The proxy list must contain reachable HTTP/HTTPS proxy URLs;
this tool does not create proxy IPs or bypass provider limits.

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
| Groq errors | Check internet, `GROQ_API_KEY`, and the model's availability in Groq Console |
| Groq rate limit | Check the model's RPM, RPD, TPM and TPD limits in Groq Console; set `GROQ_TPM_LIMIT` accordingly. Keys in one organization share capacity. Daily exhaustion requires waiting for reset or increasing the plan limit. |
| Gemini combine fails | Set `GEMINI_API_KEY` |
| Port 5000 in use | Run `./start.sh` (auto-kills existing) or change `FLASK_PORT` |
| Slow local inference | Lower `LLAMA_N_GPU_LAYERS`, reduce batch/context size, or use smaller model |
| Semantic memory offline | Set `EMBEDDING_LOCAL_FILES_ONLY=false` for a first-run model download |

## License

MIT


## Scene image prompts

Completed automatic chapters, finalized manual chapters, and polished/whole stories
now produce up to three visual highlights using the configured text model. Open
**Reader** or a saved polished story to find **Scene image prompts** above the prose.
Copy one prompt, copy all, download a text file, or generate/regenerate prompts
for existing stories. Attach the named character references in your external image
tool. Story Teller generates prompt text only; character likeness depends on that
tool's reference support. Sensitive scenes receive non-explicit adaptations.

Prompt generation uses additional language-model tokens and completes before the
completion notification. Failures keep the story saved and offer a retry. Results
are cached by final prose under the project's `image_prompts/` directory; edited
prose requires new prompts, and renamed saved versions reuse matching results.
Very long stories use beginning, middle, and end excerpts for highlight selection.
