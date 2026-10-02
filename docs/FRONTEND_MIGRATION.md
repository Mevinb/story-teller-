# React frontend migration

The primary UI is React 19 with strict TypeScript and Vite. Flask serves the built
HTML and hashed assets, using the existing API, SSE, model adapters, storage,
quota scheduler, and recovery system. This phase changes the frontend and adds
one read-only activity endpoint; it does not replace Flask or introduce a
separate generation worker or database migration.

## Source map

- `frontend/src/App.tsx`: application shell and URL-backed workspace navigation.
- `frontend/src/views/`: independent library, create/edit, premise, generation,
  manual scene, reader, polishing, bible, activity, vision, and settings views.
- `frontend/src/api.ts` and `types.ts`: shared HTTP error handling, encoded project
  paths, and response types. Settings submit only fields edited by the user.
- `frontend/src/stream.ts`: generation and polishing streams, bounded logs,
  token deduplication, native EventSource reconnection, and terminal states.
- `frontend/src/components/`: dialogs, model picker, prose, and external scene
  image prompts, including polling and clipboard/download actions.
- `web_ui.py`: same-origin serving and `/classic` fallback.

Views stay mounted after visiting them so drafts and progress survive tab
changes. Streams belong to the selected project workspace, not a particular
screen. `/api/project/<name>/activity` discovers work already running after a
page reload; reconnecting never sends a second generation request. Switching
projects closes the old browser subscriptions without stopping its server job.

The classic JS and templates remain isolated at `/classic`; the React interface
does not load them or depend on their global variables/DOM IDs. Existing stories
and settings retain their file formats.

## Build and run

```bash
bash scripts/build_frontend.sh
./start.sh
```

The launcher rebuilds when frontend source/configuration is newer than the built
HTML. The npm lockfile records dependency versions. `static/app/` and
`frontend/node_modules/` are generated, ignored files. Normal UI use needs only
the Python server after the build. The classic UI also works without Node.

```bash
# Development with hot reload; run Flask separately on port 5000.
npm --prefix frontend run dev
# Open http://localhost:5173/static/app/
```

## Validation

```bash
npm --prefix frontend run build
venv/bin/python -m pytest -q
npm --prefix frontend exec -- playwright install chromium
npm --prefix frontend test
```

Browser tests start an isolated Flask server on port 5017. Its stories and `.env`
live under a temporary directory, and automatic job recovery is disabled. Real
Flask tests cover creating, saving, reloading, and reading stories. Provider
responses and streams are simulated for model errors, generation, reconnects,
manual editing, settings, and polishing. No cloud generation or GPU inference is
claimed by these checks. Browser screenshots cover desktop and mobile layouts.

The migration also corrects chapter selection after saving, version labels from
the API, preservation of extra character memory during metadata edits, failed
request reporting, and branch suggestions (which can be sent to the manual
studio instead of reporting an unimplemented apply action).

## Remaining architecture work

Generation still runs in the existing Python process with durable SQLite job
records. A separate worker and a backend route/service split are independent
future changes. FastAPI/PostgreSQL are not prerequisites for this local app.
Cross-project browser draft persistence and multi-user access are not added by
this migration.
