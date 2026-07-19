"""
Story Teller CLI — Phase 4 Developer Experience.

Provides command-line tools for project maintenance and diagnostics:

    python -m pipeline validate <project>
    python -m pipeline migrate <project>
    python -m pipeline world-bible <project>
    python -m pipeline export <project> [--format md|txt|epub|all]
    python -m pipeline stats <project>
    python -m pipeline cache-info <project>

Run without arguments to see help.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Ensure the project root is on the path when run as ``python -m pipeline``
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ok(msg: str) -> None:
    print(f"  \033[32m✓\033[0m {msg}")

def _warn(msg: str) -> None:
    print(f"  \033[33m⚠\033[0m {msg}")

def _err(msg: str) -> None:
    print(f"  \033[31m✗\033[0m {msg}")

def _header(msg: str) -> None:
    print(f"\n\033[1m{msg}\033[0m")


def _resolve_project_dir(project_name: str) -> str:
    import config
    project_dir = os.path.join(config.PROJECTS_DIR, project_name)
    if not os.path.isdir(project_dir):
        _err(f"Project not found: {project_dir}")
        sys.exit(1)
    return project_dir


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_validate(args) -> int:
    """
    Validate project state.json against expected schema fields.
    Returns 0 on success, 1 if errors found.
    """
    _header(f"Validating project: {args.project}")
    project_dir = _resolve_project_dir(args.project)
    state_path = os.path.join(project_dir, "state.json")

    if not os.path.exists(state_path):
        _err("state.json not found")
        return 1

    try:
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    except json.JSONDecodeError as e:
        _err(f"state.json is corrupt: {e}")
        return 1

    errors = 0
    warnings = 0

    # Required top-level keys
    for key in ("metadata", "characters", "plot"):
        if key not in state:
            _err(f"Missing required key: '{key}'")
            errors += 1
        else:
            _ok(f"Key '{key}' present")

    # Metadata fields
    meta = state.get("metadata", {})
    for field in ("title", "genre", "current_chapter", "total_scenes_written"):
        if field not in meta:
            _warn(f"metadata.{field} missing (non-critical)")
            warnings += 1
        else:
            _ok(f"metadata.{field} = {meta[field]!r}")

    # Premise cursor sanity
    cursor = meta.get("premise_step_completed", -1)
    premise = meta.get("premise", "")
    if premise:
        # Count steps (split on bullet patterns)
        import re
        steps = [s.strip() for s in re.split(r"\n\s*[-•*]\s+|\n\s*\d+\.\s+", premise) if s.strip()]
        if steps:
            if cursor < -1 or cursor > len(steps):
                _warn(f"premise_step_completed={cursor} out of range (0..{len(steps)-1})")
                warnings += 1
            else:
                _ok(f"Premise cursor: {cursor}/{len(steps)-1}")

    # Characters
    chars = state.get("characters", {})
    _ok(f"Characters: {len(chars)} registered")
    for name, info in list(chars.items())[:5]:
        if not isinstance(info, dict):
            _err(f"Character '{name}' has invalid info (not a dict)")
            errors += 1

    # Chapter files vs state pointer
    chapters_dir = os.path.join(project_dir, "chapters")
    import re as _re
    _ch_re = _re.compile(r"^chapter_(\d{3,})\.md$")
    disk_chapters = []
    if os.path.exists(chapters_dir):
        for fname in os.listdir(chapters_dir):
            m = _ch_re.match(fname)
            if m:
                disk_chapters.append(int(m.group(1)))
    disk_max = max(disk_chapters) if disk_chapters else 0
    state_max = int(meta.get("current_chapter", 0))

    if disk_max != state_max:
        _warn(f"Chapter count mismatch: disk={disk_max}, state={state_max}")
        warnings += 1
    else:
        _ok(f"Chapter count consistent: {state_max} chapters")

    # WIP files
    wip_files = []
    if os.path.exists(chapters_dir):
        wip_files = [f for f in os.listdir(chapters_dir) if f.startswith(".wip_")]
    if wip_files:
        _warn(f"WIP checkpoint(s) found: {wip_files} — chapter generation was interrupted")
        warnings += 1
    else:
        _ok("No dangling WIP checkpoints")

    _header("Summary")
    if errors:
        _err(f"{errors} error(s), {warnings} warning(s) — project may not function correctly")
        return 1
    elif warnings:
        _warn(f"0 errors, {warnings} warning(s) — project usable but check warnings")
        return 0
    else:
        _ok(f"Project is healthy ({len(chars)} characters, {state_max} chapters)")
        return 0


def cmd_migrate(args) -> int:
    """
    Migrate state.json to ensure all current schema fields are present.
    Adds missing fields with safe defaults; never removes existing data.
    """
    _header(f"Migrating project: {args.project}")
    project_dir = _resolve_project_dir(args.project)
    state_path = os.path.join(project_dir, "state.json")

    if not os.path.exists(state_path):
        _err("state.json not found")
        return 1

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    changed = []

    # Ensure top-level keys
    for key, default in (
        ("metadata", {}),
        ("characters", {}),
        ("plot", {}),
        ("unresolved_threads", []),
        ("world_building", {}),
        ("narrative_state", {}),
    ):
        if key not in state:
            state[key] = default
            changed.append(f"Added missing top-level key '{key}'")

    # Ensure metadata fields
    meta_defaults = {
        "premise_step_completed": -1,
        "total_scenes_written": 0,
        "current_chapter": 0,
    }
    for field, default in meta_defaults.items():
        if field not in state["metadata"]:
            state["metadata"][field] = default
            changed.append(f"Added metadata.{field} = {default!r}")

    if not changed:
        _ok("Nothing to migrate — state.json is up to date")
        return 0

    # Atomic write
    import shutil
    tmp_path = state_path + ".migrate.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    shutil.move(tmp_path, state_path)

    for msg in changed:
        _ok(msg)
    print(f"\n  Migrated {len(changed)} field(s).")
    return 0


def cmd_world_bible(args) -> int:
    """Generate or update the World Bible for a project."""
    _header(f"Generating World Bible for: {args.project}")
    project_dir = _resolve_project_dir(args.project)

    from pipeline.world_bible import WorldBibleGenerator

    def _cb(msg: str) -> None:
        print(f"  {msg}")

    gen = WorldBibleGenerator(project_dir, progress_callback=_cb)
    bible = gen.generate(max_chapters=args.max_chapters)

    out_md = os.path.join(project_dir, "world_bible.md")
    out_json = os.path.join(project_dir, "world_bible.json")

    _ok(f"Locations : {len(bible.locations)}")
    _ok(f"Factions  : {len(bible.factions)}")
    _ok(f"Objects   : {len(bible.objects)}")
    _ok(f"Rules     : {len(bible.rules)}")
    _ok(f"Events    : {len(bible.events)}")
    print(f"\n  Saved: {out_md}")
    print(f"  Saved: {out_json}")
    return 0


def cmd_export(args) -> int:
    """Export a story project to distributable formats."""
    _header(f"Exporting project: {args.project} (format={args.format})")
    project_dir = _resolve_project_dir(args.project)

    from pipeline.exporter import StoryExporter, epub_available

    exporter = StoryExporter(project_dir)

    fmt = args.format.lower()
    if fmt == "md" or fmt == "markdown":
        path = exporter.export_markdown()
        _ok(f"Markdown: {path}")
    elif fmt == "txt":
        path = exporter.export_txt()
        _ok(f"TXT: {path}")
    elif fmt == "epub":
        if not epub_available():
            _err("ebooklib not installed. Run: pip install ebooklib")
            return 1
        path = exporter.export_epub()
        _ok(f"EPUB: {path}")
    elif fmt == "all":
        results = exporter.export_all(include_epub=epub_available())
        for fmt_name, out_path in results.items():
            _ok(f"{fmt_name.upper()}: {out_path}")
    else:
        _err(f"Unknown format: {fmt!r}. Use: md, txt, epub, all")
        return 1
    return 0


def cmd_stats(args) -> int:
    """Show project statistics."""
    _header(f"Stats for: {args.project}")
    project_dir = _resolve_project_dir(args.project)
    state_path = os.path.join(project_dir, "state.json")

    if not os.path.exists(state_path):
        _err("state.json not found")
        return 1

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    meta = state.get("metadata", {})
    chars = state.get("characters", {})
    plot = state.get("plot", {})
    threads = state.get("unresolved_threads", [])

    print(f"  Title        : {meta.get('title', '?')}")
    print(f"  Genre        : {meta.get('genre', '?')}")
    print(f"  Chapters     : {meta.get('current_chapter', 0)}")
    print(f"  Scenes       : {meta.get('total_scenes_written', 0)}")
    print(f"  Characters   : {len(chars)}")
    print(f"  Events       : {len(plot.get('major_events', []))}")
    print(f"  Open threads : {len(threads)}")

    # Word counts from files
    chapters_dir = os.path.join(project_dir, "chapters")
    import re as _re
    _ch_re = _re.compile(r"^chapter_(\d{3,})\.md$")
    total_words = 0
    if os.path.exists(chapters_dir):
        for fname in os.listdir(chapters_dir):
            if _ch_re.match(fname):
                try:
                    with open(os.path.join(chapters_dir, fname), encoding="utf-8") as f:
                        total_words += len(f.read().split())
                except OSError:
                    pass
    print(f"  Total words  : {total_words:,}")

    # Vector store
    vs_path = os.path.join(project_dir, "vector_store")
    if os.path.exists(vs_path):
        _ok(f"Vector store present at {vs_path}")
    else:
        _warn("No vector store found")

    return 0


def cmd_cache_info(args) -> int:
    """Show vector store embedding cache diagnostics for a project."""
    _header(f"Cache info for: {args.project}")
    project_dir = _resolve_project_dir(args.project)

    try:
        from memory.vector_store import VectorStore
        vs = VectorStore(project_dir)
        # Only load index metadata — don't initialize the full model
        info = vs.cache_info()
        print(f"  Cache size   : {info['cache_size']} / {info['max_size']}")
        print(f"  Cache hits   : {info['hits']}")
        print(f"  Cache misses : {info['misses']}")
        print(f"  Hit rate     : {info['hit_rate']:.1%}")
        stats = vs.get_stats()
        print(f"  Vectors      : {stats['total_vectors']}")
        print(f"  Chunks       : {stats['total_chunks']}")
    except Exception as exc:
        _err(f"Could not load vector store: {exc}")
        return 1
    return 0


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Story Teller CLI — project maintenance and diagnostics",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # validate
    p_val = sub.add_parser("validate", help="Validate project state.json")
    p_val.add_argument("project", help="Project name")

    # migrate
    p_mig = sub.add_parser("migrate", help="Migrate state.json to latest schema")
    p_mig.add_argument("project", help="Project name")

    # world-bible
    p_wb = sub.add_parser("world-bible", help="Generate World Bible from chapters")
    p_wb.add_argument("project", help="Project name")
    p_wb.add_argument("--max-chapters", type=int, default=999, metavar="N")

    # export
    p_exp = sub.add_parser("export", help="Export story to distributable format")
    p_exp.add_argument("project", help="Project name")
    p_exp.add_argument(
        "--format", default="md",
        choices=["md", "markdown", "txt", "epub", "all"],
        help="Output format (default: md)",
    )

    # stats
    p_stats = sub.add_parser("stats", help="Show project statistics")
    p_stats.add_argument("project", help="Project name")

    # cache-info
    p_cache = sub.add_parser("cache-info", help="Show vector store cache diagnostics")
    p_cache.add_argument("project", help="Project name")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    dispatch = {
        "validate": cmd_validate,
        "migrate": cmd_migrate,
        "world-bible": cmd_world_bible,
        "export": cmd_export,
        "stats": cmd_stats,
        "cache-info": cmd_cache_info,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
