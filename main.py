#!/usr/bin/env python3
"""
Story Teller — Multi-Agent Story Generation Pipeline
CLI entry point with Rich-powered terminal interface.
"""
import sys
import os
import json
import argparse
import logging


# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.panel import Panel

from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.syntax import Syntax
from rich.markdown import Markdown


import config
from pipeline.orchestrator import PipelineOrchestrator, normalize_project_name

console = Console()

# ─── Logging Setup ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("storyteller")


def print_banner():
    banner = """
[bold magenta]╔══════════════════════════════════════════════╗
║         🖊️  STORY TELLER PIPELINE  🖊️          ║
║     Multi-Agent Narrative Generation Engine   ║
╚══════════════════════════════════════════════╝[/bold magenta]
"""
    console.print(banner)


def progress_handler(event, data=None, **kwargs):
    """Handle pipeline progress events for CLI output."""
    if event == "status":
        console.print(f"  [dim]→ {data}[/dim]")
    elif event == "agent_active":
        agent = data.get("agent", "")
        step = data.get("step", "")
        console.print(f"  [cyan]⚙ {agent}[/cyan] — {step}")
    elif event == "chapter_start":
        console.print(f"\n[bold green]▶ Starting Chapter {data['chapter']}[/bold green]")
    elif event == "chapter_planned":
        plan = data.get("plan", {})
        console.print(
            f"  [yellow]📋 Plan:[/yellow] {plan.get('chapter_title', '?')} "
            f"| Tone: {plan.get('tone', '?')} "
            f"| Scenes: {plan.get('estimated_scenes', '?')}"
        )
    elif event == "scenes_planned":
        types = data.get("types", [])
        console.print(
            f"  [yellow]🎬 Scenes:[/yellow] {' → '.join(types)}"
        )
    elif event == "scene_start":
        console.print(
            f"\n  [bold]Scene {data['scene']}/{data['total']}[/bold] "
            f"[dim]({data['type']})[/dim]"
        )
    elif event == "scene_written":
        provider = data.get("provider", "?")
        icon = "☁️" if provider == "groq" else "💻"
        console.print(
            f"    {icon} Written: {data['words']} words via {provider}"
        )
    elif event == "scene_complete":
        console.print(
            f"    [green]✓[/green] Scene {data['scene']} complete "
            f"({data['words']} words)"
        )
    elif event == "chapter_complete":
        console.print(Panel(
            f"[bold green]Chapter {data['chapter_number']} Complete![/bold green]\n"
            f"Title: {data['chapter_title']}\n"
            f"Words: {data['total_words']}\n"
            f"Scenes: {data['scenes_count']}\n"
            f"File: {data['file']}",
            title="✅ Done",
            border_style="green",
        ))
    elif event == "error":
        console.print(
            f"  [bold red]✗ Error:[/bold red] {data.get('error', 'Unknown')}"
        )


# ─── Commands ─────────────────────────────────────────────────────

def cmd_new(args):
    """Create a new story project interactively."""
    print_banner()
    console.print("[bold]Create a new story project[/bold]\n")

    title = Prompt.ask("[cyan]Story title[/cyan]")
    genre = Prompt.ask("[cyan]Genre[/cyan]", default="dark fantasy")
    premise = Prompt.ask("[cyan]Premise[/cyan] (describe the core idea)")
    setting = Prompt.ask("[cyan]Setting[/cyan]", default="")
    themes = Prompt.ask(
        "[cyan]Themes[/cyan] (comma-separated)", default=""
    )
    themes_list = [t.strip() for t in themes.split(",") if t.strip()]

    # Characters
    characters = {}
    console.print("\n[bold]Define characters[/bold] (enter empty name to finish)")
    while True:
        name = Prompt.ask("  Character name", default="")
        if not name:
            break
        desc = Prompt.ask(f"    {name}'s description")
        traits = Prompt.ask(
            f"    {name}'s traits (comma-separated)", default=""
        )
        traits_list = [t.strip() for t in traits.split(",") if t.strip()]
        characters[name] = {
            "description": desc,
            "traits": traits_list,
        }
        console.print(f"    [green]✓ Added {name}[/green]")

    # Create project
    project_name = normalize_project_name(title)
    try:
        pipeline = PipelineOrchestrator(project_name, progress_handler)
        pipeline.create_project(
            title=title, genre=genre, premise=premise,
            characters=characters, themes=themes_list, setting=setting,
        )
        console.print(Panel(
            f"[bold green]Project '{title}' created![/bold green]\n"
            f"Directory: {pipeline.project_dir}\n\n"
            f"Generate your first chapter:\n"
            f"  [cyan]python main.py generate {project_name}[/cyan]",
            title="✅ Ready",
            border_style="green",
        ))
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def cmd_generate(args):
    """Generate the next chapter(s)."""
    print_banner()

    project = args.project
    chapters = args.chapters
    pacing = args.pacing

    try:
        pipeline = PipelineOrchestrator(project, progress_handler)
        pipeline.load_project()
        info = pipeline.get_project_info()

        console.print(Panel(
            f"[bold]{info['title']}[/bold] ({info['genre']})\n"
            f"Chapters written: {info['chapters_written']}\n"
            f"Characters: {', '.join(info['characters'])}\n"
            f"Vector chunks: {info['vector_chunks']}",
            title=f"📖 {project}",
            border_style="blue",
        ))

        for i in range(chapters):
            if chapters > 1:
                console.print(
                    f"\n[bold]Generating chapter {i + 1} of {chapters}...[/bold]"
                )
            result = pipeline.generate_chapter(pacing=pacing)
            if result.get("status") == "cancelled":
                console.print("[yellow]Generation cancelled.[/yellow]")
                break

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        logger.exception("Generation failed")
        sys.exit(1)


def cmd_status(args):
    """Show project status."""
    print_banner()
    try:
        pipeline = PipelineOrchestrator(args.project, lambda **kw: None)
        pipeline.load_project()
        info = pipeline.get_project_info()

        table = Table(title=f"📖 {info['title']}", border_style="blue")
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Genre", info["genre"])
        table.add_row("Current Chapter", str(info["current_chapter"]))
        table.add_row("Chapters Written", str(info["chapters_written"]))
        table.add_row("Total Scenes", str(info["total_scenes"]))
        table.add_row("Vector Chunks", str(info["vector_chunks"]))
        table.add_row("Characters", ", ".join(info["characters"]))

        console.print(table)

        if args.state:
            console.print("\n[bold]Full State:[/bold]")
            state_json = pipeline.get_state_json()
            syntax = Syntax(state_json, "json", theme="monokai")
            console.print(syntax)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def cmd_read(args):
    """Read a chapter."""
    try:
        pipeline = PipelineOrchestrator(args.project, lambda **kw: None)
        pipeline.load_project()

        chapter_num = args.chapter
        if chapter_num is None:
            chapter_num = pipeline.state_manager.get_current_chapter()

        text = pipeline.read_chapter(chapter_num)
        if text:
            console.print(Markdown(text))
        else:
            console.print(f"[yellow]Chapter {chapter_num} not found.[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def cmd_delete_chapters(args):
    """Delete chapter N and all later chapters, then sync state/memory."""
    print_banner()
    project = args.project
    from_chapter = args.from_chapter
    try:
        pipeline = PipelineOrchestrator(project, lambda **kw: None)
        pipeline.load_project()
        if not args.yes:
            if not Confirm.ask(
                f"[yellow]Delete chapter {from_chapter} and all later chapters for '{project}'?[/yellow]"
            ):
                console.print("[dim]Cancelled.[/dim]")
                return
        result = pipeline.delete_chapters_from(from_chapter)
        deleted = result.get("deleted_chapters", [])
        if deleted:
            console.print(Panel(
                f"[bold green]Deleted chapters:[/bold green] {', '.join(str(c) for c in deleted)}\n"
                f"Current chapter: {result.get('current_chapter', 0)}\n"
                f"Total scenes: {result.get('total_scenes_written', 0)}\n"
                f"Removed WIP files: {result.get('deleted_wip', 0)}\n"
                f"Removed logs: {result.get('deleted_logs', 0)}\n"
                f"Pruned vector chunks: {result.get('deleted_vector_chunks', 0)}",
                title="🧹 Chapters deleted",
                border_style="green",
            ))
        else:
            console.print(f"[yellow]No chapters found from {from_chapter} onward.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def cmd_serve(args):
    """Launch the web UI."""
    print_banner()
    console.print("[bold]Starting web UI...[/bold]")
    try:
        from app import create_app
        app = create_app()
        console.print(
            f"[green]Web UI running at "
            f"http://localhost:{config.FLASK_PORT}[/green]"
        )
        app.run(
            host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG,
        )
    except ImportError:
        console.print("[red]Flask not installed. Run: pip install flask[/red]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")


def cmd_list(args):
    """List all projects."""
    print_banner()
    projects_dir = config.PROJECTS_DIR
    if not os.path.exists(projects_dir):
        console.print("[yellow]No projects found.[/yellow]")
        return

    table = Table(title="📚 Projects", border_style="blue")
    table.add_column("Name", style="cyan")
    table.add_column("Title")
    table.add_column("Genre")
    table.add_column("Chapters")

    for name in sorted(os.listdir(projects_dir)):
        state_path = os.path.join(projects_dir, name, "state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, encoding="utf-8") as f:
                    state = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            meta = state.get("metadata", {})
            table.add_row(
                name,
                meta.get("title", "?"),
                meta.get("genre", "?"),
                str(meta.get("current_chapter", 0)),
            )

    console.print(table)


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Story Teller — Multi-Agent Narrative Generation Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # new
    sub_new = subparsers.add_parser("new", help="Create a new story project")
    sub_new.set_defaults(func=cmd_new)

    # generate
    sub_gen = subparsers.add_parser("generate", help="Generate chapter(s)")
    sub_gen.add_argument("project", help="Project name")
    sub_gen.add_argument("--chapters", "-n", type=int, default=1)
    sub_gen.add_argument(
        "--pacing", "-p", default="moderate",
        choices=["slow", "moderate", "fast", "climactic"],
    )
    sub_gen.set_defaults(func=cmd_generate)

    # status
    sub_status = subparsers.add_parser("status", help="Show project status")
    sub_status.add_argument("project", help="Project name")
    sub_status.add_argument("--state", "-s", action="store_true")
    sub_status.set_defaults(func=cmd_status)

    # read
    sub_read = subparsers.add_parser("read", help="Read a chapter")
    sub_read.add_argument("project", help="Project name")
    sub_read.add_argument("chapter", type=int, nargs="?", default=None)
    sub_read.set_defaults(func=cmd_read)

    # delete-chapters
    sub_delete_chapters = subparsers.add_parser(
        "delete-chapters",
        help="Delete chapter N and all later chapters, then sync state/WIP",
    )
    sub_delete_chapters.add_argument("project", help="Project name")
    sub_delete_chapters.add_argument("from_chapter", type=int, help="Delete from this chapter number (inclusive)")
    sub_delete_chapters.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    sub_delete_chapters.set_defaults(func=cmd_delete_chapters)

    # list
    sub_list = subparsers.add_parser("list", help="List all projects")
    sub_list.set_defaults(func=cmd_list)

    # serve
    sub_serve = subparsers.add_parser("serve", help="Launch web UI")
    sub_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
