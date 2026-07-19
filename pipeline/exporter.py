"""
EPUB / Markdown Exporter — Phase 5 Advanced Feature.

Exports completed story chapters to:
  - Markdown bundle  (single .md file, always available)
  - EPUB 3.0        (requires `ebooklib` — graceful no-op if absent)
  - Plain text       (.txt, universal)

Usage:
    exporter = StoryExporter(project_dir)
    path = exporter.export_markdown()
    path = exporter.export_epub()    # requires: pip install ebooklib
    path = exporter.export_txt()
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import ebooklib; it is optional
try:
    from ebooklib import epub as _epub
    _EPUB_AVAILABLE = True
except ImportError:
    _epub = None
    _EPUB_AVAILABLE = False


def epub_available() -> bool:
    """Return True if the ebooklib package is installed."""
    return _EPUB_AVAILABLE


_CHAPTER_FILE_RE = re.compile(r"^chapter_(\d{3,})\.md$")


class StoryExporter:
    """
    Exports a finished story project to distributable file formats.

    Parameters
    ----------
    project_dir : str
        Root directory of the story project.
    output_dir : str | None
        Where to write exported files.  Defaults to ``project_dir``.
    """

    def __init__(self, project_dir: str, output_dir: Optional[str] = None) -> None:
        self.project_dir = project_dir
        self.chapters_dir = os.path.join(project_dir, "chapters")
        self.output_dir = output_dir or project_dir
        self.state_path = os.path.join(project_dir, "state.json")

    # ─── Meta helpers ─────────────────────────────────────────────────────────

    def _load_meta(self) -> dict:
        defaults = {
            "title": "Untitled Story",
            "author": "Anonymous",
            "genre": "Fiction",
            "language": "en",
        }
        if not os.path.exists(self.state_path):
            return defaults
        try:
            with open(self.state_path, encoding="utf-8") as f:
                state = json.load(f)
            meta = state.get("metadata", {})
            defaults["title"] = meta.get("title", defaults["title"])
            defaults["genre"] = meta.get("genre", defaults["genre"])
        except Exception:
            pass
        return defaults

    def _sorted_chapter_files(self) -> list[tuple[int, str]]:
        """Return [(chapter_num, full_path), ...] sorted by chapter number."""
        if not os.path.exists(self.chapters_dir):
            return []
        result = []
        for fname in sorted(os.listdir(self.chapters_dir)):
            m = _CHAPTER_FILE_RE.match(fname)
            if m:
                result.append((int(m.group(1)), os.path.join(self.chapters_dir, fname)))
        return result

    def _read_chapter(self, path: str) -> str:
        try:
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return ""

    def _safe_filename(self, title: str) -> str:
        safe = re.sub(r"[^\w\s-]", "", title.lower())
        safe = re.sub(r"[\s_-]+", "_", safe).strip("_")
        return safe[:60] or "story"

    # ─── Markdown Export ──────────────────────────────────────────────────────

    def export_markdown(self) -> str:
        """
        Concatenate all chapters into a single Markdown file.

        Returns the path to the exported file.
        """
        meta = self._load_meta()
        chapters = self._sorted_chapter_files()

        parts = [
            f"# {meta['title']}",
            "",
            f"> Genre: {meta['genre']}  |  Exported: {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "---",
            "",
        ]

        if not chapters:
            parts.append("*No chapters written yet.*")
        else:
            for _num, path in chapters:
                text = self._read_chapter(path)
                if text:
                    parts.append(text)
                    parts.append("")
                    parts.append("---")
                    parts.append("")

        content = "\n".join(parts)
        filename = self._safe_filename(meta["title"]) + ".md"
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("Markdown export: %s (%d chars)", out_path, len(content))
        return out_path

    # ─── Plain Text Export ────────────────────────────────────────────────────

    def export_txt(self) -> str:
        """
        Export all chapters as plain text (Markdown syntax stripped).

        Returns the path to the exported file.
        """
        meta = self._load_meta()
        chapters = self._sorted_chapter_files()

        lines = [
            meta["title"].upper(),
            "=" * len(meta["title"]),
            "",
            f"Genre: {meta['genre']}",
            f"Exported: {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "",
        ]

        for _num, path in chapters:
            text = self._read_chapter(path)
            if not text:
                continue
            # Strip Markdown syntax
            clean = self._strip_markdown(text)
            lines.append(clean)
            lines.append("")
            lines.append("-" * 40)
            lines.append("")

        content = "\n".join(lines)
        filename = self._safe_filename(meta["title"]) + ".txt"
        out_path = os.path.join(self.output_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("TXT export: %s (%d chars)", out_path, len(content))
        return out_path

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove common Markdown formatting."""
        # Remove headings (must be at start of line)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
        # Remove bold/italic
        text = re.sub(r"\*{2,3}([^*]+)\*{2,3}", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"_{2,3}([^_]+)_{2,3}", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        # Remove scene dividers
        text = re.sub(r"^\s*\*\s*\*\s*\*\s*$", "\n", text, flags=re.M)
        text = re.sub(r"^\s*-{3,}\s*$", "\n", text, flags=re.M)
        # Remove inline code
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Remove links
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ─── EPUB Export ──────────────────────────────────────────────────────────

    def export_epub(self) -> str:
        """
        Export all chapters as an EPUB 3.0 file.

        Requires the ``ebooklib`` package (``pip install ebooklib``).
        Raises ``ImportError`` if it is not installed.

        Returns the path to the exported .epub file.
        """
        if not _EPUB_AVAILABLE:
            raise ImportError(
                "ebooklib is not installed. "
                "Install it with: pip install ebooklib"
            )

        meta = self._load_meta()
        chapters = self._sorted_chapter_files()

        book = _epub.EpubBook()
        book.set_identifier(f"storyteller-{self._safe_filename(meta['title'])}")
        book.set_title(meta["title"])
        book.set_language(meta["language"])
        book.add_author(meta["author"])

        # CSS
        style = _epub.EpubItem(
            uid="style",
            file_name="style/main.css",
            media_type="text/css",
            content=self._epub_css(),
        )
        book.add_item(style)

        # Title page
        title_page = _epub.EpubHtml(
            title="Title Page",
            file_name="title_page.xhtml",
            lang=meta["language"],
        )
        title_page.set_content(self._epub_title_page_html(meta))
        title_page.add_item(style)
        book.add_item(title_page)

        spine = ["nav", title_page]
        toc = []

        # Chapters
        for num, path in chapters:
            text = self._read_chapter(path)
            if not text:
                continue
            chapter_title, body_html = self._md_to_epub_html(text)
            epub_chapter = _epub.EpubHtml(
                title=chapter_title,
                file_name=f"chapter_{num:03d}.xhtml",
                lang=meta["language"],
            )
            epub_chapter.set_content(
                f'<?xml version="1.0" encoding="utf-8"?>'
                f'<!DOCTYPE html>'
                f'<html xmlns="http://www.w3.org/1999/xhtml">'
                f'<head><title>{chapter_title}</title>'
                f'<link rel="stylesheet" href="style/main.css" type="text/css"/>'
                f'</head><body>{body_html}</body></html>'
            )
            epub_chapter.add_item(style)
            book.add_item(epub_chapter)
            spine.append(epub_chapter)
            toc.append(_epub.Link(f"chapter_{num:03d}.xhtml", chapter_title, f"ch{num}"))

        book.toc = toc
        book.spine = spine
        book.add_item(_epub.EpubNcx())
        book.add_item(_epub.EpubNav())

        filename = self._safe_filename(meta["title"]) + ".epub"
        out_path = os.path.join(self.output_dir, filename)
        _epub.write_epub(out_path, book)
        logger.info("EPUB export: %s", out_path)
        return out_path

    @staticmethod
    def _epub_css() -> str:
        return """
body {
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 1em;
    line-height: 1.6;
    margin: 2em 3em;
    color: #222;
}
h1 { font-size: 1.8em; text-align: center; margin: 1.5em 0 0.5em; }
h2 { font-size: 1.4em; text-align: center; margin: 1.2em 0 0.4em; }
p { text-indent: 1.5em; margin: 0 0 0.4em; }
p.scene-break { text-align: center; text-indent: 0; margin: 1em 0; }
.title-page { text-align: center; padding-top: 4em; }
.title-page h1 { font-size: 2.5em; }
.title-page .genre { color: #555; font-style: italic; }
"""

    @staticmethod
    def _epub_title_page_html(meta: dict) -> str:
        return (
            f'<?xml version="1.0" encoding="utf-8"?>'
            f'<!DOCTYPE html>'
            f'<html xmlns="http://www.w3.org/1999/xhtml">'
            f'<head><title>{meta["title"]}</title>'
            f'<link rel="stylesheet" href="style/main.css" type="text/css"/></head>'
            f'<body><div class="title-page">'
            f'<h1>{meta["title"]}</h1>'
            f'<p class="genre">{meta["genre"]}</p>'
            f'<p>{meta["author"]}</p>'
            f'</div></body></html>'
        )

    @staticmethod
    def _md_to_epub_html(text: str) -> tuple[str, str]:
        """Convert Markdown chapter text to (title, body_html)."""
        lines = text.split("\n")
        title = "Chapter"
        html_parts: list[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Heading
            m = re.match(r"^(#{1,2})\s+(.+)", line)
            if m:
                level = len(m.group(1))
                heading_text = m.group(2).strip()
                if level == 1 and title == "Chapter":
                    title = heading_text
                tag = f"h{level}"
                html_parts.append(f"<{tag}>{heading_text}</{tag}>")
                i += 1
                continue

            # Scene break
            if re.match(r"^\s*\*\s*\*\s*\*\s*$", line) or re.match(r"^\s*-{3,}\s*$", line):
                html_parts.append('<p class="scene-break">* * *</p>')
                i += 1
                continue

            # Empty line
            if not line.strip():
                i += 1
                continue

            # Regular paragraph — collect until blank line
            para_lines = []
            while i < len(lines) and lines[i].strip():
                para_lines.append(lines[i].strip())
                i += 1
            para_text = " ".join(para_lines)

            # Convert inline Markdown
            para_text = re.sub(r"\*{2}([^*]+)\*{2}", r"<strong>\1</strong>", para_text)
            para_text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", para_text)
            para_text = re.sub(r"_([^_]+)_", r"<em>\1</em>", para_text)
            html_parts.append(f"<p>{para_text}</p>")

        return title, "\n".join(html_parts)

    # ─── Combined export ──────────────────────────────────────────────────────

    def export_all(self, include_epub: bool = False) -> dict[str, str]:
        """
        Run all available exporters.

        Returns mapping of format → output path.
        """
        results = {}
        results["markdown"] = self.export_markdown()
        results["txt"] = self.export_txt()
        if include_epub:
            if _EPUB_AVAILABLE:
                results["epub"] = self.export_epub()
            else:
                results["epub"] = "skipped (ebooklib not installed)"
        return results
