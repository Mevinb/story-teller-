"""
World Bible Generator — Phase 5 Advanced Feature.

Automatically extracts and structures world-building information from
completed chapters into a "World Bible" document:

  - Named locations and their descriptions
  - Factions, organisations, and institutions
  - Recurring objects / artefacts
  - World rules / magic systems / technology
  - Timeline of key events
  - Character relationship web summary

The generator works in two modes:
  1. LLM-assisted  — uses the active model to extract structured data
  2. Regex fallback — deterministic extraction when no LLM is available

Output is a Markdown document saved to <project_dir>/world_bible.md and
a JSON index saved to <project_dir>/world_bible.json.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class WorldEntry:
    """A single extracted world fact."""
    category: str           # location | faction | object | rule | event
    name: str
    description: str
    source_chapters: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "name": self.name,
            "description": self.description,
            "source_chapters": self.source_chapters,
        }


@dataclass
class WorldBible:
    """Full world bible for a story project."""
    title: str
    genre: str
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    locations: list[WorldEntry] = field(default_factory=list)
    factions: list[WorldEntry] = field(default_factory=list)
    objects: list[WorldEntry] = field(default_factory=list)
    rules: list[WorldEntry] = field(default_factory=list)
    events: list[WorldEntry] = field(default_factory=list)
    relationships: dict[str, list[str]] = field(default_factory=dict)

    def all_entries(self) -> list[WorldEntry]:
        return self.locations + self.factions + self.objects + self.rules + self.events

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "genre": self.genre,
            "generated_at": self.generated_at,
            "locations": [e.to_dict() for e in self.locations],
            "factions": [e.to_dict() for e in self.factions],
            "objects": [e.to_dict() for e in self.objects],
            "rules": [e.to_dict() for e in self.rules],
            "events": [e.to_dict() for e in self.events],
            "relationships": self.relationships,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# World Bible: {self.title}",
            f"> Genre: {self.genre}  |  Generated: {self.generated_at[:10]}",
            "",
        ]

        def _section(title: str, entries: list[WorldEntry]) -> None:
            if not entries:
                return
            lines.append(f"## {title}")
            lines.append("")
            for e in entries:
                chs = ", ".join(f"Ch{c}" for c in sorted(set(e.source_chapters)))
                lines.append(f"### {e.name}")
                if chs:
                    lines.append(f"*First seen: {chs}*")
                lines.append("")
                lines.append(e.description)
                lines.append("")

        _section("📍 Locations", self.locations)
        _section("⚔️ Factions & Organisations", self.factions)
        _section("🏺 Notable Objects & Artefacts", self.objects)
        _section("📜 World Rules & Systems", self.rules)
        _section("📅 Key Events Timeline", self.events)

        if self.relationships:
            lines.append("## 🔗 Character Relationships")
            lines.append("")
            for char, rels in sorted(self.relationships.items()):
                lines.append(f"**{char}**")
                for rel in rels:
                    lines.append(f"  - {rel}")
                lines.append("")

        return "\n".join(lines)


# ─── Extractor ────────────────────────────────────────────────────────────────

class WorldBibleGenerator:
    """
    Extracts world-building information from completed chapters.

    Parameters
    ----------
    project_dir : str
        Path to the project directory (contains chapters/ and state.json).
    llm : optional
        An LLM model with a ``generate_with_retry`` method.  If None, falls
        back to regex-only extraction.
    progress_callback : optional
        Called with status strings during long extractions.
    """

    # LLM extraction prompt
    _EXTRACT_SYSTEM = (
        "You are a world-building analyst for fiction. "
        "Extract factual world information from the story text provided. "
        "Return ONLY valid JSON with the exact schema requested. "
        "Do not invent information not present in the text."
    )

    _EXTRACT_PROMPT = """\
Extract world-building facts from this story excerpt:

{text}

Return JSON matching this schema:
{{
  "locations": [{{"name": "str", "description": "str"}}],
  "factions": [{{"name": "str", "description": "str"}}],
  "objects": [{{"name": "str", "description": "str"}}],
  "rules": [{{"name": "str", "description": "str"}}],
  "events": [{{"name": "str", "description": "str"}}]
}}

Rules:
- locations: named places (rooms, buildings, cities, countries)
- factions: organisations, companies, families, gangs, schools
- objects: recurring props, weapons, vehicles, technology, magic items
- rules: physics rules, magic systems, cultural laws, technology limits
- events: dated/named historical events referenced in the text
- Only include items explicitly mentioned; do not invent
- Use empty arrays if a category has no entries
"""

    # Regex patterns for fallback extraction
    _LOCATION_PATTERNS = [
        re.compile(r"\b(?:at|in|inside|outside|near|through|across)\s+(?:the\s+)?([A-Z][a-z][\w\s]{2,20})", re.M),
        re.compile(r"\b([A-Z][a-z][\w\s]{2,15})\s+(?:hotel|apartment|building|restaurant|bar|cafe|hospital|school|university|office|park|street|road|avenue|station|airport|port)\b", re.M | re.I),
    ]

    _FACTION_PATTERNS = [
        re.compile(r"\bthe\s+([A-Z][a-z][\w\s]{2,20})\s+(?:family|clan|gang|group|organisation|organization|company|firm|corp|institute|society|club|team|squad|division|department)\b", re.M),
    ]

    def __init__(
        self,
        project_dir: str,
        llm=None,
        progress_callback=None,
    ) -> None:
        self.project_dir = project_dir
        self.chapters_dir = os.path.join(project_dir, "chapters")
        self.state_path = os.path.join(project_dir, "state.json")
        self._llm = llm
        self._progress = progress_callback or (lambda msg: None)

    # ─── Public API ───────────────────────────────────────────────────────────

    def generate(self, max_chapters: int = 999) -> WorldBible:
        """
        Read all completed chapters and extract world-building data.

        Returns a :class:`WorldBible` and writes two files:
          - ``<project_dir>/world_bible.md``
          - ``<project_dir>/world_bible.json``
        """
        self._progress("Loading project state...")
        title, genre = self._load_meta()
        bible = WorldBible(title=title, genre=genre)

        chapter_texts = self._load_chapters(max_chapters)
        if not chapter_texts:
            self._progress("No chapters found — world bible will be empty.")
            self._save(bible)
            return bible

        self._progress(f"Analysing {len(chapter_texts)} chapter(s)...")

        for chapter_num, text in chapter_texts.items():
            self._progress(f"Extracting from Chapter {chapter_num}...")
            extracted = self._extract_from_chapter(text, chapter_num)
            self._merge(bible, extracted, chapter_num)

        # Deduplicate entries
        bible.locations = self._deduplicate(bible.locations)
        bible.factions = self._deduplicate(bible.factions)
        bible.objects = self._deduplicate(bible.objects)
        bible.rules = self._deduplicate(bible.rules)
        bible.events = self._deduplicate(bible.events)

        # Add character relationships from state.json
        bible.relationships = self._extract_relationships()

        self._save(bible)
        self._progress(
            f"World Bible generated: {len(bible.locations)} locations, "
            f"{len(bible.factions)} factions, {len(bible.objects)} objects, "
            f"{len(bible.rules)} rules, {len(bible.events)} events."
        )
        return bible

    # ─── Internals ────────────────────────────────────────────────────────────

    def _load_meta(self) -> tuple[str, str]:
        if not os.path.exists(self.state_path):
            return "Untitled", "fiction"
        try:
            with open(self.state_path, encoding="utf-8") as f:
                state = json.load(f)
            meta = state.get("metadata", {})
            return meta.get("title", "Untitled"), meta.get("genre", "fiction")
        except Exception:
            return "Untitled", "fiction"

    def _load_chapters(self, max_chapters: int) -> dict[int, str]:
        texts: dict[int, str] = {}
        if not os.path.exists(self.chapters_dir):
            return texts
        _re = re.compile(r"^chapter_(\d{3,})\.md$")
        for fname in sorted(os.listdir(self.chapters_dir)):
            m = _re.match(fname)
            if not m:
                continue
            num = int(m.group(1))
            if num > max_chapters:
                continue
            path = os.path.join(self.chapters_dir, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    texts[num] = f.read()
            except OSError:
                pass
        return texts

    def _extract_from_chapter(self, text: str, chapter_num: int) -> dict:
        """Try LLM extraction; fall back to regex if LLM unavailable or fails."""
        if self._llm:
            try:
                return self._llm_extract(text, chapter_num)
            except Exception as exc:
                logger.warning("LLM extraction failed for ch%s: %s", chapter_num, exc)
        return self._regex_extract(text)

    def _llm_extract(self, text: str, chapter_num: int) -> dict:
        # Truncate to avoid token overflow (first 3000 words)
        words = text.split()
        excerpt = " ".join(words[:3000])
        prompt = self._EXTRACT_PROMPT.format(text=excerpt)
        response = self._llm.generate_with_retry(
            prompt=prompt,
            system=self._EXTRACT_SYSTEM,
            schema={
                "type": "object",
                "properties": {
                    "locations": {"type": "array"},
                    "factions": {"type": "array"},
                    "objects": {"type": "array"},
                    "rules": {"type": "array"},
                    "events": {"type": "array"},
                },
            },
            temperature=0.1,
            max_tokens=1200,
        )
        data = response.as_json() if response else {}
        return data if isinstance(data, dict) else {}

    def _regex_extract(self, text: str) -> dict:
        """Fast deterministic extraction using regex patterns."""
        locations = set()
        factions = set()

        for pattern in self._LOCATION_PATTERNS:
            for m in pattern.finditer(text):
                name = m.group(1).strip().rstrip(".,;:!?")
                if 3 <= len(name) <= 40 and name[0].isupper():
                    locations.add(name)

        for pattern in self._FACTION_PATTERNS:
            for m in pattern.finditer(text):
                name = m.group(1).strip()
                if 2 <= len(name) <= 40:
                    factions.add("the " + name)

        return {
            "locations": [{"name": n, "description": f"Location mentioned in the story."} for n in sorted(locations)],
            "factions": [{"name": n, "description": f"Organisation or group in the story."} for n in sorted(factions)],
            "objects": [],
            "rules": [],
            "events": [],
        }

    def _merge(self, bible: WorldBible, extracted: dict, chapter_num: int) -> None:
        def _add(lst: list, data: list, category: str) -> None:
            for item in data or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                desc = str(item.get("description", "")).strip()
                if name and desc:
                    lst.append(WorldEntry(
                        category=category,
                        name=name,
                        description=desc,
                        source_chapters=[chapter_num],
                    ))

        _add(bible.locations, extracted.get("locations", []), "location")
        _add(bible.factions, extracted.get("factions", []), "faction")
        _add(bible.objects, extracted.get("objects", []), "object")
        _add(bible.rules, extracted.get("rules", []), "rule")
        _add(bible.events, extracted.get("events", []), "event")

    def _deduplicate(self, entries: list[WorldEntry]) -> list[WorldEntry]:
        """Merge entries with the same name (case-insensitive)."""
        seen: dict[str, WorldEntry] = {}
        for entry in entries:
            key = entry.name.lower().strip()
            if key in seen:
                # Merge chapter references
                seen[key].source_chapters.extend(entry.source_chapters)
                # Keep longer description
                if len(entry.description) > len(seen[key].description):
                    seen[key].description = entry.description
            else:
                seen[key] = WorldEntry(
                    category=entry.category,
                    name=entry.name,
                    description=entry.description,
                    source_chapters=list(entry.source_chapters),
                )
        result = list(seen.values())
        result.sort(key=lambda e: e.name.lower())
        return result

    def _extract_relationships(self) -> dict[str, list[str]]:
        """Read character relationships from state.json."""
        if not os.path.exists(self.state_path):
            return {}
        try:
            with open(self.state_path, encoding="utf-8") as f:
                state = json.load(f)
            rels: dict[str, list[str]] = {}
            for name, info in state.get("characters", {}).items():
                if not isinstance(info, dict):
                    continue
                char_rels = info.get("relationships", {})
                if isinstance(char_rels, dict) and char_rels:
                    rels[name] = [
                        f"{other}: {desc}" if isinstance(desc, str) else str(other)
                        for other, desc in char_rels.items()
                    ]
                elif isinstance(char_rels, list) and char_rels:
                    rels[name] = [str(r) for r in char_rels]
            return rels
        except Exception:
            return {}

    def _save(self, bible: WorldBible) -> None:
        md_path = os.path.join(self.project_dir, "world_bible.md")
        json_path = os.path.join(self.project_dir, "world_bible.json")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(bible.to_markdown())

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(bible.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("World Bible saved to %s", md_path)
