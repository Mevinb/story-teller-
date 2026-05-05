"""
Structured State Manager — JSON-based deterministic memory.
Provides CRUD operations for characters, world, plot, and metadata.
Thread-safe with atomic writes and change history tracking.
"""
import json
import os
import shutil
import logging
import re
from datetime import datetime
from typing import Any, Optional, Callable
from copy import deepcopy
import threading

logger = logging.getLogger(__name__)


def _canonical_character_key(name: str) -> str:
    """Normalize character names so aliases map to the same identity."""
    if not isinstance(name, str):
        return ""
    # "Raj (Bus Driver)" -> "raj"
    without_parenthetical = re.sub(r"\([^)]*\)", "", name).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", without_parenthetical)


def _empty_state() -> dict:
    """Returns a blank story state template."""
    return {
        "metadata": {
            "title": "",
            "genre": "",
            "premise": "",
            "themes": [],
            "setting": "",
            "current_chapter": 0,
            "total_scenes_written": 0,
            "state_version": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        },
        "characters": {},
        "world": {
            "locations": {},
            "rules": [],
            "timeline": [],
        },
        "plot": {
            "major_events": [],
            "unresolved_threads": [],
            "foreshadowing": [],
            "chapter_summaries": [],
        },
    }


class StateManager:
    """
    Manages the structured JSON state for a story project.
    This is the 'hard memory' — deterministic facts the model must respect.
    """

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.state_path = os.path.join(project_dir, "state.json")
        self.history_path = os.path.join(project_dir, "state_history.jsonl")
        self._lock = threading.Lock()
        self._state: Optional[dict] = None

    # ─── Core I/O ────────────────────────────────────────────────────

    def load(self) -> dict:
        """Load state from disk, or create a fresh one."""
        with self._lock:
            self._ensure_loaded_locked()
            return deepcopy(self._state)

    def reset_generated(self) -> None:
        """Reset all AI-generated content while preserving user-entered data.

        Keeps: metadata (title, genre, premise, themes, setting), characters, world
        Clears: plot data, chapter files, vectors, logs, WIP checkpoints
        """
        def mutator(current: dict) -> dict:
            current["plot"] = {
                "major_events": [],
                "unresolved_threads": [],
                "foreshadowing": [],
                "chapter_summaries": [],
            }
            current["metadata"]["current_chapter"] = 0
            current["metadata"]["total_scenes_written"] = 0
            for char in current.get("characters", {}).values():
                char["arc_progression"] = []
                char["state"] = {"emotion": "neutral", "goal": "", "location": ""}
            return current

        self._transition("reset_generated", {}, mutator)

        chapters_dir = os.path.join(self.project_dir, "chapters")
        if os.path.exists(chapters_dir):
            for f in os.listdir(chapters_dir):
                os.remove(os.path.join(chapters_dir, f))
            logger.info("Cleared all chapter files")

        logs_dir = os.path.join(self.project_dir, "logs")
        if os.path.exists(logs_dir):
            for f in os.listdir(logs_dir):
                os.remove(os.path.join(logs_dir, f))
            logger.info("Cleared all log files")

        vector_index_dir = os.path.join(self.project_dir, "vector_index")
        if os.path.exists(vector_index_dir):
            shutil.rmtree(vector_index_dir)
            os.makedirs(vector_index_dir)
            logger.info("Cleared vector store")

        if os.path.exists(self.history_path):
            os.remove(self.history_path)

        logger.info("Reset all generated content — user data preserved")

    def save(self) -> None:
        """Save current state to disk with atomic write."""
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        """Internal save — caller must hold the lock."""
        meta = self._state.setdefault("metadata", {})
        meta["state_version"] = int(meta.get("state_version", 0)) + 1
        self._state["metadata"]["updated_at"] = datetime.now().isoformat()
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False)
        shutil.move(tmp_path, self.state_path)

    def _log_change(self, action: str, data: dict) -> None:
        """Append a change entry to the history log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data,
        }
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _ensure_loaded_locked(self) -> None:
        if self._state is not None:
            return
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                self._state = json.load(f)
            self._state.setdefault("metadata", {})
            self._state["metadata"].setdefault("state_version", 0)
            logger.debug(f"State loaded from {self.state_path}")
        else:
            self._state = _empty_state()
            self._save_locked()
            logger.info(f"Created fresh state at {self.state_path}")

    def _transition(self, action: str, data: dict, mutator: Callable[[dict], dict]) -> None:
        with self._lock:
            self._ensure_loaded_locked()
            next_state = mutator(deepcopy(self._state))
            if not isinstance(next_state, dict):
                raise ValueError(f"State transition '{action}' did not return a dict")
            self._state = next_state
            self._save_locked()
            self._log_change(action, data)

    # ─── State Access ────────────────────────────────────────────────

    @property
    def state(self) -> dict:
        if self._state is None:
            self.load()
        return self._state

    def get_metadata(self) -> dict:
        return deepcopy(self.state.get("metadata", {}))

    def get_characters(self) -> dict:
        return deepcopy(self.state.get("characters", {}))

    def get_character(self, name: str) -> Optional[dict]:
        chars = self.state.get("characters", {})
        target = _canonical_character_key(name)
        # Case-insensitive lookup
        for key, val in chars.items():
            if key.lower() == name.lower() or _canonical_character_key(key) == target:
                return deepcopy(val)
        return None

    def get_world(self) -> dict:
        return deepcopy(self.state.get("world", {}))

    def get_plot(self) -> dict:
        return deepcopy(self.state.get("plot", {}))

    def get_current_chapter(self) -> int:
        return self.state.get("metadata", {}).get("current_chapter", 0)

    # ─── State Mutations ─────────────────────────────────────────────

    def initialize(
        self,
        title: str,
        genre: str,
        premise: str,
        characters: dict = None,
        themes: list = None,
        setting: str = "",
    ) -> dict:
        """Initialize a new story project."""
        payload = {
            "title": title,
            "genre": genre,
            "characters": list((characters or {}).keys()),
        }

        def mutator(_: dict) -> dict:
            next_state = _empty_state()
            next_state["metadata"].update({
                "title": title,
                "genre": genre,
                "premise": premise,
                "themes": themes or [],
                "setting": setting,
            })
            if characters:
                for name, info in characters.items():
                    next_state["characters"][name] = {
                        "description": info.get("description", ""),
                        "traits": info.get("traits", []),
                        "relationships": info.get("relationships", {}),
                        "state": info.get("state", {
                            "emotion": "neutral",
                            "location": "",
                            "goal": "",
                        }),
                        "arc_progression": [],
                    }
            return next_state

        self._transition("initialize", payload, mutator)
        logger.info(f"Story initialized: '{title}' ({genre})")
        return deepcopy(self.state)

    def update_character(self, name: str, updates: dict) -> None:
        """Update a character's state, traits, or relationships."""
        def mutator(current: dict) -> dict:
            chars = current["characters"]
            target_key = name if name in chars else None
            if target_key is None:
                canonical = _canonical_character_key(name)
                for existing_name in chars.keys():
                    if _canonical_character_key(existing_name) == canonical:
                        target_key = existing_name
                        break

            if target_key is None:
                target_key = name
                chars[target_key] = {
                    "description": "",
                    "traits": [],
                    "relationships": {},
                    "state": {"emotion": "neutral", "location": "", "goal": ""},
                    "arc_progression": [],
                }

            char = chars[target_key]
            if target_key != name:
                logger.debug(f"Resolved character alias '{name}' -> '{target_key}' for updates")

            for key, val in updates.items():
                if key == "traits" and isinstance(val, list):
                    existing = set(char.get("traits", []))
                    existing.update(val)
                    char["traits"] = list(existing)
                elif key == "relationships" and isinstance(val, dict):
                    char.setdefault("relationships", {}).update(val)
                elif key == "state" and isinstance(val, dict):
                    char.setdefault("state", {}).update(val)
                elif key == "arc_progression":
                    if isinstance(val, str):
                        entry = val.strip()
                        if entry:
                            char.setdefault("arc_progression", []).append(entry)
                    elif isinstance(val, list):
                        progression = char.setdefault("arc_progression", [])
                        for item in val:
                            entry = str(item).strip()
                            if entry and entry not in progression:
                                progression.append(entry)
                else:
                    char[key] = val
            return current

        self._transition("update_character", {"name": name, "updates": updates}, mutator)

    def add_event(self, event: str, chapter: int = None) -> None:
        """Add a major plot event."""
        entry = {
            "event": event,
            "chapter": chapter or self.state["metadata"]["current_chapter"],
            "timestamp": datetime.now().isoformat(),
        }

        def mutator(current: dict) -> dict:
            current["plot"]["major_events"].append(entry)
            return current

        self._transition("add_event", entry, mutator)

    def add_chapter_summary(self, chapter_num: int, summary: str) -> None:
        """Store a chapter summary for re-anchoring."""
        payload = {"chapter": chapter_num}

        def mutator(current: dict) -> dict:
            current["plot"]["chapter_summaries"].append({
                "chapter": chapter_num,
                "summary": summary,
            })
            current["metadata"]["current_chapter"] = chapter_num
            return current

        self._transition("add_chapter_summary", payload, mutator)

    def increment_scene_count(self, count: int = 1) -> None:
        """Increment the total scenes written counter."""
        def mutator(current: dict) -> dict:
            current["metadata"]["total_scenes_written"] += count
            return current

        self._transition("increment_scene_count", {"count": count}, mutator)

    def add_unresolved_thread(self, thread: str) -> None:
        """Track an unresolved plot thread."""
        def mutator(current: dict) -> dict:
            current["plot"]["unresolved_threads"].append(thread)
            return current

        self._transition("add_unresolved_thread", {"thread": thread}, mutator)

    def resolve_thread(self, thread: str) -> None:
        """Remove a resolved plot thread."""
        def mutator(current: dict) -> dict:
            threads = current["plot"]["unresolved_threads"]
            current["plot"]["unresolved_threads"] = [t for t in threads if t != thread]
            return current

        self._transition("resolve_thread", {"thread": thread}, mutator)

    def add_location(self, name: str, description: str) -> None:
        """Add or update a world location."""
        def mutator(current: dict) -> dict:
            current["world"]["locations"][name] = description
            return current

        self._transition("add_location", {"name": name}, mutator)

    def add_timeline_event(self, event: str) -> None:
        """Append to the world timeline."""
        def mutator(current: dict) -> dict:
            current["world"]["timeline"].append(event)
            return current

        self._transition("add_timeline_event", {"event": event}, mutator)

    def apply_state_update(self, updates: dict) -> None:
        """
        Apply a bulk state update from the Consistency Engine or Architect.
        Expects a dict with optional keys: characters, world, plot, metadata.
        """
        def merge_character(existing_char: dict, char_updates: dict) -> dict:
            for key, val in char_updates.items():
                if key == "traits" and isinstance(val, list):
                    trait_set = set(existing_char.get("traits", []))
                    trait_set.update(val)
                    existing_char["traits"] = list(trait_set)
                elif key == "relationships" and isinstance(val, dict):
                    existing_char.setdefault("relationships", {}).update(val)
                elif key == "state" and isinstance(val, dict):
                    existing_char.setdefault("state", {}).update(val)
                elif key == "arc_progression":
                    if isinstance(val, str):
                        entry = val.strip()
                        if entry:
                            existing_char.setdefault("arc_progression", []).append(entry)
                    elif isinstance(val, list):
                        progression = existing_char.setdefault("arc_progression", [])
                        for item in val:
                            entry = str(item).strip()
                            if entry and entry not in progression:
                                progression.append(entry)
                else:
                    existing_char[key] = val
            return existing_char

        def mutator(current: dict) -> dict:
            if "metadata" in updates:
                current["metadata"].update(updates["metadata"])
            if "world" in updates:
                if "locations" in updates["world"]:
                    current["world"]["locations"].update(updates["world"]["locations"])
                if "timeline" in updates["world"]:
                    current["world"]["timeline"].extend(updates["world"]["timeline"])
            if "plot" in updates:
                if "major_events" in updates["plot"]:
                    current["plot"]["major_events"].extend(updates["plot"]["major_events"])
                if "unresolved_threads" in updates["plot"]:
                    current["plot"]["unresolved_threads"].extend(updates["plot"]["unresolved_threads"])
            if "characters" in updates:
                existing = {k.lower(): k for k in current.get("characters", {}).keys()}
                canonical_existing = {
                    _canonical_character_key(k): k
                    for k in current.get("characters", {}).keys()
                }
                for name, char_updates in updates["characters"].items():
                    matched = existing.get(name.lower())
                    if not matched:
                        matched = canonical_existing.get(_canonical_character_key(name))
                    if matched:
                        current["characters"][matched] = merge_character(
                            current["characters"][matched],
                            char_updates,
                        )
                    else:
                        logger.debug(f"Ignoring state update for unknown character: '{name}'")
            return current

        self._transition("apply_state_update", {"keys": list(updates.keys())}, mutator)

    # ─── Context Helpers ──────────────────────────────────────────────

    def get_context_window(self, characters: list = None, include_premise: bool = True) -> str:
        """
        Build a compact text summary of current state for prompt injection.
        Optionally filtered to specific characters.
        """
        parts = []
        meta = self.get_metadata()
        parts.append(f"Story: {meta['title']} ({meta['genre']})")
        parts.append(f"Current Chapter: {meta['current_chapter']}")

        if include_premise and meta.get("premise"):
            parts.append(f"Premise: {meta['premise']}")

        # Characters
        chars = self.get_characters()
        if characters:
            wanted = {_canonical_character_key(c) for c in characters}
            chars = {
                k: v for k, v in chars.items()
                if (k in characters) or (_canonical_character_key(k) in wanted)
            }

        for name, info in chars.items():
            char_str = f"\nCharacter '{name}':"
            if info.get("traits"):
                char_str += f" Traits: {', '.join(info['traits'])}."
            if info.get("state"):
                state_parts = [f"{k}={v}" for k, v in info["state"].items() if v]
                if state_parts:
                    char_str += f" State: {', '.join(state_parts)}."
            if info.get("relationships"):
                rel_parts = [f"{k}: {v}" for k, v in info["relationships"].items()]
                char_str += f" Relationships: {', '.join(rel_parts)}."
            parts.append(char_str)

        # Unresolved threads
        plot = self.get_plot()
        if plot.get("unresolved_threads"):
            parts.append(f"\nOpen threads: {'; '.join(plot['unresolved_threads'][:5])}")

        # Recent events
        if plot.get("major_events"):
            recent = plot["major_events"][-5:]
            events_str = "; ".join(e["event"] for e in recent)
            parts.append(f"\nRecent events: {events_str}")

        return "\n".join(parts)

    def to_json(self) -> str:
        """Return the full state as formatted JSON string."""
        return json.dumps(self.state, indent=2, ensure_ascii=False)
