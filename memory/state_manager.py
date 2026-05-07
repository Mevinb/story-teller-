"""
Structured State Manager — JSON-based deterministic memory.
Provides CRUD operations for characters, world, plot, and metadata.
Thread-safe with atomic writes and change history tracking.

Upgraded with narrative evolution support:
- Emotional history tracking
- Dynamic importance scoring
- Character lifecycle states
- Rich relationship graphs
- Story event log
- Legend memory (permanent truths)
- Scene transition states
- Narrative phase tracking
"""
import json
import os
import shutil
import logging
import re
from datetime import datetime
from typing import Optional, Callable
from copy import deepcopy
import threading

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:
    _fuzz = None

logger = logging.getLogger(__name__)

# ─── Nickname / Alias Map ────────────────────────────────────────────
# Common nicknames mapped to canonical full names for fuzzy matching.
NICKNAME_MAP = {
    "liz": "elizabeth", "beth": "elizabeth", "lizzy": "elizabeth",
    "mike": "michael", "mikey": "michael",
    "bob": "robert", "bobby": "robert", "rob": "robert",
    "bill": "william", "billy": "william", "will": "william",
    "dick": "richard", "rick": "richard", "ricky": "richard",
    "jim": "james", "jimmy": "james", "jamie": "james",
    "joe": "joseph", "joey": "joseph",
    "tom": "thomas", "tommy": "thomas",
    "dave": "david", "davy": "david",
    "nick": "nicholas", "nicky": "nicholas",
    "alex": "alexander", "al": "albert",
    "kate": "katherine", "kathy": "katherine", "katie": "katherine",
    "jen": "jennifer", "jenny": "jennifer",
    "sam": "samuel", "sammy": "samuel",
    "dan": "daniel", "danny": "daniel",
    "ed": "edward", "eddie": "edward",
    "charlie": "charles", "chuck": "charles",
    "matt": "matthew", "matty": "matthew",
    "pat": "patrick", "paddy": "patrick",
    "tony": "anthony",
    "steve": "steven", "stevie": "steven",
    "chris": "christopher",
    "ben": "benjamin", "benny": "benjamin",
    "ted": "theodore", "teddy": "theodore",
    "max": "maximilian",
    "maggie": "margaret", "meg": "margaret",
    "sue": "susan", "suzy": "susan",
    "becky": "rebecca", "becca": "rebecca",
}

FUZZY_MATCH_THRESHOLD = 85  # rapidfuzz ratio threshold


def _canonical_character_key(name: str) -> str:
    """Normalize character names so aliases map to the same identity.

    Process:
    1. Strip parentheticals and lowercase
    2. Remove non-alphanumeric characters
    3. Apply nickname map (e.g. 'liz' -> 'elizabeth')
    """
    if not isinstance(name, str):
        return ""
    # "Raj (Bus Driver)" -> "raj"
    without_parenthetical = re.sub(r"\([^)]*\)", "", name).strip().lower()
    stripped = re.sub(r"[^a-z0-9]+", "", without_parenthetical)
    # Apply nickname map
    return NICKNAME_MAP.get(stripped, stripped)


def _fuzzy_match_character(name: str, existing_names: dict) -> Optional[str]:
    """Try to fuzzy-match a character name against existing names.

    Args:
        name: The name to look up.
        existing_names: Dict of canonical_key -> display_name.

    Returns:
        The matched display name, or None.
    """
    if _fuzz is None:
        return None
    target = _canonical_character_key(name)
    if not target:
        return None
    best_score = 0
    best_match = None
    for canonical, display in existing_names.items():
        score = _fuzz.ratio(target, canonical)
        if score > best_score and score >= FUZZY_MATCH_THRESHOLD:
            best_score = score
            best_match = display
    return best_match


def _normalize_traits_list(traits: list) -> list:
    """
    Normalize traits while removing obvious mutually-exclusive duplicates
    that cause false consistency blockers.
    """
    if not isinstance(traits, list):
        return []

    cleaned = []
    seen = set()
    for raw in traits:
        trait = str(raw).strip()
        if not trait:
            continue
        key = re.sub(r"\s+", " ", trait.lower())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(trait)

    lower_map = {trait: trait.lower() for trait in cleaned}

    def has_any(markers: tuple[str, ...]) -> bool:
        for val in lower_map.values():
            if any(marker in val for marker in markers):
                return True
        return False

    has_experienced = has_any(("experienced", "post-virginity", "no longer virgin", "lost virginity"))
    has_virgin = has_any((" virgin", "virgin ", "virgin)", "virginity")) or any(
        val == "virgin" for val in lower_map.values()
    )

    if has_experienced and has_virgin:
        filtered = []
        for trait in cleaned:
            val = trait.lower()
            # Keep timeline-style constraints like "virgin (until X)".
            if "virgin" in val and "until" not in val and "post-virginity" not in val:
                continue
            filtered.append(trait)
        cleaned = filtered

    # Re-dedupe after filtering.
    deduped = []
    seen2 = set()
    for trait in cleaned:
        key = re.sub(r"\s+", " ", trait.lower())
        if key in seen2:
            continue
        seen2.add(key)
        deduped.append(trait)
    return deduped


def _empty_state() -> dict:
    """Returns a blank story state template.

    Includes evolution engine fields:
    - narrative_phase: Current story phase
    - story_events: Structured event log
    - legend_memory: Permanent truths
    - transitions: Scene transition states
    """
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
            "narrative_phase": "introduction",
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
            "story_events": [],
            "legend_memory": [],
        },
        "transitions": [],
    }


def _merge_character_records(primary: dict, duplicate: dict) -> dict:
    """Merge duplicate character aliases without discarding richer primary data."""
    primary = primary or {}
    duplicate = duplicate or {}

    if not primary.get("description") and duplicate.get("description"):
        primary["description"] = duplicate["description"]

    if primary.get("role") != "main" and duplicate.get("role"):
        primary["role"] = duplicate["role"]

    trait_set = set(primary.get("traits", []))
    trait_set.update(duplicate.get("traits", []))
    primary["traits"] = _normalize_traits_list(list(trait_set))

    primary.setdefault("relationships", {}).update(duplicate.get("relationships", {}))

    state = primary.setdefault("state", {"emotion": "neutral", "location": "", "goal": ""})
    for key, value in duplicate.get("state", {}).items():
        if value and not state.get(key):
            state[key] = value

    progression = primary.setdefault("arc_progression", [])
    for item in duplicate.get("arc_progression", []):
        entry = str(item).strip()
        if entry and entry not in progression:
            progression.append(entry)

    return primary


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
                path = os.path.join(chapters_dir, f)
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            logger.info("Cleared all chapter files")

        logs_dir = os.path.join(self.project_dir, "logs")
        if os.path.exists(logs_dir):
            for f in os.listdir(logs_dir):
                path = os.path.join(logs_dir, f)
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
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
            if self._dedupe_characters_locked():
                self._save_locked()
            logger.debug(f"State loaded from {self.state_path}")
        else:
            self._state = _empty_state()
            self._save_locked()
            logger.info(f"Created fresh state at {self.state_path}")

    def _dedupe_characters_locked(self) -> bool:
        chars = self._state.get("characters", {})
        if not isinstance(chars, dict):
            self._state["characters"] = {}
            return True

        deduped = {}
        canonical_to_name = {}
        changed = False
        for name, info in chars.items():
            canonical = _canonical_character_key(name)
            if canonical and canonical in canonical_to_name:
                target_name = canonical_to_name[canonical]
                deduped[target_name] = _merge_character_records(deduped[target_name], info)
                changed = True
                logger.info("Merged duplicate character alias '%s' into '%s'", name, target_name)
                continue

            canonical_to_name[canonical] = name
            deduped[name] = info

        if changed:
            self._state["characters"] = deduped
        return changed

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
                        "role": info.get("role", "main"),
                        # Evolution engine fields
                        "emotional_history": [],
                        "importance_score": 0.7 if info.get("role") == "main" else 0.3,
                        "mention_count": 0,
                        "dialogue_density": 0.0,
                        "status": info.get("status", "active"),
                        "first_seen": 0,
                        "last_seen": 0,
                        "aliases": info.get("aliases", []),
                        "arc_progression_data": info.get("arc_progression_data", {}),
                        "story_events_involved": 0,
                    }
            return next_state

        self._transition("initialize", payload, mutator)
        logger.info(f"Story initialized: '{title}' ({genre})")
        return deepcopy(self.state)

    def update_character(self, name: str, updates: dict | str | list | None) -> None:
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
                    "role": "supporting",
                }

            char = chars[target_key]
            if target_key != name:
                logger.debug(f"Resolved character alias '{name}' -> '{target_key}' for updates")

            normalized_updates = updates
            if normalized_updates is None:
                normalized_updates = {}
            elif isinstance(normalized_updates, str):
                entry = normalized_updates.strip()
                normalized_updates = {"arc_progression": [entry]} if entry else {}
            elif isinstance(normalized_updates, list):
                normalized_updates = {
                    "arc_progression": [str(item).strip() for item in normalized_updates if str(item).strip()]
                }
            elif not isinstance(normalized_updates, dict):
                logger.warning(
                    "Ignoring invalid character update payload for '%s': %s",
                    target_key,
                    type(normalized_updates).__name__,
                )
                normalized_updates = {}

            for key, val in normalized_updates.items():
                if key == "traits" and isinstance(val, list):
                    existing = set(char.get("traits", []))
                    existing.update(val)
                    char["traits"] = _normalize_traits_list(list(existing))
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

    def prune_after_chapter(self, max_chapter: int, total_scenes_written: int) -> dict:
        """
        Rewind generated-state artifacts so only data up to max_chapter remains.
        Keeps user-entered metadata/character definitions/world; trims generated plot
        history and clears generated character runtime fields that cannot be safely
        attributed to remaining chapters.
        """
        max_chapter = max(0, int(max_chapter))
        total_scenes_written = max(0, int(total_scenes_written))
        summary = {
            "character_arcs_cleared": 0,
            "character_states_reset": 0,
            "generated_character_fields_removed": 0,
            "supporting_characters_removed": 0,
        }

        payload = {
            "max_chapter": max_chapter,
            "total_scenes_written": total_scenes_written,
        }

        def mutator(current: dict) -> dict:
            def chapter_num(value, default=0):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default

            metadata = current.setdefault("metadata", {})
            metadata["current_chapter"] = max_chapter
            metadata["total_scenes_written"] = total_scenes_written

            plot = current.setdefault("plot", {})
            chapter_summaries = plot.get("chapter_summaries", [])
            if isinstance(chapter_summaries, list):
                plot["chapter_summaries"] = [
                    s for s in chapter_summaries
                    if chapter_num(s.get("chapter", 0)) <= max_chapter
                ]

            major_events = plot.get("major_events", [])
            if isinstance(major_events, list):
                plot["major_events"] = [
                    e for e in major_events
                    if chapter_num(e.get("chapter", 0)) <= max_chapter
                ]

            # Character state/arc changes are generated during scene/chapter runs.
            # Older projects did not record per-character chapter provenance, so after
            # deleting chapters we must prefer a clean baseline over stale future facts.
            characters = current.setdefault("characters", {})
            if isinstance(characters, dict):
                for name in list(characters.keys()):
                    char = characters.get(name)
                    if not isinstance(char, dict):
                        continue

                    if (
                        char.get("generated")
                        or (
                            char.get("role") == "supporting"
                            and str(char.get("description", "")).strip()
                            in {"", "Supporting character introduced by the story."}
                        )
                    ):
                        characters.pop(name, None)
                        summary["supporting_characters_removed"] += 1
                        continue

                    if char.get("arc_progression"):
                        summary["character_arcs_cleared"] += len(char.get("arc_progression", []))
                    char["arc_progression"] = []

                    if char.get("state") != {"emotion": "neutral", "goal": "", "location": ""}:
                        summary["character_states_reset"] += 1
                    char["state"] = {"emotion": "neutral", "goal": "", "location": ""}

                    for generated_key in ("emotion", "location", "goal"):
                        if generated_key in char:
                            char.pop(generated_key, None)
                            summary["generated_character_fields_removed"] += 1

            return current

        self._transition("prune_after_chapter", payload, mutator)
        return summary

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
                    existing_char["traits"] = _normalize_traits_list(list(trait_set))
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
                        current["characters"][name] = merge_character(
                            {
                                "description": "",
                                "traits": [],
                                "relationships": {},
                                "state": {"emotion": "neutral", "location": "", "goal": ""},
                                "arc_progression": [],
                                "role": "supporting",
                                "generated": True,
                            },
                            char_updates,
                        )
                        logger.info("Added new supporting character from state update: '%s'", name)
            return current

        self._transition("apply_state_update", {"keys": list(updates.keys())}, mutator)

    def normalize_character_traits(self) -> dict:
        """
        Normalize stored traits across all characters and persist if changes occur.
        Returns a summary of how much stale/contradictory trait noise was removed.
        """
        summary = {"characters_touched": 0, "traits_removed": 0}
        snapshot = self.load().get("characters", {})
        has_changes = False
        if isinstance(snapshot, dict):
            for char in snapshot.values():
                if not isinstance(char, dict):
                    continue
                before = char.get("traits", [])
                after = _normalize_traits_list(before)
                if before != after:
                    has_changes = True
                    break
        if not has_changes:
            return summary

        def mutator(current: dict) -> dict:
            characters = current.get("characters", {})
            if not isinstance(characters, dict):
                return current
            for char in characters.values():
                if not isinstance(char, dict):
                    continue
                before = char.get("traits", [])
                after = _normalize_traits_list(before)
                if before != after:
                    summary["characters_touched"] += 1
                    summary["traits_removed"] += max(0, len(before) - len(after))
                    char["traits"] = after
            return current

        self._transition("normalize_character_traits", summary, mutator)
        return summary

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

    # ─── Evolution Engine Mutations ───────────────────────────────────

    def update_emotional_history(self, name: str, entry: dict) -> None:
        """Append an emotional history entry for a character.

        Entry format: {"chapter": int, "emotion": str, "cause": str, "intensity": float}
        """
        def mutator(current: dict) -> dict:
            chars = current.get("characters", {})
            # Find character by canonical key
            target_key = None
            canonical = _canonical_character_key(name)
            for existing_name in chars.keys():
                if existing_name == name or _canonical_character_key(existing_name) == canonical:
                    target_key = existing_name
                    break
            if target_key is None:
                return current
            char = chars[target_key]
            history = char.setdefault("emotional_history", [])
            history.append(entry)
            # Keep last 30 entries
            if len(history) > 30:
                char["emotional_history"] = history[-30:]
            return current

        self._transition("update_emotional_history", {"name": name}, mutator)

    def update_character_status(self, name: str, status: str) -> None:
        """Update a character's lifecycle status.

        Valid statuses: active, missing, dead, retired, imprisoned, corrupted
        """
        valid = {"active", "missing", "dead", "retired", "imprisoned", "corrupted"}
        status = str(status).strip().lower()
        if status not in valid:
            logger.warning("Invalid character status '%s' for '%s'", status, name)
            return
        self.update_character(name, {"status": status})

    def add_transition_state(self, transition: dict) -> None:
        """Store a scene transition state object."""
        def mutator(current: dict) -> dict:
            transitions = current.setdefault("transitions", [])
            transitions.append(transition)
            # Keep last 10 transitions
            if len(transitions) > 10:
                current["transitions"] = transitions[-10:]
            return current

        self._transition("add_transition_state", {}, mutator)

    def get_latest_transition(self) -> Optional[dict]:
        """Get the most recent transition state."""
        transitions = self.state.get("transitions", [])
        if transitions:
            return deepcopy(transitions[-1])
        return None

    def set_narrative_phase(self, phase: str) -> None:
        """Update the current narrative phase.

        Valid phases: introduction, escalation, midpoint, collapse, climax, aftermath
        """
        valid = {"introduction", "escalation", "midpoint", "collapse", "climax", "aftermath"}
        phase = str(phase).strip().lower()
        if phase not in valid:
            logger.debug("Non-standard narrative phase '%s' — accepting anyway", phase)

        def mutator(current: dict) -> dict:
            current.setdefault("metadata", {})["narrative_phase"] = phase
            return current

        self._transition("set_narrative_phase", {"phase": phase}, mutator)

    def get_narrative_phase(self) -> str:
        """Get the current narrative phase."""
        return self.state.get("metadata", {}).get("narrative_phase", "introduction")

    def add_legend_event(self, event: dict) -> None:
        """Add a permanent truth to legend memory."""
        def mutator(current: dict) -> dict:
            legend = current.setdefault("plot", {}).setdefault("legend_memory", [])
            legend.append(event)
            return current

        self._transition("add_legend_event", {}, mutator)

    def add_story_event(self, event: dict) -> None:
        """Add a structured story event to the event log."""
        def mutator(current: dict) -> dict:
            events = current.setdefault("plot", {}).setdefault("story_events", [])
            events.append(event)
            # Keep last 100 events
            if len(events) > 100:
                current["plot"]["story_events"] = events[-100:]
            return current

        self._transition("add_story_event", {}, mutator)

    def get_story_events(self) -> list:
        """Get all structured story events."""
        return deepcopy(self.state.get("plot", {}).get("story_events", []))

    def get_emotional_history(self, name: str, last_n: int = 5) -> list:
        """Get recent emotional history for a character."""
        char = self.get_character(name)
        if not char:
            return []
        history = char.get("emotional_history", [])
        return deepcopy(history[-last_n:])
