"""
Consistency Engine — Post-generation validator.
Compares generated text against the structured state to detect
character contradictions, timeline errors, and logical conflicts.
Runs on local model for cheap validation.
"""
import logging
import re

from models.base import LLMInterface
from .contract import AgentContract

import config

logger = logging.getLogger(__name__)

CONSISTENCY_SYSTEM = (
    "You are a Consistency Engine — a strict but context-aware narrative validator.\n"
    "Your ONLY job is to compare generated fiction text against established story facts.\n"
    "You detect: character contradictions, timeline errors, relationship inconsistencies,\n"
    "location errors, and logical conflicts.\n\n"
    "You are NOT a creative writer. You are a validator.\n"
    "Be thorough but do not flag stylistic choices as errors.\n"
    "Only flag actual factual contradictions with the established state.\n\n"
    "IMPORTANT: Distinguish immutable facts from mutable states.\n"
    "- Immutable facts (identity, chronology, prior established events) can be contradictions.\n"
    "- Mutable states (emotion, arousal, confidence, goals, relationship tension, personality shifts,\n"
    "  injuries, sexual status changes) are expected to evolve during scenes.\n"
    "- If a mutable state changes and the scene text shows that change happening, do NOT treat it as\n"
    "  a contradiction. Record it in state_updates instead.\n\n"
    "ALWAYS respond with ONLY valid JSON."
)

VALIDATE_PROMPT = (
    "Validate this scene against the established story state.\n\n"
    "=== GENERATED SCENE TEXT ===\n{scene_text}\n\n"
    "=== ESTABLISHED STATE ===\n{state_context}\n\n"
    "=== INSTRUCTIONS ===\n"
    "Check for:\n"
    "1. Character trait contradictions (e.g., shy character acting boldly without development)\n"
    "2. Relationship inconsistencies (e.g., allies acting as enemies without cause)\n"
    "3. Location errors (character in wrong place)\n"
    "4. Timeline conflicts (events out of order)\n"
    "5. Missing or contradicted facts\n"
    "6. Character evolution and personality/emotion shifts that are justified in-scene\n\n"
    "When a character changes emotional tone, personality expression, or sexual status due to events\n"
    "shown in the scene, treat that as a VALID state transition and capture it in state_updates.\n"
    "Do not mark those as blocking contradictions.\n\n"
    "Respond with this JSON:\n"
    '{{\n'
    '    "is_consistent": true/false,\n'
    '    "issues": [\n'
    '        {{\n'
    '            "type": "character_contradiction|relationship_error|'
    'location_error|timeline_error|logic_error|state_transition",\n'
    '            "detail": "What the inconsistency is",\n'
    '            "severity": "high|medium|low",\n'
    '            "blocking": true/false,\n'
    '            "suggestion": "How to fix it"\n'
    '        }}\n'
    '    ],\n'
    '    "state_updates": {{\n'
    '        "characters": {{\n'
    '            "name": {{\n'
    '                "state": {{"emotion": "new", "location": "new", "goal": "new"}},\n'
    '                "traits": ["new persistent trait if needed"],\n'
    '                "arc_progression": "short note about personality/state evolution"\n'
    '            }}\n'
    '        }}\n'
    '    }}\n'
    '}}'
)


class ConsistencyEngine(AgentContract):
    """
    Validates generated scenes against the structured state.
    This is a validator, not a generator — it checks for factual errors.
    """

    def __init__(self, model: LLMInterface):
        self.name = "critic"
        self.model = model
        self._transition_markers = {
            "emotion", "aroused", "arousal", "predatory", "hungry", "possessive",
            "satisfied", "personality", "state shift", "state change", "character development",
            "arc progression", "no longer virgin", "virgin status", "first time",
            "goal changed", "confidence shift",
        }
        self._hard_contradiction_markers = {
            "timeline", "chronology", "impossible", "cannot both", "dead", "resurrected",
            "wrong location", "different location", "identity mismatch", "name mismatch",
            "gender mismatch",
        }

    def run(self, state: dict) -> dict:
        report = self.validate(
            scene_text=state["scene_text"],
            state_context=state["state_context"],
        )
        has_blocking = self.has_blocking_issues(report)
        confidence = 0.95 if report.get("is_consistent", True) else 0.4
        return {
            "output": {"report": report},
            "confidence": confidence,
            "next_action": "decision" if has_blocking else "editor",
        }

    def validate(self, scene_text: str, state_context: str) -> dict:
        """
        Validate a scene against the current state.

        Args:
            scene_text: The generated prose to validate
            state_context: Context from the Retriever (state + past content)

        Returns:
            Validation report dict with is_consistent, issues, state_updates
        """
        prompt = VALIDATE_PROMPT.format(
            scene_text=scene_text[:3000],  # Limit to fit context window
            state_context=state_context[:2000],
        )

        schema = {
            "type": "object",
            "properties": {
                "is_consistent": {"type": "boolean"},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "detail": {"type": "string"},
                            "severity": {"type": "string"},
                            "blocking": {"type": "boolean"},
                            "suggestion": {"type": "string"},
                        },
                        "required": ["type", "detail", "severity", "blocking"],
                    },
                },
                "state_updates": {"type": "object"},
            },
            "required": ["is_consistent", "issues"],
        }

        response = self.model.generate_with_retry(
            prompt=prompt,
            system=CONSISTENCY_SYSTEM,
            schema=schema,
            temperature=config.AGENT_TEMPERATURES["critic"],
        )

        result = response.as_json()
        if result is None:
            logger.warning("Consistency Engine returned invalid JSON. Assuming consistent.")
            return {
                "is_consistent": True,
                "issues": [],
                "state_updates": {},
            }

        # Some models occasionally return a top-level list instead of the expected object.
        if isinstance(result, list):
            if len(result) == 1 and isinstance(result[0], dict):
                result = result[0]
            else:
                result = {
                    "is_consistent": True,
                    "issues": [i for i in result if isinstance(i, dict)],
                    "state_updates": {},
                }

        if not isinstance(result, dict):
            logger.warning("Consistency Engine returned non-object JSON. Assuming consistent.")
            return {
                "is_consistent": True,
                "issues": [],
                "state_updates": {},
            }

        raw_issues = result.get("issues", [])
        if isinstance(raw_issues, dict):
            raw_issues = [raw_issues]
        elif not isinstance(raw_issues, list):
            raw_issues = []

        result["issues"] = [
            self._normalize_issue(issue)
            for issue in raw_issues
            if isinstance(issue, dict)
        ]
        result["state_updates"] = self._normalize_state_updates(result.get("state_updates"))

        blocking_issues = self.get_blocking_issues(result)
        if blocking_issues:
            logger.warning(
                f"Consistency check found {len(blocking_issues)} blocking issues: "
                f"{[i['type'] for i in blocking_issues]}"
            )
            result["is_consistent"] = False
        else:
            result["is_consistent"] = True

        return result

    def has_blocking_issues(self, report: dict) -> bool:
        """Check if a validation report has issues that should block publishing."""
        return bool(self.get_blocking_issues(report))

    def get_blocking_issues(self, report: dict) -> list[dict]:
        issues = report.get("issues", []) if isinstance(report, dict) else []
        return [issue for issue in issues if self._is_blocking_issue(issue)]

    def get_non_blocking_issues(self, report: dict) -> list[dict]:
        issues = report.get("issues", []) if isinstance(report, dict) else []
        return [issue for issue in issues if not self._is_blocking_issue(issue)]

    @staticmethod
    def issue_signature(issue: dict) -> str:
        issue_type = str(issue.get("type", "logic_error")).strip().lower()
        detail = re.sub(r"\s+", " ", str(issue.get("detail", "")).strip().lower())
        return f"{issue_type}:{detail[:180]}"

    def _normalize_issue(self, issue: dict) -> dict:
        normalized = {
            "type": str(issue.get("type", "logic_error")).strip().lower() or "logic_error",
            "detail": str(issue.get("detail", "")).strip(),
            "severity": str(issue.get("severity", "low")).strip().lower() or "low",
            "blocking": issue.get("blocking"),
            "suggestion": str(issue.get("suggestion", "")).strip(),
        }

        if normalized["severity"] not in {"high", "medium", "low"}:
            normalized["severity"] = "medium"

        if not isinstance(normalized["blocking"], bool):
            normalized["blocking"] = normalized["severity"] in {"high", "medium"}

        if self._looks_like_state_transition(normalized):
            normalized["type"] = "state_transition"
            normalized["blocking"] = False
            if normalized["severity"] == "high":
                normalized["severity"] = "medium"
            return normalized

        hard_types = {
            "timeline_error",
            "location_error",
            "relationship_error",
            "logic_error",
            "character_contradiction",
        }
        if normalized["type"] in hard_types:
            normalized["blocking"] = normalized["severity"] in {"high", "medium"}

        return normalized

    def _normalize_state_updates(self, updates: dict) -> dict:
        if not isinstance(updates, dict):
            return {}

        normalized = {}
        characters = updates.get("characters")
        if isinstance(characters, dict):
            clean_chars = {}
            for name, char_update in characters.items():
                if not isinstance(name, str) or not isinstance(char_update, dict):
                    continue
                clean = {}
                if isinstance(char_update.get("state"), dict):
                    clean["state"] = char_update["state"]
                if isinstance(char_update.get("relationships"), dict):
                    clean["relationships"] = char_update["relationships"]
                if isinstance(char_update.get("traits"), list):
                    traits = [str(t).strip() for t in char_update["traits"] if str(t).strip()]
                    if traits:
                        clean["traits"] = traits
                arc = char_update.get("arc_progression")
                if isinstance(arc, str) and arc.strip():
                    clean["arc_progression"] = arc.strip()
                elif isinstance(arc, list):
                    arc_list = [str(a).strip() for a in arc if str(a).strip()]
                    if arc_list:
                        clean["arc_progression"] = arc_list
                if clean:
                    clean_chars[name] = clean
            if clean_chars:
                normalized["characters"] = clean_chars

        for key in ("world", "plot", "metadata"):
            if isinstance(updates.get(key), dict):
                normalized[key] = updates[key]

        return normalized

    def _is_blocking_issue(self, issue: dict) -> bool:
        if not isinstance(issue, dict):
            return False
        if self._looks_like_state_transition(issue):
            return False

        if isinstance(issue.get("blocking"), bool):
            return issue["blocking"]

        issue_type = str(issue.get("type", "")).lower()
        severity = str(issue.get("severity", "low")).lower()

        if issue_type in {"timeline_error", "location_error", "relationship_error", "logic_error", "character_contradiction"}:
            return severity in {"high", "medium"}

        return severity == "high"

    def _looks_like_state_transition(self, issue: dict) -> bool:
        issue_type = str(issue.get("type", "")).strip().lower()
        if issue_type in {"state_transition", "emotion_shift", "character_state_shift", "personality_shift"}:
            return True

        detail_blob = " ".join(
            [
                str(issue.get("detail", "")).lower(),
                str(issue.get("suggestion", "")).lower(),
            ]
        )
        if any(marker in detail_blob for marker in self._hard_contradiction_markers):
            return False

        if "neutral" in detail_blob and any(marker in detail_blob for marker in ("aroused", "predatory", "hungry", "satisfied", "possessive")):
            return True

        return any(marker in detail_blob for marker in self._transition_markers)
