"""
Consistency Engine — Post-generation validator.
Compares generated text against the structured state to detect
character contradictions, timeline errors, and logical conflicts.
Also performs a second premise-alignment pass to catch drift.
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

CONSISTENCY_SYSTEM_COMPACT = (
    "You are a strict continuity validator. Compare scene vs plan/state and report only factual contradictions. "
    "Do not flag style choices. Mutable emotional/relationship shifts are valid if shown in-scene. "
    "Output JSON only."
)

VALIDATE_PROMPT = (
    "Validate this scene against the established story state.\n\n"
    "=== GENERATED SCENE TEXT ===\n{scene_text}\n\n"
    "=== REQUIRED SCENE PLAN ===\n{scene_plan}\n\n"
    "=== ESTABLISHED STATE ===\n{state_context}\n\n"
    "=== INSTRUCTIONS ===\n"
    "Check for:\n"
    "1. Character trait contradictions (e.g., shy character acting boldly without development)\n"
    "2. Relationship inconsistencies (e.g., allies acting as enemies without cause)\n"
    "3. Location errors (character in wrong place)\n"
    "4. Timeline conflicts (events out of order)\n"
    "5. Missing or contradicted facts\n"
    "6. Character evolution and personality/emotion shifts that are justified in-scene\n\n"
    "The scene plan is the intended next beat. The established state is the continuity before the scene.\n"
    "If the scene plan requires a new location or situation, accept it only when the generated scene shows\n"
    "a plausible transition from the established state. Flag the scene if it ignores the required plan.\n"
    "If required key events are missing on-page (the text jumps to aftermath instead), mark that as a blocking logic_error.\n\n"
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
    '            "suggestion": "How to fix it",\n'
    '            "nearest_paragraph": "Exact sentence/paragraph from the prose where the missing event should have occurred (if applicable)"\n'
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

VALIDATE_PROMPT_COMPACT = (
    "Validate scene continuity.\n\n"
    "SCENE:\n{scene_text}\n\n"
    "PLAN:\n{scene_plan}\n\n"
    "STATE:\n{state_context}\n\n"
    "Check: character, relationship, location, timeline, logic contradictions. "
    "Required key events must happen on-page; aftermath-only scenes for required events are blocking logic errors. "
    "If emotional/personality/sexual status changes are shown in-scene, treat as state_updates (non-blocking). "
    "Return JSON: is_consistent, issues[{{type,detail,severity,blocking,suggestion,nearest_paragraph}}], state_updates."
)

# ── Premise Alignment Pass ──────────────────────────────────────────────────
PREMISE_ALIGNMENT_SYSTEM = (
    "You are a Premise Alignment Auditor. Your ONLY job is to verify that a "
    "generated scene stays within the boundaries of the active story premise steps.\n"
    "You do NOT check prose quality or character consistency — only premise adherence.\n"
    "ALWAYS respond with ONLY valid JSON."
)

PREMISE_ALIGNMENT_PROMPT = (
    "Audit this scene for premise alignment.\n\n"
    "=== ACTIVE PREMISE STEPS (allowed content for this chapter) ===\n{allowed_steps}\n\n"
    "=== FORBIDDEN FUTURE STEPS (must NOT appear yet) ===\n{forbidden_steps}\n\n"
    "=== CURRENT NARRATIVE PHASE ===\n{narrative_phase}\n"
    "=== EMOTIONAL INTENSITY TARGET ===\n{intensity_target}/1.0\n\n"
    "=== GENERATED SCENE TEXT ===\n{scene_text}\n\n"
    "=== AUDIT QUESTIONS ===\n"
    "1. Does this scene serve an ALLOWED premise step? "
    "If not, identify which forbidden future step it resembles.\n"
    "2. Has any FORBIDDEN future step leaked into the prose? "
    "Quote the specific passage if yes.\n"
    "3. Is the emotional intensity appropriate for the current narrative phase "
    "(not peaking too early, not too flat for a climax)?\n\n"
    "Respond with this JSON:\n"
    '{{\n'
    '    "premise_aligned": true/false,\n'
    '    "violations": [\n'
    '        {{\n'
    '            "type": "future_step_leak|wrong_premise_step|intensity_mismatch",\n'
    '            "detail": "What specifically is wrong",\n'
    '            "quoted_passage": "The exact sentence from the scene causing the issue",\n'
    '            "suggestion": "How the writer should fix it"\n'
    '        }}\n'
    '    ]\n'
    '}}'
)

PREMISE_ALIGNMENT_PROMPT_COMPACT = (
    "Audit scene for premise alignment.\n\n"
    "ALLOWED STEPS:\n{allowed_steps}\n"
    "FORBIDDEN STEPS:\n{forbidden_steps}\n"
    "PHASE: {narrative_phase} | INTENSITY TARGET: {intensity_target}/1.0\n"
    "SCENE:\n{scene_text}\n\n"
    "Return JSON: premise_aligned (bool), "
    "violations[{{type, detail, quoted_passage, suggestion}}]. "
    "Types: future_step_leak|wrong_premise_step|intensity_mismatch"
)


class ConsistencyEngine(AgentContract):
    """
    Validates generated scenes against the structured state.
    This is a validator, not a generator — it checks for factual errors.
    Runs two passes: internal consistency + premise alignment.
    """

    def __init__(self, model: LLMInterface):
        self.name = "critic"
        self.model = model
        self._compact_mode = False
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
        self._state_hygiene_markers = {
            "listed as both",
            "described as both",
            "in her traits",
            "in his traits",
            "in their traits",
            "in the character state",
            "in the established state",
            "mutually exclusive",
        }

    def set_compact_mode(self, enabled: bool):
        self._compact_mode = bool(enabled)

    def run(self, state: dict) -> dict:
        report = self.validate(
            scene_text=state["scene_text"],
            state_context=state["state_context"],
            scene_plan=state.get("scene_plan", {}),
        )
        has_blocking = self.has_blocking_issues(report)
        confidence = 0.95 if report.get("is_consistent", True) else 0.4
        return {
            "output": {"report": report},
            "confidence": confidence,
            "next_action": "decision" if has_blocking else "editor",
        }

    def validate(
        self,
        scene_text: str,
        state_context: str,
        scene_plan: dict = None,
        allowed_steps: list = None,
        forbidden_steps: list = None,
        narrative_phase: str = "",
        intensity_target: float = 0.5,
    ) -> dict:
        """
        Validate a scene against the current state (pass 1: internal consistency)
        then against the active premise window (pass 2: premise alignment).
        Then run pass 3: deterministic beat-presence check against scene.summary.

        Args:
            scene_text: The generated prose to validate
            state_context: Context from the Retriever (state + past content)
            scene_plan: The intended scene plan
            allowed_steps: Premise steps permitted this chapter
            forbidden_steps: Premise steps that must not appear yet
            narrative_phase: Current story phase label
            intensity_target: Expected emotional intensity 0.0–1.0

        Returns:
            Validation report dict with is_consistent, issues, state_updates
        """
        scene_text_limit = 1600 if self._compact_mode else 3000
        scene_plan_limit = 650 if self._compact_mode else 1200
        state_context_limit = 900 if self._compact_mode else 2000
        template = VALIDATE_PROMPT_COMPACT if self._compact_mode else VALIDATE_PROMPT
        prompt = template.format(
            scene_text=scene_text[:scene_text_limit],
            scene_plan=self._format_scene_plan(scene_plan or {})[:scene_plan_limit],
            state_context=state_context[:state_context_limit],
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
            system=CONSISTENCY_SYSTEM_COMPACT if self._compact_mode else CONSISTENCY_SYSTEM,
            schema=schema,
            temperature=config.AGENT_TEMPERATURES["critic"],
            max_tokens=700 if self._compact_mode else None,
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

        # ── Pass 2: Premise alignment ──────────────────────────────────────
        if allowed_steps is not None or forbidden_steps is not None:
            alignment_issues = self._check_premise_alignment(
                scene_text=scene_text,
                allowed_steps=allowed_steps or [],
                forbidden_steps=forbidden_steps or [],
                narrative_phase=narrative_phase or "unknown",
                intensity_target=intensity_target,
            )
            if alignment_issues:
                result["issues"].extend(alignment_issues)
                result["is_consistent"] = False
                logger.warning(
                    "Premise alignment check found %d blocking violation(s).",
                    len(alignment_issues),
                )

        # ── Pass 3: Deterministic beat-presence check ────────────────────
        # This is a completion gate: the scene cannot pass as consistent until
        # every beat from the summary checklist is confirmed present in the prose.
        beat_issues = self._check_beat_presence(scene_text, scene_plan or {})
        if beat_issues:
            result["issues"].extend(beat_issues)
            # Completion gate: force is_consistent False regardless of LLM verdict
            result["is_consistent"] = False
            logger.warning(
                "Beat-presence check found %d missing beat(s). Scene blocked until all beats present.",
                len(beat_issues),
            )

        return result

    def _check_premise_alignment(
        self,
        scene_text: str,
        allowed_steps: list,
        forbidden_steps: list,
        narrative_phase: str,
        intensity_target: float,
    ) -> list:
        """
        Run the premise-alignment LLM pass. Returns a list of blocking issues
        (empty list = scene is aligned).
        """
        scene_text_limit = 1600 if self._compact_mode else 3000
        allowed_text = "\n".join(f"- {s}" for s in allowed_steps) or "(none specified)"
        forbidden_text = "\n".join(f"- {s}" for s in forbidden_steps) or "(none specified)"

        template = PREMISE_ALIGNMENT_PROMPT_COMPACT if self._compact_mode else PREMISE_ALIGNMENT_PROMPT
        prompt = template.format(
            scene_text=scene_text[:scene_text_limit],
            allowed_steps=allowed_text,
            forbidden_steps=forbidden_text,
            narrative_phase=narrative_phase,
            intensity_target=round(intensity_target, 2),
        )

        schema = {
            "type": "object",
            "properties": {
                "premise_aligned": {"type": "boolean"},
                "violations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "detail": {"type": "string"},
                            "quoted_passage": {"type": "string"},
                            "suggestion": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["premise_aligned", "violations"],
        }

        try:
            response = self.model.generate_with_retry(
                prompt=prompt,
                system=PREMISE_ALIGNMENT_SYSTEM,
                schema=schema,
                temperature=config.AGENT_TEMPERATURES["critic"],
                max_tokens=600 if self._compact_mode else None,
            )
            result = response.as_json()
        except Exception as exc:
            logger.warning("Premise alignment check failed: %s", exc)
            return []

        if not isinstance(result, dict):
            return []
        if result.get("premise_aligned", True):
            return []

        blocking_issues = []
        for v in (result.get("violations") or []):
            if not isinstance(v, dict):
                continue
            detail = str(v.get("detail", "")).strip()
            quoted = str(v.get("quoted_passage", "")).strip()
            suggestion = str(v.get("suggestion", "")).strip()
            full_detail = detail
            if quoted:
                full_detail += f" | Offending passage: \"{quoted}\""
            blocking_issues.append({
                "type": "premise_alignment",
                "detail": full_detail,
                "severity": "high",
                "blocking": True,
                "suggestion": suggestion,
            })

        return blocking_issues

    @staticmethod
    def _extract_beats(scene_plan: dict) -> list[str]:
        """
        Extract individual beats from the scene plan.
        Prefers required_beats; falls back to splitting original_user_brief / summary
        on periods and commas, the same way the writer prompt does.
        """
        required = [
            str(b).strip() for b in scene_plan.get("required_beats", [])
            if str(b).strip()
        ]
        if required:
            return required

        source = (
            str(scene_plan.get("original_user_brief", "")).strip()
            or str(scene_plan.get("summary", "")).strip()
        )
        if not source:
            return []

        beats = [
            b.strip().strip('"\'') for b in re.split(r'[.,]+', source)
            if len(b.strip()) > 6
        ]
        return beats

    def _check_beat_presence(self, scene_text: str, scene_plan: dict) -> list[dict]:
        """
        Deterministic check: for each beat derived from scene.summary (or
        required_beats / original_user_brief), verify that the key phrases from
        that beat appear somewhere in the generated prose.
        Returns a list of blocking issues for any beat not found.
        """
        beats = self._extract_beats(scene_plan)
        if not beats:
            return []

        prose_lower = scene_text.lower()
        missing_issues = []

        for beat in beats:
            beat_lower = beat.lower()
            # Tokenise the beat into meaningful keywords (≥4 chars, skip stop-words)
            _STOP = {
                "then", "after", "before", "during", "while", "with", "from", "into",
                "they", "them", "this", "that", "their", "scene", "event", "chapter",
                "have", "does", "will", "when", "where", "what", "which", "also",
            }
            keywords = [
                token for token in re.findall(r"[a-z0-9']+", beat_lower)
                if len(token) >= 4 and token not in _STOP
            ]

            # A beat is considered present if:
            #   (a) its full lowercased text is a substring of the prose, OR
            #   (b) at least half of its meaningful keywords appear in the prose.
            full_match = beat_lower in prose_lower
            if not full_match and keywords:
                matched = sum(1 for kw in keywords if kw in prose_lower)
                full_match = matched >= max(1, len(keywords) // 2)

            if not full_match:
                missing_issues.append({
                    "type": "logic_error",
                    "detail": f"Required beat not found in prose: {beat}",
                    "severity": "high",
                    "blocking": True,
                    "suggestion": (
                        f"Add on-page prose that explicitly covers this beat: '{beat}'. "
                        "Do not summarise it in passing — show it happening."
                    ),
                })

        return missing_issues

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

        if self._looks_like_state_hygiene_issue(normalized):
            normalized["type"] = "state_hygiene"
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
        if self._looks_like_state_hygiene_issue(issue):
            return False

        issue_type = str(issue.get("type", "")).lower()
        severity = str(issue.get("severity", "low")).lower()
        detail_blob = " ".join(
            [
                str(issue.get("detail", "")).lower(),
                str(issue.get("suggestion", "")).lower(),
            ]
        )

        if issue_type in {"timeline_error", "location_error", "relationship_error", "logic_error", "character_contradiction"}:
            if any(marker in detail_blob for marker in self._hard_contradiction_markers):
                return True
            return severity in {"high", "medium"}

        if isinstance(issue.get("blocking"), bool):
            return issue["blocking"]

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

    def _looks_like_state_hygiene_issue(self, issue: dict) -> bool:
        detail_blob = " ".join(
            [
                str(issue.get("detail", "")).lower(),
                str(issue.get("suggestion", "")).lower(),
            ]
        )
        if any(marker in detail_blob for marker in self._hard_contradiction_markers):
            return False
        if any(marker in detail_blob for marker in self._state_hygiene_markers):
            return True

        # Common stale-trait contradiction pattern in stored state.
        has_virgin = "virgin" in detail_blob
        has_experienced = "experienced" in detail_blob or "post-virginity" in detail_blob
        return has_virgin and has_experienced and (
            "listed as both" in detail_blob or "described as both" in detail_blob
        )

    @staticmethod
    def _format_scene_plan(scene_plan: dict) -> str:
        if not isinstance(scene_plan, dict) or not scene_plan:
            return "No scene plan provided."
        fields = [
            ("Summary", scene_plan.get("summary", "")),
            ("Required setting", scene_plan.get("location", "")),
            ("Required characters", ", ".join(scene_plan.get("characters_present", []))),
            ("Required events", "; ".join(scene_plan.get("key_events", []))),
            ("Mood", scene_plan.get("mood", "")),
        ]
        return "\n".join(f"{label}: {value}" for label, value in fields if value)
