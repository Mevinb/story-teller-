"""
Post-Generation Verifier — last-line-of-defense drift guard.

Runs AFTER the scene graph (writer -> critic -> editor) and BEFORE the scene
text is persisted to WIP / vector store / state.  Deterministic checks are
always on (free, fast, low false-positive).  An optional LLM pass (enabled when
a cloud model is active) catches softer issues like identity/alias swaps.

Detects:
- Entity status violations: a *dead* character acts on-page (revival).
- Premise-step leaks: forbidden future steps bleed into the current scene.
- Implicit new entities: capitalized names that do not resolve in the
  authoritative entity registry and are not sanctioned by the scene plan.

The premise-step leak detector is delegated to the orchestrator (single source
of truth) via a callable; tests can inject a stub.
"""
import logging
import re
from typing import Callable, List, Optional

from models.base import LLMInterface
from .contract import AgentContract

import config

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM = (
    "You are a Story Verifier — the final gate before prose is persisted.\n"
    "Your ONLY job is to catch hard continuity violations in generated fiction:\n"
    "1. A character who is DEAD or irrevocably incapacitated acting again without justification.\n"
    "2. Identity/alias swaps (a character referred to by another character's name mid-scene).\n"
    "3. Forbidden future premise beats leaking into the current chapter.\n"
    "4. New named entities appearing as if the reader already knows them (implicit introduction).\n"
    "You are NOT a creative writer and NOT a style critic. Only flag factual violations.\n"
    "ALWAYS respond with ONLY valid JSON."
)

VERIFIER_SYSTEM_COMPACT = (
    "You are a strict final story verifier. Catch: dead characters acting, identity/name swaps, "
    "future premise beats, and implicitly-introduced entities. Not a style critic. JSON only."
)

VERIFIER_PROMPT = (
    "Verify this final scene prose before it is persisted.\n\n"
    "=== GENERATED SCENE TEXT ===\n{scene_text}\n\n"
    "=== SCENE PLAN ===\n{scene_plan}\n\n"
    "=== CURRENT CHAPTER ===\n{chapter_num}\n"
    "=== NARRATIVE PHASE ===\n{narrative_phase}\n\n"
    "=== KNOWN ENTITIES (name -> status) ===\n{entity_summary}\n\n"
    "=== ALLOWED PREMISE STEPS THIS CHAPTER ===\n{allowed_steps}\n"
    "=== FORBIDDEN FUTURE PREMISE STEPS ===\n{forbidden_steps}\n\n"
    "Respond with this JSON:\n"
    '{{\n'
    '    "verified": true/false,\n'
    '    "issues": [\n'
    '        {{\n'
    '            "type": "dead_character_acting|identity_swap|future_step_leak|implicit_entity",\n'
    '            "detail": "What is wrong",\n'
    '            "severity": "high|medium|low",\n'
    '            "suggestion": "How to fix it"\n'
    '        }}\n'
    '    ]\n'
    '}}'
)

VERIFIER_PROMPT_COMPACT = (
    "Verify scene prose. ENTITIES: {entity_summary}\n"
    "ALLOWED: {allowed_steps}\nFORBIDDEN: {forbidden_steps}\n"
    "SCENE:\n{scene_text}\n\n"
    'Return JSON: verified (bool), issues[{{type: "dead_character_acting|identity_swap|'
    'future_step_leak|implicit_entity", detail, severity, suggestion}}].'
)

# Tokens that look like names but are almost never entities.
_NON_ENTITY_TOKENS = {
    "there", "where", "these", "those", "which", "would", "could", "should",
    "their", "there", "about", "after", "again", "against", "being", "before",
    "between", "though", "through", "while", "within", "without", "someone",
    "somebody", "everyone", "everything", "nothing", "anything", "something",
    "maybe", "however", "because", "although", "already", "always",
    "mother", "father", "sister", "brother", "the", "this", "that", "then",
}
# Words allowed at the start of a quoted line of dialogue (speaker attribute).
_DIALOGUE_LABELS = {"said", "asked", "whispered", "shouted", "replied", "murmured",
                    "muttered", "cried", "yelled", "added", "began", "continued"}


class Verifier(AgentContract):
    name = "verifier"

    def __init__(self, model: Optional[LLMInterface] = None):
        self.model = model
        self._compact_mode = False

    def set_compact_mode(self, enabled: bool) -> None:
        self._compact_mode = bool(enabled)

    def run(self, state: dict) -> dict:
        report = self.verify(
            scene_text=state.get("scene_text", ""),
            state_manager=state.get("state_manager"),
            scene_plan=state.get("scene_plan", {}),
            chapter_num=state.get("chapter_num", 0),
            allowed_steps=state.get("allowed_steps"),
            forbidden_steps=state.get("forbidden_steps"),
            narrative_phase=state.get("narrative_phase", ""),
            premise_violation_fn=state.get("premise_violation_fn"),
            run_llm=state.get("run_llm", False),
        )
        has_blocking = bool(report.get("blocking_issues"))
        return {
            "output": {"report": report},
            "confidence": 0.95 if not has_blocking else 0.4,
            "next_action": "review" if has_blocking else "persist",
        }

    def verify(
        self,
        scene_text: str,
        state_manager,
        scene_plan: Optional[dict] = None,
        chapter_num: int = 0,
        allowed_steps: Optional[list] = None,
        forbidden_steps: Optional[list] = None,
        narrative_phase: str = "",
        premise_violation_fn: Optional[Callable] = None,
        run_llm: bool = False,
    ) -> dict:
        """Run the verifier against final scene prose.

        Args:
            scene_text: Final prose (post-editor).
            state_manager: StateManager with the entity registry.
            scene_plan: The scene plan for this scene.
            chapter_num: Current chapter number.
            allowed_steps: Premise steps allowed this chapter (LLM pass only).
            forbidden_steps: Premise steps that must not appear yet (LLM pass only).
            narrative_phase: Current narrative phase label (LLM pass only).
            premise_violation_fn: Optional callable ``fn(scene_text, chapter_num,
                state) -> list[str]`` returning future-premise violation messages.
                The orchestrator passes its existing ``_future_premise_violations``.
            run_llm: Enable the optional LLM refinement pass.

        Returns:
            Verification report dict.
        """
        blocking_issues: List[dict] = []
        non_blocking_issues: List[dict] = []
        checks = ["entity_status", "premise_leak", "implicit_entity"]

        # 1. Entity status violations (dead characters acting).
        status_issues = self._check_entity_status(scene_text, state_manager, scene_plan or {})
        for issue in status_issues:
            (blocking_issues if issue.get("blocking") else non_blocking_issues).append(issue)

        # 2. Premise-step leak — delegated to the orchestrator's detector when
        #    available, so there is a single source of truth.
        if premise_violation_fn is not None:
            try:
                state = getattr(state_manager, "state", None) or {}
                violations = premise_violation_fn(scene_text, chapter_num, state)
                for violation in violations:
                    blocking_issues.append({
                        "type": "future_step_leak",
                        "detail": str(violation),
                        "severity": "high",
                        "blocking": True,
                        "source": "verifier",
                        "suggestion": "Remove or defer this beat until its premise step becomes allowed.",
                    })
            except Exception as e:
                logger.warning("Verifier premise-leak check failed (non-fatal): %s", e)

        # 3. Implicit new entities — non-blocking notes only.
        implicit = self._check_implicit_entities(scene_text, state_manager, scene_plan or {})
        non_blocking_issues.extend(implicit)

        # 4. Optional LLM refinement pass.
        if run_llm and self.model is not None:
            checks.append("llm_pass")
            llm_issues = self._llm_pass(
                scene_text=scene_text,
                state_manager=state_manager,
                scene_plan=scene_plan or {},
                chapter_num=chapter_num,
                allowed_steps=allowed_steps or [],
                forbidden_steps=forbidden_steps or [],
                narrative_phase=narrative_phase,
            )
            for issue in llm_issues:
                (blocking_issues if issue.get("blocking") else non_blocking_issues).append(issue)

        blocking_issues = [self._normalize_issue(i) for i in blocking_issues]
        non_blocking_issues = [self._normalize_issue(i) for i in non_blocking_issues]

        return {
            "is_consistent": not blocking_issues,
            "blocking_issues": blocking_issues,
            "non_blocking_issues": non_blocking_issues,
            "issues": blocking_issues + non_blocking_issues,
            "signatures": sorted({
                self.issue_signature(issue) for issue in blocking_issues
            }),
            "checks": checks,
            "needs_review": False,
        }

    # ─── Deterministic checks ────────────────────────────────────────

    def _check_entity_status(self, scene_text: str, state_manager, scene_plan: dict) -> list[dict]:
        """Block if a character the plan lists as present is dead."""
        issues = []
        if state_manager is None:
            return issues
        present = scene_plan.get("characters_present") or []
        for char in present:
            if not isinstance(char, str) or len(char) <= 2:
                continue
            status = state_manager.get_entity_status(char)
            if status == "dead":
                issues.append({
                    "type": "dead_character_acting",
                    "detail": (
                        f"Scene plan requires character '{char}' to be present and "
                        "acting, but they are marked DEAD in the entity registry."
                    ),
                    "severity": "high",
                    "blocking": True,
                    "source": "verifier",
                    "suggestion": (
                        f"Revise the scene so '{char}' does not physically act "
                        "(appear only in memory/dialogue), or explicitly justify a revival."
                    ),
                })
            elif status == "missing":
                issues.append({
                    "type": "dead_character_acting",
                    "detail": (
                        f"Character '{char}' is marked MISSING but is present in this scene."
                    ),
                    "severity": "medium",
                    "blocking": False,
                    "source": "verifier",
                    "suggestion": "Show how/why the missing character has returned.",
                })
        return issues

    def _check_implicit_entities(self, scene_text: str, state_manager, scene_plan: dict) -> list[dict]:
        """Surface capitalized names that do not resolve to known entities.

        Non-blocking: background characters are legitimately introduced during a
        story; the note exists so the human can decide whether to register them.
        """
        issues = []
        if state_manager is None:
            return issues
        allowed_names = {str(c).strip().lower() for c in (scene_plan.get("characters_present") or [])}
        for token in self._candidate_names(scene_text):
            if token.lower() in allowed_names:
                continue
            if state_manager.resolve_entity_name(token):
                continue
            display = self._original_case(token, scene_text)
            issues.append({
                "type": "implicit_entity",
                "detail": (
                    f"Name '{display}' appears in prose but is not registered as a known "
                    "entity (and is not listed in the scene plan)."
                ),
                "severity": "low",
                "blocking": False,
                "source": "verifier",
                "suggestion": "Register the entity if it is an intentional new character, "
                              "or remove the name if it is an accidental invention.",
            })
        return issues

    @staticmethod
    def _original_case(token: str, scene_text: str) -> str:
        match = re.search(
            r"\b" + re.escape(token) + r"[A-Za-z'-]*\b",
            scene_text or "",
            flags=re.IGNORECASE,
        )
        return match.group(0) if match else token

    @staticmethod
    def _candidate_names(scene_text: str) -> set[str]:
        """Extract likely proper-noun names from prose with low false positives.

        A candidate must be capitalized, >= 3 chars, appear at least twice, and
        not be a sentence-start filler word.
        """
        if not scene_text:
            return set()
        counts: dict[str, int] = {}
        sentences = re.split(r"(?<=[.!?])\s+|\n+", scene_text)
        for sentence in sentences:
            tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
            if not tokens:
                continue
            # Skip the first token of each sentence (usually capitalised by grammar).
            for token in tokens[1:]:
                if not token or token[0].isupper():
                    key = token.lower()
                    if len(key) >= 3 and key not in _NON_ENTITY_TOKENS and key not in _DIALOGUE_LABELS:
                        counts[key] = counts.get(key, 0) + 1
        return {t for t, c in counts.items() if c >= 2}

    # ─── Optional LLM pass ───────────────────────────────────────────

    def _llm_pass(self, scene_text, state_manager, scene_plan, chapter_num,
                  allowed_steps, forbidden_steps, narrative_phase) -> list[dict]:
        if state_manager is None:
            return []
        entity_summary = self._entity_summary(state_manager)
        template = VERIFIER_PROMPT_COMPACT if self._compact_mode else VERIFIER_PROMPT
        prompt = template.format(
            scene_text=(scene_text or "")[:2500],
            scene_plan=self._format_plan(scene_plan)[:900],
            chapter_num=chapter_num,
            narrative_phase=narrative_phase or "unknown",
            entity_summary=entity_summary,
            allowed_steps=" | ".join(allowed_steps)[:1200] if allowed_steps else "None.",
            forbidden_steps=" | ".join(forbidden_steps)[:1200] if forbidden_steps else "None.",
        )
        schema = {
            "type": "object",
            "properties": {
                "verified": {"type": "boolean"},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "detail": {"type": "string"},
                            "severity": {"type": "string"},
                            "suggestion": {"type": "string"},
                        },
                        "required": ["type", "detail", "severity"],
                    },
                },
            },
            "required": ["verified", "issues"],
        }
        try:
            response = self.model.generate_with_retry(
                prompt=prompt,
                system=VERIFIER_SYSTEM_COMPACT if self._compact_mode else VERIFIER_SYSTEM,
                schema=schema,
                temperature=config.AGENT_TEMPERATURES["critic"],
                max_tokens=600 if self._compact_mode else None,
            )
            result = response.as_json()
        except Exception as e:
            logger.warning("Verifier LLM pass failed (non-fatal): %s", e)
            return []
        if not isinstance(result, dict):
            return []
        raw_issues = result.get("issues", [])
        if isinstance(raw_issues, dict):
            raw_issues = [raw_issues]
        if not isinstance(raw_issues, list):
            return []

        blocking_types = {"dead_character_acting", "identity_swap", "future_step_leak"}
        issues = []
        for issue in raw_issues:
            if not isinstance(issue, dict):
                continue
            itype = str(issue.get("type", ""))
            severity = str(issue.get("severity", "low"))
            blocking = itype in blocking_types or severity == "high"
            issues.append({
                "type": itype or "verification",
                "detail": str(issue.get("detail", "")),
                "severity": severity,
                "blocking": blocking,
                "source": "verifier_llm",
                "suggestion": str(issue.get("suggestion", "")),
            })
        return issues

    @staticmethod
    def _entity_summary(state_manager) -> str:
        entities = state_manager.get_entities() if hasattr(state_manager, "get_entities") else {}
        if not entities:
            return "No entities registered."
        lines = []
        for entity_id, record in entities.items():
            if not isinstance(record, dict):
                continue
            lines.append(
                f"{record.get('canonical_name', entity_id)} "
                f"[{record.get('type', 'character')}:{record.get('status', 'active')}]"
            )
        return "\n".join(lines) if lines else "No entities registered."

    @staticmethod
    def _format_plan(scene_plan: dict) -> str:
        try:
            import json
            return json.dumps(scene_plan, ensure_ascii=False)[:900]
        except Exception:
            return str(scene_plan)[:900]

    # ─── Issue helpers ───────────────────────────────────────────────

    def has_blocking_issues(self, report: dict) -> bool:
        return bool(report.get("blocking_issues"))

    def get_blocking_issues(self, report: dict) -> list[dict]:
        return report.get("blocking_issues", [])

    def get_non_blocking_issues(self, report: dict) -> list[dict]:
        return report.get("non_blocking_issues", [])

    @staticmethod
    def issue_signature(issue: dict) -> str:
        itype = str(issue.get("type", ""))
        detail = re.sub(r"[^a-z0-9]+", " ", str(issue.get("detail", "")).lower()).strip()
        return f"{itype}::{detail[:80]}"

    @staticmethod
    def _normalize_issue(issue: dict) -> dict:
        return {
            "type": str(issue.get("type", "verification")),
            "detail": str(issue.get("detail", "")),
            "severity": str(issue.get("severity", "low")),
            "blocking": bool(issue.get("blocking", False)),
            "source": str(issue.get("source", "verifier")),
            "suggestion": str(issue.get("suggestion", "")),
        }
