"""
Gemini-powered story combiner and polisher.

Executes a rigid three-pass completion pipeline:
1. Semantic Coverage Analysis & Pre-evaluation (Local + Gemini)
2. Pass 1: Reconciliation (Fixing conflicts, removing unmapped extras)
3. Pass 2: Completion (Writing missing premise beats)
4. Pass 3: Polish (Fixing flow and grammar)
5. Post-Processing & Final Verification
"""
import os
import re
import json
import logging
import time
import random
from functools import lru_cache
from typing import Callable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

import config

logger = logging.getLogger(__name__)


# ── Cached Embedder Singleton ─────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    """Return a cached SentenceTransformer instance (loaded once per process)."""
    model_name = getattr(config, "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    logger.info("Loading shared SentenceTransformer: %s", model_name)
    return SentenceTransformer(model_name)


_API_REQUEST_LOG = []

def _estimate_tokens(contents) -> int:
    """Rough estimation of token count from contents (approx 4 chars per token)."""
    text_len = 0
    if isinstance(contents, str):
        text_len = len(contents)
    elif isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                text_len += len(item)
            elif hasattr(item, "text"):
                text_len += len(item.text)
    return max(1, text_len // 4)

def _throttle_request(estimated_tokens: int):
    """Proactively sleep if near the TPM limit to prevent 429 rate limits."""
    global _API_REQUEST_LOG
    now = time.time()
    _API_REQUEST_LOG = [entry for entry in _API_REQUEST_LOG if now - entry[0] < 60.0]
    
    current_tpm = sum(entry[1] for entry in _API_REQUEST_LOG)
    tpm_limit = getattr(config, "GEMINI_TPM_LIMIT", 200000)
    if current_tpm + estimated_tokens > 0.85 * tpm_limit:
        sleep_needed = 60.0 - (now - _API_REQUEST_LOG[0][0]) if _API_REQUEST_LOG else 10.0
        sleep_needed = max(5.0, min(sleep_needed, 60.0))
        sleep_needed += random.uniform(1.0, 3.0)
        logger.warning(f"Proactive rate limiting: Current minute tokens ({current_tpm}) + request ({estimated_tokens}) exceeds 85% of limit. Throttling/sleeping for {sleep_needed:.2f}s...")
        time.sleep(sleep_needed)
        now = time.time()
        _API_REQUEST_LOG = [entry for entry in _API_REQUEST_LOG if now - entry[0] < 60.0]
        
    _API_REQUEST_LOG.append((now, estimated_tokens))

def _generate_content_with_retry(client: genai.Client, model: str, contents, config=None, max_retries: int = 5, initial_delay: float = 10.0, is_cancelled: Optional[Callable[[], bool]] = None):
    """Generate content with TPM-aware throttling and exponential backoff retry."""
    if is_cancelled and is_cancelled():
        raise RuntimeError("Cancellation requested by user")
    est_tokens = _estimate_tokens(contents)
    _throttle_request(est_tokens)
    
    delay = initial_delay
    for attempt in range(max_retries):
        if is_cancelled and is_cancelled():
            raise RuntimeError("Cancellation requested by user")
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                sleep_time = delay + random.uniform(0.5, 2.0)
                logger.warning(f"Gemini API Rate limit hit (429/Resource Exhausted). Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                slept = 0.0
                while slept < sleep_time:
                    if is_cancelled and is_cancelled():
                        raise RuntimeError("Cancellation requested by user")
                    time.sleep(0.5)
                    slept += 0.5
                delay *= 2.0
            else:
                raise e

_CHAPTER_FILE_RE = re.compile(r"^chapter_(\d{3,})\.md$")
_CHAPTER_HEADER_RE = re.compile(r"^(#+\s*)(Chapter\s+\d+)(.*?)$", re.IGNORECASE | re.MULTILINE)
_SCENE_BREAK_RE = re.compile(r"\n{2,}(?:\*\s*\*\s*\*|--+|===+)\n{2,}")


# ── Global Story Digest ──────────────────────────────────────────────
def _build_story_digest(state: dict, max_chars: int = 3000) -> str:
    """Compress state.json into a canonical facts block for polish prompt injection.

    This provides the polisher with ground-truth character names, relationships,
    world rules, motifs, and timeline so it never introduces factual errors.
    """
    if not state:
        return ""

    parts = []

    # Canonical character spellings and profiles
    characters = state.get("characters", {})
    if characters:
        char_lines = []
        for name, info in characters.items():
            if not isinstance(info, dict):
                continue
            desc = info.get("description", "")
            role = info.get("role", "supporting")
            rels = info.get("relationships", {})
            rel_str = "; ".join(f"{k}: {v}" for k, v in rels.items()) if rels else ""
            status = info.get("status", "active")
            line = f"  {name} ({role}, {status})"
            if desc:
                line += f": {desc[:80]}"
            if rel_str:
                line += f" | Rels: {rel_str[:100]}"
            char_lines.append(line)
        if char_lines:
            parts.append("CANONICAL CHARACTERS:\n" + "\n".join(char_lines))

    # World rules
    world = state.get("world", {})
    rules = world.get("rules", [])
    if rules:
        parts.append("WORLD RULES:\n" + "\n".join(f"  - {r}" for r in rules[:8]))

    # Locations
    locations = world.get("locations", {})
    if locations:
        loc_lines = [f"  {name}: {desc[:60]}" for name, desc in list(locations.items())[:10]]
        parts.append("LOCATIONS:\n" + "\n".join(loc_lines))

    # Motifs
    motifs = world.get("motifs", {})
    if motifs:
        motif_lines = []
        for key, data in motifs.items():
            if isinstance(data, dict) and data.get("mentions", 0) > 0:
                name_fmt = key.replace("_", " ").title()
                meaning = data.get("meaning", "")
                motif_lines.append(f"  {name_fmt}: {meaning}" if meaning else f"  {name_fmt}")
        if motif_lines:
            parts.append("RECURRING MOTIFS:\n" + "\n".join(motif_lines[:6]))

    # Timeline summary (chapter summaries)
    plot = state.get("plot", {})
    summaries = plot.get("chapter_summaries", [])
    if summaries:
        timeline = []
        for s in summaries[-12:]:
            ch = s.get("chapter", "?")
            text = s.get("summary", "")[:80]
            timeline.append(f"  Ch{ch}: {text}")
        parts.append("TIMELINE:\n" + "\n".join(timeline))

    # Unresolved threads
    threads = plot.get("unresolved_threads", [])
    if threads:
        parts.append("UNRESOLVED THREADS:\n" + "\n".join(f"  - {t}" for t in threads[:5]))

    digest = "\n\n".join(parts)
    if len(digest) > max_chars:
        digest = digest[:max_chars - 3] + "..."
    return digest


# ── Semantic Preservation Validator ───────────────────────────────────
def _semantic_preservation_check(
    original: str,
    candidate: str,
    threshold: float = 0.96,
    chunk_size: int = 200,
) -> tuple[bool, float]:
    """Check whether a polished draft preserves the semantic content of the original.

    Returns (passed: bool, coverage_score: float).
    Uses embedding cosine similarity of chunked text.
    Falls back to length ratio if embedder is unavailable.
    """
    if not original or not candidate:
        return True, 1.0

    try:
        embedder = _get_embedder()
    except Exception:
        # Fallback to length ratio
        ratio = len(candidate) / max(1, len(original))
        return ratio >= 0.92, ratio

    def _chunk(text: str) -> list[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        if not paragraphs:
            # Fall back to sentence-level chunks
            paragraphs = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
        return paragraphs[:500]  # Cap to avoid OOM

    orig_chunks = _chunk(original)
    cand_chunks = _chunk(candidate)

    if not orig_chunks or not cand_chunks:
        ratio = len(candidate) / max(1, len(original))
        return ratio >= 0.92, ratio

    try:
        orig_embs = embedder.encode(orig_chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        cand_embs = embedder.encode(cand_chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

        # For each original chunk, find max similarity to any candidate chunk
        sim_matrix = np.dot(orig_embs, cand_embs.T)
        max_sims = np.max(sim_matrix, axis=1)
        coverage = float(np.mean(max_sims >= 0.75))  # Fraction of original chunks preserved
    except Exception as e:
        logger.warning("Semantic preservation check failed: %s. Falling back to length ratio.", e)
        ratio = len(candidate) / max(1, len(original))
        return ratio >= 0.92, ratio

    return coverage >= threshold, round(coverage, 4)


def _is_substantially_shorter(candidate: str, source: str, min_ratio: float = 0.90) -> bool:
    if not source:
        return False
    return len(candidate or "") < min_ratio * len(source)


def _preserve_length_or_fallback(
    candidate: str,
    source: str,
    stage_name: str,
    min_ratio: float = 0.90,
) -> str:
    """Reject LLM outputs that look like summaries instead of preserved full drafts."""
    if _is_substantially_shorter(candidate, source, min_ratio):
        logger.warning(
            "%s output was too short (%s vs %s chars). Falling back to previous full draft.",
            stage_name,
            len(candidate or ""),
            len(source or ""),
        )
        return source
    return candidate


def _read_chapters(chapters_dir: str) -> list[tuple[int, str]]:
    """Return a sorted list of (chapter_number, chapter_text) tuples."""
    if not os.path.isdir(chapters_dir):
        return []
    chapters = []
    for fname in sorted(os.listdir(chapters_dir)):
        match = _CHAPTER_FILE_RE.match(fname)
        if not match:
            continue
        num = int(match.group(1))
        path = os.path.join(chapters_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                chapters.append((num, f.read()))
        except OSError:
            logger.warning("Could not read %s", path)
    return chapters


def _fingerprint(text: str) -> str:
    """Create a normalised fingerprint of text for fast exact-match dedup."""
    return re.sub(r"\s+", " ", text.lower().strip())[:200]


def _dedup_paragraphs(text: str, similarity_threshold: float = 0.75) -> str:
    """Remove fuzzy duplicate paragraphs while preserving order.

    Two-pass approach:
      Pass 1 — paragraph-level: uses exact matches, vector embeddings for conceptual loops, 
               and difflib as a fallback against previous prose paragraphs.
      Pass 2 — scene-level: splits on scene breaks (* * *, ---, ===) and
               removes entire duplicate scene blocks.
    """
    import difflib

    # ── Pass 1: paragraph-level dedup ────────────────────────────────
    paragraphs = re.split(r"\n{2,}", text.strip())
    if not paragraphs:
        return text

    seen_fingerprints: set[str] = set()
    prose_paragraphs: list[str] = []   # only prose (no headings/breaks)
    recent_embeddings: list[np.ndarray] = []  # sliding window of embeddings
    result: list[str] = []

    # Use cached SentenceTransformer
    model = None
    try:
        model = _get_embedder()
    except Exception as e:
        logger.warning(f"Could not load SentenceTransformer for deduplication: {e}. Using difflib only.")

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue

        is_heading = (
            stripped.startswith("#")
            or stripped.startswith("---")
            or stripped.startswith("* * *")
            or stripped.startswith("===")
        )

        if is_heading:
            result.append(stripped)
            continue

        # Fast exact-fingerprint check
        fp = _fingerprint(stripped)
        if fp in seen_fingerprints:
            continue

        # Conceptual check: Use vector similarity or fallback to edit distance
        is_duplicate = False
        if len(stripped) > 20:
            if model is not None:
                try:
                    current_emb = model.encode([stripped], convert_to_numpy=True, normalize_embeddings=True)[0]
                    if recent_embeddings:
                        similarities = np.dot(recent_embeddings, current_emb)
                        if np.max(similarities) >= 0.88:
                            is_duplicate = True
                except Exception as e:
                    logger.warning(f"Vector similarity deduplication failed: {e}")
                    model = None

            if model is None:
                for recent in prose_paragraphs:
                    ratio = difflib.SequenceMatcher(
                        None, stripped.lower(), recent.lower()
                    ).ratio()
                    if ratio >= similarity_threshold:
                        is_duplicate = True
                        break

        if not is_duplicate:
            result.append(stripped)
            seen_fingerprints.add(fp)
            prose_paragraphs.append(stripped)
            if model is not None and len(stripped) > 20:
                try:
                    emb = model.encode([stripped], convert_to_numpy=True, normalize_embeddings=True)[0]
                    recent_embeddings.append(emb)
                    if len(recent_embeddings) > 20:
                        recent_embeddings.pop(0)
                except Exception:
                    pass

    # ── Pass 2: scene-level dedup ────────────────────────────────────
    # Join paragraphs back, split on scene breaks, remove duplicate blocks.
    joined = "\n\n".join(result)
    scene_blocks = re.split(r"\n{2,}(?:\*\s*\*\s*\*|---+|===+)\n{2,}", joined)

    if len(scene_blocks) > 1:
        seen_scene_fps: set[str] = set()
        deduped_scenes: list[str] = []
        for block in scene_blocks:
            sfp = _fingerprint(block)
            if sfp in seen_scene_fps:
                continue
            # Also check fuzzy against all kept scenes
            is_scene_dup = False
            for kept in deduped_scenes:
                ratio = difflib.SequenceMatcher(
                    None, block.lower()[:600], kept.lower()[:600]
                ).ratio()
                if ratio >= 0.70:
                    is_scene_dup = True
                    break
            if not is_scene_dup:
                deduped_scenes.append(block)
                seen_scene_fps.add(sfp)
        joined = "\n\n* * *\n\n".join(deduped_scenes)

    return joined


def _resequence_chapter_headers(text: str) -> str:
    """Rewrite chapter headers so they are numbered strictly 1, 2, 3, ..."""
    chapter_counter = [0]

    def _replace_header(match: re.Match) -> str:
        chapter_counter[0] += 1
        hashes = match.group(1)
        suffix = match.group(3)
        return f"{hashes}Chapter {chapter_counter[0]}{suffix}"

    return _CHAPTER_HEADER_RE.sub(_replace_header, text)


def _extract_premise_steps(premise: str) -> list[str]:
    """Split premise into individual steps."""
    steps = []
    for line in premise.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Act ") or line.startswith("Chapter "):
            continue
        # Remove leading bullets or numbers
        clean = re.sub(r"^(\d+\.|\*|-)\s+", "", line)
        if clean:
            steps.append(clean)
    return steps


def _compute_semantic_coverage(steps: list[str], draft_text: str) -> tuple[list[str], list[str]]:
    """Use local sentence-transformers to find COVERED and MISSING steps."""
    if not steps or not draft_text:
        return [], steps

    logger.info("Computing semantic premise coverage...")
    model = _get_embedder()
    
    # Chunk the draft text into ~sentence/paragraph sizes for granular matching
    draft_chunks = [c.strip() for c in re.split(r"\n\n|\.\s+", draft_text) if len(c.strip()) > 30]
    if not draft_chunks:
        return [], steps
        
    step_embeddings = model.encode(steps, convert_to_numpy=True, normalize_embeddings=True)
    chunk_embeddings = model.encode(draft_chunks, convert_to_numpy=True, normalize_embeddings=True)
    
    # Compute cosine similarity matrix (steps x chunks)
    similarity = np.dot(step_embeddings, chunk_embeddings.T)
    max_sim_per_step = np.max(similarity, axis=1)
    
    covered = []
    missing = []
    
    for i, step in enumerate(steps):
        if max_sim_per_step[i] >= 0.72:
            covered.append(step)
        else:
            missing.append(step)
            
    return covered, missing


def _evaluate_coverage_with_gemini(
    client: genai.Client,
    model_name: str,
    draft_text: str,
    steps: list[str],
    is_cancelled: Optional[Callable[[], bool]] = None
) -> tuple[list[str], list[str]]:
    """Use Gemini to accurately determine which premise steps are covered in the draft."""
    logger.info("Computing semantic premise coverage with Gemini...")
    if not steps or not draft_text:
        return [], steps

    prompt = f"""You are a story analyst. Review the DRAFT STORY against the list of sequential steps from the MASTER PREMISE.
    
DRAFT STORY:
{draft_text}

MASTER PREMISE STEPS:
{json.dumps(steps, indent=2)}

For each step in the MASTER PREMISE STEPS, determine if that step is already written/depicted as a detailed scene or event in the DRAFT STORY.
Do NOT count a step as covered if it is only mentioned in passing or if the draft story hasn't reached that part of the timeline yet.

Return a JSON object with two keys:
1. "covered_steps": A list of steps (strings, exactly as they appear in the input list) that are already written in the draft.
2. "missing_steps": A list of steps (strings, exactly as they appear in the input list) that are NOT yet written in the draft.

Return ONLY the raw JSON object. No markdown formatting, no comments.
"""
    try:
        response = _generate_content_with_retry(
            client=client,
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            ),
            is_cancelled=is_cancelled
        )
        data = json.loads(response.text.strip())
        covered = data.get("covered_steps", [])
        missing = data.get("missing_steps", [])
        
        # Ensure all steps are accounted for and match the input strings
        steps_set = set(steps)
        covered = [s for s in covered if s in steps_set]
        missing = [s for s in missing if s in steps_set]
        
        # Any step not classified is missing by default
        classified = set(covered) | set(missing)
        for s in steps:
            if s not in classified:
                missing.append(s)
                
        return covered, missing
    except Exception as e:
        logger.warning(f"Gemini coverage evaluation failed: {e}. Falling back to local SentenceTransformer.")
        return _compute_semantic_coverage(steps, draft_text)


def _evaluate_conflicts_and_extras(client: genai.Client, model_name: str, draft_text: str, premise: str, is_cancelled: Optional[Callable[[], bool]] = None) -> dict:
    """Use Gemini to find conflicting events, evaluate unmapped scenes, and perform character state audits."""
    logger.info("Running Pre-Evaluation pass for conflicts and extra scenes...")
    
    prompt = f"""You are a story analyst. Review the DRAFT STORY against the MASTER PREMISE.
    
MASTER PREMISE:
{premise}

DRAFT STORY:
{draft_text}

Identify:
1. "conflicting": Any events in the DRAFT STORY that DIRECTLY CONTRADICT the MASTER PREMISE (e.g. a character dies prematurely, character backgrounds/roles are wrong, or a character behaves in a way that breaks a future premise step).
CRITICAL NOTE: The DRAFT STORY is a work-in-progress and may only cover the beginning or parts of the premise. A scene or step from the premise is NOT conflicting just because it hasn't happened yet in the draft.
Do NOT flag temporary transitions, character dialogue, plans, or detailed scene setups as conflicts just because they are not explicitly mentioned in the master premise steps or because they differ slightly in style or execution. For example, if characters split up temporarily but plan to reunite, this does NOT contradict them escaping together or reuniting. Only flag a conflict if there is a flat contradiction of fact (e.g. a character dies in the draft but is alive later in the premise, or characters have different names/relations, or a scene physically prevents a later premise step from occurring). If the draft matches the premise timeline but adds detail, it is NOT conflicting.
Return as a list of strings.
2. "keep_scenes": Extra scenes in the DRAFT STORY not in the premise that do NOT conflict and should be kept.
3. "remove_scenes": Extra scenes in the DRAFT STORY that contradict the premise or break world rules and must be removed.
4. "character_audit": An audit of the characters' current states at the end of the DRAFT STORY to ensure subplots and physical states remain consistent. For each character, list their physical status, emotional state, active goals, and any notable injuries or changes.

Return ONLY a raw JSON object with keys: "conflicting", "keep_scenes", "remove_scenes", "character_audit" (where character_audit is a list of objects containing keys: "name", "physical_status", "emotional_state", "active_goals"). No markdown formatting.
"""
    try:
        response = _generate_content_with_retry(
            client=client,
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            ),
            is_cancelled=is_cancelled
        )
        return json.loads(response.text.strip())
    except Exception as e:
        logger.warning(f"Conflict evaluation failed: {e}")
        return {"conflicting": [], "keep_scenes": [], "remove_scenes": [], "character_audit": []}


def _format_step_list(title: str, steps: list[str]) -> str:
    return f"{title} ({len(steps)}):\n" + ("\n".join(f"- {s}" for s in steps) if steps else "")


def _build_analysis_report(
    *,
    initial_covered: list[str],
    initial_missing: list[str],
    final_covered: list[str],
    final_missing: list[str],
    conflicting: list[str],
    remove_list: list[str],
    inserted_sections: list[str],
    character_audit: list[dict],
) -> str:
    """Build the saved continuity report from both pre- and post-completion checks."""
    parts = [
        "INITIAL COVERAGE ANALYSIS",
        _format_step_list("COVERED STEPS", initial_covered),
        _format_step_list("MISSING STEPS SENT TO COMPLETION PASS", initial_missing),
        _format_step_list("CONFLICTS", conflicting),
        _format_step_list("SCENES TO REMOVE", remove_list),
        "",
        "COMPLETION RESULTS",
        _format_step_list("INSERTED SCENE HEADINGS", inserted_sections),
        _format_step_list("FINAL COVERED STEPS", final_covered),
        _format_step_list("FINAL MISSING STEPS", final_missing),
    ]
    if final_missing:
        parts.append(
            "WARNING: Gemini still reported the above premise steps as missing after "
            "completion and final verification. Re-run Combine & Polish or generate "
            "more chapters before combining."
        )
    else:
        parts.append("FINAL STATUS: All premise steps were verified as covered in the final story.")

    if character_audit:
        char_lines = []
        for c in character_audit:
            char_lines.append(
                f"- {c.get('name', 'Unknown')}: Physical: {c.get('physical_status', 'N/A')} | "
                f"Emotional: {c.get('emotional_state', 'N/A')} | Goals: {c.get('active_goals', 'N/A')}"
            )
        parts.extend(["", "CHARACTER & SUBPLOT STATUS AUDIT", "\n".join(char_lines)])

    return "\n\n".join(parts).strip()


def _repair_final_coverage_gaps(
    client: genai.Client,
    model_name: str,
    final_text: str,
    missing_steps: list[str],
    character_audit: list[dict],
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> tuple[str, list[str]]:
    """Make one targeted final attempt to insert premise steps that verification still missed."""
    if not missing_steps:
        return final_text, []

    prompt = f"""You are a master continuity editor and novelist.

The final story below was verified and still missed specific MASTER PREMISE STEPS.
Repair the story by writing those missing steps as full prose scenes and placing them in the correct chronological positions.

CRITICAL RULES:
- Return the ENTIRE repaired story, not only the new scenes.
- Preserve all existing chapters and scenes unless a sentence-level continuity change is required.
- Insert every missing premise step below as an actual depicted scene or event, not as a summary or passing reference.
- Keep chapters in chronological story order. If new chapter headings are needed, add them at the correct position.
- Prefix every newly inserted scene heading with exactly "INSERTED: " so it can be tracked before final cleanup.
- Maintain character status, injuries, relationships, and active goals from the audit.
- Do not duplicate scenes that already exist.

CHARACTER STATUS TRACKERS:
{json.dumps(character_audit, indent=2)}

MISSING PREMISE STEPS:
{json.dumps(missing_steps, indent=2)}

FINAL STORY TO REPAIR:
{final_text}
"""
    try:
        response = _generate_content_with_retry(
            client=client,
            model=model_name,
            contents=prompt,
            is_cancelled=is_cancelled,
        )
        repaired_text = response.text.strip()
        if len(repaired_text) < 0.98 * len(final_text):
            logger.warning(
                "Final coverage repair output was too short (%s vs %s chars). Keeping previous final draft.",
                len(repaired_text),
                len(final_text),
            )
            return final_text, []
        inserted = [line for line in repaired_text.split("\n") if line.startswith("INSERTED:")]
        return repaired_text, inserted
    except Exception as e:
        logger.error("Final coverage repair failed: %s", e)
        return final_text, []


def combine_chapters(chapters_dir: str, metadata: dict) -> str:
    """Concatenate all chapter files into one Markdown document."""
    chapters = _read_chapters(chapters_dir)
    if not chapters:
        raise RuntimeError("No chapter files found to combine.")
    
    parts = []
    for _num, text in chapters:
        parts.append(text.strip())
        parts.append("\n\n---\n")
    return "\n".join(parts)


def analyze_and_polish(
    story_text: str,
    progress_cb: Optional[Callable] = None,
    premise: str = "",
    model_name: Optional[str] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
    state: Optional[dict] = None,
) -> dict:
    """
    Execute the 3-Pass Gemini reconciliation and completion engine.

    Args:
        story_text: Combined chapter text.
        progress_cb: SSE progress callback.
        premise: Master premise text.
        model_name: Gemini model to use.
        is_cancelled: Cancellation check callable.
        state: Full state.json dict for canonical facts injection.
    """
    api_key = config.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    model_name = model_name or config.GEMINI_MODEL or "gemini-2.5-flash-lite"
    client = genai.Client(api_key=api_key)

    # Build canonical story digest for polish prompt injection
    story_digest = _build_story_digest(state or {})

    # Edit provenance log — tracks what each pass changed
    provenance: list[dict] = []
    confidence_scores: dict[str, float] = {}

    def _emit(msg: str):
        if progress_cb:
            progress_cb("combine_status", {"step": msg})

    def _check_cancelled():
        if is_cancelled and is_cancelled():
            raise RuntimeError("Cancellation requested by user")

    _emit("Analyzing draft coverage against premise...")
    
    # 1. Premise Coverage Analysis
    premise_steps = _extract_premise_steps(premise)
    _check_cancelled()
    covered_steps, missing_steps = _evaluate_coverage_with_gemini(client, model_name, story_text, premise_steps, is_cancelled)
    
    # 2. Extra Scene Evaluation & Conflict Detection
    _check_cancelled()
    eval_data = _evaluate_conflicts_and_extras(client, model_name, story_text, premise, is_cancelled)
    conflicting = eval_data.get("conflicting", [])
    keep_list = eval_data.get("keep_scenes", [])
    remove_list = eval_data.get("remove_scenes", [])
    character_audit = eval_data.get("character_audit", [])
    
    # 3. Gemini Three-Pass Pipeline
    
    # --- Pass 1: Reconciliation ---
    if not conflicting and not remove_list:
        _emit("No conflicts or scenes to remove detected. Skipping Pass 1 (Reconciliation).")
        reconciled_draft = story_text
    else:
        _emit("Pass 1/3: Reconciling conflicts and removing extra scenes...")
        p1_prompt = f"""You are a master developmental editor.
        
Here is a story draft. You must ONLY do the following:
1. Remove or rewrite the specific events listed in CONFLICTS.
2. Remove the scenes listed in REMOVE LIST.

CRITICAL RULES:
- Do NOT delete entire chapters, scenes, or introductions. Keep all written text intact, editing only the specific sentences or details that cause the logical conflict.
- You must preserve the beginning/introductory scenes of the story. Do NOT delete or truncate the classroom outbreak, escape, or initial setup.
- Preserve the full length and scene-by-scene shape of the draft. This is NOT a summary task.
- Do NOT add any new story beats. Do NOT alter anything related to KEEP LIST or existing good content.
Just return the cleanly reconciled draft.

CONFLICTS: {conflicting}
REMOVE LIST: {remove_list}
KEEP LIST (Do not touch): {keep_list}

DRAFT:
{story_text}
"""
        try:
            p1_resp = _generate_content_with_retry(client=client, model=model_name, contents=p1_prompt, is_cancelled=is_cancelled)
            reconciled_draft = _preserve_length_or_fallback(
                p1_resp.text.strip(),
                story_text,
                "Pass 1 reconciliation",
                min_ratio=0.90,
            )
        except Exception as e:
            logger.error(f"Pass 1 failed: {e}")
            reconciled_draft = story_text

    # Provenance: Pass 1
    p1_ratio = len(reconciled_draft) / max(1, len(story_text))
    provenance.append({
        "pass": "reconciliation",
        "input_chars": len(story_text),
        "output_chars": len(reconciled_draft),
        "changes_summary": f"Conflicts: {len(conflicting)}, Removed: {len(remove_list)}",
    })
    confidence_scores["reconciliation"] = round(min(1.0, p1_ratio), 3)
    # --- Pass 2: Completion ---
    if not missing_steps:
        _emit("No missing premise steps detected. Skipping Pass 2 (Completion).")
        completed_draft = reconciled_draft
        inserted_sections = []
    else:
        _emit("Pass 2/3: Writing missing premise scenes...")
        p2_prompt = f"""You are a master author.
        
Here is a reconciled story draft, character status trackers, and a list of MISSING PREMISE STEPS that were skipped.
You must WRITE fully-realized, highly detailed prose chapters/scenes for EVERY missing step.
Insert the new scenes at their correct narrative chronological position within the draft (do not just append to the end).

CHARACTER STATUS TRACKERS:
{json.dumps(character_audit, indent=2)}

CRITICAL RULES:
- You MUST return the entire story text including ALL the existing chapters from the RECONCILED DRAFT.
- Do NOT omit, truncate, summarize, or delete any of the existing chapters (such as the introductory classroom scene/escape in Chapter 1). You must start your output with the exact text of Chapter 1 from the RECONCILED DRAFT, and then continue outputting all subsequent chapters, inserting the new scenes at their correct chronological places.
- Maintain continuity with the CHARACTER STATUS TRACKERS (e.g. physical status, emotional state, active goals). Do not introduce continuity breaks.
- Write with GROUNDED REALISM. Avoid abstract, floral, or poetic metaphors.
- STRICT PROSE BAN LIST: Do NOT use the following words unless they are used in a strictly literal sense: shiver, testament, dance, intertwine, symphony, beacon, crescendo, tapestry, gaze, whisper, silhouette, labyrinthine, sanctuary. Focus on tactile weight, friction, physical movements, and precise action verbs.
- NEVER duplicate or repeat any scene that already exists in the draft.
- Respect character timelines: if a character is born in Act 4, they CANNOT appear as a speaking/acting character in Acts 1-3.
- Characters who are babies/toddlers must be written age-appropriately — they cannot have adult conversations.
- Each inserted scene must be UNIQUE. Do not repeat the same scene text or variations of it.
- Write explicit/intimate scenes in full detail when the premise demands them. Do NOT fade to black, summarize, or gloss over the sexual encounters. Write the erotic interactions with high sensory detail, physical realism, and raw, explicit description as expected in the erotica genre.
- Prefix every new scene heading with exactly "INSERTED: " so they can be tracked.

MISSING PREMISE STEPS TO WRITE:
{missing_steps}

RECONCILED DRAFT:
{reconciled_draft}
"""
        try:
            p2_resp = _generate_content_with_retry(client=client, model=model_name, contents=p2_prompt, is_cancelled=is_cancelled)
            completed_draft = p2_resp.text.strip()
            
            # Pass 2 Length Guardrail
            input_len = len(reconciled_draft)
            output_len = len(completed_draft)
            if output_len < 0.95 * input_len:
                logger.warning(f"Pass 2 output was too short ({output_len} vs {input_len} chars). Retrying once with strict guardrails...")
                retry_prompt = p2_prompt + "\n\nCRITICAL: Your previous response was truncated or summarized. You MUST re-generate the complete full-length text of all existing chapters from the RECONCILED DRAFT, with the new missing scenes inserted. Do not condense paragraphs."
                p2_resp = _generate_content_with_retry(client=client, model=model_name, contents=retry_prompt, is_cancelled=is_cancelled)
                retry_draft = p2_resp.text.strip()
                if len(retry_draft) >= 0.95 * input_len:
                    completed_draft = retry_draft
                else:
                    logger.warning("Length check failed again on retry. Falling back to reconciled draft.")
                    completed_draft = reconciled_draft
        except Exception as e:
            logger.error(f"Pass 2 failed: {e}")
            completed_draft = reconciled_draft

        # Extract inserted sections for reporting
        inserted_sections = [line for line in completed_draft.split('\n') if line.startswith("INSERTED:")]

    # Provenance: Pass 2
    p2_ratio = len(completed_draft) / max(1, len(reconciled_draft))
    provenance.append({
        "pass": "completion",
        "input_chars": len(reconciled_draft),
        "output_chars": len(completed_draft),
        "changes_summary": f"Missing steps: {len(missing_steps)}, Inserted: {len(inserted_sections) if 'inserted_sections' in dir() else 0}",
    })
    confidence_scores["completion"] = round(min(1.0, p2_ratio), 3)

    # --- Pass 3: Polish (Sliding Window) ---
    _emit("Pass 3/3: Polishing continuity and prose...")
    chapters = [p.strip() for p in re.split(r'(?=(?:^|\n)(?:#+|##+)\s+Chapter\s+\d+)', completed_draft) if p.strip()]
    
    if len(chapters) <= 2:
        # Single Pass Polish for very small texts
        digest_block = ""
        if story_digest:
            digest_block = f"""\n\nCANONICAL STORY FACTS (Authoritative source of truth — override any contradictions in the draft):
{story_digest}

ENFORCEMENT RULES:
- NEVER rename characters. Use the exact canonical spellings above.
- NEVER change established relationships.
- NEVER contradict world rules.
- Preserve recurring motifs listed above.
"""
        p3_prompt = f"""You are a master copyeditor specializing in mature romance and erotica.
        
Here is a completed novel draft.
Fix tense drift, character name swaps, abrupt transitions, pacing inconsistencies, and chapter boundary continuity ONLY.
Do NOT add, remove, or restructure any events or scenes.
Preserve the full scene-by-scene story and chapter length. This is a line edit, not a rewrite or summary.
Do NOT duplicate any scenes or paragraphs. If you see repeated/duplicate content, remove the extras.
Ensure character ages are consistent with the timeline.
Do NOT censor, clean up, or tone down the explicit sexual scenes. Keep all intimate details fully detailed, explicit, and intact.
Write with GROUNDED REALISM. Avoid abstract, floral AI-isms. Banned words: shiver, testament, dance, intertwine, symphony, beacon, crescendo, tapestry, gaze, whisper, silhouette, labyrinthine, sanctuary.
Return the final polished text.
{digest_block}
COMPLETED DRAFT:
{completed_draft}
"""
        try:
            p3_resp = _generate_content_with_retry(client=client, model=model_name, contents=p3_prompt, is_cancelled=is_cancelled)
            polished_draft = _preserve_length_or_fallback(
                p3_resp.text.strip(),
                completed_draft,
                "Pass 3 polish",
                min_ratio=0.92,
            )
        except Exception as e:
            logger.error(f"Pass 3 failed: {e}")
            polished_draft = completed_draft
    else:
        # Sliding Window Polish
        polished_chapters = []
        i = 0
        window_size = 3
        while i < len(chapters):
            _check_cancelled()
            chunk = chapters[i:i+window_size]
            prev_context = chapters[i-1] if i > 0 else None
            next_context = chapters[i+window_size] if (i+window_size) < len(chapters) else None
            
            # Build digest block for this window
            digest_block = ""
            if story_digest:
                digest_block = f"""\nCANONICAL STORY FACTS (Source of truth — override contradictions):
{story_digest}

ENFORCEMENT RULES:
- NEVER rename characters. Use the exact canonical spellings above.
- NEVER change established relationships.
- NEVER contradict world rules.
- Preserve recurring motifs.
"""
            p3_prompt = f"""You are a master copyeditor specializing in mature romance and erotica.
You must polish the target chapters to resolve tense drift, name swaps, and transitions.
{digest_block}
CONTEXT FROM NEIGHBORING CHAPTERS:
"""
            if prev_context:
                p3_prompt += f"\n[PREVIOUS CHAPTER CONTEXT - DO NOT EDIT OR RETURN]:\n{prev_context}\n"
            if next_context:
                p3_prompt += f"\n[FOLLOWING CHAPTER CONTEXT - DO NOT EDIT OR RETURN]:\n{next_context}\n"
                
            p3_prompt += f"""
TARGET CHAPTERS TO POLISH (Return ONLY these polished target chapters):
{"\n\n---\n\n".join(chunk)}

CRITICAL RULES:
- Fix tense drift, character name swaps, abrupt transitions, and chapter boundary continuity.
- Do NOT add, remove, or restructure any events or scenes.
- Preserve chapter length and paragraph-level detail. This is a line edit, not a summary.
- Do NOT duplicate any scenes.
- Do NOT censor, clean up, or tone down the explicit sexual scenes. Keep all intimate details fully detailed, explicit, and intact.
- Write with GROUNDED REALISM. Avoid abstract, floral AI-isms. Banned words: shiver, testament, dance, intertwine, symphony, beacon, crescendo, tapestry, gaze, whisper, silhouette, labyrinthine, sanctuary.
- Return ONLY the polished version of the TARGET CHAPTERS. Do not output any introduction, notes, markdown block wrappers, or neighboring chapter context.
"""
            try:
                _emit(f"Polishing chapters {i+1} to {min(i+window_size, len(chapters))}...")
                p3_resp = _generate_content_with_retry(client=client, model=model_name, contents=p3_prompt, is_cancelled=is_cancelled)
                polished_chunk = p3_resp.text.strip()
                
                # Split polished chunk back into individual chapters
                polished_parts = [p.strip() for p in re.split(r'(?=(?:^|\n)(?:#+|##+)\s+Chapter\s+\d+)', polished_chunk) if p.strip()]
                original_chunk_text = "\n\n".join(chunk)
                if len(polished_parts) != len(chunk):
                    logger.warning(
                        "Polished chunk rejected (chapters %s vs %s). Using original chunk.",
                        len(polished_parts), len(chunk),
                    )
                    polished_chapters.extend(chunk)
                else:
                    # Per-window semantic preservation check
                    sem_ok, sem_score = _semantic_preservation_check(
                        original_chunk_text, polished_chunk, threshold=0.94,
                    )
                    if sem_ok:
                        polished_chapters.extend(polished_parts)
                    else:
                        logger.warning(
                            "Polished window %s-%s failed semantic check (%.3f). Using original.",
                            i + 1, min(i + window_size, len(chapters)), sem_score,
                        )
                        polished_chapters.extend(chunk)
            except Exception as e:
                logger.error(f"Failed to polish window starting at chapter {i+1}: {e}")
                polished_chapters.extend(chunk)
            
            i += window_size
            
        polished_draft = "\n\n".join(polished_chapters)

    # Pass 3 Global Semantic Preservation Guardrail
    p3_sem_ok, p3_sem_score = _semantic_preservation_check(
        completed_draft, polished_draft, threshold=0.93,
    )
    if not p3_sem_ok:
        logger.warning(
            "Polished draft failed global semantic check (%.3f). Falling back to completed draft.",
            p3_sem_score,
        )
        polished_draft = completed_draft
        p3_sem_score = 1.0  # Fallback = identical

    # Provenance: Pass 3
    provenance.append({
        "pass": "polish",
        "input_chars": len(completed_draft),
        "output_chars": len(polished_draft),
        "semantic_coverage": p3_sem_score,
        "changes_summary": "Line edit: tense, spelling, transitions",
    })
    confidence_scores["polish"] = round(p3_sem_score, 3)

    # 4. Post-Processing
    _emit("Applying post-processing (deduplication & resequencing)...")
    deduped_text = _dedup_paragraphs(polished_draft, similarity_threshold=0.90)
    final_text = _preserve_length_or_fallback(
        deduped_text,
        polished_draft,
        "Post-processing deduplication",
        min_ratio=0.95,
    )
    final_text = _resequence_chapter_headers(final_text)
    final_text = final_text.replace("INSERTED: ", "")

    # 5. Final Coverage Verification and Targeted Repair
    _emit("Verifying final premise coverage...")
    _check_cancelled()
    final_covered, final_missing = _evaluate_coverage_with_gemini(client, model_name, final_text, premise_steps, is_cancelled)

    repair_inserted_sections = []
    if final_missing:
        logger.warning(f"Final draft is still missing {len(final_missing)} premise steps.")
        _emit("Final verification found missing premise steps. Repairing coverage gaps...")
        repaired_text, repair_inserted_sections = _repair_final_coverage_gaps(
            client,
            model_name,
            final_text,
            final_missing,
            character_audit,
            is_cancelled,
        )
        if repaired_text != final_text:
            deduped_repaired_text = _dedup_paragraphs(repaired_text, similarity_threshold=0.90)
            final_text = _preserve_length_or_fallback(
                deduped_repaired_text,
                repaired_text,
                "Final repair deduplication",
                min_ratio=0.95,
            )
            final_text = _resequence_chapter_headers(final_text)
            final_text = final_text.replace("INSERTED: ", "")
            _emit("Re-checking premise coverage after repair...")
            final_covered, final_missing = _evaluate_coverage_with_gemini(
                client, model_name, final_text, premise_steps, is_cancelled
            )

    guarded_final_text = _preserve_length_or_fallback(
        final_text,
        story_text,
        "Final output",
        min_ratio=0.95,
    )
    if guarded_final_text != final_text:
        final_text = guarded_final_text
        _emit("Final output was too short. Restored full-length draft and re-checking coverage...")
        final_covered, final_missing = _evaluate_coverage_with_gemini(
            client, model_name, final_text, premise_steps, is_cancelled
        )

    inserted_sections = inserted_sections + repair_inserted_sections
    coverage_gaps = final_missing

    analysis_report = _build_analysis_report(
        initial_covered=covered_steps,
        initial_missing=missing_steps,
        final_covered=final_covered,
        final_missing=final_missing,
        conflicting=conflicting,
        remove_list=remove_list,
        inserted_sections=inserted_sections,
        character_audit=character_audit,
    )

    _emit("Story combination and completion finished successfully.")

    # 6. Output Structure
    return {
        "final_story": final_text,
        "analysis": analysis_report,
        "inserted_sections": inserted_sections,
        "kept_extra_scenes": keep_list,
        "removed_scenes": remove_list,
        "coverage_gaps": coverage_gaps,
        "model": model_name,
        "provenance": provenance,
        "confidence_scores": confidence_scores,
    }
