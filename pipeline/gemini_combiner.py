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
from typing import Callable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

import config

logger = logging.getLogger(__name__)

_CHAPTER_FILE_RE = re.compile(r"^chapter_(\d{3,})\.md$")
_CHAPTER_HEADER_RE = re.compile(r"^(#+\s*)(Chapter\s+\d+)(.*?)$", re.IGNORECASE | re.MULTILINE)
_SCENE_BREAK_RE = re.compile(r"\n{2,}(?:\*\s*\*\s*\*|--+|===+)\n{2,}")


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
      Pass 1 — paragraph-level: uses a fingerprint set for exact matches and
               difflib against ALL previous prose paragraphs for near-matches.
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
    result: list[str] = []

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

        # Fuzzy check against ALL previous prose paragraphs
        is_duplicate = False
        if len(stripped) > 20:
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
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
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
        if max_sim_per_step[i] >= 0.60:
            covered.append(step)
        else:
            missing.append(step)
            
    return covered, missing


def _evaluate_conflicts_and_extras(client: genai.Client, model_name: str, draft_text: str, premise: str) -> dict:
    """Use Gemini to find conflicting events and evaluate unmapped scenes."""
    logger.info("Running Pre-Evaluation pass for conflicts and extra scenes...")
    
    prompt = f"""You are a story analyst. Review the DRAFT against the MASTER PREMISE.
    
MASTER PREMISE:
{premise}

DRAFT STORY:
{draft_text[:30000]}... (truncated for analysis)

Identify:
1. "conflicting": Any events in the draft that directly contradict the premise (e.g. a character dies prematurely). Return as a list of strings.
2. "keep_scenes": Extra scenes not in the premise that do NOT conflict and should be kept.
3. "remove_scenes": Extra scenes that contradict the premise or break world rules and must be removed.

Return ONLY a raw JSON object with keys: "conflicting", "keep_scenes", "remove_scenes" (all list of strings). No markdown formatting.
"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        return json.loads(response.text.strip())
    except Exception as e:
        logger.warning(f"Conflict evaluation failed: {e}")
        return {"conflicting": [], "keep_scenes": [], "remove_scenes": []}


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
) -> dict:
    """
    Execute the 3-Pass Gemini reconciliation and completion engine.
    """
    api_key = config.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    model_name = model_name or config.GEMINI_MODEL or "gemini-2.5-flash-lite"
    client = genai.Client(api_key=api_key)

    def _emit(msg: str):
        if progress_cb:
            progress_cb("combine_status", {"step": msg})

    _emit("Analyzing draft coverage against premise...")
    
    # 1. Premise Coverage Analysis
    premise_steps = _extract_premise_steps(premise)
    covered_steps, missing_steps = _compute_semantic_coverage(premise_steps, story_text)
    
    # 2. Extra Scene Evaluation & Conflict Detection
    eval_data = _evaluate_conflicts_and_extras(client, model_name, story_text, premise)
    conflicting = eval_data.get("conflicting", [])
    keep_list = eval_data.get("keep_scenes", [])
    remove_list = eval_data.get("remove_scenes", [])
    
    analysis_report = (
        f"COVERED STEPS ({len(covered_steps)}):\n" + "\n".join(f"- {s}" for s in covered_steps) + "\n\n" +
        f"MISSING STEPS ({len(missing_steps)}):\n" + "\n".join(f"- {s}" for s in missing_steps) + "\n\n" +
        f"CONFLICTS ({len(conflicting)}):\n" + "\n".join(f"- {s}" for s in conflicting) + "\n\n" +
        f"SCENES TO REMOVE ({len(remove_list)}):\n" + "\n".join(f"- {s}" for s in remove_list)
    )
    
    # 3. Gemini Three-Pass Pipeline
    
    # --- Pass 1: Reconciliation ---
    _emit("Pass 1/3: Reconciling conflicts and removing extra scenes...")
    p1_prompt = f"""You are a master developmental editor.
    
Here is a story draft. You must ONLY do the following:
1. Remove or rewrite the specific events listed in CONFLICTS.
2. Remove the scenes listed in REMOVE LIST.
Do NOT add any new story beats. Do NOT alter anything related to KEEP LIST or existing good content.
Just return the cleanly reconciled draft.

CONFLICTS: {conflicting}
REMOVE LIST: {remove_list}
KEEP LIST (Do not touch): {keep_list}

DRAFT:
{story_text}
"""
    try:
        p1_resp = client.models.generate_content(model=model_name, contents=p1_prompt)
        reconciled_draft = p1_resp.text.strip()
    except Exception as e:
        logger.error(f"Pass 1 failed: {e}")
        reconciled_draft = story_text

    # --- Pass 2: Completion ---
    _emit("Pass 2/3: Writing missing premise scenes...")
    p2_prompt = f"""You are a master author.
    
Here is a reconciled story draft and a list of MISSING PREMISE STEPS that were skipped.
You must WRITE fully-realized, highly detailed prose chapters/scenes for EVERY missing step.
Insert the new scenes at their correct narrative chronological position within the draft (do not just append to the end).
Do NOT alter the existing text, just inject the new scenes.
Prefix every new scene heading with exactly "INSERTED: " so they can be tracked.

CRITICAL RULES:
- NEVER duplicate or repeat any scene that already exists in the draft.
- Respect character timelines: if a character is born in Act 4, they CANNOT appear as a speaking/acting character in Acts 1-3.
- Characters who are babies/toddlers must be written age-appropriately — they cannot have adult conversations.
- Each inserted scene must be UNIQUE. Do not repeat the same scene text or variations of it.
- Write explicit/intimate scenes in full detail when the premise demands them. Do not fade to black or skip the content.

MISSING PREMISE STEPS TO WRITE:
{missing_steps}

RECONCILED DRAFT:
{reconciled_draft}
"""
    try:
        p2_resp = client.models.generate_content(model=model_name, contents=p2_prompt)
        completed_draft = p2_resp.text.strip()
    except Exception as e:
        logger.error(f"Pass 2 failed: {e}")
        completed_draft = reconciled_draft

    # Extract inserted sections for reporting
    inserted_sections = [line for line in completed_draft.split('\n') if line.startswith("INSERTED:")]

    # --- Pass 3: Polish ---
    _emit("Pass 3/3: Polishing continuity and prose...")
    p3_prompt = f"""You are a master copyeditor.
    
Here is a completed novel draft.
Fix tense drift, character name swaps, abrupt transitions, pacing inconsistencies, and chapter boundary continuity ONLY.
Do NOT add, remove, or restructure any events or scenes.
Do NOT duplicate any scenes or paragraphs. If you see repeated/duplicate content, remove the extras.
Ensure character ages are consistent with the timeline (e.g. babies cannot speak or act like adults).
Return the final polished text.

COMPLETED DRAFT:
{completed_draft}
"""
    try:
        p3_resp = client.models.generate_content(model=model_name, contents=p3_prompt)
        polished_draft = p3_resp.text.strip()
    except Exception as e:
        logger.error(f"Pass 3 failed: {e}")
        polished_draft = completed_draft

    # 4. Post-Processing
    _emit("Applying post-processing (deduplication & resequencing)...")
    final_text = _dedup_paragraphs(polished_draft, similarity_threshold=0.85)
    final_text = _resequence_chapter_headers(final_text)
    final_text = final_text.replace("INSERTED: ", "")

    # 5. Final Coverage Verification
    _emit("Verifying final premise coverage...")
    final_covered, final_missing = _compute_semantic_coverage(premise_steps, final_text)
    
    coverage_gaps = []
    if final_missing:
        logger.warning(f"Final draft is still missing {len(final_missing)} premise steps.")
        coverage_gaps = final_missing

    _emit("Story combination and completion finished successfully.")

    # 6. Output Structure
    return {
        "final_story": final_text,
        "analysis": analysis_report,
        "inserted_sections": inserted_sections,
        "kept_extra_scenes": keep_list,
        "removed_scenes": remove_list,
        "coverage_gaps": coverage_gaps,
        "model": model_name
    }



