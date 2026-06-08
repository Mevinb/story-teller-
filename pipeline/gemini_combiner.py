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
from typing import Callable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

import config

logger = logging.getLogger(__name__)


_API_REQUEST_LOG = []
_TPM_LIMIT = 200000  # Estimate limit for gemini-2.5-flash-lite free tier

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
    if current_tpm + estimated_tokens > 0.85 * _TPM_LIMIT:
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

    # Pre-load SentenceTransformer if possible
    model = None
    try:
        model = SentenceTransformer(config.EMBEDDING_MODEL)
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
    
    analysis_report = (
        f"COVERED STEPS ({len(covered_steps)}):\n" + "\n".join(f"- {s}" for s in covered_steps) + "\n\n" +
        f"MISSING STEPS ({len(missing_steps)}):\n" + "\n".join(f"- {s}" for s in missing_steps) + "\n\n" +
        f"CONFLICTS ({len(conflicting)}):\n" + "\n".join(f"- {s}" for s in conflicting) + "\n\n" +
        f"SCENES TO REMOVE ({len(remove_list)}):\n" + "\n".join(f"- {s}" for s in remove_list)
    )
    if character_audit:
        char_lines = []
        for c in character_audit:
            char_lines.append(f"- {c.get('name', 'Unknown')}: Physical: {c.get('physical_status', 'N/A')} | Emotional: {c.get('emotional_state', 'N/A')} | Goals: {c.get('active_goals', 'N/A')}")
        analysis_report += "\n\nCHARACTER & SUBPLOT STATUS AUDIT:\n" + "\n".join(char_lines)
    
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
            reconciled_draft = p1_resp.text.strip()
        except Exception as e:
            logger.error(f"Pass 1 failed: {e}")
            reconciled_draft = story_text

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
            if output_len < 0.90 * input_len:
                logger.warning(f"Pass 2 output was too short ({output_len} vs {input_len} chars). Retrying once with strict guardrails...")
                retry_prompt = p2_prompt + "\n\nCRITICAL: Your previous response was truncated. You MUST re-generate the text and include the complete text of all existing chapters from the RECONCILED DRAFT."
                p2_resp = _generate_content_with_retry(client=client, model=model_name, contents=retry_prompt, is_cancelled=is_cancelled)
                retry_draft = p2_resp.text.strip()
                if len(retry_draft) >= 0.90 * input_len:
                    completed_draft = retry_draft
                else:
                    logger.warning("Length check failed again on retry. Falling back to reconciled draft.")
                    completed_draft = reconciled_draft
        except Exception as e:
            logger.error(f"Pass 2 failed: {e}")
            completed_draft = reconciled_draft

        # Extract inserted sections for reporting
        inserted_sections = [line for line in completed_draft.split('\n') if line.startswith("INSERTED:")]

    # --- Pass 3: Polish (Sliding Window) ---
    _emit("Pass 3/3: Polishing continuity and prose...")
    chapters = [p.strip() for p in re.split(r'(?=(?:^|\n)(?:#+|##+)\s+Chapter\s+\d+)', completed_draft) if p.strip()]
    
    if len(chapters) <= 2:
        # Single Pass Polish for very small texts
        p3_prompt = f"""You are a master copyeditor specializing in mature romance and erotica.
        
Here is a completed novel draft.
Fix tense drift, character name swaps, abrupt transitions, pacing inconsistencies, and chapter boundary continuity ONLY.
Do NOT add, remove, or restructure any events or scenes.
Do NOT duplicate any scenes or paragraphs. If you see repeated/duplicate content, remove the extras.
Ensure character ages are consistent with the timeline.
Do NOT censor, clean up, or tone down the explicit sexual scenes. Keep all intimate details fully detailed, explicit, and intact.
Write with GROUNDED REALISM. Avoid abstract, floral AI-isms. Banned words: shiver, testament, dance, intertwine, symphony, beacon, crescendo, tapestry, gaze, whisper, silhouette, labyrinthine, sanctuary.
Return the final polished text.

COMPLETED DRAFT:
{completed_draft}
"""
        try:
            p3_resp = _generate_content_with_retry(client=client, model=model_name, contents=p3_prompt, is_cancelled=is_cancelled)
            polished_draft = p3_resp.text.strip()
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
            
            p3_prompt = f"""You are a master copyeditor specializing in mature romance and erotica.
You must polish the target chapters to resolve tense drift, name swaps, and transitions.

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
                if len(polished_parts) == len(chunk):
                    polished_chapters.extend(polished_parts)
                else:
                    logger.warning(f"Polished chapter count mismatch ({len(polished_parts)} vs {len(chunk)}). Using original chunk.")
                    polished_chapters.extend(chunk)
            except Exception as e:
                logger.error(f"Failed to polish window starting at chapter {i+1}: {e}")
                polished_chapters.extend(chunk)
            
            i += window_size
            
        polished_draft = "\n\n".join(polished_chapters)

    # Pass 3 Global Length Guardrail
    if len(polished_draft) < 0.85 * len(completed_draft):
        logger.warning(f"Polished draft is too short ({len(polished_draft)} vs {len(completed_draft)} chars). Falling back to completed draft.")
        polished_draft = completed_draft

    # 4. Post-Processing
    _emit("Applying post-processing (deduplication & resequencing)...")
    final_text = _dedup_paragraphs(polished_draft, similarity_threshold=0.85)
    final_text = _resequence_chapter_headers(final_text)
    final_text = final_text.replace("INSERTED: ", "")

    # 5. Final Coverage Verification
    _emit("Verifying final premise coverage...")
    _check_cancelled()
    final_covered, final_missing = _evaluate_coverage_with_gemini(client, model_name, final_text, premise_steps, is_cancelled)
    
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



