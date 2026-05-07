"""
Gemini-powered story combiner and polisher.

Concatenates all chapter files into a single document, then sends it to
Google Gemini for continuity analysis and revision.  The large context
window makes it possible to process the entire story in one call.
"""
import os
import re
import logging
from typing import Callable, Optional

from google import genai

import config

logger = logging.getLogger(__name__)

_CHAPTER_FILE_RE = re.compile(r"^chapter_(\d{3,})\.md$")


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


def combine_chapters(chapters_dir: str, metadata: dict) -> str:
    """Concatenate all chapter files into one Markdown document."""
    chapters = _read_chapters(chapters_dir)
    if not chapters:
        raise RuntimeError("No chapter files found to combine.")

    parts = []
    title = metadata.get("title", "Untitled")
    genre = metadata.get("genre", "")
    premise = metadata.get("premise", "")

    parts.append(f"# {title}\n")
    if genre:
        parts.append(f"*{genre}*\n")
    if premise:
        parts.append(f"> {premise}\n")
    parts.append("---\n")

    for _num, text in chapters:
        parts.append(text.strip())
        parts.append("\n\n---\n")

    return "\n".join(parts)


def analyze_and_polish(
    story_text: str,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Send the combined story to Gemini for continuity analysis and revision.

    Returns a dict with keys:
        - analysis: str   — the continuity/consistency analysis
        - revised:  str   — the polished, revised story text
        - model:    str   — which Gemini model was used
    """
    api_key = config.GEMINI_API_KEY
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set.  Add it to your .env file."
        )

    model_name = config.GEMINI_MODEL or "gemini-2.5-flash-lite"

    def _emit(msg: str):
        if progress_cb:
            progress_cb("combine_status", {"step": msg})

    _emit(f"Initializing Gemini client ({model_name})...")
    client = genai.Client(api_key=api_key)

    prompt = f"""You are a developmental editor specializing in multi-chapter narratives.

Analyze this story for:

1. **Character Consistency Issues**
   - Does each character maintain consistent motivations and personality?
   - Are there unexplained behavior shifts?
   - Do physical descriptions remain consistent?

2. **Continuity Errors**
   - Timeline inconsistencies
   - Events that contradict earlier chapters
   - Character whereabouts/relationships that don't align
   - Emotional arcs that don't flow logically

3. **Narrative Flow**
   - Are transitions smooth between chapters?
   - Do cliffhangers resolve appropriately?
   - Is pacing consistent?
   - Are there plot holes?

Then provide:
- A brief summary of issues found
- A revised version of the story with fixes applied
- Keep ALL explicit content intact
- Maintain the narrative voice and tense
- Keep the same chapter structure
- Do NOT summarize or condense — output the FULL revised story

Format your response as:
---ANALYSIS---
[Your analysis here]

---REVISED_STORY---
[Complete revised story here]

STORY TO ANALYZE:
{story_text}
"""

    _emit("Sending story to Gemini for analysis (this may take a few minutes)...")
    logger.info(
        "Gemini combine: sending %d chars to %s", len(story_text), model_name
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        output = response.text.strip()
    except Exception as e:
        logger.error("Gemini API call failed: %s", e)
        raise RuntimeError(f"Gemini API call failed: {e}") from e

    _emit("Parsing Gemini response...")

    # Parse structured response
    if "---ANALYSIS---" in output and "---REVISED_STORY---" in output:
        analysis_part, rest = output.split("---REVISED_STORY---", 1)
        analysis = analysis_part.replace("---ANALYSIS---", "").strip()
        revised = rest.strip()
    else:
        # Fallback: treat entire output as revised story
        logger.warning("Gemini response did not contain expected markers.")
        analysis = (
            "Gemini did not return a structured analysis.  "
            "The full response has been saved as the revised story."
        )
        revised = output

    logger.info(
        "Gemini combine complete: analysis=%d chars, revised=%d chars",
        len(analysis),
        len(revised),
    )

    return {
        "analysis": analysis,
        "revised": revised,
        "model": model_name,
    }
