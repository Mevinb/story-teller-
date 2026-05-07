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

    prompt = f"""You are an elite developmental editor and narrative continuity expert specializing in multi-chapter stories.

Your objective is to take the following collection of chapters and fuse them into a seamless, high-quality story. The primary issues you must fix are abrupt scene shifts and lack of persistent character context between chapters.

Analyze and heavily revise the story with extreme focus on the following core areas:

1. **Character Context & Continuity (CRITICAL)**
   - Maintain deep context about each person across chapters. Do not let characters "reset" their emotional state, relationships, or physical descriptions between scenes.
   - Weave in relevant past context where needed to remind the reader of characters' ongoing internal struggles, motivations, and evolving dynamics.
   - Fix any unexplained behavior shifts. Characters must act consistently based on everything that happened to them previously.

2. **Scene Shifts & Transitions (CRITICAL)**
   - Eliminate abrupt jumps in time or location. 
   - When a scene shifts to another place or time, explicitly define the transition. Add necessary descriptive prose, bridging paragraphs, and environmental context so the reader feels the movement from one location to the next.
   - Ensure the new location is painted vividly before characters begin interacting.

3. **Narrative Flow & Polish**
   - Ensure transitions between the chapters are perfectly smooth.
   - Resolve any timeline inconsistencies, plot holes, or contradictions.
   - Enhance the prose to be engaging, atmospheric, and highly immersive.

4. **Explicit Content Preservation (CRITICAL)**
   - You MUST keep ALL explicit content, adult themes, and mature scenes fully intact.
   - Do NOT censor, tone down, fade-to-black, or reduce the explicitness of any scene under any circumstances.
   - You may enhance the prose and flow of explicit scenes, but you must retain their full length, detail, and adult nature.

Then provide:
- A brief summary of the major continuity issues and scene shifts you fixed.
- A heavily revised, polished version of the story with all fixes and expanded transitions applied.
- Maintain the narrative voice and tense.
- Do NOT summarize or condense — you must output the FULL revised story, taking care to flesh out under-defined scenes rather than shrinking them.

Format your response exactly as:
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
