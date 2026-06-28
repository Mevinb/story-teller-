# Story Teller Agent System Architecture & Extension Guide

This technical report details the inner mechanics, interfaces, prompt structures, and coordination loops of the multi-agent system in **Story Teller**. Use this guide to understand how to safely modify prompts, change JSON schemas, adjust LLM configurations, or extend agent logic.

---

## 1. High-Level Agent Coordination Loop

The storytelling generation pipeline uses a hierarchical multi-agent loop. A single-prompt generation for long chapters typically leads to narrative drift, pacing loss, or repeating ideas. Story Teller addresses this by breaking down the writing process into planning, scene decomposition, prose writing, continuity critique, and styling steps.

The **Pipeline Orchestrator** (`pipeline/orchestrator.py`) coordinates the flow:

```mermaid
graph TD
    PO[Pipeline Orchestrator] <--> |Query/Update state.json| SM[State Manager]
    PO --> |1. Plan Chapter| Arch[Story Architect Agent]
    PO --> |2. Decompose into Scenes| Planner[Scene Planner Agent]
    
    subgraph Scene Graph [3. Progressive Scene Generation Graph]
        Writer[Scene Writer] --> Critic[Consistency Critic]
        Critic --> Dec{Has Blocking Issues?}
        Dec -->|Yes & Iterations < 5| Writer
        Dec -->|No or Max Retries Hit| Editor[Style Editor]
    end
    
    PO --> Scene Graph
    Scene Graph --> |Prose Chunks| VS[FAISS Vector Store]
    Scene Graph --> |Fact Evolution| EE[Evolution Engine]
    EE --> SM
```

---

## 2. Detailed Agent Breakdown

Each agent implements a standard `AgentContract` (defined in `agents/contract.py`) requiring a `run(self, state: dict) -> dict` interface:
```python
class AgentContract(ABC):
    name: str = "agent"
    
    @abstractmethod
    def run(self, state: dict) -> dict:
        """
        Execute one agent step. Must return:
        {
            "output": ...,
            "confidence": float,
            "next_action": str
        }
        """
```

Here is the exhaustive technical blueprint of the five agents:

---

### A. Story Architect Agent (`agents/architect.py`)

*   **Responsibility**: Plans the high-level outline and goals for a specific chapter. It does not write prose; it creates the high-level roadmap.
*   **Execution Environment**: Local llama.cpp model for low-cost, structured schema compliance.
*   **Core Logic**:
    *   **Seed Mode (Chapter 1)**: Focuses on setting up characters, starting setting details, and initiating the first 1-2 steps of the premise.
    *   **Continuation Mode (Chapter 2+)**: Inspects summaries of previous chapters, unfinished subplots, and maps out events matching the exact *mandatory premise steps* assigned to the current chapter cursor.
    *   **Narrative Position Anchor**: Forces the model to operate under a specific story phase (e.g., `RISING_ACTION`, `CLIMAX`) and intensity target (e.g., `0.5/1.0`), preventing it from rushing the climax.
*   **System Prompts**:
    *   `ARCHITECT_SYSTEM`: Instructs the agent to prioritize chapter pacing, arcs, and constraints.
    *   `ARCHITECT_SYSTEM_COMPACT`: Used for cloud models to minimize tokens.
*   **Expected JSON Output Schema**:
    ```json
    {
      "chapter_number": 2,
      "chapter_title": "Title of the Chapter",
      "plot_direction": "High-level summary of what happens",
      "character_arcs": {
        "CharacterName": "How they develop or react in this chapter"
      },
      "key_events": [
        "First key premise step rephrased as a concrete action",
        "Second key premise step rephrased as a concrete action"
      ],
      "constraints": [
        "Special constraints, e.g., 'Do not resolve conflict X yet'"
      ],
      "tone": "Suspenseful / Romantic / Tense",
      "estimated_scenes": 3,
      "unresolved_threads_to_address": ["subplot_a"],
      "new_threads_to_introduce": []
    }
    ```
*   **Self-Correction Loop (`_self_check_plan`)**:
    *   Checks if the planned title is already used.
    *   Validates that *every* mandatory premise step is covered by at least one key event (via word overlap checking).
    *   Ensures characters mentioned in `character_arcs` have actually been introduced in the story state.
    *   If violations are found, it automatically sends a correction prompt listing the errors and re-plans the chapter.

---

### B. Scene Planner Agent (`agents/planner.py`)

*   **Responsibility**: Receives the high-level chapter plan and decomposes it into individual, granular scene plans (typically one scene per key event).
*   **Execution Environment**: Local llama.cpp model.
*   **Core Logic**:
    *   Receives the story context, active character names, and the chapter plan.
    *   Generates scene metadata, including setting, present characters, mood, and target word counts.
    *   **Intimacy Level Detection**: Categorizes scene intimacy as `none`, `romantic`, `sensual`, or `explicit` based on keywords in the scene summary or explicit fields. If `explicit`, it splits the scene into granular beats (buildup $\to$ foreplay $\to$ climax) so the writer has a graphic roadmap.
    *   **Narrative Bridge**: Generates a one-sentence logical link explaining *why* this scene follows the previous scene.
*   **System Prompts**:
    *   `PLANNER_SYSTEM`: Mandates exact usage of user scene summaries (without paraphrasing) and structural planning.
    *   `PLANNER_SYSTEM_COMPACT`: Compressed version for cloud models.
*   **Expected JSON Output Schema**:
    ```json
    {
      "chapter_number": 2,
      "chapter_title": "Title",
      "scenes": [
        {
          "scene_number": 1,
          "type": "setup | build_tension | peak | resolution | transition",
          "summary": "Verbatim key event or brief scene summary",
          "characters_present": ["CharacterA", "CharacterB"],
          "location": "Setting name",
          "mood": "Atmosphere/Emotional tone",
          "key_events": ["Detailed beat 1", "Detailed beat 2"],
          "dialogue_notes": "Important conversational beats to cover",
          "sensory_details": "Sensory highlights (e.g. smell of ozone, flickering candles)",
          "intimacy_level": "none | romantic | sensual | explicit",
          "word_target": 600,
          "narrative_bridge": "Connector sentence linking the end of the previous scene to this scene."
        }
      ]
    }
    ```

---

### C. Scene Writer Agent (`agents/writer.py`)

*   **Responsibility**: Converts the scene-level beats into immersive, narrative prose.
*   **Execution Environment**:
    *   *Primary*: High-quality cloud model (e.g., Groq Llama 3 70B) for rich prose.
    *   *Fallback*: Local uncensored model (llama.cpp) used when content filters trigger blocks (especially on mature/explicit scenes) or during offline/network failures.
*   **Core Logic**:
    *   **Grounding & Continuity Blocks**: Dynamically inserts instructions based on scene location. If the setting has changed since the last scene, it forces a transition paragraph showing travel or passage of time (no teleporting).
    *   **Anti-Mirror Constraint**: Injects the final paragraph of the preceding scene and enforces that the writer must not match its sentence structure or mood. If the previous scene ended in reflection, it forces this scene to open with dialogue or physical action.
    *   **Anti-Censorship/Refusal Guard**: Switches to the local uncensored model if Groq returns a `ContentBlockedError` due to sensitive or explicit themes.
*   **System Prompts**:
    *   `WRITER_SYSTEM` / `WRITER_SYSTEM_COMPACT`: Standard immersive third-person writing parameters.
    *   `WRITER_SYSTEM_EXPLICIT`: Anatomical, uninhibited pacing instructions (never fade to black, never self-censor).
    *   `WRITER_SYSTEM_MATURE_NON_EXPLICIT`: Non-graphic, high-tension romance guidelines.
    *   **Genre-Specific Routers**: Selecting specialized guidelines based on metadata (e.g., `dark fantasy` focusing on grim atmosphere; `thriller` focusing on rapid pacing).
*   **Prose Post-Processing Checks**:
    *   *Length check*: If the output is too brief (e.g. summarizing rather than showing), it triggers a rewrite/expansion prompt.
    *   *Overlap check*: Re-rolls generation if it duplicates sentences from the preceding scene (using 8-gram overlap checks).
    *   *Truncation check*: If the text ends mid-sentence, it requests a continuation starting from the last complete paragraph.

---

### D. Consistency Engine / Critic Agent (`agents/consistency.py`)

*   **Responsibility**: Validates generated scene prose against the world rules, timeline, character traits, and active premise steps.
*   **Execution Environment**: Local llama.cpp model.
*   **Core Logic**:
    *   **Deterministic Fast Pass (Pre-LLM)**:
        1. Checks character presence: Flags a blocking error if a character listed in the scene plan is missing in the generated text.
        2. Checks beat coverage: Verifies that keywords from each planned scene beat appear in the prose. If beats are missing, it triggers a blocking `logic_error`.
    *   **Factual vs. State Transitions**: Distinguishes between **Immutable Facts** (e.g., character identity, dead characters revived, incorrect locations) and **Mutable States** (e.g., emotional changes, relationships, injuries). Mutable state changes are normalized and added to `state_updates` rather than flagged as errors.
    *   **Premise Alignment Pass**: Validates the scene against the allowed premise steps. If the writer accidentally leaks future premise steps, it flags a `future_step_leak` violation.
*   **System Prompts**:
    *   `CONSISTENCY_SYSTEM` / `CONSISTENCY_SYSTEM_COMPACT`: Defines continuity rules, immutable vs. mutable differences, and schema instructions.
    *   `PREMISE_ALIGNMENT_SYSTEM`: Checks boundaries against active premise steps.
*   **Expected JSON Output Schema**:
    ```json
    {
      "is_consistent": false,
      "issues": [
        {
          "type": "character_contradiction | relationship_error | location_error | timeline_error | logic_error | premise_alignment | state_transition | state_hygiene",
          "detail": "Alice is walking in the dungeons, but she is currently locked in the tower.",
          "severity": "high | medium | low",
          "blocking": true,
          "suggestion": "Show Alice escaping the tower or change the scene location.",
          "nearest_paragraph": "The exact paragraph in the prose where the error is."
        }
      ],
      "state_updates": {
        "characters": {
          "Alice": {
            "state": { "emotion": "terrified", "location": "dungeons" },
            "arc_progression": "Alice escaped the tower but was captured in the dungeons."
          }
        }
      }
    }
    ```

---

### E. Style Editor Agent (`agents/editor.py`)

*   **Responsibility**: Polishes grammar, vocabulary, paragraph structure, and dialogue formatting without altering the plot, character choices, or events.
*   **Execution Environment**: Local llama.cpp model.
*   **Core Logic**:
    *   Accepts raw prose and copies style details.
    *   **Safety check**: If the output contains meta-text or the token count has drifted by more than 50% (under-generation) or 200% (over-generation), it discards the edits and keeps the original writer draft.
    *   **Censorship Guard**: Explicitly instructed not to alter, soft-censor, or sanitize graphic/erotic content during copy edits.
*   **System Prompts**:
    *   `EDITOR_SYSTEM` / `EDITOR_SYSTEM_COMPACT`: General fiction editing and dialogue styling.

---

## 3. The Scene Generation Graph Cycle

Once the orchestrator receives the scene plans, it starts a cyclic state machine (`_run_scene_graph`) for each scene:

```
[Start Scene]
      │
      ▼
┌──────────────┐
│ Scene Writer │◄──────────────────┐
└──────┬───────┘                   │
       │ (Prose)                   │ (Blocking issues/Rewrite)
       ▼                           │
┌──────────────┐                   │
│  Consistency │───────────────────┘
│    Critic    │ (Blocking == True & Iterations < 5)
└──────┬───────┘
       │ (No Blocking Issues OR Iterations >= 5)
       ▼
┌──────────────┐
│ Style Editor │
└──────┬───────┘
       │ (Polished Prose)
       ▼
  [End Scene]
```

1.  **Scene Writer** generates the raw draft based on the scene plan.
2.  **Consistency Critic** runs validation.
    *   If **blocking issues** are found (e.g., wrong location, dead characters, missing beats) and we have attempted fewer than 5 rewrites, the orchestrator requests a rewrite from the Scene Writer, feeding the issues list back into the prompt.
    *   If the issues are **non-blocking** (e.g., emotional evolution), the orchestrator notes the changes to apply to `state.json` later.
3.  Once consistent or the retry limit (5) is reached, the prose goes to the **Style Editor**.
4.  The final polished prose is saved to a WIP JSON, indexed into the **FAISS Vector Store**, and the **Evolution Engine** extracts state changes to update `state.json`.

---

## 4. Practical Customization Guide

### A. How to Modify System Prompts
*   To adjust the narrative voice of the story or specific genres, modify the `GENRE_SYSTEM_PROMPTS` dict inside `agents/writer.py`.
*   To tweak scene-planning rules, update `PLANNER_SYSTEM` inside `agents/planner.py`.
*   To change high-level planning constraints, update `ARCHITECT_SYSTEM` inside `agents/architect.py`.

### B. How to Modify JSON Output Schemas
If you add fields to the JSON schemas, you must update:
1.  The JSON schema definition passed to `generate_with_retry` (e.g., `schema` dict in `StoryArchitect.plan_chapter`).
2.  The fallback parser/coercer functions (e.g., `_coerce_plan` or `_coerce_scene_plan`) to ensure default values are set in case the LLM omits the new field.

### C. How to Adjust Retry and Loop Parameters
All parameters regulating pacing, retry counts, vector weights, and model boundaries are located in `config.py`.
*   `MAX_REPETITION_RATIO`: Controls how strict the writer's repetition checks are.
*   `REANCHOR_EVERY_N_CHAPTERS`: Controls how frequently the Story Architect executes a summary compaction.
*   `WORDS_PER_SCENE_MIN` / `WORDS_PER_SCENE_MAX`: Regulates word count limits.

---

## 5. Summary Checklist for Making Code Changes

| Goal | Files to Modify | Notes |
| :--- | :--- | :--- |
| **Change how chapters are planned** | `agents/architect.py` | Update prompts (`PLAN_CHAPTER_SEED` / `PLAN_CHAPTER_CONTINUE`) or validation logic (`_collect_plan_violations`). |
| **Change how scenes are structured** | `agents/planner.py` | Modify the scene template/schema or intimacy detection functions (`_normalize_intimacy_level`). |
| **Adjust writing style or vocabulary** | `agents/writer.py` | Modify `WRITER_SYSTEM` or add new rules to `GENRE_SYSTEM_PROMPTS`. |
| **Add a new check (e.g., word frequency, pacing check)** | `agents/consistency.py` | Add the check to `validate()`. Ensure you categorize it as blocking/non-blocking correctly. |
| **Change how agents are run sequentially** | `pipeline/orchestrator.py` | Modify `_run_scene_graph()` or `generate_chapter()`. |
