import { useEffect, useState } from "react";
import { projectUrl, send, splitList } from "../api";
import type { StoryState } from "../types";
import { useAction, useResource } from "../hooks";
import {
  ErrorNotice,
  Field,
  Loading,
  PageTitle,
  useUi,
} from "../components/ui";

export function parseBeats(text: string) {
  return text
    .split("\n")
    .map((s) => s.trim())
    .filter(
      (s) =>
        s &&
        !/^(?:#+\s*|act\s+|chapter\s+|part\s+|section\s+|phase\s+|prologue|epilogue)/i.test(
          s,
        ) &&
        !(s.endsWith(":") && s.split(/\s+/).length <= 5),
    )
    .map((s) => s.replace(/^(?:[-*•]|\d+[).:-]|(?:\*{0,2}(?:step|beat|scene)\s*\d+[\s:.-]*\*{0,2}))\s*/i, "").trim())
    .map((s) => s.replace(/^\*{1,2}(.*?)\*{1,2}\s*:\s*/, "$1: ").trim())
    .filter(Boolean);
}

export function Premise({
  project,
  active,
}: {
  project: string;
  active: boolean;
}) {
  const resource = useResource<{ state: StoryState }>(projectUrl(project));
  const [steps, setSteps] = useState<string[]>([]);
  const [idea, setIdea] = useState("");
  const [setting, setSetting] = useState("");
  const [themes, setThemes] = useState("");
  const [mode, setMode] = useState<"continue" | "rearchitect" | "auto">("continue");
  const [targetScope, setTargetScope] = useState<string>("auto");
  const [customBeats, setCustomBeats] = useState<number>(48);
  const [storyGrasp, setStoryGrasp] = useState<{
    mode_applied?: string;
    cutoff_point?: string;
    central_conflict?: string;
    summary?: string;
  } | null>(null);

  const action = useAction(),
    ui = useUi();

  useEffect(() => {
    if (resource.data) {
      const m = resource.data.state.metadata;
      setSteps(parseBeats(m.premise || ""));
      setSetting(m.setting || "");
      setThemes((m.themes || []).join(", "));
    }
  }, [resource.data]);

  const saved = resource.data?.state.metadata;
  const dirty =
    !!saved &&
    (JSON.stringify(steps) !==
      JSON.stringify(parseBeats(saved.premise || "")) ||
      setting !== (saved.setting || "") ||
      themes !== (saved.themes || []).join(", "));

  useEffect(() => {
    if (active && !dirty) resource.reload();
  }, [active, resource.reload]);

  const generate = (actionType: string) =>
    action.run(async () => {
      interface GenResult {
        steps: string[];
        characters?: Record<string, any>;
        setting?: string;
        themes?: string[];
        summary?: string;
        story_analysis?: {
          mode_applied?: string;
          cutoff_point?: string;
          central_conflict?: string;
        };
      }
      const payload: Record<string, any> = {
        idea_text: idea,
        mode,
        apply_to_bible: true,
        setting,
        themes: splitList(themes),
      };
      if (targetScope === "custom") {
        payload.target_beats = customBeats;
      } else if (targetScope !== "auto") {
        payload.target_chapters = Number(targetScope);
      }

      const result = await send<GenResult>(
        projectUrl(project, `/premise/${actionType}`),
        actionType === "generate"
          ? payload
          : { steps, setting, themes: splitList(themes) },
      );

      setSteps(result.steps || []);
      if (result.setting && !setting) setSetting(result.setting);
      if (result.themes && result.themes.length && !themes) setThemes(result.themes.join(", "));
      if (result.story_analysis) {
        setStoryGrasp({
          ...result.story_analysis,
          summary: result.summary,
        });
      }
      if (actionType === "generate") {
        ui.notify(`Generated ${result.steps?.length || 0} narrative beats!`);
        resource.reload();
      }
    });

  return (
    <>
      <PageTitle title="Shape the story" eyebrow="Premise & timeline" />
      <ErrorNotice error={resource.error} retry={resource.reload} />
      {resource.loading && <Loading />}
      <div className="split-layout">
        <aside className="panel">
          <h2>Story Architect</h2>

          <Field label="Strategy">
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "6px" }}>
              <button
                type="button"
                className={mode === "continue" ? "primary" : ""}
                style={{ fontSize: "0.78rem", padding: "6px 4px" }}
                onClick={() => setMode("continue")}
                title="Preserve written scenes from half-made story, then architect remaining beats to the climax"
              >
                ⏩ Continue
              </button>
              <button
                type="button"
                className={mode === "rearchitect" ? "primary" : ""}
                style={{ fontSize: "0.78rem", padding: "6px 4px" }}
                onClick={() => setMode("rearchitect")}
                title="Extract premise & characters, rebuild full story from Chapter 1 to finale"
              >
                🔄 From Start
              </button>
              <button
                type="button"
                className={mode === "auto" ? "primary" : ""}
                style={{ fontSize: "0.78rem", padding: "6px 4px" }}
                onClick={() => setMode("auto")}
                title="Auto-detect draft vs idea"
              >
                ⚡ Auto
              </button>
            </div>
          </Field>

          <Field label="Story Scope">
            <select
              value={targetScope}
              onChange={(e) => setTargetScope(e.target.value)}
            >
              <option value="auto">⚡ Auto (Match Draft Density & Length)</option>
              <option value="5">Short Story (5 Ch / ~15 Beats)</option>
              <option value="8">Standard Story (8 Ch / ~24 Beats)</option>
              <option value="12">Novella (12 Ch / ~36 Beats)</option>
              <option value="16">Full Novel (16 Ch / ~48 Beats)</option>
              <option value="20">Extended Novel (20 Ch / ~60 Beats)</option>
              <option value="25">Epic Saga (25 Ch / ~75 Beats)</option>
              <option value="30">Dense Chronicle (30 Ch / ~90 Beats)</option>
              <option value="40">Massive Manuscript (40 Ch / ~120 Beats)</option>
              <option value="50">Mega Narrative (50 Ch / ~150 Beats)</option>
              <option value="custom">Custom Beat Count...</option>
            </select>
            {targetScope === "custom" && (
              <div style={{ marginTop: "6px" }}>
                <input
                  type="number"
                  min={6}
                  max={150}
                  value={customBeats}
                  onChange={(e) => setCustomBeats(Math.max(6, Math.min(150, Number(e.target.value) || 6)))}
                  placeholder="Beat count (6-150)"
                  style={{ width: "100%", padding: "6px 10px", fontSize: "0.85rem" }}
                />
              </div>
            )}
          </Field>

          <Field label="Story draft or rough notes">
            <textarea
              rows={8}
              value={idea}
              onChange={(e) => setIdea(e.target.value)}
              placeholder="Paste half-made story draft, written scenes, or premise outline here..."
            />
          </Field>

          <button
            className="primary"
            disabled={action.busy || !idea.trim()}
            onClick={() => generate("generate")}
          >
            {action.busy ? "Architecting..." : "Architect story beats"}
          </button>

          <Field label="Setting">
            <textarea
              rows={2}
              value={setting}
              onChange={(e) => setSetting(e.target.value)}
              placeholder="e.g. Victorian London, 1888"
            />
          </Field>
          <Field label="Themes">
            <input
              value={themes}
              onChange={(e) => setThemes(e.target.value)}
              placeholder="e.g. betrayal, redemption"
            />
          </Field>
          <p className="muted">
            Cast:{" "}
            {Object.keys(resource.data?.state.characters || {}).join(", ") ||
              "Discovered during generation or add in Story Bible."}
          </p>
        </aside>

        <section className="panel">
          <div className="section-heading">
            <h2>Narrative beats ({steps.length})</h2>
            <button
              disabled={action.busy}
              onClick={() => setSteps([...steps, ""])}
            >
              + Add beat
            </button>
          </div>

          {storyGrasp && (
            <div
              style={{
                background: "rgba(139,92,246,0.12)",
                border: "1px solid rgba(139,92,246,0.3)",
                borderRadius: "6px",
                padding: "10px 12px",
                marginBottom: "12px",
                fontSize: "0.82rem",
              }}
            >
              <strong>
                {storyGrasp.mode_applied === "continue"
                  ? "⏩ Story Draft Continued to Climax"
                  : "🔄 Full Story Re-architected"}
              </strong>
              {storyGrasp.cutoff_point && (
                <p style={{ margin: "4px 0 0", color: "var(--text-dim)" }}>
                  <em>Bridge:</em> {storyGrasp.cutoff_point}
                </p>
              )}
              {storyGrasp.summary && (
                <p style={{ margin: "4px 0 0" }}>{storyGrasp.summary}</p>
              )}
            </div>
          )}

          <fieldset disabled={action.busy}>
            {steps.map((step, i) => (
              <div className="beat" key={i}>
                <span className="beat-index">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <textarea
                  aria-label={`Beat ${i + 1}`}
                  value={step}
                  onChange={(e) =>
                    setSteps(
                      steps.map((s, j) => (i === j ? e.target.value : s)),
                    )
                  }
                />
                <div className="beat-controls">
                  {[-1, 1].map((dir) => (
                    <button
                      key={dir}
                      aria-label={`Move beat ${i + 1} ${dir < 0 ? "up" : "down"}`}
                      disabled={i + dir < 0 || i + dir >= steps.length}
                      onClick={() => {
                        const next = [...steps];
                        [next[i], next[i + dir]] = [next[i + dir], next[i]];
                        setSteps(next);
                      }}
                    >
                      {dir < 0 ? "↑" : "↓"}
                    </button>
                  ))}
                  <button
                    aria-label={`Remove beat ${i + 1}`}
                    onClick={() => setSteps(steps.filter((_, j) => i !== j))}
                  >
                    ×
                  </button>
                </div>
              </div>
            ))}
          </fieldset>
          <ErrorNotice error={action.error} />
          <div className="actions">
            <button
              disabled={action.busy || !steps.length}
              onClick={() => generate("refine")}
            >
              Refine flow
            </button>
            <button
              disabled={action.busy || !steps.length}
              onClick={() => generate("expand")}
            >
              Expand beats
            </button>
            <button
              className="primary"
              disabled={action.busy || !resource.data}
              onClick={() =>
                action.run(async () => {
                  await send(
                    projectUrl(project, "/state"),
                    {
                      metadata: {
                        premise: steps
                          .map((s, i) => `${i + 1}. ${s}`)
                          .join("\n\n"),
                        setting,
                        themes: splitList(themes),
                      },
                    },
                    "PUT",
                  );
                  resource.reload();
                  ui.notify("Premise saved to the story bible.");
                })
              }
            >
              {action.busy ? "Working…" : "Save timeline"}
            </button>
          </div>
        </section>
      </div>
    </>
  );
}
