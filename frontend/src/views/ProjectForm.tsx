import { useEffect, useState } from "react";
import { projectUrl, send, splitList } from "../api";
import type { Catalog, Character, StoryState } from "../types";
import { useAction, useResource } from "../hooks";
import {
  ErrorNotice,
  Field,
  Loading,
  PageTitle,
  useUi,
} from "../components/ui";
interface Form {
  title: string;
  genre: string;
  premise: string;
  setting: string;
  themes: string;
}
const blank: Form = {
  title: "",
  genre: "Fantasy",
  premise: "",
  setting: "",
  themes: "",
};
export function ProjectForm({
  project,
  active = true,
  onSaved,
  onDeleted,
}: {
  project?: string;
  active?: boolean;
  onSaved: (name: string) => void;
  onDeleted: () => void;
}) {
  const catalog = useResource<Catalog>("/api/models");
  const resource = useResource<{ state: StoryState }>(
    project ? projectUrl(project) : null,
  );
  const [form, setForm] = useState<Form>(blank);
  const [characters, setCharacters] = useState<Record<string, Character>>({});
  const [character, setCharacter] = useState({
    name: "",
    description: "",
    traits: "",
  });
  const [editingName, setEditingName] = useState<string>();
  const [dirty, setDirty] = useState(false);
  const [draftText, setDraftText] = useState("");
  const [archMode, setArchMode] = useState<"continue" | "rearchitect" | "auto">("continue");
  const [targetScope, setTargetScope] = useState<string>("auto");
  const [customBeats, setCustomBeats] = useState<number>(48);
  const [archBusy, setArchBusy] = useState(false);
  const [storyGrasp, setStoryGrasp] = useState<{
    mode_applied?: string;
    cutoff_point?: string;
    central_conflict?: string;
  } | null>(null);

  const action = useAction(),
    ui = useUi();

  const handleAutoArchitect = () => {
    if (!draftText.trim()) return;
    setArchBusy(true);
    action.run(async () => {
      try {
        const payload: Record<string, any> = {
          idea_text: draftText,
          mode: archMode,
          model: catalog.data?.active,
          setting: form.setting,
          themes: splitList(form.themes),
        };
        if (targetScope === "custom") {
          payload.target_beats = customBeats;
        } else if (targetScope !== "auto") {
          payload.target_chapters = Number(targetScope);
        }

        const result = await send<{
          status: string;
          title?: string;
          genre?: string;
          summary?: string;
          setting?: string;
          themes?: string[];
          characters?: Record<string, Character>;
          steps?: string[];
          story_analysis?: {
            mode_applied?: string;
            cutoff_point?: string;
            central_conflict?: string;
          };
        }>("/api/premise/generate", payload);

        const formattedPremise = (result.steps || [])
          .map((s, i) => `${i + 1}. ${s}`)
          .join("\n\n");

        if (!result.steps || result.steps.length === 0) {
          ui.notify("Generation failed to produce narrative beats. Please try again.");
          return;
        }

        setForm((prev) => ({
          ...prev,
          title: (result.title && (!prev.title || prev.title === "USER:" || prev.title.startsWith("Untitled")))
            ? result.title
            : (prev.title.trim() ? prev.title : (result.title || "")),
          genre: result.genre ? result.genre : prev.genre,
          premise: formattedPremise || result.summary || prev.premise,
          setting: result.setting ? result.setting : prev.setting,
          themes: result.themes && result.themes.length > 0 ? result.themes.join(", ") : prev.themes,
        }));

        if (result.characters && Object.keys(result.characters).length > 0) {
          setCharacters((prev) => ({
            ...prev,
            ...result.characters,
          }));
        }

        if (result.story_analysis) {
          setStoryGrasp(result.story_analysis);
        }

        setDirty(true);
        ui.notify("Story details and characters auto-populated!");
      } finally {
        setArchBusy(false);
      }
    });
  };

  useEffect(() => {
    if (!resource.data) return;
    const m = resource.data.state.metadata;
    setForm({
      title: m.title || "",
      genre: m.genre || "",
      premise: m.premise || "",
      setting: m.setting || "",
      themes: (m.themes || []).join(", "),
    });
    setCharacters(resource.data.state.characters || {});
    setDirty(false);
  }, [resource.data]);
  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);
  useEffect(() => {
    if (active && project && !dirty) resource.reload();
  }, [active, project, resource.reload]);
  const field = (key: keyof Form, label: string, multiline = false) => (
    <Field label={label}>
      {multiline ? (
        <textarea
          rows={key === "premise" ? 7 : 3}
          value={form[key]}
          onChange={(e) => {
            setForm({ ...form, [key]: e.target.value });
            setDirty(true);
          }}
        />
      ) : (
        <input
          required={key === "title"}
          maxLength={key === "title" ? 120 : undefined}
          value={form[key]}
          onChange={(e) => {
            setForm({ ...form, [key]: e.target.value });
            setDirty(true);
          }}
        />
      )}
    </Field>
  );
  return (
    <>
      <PageTitle
        title={project ? "Story details" : "Begin a new story"}
        eyebrow={project ? "Your foundation" : "A blank page, a possibility"}
      />
      <ErrorNotice error={resource.error} retry={resource.reload} />
      {resource.loading && <Loading />}
      <form
        className="panel"
        onSubmit={(e) => {
          e.preventDefault();
          action.run(async () => {
            if (character.name.trim())
              throw new Error(
                "Add the character below before saving the story.",
              );
            const metadata = {
              ...form,
              title: form.title.trim(),
              themes: splitList(form.themes),
            };
            if (!metadata.title) throw new Error("Enter a story title.");
            const result = await send<{ project: string }>(
              project ? projectUrl(project, "/state") : "/api/project/create",
              project ? { metadata, characters } : { ...metadata, characters },
              project ? "PUT" : "POST",
            );
            setDirty(false);
            if (project) resource.reload();
            ui.notify(project ? "Story details saved." : "Story created.");
            onSaved(project || result.project);
          });
        }}
      >
        <fieldset disabled={action.busy || (!!project && !resource.data)}>
          {/* Premise & Story Architect Assistant for New Stories */}
          {!project && (
            <div
              style={{
                background: "rgba(139, 92, 246, 0.08)",
                border: "1px solid rgba(139, 92, 246, 0.35)",
                borderRadius: "var(--radius-md, 8px)",
                padding: "16px",
                marginBottom: "20px",
              }}
            >
              <div className="section-heading" style={{ marginBottom: "8px" }}>
                <h2 style={{ fontSize: "1.05rem", color: "#c4b5fd" }}>
                  ✨ Story Architect — Generate from Draft or Notes
                </h2>
                <span className="muted" style={{ fontSize: "0.8rem" }}>
                  Auto-fills title, genre, premise beats, setting & cast
                </span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1.2fr 0.8fr", gap: "12px", marginBottom: "12px" }}>
                <div>
                  <label className="field-label" style={{ fontSize: "0.8rem", color: "var(--text-dim, #888)", display: "block", marginBottom: "4px" }}>
                    Generation Strategy
                  </label>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "6px" }}>
                    <button
                      type="button"
                      className={archMode === "continue" ? "primary" : ""}
                      style={{ fontSize: "0.76rem", padding: "6px 4px" }}
                      onClick={() => setArchMode("continue")}
                      title="Grasp written scenes from half-made story and finish the rest of the story"
                    >
                      ⏩ Continue Draft
                    </button>
                    <button
                      type="button"
                      className={archMode === "rearchitect" ? "primary" : ""}
                      style={{ fontSize: "0.76rem", padding: "6px 4px" }}
                      onClick={() => setArchMode("rearchitect")}
                      title="Extract premise & characters, rebuild full story from Chapter 1 to finale"
                    >
                      🔄 From Start
                    </button>
                    <button
                      type="button"
                      className={archMode === "auto" ? "primary" : ""}
                      style={{ fontSize: "0.76rem", padding: "6px 4px" }}
                      onClick={() => setArchMode("auto")}
                      title="Auto-detect draft vs idea"
                    >
                      ⚡ Auto
                    </button>
                  </div>
                </div>

                <div>
                  <label className="field-label" style={{ fontSize: "0.8rem", color: "var(--text-dim, #888)", display: "block", marginBottom: "4px" }}>
                    Story Scope
                  </label>
                  <select
                    style={{ width: "100%", padding: "7px 10px", fontSize: "0.85rem" }}
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
                </div>
              </div>

              <Field label="Paste half-made story, written scenes, or raw outline">
                <textarea
                  rows={6}
                  value={draftText}
                  onChange={(e) => setDraftText(e.target.value)}
                  placeholder="Paste your unfinished story draft, scene text, or plot dump here. The Story Architect will grasp the story, extract characters and lore, and populate the entire story blueprint below..."
                />
              </Field>

              <button
                type="button"
                className="primary"
                disabled={archBusy || !draftText.trim()}
                onClick={handleAutoArchitect}
                style={{ marginTop: "10px" }}
              >
                {archBusy ? "✨ Analyzing & Architecting Story..." : "✨ Generate & Auto-Fill Story Details"}
              </button>

              {storyGrasp && (
                <div
                  style={{
                    background: "rgba(139,92,246,0.15)",
                    border: "1px solid rgba(139,92,246,0.35)",
                    borderRadius: "6px",
                    padding: "10px 12px",
                    marginTop: "12px",
                    fontSize: "0.82rem",
                  }}
                >
                  <strong>
                    {storyGrasp.mode_applied === "continue"
                      ? "⏩ Story Draft Grasped & Completed"
                      : "🔄 Full Story Re-architected from Beginning"}
                  </strong>
                  {storyGrasp.cutoff_point && (
                    <p style={{ margin: "4px 0 0", color: "var(--text-dim)" }}>
                      <em>Bridge:</em> {storyGrasp.cutoff_point}
                    </p>
                  )}
                  {storyGrasp.central_conflict && (
                    <p style={{ margin: "4px 0 0" }}>
                      <strong>Conflict:</strong> {storyGrasp.central_conflict}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}

          <div className="two-col">
            {field("title", "Story title")}
            {field("genre", "Genre")}
          </div>
          {field("premise", "Premise", true)}
          <div className="two-col">
            {field("setting", "Setting", true)}
            {field("themes", "Themes, separated by commas")}
          </div>
          <div className="section-heading">
            <h2>Characters</h2>
            <span>{Object.keys(characters).length} in your cast</span>
          </div>
          <div className="character-list">
            {Object.entries(characters).map(([name, c]) => (
              <article key={name}>
                <div>
                  <strong>{name}</strong>
                  <p>{c.description}</p>
                  <small>{c.traits?.join(" · ")}</small>
                </div>
                <div className="actions">
                  <button
                    type="button"
                    onClick={() => {
                      setEditingName(name);
                      setCharacter({
                        name,
                        description: c.description || "",
                        traits: (c.traits || []).join(", "),
                      });
                    }}
                  >
                    Edit
                  </button>
                  <button
                    type="button"
                    aria-label={`Remove ${name}`}
                    onClick={() => {
                      setCharacters(
                        Object.fromEntries(
                          Object.entries(characters).filter(
                            ([n]) => n !== name,
                          ),
                        ),
                      );
                      setDirty(true);
                    }}
                  >
                    Remove
                  </button>
                </div>
              </article>
            ))}
          </div>
          <div className="inset">
            <div className="three-col">
              {(["name", "description", "traits"] as const).map((key) => (
                <Field
                  key={key}
                  label={
                    key === "traits"
                      ? "Traits, separated by commas"
                      : `Character ${key}`
                  }
                >
                  <input
                    value={character[key]}
                    onChange={(e) =>
                      setCharacter({ ...character, [key]: e.target.value })
                    }
                  />
                </Field>
              ))}
            </div>
            <button
              type="button"
              onClick={() => {
                const name = character.name.trim();
                if (!name) {
                  action.setError("Enter a character name.");
                  return;
                }
                if (characters[name] && editingName !== name) {
                  action.setError("A character with that name already exists.");
                  return;
                }
                const next = { ...characters };
                const previous = editingName ? next[editingName] : {};
                if (editingName) delete next[editingName];
                next[name] = {
                  ...previous,
                  description: character.description,
                  traits: splitList(character.traits),
                };
                setCharacters(next);
                setCharacter({ name: "", description: "", traits: "" });
                setEditingName(undefined);
                setDirty(true);
                action.setError("");
              }}
            >
              {editingName ? "Update character" : "Add character"}
            </button>
            {editingName && (
              <button
                type="button"
                onClick={() => {
                  setCharacter({ name: "", description: "", traits: "" });
                  setEditingName(undefined);
                }}
              >
                Cancel character edit
              </button>
            )}
          </div>
        </fieldset>
        <ErrorNotice error={action.error} />
        <div className="actions">
          <button
            className="primary"
            disabled={action.busy || (!!project && !resource.data)}
          >
            {action.busy
              ? "Saving…"
              : project
                ? "Save details"
                : "Create story"}
          </button>
          {dirty && <small>Unsaved changes</small>}
          {project && (
            <button
              type="button"
              className="danger push-right"
              disabled={action.busy}
              onClick={() =>
                action.run(async () => {
                  if (
                    await ui.confirm(
                      `Delete “${form.title}” and all its chapters permanently?`,
                    )
                  ) {
                    await send(projectUrl(project, "/delete"));
                    onDeleted();
                  }
                })
              }
            >
              Delete story
            </button>
          )}
        </div>
      </form>
    </>
  );
}
