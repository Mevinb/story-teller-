import { useEffect, useState } from "react";
import { send } from "../api";
import type { Catalog, Settings as SettingsData, SettingValue } from "../types";
import { useAction, useResource } from "../hooks";
import {
  ErrorNotice,
  Field,
  JsonView,
  Loading,
  PageTitle,
  useUi,
} from "../components/ui";
import { modelOptions } from "../components/Models";
const labels: Record<string, string> = {
  BACKEND_MODE: "Generation backend",
  GROQ_API_KEY: "Groq API key",
  GEMINI_API_KEY: "Gemini API key",
  OPENROUTER_API_KEY: "OpenRouter API key",
  GROQ_API_KEYS: "Additional Groq keys",
  GEMINI_API_KEYS: "Additional Gemini keys",
  OPENROUTER_API_KEYS: "Additional OpenRouter keys",
  GROQ_MODEL: "Groq model",
  GEMINI_MODEL: "Gemini model",
  OPENROUTER_MODEL: "OpenRouter model",
  WORDS_PER_SCENE_MIN: "Target scene words · minimum",
  WORDS_PER_SCENE_MAX: "Target scene words · maximum",
  LLAMA_MODEL_PATH: "Local GGUF model path",
  LLAMA_N_GPU_LAYERS: "GPU layers",
};
export function Settings({
  catalog,
  onChanged,
}: {
  catalog?: Catalog;
  onChanged: () => void;
}) {
  const resource = useResource<SettingsData>("/api/settings");
  const keys = useResource<Record<string, unknown>>("/api/keys/status");
  const [changes, setChanges] = useState<Record<string, SettingValue>>({}),
    [search, setSearch] = useState("");
  const action = useAction(),
    ui = useUi();
  useEffect(() => {
    setChanges({});
  }, [resource.data]);
  const values = { ...resource.data?.settings, ...changes };
  const groups = {
    "Models & providers": Object.keys(values).filter((k) =>
      /^(BACKEND|GROQ_MODEL|GEMINI_MODEL|OPENROUTER_MODEL)|API_KEY/.test(k),
    ),
    "Writing & memory": Object.keys(values).filter((k) =>
      /^(WORDS_|MIN_SCENE|MAX_SCENE|BEST_OF|MAX_PIPELINE|MAX_TOKEN|TOP_K|CONTEXT_)/.test(
        k,
      ),
    ),
    "Advanced settings": Object.keys(values).filter(
      (k) =>
        k !== "ACTIVE_MODEL" &&
        !/^(BACKEND|GROQ_MODEL|GEMINI_MODEL|OPENROUTER_MODEL|WORDS_|MIN_SCENE|MAX_SCENE|BEST_OF|MAX_PIPELINE|MAX_TOKEN|TOP_K|CONTEXT_)|API_KEY/.test(
          k,
        ),
    ),
  };
  const input = (key: string) => {
    const value = values[key],
      secret = key.includes("API_KEY");
    const update = (v: SettingValue) => setChanges((c) => ({ ...c, [key]: v }));
    const provider = key.replace("_MODEL", "").toLowerCase();
    const options = modelOptions(catalog).filter(
      (m) => m.provider === provider,
    );
    if (typeof value === "boolean")
      return (
        <input
          type="checkbox"
          checked={value}
          onChange={(e) => update(e.target.checked)}
        />
      );
    if (key === "BACKEND_MODE")
      return (
        <select value={String(value)} onChange={(e) => update(e.target.value)}>
          {["local", "groq", "gemini", "openrouter", "hybrid"].map((m) => (
            <option key={m}>{m}</option>
          ))}
        </select>
      );
    if (options.length)
      return (
        <>
          <input
            list={`models-${key}`}
            value={String(value)}
            onChange={(e) => update(e.target.value)}
          />
          <datalist id={`models-${key}`}>
            {options.map((m) => (
              <option key={m.id} value={m.id.slice(m.provider.length + 1)}>
                {m.name}
              </option>
            ))}
          </datalist>
        </>
      );
    return (
      <input
        type={
          secret ? "password" : typeof value === "number" ? "number" : "text"
        }
        step={typeof value === "number" ? "any" : undefined}
        autoComplete={secret ? "new-password" : undefined}
        value={String(value)}
        onChange={(e) =>
          update(
            typeof value === "number" ? Number(e.target.value) : e.target.value,
          )
        }
      />
    );
  };
  return (
    <>
      <PageTitle eyebrow="Your tools" title="Settings & models" />
      <ErrorNotice error={resource.error} retry={resource.reload} />
      {resource.loading && <Loading />}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          action.run(async () => {
            await send("/api/settings", { settings: changes }, "PUT");
            resource.reload();
            keys.reload();
            onChanged();
            ui.notify("Settings saved.");
          });
        }}
      >
        <div className="toolbar">
          <input
            aria-label="Search settings"
            placeholder="Search settings…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <button
            className="primary"
            disabled={action.busy || !Object.keys(changes).length}
          >
            {action.busy ? "Saving…" : "Save settings"}
          </button>
        </div>
        <ErrorNotice error={action.error} />
        {Object.entries(groups).map(([group, fields]) => (
          <details
            className="panel"
            open={group !== "Advanced settings" || !!search}
            key={group}
          >
            <summary>{group}</summary>
            <fieldset disabled={action.busy}>
              <div className="two-col">
                {fields
                  .filter((k) =>
                    `${k} ${labels[k] || ""}`
                      .toLowerCase()
                      .includes(search.toLowerCase()),
                  )
                  .map((k) => (
                    <Field
                      key={k}
                      label={labels[k] || k.toLowerCase().replaceAll("_", " ")}
                    >
                      {input(k)}
                    </Field>
                  ))}
              </div>
            </fieldset>
          </details>
        ))}
      </form>
      <details className="panel">
        <summary>Provider availability & quota status</summary>
        <button onClick={keys.reload}>Refresh provider status</button>
        <ErrorNotice error={keys.error} />
        <JsonView value={keys.data} />
      </details>
    </>
  );
}
