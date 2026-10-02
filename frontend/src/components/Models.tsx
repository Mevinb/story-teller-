import { useState } from "react";
import type { Catalog } from "../types";
import { send } from "../api";
import { useAction } from "../hooks";
import { Dialog, ErrorNotice } from "./ui";

export function modelOptions(catalog?: Catalog) {
  if (!catalog) return [];
  return [
    ...(["groq", "gemini", "openrouter"] as const).flatMap((provider) =>
      (catalog[`${provider}_models`] || []).map((m) => ({
        id: `${provider}:${m.id}`,
        name: m.name || m.id,
        provider,
        description: m.description || "",
      })),
    ),
    ...(catalog.local_models || []).map((id) => ({
      id,
      name: id,
      provider: "local",
      description: "Local GGUF model",
    })),
    {
      id: "hybrid",
      name: "Hybrid",
      provider: "hybrid",
      description: "Cloud writing with local planning when available",
    },
  ];
}
export const modelName = (id = "") =>
  id
    .replace(/^(groq|gemini|openrouter):/, "")
    .split("/")
    .pop()
    ?.replace(".gguf", "") || "Choose model";
export function ModelPicker({
  catalog,
  onClose,
  onChanged,
}: {
  catalog?: Catalog;
  onClose: () => void;
  onChanged: () => void;
}) {
  const [filter, setFilter] = useState("all");
  const action = useAction();
  return (
    <Dialog title="Choose a writing model" onClose={onClose}>
      <div className="segmented">
        {["all", "groq", "gemini", "openrouter", "local", "hybrid"].map((p) => (
          <button
            key={p}
            aria-pressed={filter === p}
            onClick={() => setFilter(p)}
          >
            {p}
          </button>
        ))}
      </div>
      <ErrorNotice error={action.error} />
      <div className="model-list">
        {modelOptions(catalog)
          .filter((m) => filter === "all" || m.provider === filter)
          .map((m) => (
            <button
              className="model-option"
              key={m.id}
              disabled={action.busy}
              aria-pressed={catalog?.active === m.id}
              onClick={() =>
                action.run(async () => {
                  await send("/api/models/switch", { model: m.id });
                  onChanged();
                  onClose();
                })
              }
            >
              <span className="eyebrow">{m.provider}</span>
              <strong>{m.name}</strong>
              <small>{m.description || m.id}</small>
              {catalog?.active === m.id && <span className="tag">Active</span>}
            </button>
          ))}
      </div>
      {action.busy && <p role="status">Checking model availability…</p>}
    </Dialog>
  );
}
