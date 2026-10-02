import { useEffect } from "react";
import { downloadText, projectUrl, send } from "../api";
import type { PromptResult } from "../types";
import { useAction, useResource } from "../hooks";
import { ErrorNotice, useUi } from "./ui";
export function ImagePrompts({
  project,
  source,
}: {
  project: string;
  source: string;
}) {
  const resource = useResource<PromptResult>(
    projectUrl(project, `/image-prompts?source=${encodeURIComponent(source)}`),
  );
  const action = useAction(),
    ui = useUi();
  useEffect(() => {
    if (resource.data?.status === "generating") {
      const timer = setTimeout(resource.reload, 2000);
      return () => clearTimeout(timer);
    }
  }, [resource.data, resource.reload]);
  const highlights = resource.data?.highlights || [];
  const all = highlights
    .map(
      (h) =>
        `${h.title}\n${h.context}\n${h.references.join("\n")}\n${h.prompt}`,
    )
    .join("\n\n");
  return (
    <details className="panel image-prompts">
      <summary>
        Scene image prompts{" "}
        <span className="muted">
          {highlights.length ? `· ${highlights.length} highlights` : ""}
        </span>
      </summary>
      <p className="muted">
        Use these prompts with named character references in your image tool.
      </p>
      <ErrorNotice
        error={resource.error || action.error}
        retry={resource.reload}
      />
      {resource.data?.status === "failed" && (
        <p role="alert">
          {resource.data.error || "Prompt generation failed. Try again."}
        </p>
      )}
      <div className="actions">
        <button
          disabled={action.busy || resource.data?.status === "generating"}
          onClick={() =>
            action.run(async () => {
              await send(projectUrl(project, "/image-prompts"), { source });
              resource.reload();
            })
          }
        >
          {resource.data?.status === "generating"
            ? "Writing prompts…"
            : highlights.length
              ? "Regenerate prompts"
              : "Generate prompts"}
        </button>
        {!!highlights.length && (
          <>
            <button
              onClick={() =>
                action.run(async () => {
                  await navigator.clipboard.writeText(all);
                  ui.notify("Prompts copied.");
                })
              }
            >
              Copy all
            </button>
            <button
              onClick={() => downloadText(all, "scene-image-prompts.txt")}
            >
              Download prompts
            </button>
          </>
        )}
      </div>
      {highlights.map((h, i) => (
        <article className="scene" key={i}>
          <h3>{h.title}</h3>
          <p>{h.context}</p>
          <small>{h.references.join(" · ")}</small>
          <p className="pre-wrap">{h.prompt}</p>
          <button
            onClick={() =>
              action.run(async () => {
                await navigator.clipboard.writeText(h.prompt);
                ui.notify("Prompt copied.");
              })
            }
          >
            Copy prompt
          </button>
        </article>
      ))}
    </details>
  );
}
