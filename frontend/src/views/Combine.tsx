import { useEffect, useState } from "react";
import { projectUrl, send, words } from "../api";
import type { Catalog, Version } from "../types";
import type { StreamController } from "../stream";
import { useAction, useResource } from "../hooks";
import {
  Empty,
  ErrorNotice,
  Field,
  Loading,
  PageTitle,
  Prose,
  useUi,
} from "../components/ui";
import { ImagePrompts } from "../components/ImagePrompts";
export function Combine({
  project,
  catalog,
  stream,
}: {
  project: string;
  catalog?: Catalog;
  stream: StreamController;
}) {
  const versions = useResource<{ versions: Version[] }>(
    projectUrl(project, "/combine/versions"),
  );
  const [suffix, setSuffix] = useState(""),
    [model, setModel] = useState(""),
    [label, setLabel] = useState("");
  const version = useResource<{ revised: string; analysis: string }>(
    suffix
      ? projectUrl(project, `/combine/version/${encodeURIComponent(suffix)}`)
      : null,
  );
  const action = useAction(),
    ui = useUi();
  useEffect(() => {
    if (catalog) setModel((m) => m || catalog.gemini_model);
  }, [catalog]);
  useEffect(() => {
    if (versions.data)
      setSuffix((s) =>
        versions.data!.versions.some((v) => v.suffix === s)
          ? s
          : versions.data!.versions.at(-1)?.suffix || "",
      );
  }, [versions.data]);
  useEffect(() => {
    versions.reload();
    version.reload();
  }, [stream.revision, versions.reload, version.reload]);
  const start = (mode: string) =>
    action.run(async () => {
      await send(projectUrl(project, "/combine"), { model, mode });
      stream.start();
    });
  return (
    <>
      <PageTitle eyebrow="The finished work" title="Combine & polish" />
      <section className="panel">
        <div className="two-col">
          <div>
            <h2>A complete manuscript</h2>
            <p className="muted">
              Combine existing chapters for continuity editing, or write a whole
              story from the saved premise.
            </p>
          </div>
          <Field label="Gemini model">
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {catalog?.gemini_models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name || m.id}
                </option>
              ))}
              {model && !catalog?.gemini_models.some((m) => m.id === model) && (
                <option>{model}</option>
              )}
            </select>
          </Field>
        </div>
        <ErrorNotice error={action.error || stream.error} />
        <div className="actions">
          <button
            className="primary"
            disabled={action.busy || stream.active || !model}
            onClick={() => start("polish")}
          >
            Polish existing chapters
          </button>
          <button
            disabled={action.busy || stream.active || !model}
            onClick={() => start("whole")}
          >
            Generate whole story
          </button>
          {stream.active && (
            <button
              className="danger"
              disabled={action.busy}
              onClick={() =>
                action.run(async () => {
                  await send(projectUrl(project, "/combine/cancel"));
                })
              }
            >
              Stop processing
            </button>
          )}
        </div>
        {stream.active && (
          <p className="stream-status" role="status">
            {stream.status}
          </p>
        )}
      </section>
      <section className="panel">
        <div className="section-heading">
          <h2>Saved versions</h2>
          <button onClick={versions.reload}>Refresh versions</button>
        </div>
        <ErrorNotice
          error={versions.error || version.error}
          retry={versions.reload}
        />
        {versions.loading && <Loading />}
        {versions.data?.versions.length ? (
          <>
            <Field label="Version">
              <select
                value={suffix}
                onChange={(e) => setSuffix(e.target.value)}
              >
                {versions.data.versions.map((v) => (
                  <option value={v.suffix} key={v.suffix}>
                    {v.label}
                  </option>
                ))}
              </select>
            </Field>
            <div className="toolbar">
              <input
                aria-label="New version label"
                placeholder="Label this version"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
              />
              <button
                disabled={action.busy || !label.trim()}
                onClick={() =>
                  action.run(async () => {
                    const result = await send<{ new_suffix: string }>(
                      projectUrl(project, "/combine/rename"),
                      { suffix, new_name: label },
                    );
                    setSuffix(result.new_suffix);
                    setLabel("");
                    versions.reload();
                  })
                }
              >
                Rename
              </button>
              <button
                className="danger"
                disabled={action.busy}
                onClick={() =>
                  action.run(async () => {
                    if (
                      await ui.confirm(
                        "Permanently delete this polished version?",
                      )
                    ) {
                      await send(
                        projectUrl(project, "/combine/delete_version"),
                        { suffix },
                      );
                      setSuffix("");
                      versions.reload();
                    }
                  })
                }
              >
                Delete version
              </button>
            </div>
            <div className="actions">
              {["polished", "analysis", "original"].map((type) => (
                <a
                  className="button"
                  key={type}
                  href={projectUrl(
                    project,
                    `/combine/download/${type}?suffix=${encodeURIComponent(suffix)}`,
                  )}
                >
                  Download {type}
                </a>
              ))}
            </div>
          </>
        ) : (
          !versions.loading && <Empty title="No polished versions yet." />
        )}
      </section>
      {suffix &&
        (version.loading ? (
          <Loading />
        ) : (
          version.data && (
            <>
              <details className="panel">
                <summary>Editorial analysis</summary>
                <Prose
                  text={
                    version.data.analysis ||
                    "No analysis saved for this version."
                  }
                />
              </details>
              <section className="panel">
                <div className="section-heading">
                  <h2>Polished manuscript</h2>
                  <small>
                    {words(version.data.revised).toLocaleString()} words
                  </small>
                </div>
                <Prose text={version.data.revised} />
              </section>
              <ImagePrompts
                key={`${suffix}:${version.data.revised}`}
                project={project}
                source={`story:${suffix}`}
              />
            </>
          )
        ))}
    </>
  );
}
