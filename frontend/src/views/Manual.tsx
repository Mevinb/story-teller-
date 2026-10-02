import { useEffect, useState } from "react";
import { projectUrl, send, words } from "../api";
import type { ManualStatus } from "../types";
import type { StreamController } from "../stream";
import { useAction, useResource } from "../hooks";
import {
  Dialog,
  ErrorNotice,
  Field,
  Loading,
  PageTitle,
  Prose,
  useUi,
} from "../components/ui";
import { Pacing } from "./Generation";
export function Manual({
  project,
  stream,
  suggestedBrief,
  active,
}: {
  project: string;
  stream: StreamController;
  suggestedBrief: string;
  active: boolean;
}) {
  const resource = useResource<ManualStatus>(
    projectUrl(project, "/generate/manual/status"),
  );
  const [title, setTitle] = useState(""),
    [pacing, setPacing] = useState("moderate"),
    [brief, setBrief] = useState(suggestedBrief),
    [typed, setTyped] = useState("");
  const [editing, setEditing] = useState<{ index: number; text: string }>();
  const action = useAction(),
    ui = useUi();
  useEffect(() => {
    resource.reload();
  }, [stream.revision, resource.reload]);
  useEffect(() => {
    if (suggestedBrief) setBrief(suggestedBrief);
  }, [suggestedBrief]);
  useEffect(() => {
    if (active) resource.reload();
  }, [active, resource.reload]);
  useEffect(() => {
    if (resource.data?.is_generating && !stream.active) {
      const timer = setTimeout(resource.reload, 1000);
      return () => clearTimeout(timer);
    }
  }, [resource.data, stream.active, resource.reload]);
  const status = resource.data;
  const busy = action.busy || stream.active || status?.is_generating;
  const mutate = (
    suffix: string,
    body?: unknown,
    method = "POST",
    streaming = false,
  ) =>
    action.run(async () => {
      await send(projectUrl(project, suffix), body, method);
      if (streaming) stream.start();
      else resource.reload();
    });
  return (
    <>
      <PageTitle eyebrow="Write scene by scene" title="The manual studio" />
      <ErrorNotice error={resource.error} retry={resource.reload} />
      {resource.loading && !status && <Loading />}
      {!status?.active ? (
        <section className="panel narrow">
          <h2>Open a chapter</h2>
          <Field label="Chapter title">
            <input value={title} onChange={(e) => setTitle(e.target.value)} />
          </Field>
          <Pacing value={pacing} onChange={setPacing} />
          <button
            className="primary"
            disabled={!!busy || !title.trim() || !status}
            onClick={() =>
              mutate("/generate/manual/start", { chapter_title: title, pacing })
            }
          >
            Start chapter
          </button>
          <p className="muted">
            To continue a saved draft, open it in the Reader and choose Resume.
          </p>
        </section>
      ) : (
        <>
          <section className="panel">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Chapter {status.chapter_num}</p>
                <h2>{status.chapter_title}</h2>
              </div>
              <span className="tag">{status.scenes_completed || 0} scenes</span>
            </div>
            <div className="two-col">
              <div>
                <Field label="What happens in the next scene?">
                  <textarea
                    rows={6}
                    value={brief}
                    onChange={(e) => setBrief(e.target.value)}
                  />
                </Field>
                <button
                  className="primary"
                  disabled={!!busy || !brief.trim()}
                  onClick={() =>
                    mutate(
                      "/generate/manual/scene",
                      { scene_brief: brief },
                      "POST",
                      true,
                    )
                  }
                >
                  Generate scene
                </button>
              </div>
              <div>
                <Field label="Or write your own scene">
                  <textarea
                    rows={6}
                    value={typed}
                    onChange={(e) => setTyped(e.target.value)}
                  />
                </Field>
                <button
                  disabled={!!busy || !typed.trim()}
                  onClick={() =>
                    action.run(async () => {
                      await send(
                        projectUrl(project, "/generate/manual/scene/typed"),
                        { text: typed },
                      );
                      setTyped("");
                      resource.reload();
                    })
                  }
                >
                  Add written scene
                </button>
              </div>
            </div>
            <div className="actions">
              {stream.active && (
                <button
                  className="danger"
                  disabled={action.busy}
                  onClick={() => mutate("/generate/manual/cancel_scene")}
                >
                  Stop scene
                </button>
              )}
              <button
                disabled={!!busy || !status.scenes_completed}
                onClick={() =>
                  mutate("/generate/manual/finish", undefined, "POST", true)
                }
              >
                Finish chapter
              </button>
              <button
                disabled={!!busy}
                onClick={() =>
                  action.run(async () => {
                    if (
                      await ui.confirm(
                        "Stop this session and keep the chapter draft for later?",
                      )
                    ) {
                      await send(projectUrl(project, "/generate/manual/abort"));
                      resource.reload();
                    }
                  })
                }
              >
                Save draft & stop
              </button>
              <button onClick={resource.reload}>Refresh scenes</button>
            </div>
          </section>
          {(stream.active || stream.draft) && (
            <section className="panel">
              <p className="stream-status" role="status">
                {stream.status}
              </p>
              <Prose text={stream.draft} />
            </section>
          )}
          {(status.completed_scenes || []).map((scene, i, all) => {
            const text = typeof scene === "string" ? scene : scene.text;
            return (
              <article className="panel" key={i}>
                <div className="section-heading">
                  <h2>Scene {i + 1}</h2>
                  <small>{words(text)} words</small>
                </div>
                <Prose text={text} />
                <div className="actions">
                  <button
                    disabled={!!busy}
                    onClick={() => setEditing({ index: i, text })}
                  >
                    Edit scene {i + 1}
                  </button>
                  {i === all.length - 1 && (
                    <button
                      disabled={!!busy}
                      onClick={() =>
                        action.run(async () => {
                          if (
                            await ui.confirm(
                              `Replace scene ${i + 1} with a newly generated version?`,
                            )
                          ) {
                            await send(
                              projectUrl(
                                project,
                                `/generate/manual/scene/${i}/regenerate`,
                              ),
                              {},
                            );
                            stream.start();
                          }
                        })
                      }
                    >
                      Regenerate
                    </button>
                  )}
                  <button
                    className="danger"
                    disabled={!!busy}
                    onClick={() =>
                      action.run(async () => {
                        if (await ui.confirm(`Delete scene ${i + 1}?`)) {
                          await send(
                            projectUrl(project, `/generate/manual/scene/${i}`),
                            undefined,
                            "DELETE",
                          );
                          resource.reload();
                        }
                      })
                    }
                  >
                    Delete scene
                  </button>
                </div>
              </article>
            );
          })}
        </>
      )}
      <ErrorNotice error={action.error || stream.error} />
      {editing && (
        <Dialog
          title={`Edit scene ${editing.index + 1}`}
          onClose={() => setEditing(undefined)}
        >
          <Field label="Scene prose">
            <textarea
              rows={16}
              value={editing.text}
              onChange={(e) => setEditing({ ...editing, text: e.target.value })}
            />
          </Field>
          <ErrorNotice error={action.error} />
          <button
            className="primary"
            disabled={action.busy || !editing.text.trim()}
            onClick={() =>
              action.run(async () => {
                await send(
                  projectUrl(
                    project,
                    `/generate/manual/scene/${editing.index}`,
                  ),
                  { text: editing.text },
                  "PUT",
                );
                setEditing(undefined);
                resource.reload();
              })
            }
          >
            Save scene
          </button>
        </Dialog>
      )}
    </>
  );
}
