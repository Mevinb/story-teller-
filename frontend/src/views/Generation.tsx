import { useState } from "react";
import { projectUrl, send, words } from "../api";
import { useAction, useResource } from "../hooks";
import type { StreamController } from "../stream";
import {
  Empty,
  ErrorNotice,
  Field,
  JsonView,
  PageTitle,
  Prose,
} from "../components/ui";
export function Pacing({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <Field label="Pacing">
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {["slow", "moderate", "fast"].map((p) => (
          <option key={p}>{p}</option>
        ))}
      </select>
    </Field>
  );
}
export function Generation({
  project,
  stream,
  onBranch,
}: {
  project: string;
  stream: StreamController;
  onBranch: (summary: string) => void;
}) {
  const [pacing, setPacing] = useState("moderate"),
    [count, setCount] = useState(1),
    [hint, setHint] = useState("");
  const [options, setOptions] = useState<{ title: string; summary: string }[]>(
      [],
    ),
    [report, setReport] = useState(false);
  const action = useAction();
  const continuity = useResource<unknown>(
    report ? projectUrl(project, "/continuity") : null,
  );
  return (
    <>
      <PageTitle
        eyebrow="Automatic writing"
        title="Let the next chapter unfold."
      />
      <div className="split-layout">
        <aside className="panel">
          <h2>Chapter plan</h2>
          <Pacing value={pacing} onChange={setPacing} />
          <Field label="Chapters">
            <select
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
            >
              {[1, 2, 3, 5, 10, 20].map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
              <option value={-1}>Continue until stopped</option>
              <option value={-2}>Until premise is complete</option>
            </select>
          </Field>
          <ErrorNotice error={action.error} />
          <div className="actions">
            {stream.active ? (
              <button
                className="danger"
                disabled={action.busy}
                onClick={() =>
                  action.run(async () => {
                    await send(projectUrl(project, "/generate/cancel"));
                  })
                }
              >
                Stop generation
              </button>
            ) : (
              <button
                className="primary"
                disabled={action.busy}
                onClick={() =>
                  action.run(async () => {
                    await send(projectUrl(project, "/generate"), {
                      pacing,
                      chapter_count: count,
                    });
                    stream.start();
                  })
                }
              >
                {action.busy ? "Starting…" : "Write chapters"}
              </button>
            )}
          </div>
          <hr />
          <h3>Explore a direction</h3>
          <Field label="Direction hint">
            <textarea
              value={hint}
              onChange={(e) => setHint(e.target.value)}
              placeholder="A secret comes to light…"
            />
          </Field>
          <button
            disabled={action.busy || stream.active}
            onClick={() =>
              action.run(async () => {
                const data = await send<{
                  options: { title: string; summary: string }[];
                }>(projectUrl(project, "/branch-options"), {
                  direction_hint: hint,
                });
                setOptions(data.options || []);
              })
            }
          >
            Suggest paths
          </button>
          <button
            onClick={() => {
              setReport(!report);
              if (!report) continuity.reload();
            }}
          >
            Continuity report
          </button>
        </aside>
        <section className="panel">
          <div className="section-heading">
            <h2>Live manuscript</h2>
            <span className="tag">
              {stream.scenes
                .reduce((n, s) => n + words(s.text), 0)
                .toLocaleString()}{" "}
              words
            </span>
          </div>
          <p className="stream-status" role="status">
            <span
              className={stream.active ? "status-dot busy" : "status-dot"}
            />
            {stream.status}
          </p>
          <ErrorNotice error={stream.error} />
          <div className="pipeline">
            {["Architect", "Planner", "Writer", "Consistency", "Editor"].map(
              (name) => (
                <span
                  key={name}
                  className={
                    stream.agent.toLowerCase().includes(name.toLowerCase())
                      ? "active"
                      : ""
                  }
                >
                  {name}
                </span>
              ),
            )}
          </div>
          {!stream.scenes.length && (
            <Empty title="Your next page is waiting.">
              <p>Choose a pace and chapter count to start.</p>
            </Empty>
          )}
          {stream.scenes.map((scene) => (
            <article className="scene" key={scene.key}>
              <div className="section-heading">
                <h3>
                  Chapter {scene.chapter || "—"} · Scene {scene.number}
                </h3>
                <small>
                  {scene.complete ? `${words(scene.text)} words` : "Writing…"}
                  {scene.score !== undefined &&
                    ` · Score ${scene.score.toFixed(2)}`}
                </small>
              </div>
              <Prose text={scene.text} />
            </article>
          ))}
        </section>
      </div>
      {!!options.length && (
        <section className="panel">
          <h2>Possible paths</h2>
          {options.map((o, i) => (
            <article className="scene" key={i}>
              <h3>{o.title}</h3>
              <p>{o.summary}</p>
              <button onClick={() => onBranch(o.summary)}>
                Use in manual studio
              </button>
            </article>
          ))}
        </section>
      )}
      {report && (
        <section className="panel">
          <h2>Continuity report</h2>
          <ErrorNotice error={continuity.error} retry={continuity.reload} />
          <JsonView value={continuity.data} />
        </section>
      )}
    </>
  );
}
