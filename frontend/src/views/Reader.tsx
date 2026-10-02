import { useEffect, useState } from "react";
import { projectUrl, send } from "../api";
import type { Chapter, ChapterText } from "../types";
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
export function Reader({
  project,
  onResume,
  active,
}: {
  project: string;
  onResume: () => void;
  active: boolean;
}) {
  const list = useResource<{ chapters: Chapter[] }>(
    projectUrl(project, "/chapters"),
  );
  const [number, setNumber] = useState<number>(),
    [editing, setEditing] = useState(false),
    [draft, setDraft] = useState("");
  const [size, setSize] = useState(19),
    [theme, setTheme] = useState("paper"),
    [font, setFont] = useState("serif"),
    [focus, setFocus] = useState(false),
    [deleteFrom, setDeleteFrom] = useState(1);
  const chapter = useResource<ChapterText>(
    number ? projectUrl(project, `/chapter/${number}`) : null,
  );
  const action = useAction(),
    ui = useUi();
  useEffect(() => {
    if (list.data)
      setNumber((n) =>
        list.data!.chapters.some((c) => c.number === n)
          ? n
          : list.data!.chapters[0]?.number,
      );
  }, [list.data]);
  useEffect(() => {
    if (chapter.data) setDraft(chapter.data.content);
  }, [chapter.data]);
  useEffect(() => {
    const escape = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFocus(false);
    };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, []);
  useEffect(() => {
    if (active && !editing) {
      list.reload();
      chapter.reload();
    }
  }, [active, editing, list.reload, chapter.reload]);
  return (
    <>
      <PageTitle eyebrow="The manuscript" title="Read with fresh eyes.">
        <button onClick={list.reload}>Refresh chapters</button>
      </PageTitle>
      <ErrorNotice error={list.error} retry={list.reload} />
      {list.loading && <Loading />}
      {!list.loading && list.data && !list.data.chapters.length ? (
        <Empty title="No chapters yet.">
          <p>
            Write a chapter in the automatic or manual studio, then return here.
          </p>
        </Empty>
      ) : (
        <div className="reader-layout">
          <aside className="chapter-list">
            <h2>Chapters</h2>
            {list.data?.chapters.map((c) => (
              <button
                className={number === c.number ? "active" : ""}
                key={c.number}
                aria-current={number === c.number ? "page" : undefined}
                onClick={async () => {
                  if (
                    editing &&
                    !(await ui.confirm("Discard your unsaved chapter edits?"))
                  )
                    return;
                  setEditing(false);
                  setNumber(c.number);
                }}
              >
                <strong>{c.title}</strong>
                <small>
                  {c.words.toLocaleString()} words{" "}
                  {c.status === "writing" && "· Draft"}
                </small>
              </button>
            ))}
            <details className="maintenance">
              <summary>Manage chapters</summary>
              <Field label="Delete from chapter">
                <input
                  type="number"
                  min={1}
                  value={deleteFrom}
                  onChange={(e) => setDeleteFrom(Number(e.target.value))}
                />
              </Field>
              <button
                className="danger"
                disabled={action.busy || deleteFrom < 1}
                onClick={() =>
                  action.run(async () => {
                    if (
                      await ui.confirm(
                        `Delete chapter ${deleteFrom} and every chapter after it?`,
                      )
                    ) {
                      await send(projectUrl(project, "/chapters/delete"), {
                        from_chapter: deleteFrom,
                      });
                      setNumber(undefined);
                      setEditing(false);
                      list.reload();
                    }
                  })
                }
              >
                Delete chapters
              </button>
            </details>
          </aside>
          <section
            className={`reader ${focus ? "reader-focus" : ""} theme-${theme}`}
          >
            <div className="reader-toolbar">
              <h2>{chapter.data?.title || "Select a chapter"}</h2>
              <div className="actions">
                <button
                  aria-label="Decrease font size"
                  onClick={() => setSize((n) => Math.max(12, n - 1))}
                >
                  A−
                </button>
                <button
                  aria-label="Increase font size"
                  onClick={() => setSize((n) => Math.min(30, n + 1))}
                >
                  A+
                </button>
                <select
                  aria-label="Reader theme"
                  value={theme}
                  onChange={(e) => setTheme(e.target.value)}
                >
                  <option value="paper">Paper</option>
                  <option value="dark">Night</option>
                  <option value="sepia">Sepia</option>
                </select>
                <select
                  aria-label="Reader font"
                  value={font}
                  onChange={(e) => setFont(e.target.value)}
                >
                  <option value="serif">Serif</option>
                  <option value="sans-serif">Sans</option>
                  <option value="monospace">Mono</option>
                </select>
                <button onClick={() => setFocus(!focus)}>
                  {focus ? "Exit focus" : "Focus"}
                </button>
              </div>
            </div>
            <ErrorNotice
              error={chapter.error || action.error}
              retry={chapter.reload}
            />
            {chapter.loading ? (
              <Loading />
            ) : (
              number &&
              chapter.data && (
                <>
                  <div className="actions reader-actions">
                    {chapter.data.status === "writing" ? (
                      <button
                        className="primary"
                        disabled={action.busy}
                        onClick={() =>
                          action.run(async () => {
                            await send(
                              projectUrl(project, `/chapter/${number}/resume`),
                            );
                            onResume();
                          })
                        }
                      >
                        Resume draft
                      </button>
                    ) : editing ? (
                      <>
                        <button
                          className="primary"
                          disabled={action.busy}
                          onClick={() =>
                            action.run(async () => {
                              await send(
                                projectUrl(project, `/chapter/${number}`),
                                { content: draft },
                                "PUT",
                              );
                              setEditing(false);
                              chapter.reload();
                              list.reload();
                              ui.notify("Chapter saved.");
                            })
                          }
                        >
                          Save chapter
                        </button>
                        <button
                          onClick={() => {
                            setEditing(false);
                            setDraft(chapter.data!.content);
                          }}
                        >
                          Cancel edit
                        </button>
                      </>
                    ) : (
                      <button onClick={() => setEditing(true)}>
                        Edit chapter
                      </button>
                    )}
                  </div>
                  {editing ? (
                    <textarea
                      aria-label="Chapter prose"
                      className="chapter-editor"
                      value={draft}
                      onChange={(e) => setDraft(e.target.value)}
                    />
                  ) : (
                    <div
                      className="reader-page"
                      style={{ fontSize: size, fontFamily: font }}
                    >
                      <Prose text={chapter.data.content} />
                    </div>
                  )}
                </>
              )
            )}
          </section>
        </div>
      )}
      {!!number && chapter.data && chapter.data.status !== "writing" && (
        <ImagePrompts
          key={`${number}:${chapter.data.content}`}
          project={project}
          source={`chapter:${number}`}
        />
      )}
    </>
  );
}
