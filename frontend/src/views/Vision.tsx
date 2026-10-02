import { useEffect, useState } from "react";
import { api, send } from "../api";
import { useAction, useResource } from "../hooks";
import {
  ErrorNotice,
  Field,
  JsonView,
  PageTitle,
  Prose,
} from "../components/ui";
export function Vision() {
  const status = useResource<{ data: unknown }>("/api/vision/status");
  const [file, setFile] = useState<File>(),
    [preview, setPreview] = useState(""),
    [mode, setMode] = useState("auto"),
    [extra, setExtra] = useState("");
  const [analysis, setAnalysis] = useState<unknown>(),
    [session, setSession] = useState(""),
    [text, setText] = useState("");
  const [chat, setChat] = useState<{ role: string; text: string }[]>([]);
  const action = useAction();
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  const pick = (next?: File) => {
    setFile(next);
    setAnalysis(undefined);
    setSession("");
    setChat([]);
  };
  return (
    <>
      <PageTitle eyebrow="Visual references" title="A character in focus" />
      <ErrorNotice error={status.error} retry={status.reload} />
      <div className="two-col">
        <section className="panel">
          <h2>Image reference</h2>
          <Field label="Choose an image">
            <input
              disabled={action.busy}
              type="file"
              accept="image/*"
              onChange={(e) => pick(e.target.files?.[0])}
            />
          </Field>
          {preview && (
            <img
              className="vision-preview"
              src={preview}
              alt="Selected character reference"
            />
          )}
          <Field label="Vision backend">
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              {["auto", "local", "cloud"].map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </Field>
          <Field label="Additional instructions">
            <textarea
              value={extra}
              onChange={(e) => setExtra(e.target.value)}
            />
          </Field>
          <button
            className="primary"
            disabled={!file || action.busy}
            onClick={() =>
              action.run(async () => {
                const form = new FormData();
                form.append("image", file!);
                form.append("mode", mode);
                form.append("prompt_extra", extra);
                const result = await api<{ data: unknown }>(
                  "/api/vision/analyze",
                  { method: "POST", body: form },
                );
                setAnalysis(result.data);
                const chatForm = new FormData();
                chatForm.append("image", file!);
                const resultChat = await api<{
                  data: { session_id: string; initial_message: string };
                }>("/api/vision/chat/start", {
                  method: "POST",
                  body: chatForm,
                });
                setSession(resultChat.data.session_id);
                setChat([
                  {
                    role: "assistant",
                    text:
                      resultChat.data.initial_message || "Ask about the image.",
                  },
                ]);
              })
            }
          >
            {action.busy ? "Working…" : "Analyze image"}
          </button>
          <details>
            <summary>Backend availability</summary>
            <JsonView value={status.data?.data} />
          </details>
        </section>
        <section className="panel">
          <h2>Visual notes</h2>
          {analysis !== undefined ? (
            <JsonView value={analysis} />
          ) : (
            <p className="muted">
              Choose an image to describe its visual details.
            </p>
          )}
          {chat.map((entry, i) => (
            <article className={`chat-message ${entry.role}`} key={i}>
              <strong>{entry.role === "user" ? "You" : "Assistant"}</strong>
              <Prose text={entry.text} />
            </article>
          ))}
          {session && (
            <form
              onSubmit={(e) => {
                e.preventDefault();
                action.run(async () => {
                  const result = await send<{ data: { response: string } }>(
                    "/api/vision/chat/message",
                    { session_id: session, message: text },
                  );
                  setChat([
                    ...chat,
                    { role: "user", text },
                    { role: "assistant", text: result.data.response },
                  ]);
                  setText("");
                });
              }}
            >
              <Field label="Ask about the image">
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                />
              </Field>
              <button disabled={action.busy || !text.trim()}>
                Send message
              </button>
            </form>
          )}
          <ErrorNotice error={action.error} />
        </section>
      </div>
    </>
  );
}
