import { useCallback, useEffect, useReducer, useRef } from "react";
import type { StreamEvent } from "./types";
export interface Scene {
  key: string;
  chapter: number;
  number: number;
  text: string;
  complete: boolean;
  score?: number;
}
export interface StreamState {
  active: boolean;
  status: string;
  error: string;
  agent: string;
  chapter: number;
  scenes: Scene[];
  draft: string;
  logs: StreamEvent[];
  revision: number;
}
export const initialStream: StreamState = {
  active: false,
  status: "Ready",
  error: "",
  agent: "",
  chapter: 0,
  scenes: [],
  draft: "",
  logs: [],
  revision: 0,
};
type Action =
  | { type: "start" }
  | { type: "connection"; status: string }
  | { type: "event"; event: StreamEvent }
  | { type: "clearLogs" };
export function streamReducer(state: StreamState, action: Action): StreamState {
  if (action.type === "clearLogs") return { ...state, logs: [] };
  if (action.type === "start")
    return {
      ...initialStream,
      active: true,
      status: "Starting…",
      logs: state.logs,
      revision: state.revision,
    };
  if (action.type === "connection") return { ...state, status: action.status };
  const event = action.event;
  if (event.type === "heartbeat") return state;
  const payload = event.payload ?? event.data ?? {};
  const data = typeof payload === "string" ? {} : payload;
  const content =
    typeof payload === "string"
      ? payload
      : String(data.content || data.step || "");
  const next = { ...state, logs: [...state.logs, event].slice(-400) };
  if (event.type === "status" || event.type === "combine_status")
    next.status = content;
  if (event.type === "quota")
    next.status =
      data.phase === "waiting"
        ? `Waiting for ${data.reason}: ${Math.ceil(Number(data.wait_seconds || 0))}s`
        : data.phase === "backup"
          ? `Using backup: ${data.model}`
          : state.status;
  if (event.type === "agent_active") next.agent = String(data.agent || "");
  if (event.type === "chapter_start") next.chapter = Number(data.chapter || 0);
  if (event.type === "scene_start") {
    const number = Number(data.scene || data.scene_number || 1),
      key = `${next.chapter}:${number}`;
    if (!next.scenes.some((s) => s.key === key))
      next.scenes = [
        ...next.scenes,
        { key, chapter: next.chapter, number, text: "", complete: false },
      ];
    next.draft = "";
  }
  if (event.type === "token" || event.type === "stream_token") {
    const token =
      typeof payload === "string"
        ? payload
        : String(data.token || data.content || "");
    next.draft += token;
    next.scenes = next.scenes.map((s, i) =>
      i === next.scenes.length - 1 ? { ...s, text: s.text + token } : s,
    );
    // Token contents live in the draft; do not duplicate every token in the log.
    next.logs = state.logs;
  }
  if (event.type === "scene_complete" || event.type === "scene_written") {
    const number = Number(data.scene || data.scene_number || 1),
      key = `${next.chapter}:${number}`;
    next.scenes = next.scenes.map((s) =>
      s.key === key
        ? {
            ...s,
            text: String(data.text || s.text),
            complete: event.type === "scene_complete" || s.complete,
            score: typeof data.score === "number" ? data.score : s.score,
          }
        : s,
    );
    if (typeof data.text === "string") next.draft = data.text;
  }
  if (
    [
      "done",
      "error",
      "manual_scene_done",
      "combine_done",
      "combine_error",
    ].includes(event.type)
  ) {
    next.active = false;
    next.revision++;
    next.error = String(
      data.error ||
        (event.type.endsWith("error") ? content || "Generation failed." : ""),
    );
    next.status = next.error
      ? "Needs attention"
      : data.status === "cancelled"
        ? "Stopped"
        : "Complete";
  }
  if (event.type === "chapter_complete") next.revision++;
  if (event.type === "needs_review")
    next.status = String(data.reason || "This scene needs review.");
  return next;
}
export function useStream(url: string) {
  const [state, dispatch] = useReducer(streamReducer, initialStream);
  const source = useRef<EventSource | null>(null);
  const close = useCallback(() => {
    source.current?.close();
    source.current = null;
  }, []);
  const start = useCallback(() => {
    close();
    dispatch({ type: "start" });
    const stream = new EventSource(url);
    source.current = stream;
    // EventSource automatically reconnects with Last-Event-ID. Avoid duplicate
    // reducer updates if a proxy replays a message at the reconnect boundary.
    const seen = new Set<string>();
    stream.onopen = () =>
      dispatch({ type: "connection", status: "Connected to live progress" });
    stream.onmessage = (e) => {
      if (source.current !== stream) return;
      if (e.lastEventId && seen.has(e.lastEventId)) return;
      try {
        const event = JSON.parse(e.data) as StreamEvent;
        if (e.lastEventId) {
          seen.add(e.lastEventId);
          if (seen.size > 2000) seen.delete(seen.values().next().value!);
        }
        dispatch({ type: "event", event });
        if (
          [
            "done",
            "error",
            "manual_scene_done",
            "combine_done",
            "combine_error",
          ].includes(event.type)
        )
          close();
      } catch {
        dispatch({
          type: "connection",
          status: "An unreadable progress update was skipped.",
        });
      }
    };
    stream.onerror = () => {
      if (source.current !== stream) return;
      dispatch({
        type: "connection",
        status: "Reconnecting to live progress…",
      });
      // Keep cancellation available: losing transport does not stop the worker.
    };
  }, [url, close]);
  useEffect(() => close, [close, url]);
  return {
    ...state,
    start,
    close,
    clearLogs: () => dispatch({ type: "clearLogs" }),
  };
}
export type StreamController = ReturnType<typeof useStream>;
