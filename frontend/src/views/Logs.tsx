import { useState } from "react";
import type { StreamController } from "../stream";
import { useResource } from "../hooks";
import { ErrorNotice, JsonView, PageTitle } from "../components/ui";
export function Logs({
  generation,
  combine,
}: {
  generation: StreamController;
  combine: StreamController;
}) {
  const [filter, setFilter] = useState("");
  const jobs = useResource<{ jobs: unknown[] }>("/api/jobs");
  const logs = [...generation.logs, ...combine.logs]
    .sort((a, b) => (a.timestamp || "").localeCompare(b.timestamp || ""))
    .filter((e) =>
      JSON.stringify(e).toLowerCase().includes(filter.toLowerCase()),
    );
  return (
    <>
      <PageTitle eyebrow="Activity" title="Pipeline logs">
        <button
          onClick={() => {
            generation.clearLogs();
            combine.clearLogs();
          }}
        >
          Clear visible logs
        </button>
      </PageTitle>
      <div className="toolbar">
        <input
          aria-label="Filter logs"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter by agent, status, or message…"
        />
      </div>
      <section className="terminal" aria-label="Pipeline events">
        {logs.length ? (
          logs.map((e, i) => (
            <div className="log-row" key={i}>
              <time>{e.timestamp?.slice(11, 19)}</time>
              <strong>{e.type}</strong>
              <pre>
                {typeof e.payload === "string"
                  ? e.payload
                  : JSON.stringify(e.payload ?? e.data)}
              </pre>
            </div>
          ))
        ) : (
          <p>No events received in this session.</p>
        )}
      </section>
      <details className="panel">
        <summary>Recoverable jobs</summary>
        <button onClick={jobs.reload}>Refresh jobs</button>
        <ErrorNotice error={jobs.error} />
        <JsonView value={jobs.data?.jobs || []} />
      </details>
    </>
  );
}
