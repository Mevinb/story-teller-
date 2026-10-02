import { useState } from "react";
import type { Project, View } from "../types";
import { useResource } from "../hooks";
import { Empty, ErrorNotice, Loading, PageTitle } from "../components/ui";
export function Library({
  open,
  create,
}: {
  open: (project: string, view: View) => void;
  create: () => void;
}) {
  const resource = useResource<{ projects: Project[] }>("/api/projects");
  const [search, setSearch] = useState(""),
    [genre, setGenre] = useState("");
  const projects = resource.data?.projects || [];
  const filtered = projects.filter(
    (p) =>
      (!genre || p.genre === genre) &&
      `${p.title} ${p.premise}`.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <>
      <PageTitle eyebrow="Your collection" title="Every story starts here.">
        <button className="primary" onClick={create}>
          + New story
        </button>
      </PageTitle>
      <div className="stats">
        <span>
          <strong>{projects.length}</strong> stories
        </span>
        <span>
          <strong>{projects.reduce((n, p) => n + p.total_chapters, 0)}</strong>{" "}
          chapters
        </span>
        <span>
          <strong>
            {projects.reduce((n, p) => n + p.word_count, 0).toLocaleString()}
          </strong>{" "}
          words written
        </span>
      </div>
      <div className="toolbar">
        <input
          aria-label="Search stories"
          type="search"
          placeholder="Find a story, an idea, a world…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          aria-label="Filter genre"
          value={genre}
          onChange={(e) => setGenre(e.target.value)}
        >
          <option value="">All genres</option>
          {[...new Set(projects.map((p) => p.genre))].sort().map((g) => (
            <option key={g}>{g}</option>
          ))}
        </select>
        <button onClick={resource.reload}>Refresh</button>
      </div>
      <ErrorNotice error={resource.error} retry={resource.reload} />
      {resource.loading && <Loading />}
      {!resource.loading && !resource.error && !filtered.length && (
        <Empty
          title={
            projects.length
              ? "No matching stories"
              : "Make room for your first story."
          }
        >
          <p>
            {projects.length
              ? "Try another search or genre."
              : "Start with a premise, a character, or a place you can’t stop thinking about."}
          </p>
          <button onClick={create}>Create a story</button>
        </Empty>
      )}
      <div className="library-grid">
        {filtered.map((p, i) => (
          <article className="story-card" key={p.name}>
            <div className="story-heading">
              <span className="eyebrow">{p.genre || "Fiction"}</span>
              <span className="story-number">
                {String(i + 1).padStart(2, "0")}
              </span>
            </div>
            <h2>{p.title}</h2>
            <p className="story-premise">
              {p.premise || "An unwritten world. Add your premise to begin."}
            </p>
            <div className="story-meta">
              {p.total_chapters} chapters <span>·</span>{" "}
              {p.word_count.toLocaleString()} words
            </div>
            <div className="actions">
              <button
                className="primary"
                onClick={() => open(p.name, "generate")}
              >
                Open studio ↗
              </button>
              <button onClick={() => open(p.name, "reader")}>Read</button>
              <button onClick={() => open(p.name, "edit")}>Details</button>
            </div>
          </article>
        ))}
      </div>
    </>
  );
}
