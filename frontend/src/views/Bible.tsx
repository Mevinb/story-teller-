import { useEffect, useState } from "react";
import { projectUrl } from "../api";
import type { StoryState } from "../types";
import { useResource } from "../hooks";
import { ErrorNotice, JsonView, Loading, PageTitle } from "../components/ui";
export function Bible({
  project,
  active,
}: {
  project: string;
  active: boolean;
}) {
  const resource = useResource<{ state: StoryState }>(
    projectUrl(project, "/state"),
  );
  useEffect(() => {
    if (active) resource.reload();
  }, [active, resource.reload]);
  const [tab, setTab] = useState("characters");
  const state = resource.data?.state;
  const sections: Record<string, unknown> = {
    characters: state?.characters,
    plot: state?.plot,
    world: { metadata: state?.metadata, world: state?.world },
    full: state,
  };
  return (
    <>
      <PageTitle eyebrow="Memory & continuity" title="The story bible">
        <button onClick={resource.reload}>Refresh memory</button>
      </PageTitle>
      <ErrorNotice error={resource.error} retry={resource.reload} />
      {resource.loading && <Loading />}
      <div className="segmented">
        {["characters", "plot", "world", "full"].map((t) => (
          <button key={t} aria-pressed={tab === t} onClick={() => setTab(t)}>
            {t === "full" ? "Full state" : t}
          </button>
        ))}
      </div>
      {tab === "characters" ? (
        <div className="library-grid">
          {Object.entries(state?.characters || {}).map(([name, c]) => (
            <article className="panel" key={name}>
              <h2>{name}</h2>
              <p>{c.description}</p>
              <div className="actions">
                {c.traits?.map((t, i) => (
                  <span className="tag" key={i}>
                    {t}
                  </span>
                ))}
              </div>
              <details>
                <summary>Character memory</summary>
                <JsonView value={c} />
              </details>
            </article>
          ))}
        </div>
      ) : (
        <section className="panel">
          <JsonView value={sections[tab] || {}} />
        </section>
      )}
      <a className="button" href={projectUrl(project, "/export")}>
        Download full manuscript
      </a>
    </>
  );
}
