import { Component, useEffect, useState, type ReactNode } from "react";
import type { Activity, Catalog, View } from "./types";
import { projectUrl } from "./api";
import { useResource } from "./hooks";
import { useStream } from "./stream";
import { ErrorNotice } from "./components/ui";
import { ModelPicker, modelName } from "./components/Models";
import { Library } from "./views/Library";
import { ProjectForm } from "./views/ProjectForm";
import { Premise } from "./views/Premise";
import { Generation } from "./views/Generation";
import { Manual } from "./views/Manual";
import { Reader } from "./views/Reader";
import { Combine } from "./views/Combine";
import { Bible } from "./views/Bible";
import { Logs } from "./views/Logs";
import { Vision } from "./views/Vision";
import { Settings } from "./views/Settings";

const storyViews: { id: View; label: string }[] = [
  { id: "premise", label: "Premise" },
  { id: "generate", label: "Auto studio" },
  { id: "manual", label: "Manual studio" },
  { id: "reader", label: "Reader" },
  { id: "combine", label: "Polish" },
  { id: "bible", label: "Story bible" },
  { id: "logs", label: "Activity" },
  { id: "edit", label: "Story details" },
];
const allViews = [
  "library",
  "create",
  "vision",
  "settings",
  ...storyViews.map((v) => v.id),
];
function readLocation(): { view: View; project: string } {
  const params = new URLSearchParams(location.hash.slice(1));
  const project = params.get("project") || "";
  let view = params.get("view") || "library";
  if (
    !allViews.includes(view) ||
    (!project && storyViews.some((v) => v.id === view))
  )
    view = "library";
  return { view: view as View, project };
}
export class ErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? (
      <main className="app-shell">
        <h1>The studio couldn’t display this page.</h1>
        <p>Your stories are stored on the server.</p>
        <a className="button" href="/">
          Reload studio
        </a>{" "}
        <a className="button" href="/classic">
          Open classic interface
        </a>
      </main>
    ) : (
      this.props.children
    );
  }
}
export default function App() {
  const [route, setRoute] = useState(readLocation);
  const [picker, setPicker] = useState(false),
    [createVersion, setCreateVersion] = useState(0);
  const catalog = useResource<Catalog>("/api/models");
  const [visited, setVisited] = useState<Set<View>>(new Set([route.view]));
  useEffect(() => {
    const change = () => setRoute(readLocation());
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  useEffect(() => {
    setVisited((old) => new Set([...old, route.view]));
    document.title = `${route.project ? `${route.project} · ` : ""}Story Teller`;
  }, [route]);
  const navigate = (view: View, project = route.project) => {
    location.hash = new URLSearchParams({
      view,
      ...(project ? { project } : {}),
    }).toString();
  };
  const mounted = (view: View) => visited.has(view) || route.view === view;
  return (
    <>
      <a
        className="skip-link"
        href="#workspace"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById("workspace")?.focus();
        }}
      >
        Skip to workspace
      </a>
      <div className="app-shell">
        <header className="app-header">
          <button
            className="wordmark"
            aria-label="Story Teller library"
            onClick={() => navigate("library")}
          >
            <span className="brand-mark">S.</span>
            <span>
              Story Teller<small>A place for the unwritten.</small>
            </span>
          </button>
          <div className="header-actions">
            <button
              className="model-trigger"
              onClick={() => {
                catalog.reload();
                setPicker(true);
              }}
            >
              <span className="status-dot" />
              <span>{modelName(catalog.data?.active)}</span>
              <span aria-hidden="true">⌄</span>
            </button>
            <button className="primary" onClick={() => navigate("create")}>
              + New story
            </button>
          </div>
        </header>
        <nav className="primary-nav" aria-label="Main navigation">
          {(
            [
              { id: "library", label: "Story library" },
              { id: "vision", label: "Visual references" },
              { id: "settings", label: "Settings" },
            ] as const
          ).map((v) => (
            <button
              key={v.id}
              aria-current={route.view === v.id ? "page" : undefined}
              onClick={() => navigate(v.id)}
            >
              {v.label}
            </button>
          ))}
          <span className="nav-note">Local workspace</span>
        </nav>
        {route.project && (
          <div className="project-bar">
            <button className="project-name" onClick={() => navigate("edit")}>
              {route.project.replaceAll("_", " ")}
            </button>
            <nav aria-label="Story workspace">
              {storyViews.map((v) => (
                <button
                  key={v.id}
                  aria-current={route.view === v.id ? "page" : undefined}
                  onClick={() => navigate(v.id)}
                >
                  {v.label}
                </button>
              ))}
            </nav>
          </div>
        )}
        <main id="workspace" tabIndex={-1}>
          <ErrorNotice error={catalog.error} retry={catalog.reload} />
          {route.view === "library" && (
            <Library
              open={(p, v) => navigate(v, p)}
              create={() => navigate("create")}
            />
          )}
          {mounted("create") && (
            <div hidden={route.view !== "create"}>
              <ProjectForm
                key={createVersion}
                onSaved={(p) => {
                  setCreateVersion((n) => n + 1);
                  navigate("premise", p);
                }}
                onDeleted={() => {}}
              />
            </div>
          )}
          {mounted("vision") && (
            <div hidden={route.view !== "vision"}>
              <Vision />
            </div>
          )}
          {mounted("settings") && (
            <div hidden={route.view !== "settings"}>
              <Settings catalog={catalog.data} onChanged={catalog.reload} />
            </div>
          )}
          {route.project && (
            <Workspace
              key={route.project}
              project={route.project}
              view={route.view}
              catalog={catalog.data}
              navigate={navigate}
            />
          )}
        </main>
        <footer className="app-footer">
          <span>
            Story Teller <span className="muted">/ Writing studio</span>
          </span>
          <a href="/classic">Classic interface</a>
        </footer>
      </div>
      {picker && (
        <ModelPicker
          catalog={catalog.data}
          onClose={() => setPicker(false)}
          onChanged={catalog.reload}
        />
      )}
    </>
  );
}
function Workspace({
  project,
  view,
  catalog,
  navigate,
}: {
  project: string;
  view: View;
  catalog?: Catalog;
  navigate: (view: View, project?: string) => void;
}) {
  const generation = useStream(projectUrl(project, "/generate/stream"));
  const combine = useStream(projectUrl(project, "/combine/stream"));
  const activity = useResource<Activity>(projectUrl(project, "/activity"));
  const [visited, setVisited] = useState(new Set<View>([view]));
  const [brief, setBrief] = useState("");
  useEffect(() => {
    setVisited((old) => new Set([...old, view]));
  }, [view]);
  useEffect(() => {
    if (activity.data?.generation) generation.start();
    if (activity.data?.combine) combine.start();
  }, [activity.data, generation.start, combine.start]);
  const views: Partial<Record<View, ReactNode>> = {
    premise: <Premise project={project} active={view === "premise"} />,
    generate: (
      <Generation
        project={project}
        stream={generation}
        onBranch={(text) => {
          setBrief(text);
          navigate("manual");
        }}
      />
    ),
    manual: (
      <Manual
        project={project}
        stream={generation}
        suggestedBrief={brief}
        active={view === "manual"}
      />
    ),
    reader: (
      <Reader
        project={project}
        active={view === "reader"}
        onResume={() => navigate("manual")}
      />
    ),
    combine: <Combine project={project} catalog={catalog} stream={combine} />,
    bible: <Bible project={project} active={view === "bible"} />,
    logs: <Logs generation={generation} combine={combine} />,
    edit: (
      <ProjectForm
        project={project}
        active={view === "edit"}
        onSaved={() => {}}
        onDeleted={() => navigate("library", "")}
      />
    ),
  };
  return (
    <>
      {(generation.active || combine.active) && (
        <div className="live-banner" role="status">
          <span className="status-dot busy" />
          {generation.active ? generation.status : combine.status}
          <button
            onClick={() =>
              navigate(
                generation.active
                  ? activity.data?.manual
                    ? "manual"
                    : "generate"
                  : "combine",
              )
            }
          >
            View progress
          </button>
        </div>
      )}
      <ErrorNotice error={activity.error} retry={activity.reload} />
      {storyViews
        .filter((v) => visited.has(v.id) || view === v.id)
        .map((v) => (
          <div key={v.id} hidden={view !== v.id}>
            {views[v.id]}
          </div>
        ))}
    </>
  );
}
