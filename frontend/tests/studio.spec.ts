import { test, expect, type Page } from "@playwright/test";
import { streamReducer, initialStream } from "../src/stream";

const state = {
  metadata: {
    title: "The Quiet Atlas",
    genre: "Mystery",
    premise: "1. Find the map.\n\n2. Follow the coast.",
    setting: "An island archive",
    themes: ["memory"],
  },
  characters: {
    Mira: {
      description: "A mapmaker",
      traits: ["patient"],
      status: "active",
      knowledge: ["The map is incomplete"],
    },
  },
};
const project = {
  name: "atlas",
  title: "The Quiet Atlas",
  genre: "Mystery",
  premise: "A cartographer discovers a coastline that moves.",
  total_chapters: 2,
  word_count: 8400,
  characters: ["Mira"],
};
const catalog = {
  active: "groq:qwen/test",
  backend_mode: "groq",
  groq_models: [{ id: "qwen/test", name: "Qwen writer" }],
  gemini_models: [{ id: "gemini-test", name: "Gemini editor" }],
  openrouter_models: [],
  local_models: [],
  gemini_model: "gemini-test",
};
async function fixtures(page: Page, overrides: Record<string, unknown> = {}) {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    const data: Record<string, unknown> = {
      "/api/models": catalog,
      "/api/projects": { projects: [project] },
      "/api/project/atlas": { state },
      "/api/project/atlas/state": { state },
      "/api/project/atlas/activity": {
        generation: false,
        combine: false,
        manual: false,
      },
      "/api/project/atlas/chapters": {
        chapters: [
          { number: 1, title: "Chapter 1", words: 4200, status: "completed" },
          { number: 2, title: "Chapter 2", words: 4200, status: "completed" },
        ],
      },
      "/api/project/atlas/chapter/1": {
        chapter: 1,
        title: "Chapter 1",
        content: "The first map lay open.",
      },
      "/api/project/atlas/chapter/2": {
        chapter: 2,
        title: "Chapter 2",
        content: "The coastline shifted in the dark.",
      },
      "/api/project/atlas/image-prompts": {
        status: "ready",
        highlights: [
          {
            title: "The archive",
            context: "Mira finds the map.",
            references: ["Reference 1: Mira"],
            prompt: "Mira examines a map under warm lamplight.",
          },
        ],
      },
      "/api/project/atlas/generate/manual/status": { active: false },
      "/api/project/atlas/combine/versions": {
        versions: [
          { suffix: "latest", label: "Latest (Active)", timestamp: 1 },
        ],
      },
      "/api/project/atlas/combine/version/latest": {
        revised: "A polished story.",
        analysis: "The threads now connect.",
      },
      "/api/project/atlas/continuity": {
        threads: ["Find the original map"],
        seeds: [],
      },
      "/api/jobs": { jobs: [] },
      "/api/vision/status": {
        success: true,
        data: { local: false, cloud: true },
      },
      "/api/settings": {
        settings: {
          BACKEND_MODE: "groq",
          GROQ_API_KEY: "********",
          GROQ_MODEL: "qwen/test",
          WORDS_PER_SCENE_MIN: 500,
        },
      },
      "/api/keys/status": {},
      ...overrides,
    };
    await route.fulfill({
      status: 200,
      json: data[url.pathname] || { status: "ok" },
    });
  });
}
async function fakeStreams(page: Page) {
  await page.addInitScript(() => {
    class FakeSource {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSED = 2;
      readyState = 1;
      onmessage: ((e: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      onopen: (() => void) | null = null;
      url: string;
      constructor(url: string) {
        this.url = url;
        sources.push(this);
        setTimeout(() => this.onopen?.(), 0);
      }
      close() {
        this.readyState = 2;
      }
    }
    const sources: FakeSource[] = [];
    Object.assign(window, {
      EventSource: FakeSource,
      emitStoryEvent: (type: string, payload: unknown, id: string) =>
        sources
          .filter((s) => s.readyState !== 2)
          .forEach((s) =>
            s.onmessage?.({
              data: JSON.stringify({ type, payload }),
              lastEventId: id,
            } as MessageEvent),
          ),
      disconnectStory: () =>
        sources.forEach((s) => {
          s.readyState = 0;
          s.onerror?.();
        }),
      storySources: () => sources.length,
    });
  });
}
async function emit(page: Page, type: string, payload: unknown, id: string) {
  await page.evaluate(
    ({ type, payload, id }) =>
      (
        window as unknown as {
          emitStoryEvent: (type: string, payload: unknown, id: string) => void;
        }
      ).emitStoryEvent(type, payload, id),
    { type, payload, id },
  );
}

test("real Flask: create story, preserve character metadata, and retain stories after reload", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "The Lantern Archive" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "+ New story", exact: true })
    .first()
    .click();
  await page
    .getByRole("textbox", { name: "Story title", exact: true })
    .fill("Browser journey");
  await page
    .getByLabel("Premise", { exact: true })
    .fill("A courier brings a letter to an empty city.");
  await page.getByLabel("Character name").fill("Iona");
  await page.getByLabel("Character description").fill("A courier");
  await page
    .getByLabel("Traits, separated by commas", { exact: true })
    .fill("persistent, curious");
  await page
    .getByRole("button", { name: "Add character", exact: true })
    .click();
  await page.getByRole("button", { name: "Create story", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Shape the story" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Story details", exact: true })
    .click();
  await expect(
    page.getByRole("textbox", { name: "Story title", exact: true }),
  ).toHaveValue("Browser journey");
  await page
    .getByRole("textbox", { name: "Setting", exact: true })
    .fill("An abandoned coastal city");
  await page.getByRole("button", { name: "Save details", exact: true }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "Story details saved." }),
  ).toBeVisible();
  await page.reload();
  await expect(
    page.getByRole("textbox", { name: "Setting", exact: true }),
  ).toHaveValue("An abandoned coastal city");
});

test("real Flask serves compiled assets and chapters on the same origin", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/#view=reader&project=browser_fixture");
  await expect(
    page.getByRole("heading", { name: "Read with fresh eyes." }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Chapter 2", exact: false }).click();
  await expect(page.locator(".reader-page")).toContainText("Chapter 2");
  await expect(page.locator('script[type="module"]')).toHaveAttribute(
    "src",
    /\/static\/app\/assets\//,
  );
  expect(errors).toEqual([]);
});

test("library filters and mobile layout have no horizontal overflow", async ({
  page,
}) => {
  await fixtures(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByLabel("Search stories").fill("missing");
  await expect(
    page.getByRole("heading", { name: "No matching stories" }),
  ).toBeVisible();
  await page.getByLabel("Search stories").fill("atlas");
  await expect(
    page.getByRole("heading", { name: "The Quiet Atlas" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "test-results/studio-mobile.png",
    fullPage: true,
  });
});

test("reader saves selected chapter and renders prompts as text", async ({
  page,
}) => {
  await fixtures(page);
  let saved: unknown;
  await page.route("**/api/project/atlas/chapter/2", async (route) => {
    if (route.request().method() === "PUT") {
      saved = route.request().postDataJSON();
      await route.fulfill({ json: { status: "ok" } });
    } else
      await route.fulfill({
        json: {
          chapter: 2,
          title: "Chapter 2",
          content: "The coastline shifted in the dark.",
        },
      });
  });
  await page.goto("/#view=reader&project=atlas");
  await page.getByRole("button", { name: "Chapter 2", exact: false }).click();
  await expect(page.locator(".reader-page")).toContainText(
    "The coastline shifted",
  );
  await page.getByRole("button", { name: "Edit chapter", exact: true }).click();
  await page.getByLabel("Chapter prose").fill("A revised coastline.");
  await page.getByRole("button", { name: "Save chapter", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Chapter 2", exact: false }),
  ).toHaveAttribute("aria-current", "page");
  expect(saved).toEqual({ content: "A revised coastline." });
  await page.getByText("Scene image prompts", { exact: false }).click();
  await expect(
    page.getByText("Mira examines a map under warm lamplight."),
  ).toBeVisible();
});

test("stream reconnect preserves cancellation, deduplicates tokens, and survives tab changes", async ({
  page,
}) => {
  await fixtures(page);
  await fakeStreams(page);
  await page.goto("/#view=generate&project=atlas");
  await page.getByRole("button", { name: "Write chapters" }).click();
  await expect(
    page.getByRole("button", { name: "Stop generation" }),
  ).toBeVisible();
  await emit(page, "chapter_start", { chapter: 3 }, "run:1");
  await emit(page, "scene_start", { scene: 1 }, "run:2");
  await emit(page, "token", "A new shore.", "run:3");
  await emit(page, "token", "A new shore.", "run:3");
  await expect(page.locator(".scene .prose")).toHaveText("A new shore.");
  await page.evaluate(() =>
    (window as unknown as { disconnectStory: () => void }).disconnectStory(),
  );
  await expect(
    page.getByRole("button", { name: "Stop generation" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Story bible", exact: true }).click();
  await emit(page, "done", { status: "ok" }, "run:4");
  await page.getByRole("button", { name: "Auto studio", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Write chapters" }),
  ).toBeVisible();
  await expect(page.locator(".scene .prose")).toHaveText("A new shore.");
});

test("active work reconnects after a page load without starting a generation", async ({
  page,
}) => {
  await fixtures(page, {
    "/api/project/atlas/activity": {
      generation: true,
      combine: false,
      manual: false,
    },
  });
  await fakeStreams(page);
  let starts = 0;
  await page.route("**/api/project/atlas/generate", async (route) => {
    starts++;
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.goto("/#view=generate&project=atlas");
  await expect(
    page.getByRole("button", { name: "Stop generation" }),
  ).toBeVisible();
  expect(starts).toBe(0);
});

test("manual typed scenes and edits use their exact API payloads", async ({
  page,
}) => {
  await fixtures(page, {
    "/api/project/atlas/generate/manual/status": {
      active: true,
      chapter_num: 3,
      chapter_title: "A new shore",
      scenes_completed: 1,
      completed_scenes: [{ text: "The boat arrived." }],
      is_generating: false,
    },
  });
  let typed: unknown, edited: unknown;
  await page.route(
    "**/api/project/atlas/generate/manual/scene/typed",
    async (route) => {
      typed = route.request().postDataJSON();
      await route.fulfill({ json: { status: "ok" } });
    },
  );
  await page.route(
    "**/api/project/atlas/generate/manual/scene/0",
    async (route) => {
      edited = route.request().postDataJSON();
      await route.fulfill({ json: { status: "ok" } });
    },
  );
  await page.goto("/#view=manual&project=atlas");
  await page.getByLabel("Or write your own scene").fill("Mira stepped ashore.");
  await page.getByRole("button", { name: "Add written scene" }).click();
  await expect(page.getByLabel("Or write your own scene")).toHaveValue("");
  expect(typed).toEqual({ text: "Mira stepped ashore." });
  await page.getByRole("button", { name: "Edit scene 1" }).click();
  await page.getByLabel("Scene prose").fill("The boat arrived at dawn.");
  await page.getByRole("button", { name: "Save scene", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  expect(edited).toEqual({ text: "The boat arrived at dawn." });
});

test("settings submit only edited fields and keep masked keys out of updates", async ({
  page,
}) => {
  await fixtures(page);
  let submitted: unknown;
  await page.route("**/api/settings", async (route) => {
    if (route.request().method() === "PUT") {
      submitted = route.request().postDataJSON();
      await route.fulfill({ json: { status: "ok" } });
    } else
      await route.fulfill({
        json: {
          settings: {
            BACKEND_MODE: "groq",
            GROQ_API_KEY: "********",
            GROQ_MODEL: "qwen/test",
            WORDS_PER_SCENE_MIN: 500,
          },
        },
      });
  });
  await page.goto("/#view=settings");
  await page.getByLabel("Target scene words · minimum").fill("650");
  await page.getByRole("button", { name: "Save settings" }).click();
  await expect(
    page.getByRole("button", { name: "Save settings" }),
  ).toBeDisabled();
  expect(submitted).toEqual({ settings: { WORDS_PER_SCENE_MIN: 650 } });
});

test("model switch errors stay visible in an accessible dialog", async ({
  page,
}) => {
  await fixtures(page);
  await page.route("**/api/models/switch", (route) =>
    route.fulfill({
      status: 503,
      json: { error: "Provider is temporarily unavailable." },
    }),
  );
  await page.goto("/");
  await page.locator(".model-trigger").click();
  await page.getByRole("button", { name: /Qwen writer/ }).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.getByRole("alert")).toHaveText(
    "Provider is temporarily unavailable.",
  );
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
});

test("project edit retains extended character memory", async ({ page }) => {
  await fixtures(page);
  let saved: { characters?: typeof state.characters } = {};
  await page.route("**/api/project/atlas/state", async (route) => {
    saved = route.request().postDataJSON();
    await route.fulfill({ json: { status: "ok" } });
  });
  await page.goto("/#view=edit&project=atlas");
  await page
    .getByRole("textbox", { name: "Story title", exact: true })
    .fill("The Quiet Atlas revised");
  await page.getByRole("button", { name: "Save details" }).click();
  await expect(
    page.getByRole("status").filter({ hasText: "Story details saved." }),
  ).toBeVisible();
  expect(saved.characters?.Mira.knowledge).toEqual(["The map is incomplete"]);
});

test("all workspaces render and desktop screenshot is available", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await fixtures(page);
  await page.goto("/#view=generate&project=atlas");
  for (const name of [
    "Premise",
    "Manual studio",
    "Reader",
    "Polish",
    "Story bible",
    "Activity",
    "Story details",
    "Visual references",
    "Settings",
    "Story library",
  ]) {
    await page.getByRole("button", { name, exact: true }).click();
    await expect(page.locator("main h1:visible")).toHaveCount(1);
  }
  await page.screenshot({
    path: "test-results/studio-desktop.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("stream reducer retains zero scores and separates chapter scenes", () => {
  let state = streamReducer(initialStream, { type: "start" });
  const events = [
    { type: "chapter_start", payload: { chapter: 1 } },
    { type: "scene_start", payload: { scene: 1 } },
    { type: "token", payload: "First" },
    {
      type: "scene_complete",
      payload: { scene: 1, text: "Edited first", score: 0 },
    },
    { type: "chapter_start", payload: { chapter: 2 } },
    { type: "scene_start", payload: { scene: 1 } },
    { type: "token", payload: "Second" },
    { type: "done", payload: { status: "ok" } },
  ];
  for (const event of events)
    state = streamReducer(state, { type: "event", event });
  expect(state.scenes.map((s) => s.text)).toEqual(["Edited first", "Second"]);
  expect(state.scenes[0].score).toBe(0);
  expect(state.active).toBe(false);
});

test("skip link focuses the workspace without changing the active story", async ({
  page,
}) => {
  await fixtures(page);
  await page.goto("/#view=reader&project=atlas");
  const link = page.getByRole("link", { name: "Skip to workspace" });
  await link.focus();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/view=reader&project=atlas/);
  await expect(page.locator("#workspace")).toBeFocused();
});

test("failed prompt regeneration still displays previously saved prompts", async ({
  page,
}) => {
  await fixtures(page, {
    "/api/project/atlas/image-prompts": {
      status: "failed",
      error: "Provider unavailable. Saved prompts retained.",
      highlights: [
        {
          title: "Saved highlight",
          context: "The earlier version.",
          references: [],
          prompt: "The saved prompt is still here.",
        },
      ],
    },
  });
  await page.goto("/#view=reader&project=atlas");
  await page.getByText("Scene image prompts", { exact: false }).click();
  await expect(
    page.getByText("Provider unavailable. Saved prompts retained."),
  ).toBeVisible();
  await expect(page.getByText("The saved prompt is still here.")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Regenerate prompts" }),
  ).toBeEnabled();
});

test("reader refreshes its chapters when returning from another workspace", async ({
  page,
}) => {
  await fixtures(page);
  let expanded = false;
  await page.route("**/api/project/atlas/chapters", (route) =>
    route.fulfill({
      json: {
        chapters: [
          { number: 1, title: "Chapter 1", words: 100, status: "completed" },
          ...(expanded
            ? [
                {
                  number: 2,
                  title: "Chapter 2",
                  words: 200,
                  status: "completed",
                },
              ]
            : []),
        ],
      },
    }),
  );
  await page.goto("/#view=reader&project=atlas");
  await expect(page.getByRole("button", { name: /Chapter 1/ })).toBeVisible();
  await page.getByRole("button", { name: "Auto studio", exact: true }).click();
  expanded = true;
  await page.getByRole("button", { name: "Reader", exact: true }).click();
  await expect(page.getByRole("button", { name: /Chapter 2/ })).toBeVisible();
});
