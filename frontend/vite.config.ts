import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = process.env.STORY_API_URL || "http://127.0.0.1:5000";

export default defineConfig({
  plugins: [react()],
  base: "/static/app/",
  build: { outDir: "../static/app", emptyOutDir: true, manifest: true },
  server: {
    proxy: {
      "/api": backend,
      "/classic": backend,
      "/static/favicon.svg": backend,
    },
  },
});
