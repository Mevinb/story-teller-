import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  fullyParallel: false,
  workers: 1,
  timeout: 30000,
  use: {
    baseURL: "http://127.0.0.1:5017",
    viewport: { width: 1440, height: 1000 },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "../venv/bin/python ../scripts/serve_frontend_fixture.py",
    url: "http://127.0.0.1:5017",
    timeout: 60000,
    reuseExistingServer: false,
  },
});
