import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  // Perguntas de referência para o relatório de qualidade: só na demonstração
  // em Docker (playwright.demo.config.ts), que tem os dois RAGs no ar.
  testIgnore: "**/goldens.spec.ts",
  timeout: 90_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: "list",
  use: {
    baseURL: "http://localhost:5510",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "python -m http.server 5510 --directory ..",
    url: "http://localhost:5510/index.html",
    reuseExistingServer: true,
    timeout: 15_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
