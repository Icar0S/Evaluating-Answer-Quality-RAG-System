import { defineConfig, devices } from "@playwright/test";
import base from "./playwright.config";

// Modo demonstração: os MESMOS testes de e2e/, para uma turma assistir ao vivo.
//
//   npm run demo        navegador visível, ações em câmera lenta
//   npm run demo:ui     modo UI do Playwright (lista de testes, linha do tempo, DOM a cada passo)
//   npm run demo:report reabre o relatório HTML com vídeo e trace de cada teste
//
// O resto vem de playwright.config.ts. O frontend é reaproveitado se já estiver
// na 5510 (o nginx do docker-compose, por exemplo).

// Liga a faixa com o teste em execução (e2e/fixtures.ts) e as pausas de leitura.
process.env.DEMO = process.env.DEMO ?? "1";

const slowMo = Number(process.env.DEMO_SLOWMO ?? 400);

export default defineConfig({
  ...base,
  timeout: 420_000,
  workers: 1,
  reporter: [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]],
  use: {
    ...base.use,
    trace: "on",
    video: "on",
    screenshot: "on",
  },
  projects: [
    {
      name: "chromium-demo",
      use: {
        ...devices["Desktop Chrome"],
        headless: false,
        viewport: { width: 1366, height: 800 },
        launchOptions: { slowMo },
      },
    },
  ],
});
