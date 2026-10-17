import { defineConfig, devices, ReporterDescription } from "@playwright/test";
import base from "./playwright.config";
import { RagTarget } from "./e2e/helpers";

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

// Os dois RAGs do docker-compose.yml: mesmo gerador, mesmos embeddings,
// acervos diferentes.
const RAGS: RagTarget[] = [
  { index: "pdf-v1", label: "PDFs", apiBaseURL: process.env.DEMO_PDF_API_URL ?? "http://localhost:8010" },
  { index: "md-v1", label: "Markdown", apiBaseURL: process.env.DEMO_MD_API_URL ?? "http://localhost:8011" },
];

// Specs que dependem do acervo: rodam uma vez por RAG. O resto (interface,
// monitor, seletor de provider) roda uma vez só, contra o backend padrão.
const RAG_SPECS = /(chat-flow|goldens|judge-vs-generator)\.spec\.ts$/;
// As perguntas de referência só entram quando há onde gravar as respostas para
// o relatório (tests/demo_e2e.py define DEMO_ANSWERS_DIR). Em `npm run demo` e
// `npm run demo:ui`, ficam fora da lista.
const SKIP_GOLDENS = process.env.DEMO_ANSWERS_DIR ? [] : [/goldens\.spec\.ts$/];

const browser = {
  ...devices["Desktop Chrome"],
  headless: false,
  viewport: { width: 1366, height: 800 },
  launchOptions: { slowMo },
};

const reporter: ReporterDescription[] = [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]];
// tests/demo_e2e.py lê o resultado de cada projeto daqui.
if (process.env.DEMO_RESULTS_JSON) reporter.push(["json", { outputFile: process.env.DEMO_RESULTS_JSON }]);

export default defineConfig<{ rag: RagTarget | null }>({
  ...base,
  // O padrão ignora goldens.spec.ts; aqui, quem decide são os projetos.
  testIgnore: [],
  timeout: 420_000,
  workers: 1,
  reporter,
  use: {
    ...base.use,
    trace: "on",
    video: "on",
    screenshot: "on",
  },
  // Nesta ordem: a interface; depois o RAG dos PDFs; depois o dos Markdown.
  projects: [
    { name: "interface", testIgnore: RAG_SPECS, use: browser },
    ...RAGS.map((rag) => ({
      name: `RAG ${rag.index} (${rag.label})`,
      testMatch: RAG_SPECS,
      testIgnore: SKIP_GOLDENS,
      use: { ...browser, rag },
    })),
  ],
});
