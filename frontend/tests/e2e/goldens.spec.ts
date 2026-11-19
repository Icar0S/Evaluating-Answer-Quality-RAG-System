import * as fs from "fs";
import * as path from "path";
import { test, expect } from "./fixtures";
import { DEMO, showSidePanel } from "./helpers";

// As perguntas de referência (goldens) feitas pelo chat a cada RAG, como um
// usuário faria. Só roda na demonstração com relatório (scripts\demo_e2e.bat,
// que define DEMO_ANSWERS_DIR): cada resposta, com os trechos que o chat
// devolveu, vai para um arquivo por RAG, e tests/demo_e2e.py entrega esses
// arquivos ao RAGAS. O relatório compara a qualidade dos dois RAGs com as
// respostas que a turma viu na tela.
//
// Os dois RAGs recebem as MESMAS perguntas, as dos PDFs e as dos Markdown. Se
// o índice do RAG não tem os artigos do acervo da pergunta, ela mede se o RAG
// se abstém. Quem diz é o /health (rag_index_origins): o pdf-v1 e o md-v1 têm
// um acervo cada; o pdf-v2 e o md-v2, os dois.

const ANSWERS_DIR = process.env.DEMO_ANSWERS_DIR;
// DEMO_GOLDENS=N: só os N primeiros de cada conjunto (para um ensaio rápido).
const PER_SET = Number(process.env.DEMO_GOLDENS) || undefined;
const GOLDENS_DIR = path.resolve(__dirname, "..", "..", "..", "tests", "deepeval", "goldens");
const SETS = [
  { origin: "pdf", file: "dataset.json", label: "acervo dos PDFs" },
  { origin: "md", file: "dataset_md.json", label: "acervo dos Markdown" },
];

type Golden = { name: string; input: string; expected_output: string };
type Source = { document: string; page: number | null; text: string };

const CASES = SETS.flatMap((set) =>
  (JSON.parse(fs.readFileSync(path.join(GOLDENS_DIR, set.file), "utf-8")) as Golden[])
    .slice(0, PER_SET)
    .map((golden) => ({ ...set, golden })),
);

function clip(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max - 1).trimEnd()}…`;
}

test.describe("Perguntas de referência", () => {
  for (const { origin, label, golden } of CASES) {
    test(`${origin} · ${golden.name}`, async ({ page, rag, request }) => {
      test.skip(!ANSWERS_DIR || !rag, "Só na demonstração com relatório (scripts\\demo_e2e.bat).");
      test.setTimeout(360_000);
      const target = rag!;
      const health = await (await request.get(`${target.apiBaseURL}/health`, { timeout: 15_000 })).json();
      // Backend sem o campo (imagem antiga): vale o prefixo do nome do índice, como antes.
      const origins: string[] = health.rag_index_origins ?? [target.index.split("-")[0]];
      const ownCorpus = origins.includes(origin);

      await page.goto("/chat.html");
      await showSidePanel(page, {
        title: "Pergunta de referência (golden)",
        rows: [
          ["RAG", `${target.label} (${target.index})`],
          ["Pergunta do", label],
        ],
        note: ownCorpus
          ? `Resposta esperada: ${clip(golden.expected_output, 220)}`
          : "Assunto de fora deste acervo: o certo é o RAG dizer que não encontrou.",
      });

      const answered = page.waitForResponse(
        (r) => r.url().endsWith("/chat") && r.request().method() === "POST",
        { timeout: 300_000 },
      );
      const started = Date.now();
      await page.fill("#question-input", golden.input);
      await page.click("#send-button");
      const response = await answered;
      const body = await response.json().catch(() => null);
      const ok = response.ok() && body !== null;
      const sources: Source[] = ok ? body.sources : [];
      const record = {
        rag_index: target.index,
        name: golden.name,
        origin,
        input: golden.input,
        actual_output: ok ? body.answer : null,
        retrieval_context: sources.map((s) => s.text),
        sources: sources.map((s) => (s.page == null ? s.document : `${s.document} (pág. ${s.page})`)),
        source_documents: sources.map((s) => s.document),
        error: ok ? null : `HTTP ${response.status()}: ${body?.detail ?? "sem corpo"}`,
        // Latência do backend (busca + geração), comparável à de tests/generate_answers.py.
        seconds: ok ? Math.round(body.metadata.latency_ms / 100) / 10 : null,
        wall_seconds: Math.round((Date.now() - started) / 100) / 10,
        metadata: ok ? body.metadata : null,
        finished_at: new Date().toISOString(),
      };
      // Grava antes de conferir: uma resposta com erro também entra no
      // relatório, como nota não medida.
      fs.mkdirSync(ANSWERS_DIR!, { recursive: true });
      fs.appendFileSync(path.join(ANSWERS_DIR!, `respostas_${target.index}.jsonl`), `${JSON.stringify(record)}\n`, "utf-8");

      expect(ok, record.error ?? "").toBeTruthy();
      expect(body.metadata.rag_index, "a resposta veio de outro índice").toBe(target.index);
      await expect(page.locator(".message-row.assistant").last()).toBeVisible({ timeout: 60_000 });
      if (DEMO) {
        // Tempo para a turma ver a resposta antes da próxima pergunta.
        await page.waitForTimeout(1_500);
      }
    });
  }
});
