import { Page } from "@playwright/test";
import { test, expect } from "./fixtures";

// O "chat limpo" do protocolo da pergunta única (plano-escopo-final.md, §4): o
// histórico da conversa fica só na tela (localStorage) e nunca chega ao modelo.
// Cada POST /chat leva só a pergunta. Por isso perguntar no mesmo chat ou num
// chat novo dá ao modelo exatamente a mesma entrada; o estado que pode mudar a
// resposta é o do servidor (o cache do Ollama), e esse o protocolo zera em
// tests/generate_answers.py --estado-limpo.
//
// Backend mockado: roda no CI do GitHub, sem API nem Ollama. Se um dia o chat
// passar a mandar o histórico, este teste falha, e o protocolo tem de mudar junto.

// A chave das conversas no localStorage (frontend/components/chat.js, SESSIONS_KEY).
const SESSIONS_KEY = "rag_sessions_v1";

async function mockBackend(page: Page) {
  const bodies: Record<string, unknown>[] = [];
  await page.route("**/health", (route) =>
    route.fulfill({
      json: {
        status: "ok",
        ollama_reachable: true,
        generation_model: "gemma3:4b",
        embedding_model: "nomic-embed-text",
        vector_store_documents: 857,
        rag_index: "md-v1",
      },
    }),
  );
  await page.route("**/providers", (route) => route.fulfill({ json: { active: "local", providers: [] } }));
  await page.route("**/metrics", (route) => route.fulfill({ status: 503, json: { detail: "sem monitor no teste" } }));
  await page.routeWebSocket(/\/ws\/metrics/, () => {});
  await page.route("**/chat", async (route) => {
    const body = route.request().postDataJSON();
    bodies.push(body);
    await route.fulfill({
      json: {
        answer: `Resposta ${bodies.length}: os vieses de posição e de verbosidade.`,
        sources: [
          {
            chunk_id: "mtbench::c0",
            document: "mtbench.md",
            page: null,
            section: null,
            text: "position, verbosity, and self-enhancement biases",
            similarity_score: 0.74,
          },
        ],
        metadata: {
          model: "gemma3:4b",
          embedding_model: "nomic-embed-text",
          provider: "local",
          prompt_version: "v1",
          latency_ms: 10,
          retrieval_latency_ms: 1,
          generation_latency_ms: 9,
          prompt_tokens: 900,
          completion_tokens: 40,
          top_k: 4,
          rag_index: "md-v1",
          temperature: null,
          seed: null,
        },
        grounded: true,
      },
    });
  });
  return bodies;
}

async function ask(page: Page, question: string, expectedAnswers: number) {
  await page.fill("#question-input", question);
  await page.click("#send-button");
  await expect(page.locator(".message-row.assistant")).toHaveCount(expectedAnswers);
}

test.describe("Chat limpo: o histórico não chega ao modelo", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/chat.html");
    await page.evaluate(() => localStorage.clear());
  });

  test("a segunda pergunta da mesma conversa vai sozinha, sem a primeira nem a resposta", async ({ page }) => {
    const bodies = await mockBackend(page);
    await page.reload();

    await ask(page, "Quais vieses o MT-Bench examina?", 1);
    await ask(page, "E qual a concordância do GPT-4?", 2);

    // A tela tem a conversa inteira...
    await expect(page.locator(".message-row.user")).toHaveCount(2);
    const stored = await page.evaluate((key) => JSON.parse(localStorage.getItem(key) || "[]"), SESSIONS_KEY);
    expect(stored[0].messages).toHaveLength(4);
    // ...e cada POST /chat levou só a pergunta: nenhum campo de histórico, nenhuma mensagem anterior.
    expect(bodies).toEqual([
      { question: "Quais vieses o MT-Bench examina?" },
      { question: "E qual a concordância do GPT-4?" },
    ]);
    expect(JSON.stringify(bodies[1])).not.toContain("MT-Bench");
    expect(JSON.stringify(bodies[1])).not.toContain("Resposta 1");
  });

  test("a mesma pergunta no mesmo chat e num chat novo manda o mesmo corpo", async ({ page }) => {
    const bodies = await mockBackend(page);
    await page.reload();
    const question = "Quais vieses o MT-Bench examina?";

    await ask(page, "Uma pergunta antes, para sujar a conversa.", 1);
    await ask(page, question, 2);
    await page.click("#new-session-btn");
    await expect(page.locator(".message-row")).toHaveCount(0);
    await ask(page, question, 1);

    expect(bodies[1]).toEqual({ question });
    expect(bodies[2]).toEqual(bodies[1]);
  });
});
