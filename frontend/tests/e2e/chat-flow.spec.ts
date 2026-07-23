import { test, expect } from "@playwright/test";
import { isBackendUp } from "./helpers";

// Estes testes fazem chamadas reais ao Ollama via backend — são os mais lentos
// da suíte (uma geração local pode levar 15-30s). Pulados automaticamente se a
// API não estiver no ar em http://localhost:8000.

test.describe("Fluxo real de chat (usa o backend/Ollama)", () => {
  test.beforeEach(async ({ page, request }) => {
    test.skip(!(await isBackendUp(request)), "Backend indisponível em http://localhost:8000 — inicie a API antes de rodar este teste.");
    await page.goto("/chat.html");
    await page.evaluate(() => localStorage.clear());
    await page.reload();
  });

  test("envia uma pergunta e recebe resposta com sessão salva", async ({ page }) => {
    await page.fill("#question-input", "O que e RAG?");
    await page.click("#send-button");

    await expect(page.locator(".message-row.user")).toHaveCount(1);
    await expect(page.locator(".message-row.assistant")).toBeVisible({ timeout: 90_000 });

    const answerText = await page.locator(".message-row.assistant .bubble").innerText();
    expect(answerText.length).toBeGreaterThan(0);
    await expect(page.locator(".session-item")).toHaveCount(1);
  });

  test("scroll e digitação continuam funcionando após múltiplas mensagens", async ({ page }) => {
    const questions = ["O que e chunking?", "O que e embeddings?"];
    for (const question of questions) {
      await page.fill("#question-input", question);
      await page.click("#send-button");
      await expect(page.locator(".message-row.assistant").last()).toBeVisible({ timeout: 90_000 });
    }

    // Regressão: a barra de input não pode ser empurrada para fora do viewport.
    const inputBox = await page.locator("#question-input").boundingBox();
    const viewport = page.viewportSize();
    expect(inputBox!.y + inputBox!.height).toBeLessThanOrEqual(viewport!.height);

    await page.fill("#question-input", "ainda consigo digitar");
    await expect(page.locator("#question-input")).toHaveValue("ainda consigo digitar");

    // Regressão: o chat-log deve ter overflow interno de verdade, não crescer sem limite.
    const scrollInfo = await page.evaluate(() => {
      const log = document.getElementById("chat-log")!;
      return { scrollHeight: log.scrollHeight, clientHeight: log.clientHeight };
    });
    expect(scrollInfo.scrollHeight).toBeGreaterThan(scrollInfo.clientHeight);
  });
});
