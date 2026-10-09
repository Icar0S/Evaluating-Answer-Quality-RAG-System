import { Page } from "@playwright/test";
import { test, expect } from "./fixtures";
import { API_BASE_URL, DEMO, JUDGE_MODEL, JUDGE_OLLAMA_URL, isBackendUp } from "./helpers";

// LLM-as-a-judge de ponta a ponta, na frente de quem assiste:
//   1. o juiz é OUTRO modelo, nunca o que gera as respostas (self-grading);
//   2. o gerador responde a uma pergunta pelo chat, como um usuário faria;
//   3. o juiz recebe a pergunta, os trechos recuperados e a resposta, e diz se
//      a resposta está apoiada nos trechos (fidelidade);
//   4. um painel na página mostra quem respondeu, quem julgou e o veredito.
// O veredito em si não é assertado (é opinião de um LLM); o que o teste garante
// é a separação dos papéis e que o julgamento aconteceu.

const QUESTION = "O que é o RAGAS e que tipo de avaliação ele propõe para sistemas RAG?";

const VERDICT_SCHEMA = {
  type: "object",
  properties: {
    veredito: { type: "string", enum: ["fiel", "infiel"] },
    justificativa: { type: "string" },
  },
  required: ["veredito", "justificativa"],
};

type Verdict = { veredito: "fiel" | "infiel"; justificativa: string };

function judgePrompt(question: string, contexts: string[], answer: string): string {
  const trechos = contexts.map((t, i) => `[Trecho ${i + 1}]\n${t}`).join("\n\n");
  return [
    "Você é um avaliador de sistemas RAG (LLM-as-a-judge). Decida se a RESPOSTA é fiel aos TRECHOS:",
    "'fiel' se todas as afirmações da resposta podem ser inferidas dos trechos; 'infiel' se alguma",
    "afirmação não está nos trechos ou os contradiz. Justifique em uma frase, em português.",
    "",
    `PERGUNTA: ${question}`,
    "",
    `TRECHOS:\n${trechos}`,
    "",
    `RESPOSTA: ${answer}`,
  ].join("\n");
}

type PanelState = {
  generator: string;
  judge: string;
  step: string;
  verdict?: Verdict;
};

/** Painel no canto da página com os dois papéis; atualizado a cada etapa do teste. */
async function showPanel(page: Page, state: PanelState): Promise<void> {
  await page.evaluate((s: PanelState) => {
    let box = document.getElementById("e2e-judge-panel");
    if (!box) {
      box = document.createElement("div");
      box.id = "e2e-judge-panel";
      box.setAttribute("aria-hidden", "true");
      Object.assign(box.style, {
        // Na coluna da esquerda (lista de conversas, quase vazia durante o
        // teste), acima da faixa do teste: assim não cobre a conversa.
        position: "fixed",
        left: "12px",
        bottom: "128px",  // folga para a faixa do teste, mesmo com título de 4 linhas
        zIndex: "2147483647",
        width: "336px",
        padding: "16px 18px",
        borderRadius: "12px",
        background: "#fcfcfb",
        color: "#0b0b0b",
        border: "1px solid rgba(11, 11, 11, 0.12)",
        boxShadow: "0 10px 30px rgba(0, 0, 0, 0.18)",
        font: "14px/1.5 system-ui, 'Segoe UI', sans-serif",
        pointerEvents: "none",
      });
      document.body.appendChild(box);
    }
    const row = (label: string, value: string) =>
      `<div style="display:flex;justify-content:space-between;gap:12px;margin:4px 0">` +
      `<span style="color:#52514e">${label}</span><strong>${value}</strong></div>`;
    let verdictHtml = "";
    if (s.verdict) {
      const good = s.verdict.veredito === "fiel";
      // Cor de status sempre com ícone + rótulo, nunca a cor sozinha.
      verdictHtml =
        `<div style="margin-top:10px;padding:10px 12px;border-radius:8px;` +
        `border-left:4px solid ${good ? "#0ca30c" : "#d03b3b"};background:#f9f9f7">` +
        `<div style="font-weight:700">${good ? "✓ Fiel aos trechos" : "✗ Infiel aos trechos"}</div>` +
        `<div style="color:#52514e;margin-top:4px">${s.verdict.justificativa.replace(/</g, "&lt;")}</div></div>`;
    }
    box.innerHTML =
      `<div style="font-weight:700;font-size:15px;margin-bottom:8px">LLM-as-a-judge: quem responde não julga</div>` +
      row("Responde (gerador)", s.generator) +
      row("Julga (juiz)", s.judge) +
      `<div style="margin-top:8px;color:#52514e">${s.step}</div>` +
      verdictHtml;
  }, state);
}

test.describe("Quem responde não é quem julga (LLM-as-a-judge)", () => {
  test("a resposta do chat é julgada por um modelo diferente do que a gerou", async ({ page, request }, testInfo) => {
    test.skip(!(await isBackendUp(request)), `Backend indisponível em ${API_BASE_URL}.`);
    const tags = await request.get(`${JUDGE_OLLAMA_URL}/api/tags`, { timeout: 10_000 }).catch(() => null);
    test.skip(!tags || !tags.ok(), `Ollama do juiz indisponível em ${JUDGE_OLLAMA_URL}.`);
    const models: string[] = ((await tags!.json()).models ?? []).map((m: { name: string }) => m.name);
    test.skip(
      !models.some((m) => m === JUDGE_MODEL || m === `${JUDGE_MODEL}:latest`),
      `Modelo juiz ${JUDGE_MODEL} não está baixado em ${JUDGE_OLLAMA_URL} (ollama pull ${JUDGE_MODEL}).`,
    );
    test.setTimeout(420_000);

    const generator: string = (await (await request.get(`${API_BASE_URL}/health`)).json()).generation_model;
    testInfo.annotations.push({ type: "gerador", description: generator }, { type: "juiz", description: JUDGE_MODEL });

    await test.step(`o juiz (${JUDGE_MODEL}) é um modelo diferente do gerador (${generator})`, async () => {
      expect(
        JUDGE_MODEL,
        "o juiz não pode ser o mesmo modelo que gera as respostas: ele estaria avaliando a si mesmo (self-grading)",
      ).not.toBe(generator);
    });

    await page.goto("/chat.html");
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    const panel: PanelState = { generator, judge: JUDGE_MODEL, step: "1/3 · enviando a pergunta pelo chat…" };
    await showPanel(page, panel);

    const chat = await test.step(`o gerador (${generator}) responde pelo chat`, async () => {
      const response = page.waitForResponse(
        (r) => r.url().endsWith("/chat") && r.request().method() === "POST",
        { timeout: 300_000 },
      );
      await page.fill("#question-input", QUESTION);
      await page.click("#send-button");
      const body = await (await response).json();
      await expect(page.locator(".message-row.assistant").last()).toBeVisible({ timeout: 60_000 });
      // O próprio chat mostra quem respondeu, embaixo da resposta.
      await expect(page.locator(".message-row.assistant .msg-meta").last()).toContainText(generator);
      expect(body.metadata.model).toBe(generator);
      return body as { answer: string; sources: { text: string }[] };
    });

    await showPanel(page, { ...panel, step: `2/3 · resposta de ${generator} recebida; ${JUDGE_MODEL} está julgando…` });

    const verdict = await test.step(`o juiz (${JUDGE_MODEL}) avalia a fidelidade da resposta`, async () => {
      const resp = await request.post(`${JUDGE_OLLAMA_URL}/api/chat`, {
        timeout: 300_000,
        data: {
          model: JUDGE_MODEL,
          stream: false,
          think: false,
          format: VERDICT_SCHEMA,
          options: { temperature: 0, num_ctx: 8192 },
          messages: [
            {
              role: "user",
              content: judgePrompt(QUESTION, chat.sources.map((s) => s.text), chat.answer),
            },
          ],
        },
      });
      expect(resp.ok(), `o juiz respondeu ${resp.status()}`).toBeTruthy();
      const data = await resp.json();
      return JSON.parse(data.message.content) as Verdict;
    });

    await showPanel(page, { ...panel, step: "3/3 · julgamento concluído", verdict });
    testInfo.annotations.push({ type: "veredito do juiz", description: `${verdict.veredito}: ${verdict.justificativa}` });
    expect(["fiel", "infiel"]).toContain(verdict.veredito);
    expect(verdict.justificativa.trim().length).toBeGreaterThan(0);

    if (DEMO) {
      // Tempo para a turma ler o veredito antes do próximo teste.
      await page.waitForTimeout(8_000);
    }
  });
});
