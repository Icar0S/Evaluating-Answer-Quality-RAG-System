import { test as base, expect } from "@playwright/test";
import { API_BASE_URL, DEMO, JUDGE_MODEL, RagTarget } from "./helpers";

/**
 * O `test` de todos os specs. Igual ao do Playwright, mais três coisas:
 *
 * - `rag`: o RAG sob teste, uma opção de projeto. A demonstração em Docker
 *   (playwright.demo.config.ts) roda os specs do chat uma vez por RAG: o dos
 *   PDFs e o dos Markdown. Sem ela (playwright.config.ts), `null`.
 * - `apiBaseURL`: a API do RAG sob teste, ou API_BASE_URL sem RAG definido.
 * - Num projeto de RAG, antes de cada teste: pula se o backend daquele RAG
 *   estiver fora do ar ou com o índice vazio, confere que ele serve mesmo
 *   aquele índice (uma porta trocada mediria o RAG errado) e aponta o chat da
 *   página para ele.
 *
 * Em modo demonstração (DEMO=1), cada página ganha uma faixa no canto com o
 * teste em execução, o RAG e os dois modelos: quem RESPONDE (gerador, lido do
 * /health do backend) e quem JULGA (juiz). A faixa ignora o mouse
 * (pointer-events: none) e é `position: fixed`: não muda o layout nem
 * intercepta cliques dos testes.
 */
export const test = base.extend<{ rag: RagTarget | null; apiBaseURL: string; ragSetup: void }>({
  rag: [null, { option: true }],

  apiBaseURL: async ({ rag }, use) => {
    await use(rag?.apiBaseURL ?? API_BASE_URL);
  },

  ragSetup: [
    async ({ page, request, rag, apiBaseURL }, use, testInfo) => {
      let health: { generation_model?: string; rag_index?: string; vector_store_documents?: number } | null = null;
      if (rag || DEMO) {
        try {
          const resp = await request.get(`${apiBaseURL}/health`, { timeout: 15_000 });
          if (resp.ok()) health = await resp.json();
        } catch {
          // sem backend: num projeto de RAG o teste pula; fora dele, a faixa mostra isso
        }
      }
      if (rag) {
        testInfo.skip(!health, `RAG ${rag.index} indisponível em ${rag.apiBaseURL}.`);
        testInfo.skip(
          health?.vector_store_documents === 0,
          `Índice ${rag.index} vazio em ${rag.apiBaseURL} (faltam os documentos do acervo).`,
        );
        expect(health?.rag_index, `o backend em ${rag.apiBaseURL} deveria servir o índice ${rag.index}`).toBe(
          rag.index,
        );
        await page.addInitScript((url) => {
          (window as unknown as { __RAG_API_BASE_URL__: string }).__RAG_API_BASE_URL__ = url;
        }, rag.apiBaseURL);
      }
      if (DEMO) {
        await page.addInitScript(installBanner, {
          title: testInfo.titlePath.slice(1).join(" › "),
          generator: health?.generation_model ?? "backend indisponível",
          judge: JUDGE_MODEL,
          rag: rag ? `RAG dos ${rag.label} (${rag.index})` : null,
        });
      }
      await use();
    },
    { auto: true },
  ],
});

export { expect };

function installBanner(info: { title: string; generator: string; judge: string; rag: string | null }) {
  const render = () => {
    if (document.getElementById("e2e-demo-banner")) return;
    const box = document.createElement("div");
    box.id = "e2e-demo-banner";
    box.setAttribute("aria-hidden", "true");
    Object.assign(box.style, {
      position: "fixed",
      left: "12px",
      bottom: "12px",
      zIndex: "2147483647",
      width: "336px",
      padding: "10px 14px",
      borderRadius: "10px",
      background: "rgba(11, 11, 11, 0.88)",
      color: "#ffffff",
      font: "13px/1.45 system-ui, 'Segoe UI', sans-serif",
      boxShadow: "0 6px 24px rgba(0, 0, 0, 0.25)",
      pointerEvents: "none",
    });
    const title = document.createElement("div");
    title.style.fontWeight = "600";
    title.textContent = `▶ Teste E2E: ${info.title}`;
    box.append(title);
    if (info.rag) {
      const rag = document.createElement("div");
      rag.textContent = info.rag;
      box.append(rag);
    }
    const models = document.createElement("div");
    models.style.color = "#c3c2b7";
    models.textContent = `responde: ${info.generator}  ·  julga: ${info.judge}`;
    box.append(models);
    document.body.appendChild(box);
  };
  if (document.body) render();
  else document.addEventListener("DOMContentLoaded", render);
}
