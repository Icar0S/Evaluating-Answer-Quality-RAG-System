import { APIRequestContext, Page } from "@playwright/test";

export const API_BASE_URL = process.env.RAG_API_BASE_URL || "http://localhost:8010";

// Juiz do teste judge-vs-generator: um modelo DIFERENTE do que gera as
// respostas (o backend diz qual é o gerador em /health). Na demonstração em
// Docker o Ollama do container fica na 11435 (scripts/demo_e2e.bat define);
// fora dela, o Ollama da máquina, na 11434.
export const JUDGE_MODEL = process.env.RAG_JUDGE_MODEL || "qwen3:8b";
export const JUDGE_OLLAMA_URL = process.env.RAG_JUDGE_OLLAMA_URL || "http://localhost:11434";

// Modo demonstração (playwright.demo.config.ts liga): faixa com o teste em
// execução em cada página e pausas para a plateia acompanhar.
export const DEMO = Boolean(process.env.DEMO) && process.env.DEMO !== "0";

/**
 * Um RAG sob teste: a versão do índice (backend/app/config.py, RAG_INDICES) e
 * a API que a serve. A demonstração em Docker tem dois, um por acervo
 * (playwright.demo.config.ts); fora dela, nenhum, e vale API_BASE_URL.
 */
export type RagTarget = { index: string; label: string; apiBaseURL: string };

export async function isBackendUp(request: APIRequestContext, baseURL: string = API_BASE_URL): Promise<boolean> {
  try {
    // 15s, não 5s: com o Ollama ocupado (gerando para o teste anterior), o
    // /health chegou a passar de 5s e três testes se auto-pularam com o backend
    // no ar. Backend fora do ar continua falhando na hora (conexão recusada).
    const resp = await request.get(`${baseURL}/health`, { timeout: 15_000 });
    return resp.ok();
  } catch {
    return false;
  }
}

/**
 * Painel no canto esquerdo da página (sobre a lista de conversas, quase vazia
 * durante os testes), acima da faixa do teste. Ignora o mouse e não muda o
 * layout. Texto montado com textContent: nada do que vem dos goldens vira HTML.
 */
export async function showSidePanel(
  page: Page,
  panel: { title: string; rows: [string, string][]; note?: string },
): Promise<void> {
  await page.evaluate((p) => {
    let box = document.getElementById("e2e-side-panel");
    if (!box) {
      box = document.createElement("div");
      box.id = "e2e-side-panel";
      box.setAttribute("aria-hidden", "true");
      Object.assign(box.style, {
        position: "fixed",
        left: "12px",
        bottom: "150px", // folga para a faixa do teste (fixtures.ts), mesmo com título longo e a linha do RAG
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
    box.replaceChildren();
    const title = document.createElement("div");
    Object.assign(title.style, { fontWeight: "700", fontSize: "15px", marginBottom: "8px" });
    title.textContent = p.title;
    box.appendChild(title);
    for (const [label, value] of p.rows) {
      const row = document.createElement("div");
      Object.assign(row.style, { display: "flex", justifyContent: "space-between", gap: "12px", margin: "4px 0" });
      const name = document.createElement("span");
      name.style.color = "#52514e";
      name.textContent = label;
      const strong = document.createElement("strong");
      strong.style.textAlign = "right";
      strong.textContent = value;
      row.append(name, strong);
      box.appendChild(row);
    }
    if (p.note) {
      const note = document.createElement("div");
      Object.assign(note.style, { marginTop: "8px", color: "#52514e", fontSize: "13px" });
      note.textContent = p.note;
      box.appendChild(note);
    }
  }, panel);
}
