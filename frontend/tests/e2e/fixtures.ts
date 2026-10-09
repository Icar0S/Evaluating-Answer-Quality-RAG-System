import { test as base, expect } from "@playwright/test";
import { API_BASE_URL, DEMO, JUDGE_MODEL } from "./helpers";

/**
 * O `test` de todos os specs. Igual ao do Playwright, mais uma coisa só em modo
 * demonstração (DEMO=1, ligado por playwright.demo.config.ts): cada página do
 * teste ganha uma faixa no canto com o nome do teste em execução e os dois
 * modelos — quem RESPONDE (gerador, lido do /health do backend) e quem JULGA
 * (juiz). É o que permite à turma acompanhar, ao vivo, qual teste está mexendo
 * no sistema. Fora do modo demonstração, não faz nada.
 *
 * A faixa ignora o mouse (pointer-events: none) e é `position: fixed`: não
 * muda o layout nem intercepta cliques dos testes.
 */
export const test = base.extend<{ demoBanner: void }>({
  demoBanner: [
    async ({ page, request }, use, testInfo) => {
      if (DEMO) {
        let generator = "backend indisponível";
        try {
          const resp = await request.get(`${API_BASE_URL}/health`, { timeout: 15_000 });
          if (resp.ok()) generator = (await resp.json()).generation_model;
        } catch {
          // sem backend: a faixa mostra isso, e os testes que dependem dele se auto-pulam
        }
        await page.addInitScript(installBanner, {
          title: testInfo.titlePath.slice(1).join(" › "),
          generator,
          judge: JUDGE_MODEL,
        });
      }
      await use();
    },
    { auto: true },
  ],
});

export { expect };

function installBanner(info: { title: string; generator: string; judge: string }) {
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
    const models = document.createElement("div");
    models.style.color = "#c3c2b7";
    models.textContent = `responde: ${info.generator}  ·  julga: ${info.judge}`;
    box.append(title, models);
    document.body.appendChild(box);
  };
  if (document.body) render();
  else document.addEventListener("DOMContentLoaded", render);
}
