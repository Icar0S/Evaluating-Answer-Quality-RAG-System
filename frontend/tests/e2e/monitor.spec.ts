import { test, expect } from "@playwright/test";
import { isBackendUp } from "./helpers";

test.describe("Painel de Monitor na sidebar", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/chat.html");
  });

  test("por padrão mostra o histórico de sessões, não o monitor", async ({ page }) => {
    await expect(page.locator("#sidebar-sessions")).toBeVisible();
    await expect(page.locator("#sidebar-monitor")).toBeHidden();
  });

  test("alterna para o monitor mantendo o chat visível, sem colapsar a largura", async ({ page }) => {
    const before = await page.locator("#chat-log").boundingBox();

    await page.click('.rail-btn[data-sidebar="monitor"]');

    await expect(page.locator("#sidebar-monitor")).toBeVisible();
    await expect(page.locator("#sidebar-sessions")).toBeHidden();
    await expect(page.locator("#chat-log")).toBeVisible();

    const after = await page.locator("#chat-log").boundingBox();
    // Regressão do bug de CSS Grid: main-panel não pode colapsar para largura ~0
    // quando a sidebar troca de conteúdo (grid agora é estático, não deveria mudar nada).
    expect(after!.width).toBeGreaterThan(400);
    expect(after!.width).toBeCloseTo(before!.width, -1);
  });

  test("voltar para sessões restaura a lista original", async ({ page }) => {
    await page.click('.rail-btn[data-sidebar="monitor"]');
    await expect(page.locator("#sidebar-monitor")).toBeVisible();

    await page.click('.rail-btn[data-sidebar="sessions"]');
    await expect(page.locator("#sidebar-sessions")).toBeVisible();
    await expect(page.locator("#sidebar-monitor")).toBeHidden();
  });

  test("métricas populam quando o backend está no ar", async ({ page, request }) => {
    test.skip(!(await isBackendUp(request)), "Backend indisponível em http://localhost:8000.");
    await page.click('.rail-btn[data-sidebar="monitor"]');
    await expect(page.locator("#metric-model")).not.toHaveText("—", { timeout: 10_000 });
    await expect(page.locator("#metric-cpu")).not.toHaveText("—", { timeout: 10_000 });
  });
});
