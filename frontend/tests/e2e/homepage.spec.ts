import { test, expect } from "@playwright/test";
import { isBackendUp } from "./helpers";

test.describe("Homepage", () => {
  test("hero carrega e o HUD não bloqueia a navegação para o chat", async ({ page }) => {
    await page.goto("/index.html");
    await expect(page.locator(".hero-title")).toBeVisible();

    // Regressão: o HUD fixo no canto já sobrepôs e bloqueou este link antes.
    await page.click(".home-nav .nav-cta");
    await expect(page).toHaveURL(/chat\.html/);
  });

  test("números da seção de estatísticas populam quando o backend está no ar", async ({ page, request }) => {
    test.skip(!(await isBackendUp(request)), "Backend indisponível em http://localhost:8000.");
    await page.goto("/index.html");
    await expect(page.locator("#stat-documents")).not.toHaveText("—", { timeout: 10_000 });
  });
});
