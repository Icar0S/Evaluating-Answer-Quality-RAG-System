import { test, expect } from "@playwright/test";
import { isBackendUp } from "./helpers";

test.describe("Composer da mensagem", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/chat.html");
  });

  test("não tem botão de emoji (removido a pedido)", async ({ page }) => {
    await expect(page.locator("#emoji-btn")).toHaveCount(0);
    await expect(page.locator(".emoji-popover")).toHaveCount(0);
  });

  test("botão de enviar começa desabilitado e habilita ao digitar", async ({ page }) => {
    await expect(page.locator("#send-button")).toBeDisabled();
    await page.fill("#question-input", "teste");
    await expect(page.locator("#send-button")).toBeEnabled();
    await page.fill("#question-input", "");
    await expect(page.locator("#send-button")).toBeDisabled();
  });

  test("Shift+Enter insere nova linha sem enviar a pergunta", async ({ page }) => {
    const input = page.locator("#question-input");
    await input.fill("linha 1");
    await input.press("Shift+Enter");
    await input.type("linha 2");
    await expect(input).toHaveValue("linha 1\nlinha 2");
    await expect(page.locator(".message-row.user")).toHaveCount(0);
  });

  test("contador de caracteres só aparece perto do limite de 4000", async ({ page }) => {
    await expect(page.locator("#hint-char-count")).toBeHidden();
    await page.fill("#question-input", "a".repeat(3600));
    await expect(page.locator("#hint-char-count")).toBeVisible();
    await expect(page.locator("#hint-char-count")).toHaveText("3600/4000");
  });

  test("mostra o modelo ativo no rodapé quando o backend está no ar", async ({ page, request }) => {
    test.skip(!(await isBackendUp(request)), "Backend indisponível em http://localhost:8000.");
    await expect(page.locator("#hint-model")).not.toHaveText("—", { timeout: 10_000 });
  });
});
