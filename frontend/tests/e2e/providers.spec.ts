import { test, expect, Page } from "@playwright/test";

// Seletor Local/Servidor do painel de Monitor, com o backend inteiro mockado
// (HTTP + WebSocket). Nao depende de API no ar nem de Ollama, entao roda no CI
// do GitHub igual roda na maquina — e falha de verdade quando quebra, em vez de
// ser pulado como os testes que precisam de geracao real.

type Snapshot = Record<string, unknown>;

const SNAPSHOT_LOCAL: Snapshot = {
  generation_model: "qwen3:8b",
  provider: "local",
  host_metrics_available: true,
  status: "idle",
  cpu_percent: 23,
  ram: { used_gb: 12.5, total_gb: 15.71, percent: 79.6 },
  gpu: { available: true, name: "GPU de teste", utilization_percent: 41, vram_used_gb: 5.2, vram_total_gb: 8 },
  last_generation: null,
  cpu_history: [10, 20, 23],
  gpu_history: [30, 35, 41],
};

const SNAPSHOT_REMOTO: Snapshot = {
  generation_model: "coder-32k",
  provider: "remote",
  host_metrics_available: false,
  status: "idle",
  cpu_percent: 0,
  ram: { used_gb: 0, total_gb: 0, percent: 0 },
  gpu: { available: false },
  last_generation: null,
  cpu_history: [],
  gpu_history: [],
};

function providerPayload(active: string, remoteReachable = true) {
  return {
    active,
    providers: [
      {
        name: "local",
        label: "Local",
        base_url: "http://localhost:11434",
        generation_model: "qwen3:8b",
        embedding_model: "nomic-embed-text",
        reachable: true,
      },
      {
        name: "remote",
        label: "Servidor (Mac mini)",
        base_url: "https://llm.smartdatatest.com",
        generation_model: "coder-32k",
        embedding_model: "nomic-embed-text",
        reachable: remoteReachable,
      },
    ],
  };
}

/** Sobe um backend falso e devolve o estado mutavel que o teste controla. */
async function mockBackend(page: Page, options: { remoteReachable?: boolean } = {}) {
  const state = {
    active: "local",
    snapshot: SNAPSHOT_LOCAL,
    trocasPedidas: [] as string[],
  };

  await page.route("**/health", (route) =>
    route.fulfill({
      json: {
        status: "ok",
        ollama_reachable: true,
        generation_model: "qwen3:8b",
        embedding_model: "nomic-embed-text",
        vector_store_documents: 536,
      },
    }),
  );

  await page.route("**/providers/active", async (route) => {
    const { name } = route.request().postDataJSON();
    state.trocasPedidas.push(name);
    state.active = name;
    state.snapshot = name === "remote" ? SNAPSHOT_REMOTO : SNAPSHOT_LOCAL;
    await route.fulfill({ json: providerPayload(state.active, options.remoteReachable ?? true) });
  });

  await page.route("**/providers", (route) =>
    route.fulfill({ json: providerPayload(state.active, options.remoteReachable ?? true) }),
  );

  await page.route("**/metrics", (route) => route.fulfill({ json: state.snapshot }));

  // O monitor usa WebSocket como transporte principal; sem isso ele so cairia no
  // polling depois de um timeout, deixando o teste lento e nao-deterministico.
  await page.routeWebSocket(/\/ws\/metrics/, (ws) => {
    const timer = setInterval(() => ws.send(JSON.stringify(state.snapshot)), 150);
    ws.onClose(() => clearInterval(timer));
  });

  return state;
}

async function abrirMonitor(page: Page) {
  await page.goto("/chat.html");
  await page.click('.rail-btn[data-sidebar="monitor"]');
  await expect(page.locator("#sidebar-monitor")).toBeVisible();
}

test.describe("Seletor de provider no monitor", () => {
  test("mostra uma aba por provider configurado, com o ativo destacado", async ({ page }) => {
    await mockBackend(page);
    await abrirMonitor(page);

    const abas = page.locator("#provider-tabs .header-tab");
    await expect(abas).toHaveCount(2);
    await expect(abas.nth(0)).toHaveText(/Local/);
    await expect(abas.nth(1)).toHaveText(/Servidor \(Mac mini\)/);
    await expect(abas.nth(0)).toHaveClass(/active/);
    await expect(abas.nth(1)).not.toHaveClass(/active/);
  });

  test("sinaliza no ponto de status qual provider esta acessivel", async ({ page }) => {
    await mockBackend(page, { remoteReachable: false });
    await abrirMonitor(page);

    await expect(page.locator('#provider-tabs .header-tab[data-provider="local"] .badge-dot')).toHaveClass(/ok/);
    await expect(page.locator('#provider-tabs .header-tab[data-provider="remote"] .badge-dot')).toHaveClass(/down/);
  });

  test("clicar em Servidor troca o provider ativo no backend", async ({ page }) => {
    const state = await mockBackend(page);
    await abrirMonitor(page);

    await page.click('#provider-tabs .header-tab[data-provider="remote"]');

    await expect(page.locator('#provider-tabs .header-tab[data-provider="remote"]')).toHaveClass(/active/);
    await expect(page.locator('#provider-tabs .header-tab[data-provider="local"]')).not.toHaveClass(/active/);
    expect(state.trocasPedidas).toEqual(["remote"]);
  });

  test("com o servidor remoto ativo, CPU/RAM/GPU aparecem como indisponiveis", async ({ page }) => {
    await mockBackend(page);
    await abrirMonitor(page);

    // Confere primeiro que o modo local mostra numeros de verdade...
    await expect(page.locator("#metric-cpu")).toHaveText("23%");
    await expect(page.locator("#metric-model")).toHaveText("qwen3:8b");

    await page.click('#provider-tabs .header-tab[data-provider="remote"]');

    // ...e que no remoto eles viram um aviso explicito, nao os numeros da
    // maquina local (que seriam tecnicamente corretos e completamente enganosos).
    await expect(page.locator("#metric-model")).toHaveText("coder-32k");
    await expect(page.locator("#metric-cpu")).toHaveText(/[Ii]ndispon/);
    await expect(page.locator("#metric-ram")).toHaveText(/[Ii]ndispon/);
    await expect(page.locator("#metric-gpu")).toHaveText(/[Ii]ndispon/);
  });

  test("voltar para Local restaura as metricas do host", async ({ page }) => {
    await mockBackend(page);
    await abrirMonitor(page);

    await page.click('#provider-tabs .header-tab[data-provider="remote"]');
    await expect(page.locator("#metric-cpu")).toHaveText(/[Ii]ndispon/);

    await page.click('#provider-tabs .header-tab[data-provider="local"]');

    await expect(page.locator("#metric-cpu")).toHaveText("23%");
    await expect(page.locator("#metric-ram")).toHaveText("12.5 / 15.71 GB");
    await expect(page.locator("#metric-gpu")).toHaveText("41%");
  });

  test("nao quebra o layout do chat ao trocar de provider", async ({ page }) => {
    await mockBackend(page);
    await abrirMonitor(page);
    const antes = await page.locator("#chat-log").boundingBox();

    await page.click('#provider-tabs .header-tab[data-provider="remote"]');

    const depois = await page.locator("#chat-log").boundingBox();
    expect(depois!.width).toBeCloseTo(antes!.width, -1);
    expect(depois!.width).toBeGreaterThan(400);
  });
});
