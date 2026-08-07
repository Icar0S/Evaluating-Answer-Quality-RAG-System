// Painel de monitoramento do LLM local: tenta WebSocket (tempo real) e cai para
// polling via GET /metrics se a conexão falhar. Reaproveita API_BASE_URL de chat.js.

const monitorStatusDot = document.getElementById("monitor-status-dot");
const monitorStatusText = document.getElementById("monitor-status-text");
const metricModelEl = document.getElementById("metric-model");
const metricCpuEl = document.getElementById("metric-cpu");
const metricRamEl = document.getElementById("metric-ram");
const metricRamBarEl = document.getElementById("metric-ram-bar");
const metricGpuEl = document.getElementById("metric-gpu");
const metricVramEl = document.getElementById("metric-vram");
const metricVramBarEl = document.getElementById("metric-vram-bar");
const metricTpsEl = document.getElementById("metric-tps");
const metricLatencyTotalEl = document.getElementById("metric-latency-total");
const metricLatencyPromptEl = document.getElementById("metric-latency-prompt");
const metricLatencyGenEl = document.getElementById("metric-latency-gen");
const sparklineCpuEl = document.getElementById("sparkline-cpu");
const sparklineGpuEl = document.getElementById("sparkline-gpu");
const providerTabsEl = document.getElementById("provider-tabs");

const MONITOR_POLL_INTERVAL_MS = 1500;
const PROVIDERS_POLL_INTERVAL_MS = 15000;

// A sparkline é telemetria, então usa o ciano de instrumento dos tokens.
// Lido do CSS para que a paleta continue tendo uma fonte única de verdade.
const probeColor = getComputedStyle(document.documentElement).getPropertyValue("--probe-500").trim() || "#3aafc9";
const probeFill = getComputedStyle(document.documentElement).getPropertyValue("--probe-dim").trim() || "rgba(58, 175, 201, 0.16)";

/** Escreve uma métrica marcando se é leitura real ou ausência de leitura. */
function setMetric(el, text, available) {
  el.textContent = text;
  if (available) delete el.dataset.unavailable;
  else el.dataset.unavailable = "true";
}

// ---------- Seletor de provider (Local / Servidor) ----------

function renderProviderTabs(data) {
  providerTabsEl.innerHTML = "";
  data.providers.forEach((provider) => {
    const btn = document.createElement("button");
    btn.className = "header-tab" + (provider.name === data.active ? " active" : "");
    btn.dataset.provider = provider.name;
    btn.title = `${provider.base_url} · ${provider.reachable ? "conectado" : "inacessível"}`;

    const dot = document.createElement("span");
    dot.className = "badge-dot" + (provider.reachable ? " ok" : " down");
    btn.appendChild(dot);
    btn.append(provider.label);

    btn.addEventListener("click", () => switchProvider(provider.name));
    providerTabsEl.appendChild(btn);
  });
}

async function loadProviders() {
  try {
    const resp = await fetch(`${API_BASE_URL}/providers`);
    if (!resp.ok) throw new Error("providers indisponível");
    renderProviderTabs(await resp.json());
  } catch {
    // painel de status já sinaliza desconexão; sem providers pra listar por ora
  }
}

async function switchProvider(name) {
  try {
    await fetch(`${API_BASE_URL}/providers/active`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
  } catch {
    // loadProviders() abaixo reflete o estado real independente do resultado
  } finally {
    loadProviders();
  }
}

function drawSparkline(canvas, values) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!values || values.length < 2) return;

  const max = 100;
  const step = w / (values.length - 1);

  ctx.beginPath();
  values.forEach((v, i) => {
    const x = i * step;
    const y = h - (Math.min(v, max) / max) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = probeColor;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();

  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = probeFill;
  ctx.fill();
}

function formatMs(value) {
  if (value === null || value === undefined) return "—";
  return `${Math.round(value)} ms`;
}

function updateMonitorUI(data) {
  monitorStatusDot.classList.remove("ok", "processing", "down");
  if (data.status === "processing") {
    monitorStatusDot.classList.add("processing");
    monitorStatusText.textContent = "processando";
  } else {
    monitorStatusDot.classList.add("ok");
    monitorStatusText.textContent = "ocioso";
  }

  setMetric(metricModelEl, data.generation_model, true);

  if (data.host_metrics_available) {
    setMetric(metricCpuEl, `${data.cpu_percent.toFixed(0)}%`, true);
    drawSparkline(sparklineCpuEl, data.cpu_history);
    setMetric(metricRamEl, `${data.ram.used_gb} / ${data.ram.total_gb} GB`, true);
    metricRamBarEl.style.width = `${data.ram.percent}%`;
  } else {
    setMetric(metricCpuEl, "Indisponível no servidor remoto", false);
    sparklineCpuEl.getContext("2d").clearRect(0, 0, sparklineCpuEl.width, sparklineCpuEl.height);
    setMetric(metricRamEl, "Indisponível no servidor remoto", false);
    metricRamBarEl.style.width = "0%";
  }

  if (data.gpu.available) {
    setMetric(metricGpuEl, `${data.gpu.utilization_percent.toFixed(0)}%`, true);
    setMetric(metricVramEl, `${data.gpu.vram_used_gb} / ${data.gpu.vram_total_gb} GB`, true);
    metricVramBarEl.style.width = `${(data.gpu.vram_used_gb / data.gpu.vram_total_gb) * 100}%`;
    drawSparkline(sparklineGpuEl, data.gpu_history);
  } else {
    setMetric(
      metricGpuEl,
      data.host_metrics_available ? "Sem GPU NVIDIA detectada" : "Indisponível no servidor remoto",
      false
    );
    setMetric(metricVramEl, "—", false);
    metricVramBarEl.style.width = "0%";
    sparklineGpuEl.getContext("2d").clearRect(0, 0, sparklineGpuEl.width, sparklineGpuEl.height);
  }

  if (data.last_generation && data.last_generation.tokens_per_second != null) {
    setMetric(metricTpsEl, `${data.last_generation.tokens_per_second} tok/s`, true);
    setMetric(metricLatencyTotalEl, formatMs(data.last_generation.total_latency_ms), true);
    setMetric(metricLatencyPromptEl, formatMs(data.last_generation.prompt_eval_latency_ms), true);
    setMetric(metricLatencyGenEl, formatMs(data.last_generation.generation_latency_ms), true);
  } else {
    setMetric(metricTpsEl, "Aguardando geração...", false);
    setMetric(metricLatencyTotalEl, "—", false);
    setMetric(metricLatencyPromptEl, "—", false);
    setMetric(metricLatencyGenEl, "—", false);
  }
}

function startPollingFallback() {
  setInterval(async () => {
    try {
      const resp = await fetch(`${API_BASE_URL}/metrics`);
      if (!resp.ok) throw new Error("metrics unavailable");
      updateMonitorUI(await resp.json());
    } catch {
      monitorStatusDot.classList.remove("ok", "processing");
      monitorStatusDot.classList.add("down");
      monitorStatusText.textContent = "desconectado";
    }
  }, MONITOR_POLL_INTERVAL_MS);
}

function connectMetricsSocket() {
  const wsUrl = API_BASE_URL.replace(/^http/, "ws") + "/ws/metrics";
  let socket;
  try {
    socket = new WebSocket(wsUrl);
  } catch {
    startPollingFallback();
    return;
  }

  socket.addEventListener("message", (event) => {
    try {
      updateMonitorUI(JSON.parse(event.data));
    } catch {
      // ignora mensagens malformadas
    }
  });

  socket.addEventListener("error", () => {
    socket.close();
  });

  socket.addEventListener("close", () => {
    monitorStatusDot.classList.remove("ok", "processing");
    monitorStatusDot.classList.add("down");
    monitorStatusText.textContent = "desconectado (via polling)";
    startPollingFallback();
  });
}

connectMetricsSocket();
loadProviders();
setInterval(loadProviders, PROVIDERS_POLL_INTERVAL_MS);
