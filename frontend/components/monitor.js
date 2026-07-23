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

const MONITOR_POLL_INTERVAL_MS = 1500;

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
  ctx.strokeStyle = "#7c83fd";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();

  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = "rgba(124, 131, 253, 0.14)";
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

  metricModelEl.textContent = data.generation_model;
  metricCpuEl.textContent = `${data.cpu_percent.toFixed(0)}%`;
  drawSparkline(sparklineCpuEl, data.cpu_history);

  metricRamEl.textContent = `${data.ram.used_gb} / ${data.ram.total_gb} GB`;
  metricRamBarEl.style.width = `${data.ram.percent}%`;

  if (data.gpu.available) {
    metricGpuEl.textContent = `${data.gpu.utilization_percent.toFixed(0)}%`;
    metricVramEl.textContent = `${data.gpu.vram_used_gb} / ${data.gpu.vram_total_gb} GB`;
    metricVramBarEl.style.width = `${(data.gpu.vram_used_gb / data.gpu.vram_total_gb) * 100}%`;
    drawSparkline(sparklineGpuEl, data.gpu_history);
  } else {
    metricGpuEl.textContent = "Sem GPU NVIDIA detectada";
    metricVramEl.textContent = "—";
    metricVramBarEl.style.width = "0%";
    sparklineGpuEl.getContext("2d").clearRect(0, 0, sparklineGpuEl.width, sparklineGpuEl.height);
  }

  if (data.last_generation && data.last_generation.tokens_per_second != null) {
    metricTpsEl.textContent = `${data.last_generation.tokens_per_second} tok/s`;
    metricLatencyTotalEl.textContent = formatMs(data.last_generation.total_latency_ms);
    metricLatencyPromptEl.textContent = formatMs(data.last_generation.prompt_eval_latency_ms);
    metricLatencyGenEl.textContent = formatMs(data.last_generation.generation_latency_ms);
  } else {
    metricTpsEl.textContent = "Aguardando geração...";
    metricLatencyTotalEl.textContent = "—";
    metricLatencyPromptEl.textContent = "—";
    metricLatencyGenEl.textContent = "—";
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
