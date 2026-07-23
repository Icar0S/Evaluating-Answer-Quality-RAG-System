const API_BASE_URL = window.__RAG_API_BASE_URL__ || "http://localhost:8000";

const hudDot = document.getElementById("hud-dot");
const hudText = document.getElementById("hud-text");

async function updateHud() {
  try {
    const resp = await fetch(`${API_BASE_URL}/metrics`);
    if (!resp.ok) throw new Error("unavailable");
    const data = await resp.json();
    hudDot.className = "badge-dot " + (data.status === "processing" ? "processing" : "ok");
    const tps = data.last_generation?.tokens_per_second;
    const tpsText = tps != null ? `${tps} tok/s` : "aguardando geração";
    hudText.textContent = `${data.generation_model} · ${tpsText} · CPU ${data.cpu_percent.toFixed(0)}%`;
  } catch {
    hudDot.className = "badge-dot down";
    hudText.textContent = "backend indisponível — inicie a API para ver dados ao vivo";
  }
}

function formatLatency(ms) {
  if (ms == null) return "—";
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function formatPercent(fraction) {
  if (fraction == null) return "—";
  return `${Math.round(fraction * 100)}%`;
}

async function loadStats() {
  try {
    const resp = await fetch(`${API_BASE_URL}/stats`);
    if (!resp.ok) throw new Error("unavailable");
    const data = await resp.json();
    document.getElementById("stat-documents").textContent = data.vector_store_documents;
    document.getElementById("stat-interactions").textContent = data.total_interactions;
    document.getElementById("stat-latency").textContent = formatLatency(data.average_latency_ms);
    document.getElementById("stat-grounded").textContent = formatPercent(data.grounded_rate);
    document.getElementById("stat-ragas").textContent = data.ragas_faithfulness_avg ?? "em breve";
  } catch {
    ["stat-documents", "stat-interactions", "stat-latency", "stat-grounded", "stat-ragas"].forEach((id) => {
      document.getElementById(id).textContent = "offline";
    });
  }
}

function setupScrollReveal() {
  const targets = document.querySelectorAll(".reveal");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );
  targets.forEach((el) => observer.observe(el));
}

setupScrollReveal();
loadStats();
updateHud();
setInterval(updateHud, 3000);
