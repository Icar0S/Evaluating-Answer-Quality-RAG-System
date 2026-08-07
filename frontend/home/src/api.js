// window.__RAG_API_BASE_URL__ é o override usado pelos testes Playwright —
// mesmo contrato de components/chat.js e components/monitor.js.
export const API_BASE_URL = window.__RAG_API_BASE_URL__ || "http://localhost:8000";

export async function getJson(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) throw new Error(`${path} indisponível`);
  return response.json();
}

/** Painel de leitura: sempre caixa alta e curto o suficiente para caber no board. */
export function formatStat(value, kind) {
  if (value === null || value === undefined) return null;

  switch (kind) {
    case "latency":
      return value < 1000 ? `${Math.round(value)}MS` : `${(value / 1000).toFixed(1)}S`;
    case "percent":
      return `${Math.round(value * 100)}%`;
    default:
      return String(value);
  }
}
