import { APIRequestContext } from "@playwright/test";

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

export async function isBackendUp(request: APIRequestContext): Promise<boolean> {
  try {
    // 15s, não 5s: com o Ollama ocupado (gerando para o teste anterior), o
    // /health chegou a passar de 5s e três testes se auto-pularam com o backend
    // no ar. Backend fora do ar continua falhando na hora (conexão recusada).
    const resp = await request.get(`${API_BASE_URL}/health`, { timeout: 15_000 });
    return resp.ok();
  } catch {
    return false;
  }
}
