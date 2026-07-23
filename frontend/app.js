// Cliente de chat simples, sem dependências, para facilitar testes E2E isolados.
// Permite override da URL da API via window.__RAG_API_BASE_URL__ (usado pelos testes Playwright).
const API_BASE_URL = window.__RAG_API_BASE_URL__ || "http://localhost:8000";

const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const sendButton = document.getElementById("send-button");
const errorBanner = document.getElementById("error-banner");
const statusIndicator = document.getElementById("status-indicator");

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

function appendMessage(role, text) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.textContent = text;
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
  return el;
}

function appendAssistantMessage(data) {
  const el = document.createElement("div");
  el.className = "message assistant" + (data.grounded ? "" : " ungrounded");

  const answerEl = document.createElement("div");
  answerEl.className = "answer-text";
  answerEl.textContent = data.answer;
  el.appendChild(answerEl);

  if (data.sources && data.sources.length > 0) {
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `Fontes (${data.sources.length})`;
    details.appendChild(summary);
    data.sources.forEach((source) => {
      const p = document.createElement("p");
      p.textContent = `${source.document} (pág. ${source.page ?? "?"}) — similaridade ${source.similarity_score.toFixed(3)}`;
      details.appendChild(p);
    });
    el.appendChild(details);
  }

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = `${data.metadata.model} · ${Math.round(data.metadata.latency_ms)}ms`;
  el.appendChild(meta);

  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function setLoading(isLoading) {
  sendButton.disabled = isLoading;
  questionInput.disabled = isLoading;
  let indicator = document.getElementById("loading-indicator");
  if (isLoading) {
    if (!indicator) {
      indicator = document.createElement("div");
      indicator.id = "loading-indicator";
      indicator.className = "loading-indicator";
      indicator.textContent = "Pensando...";
      chatLog.appendChild(indicator);
      chatLog.scrollTop = chatLog.scrollHeight;
    }
  } else if (indicator) {
    indicator.remove();
  }
}

async function checkHealth() {
  try {
    const resp = await fetch(`${API_BASE_URL}/health`);
    if (!resp.ok) throw new Error("unhealthy");
    const data = await resp.json();
    statusIndicator.classList.toggle("ok", data.ollama_reachable);
    statusIndicator.classList.toggle("down", !data.ollama_reachable);
    statusIndicator.title = data.ollama_reachable ? "API e Ollama disponíveis" : "Ollama indisponível";
  } catch (err) {
    statusIndicator.classList.remove("ok");
    statusIndicator.classList.add("down");
    statusIndicator.title = "API indisponível";
  }
}

async function sendQuestion(question) {
  clearError();
  appendMessage("user", question);
  setLoading(true);

  try {
    const resp = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!resp.ok) {
      const errBody = await resp.json().catch(() => ({}));
      throw new Error(errBody.detail || `Erro ${resp.status} ao consultar a API.`);
    }

    const data = await resp.json();
    appendAssistantMessage(data);
  } catch (err) {
    showError(err.message || "Falha ao conectar com a API. Verifique se o backend está em execução.");
  } finally {
    setLoading(false);
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  questionInput.value = "";
  sendQuestion(question);
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

checkHealth();
