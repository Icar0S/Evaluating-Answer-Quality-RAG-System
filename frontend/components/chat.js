// Chat RAG: histórico de sessões (localStorage), bolhas com reações, filtro por fundamentação.
// Override de URL da API via window.__RAG_API_BASE_URL__ (usado pelos testes Playwright).
const API_BASE_URL = window.__RAG_API_BASE_URL__ || "http://localhost:8000";

const SESSIONS_KEY = "rag_sessions_v1";
const ACTIVE_SESSION_KEY = "rag_active_session_v1";
const REACTION_EMOJIS = ["👍", "👎", "💡", "❓"];
const MAX_QUESTION_LENGTH = 4000;
const CHAR_COUNT_WARNING_THRESHOLD = 3500;

// ---------- Persistência de sessões ----------

function loadSessions() {
  try {
    return JSON.parse(localStorage.getItem(SESSIONS_KEY)) || [];
  } catch {
    return [];
  }
}

function persistSessions(sessions) {
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
}

function getActiveSessionId() {
  return localStorage.getItem(ACTIVE_SESSION_KEY);
}

function setActiveSessionId(id) {
  localStorage.setItem(ACTIVE_SESSION_KEY, id);
}

function newSessionId() {
  return `s_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function createSession() {
  const sessions = loadSessions();
  const session = {
    id: newSessionId(),
    title: "Nova conversa",
    createdAt: Date.now(),
    starred: false,
    messages: [],
  };
  sessions.unshift(session);
  persistSessions(sessions);
  setActiveSessionId(session.id);
  return session;
}

function findSession(sessions, id) {
  return sessions.find((s) => s.id === id);
}

function ensureActiveSession() {
  let sessions = loadSessions();
  let activeId = getActiveSessionId();
  let session = activeId ? findSession(sessions, activeId) : null;
  if (!session) {
    if (sessions.length > 0) {
      session = sessions[0];
      setActiveSessionId(session.id);
    } else {
      session = createSession();
    }
  }
  return session;
}

// ---------- DOM refs ----------

const sessionListEl = document.getElementById("session-list");
const sessionSearchEl = document.getElementById("session-search");
const sidebarTabs = document.querySelectorAll(".sidebar-tabs .tab");
const newSessionBtn = document.getElementById("new-session-btn");
const sessionTitleEl = document.getElementById("session-title");
const chatLogEl = document.getElementById("chat-log");
const chatFormEl = document.getElementById("chat-form");
const questionInputEl = document.getElementById("question-input");
const sendButtonEl = document.getElementById("send-button");
const errorBannerEl = document.getElementById("error-banner");
const railStatusDot = document.getElementById("rail-status-dot");
const msgFilterTabs = document.querySelectorAll("#msg-filter-tabs .header-tab");
const railButtons = document.querySelectorAll(".rail-btn[data-sidebar]");
const sidebarSessionsEl = document.getElementById("sidebar-sessions");
const sidebarMonitorEl = document.getElementById("sidebar-monitor");
const hintModelEl = document.getElementById("hint-model");
const hintCharCountEl = document.getElementById("hint-char-count");

let sidebarFilter = "all"; // all | starred

// ---------- Sessão / lista lateral ----------

function relativeTime(ms) {
  const diffMin = Math.round((Date.now() - ms) / 60000);
  if (diffMin < 1) return "agora";
  if (diffMin < 60) return `${diffMin}min`;
  const diffH = Math.round(diffMin / 60);
  if (diffH < 24) return `${diffH}h`;
  return `${Math.round(diffH / 24)}d`;
}

function initials(title) {
  return title
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0].toUpperCase())
    .join("") || "?";
}

function renderSessionList() {
  const sessions = loadSessions();
  const query = (sessionSearchEl.value || "").trim().toLowerCase();
  const activeId = getActiveSessionId();

  const filtered = sessions.filter((s) => {
    if (sidebarFilter === "starred" && !s.starred) return false;
    if (query && !s.title.toLowerCase().includes(query)) return false;
    return true;
  });

  sessionListEl.innerHTML = "";

  if (filtered.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-sessions";
    empty.textContent = "Nenhuma conversa encontrada.";
    sessionListEl.appendChild(empty);
    return;
  }

  filtered.forEach((session) => {
    const item = document.createElement("div");
    item.className = "session-item" + (session.id === activeId ? " active" : "");

    const avatar = document.createElement("div");
    avatar.className = "session-avatar";
    avatar.textContent = initials(session.title);

    const info = document.createElement("div");
    info.className = "session-info";

    const titleRow = document.createElement("div");
    titleRow.className = "session-title-row";
    const titleSpan = document.createElement("span");
    titleSpan.className = "title";
    titleSpan.textContent = session.title;
    const timeSpan = document.createElement("span");
    timeSpan.className = "time";
    timeSpan.textContent = relativeTime(session.createdAt);
    titleRow.append(titleSpan, timeSpan);

    const snippet = document.createElement("div");
    snippet.className = "session-snippet";
    const lastMsg = session.messages[session.messages.length - 1];
    snippet.textContent = lastMsg ? lastMsg.text.slice(0, 60) : "Sem mensagens ainda";

    info.append(titleRow, snippet);

    const starBtn = document.createElement("button");
    starBtn.className = "star-btn" + (session.starred ? " starred" : "");
    starBtn.textContent = session.starred ? "★" : "☆";
    starBtn.title = "Favoritar conversa";
    starBtn.addEventListener("click", (evt) => {
      evt.stopPropagation();
      toggleStar(session.id);
    });

    item.append(avatar, info, starBtn);
    item.addEventListener("click", () => selectSession(session.id));
    sessionListEl.appendChild(item);
  });
}

function toggleStar(id) {
  const sessions = loadSessions();
  const session = findSession(sessions, id);
  if (!session) return;
  session.starred = !session.starred;
  persistSessions(sessions);
  renderSessionList();
}

function selectSession(id) {
  setActiveSessionId(id);
  renderSessionList();
  renderActiveSession();
}

function renderActiveSession() {
  const session = ensureActiveSession();
  sessionTitleEl.textContent = session.title;
  renderChatLog(session);
}

// ---------- Chat log ----------

function formatDateLabel(ms) {
  const d = new Date(ms);
  const today = new Date();
  const isSameDay = (a, b) =>
    a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
  if (isSameDay(d, today)) return "Hoje";
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  if (isSameDay(d, yesterday)) return "Ontem";
  return d.toLocaleDateString("pt-BR", { day: "2-digit", month: "long" });
}

function formatTime(ms) {
  return new Date(ms).toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });
}

function buildReactionBar(message) {
  const bar = document.createElement("div");
  bar.className = "reaction-bar";
  REACTION_EMOJIS.forEach((emoji) => {
    const btn = document.createElement("button");
    btn.className = "reaction-btn" + (message.reactions?.includes(emoji) ? " active" : "");
    btn.textContent = emoji;
    btn.addEventListener("click", () => toggleReaction(message.id, emoji));
    bar.appendChild(btn);
  });
  return bar;
}

function toggleReaction(messageId, emoji) {
  const sessions = loadSessions();
  const session = findSession(sessions, getActiveSessionId());
  if (!session) return;
  const message = session.messages.find((m) => m.id === messageId);
  if (!message) return;
  message.reactions = message.reactions || [];
  const idx = message.reactions.indexOf(emoji);
  if (idx >= 0) message.reactions.splice(idx, 1);
  else message.reactions.push(emoji);
  persistSessions(sessions);
  renderChatLog(session);
}

function buildMessageRow(message) {
  const row = document.createElement("div");
  row.className = `message-row ${message.role}` + (message.role === "assistant" && !message.grounded ? " ungrounded" : "");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = message.text;
  row.appendChild(bubble);

  if (message.role === "assistant") {
    if (message.sources && message.sources.length > 0) {
      const details = document.createElement("details");
      details.className = "sources";
      const summary = document.createElement("summary");
      summary.textContent = `Fontes (${message.sources.length})`;
      details.appendChild(summary);
      message.sources.forEach((source) => {
        const chip = document.createElement("div");
        chip.className = "source-chip";
        chip.textContent = `${source.document} (pág. ${source.page ?? "?"}) — similaridade ${source.similarity_score.toFixed(3)}`;
        details.appendChild(chip);
      });
      row.appendChild(details);
    }

    if (message.metadata) {
      const meta = document.createElement("div");
      meta.className = "msg-meta";
      meta.textContent = `${message.metadata.model} · ${Math.round(message.metadata.latency_ms)}ms`;
      row.appendChild(meta);
    }

    row.appendChild(buildReactionBar(message));

    if (message.reactions && message.reactions.length > 0) {
      const display = document.createElement("div");
      display.className = "reactions-display";
      message.reactions.forEach((emoji) => {
        const badge = document.createElement("span");
        badge.className = "reaction-badge";
        badge.textContent = emoji;
        display.appendChild(badge);
      });
      row.appendChild(display);
    }
  }

  const timestamp = document.createElement("span");
  timestamp.className = "timestamp";
  timestamp.textContent = formatTime(message.timestamp);
  row.appendChild(timestamp);

  return row;
}

function renderChatLog(session) {
  chatLogEl.innerHTML = "";
  let lastDateLabel = null;

  session.messages.forEach((message) => {
    const dateLabel = formatDateLabel(message.timestamp);
    if (dateLabel !== lastDateLabel) {
      const sep = document.createElement("div");
      sep.className = "date-separator";
      sep.innerHTML = `<span>${dateLabel}</span>`;
      chatLogEl.appendChild(sep);
      lastDateLabel = dateLabel;
    }
    chatLogEl.appendChild(buildMessageRow(message));
  });

  chatLogEl.scrollTop = chatLogEl.scrollHeight;
}

function showLoadingRow() {
  const row = document.createElement("div");
  row.className = "loading-row";
  row.id = "loading-row";
  row.innerHTML = '<span class="loading-dot"></span><span class="loading-dot"></span><span class="loading-dot"></span>';
  chatLogEl.appendChild(row);
  chatLogEl.scrollTop = chatLogEl.scrollHeight;
}

function removeLoadingRow() {
  document.getElementById("loading-row")?.remove();
}

function showError(message) {
  errorBannerEl.textContent = message;
  errorBannerEl.hidden = false;
}

function clearError() {
  errorBannerEl.hidden = true;
  errorBannerEl.textContent = "";
}

function setSending(isSending) {
  questionInputEl.disabled = isSending;
  if (isSending) {
    sendButtonEl.disabled = true;
  } else {
    updateComposerState();
    questionInputEl.focus();
  }
}

// ---------- Envio de pergunta ----------

async function sendQuestion(question) {
  clearError();
  const sessions = loadSessions();
  const session = findSession(sessions, getActiveSessionId());
  if (!session) return;

  const userMessage = { id: `m_${Date.now()}`, role: "user", text: question, timestamp: Date.now() };
  session.messages.push(userMessage);

  if (session.title === "Nova conversa") {
    session.title = question.slice(0, 42) + (question.length > 42 ? "…" : "");
  }

  persistSessions(sessions);
  sessionTitleEl.textContent = session.title;
  renderSessionList();
  renderChatLog(session);
  setSending(true);
  showLoadingRow();

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
    const assistantMessage = {
      id: `m_${Date.now()}_a`,
      role: "assistant",
      text: data.answer,
      timestamp: Date.now(),
      grounded: data.grounded,
      sources: data.sources,
      metadata: data.metadata,
      reactions: [],
    };

    const freshSessions = loadSessions();
    const freshSession = findSession(freshSessions, session.id);
    freshSession.messages.push(assistantMessage);
    persistSessions(freshSessions);
    renderChatLog(freshSession);
    renderSessionList();
  } catch (err) {
    showError(err.message || "Falha ao conectar com a API. Verifique se o backend está em execução.");
  } finally {
    removeLoadingRow();
    setSending(false);
  }
}

// ---------- Health / status ----------

async function checkHealth() {
  try {
    const resp = await fetch(`${API_BASE_URL}/health`);
    if (!resp.ok) throw new Error("unhealthy");
    const data = await resp.json();
    railStatusDot.classList.toggle("ok", data.ollama_reachable);
    railStatusDot.classList.toggle("down", !data.ollama_reachable);
    railStatusDot.title = data.ollama_reachable ? "API e Ollama disponíveis" : "Ollama indisponível";
    hintModelEl.textContent = data.generation_model;
  } catch {
    railStatusDot.classList.remove("ok");
    railStatusDot.classList.add("down");
    railStatusDot.title = "API indisponível";
    hintModelEl.textContent = "backend indisponível";
  }
}

// ---------- Eventos ----------

chatFormEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInputEl.value.trim();
  if (!question) return;
  questionInputEl.value = "";
  questionInputEl.style.height = "auto";
  updateComposerState();
  sendQuestion(question);
});

questionInputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatFormEl.requestSubmit();
  }
});

function updateComposerState() {
  questionInputEl.style.height = "auto";
  questionInputEl.style.height = `${Math.min(questionInputEl.scrollHeight, 140)}px`;

  const length = questionInputEl.value.length;
  sendButtonEl.disabled = questionInputEl.value.trim().length === 0;

  if (length >= CHAR_COUNT_WARNING_THRESHOLD) {
    hintCharCountEl.hidden = false;
    hintCharCountEl.textContent = `${length}/${MAX_QUESTION_LENGTH}`;
  } else {
    hintCharCountEl.hidden = true;
  }
}

questionInputEl.addEventListener("input", updateComposerState);

newSessionBtn.addEventListener("click", () => {
  const session = createSession();
  renderSessionList();
  sessionTitleEl.textContent = session.title;
  renderChatLog(session);
  questionInputEl.focus();
});

sessionSearchEl.addEventListener("input", renderSessionList);

sidebarTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    sidebarTabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    sidebarFilter = tab.dataset.filter;
    renderSessionList();
  });
});

msgFilterTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    msgFilterTabs.forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    chatLogEl.dataset.msgFilter = tab.dataset.msgFilter;
  });
});

railButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    railButtons.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const target = btn.dataset.sidebar;
    sidebarSessionsEl.hidden = target !== "sessions";
    sidebarMonitorEl.hidden = target !== "monitor";
  });
});

// ---------- Inicialização ----------

ensureActiveSession();
renderSessionList();
renderActiveSession();
checkHealth();
setInterval(checkHealth, 15000);
