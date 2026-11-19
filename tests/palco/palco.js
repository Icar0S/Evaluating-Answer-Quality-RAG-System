"use strict";
// O palco: lê /estado a cada segundo e o log do passo atual a cada 400 ms (tests/apresentacao.py). Tudo vira
// DOM por textContent: nada do que vem dos logs, dos goldens ou dos modelos é interpretado como HTML.

const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...filhos) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else n.setAttribute(k, v);
  }
  for (const f of filhos) if (f !== null && f !== undefined) n.append(f);
  return n;
}

const ICONE = {
  passou: "✓", "concluída": "✓", falhou: "✗", erro: "✗", "com falhas": "✗",
  pulada: "⊘", "em andamento": "●", pendente: "○",
};
const ORDEM_METRICAS = [
  "faithfulness", "answer_relevancy", "context_precision", "context_recall", "context_relevance", "correctness",
];

const num = (v, casas = 2) =>
  v === null || v === undefined
    ? "—"
    : v.toLocaleString("pt-BR", { minimumFractionDigits: casas, maximumFractionDigits: casas });

function relogio(s) {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const x = s % 60;
  return (h ? String(h).padStart(2, "0") + ":" : "") + String(m).padStart(2, "0") + ":" + String(x).padStart(2, "0");
}

function duracao(s) {
  if (s === null || s === undefined) return "";
  return s >= 60 ? `${Math.floor(s / 60)} min ${String(Math.floor(s % 60)).padStart(2, "0")} s` : `${Math.round(s)} s`;
}

let estado = null;
let recebidoEm = 0;
let faseMostrada = 0;
let visualMontado = "";

// ---------------------------------------------------------------- o topo e a trilha

function decorrido() {
  if (!estado) return 0;
  return estado.decorrido_s + (performance.now() - recebidoEm) / 1000;
}

function renderTopo() {
  $("titulo").textContent = estado.titulo;
  const ctx = $("contexto");
  ctx.replaceChildren(
    ...[
      ["gerador", estado.modelos.gerador],
      ["juiz", estado.modelos.juiz],
      ["RAGs", estado.rags.map((r) => r.index).join(" · ")],
    ].map(([dt, dd]) => el("div", {}, el("dt", { text: dt }), el("dd", { text: dd }))),
  );
}

function renderFases() {
  const lista = $("fases");
  lista.replaceChildren(
    ...estado.fases.map((fase) => {
      const passos = fase.status === "pendente" ? [] : fase.passos.slice(-7);
      return el(
        "li",
        { class: "fase", "data-status": fase.status },
        el("span", { class: "fase-n", text: fase.status === "concluída" ? "✓" : String(fase.n) }),
        el("p", { class: "fase-nome", text: fase.nome }),
        passos.length
          ? el(
              "ol",
              { class: "passos" },
              ...passos.map((p) =>
                el(
                  "li",
                  { "data-status": p.status, title: p.detalhe || "" },
                  el("span", { class: "icone", text: ICONE[p.status] ?? "•" }),
                  el("span", { class: "p-nome", text: p.nome }),
                  el("span", { class: "p-seg", text: p.segundos === null ? "" : duracao(p.segundos) }),
                ),
              ),
            )
          : null,
      );
    }),
  );
}

function renderPasso() {
  const passo = estado.passo;
  const fase = estado.fases.find((f) => f.n === (passo ? passo.fase : estado.fase_atual));
  $("passo-fase").textContent = fase ? `Fase ${fase.n} de ${estado.fases.length} · ${fase.nome}` : "";
  $("passo-nome").textContent = passo ? passo.nome : "";
  $("passo-comando").textContent = passo && passo.comando ? `$ ${passo.comando}` : "";
  const aviso = (estado.dados || {}).aviso;
  $("aviso").hidden = !aviso;
  $("aviso").textContent = aviso || "";
}

function tique() {
  if (!estado) return;
  const agora = decorrido();
  $("relogio").textContent = relogio(agora);
  const passo = estado.passo;
  $("passo-tempo").textContent = passo && !estado.fim ? `há ${duracao(agora - passo.desde)}` : "";
}

// ---------------------------------------------------------------- o cartão de cada fase

let cartaoTimer = null;

function mostrarFase(n) {
  const fase = estado.fases.find((f) => f.n === n);
  if (!fase) return;
  $("cartao-n").textContent = `Fase ${fase.n} de ${estado.fases.length}`;
  $("cartao-nome").textContent = fase.nome;
  $("cartao-desc").textContent = fase.descricao;
  const cartao = $("cartao-fase");
  cartao.classList.add("visivel");
  clearTimeout(cartaoTimer);
  cartaoTimer = setTimeout(() => cartao.classList.remove("visivel"), 2800);
}

// ---------------------------------------------------------------- os modos da cena

function montarAbertura(visual) {
  visual.replaceChildren(
    el(
      "div",
      { class: "abertura" },
      el("h2", { text: "Do docker compose ao relatório, sem tocar no teclado" }),
      el("p", {
        text:
          "A aplicação sobe, os testes locais rodam, o Playwright usa o chat como uma pessoa e três " +
          `frameworks de qualidade (${estado.avaliadores.join(", ")}) julgam as mesmas respostas.`,
      }),
      el(
        "ol",
        {},
        ...estado.fases.map((f) =>
          el("li", {}, el("span", { text: String(f.n) }), el("div", {}, f.nome, el("small", { text: f.descricao }))),
        ),
      ),
    ),
  );
}

function montarApp(visual, dados) {
  const quadro = dados.quadro || {};
  const blocoRags = el(
    "div",
    {},
    el("h3", { text: "Os dois RAGs" }),
    ...(quadro.rags || []).map((r) =>
      el(
        "div",
        { class: "rag-linha" },
        el("strong", { text: r.index || "?" }),
        el("div", { class: "trechos" }, String(r.trechos ?? "—"), el("small", { text: "trechos" })),
        el("span", { text: `${r.label} · ${r.api}` }),
      ),
    ),
  );
  const blocoModelos = el(
    "div",
    {},
    el("h3", { text: "Modelos no Ollama" }),
    el(
      "ul",
      { class: "modelos" },
      ...(quadro.modelos || []).map((m) =>
        el(
          "li",
          { "data-status": m.ok ? "passou" : "erro" },
          el("span", { class: "icone", text: m.ok ? "✓" : "✗" }),
          el("span", { class: "papel", text: m.papel }),
          el("code", { text: m.modelo }),
        ),
      ),
    ),
  );
  const avisos = quadro.avisos && quadro.avisos.length
    ? el("div", {}, el("h3", { text: "Avisos" }), ...quadro.avisos.map((a) => el("p", { class: "aviso", text: a })))
    : null;
  visual.replaceChildren(
    el(
      "div",
      { class: "app" },
      el("div", { class: "quadro" }, blocoRags, blocoModelos, avisos),
      el("div", { class: "tela-app" }, el("iframe", { src: estado.app_url, title: "A aplicação no ar" })),
    ),
  );
}

const cartoes = new Map();

function montarPlacar(visual) {
  cartoes.clear();
  const limiarGeral = num(0.7);
  visual.replaceChildren(
    el(
      "div",
      { class: "placar" },
      el(
        "p",
        { class: "legenda" },
        el("span", {}, el("span", { class: "amostra" }), "barra: a média das notas até aqui; traço: o limiar"),
        el("span", { text: `limiar ${limiarGeral} (na correção do JevEval, ${num(0.5)})` }),
        el("span", { text: "✓ média no limiar ou acima · ✗ abaixo" }),
      ),
      el("div", { class: "cartoes", id: "cartoes" }),
    ),
  );
}

function cartaoDe(p) {
  const chave = `${p.framework}|${p.rag}`;
  if (cartoes.has(chave)) return cartoes.get(chave);
  const rag = estado.rags.find((r) => r.index === p.rag);
  const progresso = el("span", { class: "progresso" });
  const medidores = el("ul", { class: "medidores" });
  const caso = el("p", { class: "caso-atual" });
  const raiz = el(
    "article",
    { class: "cartao" },
    el(
      "header",
      {},
      el("h3", { text: p.framework }),
      el("span", { class: "rag", text: rag ? `${p.rag} · ${rag.label}` : p.rag }),
      progresso,
    ),
    medidores,
    caso,
  );
  const linhas = new Map();
  for (const chaveM of ORDEM_METRICAS) {
    if (!p.metricas[chaveM]) continue;
    const barra = el("span", { class: "m-barra" });
    const limiar = el("span", { class: "m-limiar" });
    const valor = el("span", { class: "m-valor" });
    const icone = el("span", { class: "m-icone" });
    const linha = el(
      "li",
      { class: "medidor" },
      el("span", { class: "m-nome", text: p.metricas[chaveM].rotulo }),
      el("span", { class: "m-trilho" }, barra, limiar),
      valor,
      icone,
    );
    medidores.append(linha);
    linhas.set(chaveM, { linha, barra, limiar, valor, icone });
  }
  const cartao = { raiz, progresso, caso, linhas };
  cartoes.set(chave, cartao);
  return cartao;
}

function atualizarPlacar() {
  const grade = $("cartoes");
  if (!grade) return;
  const placar = estado.placar || [];
  if (!placar.length) {
    grade.replaceChildren(el("p", { class: "aviso", text: "Esperando as respostas do E2E para julgar." }));
    return;
  }
  const frameworks = [...new Set(placar.map((p) => p.framework))];
  const rags = estado.rags.map((r) => r.index).filter((i) => placar.some((p) => p.rag === i));
  grade.style.gridTemplateColumns = `repeat(${frameworks.length}, minmax(0, 1fr))`;
  grade.style.gridTemplateRows = `repeat(${rags.length}, minmax(0, 1fr))`;
  const ordem = [];
  for (const rag of rags) for (const fw of frameworks) {
    const p = placar.find((x) => x.rag === rag && x.framework === fw);
    if (p) ordem.push(p);
  }
  const elementos = ordem.map((p) => {
    const c = cartaoDe(p);
    c.raiz.dataset.status = p.status;
    const total = p.casos_total ?? "?";
    c.progresso.textContent = {
      pendente: "na fila",
      pulada: "pulada",
      erro: "erro",
    }[p.status] ?? `${p.casos_feitos} de ${total} respostas`;
    c.caso.textContent = p.caso_atual ? `julgando: ${p.caso_atual}` : "";
    for (const [chaveM, m] of Object.entries(p.metricas)) {
      const l = c.linhas.get(chaveM);
      if (!l) continue;
      const estadoM = m.media === null ? (m.na ? "na" : "pendente") : m.passa ? "passa" : "abaixo";
      l.linha.dataset.estado = estadoM;
      l.barra.style.width = `${Math.max(0, Math.min(1, m.media ?? 0)) * 100}%`;
      l.limiar.style.left = `${m.limiar * 100}%`;
      l.valor.textContent = m.media === null ? (m.na ? "N/A" : "—") : num(m.media);
      l.icone.textContent = estadoM === "passa" ? "✓" : estadoM === "abaixo" ? "✗" : "";
      l.linha.title = `${m.rotulo}: ${m.n} nota(s)` + (m.na ? `, ${m.na} N/A` : "") + (m.erros ? `, ${m.erros} erro(s)` : "")
        + `; limiar ${num(m.limiar)}`;
    }
    return c.raiz;
  });
  if (elementos.length !== grade.children.length || elementos.some((e, i) => grade.children[i] !== e)) {
    grade.replaceChildren(...elementos);
  }
}

const LEGENDA_FIGURA = [
  [/^medias_por_dimensao$/, () => "A média de cada dimensão, rodada a rodada"],
  [/^historico$/, () => "A evolução das médias ao longo das execuções"],
  [/^notas_\d+_(\w+)$/, (m) => `A nota de cada pergunta · ${{ ragas: "RAGAS", deepeval: "DeepEval", jeveval: "JevEval" }[m[1]] ?? m[1]}`],
];

function legendaDe(caminho) {
  const nome = caminho.split("/").pop().replace(/\.png$/, "");
  for (const [re, f] of LEGENDA_FIGURA) {
    const m = nome.match(re);
    if (m) return f(m);
  }
  return nome;
}

let carrossel = null;

function montarRelatorio(visual, dados) {
  clearInterval(carrossel);
  const figuras = dados.figuras || [];
  const img = el("img", { alt: "" });
  const legenda = el("figcaption");
  const pontos = el("span", { class: "pontos" }, ...figuras.map(() => el("span")));
  let i = -1;
  const proxima = () => {
    if (!figuras.length) return;
    i = (i + 1) % figuras.length;
    img.src = `/arquivo/${figuras[i]}`;
    img.alt = legendaDe(figuras[i]);
    legenda.textContent = legendaDe(figuras[i]);
    [...pontos.children].forEach((p, j) => p.setAttribute("aria-current", String(j === i)));
  };
  visual.replaceChildren(
    el(
      "div",
      { class: "relatorio" },
      el("figure", {}, img, legenda),
      el("p", { class: "rodape-relatorio" }, el("span", { text: `Relatório em ${dados.pdf}` }), pontos),
    ),
  );
  proxima();
  carrossel = setInterval(proxima, (dados.segundos_por_figura || 7) * 1000);
}

function montarFim(visual, dados) {
  clearInterval(carrossel);
  const r = dados.resumo || {};
  const tile = (rotulo, valor, status) =>
    el(
      "div",
      { "data-status": status },
      el("dt", { text: rotulo }),
      el("dd", {}, status ? el("span", { class: "icone", text: ICONE[status] }) : null, String(valor)),
    );
  visual.replaceChildren(
    el(
      "div",
      { class: "fim" },
      el("h2", { text: r.falhas ? "Rotina terminada, com falhas" : "Rotina completa" }),
      el(
        "dl",
        { class: "numeros" },
        tile("passos que passaram", r.ok ?? 0, "passou"),
        tile("falharam", r.falhas ?? 0, r.falhas ? "falhou" : null),
        tile("pulados", r.pulados ?? 0, r.pulados ? "pulada" : null),
        tile("respostas julgadas", r.respostas_julgadas ?? 0),
        tile("avaliações", r.avaliacoes ?? 0),
        tile("duração", r.duracao ?? ""),
      ),
      el(
        "div",
        { class: "caminhos" },
        r.relatorio ? el("p", {}, "Relatório: ", el("code", { text: r.relatorio })) : null,
        r.video ? el("p", {}, "Vídeo: ", el("code", { text: r.video })) : null,
      ),
    ),
  );
}

// O espelho: a página do navegador do E2E, que roda sem janela e manda cada quadro ao palco (e2e/fixtures.ts).
let espelhoTimer = null;
let espelhoN = -1;
let espelhoUrl = null;
let espelhoPedindo = false;

function montarEspelho(visual) {
  espelhoN = -1;
  visual.replaceChildren(
    el(
      "figure",
      { class: "espelho" },
      el(
        "div",
        { class: "espelho-tela" },
        el("p", { class: "espelho-espera", id: "espelho-espera", text: "Esperando o primeiro teste abrir a página…" }),
        el("img", { id: "espelho-img", alt: "A página que o Playwright está testando, ao vivo", hidden: "" }),
      ),
      el(
        "figcaption",
        {},
        el("span", { class: "espelho-rotulo", text: "navegador do Playwright, ao vivo" }),
        el("span", { class: "espelho-teste", id: "espelho-teste" }),
      ),
    ),
  );
  clearInterval(espelhoTimer);
  espelhoTimer = setInterval(lerEspelho, 110);
}

async function lerEspelho() {
  if (!estado || estado.visual !== "espelho") {
    clearInterval(espelhoTimer);
    return;
  }
  if (espelhoPedindo) return;
  espelhoPedindo = true;
  try {
    const resp = await fetch("/espelho.jpg", { cache: "no-store" });
    const n = Number(resp.headers.get("X-Quadro"));
    if (resp.status !== 200 || n === espelhoN) return;
    espelhoN = n;
    const url = URL.createObjectURL(await resp.blob());
    const img = $("espelho-img");
    if (!img) {
      URL.revokeObjectURL(url);
      return;
    }
    // Troca só depois de carregar: o quadro novo nunca pisca em branco no vídeo.
    img.onload = () => {
      if (espelhoUrl) URL.revokeObjectURL(espelhoUrl);
      espelhoUrl = url;
      img.hidden = false;
      $("espelho-espera").hidden = true;
    };
    img.src = url;
  } catch {
    // a próxima leitura tenta de novo
  } finally {
    espelhoPedindo = false;
  }
}

function renderVisual() {
  const modo = estado.visual;
  const dados = estado.dados || {};
  document.body.className = `modo-${modo}`;
  // Remonta só quando o modo muda: o iframe da aplicação e o carrossel não recomeçam a cada segundo.
  const chave = `${modo}|${modo === "relatorio" ? (dados.figuras || []).join(",") : ""}`;
  const visual = $("visual");
  if (chave !== visualMontado) {
    visualMontado = chave;
    if (modo === "abertura") montarAbertura(visual);
    else if (modo === "app") montarApp(visual, dados);
    else if (modo === "espelho") montarEspelho(visual);
    else if (modo === "placar") montarPlacar(visual);
    else if (modo === "relatorio") montarRelatorio(visual, dados);
    else if (modo === "fim") montarFim(visual, dados);
    else visual.replaceChildren();
  }
  if (modo === "placar") atualizarPlacar();
  if (modo === "espelho") $("espelho-teste").textContent = (estado.espelho || {}).teste || "";
}

// ---------------------------------------------------------------- o log ao vivo

const MAX_LINHAS = 220;
let logPasso = null;
let logOffset = 0;
let logResto = "";
let logLinhas = [];
let logPedindo = false;

function classeDaLinha(linha) {
  const t = linha.trim();
  if (t.startsWith("$ ")) return "l-cmd";
  if (/\b(FAILED|ERROR|ERRO|FALHA|FAIL)\b|✘/.test(linha)) return "l-falha";
  if (/\bSKIPPED\b|^-\s{2}\d|\bPULOU\b/.test(t)) return "l-pulado";
  if (/^\[\d+\/\d+\]/.test(t)) return "l-caso";
  if (/\b(PASSED|PASS|Healthy|Started)\b|✓|\b\d+ passed\b|\[ OK/.test(linha)) return "l-ok";
  if (/^\s{2,}[A-Za-z][A-Za-z_ ]*?\s{2,}-?\d/.test(linha)) return "l-nota";
  return "";
}

function pintarLog() {
  const pre = $("log");
  pre.replaceChildren(...logLinhas.map((l) => el("span", { class: classeDaLinha(l), text: l + "\n" })));
}

async function lerLog() {
  if (logPedindo || !estado || !estado.passo) return;
  logPedindo = true;
  try {
    const chave = estado.passo.key;
    if (chave !== logPasso) {
      logPasso = chave;
      logOffset = 0;
      logResto = "";
      logLinhas = [];
      pintarLog();
    }
    const resp = await fetch(`/log?desde=${logOffset}`, { cache: "no-store" });
    const dados = await resp.json();
    if (dados.passo !== logPasso) return;
    logOffset = dados.proximo;
    if (!dados.texto) return;
    const partes = (logResto + dados.texto).replace(/\r/g, "").split("\n");
    logResto = partes.pop();
    logLinhas.push(...partes);
    if (logLinhas.length > MAX_LINHAS) logLinhas = logLinhas.slice(-MAX_LINHAS);
    pintarLog();
  } catch {
    // o orquestrador terminou ou está ocupado: a próxima leitura tenta de novo
  } finally {
    logPedindo = false;
  }
}

// ---------------------------------------------------------------- o laço

async function lerEstado() {
  try {
    const resp = await fetch("/estado", { cache: "no-store" });
    estado = await resp.json();
    recebidoEm = performance.now();
  } catch {
    return;
  }
  renderTopo();
  renderFases();
  renderPasso();
  renderVisual();
  if (estado.fase_atual > faseMostrada && !estado.fim) {
    faseMostrada = estado.fase_atual;
    mostrarFase(faseMostrada);
  }
  tique();
}

lerEstado();
setInterval(lerEstado, 1000);
setInterval(lerLog, 400);
setInterval(tique, 250);
