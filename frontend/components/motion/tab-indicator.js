// Indicador deslizante das abas do chat.
//
// Vale para os dois grupos: #provider-tabs (Local ↔ Servidor) e #msg-filter-tabs
// (Todas / Fundamentadas / Sem contexto), além das abas da sidebar.
//
// É a alternativa listada no plano para o GooeyNav do React Bits: o filtro SVG
// de "goo" precisa de uma camada isolada só para os blobs, e sobre a pílula
// translúcida escura destes controles o resultado ficava turvo em vez de líquido.
// A sensação de peso vem do overshoot da curva de transição (definida em
// styles/motion.css), não de um filtro.
(function () {
  "use strict";

  const GROUPS = [".header-tabs", ".sidebar-tabs"];

  /** Reposiciona a pílula sob a aba ativa do grupo. */
  function position(group) {
    const active = group.querySelector(".header-tab.active, .tab.active");

    if (!active) {
      group.style.setProperty("--tab-ready", "0");
      return;
    }

    // offsetLeft é relativo ao contêiner posicionado, que é o próprio grupo.
    group.style.setProperty("--tab-w", `${active.offsetWidth}px`);
    group.style.setProperty("--tab-x", `${active.offsetLeft}px`);
    group.style.setProperty("--tab-ready", "1");
    group.dataset.indicator = "on";
  }

  function watch(group) {
    position(group);

    // As abas de provider são recriadas a cada polling (~15s) e ao trocar de
    // provider; o filtro de mensagens só muda de classe. Um observer cobre os
    // dois casos sem precisar interceptar cliques.
    const observer = new MutationObserver(() => position(group));
    observer.observe(group, { childList: true, subtree: true, attributes: true, attributeFilter: ["class"] });

    return observer;
  }

  function init() {
    const groups = GROUPS.flatMap((selector) => Array.from(document.querySelectorAll(selector)));
    groups.forEach(watch);

    // A largura das abas muda com a da janela (e a sidebar tem largura fixa, mas
    // o painel principal não).
    window.addEventListener("resize", () => groups.forEach(position));

    // As faces só chegam depois do primeiro paint; a largura do texto muda com
    // elas e o indicador precisa remedir.
    if (document.fonts?.ready) {
      document.fonts.ready.then(() => groups.forEach(position));
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
