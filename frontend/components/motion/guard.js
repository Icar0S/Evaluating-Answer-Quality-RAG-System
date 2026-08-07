// Fonte única de verdade para "o movimento está ligado?".
//
// Carregado como script bloqueante no <head> das duas páginas (a homepage React
// e o chat vanilla) para que <html data-motion> já esteja definido antes do
// primeiro paint — senão a interface pisca a animação e só depois a desliga.
//
// Desliga o movimento quando qualquer uma destas for verdadeira:
//   1. ?motion=off na URL              (E2E, depuração pontual)
//   2. localStorage.motion === 'off'   (preferência persistente do usuário)
//   3. VITE_DISABLE_MOTION no build    (injetado como window.__MOTION_BUILD_OFF__)
//   4. prefers-reduced-motion: reduce  (preferência do sistema operacional)
//
// ?motion=on força ligado e vence tudo, inclusive a preferência do SO — existe
// para conseguir inspecionar o efeito numa máquina configurada para reduzir
// movimento, e não deve ser usado em nada automatizado.
(function () {
  "use strict";

  function readParam() {
    try {
      return new URLSearchParams(window.location.search).get("motion");
    } catch {
      return null;
    }
  }

  function readStorage() {
    try {
      return localStorage.getItem("motion");
    } catch {
      // localStorage pode lançar em contexto sem permissão (file://, modo restrito)
      return null;
    }
  }

  function prefersReduced() {
    return typeof window.matchMedia === "function" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function resolve() {
    const param = readParam();
    if (param === "on") return false;
    if (param === "off") return true;
    if (readStorage() === "off") return true;
    if (window.__MOTION_BUILD_OFF__ === true) return true;
    return prefersReduced();
  }

  const off = resolve();

  window.__MOTION_OFF__ = off;
  document.documentElement.dataset.motion = off ? "off" : "on";
})();
