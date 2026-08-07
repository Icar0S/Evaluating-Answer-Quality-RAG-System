// Efeitos de superfície do React Bits, portados para vanilla e dirigidos por
// atributo em vez de por componente:
//
//   [data-spotlight]  SpotlightCard — halo radial seguindo o cursor
//   [data-magnet]     Magnet        — elemento atraído ao cursor, volta sozinho
//   (GlareHover é 100% CSS — ver styles/home.css, .glare-hover)
//
// Por que vanilla e não os componentes React: a homepage é uma ilha. Nav, hero,
// narrativa, stack e rodapé são HTML estático justamente para o LCP não depender
// do JS. Envolver esses blocos em componentes React desfaria isso. Uma única
// implementação por delegação de evento atende os dois mundos — o HTML estático
// e os cards que o React renderiza dentro de #stats-root — sem duplicar código e
// sem somar nada ao bundle.
//
// Tudo aqui respeita o guard: com movimento desligado, nenhum listener é criado.
(function () {
  "use strict";

  const MAGNET_PADDING = 90; // distância em que o ímã começa a puxar
  const MAGNET_STRENGTH = 3; // divisor do deslocamento: maior = mais sutil

  /** SpotlightCard: só escreve as coordenadas; quem desenha é o CSS. */
  function trackSpotlight(event) {
    const card = event.target.closest("[data-spotlight]");
    if (!card) return;

    const rect = card.getBoundingClientRect();
    card.style.setProperty("--mouse-x", `${event.clientX - rect.left}px`);
    card.style.setProperty("--mouse-y", `${event.clientY - rect.top}px`);
  }

  /** Magnet: um único mousemove para todos os alvos, em vez de um por elemento. */
  function trackMagnets(magnets, event) {
    magnets.forEach((magnet) => {
      const { left, top, width, height } = magnet.getBoundingClientRect();
      const centerX = left + width / 2;
      const centerY = top + height / 2;
      const withinX = Math.abs(centerX - event.clientX) < width / 2 + MAGNET_PADDING;
      const withinY = Math.abs(centerY - event.clientY) < height / 2 + MAGNET_PADDING;

      if (withinX && withinY) {
        const offsetX = (event.clientX - centerX) / MAGNET_STRENGTH;
        const offsetY = (event.clientY - centerY) / MAGNET_STRENGTH;
        magnet.style.transform = `translate3d(${offsetX}px, ${offsetY}px, 0)`;
        magnet.dataset.magnetActive = "true";
      } else if (magnet.dataset.magnetActive === "true") {
        magnet.style.transform = "";
        delete magnet.dataset.magnetActive;
      }
    });
  }

  function init() {
    if (window.__MOTION_OFF__ === true) return;

    // Ponteiro grosso (toque) não tem hover: o ímã só atrapalharia o alvo do
    // dedo, e o spotlight nunca apareceria.
    if (!window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;

    document.addEventListener("mousemove", trackSpotlight, { passive: true });

    const magnets = Array.from(document.querySelectorAll("[data-magnet]"));
    if (magnets.length > 0) {
      document.addEventListener("mousemove", (event) => trackMagnets(magnets, event), { passive: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
