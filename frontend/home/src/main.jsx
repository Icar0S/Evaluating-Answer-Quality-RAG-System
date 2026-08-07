import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import Background from "./Background";
import Hud from "./Hud";
import StatsBoard from "./StatsBoard";

// VITE_DISABLE_MOTION=true no build desliga o movimento de forma permanente no
// artefato gerado. O guard (carregado antes deste módulo) lê esta flag.
if (import.meta.env.VITE_DISABLE_MOTION === "true") {
  window.__MOTION_BUILD_OFF__ = true;
  window.__MOTION_OFF__ = true;
  document.documentElement.dataset.motion = "off";
}

// Três ilhas, não uma aplicação. Todo o resto da página é HTML estático.
const mount = (id, node) => {
  const host = document.getElementById(id);
  if (host) createRoot(host).render(<StrictMode>{node}</StrictMode>);
};

mount("bg-root", <Background />);
mount("hud-root", <Hud />);
mount("stats-root", <StatsBoard />);
