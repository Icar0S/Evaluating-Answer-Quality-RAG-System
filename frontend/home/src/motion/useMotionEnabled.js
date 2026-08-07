import { useEffect, useState } from "react";

// Lê a decisão já tomada por /components/motion/guard.js (script bloqueante no
// <head>) e continua ouvindo a preferência do SO, para o caso de ela mudar com a
// página aberta. Nenhum componente consulta a media query por conta própria.
export function useMotionEnabled() {
  const [enabled, setEnabled] = useState(() => window.__MOTION_OFF__ !== true);

  useEffect(() => {
    // ?motion=off / ?motion=on e localStorage são decisões explícitas: uma
    // mudança de preferência do SO não deve sobrescrevê-las no meio da sessão.
    const forced = new URLSearchParams(window.location.search).get("motion");
    if (forced === "on" || forced === "off") return undefined;
    if (window.__MOTION_OFF__ === true) return undefined;
    if (typeof window.matchMedia !== "function") return undefined;

    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => {
      setEnabled(!query.matches);
      document.documentElement.dataset.motion = query.matches ? "off" : "on";
    };

    query.addEventListener("change", sync);
    return () => query.removeEventListener("change", sync);
  }, []);

  return enabled;
}
