import { useEffect, useState } from "react";
import { getJson } from "./api";

const HUD_POLL_INTERVAL_MS = 3000;

// Barra fixa no canto: modelo carregado, tok/s e CPU, relidos a cada 3s.
// O contêiner (.hud, com pointer-events: none) vive no HTML estático — foi uma
// regressão real: sem isso o HUD cobre e bloqueia o CTA da nav.
export default function Hud() {
  const [hud, setHud] = useState({ state: "", text: "conectando ao backend..." });

  useEffect(() => {
    let alive = true;

    const poll = async () => {
      try {
        const data = await getJson("/metrics");
        if (!alive) return;
        const tps = data.last_generation?.tokens_per_second;
        const rate = tps != null ? `${tps} tok/s` : "aguardando geração";
        setHud({
          state: data.status === "processing" ? "processing" : "ok",
          text: `${data.generation_model} · ${rate} · CPU ${data.cpu_percent.toFixed(0)}%`,
        });
      } catch {
        if (!alive) return;
        setHud({ state: "down", text: "backend indisponível — inicie a API para ver dados ao vivo" });
      }
    };

    poll();
    const timer = setInterval(poll, HUD_POLL_INTERVAL_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <>
      <span className={`badge-dot ${hud.state}`.trim()} id="hud-dot" />
      <span id="hud-text">{hud.text}</span>
    </>
  );
}
