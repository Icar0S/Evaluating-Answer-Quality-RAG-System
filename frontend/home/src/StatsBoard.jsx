import { useEffect, useState } from "react";
import AnimatedContent from "./motion/AnimatedContent";
import SplitFlapText from "./motion/SplitFlapText";
import { formatStat, getJson } from "./api";

// Placeholder do painel enquanto a leitura não chegou. Tracinhos, não zeros:
// zero é um valor, e alegar um valor que não foi lido é exatamente o que esta
// página existe para não fazer.
const PENDING = "-----";
const OFFLINE = "OFF";

/** Devolve o par (texto acessível, texto do painel) para um valor lido. */
function reading(raw, offline, kind) {
  if (offline) return { plain: "offline", flap: OFFLINE };
  const formatted = formatStat(raw, kind);
  if (formatted === null) return { plain: "—", flap: PENDING };
  return { plain: formatted, flap: formatted };
}

// O painel mecânico é decorativo do ponto de vista de acessibilidade: quem
// carrega o valor é o .sr-only, com o rótulo junto para virar uma frase só
// ("536 chunks indexados") em vez de cinco caracteres soltos.
function StatCard({ id, label, plain, flap }) {
  return (
    <div className="stat-card" data-spotlight>
      <p className="sr-only" aria-live="polite">
        <span id={id}>{plain}</span> {label}
      </p>
      <SplitFlapText value={flap} width={5} />
      <span className="stat-label" aria-hidden="true">
        {label}
      </span>
    </div>
  );
}

export default function StatsBoard() {
  const [stats, setStats] = useState(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let alive = true;
    getJson("/stats")
      .then((data) => alive && setStats(data))
      .catch(() => alive && setOffline(true));
    return () => {
      alive = false;
    };
  }, []);

  const documents = reading(stats?.vector_store_documents, offline, "count");
  const interactions = reading(stats?.total_interactions, offline, "count");
  const latency = reading(stats?.average_latency_ms, offline, "latency");
  const grounded = reading(stats?.grounded_rate, offline, "percent");
  const deepevalFaithfulness = reading(stats?.deepeval_faithfulness_avg, offline, "percent");

  return (
    <AnimatedContent>
      <div className="stats-grid" id="stats-grid">
        <StatCard id="stat-documents" label="chunks indexados" {...documents} />
        <StatCard id="stat-interactions" label="interações registradas" {...interactions} />
        <StatCard id="stat-latency" label="latência média de resposta" {...latency} />
        <StatCard id="stat-grounded" label="respostas fundamentadas no contexto" {...grounded} />
      </div>

      {/* Fora da grade de propósito: a grade é 4/2/1 colunas pra nunca deixar
          card órfão numa linha (ver home.css). Uma 5ª métrica quebraria isso
          em qualquer breakpoint, então fica como nota — não por ser menos
          real que as outras, só por não caber no ritmo 4/2/1. */}
      <p className="stats-pending-note">
        <span className="stat-value-pending" id="stat-deepeval">
          {deepevalFaithfulness.plain === "—" ? "em breve" : deepevalFaithfulness.plain}
        </span>
        <span>fidelidade média (DeepEval, Fase 2)</span>
      </p>
    </AnimatedContent>
  );
}
