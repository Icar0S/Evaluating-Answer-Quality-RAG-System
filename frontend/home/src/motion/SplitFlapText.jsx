// SplitFlapText — React Bits (TextAnimations/SplitFlapText, variante JS-CSS),
// adaptado para o domínio deste projeto. Diferenças em relação ao original:
//
//  1. O original é um CARROSSEL: recebe `words[]` e cicla sozinho num timer.
//     Aqui não existe carrossel — existe UM valor que chega do backend e muda.
//     Trocado por uma prop `value` que dispara a virada quando muda. Isso removeu
//     scheduleNext/phraseIndex/loop/cycleDelay junto.
//  2. O original expõe o painel à árvore de acessibilidade com role="text" +
//     aria-label. role="text" não é ARIA válido, e ler 5 caracteres fatiados é
//     ruim de qualquer forma. Aqui o painel é 100% aria-hidden e quem carrega o
//     valor é um <span class="sr-only"> no componente de cima.
//  3. prefers-reduced-motion vinha de um hook interno próprio. Agora vem do guard
//     compartilhado, que também cobre ?motion=off e localStorage.
//  4. Sem animação na primeira renderização: o painel nasce no placeholder e a
//     primeira virada é a chegada do dado real. Virar no mount seria enfeite;
//     virar quando o número chega é a informação.
import { useEffect, useMemo, useRef, useState } from "react";
import "./SplitFlapText.css";
import { useMotionEnabled } from "./useMotionEnabled";

const CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.%";

const toCssUnit = (value) => (typeof value === "number" ? `${value}px` : value);

const normalize = (value, width) => String(value ?? "").padStart(width, " ").slice(-width);

const createTiles = (phrase) =>
  phrase.split("").map((char) => ({ current: char, next: char, flipping: false, tick: 0 }));

// Tile vazio ainda precisa ocupar largura: espaco comum colapsaria.
const glyph = (char) => (char === " " ? " " : char);

const sampleChar = () => CHARSET.charAt(Math.floor(Math.random() * CHARSET.length));

const buildSequence = (target, flips) => {
  const steps = [];
  for (let i = 0; i < flips; i += 1) steps.push(sampleChar());
  steps.push(target);
  return steps;
};

const SplitFlapText = ({
  value,
  width = 5,
  // 6 viradas x 55ms + 50ms de stagger sobre 5 tiles = ~530ms no pior caso.
  // Acima de 800ms o usuário lê como travamento, não como mecanismo.
  flipDuration = 0.055,
  flipsPerChar = 6,
  stagger = 0.05,
  // Tile no fundo mais escuro e glifo no âmbar claro: o gradiente do tile
  // clareia o topo, e é lá que o contraste tem de fechar 4,5:1.
  tileColor = "var(--bg-0)",
  textColor = "var(--cite-400)",
  tileRadius = 4,
  gap = 3,
  fontSize = "1.9rem",
  className = "",
}) => {
  const motionEnabled = useMotionEnabled();
  const rafRef = useRef(null);
  const currentRef = useRef(null);
  const mountedRef = useRef(false);

  const target = useMemo(() => normalize(value, width), [value, width]);
  const [tiles, setTiles] = useState(() => createTiles(target));

  useEffect(() => {
    const stop = () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };

    stop();

    const settle = () => {
      currentRef.current = target;
      setTiles(createTiles(target));
    };

    // Primeira renderização, movimento desligado, ou nada mudou: assenta direto.
    if (!mountedRef.current || !motionEnabled || currentRef.current === target) {
      mountedRef.current = true;
      settle();
      return stop;
    }

    const from = normalize(currentRef.current, width);
    const flipMs = Math.max(40, flipDuration * 1000);
    const staggerMs = Math.max(0, stagger * 1000);
    const flips = Math.max(0, Math.floor(flipsPerChar));

    const plans = target
      .split("")
      .map((targetChar, index) => {
        if (from[index] === targetChar) return null;
        return {
          index,
          from: from[index] || " ",
          target: targetChar,
          sequence: buildSequence(targetChar, flips),
          start: index * staggerMs,
          step: -1,
          done: false,
        };
      })
      .filter(Boolean);

    if (plans.length === 0) {
      settle();
      return stop;
    }

    let cancelled = false;
    const startedAt = performance.now();

    const tick = (now) => {
      if (cancelled) return;

      const elapsed = now - startedAt;
      const updates = [];
      let running = false;

      plans.forEach((plan) => {
        const local = elapsed - plan.start;
        if (local < 0) {
          running = true;
          return;
        }

        const step = Math.floor(local / flipMs);

        if (step < plan.sequence.length) {
          running = true;
          if (step !== plan.step) {
            plan.step = step;
            updates.push({
              index: plan.index,
              current: step === 0 ? plan.from : plan.sequence[step - 1],
              next: plan.sequence[step],
              done: false,
            });
          }
        } else if (!plan.done) {
          plan.done = true;
          updates.push({ index: plan.index, current: plan.target, next: plan.target, done: true });
        }
      });

      if (updates.length > 0) {
        setTiles((previous) => {
          const nextTiles = [...previous];
          updates.forEach((update) => {
            const tile = nextTiles[update.index];
            if (!tile) return;
            nextTiles[update.index] = {
              current: update.current,
              next: update.next,
              flipping: !update.done,
              tick: tile.tick + 1,
            };
          });
          return nextTiles;
        });
      }

      if (running) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        currentRef.current = target;
        rafRef.current = null;
      }
    };

    rafRef.current = requestAnimationFrame(tick);

    return () => {
      cancelled = true;
      stop();
    };
  }, [target, width, motionEnabled, flipDuration, flipsPerChar, stagger]);

  const style = {
    "--split-flap-tile-color": tileColor,
    "--split-flap-text-color": textColor,
    "--split-flap-radius": toCssUnit(tileRadius),
    "--split-flap-gap": toCssUnit(gap),
    "--split-flap-font-size": toCssUnit(fontSize),
    "--split-flap-flip-duration": `${Math.max(0.04, flipDuration)}s`,
  };

  return (
    <div className={`split-flap-text ${className}`.trim()} style={style} aria-hidden="true">
      {tiles.map((tile, index) => (
        <span className="split-flap-text__tile" key={`tile-${index}`}>
          <span className="split-flap-text__half split-flap-text__half--top">
            <span className="split-flap-text__char">{glyph(tile.current)}</span>
          </span>
          <span className="split-flap-text__half split-flap-text__half--bottom">
            <span className="split-flap-text__char">{glyph(tile.flipping ? tile.next : tile.current)}</span>
          </span>

          {tile.flipping && (
            <>
              <span className="split-flap-text__flap split-flap-text__flap--front" key={`front-${tile.tick}`}>
                <span className="split-flap-text__char">{glyph(tile.current)}</span>
              </span>
              <span className="split-flap-text__flap split-flap-text__flap--back" key={`back-${tile.tick}`}>
                <span className="split-flap-text__char">{glyph(tile.next)}</span>
              </span>
            </>
          )}
        </span>
      ))}
    </div>
  );
};

export default SplitFlapText;
