// AnimatedContent — React Bits (Animations/AnimatedContent, variante JS-CSS).
// Mesma API e mesmo comportamento visual do original; o motor é que mudou.
//
// Por que o motor mudou: o original anima via GSAP + ScrollTrigger. Medido neste
// projeto, gsap + ScrollTrigger custam 46,2 KB gzipped — 48% do bundle inteiro
// da homepage — para fazer dois blocos entrarem com fade e 24px de deslocamento.
// IntersectionObserver + uma transition CSS produzem o mesmo resultado por ~0,3 KB.
// A troca está documentada aqui porque é reversível: reinstalar gsap e voltar ao
// efeito do catálogo é um commit.
//
// Outras diferenças em relação ao original:
//  1. O original renderiza com visibility:hidden e só revela dentro do efeito.
//     Se o observer não disparar (seção fora da viewport em teste headless), o
//     conteúdo nunca aparece. Aqui, com movimento desligado o filho renderiza
//     visível e nenhum observer é criado.
//  2. Props do domínio "aparecer e sumir de novo" (disappearAfter e afins) e o
//     fallback de scroller '#snap-main-container' foram removidos: nada usa.
import { useEffect, useRef, useState } from "react";
import "./AnimatedContent.css";
import { useMotionEnabled } from "./useMotionEnabled";

const AnimatedContent = ({ children, distance = 24, threshold = 0.15, delay = 0, className = "", ...props }) => {
  const ref = useRef(null);
  const motionEnabled = useMotionEnabled();
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || !motionEnabled) return undefined;

    // Já está na viewport na primeira medição (topo da página): revela sem esperar.
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          setRevealed(true);
          observer.unobserve(entry.target);
        });
      },
      { threshold }
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [motionEnabled, threshold]);

  const hidden = motionEnabled && !revealed;

  return (
    <div
      ref={ref}
      className={`animated-content ${className}`.trim()}
      data-revealed={hidden ? "false" : "true"}
      style={{ "--animated-content-distance": `${distance}px`, "--animated-content-delay": `${delay}s` }}
      {...props}
    >
      {children}
    </div>
  );
};

export default AnimatedContent;
