import { lazy, Suspense, useEffect, useState } from "react";
import ShaderBoundary from "./motion/ShaderBoundary";
import { useMotionEnabled } from "./motion/useMotionEnabled";

// three.js é a dependência mais cara do projeto inteiro: 133,5 KB gzipped contra
// 50 KB do restante da homepage. React.lazy mantém ela fora do bundle
// inicial — a página pinta e fica interativa sem three, e o shader chega depois,
// num chunk separado. Medido: o bundle inicial saiu de 48,96 para 49,98 KB gz.
//
// O fallback do Suspense é null de propósito. O gradiente estático que fica
// atrás já está pintado pelo CSS, então não há nada a preencher e nenhum
// deslocamento de layout quando o canvas monta.
const LiquidEther = lazy(() => import("./motion/LiquidEther"));

const DESKTOP_QUERY = "(min-width: 768px)";

// Banda fria e dessaturada, e não a paleta roxo/rosa que vem no componente.
// Não é escolha estética solta: --cite-500 (âmbar) e --probe-500 (ciano) são
// semânticos neste projeto — significam "evidência" e "telemetria". Um fundo
// nesses tons roubaria o significado deles. Estes três ficam abaixo da croma do
// ciano e nunca são lidos como um dado.
//
// A intensidade foi calibrada por medição, não no olho: com esta paleta e o véu
// de 0,52, o pixel mais claro do fundo em 10 frames animados fica em L=0,0172,
// o que deixa o --text-secondary em 6,6:1 e o --text-primary em 13,5:1.
const PALETTE = ["#1b4a6e", "#2f7fa8", "#57bcd8"];

export default function Background() {
  const motionEnabled = useMotionEnabled();
  const [isDesktop, setIsDesktop] = useState(() => window.matchMedia(DESKTOP_QUERY).matches);

  useEffect(() => {
    const query = window.matchMedia(DESKTOP_QUERY);
    const sync = () => setIsDesktop(query.matches);
    query.addEventListener("change", sync);
    return () => query.removeEventListener("change", sync);
  }, []);

  // Abaixo de 768px o shader sai por completo, não é degradado: num celular ele
  // custa bateria e frame rate para decorar uma tela onde o conteúdo já ocupa
  // tudo. O gradiente estático do CSS assume. Mesma coisa com o movimento
  // desligado — e nesses dois casos o three.js sequer é baixado, porque o
  // import() nunca chega a ser avaliado.
  if (!motionEnabled || !isDesktop) return null;

  return (
    <ShaderBoundary>
      <Suspense fallback={null}>
        <LiquidEther
          colors={PALETTE}
          resolution={0.5}
          mouseForce={26}
          cursorSize={110}
          autoDemo
          autoSpeed={0.5}
          autoIntensity={2.2}
          dt={0.014}
        />
        {/* Depois do shader no DOM, de propósito: irmãos posicionados empilham
            na ordem do documento, e o véu precisa ficar por cima do canvas.
            Fica aqui e não no HTML estático porque só deve existir quando o
            shader existe — senão escureceria o gradiente do celular e do modo
            reduzido sem ter o que velar. */}
        <div className="bg-veil" />
      </Suspense>
    </ShaderBoundary>
  );
}
