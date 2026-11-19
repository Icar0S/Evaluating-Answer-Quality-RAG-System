// O palco da apresentação gravada (tests/apresentacao.py, scripts\apresentacao.bat): abre a página que conta a
// rotina num Chromium SEM JANELA e grava só essa página, quadro a quadro, pelo CDP.
//
//   node frontend/tests/apresentacao/palco.mjs <url-do-palco> [<ffmpeg> <saida.mkv> [<fps>]]
//
// Por que a página, e não a tela: na primeira versão, o ffmpeg gravava a tela inteira (gdigrab), e o vídeo pegou
// as outras janelas abertas durante a rotina (19/11/2026). Aqui o vídeo só pode conter o palco, e o computador
// fica livre durante a gravação. O E2E também roda sem janela e manda os quadros dele ao palco (DEMO_ESPELHO,
// e2e/fixtures.ts), que os mostra ao vivo.
//
// Sem ffmpeg e saída, só abre o palco (ensaio, --sem-video). O vídeo sai em MKV, que sobrevive a uma queda;
// o orquestrador converte para MP4 no fim.
import { chromium } from "@playwright/test";
import { spawn, spawnSync } from "node:child_process";

const [url = "http://127.0.0.1:5590/", ffmpeg, saida, fpsArg] = process.argv.slice(2);
const fps = Number(fpsArg || 15);

// 1536x864 a 1,25: os mesmos 1920x1080 de uma tela de notebook com escala de 125%, onde o palco foi desenhado.
const LARGURA = 1536;
const ALTURA = 864;
const ESCALA = 1.25;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: LARGURA, height: ALTURA }, deviceScaleFactor: ESCALA });
await page.goto(url);

// O H.264 no padrão de vídeo em HD (BT.709, faixa limitada), que todo player mostra igual. O JPEG do Chromium vem
// em faixa cheia (BT.601), e o zscale converte sem escurecer a página (medido em 19/11/2026: o scale do ffmpeg
// escurecia uns 3 níveis). Num ffmpeg sem o zscale, o vídeo fica em faixa cheia, marcada no arquivo.
function argumentosDoFfmpeg() {
  const filtros = spawnSync(ffmpeg, ["-hide_banner", "-filters"], { encoding: "utf8" }).stdout || "";
  const cor = / zscale /.test(filtros)
    ? ["-vf", "zscale=matrixin=470bg:rangein=full:matrix=709:range=limited,format=yuv420p",
       "-colorspace", "bt709", "-color_range", "tv"]
    : ["-pix_fmt", "yuv420p"];
  return ["-hide_banner", "-loglevel", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg",
          "-framerate", String(fps), "-i", "-", ...cor,
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", String(fps * 5), saida];
}

let gravador = null;
let quadro = null;
let capturando = false;
let captura = Promise.resolve();
if (ffmpeg && saida) {
  gravador = spawn(ffmpeg, argumentosDoFfmpeg(), { stdio: ["pipe", "ignore", "inherit"] });
  // Se o ffmpeg cair, o palco continua no ar até o fim (o orquestrador espera por ele); o vídeo é que se perde.
  gravador.stdin.on("error", (erro) => console.error(`o ffmpeg parou de receber quadros: ${erro.message}`));
  // Os quadros: um Page.captureScreenshot por vez, no ritmo do vídeo. O screencast do CDP sairia mais barato, mas
  // entrega a página no tamanho CSS (1536x864), e o texto ficaria borrado em tela cheia. A captura com clip.scale
  // sai em pixels físicos (1920x1080, ~50 ms cada). A escala vai explícita porque esta sessão do CDP não enxerga a
  // que o Playwright emula: sem ela, a captura também sai em 1536x864.
  const cdp = await page.context().newCDPSession(page);
  const clip = { x: 0, y: 0, width: LARGURA, height: ALTURA, scale: ESCALA };
  capturando = true;
  captura = (async () => {
    while (capturando) {
      const inicio = Date.now();
      try {
        const { data } = await cdp.send("Page.captureScreenshot", { format: "jpeg", quality: 90, clip });
        quadro = Buffer.from(data, "base64");
      } catch {
        // um quadro perdido (a página fechando): o relógio repete o anterior
      }
      await new Promise((resolve) => setTimeout(resolve, Math.max(0, 1000 / fps - (Date.now() - inicio))));
    }
  })();
}

// Para o vídeo durar o que a rotina durou, cada tique escreve quantos quadros o relógio pede, repetindo o último
// quando a captura atrasa.
let inicio = 0;
let escritos = 0;
const tique = setInterval(() => {
  if (!gravador || !quadro || !gravador.stdin.writable) return;
  if (!inicio) inicio = Date.now();
  const devidos = Math.floor(((Date.now() - inicio) * fps) / 1000) + 1;
  while (escritos < devidos) {
    gravador.stdin.write(quadro);
    escritos++;
  }
}, 20);

// Até o orquestrador pedir para fechar, ou sumir (30 s sem resposta).
let falhas = 0;
for (;;) {
  await new Promise((resolve) => setTimeout(resolve, 1000));
  try {
    const estado = await (await fetch(new URL("/estado", url))).json();
    falhas = 0;
    if (estado.fechar) break;
  } catch {
    if (++falhas >= 30) break;
  }
}
capturando = false;
await captura;
clearInterval(tique);
if (gravador && gravador.exitCode === null && gravador.signalCode === null) {
  const fechou = new Promise((resolve) => gravador.on("close", resolve));
  gravador.stdin.end();
  await fechou;
}
await browser.close();
