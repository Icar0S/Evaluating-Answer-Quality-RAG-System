import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// A homepage é uma ilha: só ela é React. O build emite para frontend/, de onde
// `python -m http.server` (README, playwright.config.ts e o CI) serve tudo como
// arquivo estático — nenhum passo de build é necessário para RODAR o projeto,
// só para MUDAR a homepage. Por isso o resultado do build é versionado.
//
//   frontend/index.html      <- gerado a partir de frontend/home/index.html
//   frontend/home-assets/    <- gerado (JS/CSS com hash)
//
// publicDir aponta para frontend/ para que `npm run dev` sirva /styles/*.css,
// /assets/fonts/* e /components/* nos mesmos caminhos que a produção usa.
// copyPublicDir fica desligado porque em build a saída JÁ É essa pasta —
// copiá-la sobre si mesma seria recursivo.
export default defineConfig({
  base: "/",
  publicDir: "..",
  plugins: [react()],
  build: {
    outDir: "..",
    assetsDir: "home-assets",
    emptyOutDir: false,
    copyPublicDir: false,
    target: "es2020",
  },
});
