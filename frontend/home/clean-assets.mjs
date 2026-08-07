// O build não pode usar emptyOutDir (outDir é frontend/, apagá-la levaria junto
// chat.html, styles/ e components/). Sem isso, home-assets/ vai acumulando os
// JS/CSS com hash de todo build anterior. Este script limpa só essa pasta.
import { rmSync } from "node:fs";
import { fileURLToPath } from "node:url";

rmSync(fileURLToPath(new URL("../home-assets", import.meta.url)), { recursive: true, force: true });
