# Setup do Ambiente de LLM Local — Servidor Mac mini + Cliente Windows

> **Como usar este arquivo:** coloque-o na raiz do repositório e abra o Claude Code (ou o Aider) na pasta. Peça: *"leia o SETUP-LLM-LOCAL.md, detecte em qual máquina estamos e execute o setup correspondente"*. O documento é escrito para ser executável por um agente, com verificações após cada etapa.

---

## 0. Contexto e arquitetura

Ambiente de dois computadores para desenvolvimento assistido por LLM 100% local:

| Papel | Máquina | Specs |
|---|---|---|
| **Servidor** | Mac mini M4 | 16GB memória unificada, ~197GB livres, macOS |
| **Cliente** | Notebook Windows 11 | i7-13620H, 16GB RAM, RTX 4060 Laptop 8GB |

O Mac mini roda o Ollama e serve o modelo pela rede local (WiFi). O notebook Windows é onde se trabalha: VS Code + agente de código (Aider ou Claude Code) apontando para o Mac.

**Estado validado que este documento reproduz:**
- Modelo `coder-32k` (custom sobre `qwen2.5-coder:7b`, 32k de contexto)
- 6,6GB em memória, 100% GPU, sem offload para CPU
- IP fixo `192.168.18.200`, porta `11434`
- Aider funcionando do Windows com commits automáticos em git

**Por que WiFi basta:** apenas tokens de texto trafegam (poucos KB/s). A latência de rede é irrelevante frente ao tempo de geração do modelo. Ethernet só ajudaria em estabilidade, não em velocidade.

---

## 1. Detecção de plataforma

O agente deve identificar a máquina antes de qualquer coisa:

```bash
# POSIX (macOS)
uname -s        # "Darwin" = Mac
```

```powershell
# Windows
$PSVersionTable.Platform   # ou simplesmente: se o comando uname falhar, é Windows
```

Regra de decisão:
- **Darwin** → executar a Parte A (servidor)
- **Windows** → executar a Parte B (cliente)

Se ambos estiverem sendo configurados, faça o Mac primeiro — o cliente depende do servidor estar no ar.

---

# PARTE A — Servidor (Mac mini)

## A1. Instalação do Ollama

```bash
ollama --version
```

Se não existir: baixar em ollama.com/download ou `brew install --cask ollama`. Abrir o app uma vez para registrar o serviço em background.

## A2. Expor na rede

**Método preferencial (funciona de forma confiável):** ícone do Ollama na barra de menus → **Settings** → ativar **"Expose Ollama to the network"** → **Quit Ollama** e reabrir.

**Método alternativo (variável de ambiente):**
```bash
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
```

> ⚠️ **Armadilha conhecida:** `launchctl setenv` **não persiste após reboot** e frequentemente é ignorado pelo app da barra de menus. Neste ambiente, o toggle da UI foi o que funcionou. Sempre verifique o resultado em vez de assumir.

**Verificação obrigatória:**
```bash
lsof -iTCP:11434 -sTCP:LISTEN
```
Deve mostrar `TCP *:11434 (LISTEN)`. Se aparecer `localhost:11434` ou `127.0.0.1:11434`, não funcionou.

## A3. Firewall

System Settings → Network → Firewall → Options: "Block all incoming connections" **desligado**, Ollama com permissão para aceitar conexões.

Via CLI:
```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /Applications/Ollama.app/Contents/MacOS/ollama
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblock /Applications/Ollama.app/Contents/MacOS/ollama
```

## A4. IP fixo

**Descobrir a interface correta** — atenção: no Mac mini, `en0` costuma ser Ethernet e `en1` o WiFi (o inverso do que muitos tutoriais assumem):

```bash
networksetup -listallhardwareports
ipconfig getifaddr en0
ipconfig getifaddr en1
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**Fixar (recomendado pela UI):** System Settings → Network → Wi-Fi → Details… → TCP/IP → "Configure IPv4: **Using DHCP with manual address**" → `192.168.18.200`.

Essa opção fixa só o endereço; gateway e DNS continuam vindo do roteador.

Via CLI:
```bash
sudo networksetup -setmanualwithdhcprouter "Wi-Fi" 192.168.18.200
```

> ⚠️ Escolha um IP **alto** (`.200`, `.250`) para ficar fora do pool que o roteador distribui, evitando conflito. Verifique antes que está livre: `ping 192.168.18.200` de outra máquina — ninguém deve responder.

**Sobre mDNS:** `Mac-mini-de-Icaro.local` **não funcionou** a partir do Windows neste ambiente (a resolução `.local` no Windows é inconsistente). Não perca tempo depurando — use IP fixo. Se quiser um apelido, veja B4.

## A5. Modelos

```bash
ollama pull qwen2.5-coder:7b
```

**Escolha do tamanho — matemática de memória em 16GB (~12GB utilizáveis):**

| Modelo | Pesos | KV cache @32k | Total | Cabe? |
|---|---|---|---|---|
| qwen2.5-coder:**14b** | 9,0GB | ~6GB | ~15GB | ❌ |
| qwen2.5-coder:**7b** | 4,7GB | ~1,9GB | ~6,6GB | ✅ |

**Para trabalho agêntico, contexto vale mais que parâmetros.** Um 7B enxergando o projeto rende mais que um 14B cego com 4k de contexto.

## A6. Contexto de 32k — o passo mais importante

O Ollama define contexto por VRAM disponível: **menos de 24GB → apenas 4k**. Para agentes, a própria documentação recomenda 64k+. Com 4k, o agente falha de formas bizarras e **o Ollama descarta o excedente silenciosamente**.

> ⚠️ **Armadilha central deste setup:** tanto `launchctl setenv OLLAMA_CONTEXT_LENGTH` quanto o restart do app **falharam** em aplicar o contexto. O `ollama ps` continuou mostrando `4096`. A solução que funcionou foi assar o parâmetro no próprio modelo — `PARAMETER num_ctx` no Modelfile tem precedência sobre variáveis de ambiente.

```bash
cat > ~/Modelfile.coder << 'EOF'
FROM qwen2.5-coder:7b
PARAMETER num_ctx 32768
EOF

ollama create coder-32k -f ~/Modelfile.coder
```

Instantâneo e não rebaixa o modelo — só aplica uma camada de configuração.

**Verificação obrigatória:**
```bash
ollama stop qwen2.5-coder:7b
ollama run coder-32k "oi"
ollama ps
```

Resultado esperado:
```
NAME                ID              SIZE      PROCESSOR    CONTEXT    UNTIL
coder-32k:latest    cf0e39a496ed    6.6 GB    100% GPU     32768      5 minutes from now
```

Critérios de aceite:
- `CONTEXT` = `32768`
- `SIZE` subiu de 4,7GB para ~6,6GB (o KV cache aparecendo)
- `PROCESSOR` = **100% GPU** — se houver qualquer % de CPU, o contexto não coube; reduza para 16k

## A7. Tuning

```bash
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
launchctl setenv OLLAMA_KEEP_ALIVE 30m
```

Depois: **Quit Ollama** e reabrir. Verificar com `launchctl getenv OLLAMA_KEEP_ALIVE`.

> ⚠️ Estas variáveis têm o mesmo problema de confiabilidade do contexto. Se `OLLAMA_KEEP_ALIVE` não pegar (o `ollama ps` continuar mostrando ~5 min), o fallback manual é:
> ```bash
> curl -s http://localhost:11434/api/generate -d '{"model":"coder-32k","keep_alive":"30m"}' > /dev/null
> ```
> Sem `prompt`, só carrega o modelo e aplica o tempo.

**Memória para a GPU** (padrão é ~66% em Macs de 16GB):
```bash
sudo sysctl iogpu.wired_limit_mb=14336
```
Não passe de 14GB — deixe ao menos 2GB para o SO. Reseta no reboot.

> ⚠️ Nunca use `keep_alive: -1` (infinito) em 16GB — travaria a memória permanentemente.

## A8. Always-on / headless

**System Settings → Energy:**
- ☑ Prevent automatic sleeping when the display is off
- ☑ Start up automatically after a power failure
- ☑ Wake for network access

**Terminal:**
```bash
sudo pmset -a sleep 0 displaysleep 0 disksleep 0
sudo pmset -a powernap 0
sudo pmset -a womp 1        # wake on network
pmset -g                     # verificar
```

> ⚠️ No Mac mini M4, `caffeinate` segura a assertion mas **não impede o sleep de forma confiável**. Use os settings de Energy + `pmset`, não confie no caffeinate sozinho.

**Acesso remoto:** System Settings → General → Sharing → ligar **Remote Login** (SSH) e **Screen Sharing** (VNC, útil para setup inicial).

**Autostart:** System Settings → General → Login Items → adicionar **Ollama.app**.

> ⚠️ **Trade-off de segurança a decidir conscientemente:** o Mac mini vem com FileVault ligado, que exige senha no boot e **impede o auto-login**. Sem auto-login, o Ollama (que é app de GUI, roda como LaunchAgent) não sobe sozinho após queda de energia. Para servidor que se recupera sozinho, é preciso desligar o FileVault em Privacy & Security. Aceitável numa máquina LAN-only em casa; é decisão do usuário, não automatize sem perguntar.

**Comportamento em repouso:**

| Situação | Servidor continua? |
|---|---|
| Tela apaga / VNC fechado | ✅ Sim |
| Sleep automático | ❌ Não (volta ao acordar, modelo recarrega) |
| Apple → Sleep (manual) | ❌ Não |
| Desligar / reiniciar | ❌ Depende de auto-login + Login Item |

Deixe ligado com a tela apagada. O Mac mini gasta ~5-10W em idle.

## A9. Monitoramento

```bash
# estado do modelo (macOS não tem `watch`)
while true; do clear; ollama ps; sleep 2; done

# requisições chegando em tempo real
tail -f ~/.ollama/logs/server.log

# GPU
sudo powermetrics --samplers gpu_power -i 1000
# ou, melhor: pip install asitop && sudo asitop
```

**Métricas de latência via API** (os campos `eval_count`, `eval_duration`, `total_duration` são a base do painel de monitoramento do frontend deste projeto):

```bash
curl -s http://localhost:11434/api/generate -d '{
  "model": "coder-32k",
  "prompt": "escreva uma função de fibonacci em python",
  "stream": false
}' | python3 -c '
import json,sys
d = json.load(sys.stdin)
print(f"tokens gerados : {d[\"eval_count\"]}")
print(f"tokens/s       : {d[\"eval_count\"]/d[\"eval_duration\"]*1e9:.1f}")
print(f"tempo total    : {d[\"total_duration\"]/1e9:.2f}s")
print(f"prompt eval    : {d[\"prompt_eval_duration\"]/1e9:.2f}s")
'
```

> 📖 **Como ler o `UNTIL`:** não é cronômetro que sobe — é a hora do descarregamento. Reseta para o valor do keep_alive no instante da requisição e conta para baixo. Ver o número diminuindo é comportamento normal, não sinal de que o servidor está ocioso.

**O que observar sob carga real:** `PROCESSOR` deve manter 100% GPU; Memory Pressure no Activity Monitor verde (se ficar amarelo, 32k é o teto — não tente 64k).

---

# PARTE B — Cliente (Windows)

## B1. Verificar conectividade

```powershell
Invoke-RestMethod -Uri http://192.168.18.200:11434/api/tags
```

Deve retornar a lista de modelos incluindo `coder-32k`.

> ⚠️ No PowerShell, `curl` é alias de `Invoke-WebRequest` e se comporta diferente do curl real. Use `Invoke-RestMethod` ou `curl.exe`.

**Se falhar, na ordem de probabilidade:**
1. Firewall do macOS bloqueando
2. Máquinas em redes diferentes (rede de convidados, AP isolation, bandas 2.4/5GHz isoladas)
3. IP errado — reconferir na Parte A4

## B2. Variáveis de ambiente

```powershell
setx OLLAMA_API_BASE "http://192.168.18.200:11434"
```

> ⚠️ **Armadilha recorrente:** `setx` grava permanentemente mas **não afeta a sessão atual**. Para usar imediatamente:
> ```powershell
> $env:OLLAMA_API_BASE = "http://192.168.18.200:11434"
> ```
> Sempre confirme com `$env:OLLAMA_API_BASE` antes de rodar o agente.

## B3. Agente de código — Aider (recomendado)

```powershell
pip install aider-chat
aider --model ollama_chat/coder-32k
```

Use o prefixo `ollama_chat/`, não `ollama/`.

**Por que Aider e não Claude Code:** ver B5. Resumo: o Aider trabalha com diffs em vez de tool calls, e por isso tolera modelos pequenos.

Requer git no projeto (ele commita cada alteração — é a rede de proteção). Se necessário: `git init` + commit inicial.

**Formatos de edição:**
- `whole` (padrão para modelos desconhecidos): reescreve o arquivo inteiro. Seguro, mas lento e gasta muito token.
- `diff`: só o trecho alterado. Bem mais rápido.

```powershell
aider --model ollama_chat/coder-32k --edit-format diff
```

Se aparecer "SEARCH/REPLACE block failed to match" repetidamente, volte ao `whole`.

**Comandos essenciais:**

| Comando | Função |
|---|---|
| `/add arquivo.py` | põe arquivo no contexto |
| `/drop arquivo.py` | remove do contexto |
| `/tokens` | quanto do contexto está ocupado |
| `/undo` | desfaz o último commit do Aider |
| `/diff` | mostra as mudanças |
| `/clear` | limpa histórico da conversa |
| `/architect` | modo raciocínio + edição em duas passadas |

**Economia de contexto:** o repo-map custa ~5k tokens em toda mensagem (com 43 arquivos). Ou seja, ~15% da janela ocupada antes de adicionar qualquer arquivo. Para reduzir:

```powershell
aider --model ollama_chat/coder-32k --map-tokens 1024
```

## B4. Apelido no Windows (opcional)

Como mDNS não funciona, para usar nome em vez de IP — editar como administrador `C:\Windows\System32\drivers\etc\hosts`:
```
192.168.18.200    macmini
```
Depois `http://macmini:11434` funciona em qualquer cliente.

## B5. Claude Code — por que NÃO funcionou

Tentativa feita e documentada aqui para não ser repetida.

**Setup:**
```powershell
irm https://claude.ai/install.ps1 | iex
setx ANTHROPIC_AUTH_TOKEN "ollama"
setx ANTHROPIC_API_KEY ""
setx ANTHROPIC_BASE_URL "http://192.168.18.200:11434"
setx ANTHROPIC_MODEL "coder-32k"
setx ANTHROPIC_SMALL_FAST_MODEL "coder-32k"
```

Problemas encontrados, em ordem:

1. **PATH:** o instalador coloca em `C:\Users\<user>\.local\bin` mas **não adiciona ao PATH**. Correção:
   ```powershell
   [Environment]::SetEnvironmentVariable("Path",
     [Environment]::GetEnvironmentVariable("Path","User") + ";C:\Users\Icaro\.local\bin", "User")
   ```
   (Não use `setx PATH` — funde PATH de sistema com o de usuário e trunca em 1024 caracteres.)

   Em ambientes com conda + venv, o mais robusto é o perfil do PowerShell:
   ```powershell
   if (!(Test-Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force }
   Add-Content $PROFILE "`n`$env:Path += ';C:\Users\Icaro\.local\bin'"
   ```

2. **A flag `--model` foi ignorada** — subiu com Sonnet 5 como padrão. Corrigir com `/model coder-32k` dentro da sessão.

3. **Falha terminal — tool calling:** o modelo imprimiu a chamada como texto:
   ```json
   { "name": "Read", "arguments": { "file_path": "README.md" } }
   ```
   O Claude Code não reconheceu como tool call. Nada foi executado.

**Causa raiz:** Qwen2.5-Coder foi treinado para *completar código*, não para agir como agente. Modelos de code completion preveem a próxima linha; tool calling exige entender o pedido → decidir a ferramenta → formar o JSON. Verificar com:
```bash
ollama show coder-32k    # seção Capabilities — provavelmente só "completion", sem "tools"
```

**Alternativas não testadas** (se quiser insistir no Claude Code):
- `hhao/qwen2.5-coder-tools` — variante com template de ferramentas
- `qwen3:8b` (~5,2GB) — suporte nativo a tools, geração mais nova

---

## 2. Workflow para tirar o máximo de um modelo 7B

Modelos pequenos funcionam bem **se as tarefas forem pequenas e bem delimitadas**. Regras práticas derivadas do uso real:

1. **Uma tarefa, um arquivo.** `/add` só o que é relevante; `/drop` o resto. Rode `/tokens` regularmente.
2. **Peça mudanças concretas**, não objetivos vagos. ✅ "adicione uma seção de instalação com os pré-requisitos" · ❌ "melhore o README"
3. **Use `/architect`** em tarefas com qualquer complexidade — planejar e editar em passadas separadas é o que mais ajuda modelo pequeno.
4. **Verifique cada commit** com `git diff HEAD~1`. No formato `whole` o modelo regenera o arquivo inteiro; risco real de alterar o que não foi pedido ou truncar.
5. **`/undo` sem hesitar.** É barato. O git é a rede de proteção.
6. **Uma responsabilidade por mensagem.** Em vez de "crie o endpoint, os testes e a documentação", faça três pedidos.
7. **Escreva o teste antes** e peça ao modelo para fazer passar — dá critério objetivo de sucesso, evitando a ambiguidade em que modelos pequenos se perdem.

**Expectativa honesta:** modelos de 7–14B são bons em edições focadas, refatorações e boilerplate; perceptivelmente mais fracos que modelos de fronteira em arquitetura multi-arquivo. A primeira resposta pode levar minutos (carregamento); com o modelo quente, é bem mais rápido. Se as respostas seguirem lentas com modelo quente, o gargalo é a banda de memória do M4 base (120 GB/s) — reduza o contexto para 16k.

---

## 3. Checklist de verificação

**Servidor (Mac):**
- [ ] `lsof -iTCP:11434 -sTCP:LISTEN` → `*:11434`
- [ ] `ollama ps` → `CONTEXT 32768`, `PROCESSOR 100% GPU`, `SIZE ~6,6GB`
- [ ] `pmset -g` → `sleep 0`
- [ ] Ollama nos Login Items
- [ ] Remote Login ativo
- [ ] IP fixo confirmado

**Cliente (Windows):**
- [ ] `Invoke-RestMethod http://192.168.18.200:11434/api/tags` retorna `coder-32k`
- [ ] `$env:OLLAMA_API_BASE` preenchido
- [ ] `aider --model ollama_chat/coder-32k` sobe sem erro 404
- [ ] Projeto é repositório git
- [ ] Uma edição de teste aplicada e commitada

---

## 4. Troubleshooting

| Sintoma | Causa provável | Correção |
|---|---|---|
| `Impossível conectar ao servidor remoto` | IP errado ou firewall | Reconferir A4, A3 |
| `ollama ps` mostra `4096` | Var. de ambiente ignorada | Modelfile com `num_ctx` (A6) |
| `PROCESSOR` com % de CPU | Contexto não cabe | Reduzir para 16k |
| Aider: `404 /api/show` em `localhost` | `OLLAMA_API_BASE` não setada na sessão | `$env:OLLAMA_API_BASE = ...` |
| `claude` não reconhecido | PATH | B5, item 1 |
| Modelo imprime JSON de tool como texto | Modelo sem capacidade de tools | Usar Aider (B3) |
| `SEARCH/REPLACE block failed` | Modelo não dá conta de diff | `--edit-format whole` |
| Primeira resposta muito lenta | Carregamento do modelo | Normal; ajustar keep_alive |
| Servidor some após reboot | FileVault bloqueia auto-login | A8 |
| Duas linhas no `ollama ps` | Duas instâncias carregadas | Verificar se há Ollama rodando no Windows |

---

## 5. Segurança

- **LAN-only.** O Ollama **não tem autenticação** — quem alcançar a porta 11434 usa seus modelos. Em janeiro de 2026, uma varredura encontrou 175 mil instâncias Ollama expostas publicamente.
- **Nunca faça port-forward da 11434** no roteador.
- Para acesso fora de casa: **Tailscale** (mesh WireGuard, sem abrir portas).
- Se precisar de autenticação real: LM Studio (suporta tokens) ou Open WebUI na frente.

---

## 6. Tarefas para o agente executar

Ao ler este documento, o agente deve:

1. Detectar a plataforma (seção 1)
2. Rodar o checklist da seção 3 para a plataforma detectada, reportando o que já está OK
3. Executar apenas os passos pendentes
4. **Perguntar antes** de qualquer ação com trade-off de segurança (desligar FileVault) ou que altere configuração de rede
5. Após cada etapa, rodar a verificação correspondente e mostrar o output — não assumir sucesso
6. Ao final, rodar o checklist completo e reportar o estado

**Princípio geral:** neste ambiente, várias configurações "oficiais" falharam silenciosamente (variáveis de ambiente do Ollama no macOS, mDNS, PATH no Windows). **Sempre verifique o resultado em vez de confiar no comando ter retornado sucesso.**
