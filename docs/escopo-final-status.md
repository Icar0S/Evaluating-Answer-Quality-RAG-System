# Escopo final: estado da implementação

**Plano:** [plano-escopo-final.md](../plano-escopo-final.md) · **Branch:** `feat/escopo-final` (um commit por
fase) · **Última atualização:** 23/11/2026

Este documento registra o que já foi feito do plano, fase por fase: o que mudou, onde, como usar, os testes que
rodaram e as decisões tomadas no caminho. Os números vêm das rodadas, com a pasta de cada uma.

## 0. Resumo

| Fase | Estado | O que entregou |
| :--- | :--- | :--- |
| 0. Preparação | Feita | As quatro premissas medidas (§2); D1 a D5 adotadas como propostas (§1) |
| 1. Protocolo das 5 respostas | Feita | Semente por requisição no `/chat`; o protocolo no `generate_answers.py`; o controle quente × frio; `chat-limpo.spec.ts`. 5 respostas diferentes, que se reproduzem, com os mesmos trechos (§3) |
| 2. Juízes repetidos e anatomia | Código feito; o dado espera a rodada de análise (§10) | Ordem embaralhada com semente, número do julgamento e passos da nota nos três juízes; a decomposição gerador × juiz (§4) |
| 3. Comparação em pares | Código feito; o dado espera a rodada | `ArenaGEval` com ordem e nomes controlados, a Choice do Jev e a preferência implícita (§5) |
| 4. Relatório | Código feito; falta ver com a rodada completa | `build_report.py --pergunta-unica` (§6) |
| 5. Explicação dos juízes | README feito; falta o card do palco (fase 8) | "Como cada juiz chega à nota" (§6) |
| 6. Mutação na plataforma | Código feito; o dado espera a rodada | C1, C2, P1, R1 e o controle N0 (§7) |
| 7. Checklist | Código feito; os estados esperam a rodada | 16 itens, com o critério escrito antes (§8) |
| 8. Apresentação | Não começou | |
| 9. Rodada de análise e fechamento | A 1.ª rodada falhou; a auditoria e as correções estão em §9 | |

## 1. Decisões

As cinco primeiras seguem a proposta do plano (§15) e podem ser revistas com o orientador. Mudar uma delas
depois de uma rodada de análise vira desvio registrado aqui.

| | Decisão | Adotada |
| :--- | :--- | :--- |
| D1 | A pergunta e o índice | `md-mtbench-vieses` no `md-v1` |
| D2 | Semente por requisição na API | Sim: campo opcional `seed` no `POST /chat`; o frontend não manda |
| D3 | O critério do par e o empate | O critério do §6 do plano; empate onde o juiz aceita (o Jev aceita, o `ArenaGEval` não) |
| D4 | Julgamentos por resposta e margem da preferência implícita | 3 julgamentos; margem de 0,05 |
| D5 | Os operadores de mutação | C1, C2, P1 e R1 |
| D6 | O que entra na tese | **Em aberto** (orientador). Não bloqueia nenhuma fase |
| D7 | A ordem das fases em relação ao IST | **Em aberto** (orientador). As fases seguem a ordem do plano; nenhuma mexe na campanha do IST |

Os limites do checklist (90% de consistência nas duas ordens, 70% de vitórias da mais longa) também foram
escolhidos na implementação, antes dos dados, e ainda não passaram pelo orientador.

## 2. Fase 0: as premissas, medidas

Medido em 23/11/2026, no Ollama da máquina (0.40.16, `127.0.0.1:11434`) e no índice `md-v1` local.

| Premissa | Medido | Consequência |
| :--- | :--- | :--- |
| A temperatura padrão do gemma3:4b | `ollama show`: `temperature 1`, `top_k 64`, `top_p 0.95` | O protocolo fixa a temperatura em **1,0**, a padrão, e grava o valor. Fixar não muda a amostragem de quem usa o chat: só a torna explícita |
| O artigo chega no top-k | Sim: `NeurIPS-2023-judging-llm-as-a-judge…::c0` em 3.º lugar (similaridade 0,739). Os outros três trechos são do *Testing LLM-Based Systems* (0,760, 0,753, 0,728) | A pergunta tem a evidência. O item "o artigo certo chega no top-k" do checklist já passa |
| Os mesmos trechos no mesmo processo | Sim: três buscas seguidas, o mesmo hash (`9979d94203e4`) | O `md-v1` tem 857 trechos, abaixo de 1.000: o grafo HNSW é refeito a cada processo, e entre processos o top-k pode mudar. A conferência de hash do protocolo fica, como o plano prevê |
| O `ArenaGEval` deixa fixar a ordem | Sim, pelo código (deepeval 4.2.0, `metrics/arena_g_eval/utils.py`): a ordem dos concorrentes e os nomes falsos saem do `random.shuffle` do módulo, sem semente | O par troca o `random` daquele módulo por um de ordem fixa: o primeiro concorrente é sempre "Alice", o segundo "Bob". O teste da fase 3 confere que a ordem enviada ao juiz é a pedida |
| O estado do par cabe no Jev | Sim: com os 4 trechos reais e duas respostas de 2.500 caracteres, 16.228 caracteres, **5.410 tokens estimados** (a estimativa do deepeval, 3 caracteres por token), contra o limite de 32.000 | O par leva os 4 trechos inteiros; não precisa cortar |
| O custo de reindexar no C1 e no C2 | Não precisa reindexar. **C1:** copiar os vetores do `md-v1` sem os 36 trechos do artigo leva 0,9 s. **C2:** embedar os 36 trechos do artigo mutado leva 1,5 s | Os mutantes de corpus usam uma coleção temporária montada por cópia, sem tocar na coleção `rag_md_v1` |

**Juiz e gerador na máquina:** `gemma3:4b` e `qwen3:8b` estão no Ollama local; o `.env` da máquina já gera com
o gemma3:4b e julga com o qwen3:8b (`DEEPEVAL_JUDGE_MODEL`, `RAGAS_JUDGE_MODEL`).

## 3. Fase 1: o protocolo das 5 respostas

### O que mudou

| Onde | O quê |
| :--- | :--- |
| `backend/app/schemas.py`, `rag/generation.py`, `main.py` | `ChatRequest.seed` (opcional, 0 a 2³¹−1): a semente vale só naquela chamada, no lugar de `GENERATION_SEED`. O `ChatMetadata` ganha `temperature` e `seed`, o que foi de fato ao Ollama (`None` = o padrão do modelo, ou o provider remoto, que não aceita nenhum dos dois). O frontend não manda o campo: o chat da tela não muda |
| `tests/generate_answers.py` | O protocolo, nos dois modos (in-process e `--api`): `--only`, `--sementes 1-5`, `--repeticoes`, `--estado-limpo`, `--ollama`, `--controle-estado` e `--aquecimento`. Sem essas opções, o plano de perguntas é o de sempre (um teste confere) |
| `tests/benchmark/test_protocolo_pergunta_unica.py` | 12 testes herméticos: um servidor local faz a API do RAG e o Ollama ao mesmo tempo e registra a ordem das chamadas |
| `tests/api/test_chat.py` | 5 testes da semente por requisição |
| `frontend/tests/e2e/chat-limpo.spec.ts` | Dois testes com o backend mockado (rodam no CI): na mesma conversa, a segunda pergunta vai sozinha; a mesma pergunta no mesmo chat e num chat novo manda o mesmo corpo |

**Como o protocolo se protege:**

- **Servidor limpo.** Antes de cada resposta, o gerador sai da memória do Ollama (`keep_alive: 0`), e o script
  espera ele sumir do `/api/ps`. Se não sair em 30 s, o protocolo para antes de perguntar.
- **Os mesmos trechos.** Cada resposta grava o hash dos trechos (`sources_sha256`: o id e o texto de cada um,
  na ordem). Uma resposta da mesma pergunta com outros trechos para o protocolo (código 3).
- **A semente valeu.** Se a resposta não devolve a semente pedida, o protocolo para (código 3). É o caso de um
  backend anterior a esta fase: o Pydantic ignora o campo desconhecido em silêncio, e as 5 respostas sairiam
  sem semente. **No Docker, é preciso reconstruir a imagem do backend** (`docker compose build backend`).
- **A temperatura que valeu fica gravada:** a configurada, ou a padrão do modelo, lida do `/api/show`.

### Como rodar

```powershell
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
backend\.venv\Scripts\python.exe tests\generate_answers.py --out tests\pergunta_unica\out\r.json `
    --index md-v1 --local --goldens md=tests/deepeval/goldens/dataset_md.json `
    --only md-mtbench-vieses --sementes 1-5 --estado-limpo --set generation_temperature=1
```

Pela API do Docker: `--api http://localhost:8010 --ollama http://127.0.0.1:11435`, sem `--index`, `--local`
e `--set`. O controle troca `--estado-limpo` por `--controle-estado`.

### A rodada de validação (23/11/2026)

Pasta: `tests/pergunta_unica/out/fase1/` (fora do git). gemma3:4b no Ollama da máquina, `md-v1`, temperatura 1
(configurada, igual à padrão), sementes 1 a 5.

| Critério da fase | Resultado |
| :--- | :--- |
| As 5 respostas variam entre si | Sim: 5 textos diferentes, de 580 a 747 caracteres; semelhança entre pares de 0,30 a 0,67 (`difflib`) |
| Saem iguais quando a rodada se repete | Sim: a rodada B, outro processo, deu os mesmos 5 textos, semente a semente. O "frio" do controle, um terceiro processo, também |
| Os mesmos trechos | Sim: o mesmo hash (`9979d94203e48277`) nas 5 respostas das rodadas A e B e nas 15 do controle. São três processos: o HNSW não mudou o top-4 desta pergunta |
| Custo | ~8 s por resposta, com a recarga do modelo; descarregar leva ~0,15 s. A rodada de 5 respostas leva 49 s |

**O controle quente × frio** (`controle.json`), por semente: a resposta fria, de novo fria, e quente, logo
depois de outra pergunta (`md-seven-failure-points-lista`), sem recarregar.

| Semente | O frio se reproduz | O quente sai igual ao frio |
| :--- | :--- | :--- |
| 1 a 5 | 5 de 5 | 5 de 5 (semelhança 1,0) |

Nesta pergunta, com semente fixa, o cache de prefixo **não mudou nenhuma resposta**. A resposta quente só saiu
mais rápida (~4 s, contra ~8 s da fria). No E0 do benchmark, com temperatura 0, a ordem das perguntas mudou
2 de 103 respostas: o efeito existe, mas é raro. O protocolo continua zerando o estado, porque o custo é
pequeno (~4 s por resposta) e tira a dúvida. Para a apresentação, a leitura é: o viés que a turma apontou não
vem do histórico do chat (que nunca chega ao modelo), e o do servidor, medido aqui, foi zero em 5.

**O que as 5 respostas trazem** (material para os juízes): as 5 citam os três vieses e os 80%; a 5.ª cita
também os 75% do viés de posição. Mas todas misturam números do outro artigo dos trechos (*Testing LLM-Based
Systems*): a "deflação de kappa" atribuída ao viés de posição, os "10–25%", o relatório da Atlan. É o tipo de
erro que separa os juízes: a afirmação está nos trechos (fiel), mas não é do MT-Bench (correção).

## 4. Fase 2: os juízes repetidos e a anatomia

| Onde | O quê |
| :--- | :--- |
| `tests/{ragas,deepeval,jeveval}/run_and_export.py` | `--repeticao`, `--ordem-semente`, `--out-dir` e, no RAGAS e no DeepEval, `--passos`. Sem essas opções, o export é o de sempre |
| `tests/pergunta_unica/rodar.py` | O orquestrador: as respostas, os 3 juízes × K, os pares, os mutantes, a análise e o relatório |
| `tests/pergunta_unica/analise.py` | Por juiz e métrica: a média, a faixa, o desvio entre as respostas (gerador) e entre os julgamentos (juiz), os vereditos que mudam |

**Nota que falta é buraco.** O passo tenta de novo só os casos com buraco (`--tentativas`), preenche só o que
faltava e nunca troca uma nota medida. Se sobrar buraco, o passo sai "incompleta" e a rodada não termina com código
0. O `--retomar` completa buracos e põe de lado um export interrompido no meio.

## 5. Fase 3: os pares

| Onde | O quê |
| :--- | :--- |
| `tests/pergunta_unica/pares.py` | O plano (10 pares × 2 ordens), o critério (D3), a preferência implícita (D4) e as contas |
| `tests/deepeval/pairwise.py` | O `ArenaGEval` com a ordem pedida. O nome ("Alice", "Bob") fica preso à resposta nas duas ordens, e cada resposta é "Alice" em 2 dos 4 pares dela: a posição e o nome são medidos à parte |
| `tests/jeveval/pairwise.py` | Uma Choice A / B / Empate, com as probabilidades |

Conferido com um juiz falso e as 5 respostas reais: nos 20 julgamentos, o prompt sai com a ordem e os nomes pedidos;
um juiz que só olha o nome dá consistência 100% nas duas ordens e ranking empatado, e aparece no viés de nome.

## 6. Fases 4 e 5: o relatório e a explicação

`build_report.py --pergunta-unica <pasta>` gera `relatorio.md`, `.pdf` e `.xlsx`: o protocolo (com a máquina: a
tomada e as suspensões), os passos que não fecharam, as respostas, as notas e a decomposição, a anatomia da
fidelidade (com a conta refeita e conferida em todos os julgamentos), os pares, os mutantes e o checklist. A
explicação de como cada juiz chega à nota está no README da raiz ("A pergunta única").

## 7. Fase 6: os mutantes

`mutantes.py` prepara, `rodar.py --mutantes` gera e julga, `mutacao.py` decide. C1 e C2 são cópias dos vetores do
`md-v1` numa pasta da rodada; P1 e R1 são sobrescritas. **N0** é o controle: a cópia sem mudança. A morte só conta
se caíram mais respostas do que na pior comparação nula (o N0 e cada julgamento do original contra os outros). O
efeito na resposta é contado por palavra-chave, sem juiz (`fatos.py`).

## 8. Fase 7: o checklist

16 itens em `tests/checklist/checklist.yaml` (MFT, INV e DIR), avaliados por `avaliar.py` sobre a pasta da rodada.
Um juiz que faltou deixa o item "sem dado": só o que foi medido reprova, e, enquanto faltar um, o item não passa.

## 9. A 1.ª rodada, a auditoria e os desvios (23/11/2026)

**A 1.ª rodada de análise falhou** (`out/20261123T110351`, fora do git). O notebook saiu da tomada às 11:28 e
suspendeu de 11:31 a 13:22 (log de eventos do Windows), no meio do DeepEval. Saíram 25 de 25 notas do RAGAS, 23 de
25 do DeepEval e 3 de 30 do Jev, que rodou logo na volta, com erro de conexão. Uma das duas notas do DeepEval caiu
antes disso, às 11:10, por falta de memória na GPU: o modelo de embeddings do RAGAS, usado pela última vez às 11:08,
ainda estava nos 5 minutos em que o Ollama o mantém carregado. **Essa pasta não é retomada:** a rodada válida é outra.

**O que a auditoria achou e o que mudou:**

| | Defeito | Correção |
| :--- | :--- | :--- |
| 1 | O `--retomar` e o estado do passo ignoravam notas não medidas: 3 de 30 saía "concluída" | Buraco é detectado, medido de novo e, se sobrar, o passo sai "incompleta" (`test_analise.py`) |
| 2 | Nada impedia a suspensão numa rodada de horas | `energia.py`: não começa na bateria, segura a suspensão por inatividade, registra a suspensão no passo. E o embedding sai da GPU antes do DeepEval |
| 3 | O esperado do C2 ("a correção cai") ignorava que o survey mantém os fatos certos | Esperado revisto, e os fatos certos e trocados de cada resposta passam a ser contados |
| 4 | Mutantes sem controle nulo, com 3 julgamentos de um lado e 1 do outro | N0 e o controle "deixando um julgamento de fora" |
| 5 | No `ArenaGEval`, o primeiro era sempre "Alice": nome e posição confundidos | O nome preso à resposta, balanceado |
| 6 | O checklist dizia "falhou" quando faltava um juiz | "sem dado", e só o que foi medido reprova |
| 7 | `tests/pergunta_unica/out` entrava na imagem de testes | `docker/tests.Dockerfile.dockerignore` |
| 8 | O `set VAR=valor` dos documentos não vale no PowerShell | `$env:VAR = "valor"` |
| 9 | Este documento estava em branco nas fases 2 a 7 | §4 a §8 |
| 10 | O ARCHITECTURE citava um teste da conta que não existia | O relatório confere a conta em todos os julgamentos, e o teste existe |
| 11 | Achado na fumaça ao vivo, depois da auditoria: o qwen3:8b às vezes devolve o vencedor como `$Alice$`, e o `ArenaGEval` quebrava com `KeyError` | O nome é limpo na leitura e o texto bruto fica no export (`vencedor_bruto`); o prompt não muda |
| – | O README do JevEval dizia que os 15 julgamentos e os 20 pares já tinham rodado | Corrigido |

**Desvios do plano, todos antes de qualquer mutante ou par ser julgado por um juiz de verdade:**

- **C1:** o plano esperava abstenção. O acervo tem os fatos por outra fonte (o survey *Testing LLM-Based Systems*):
  responder pelo survey não é falha.
- **C2:** o plano esperava que a correção caísse. O 1.º trecho do top-4 é do survey, com os fatos certos, e o artigo
  trocado é o 3.º: a correção só cai nas respostas que seguirem o artigo trocado.
- **JUI-5 do checklist:** a morte precisa ser confirmada contra o controle nulo.
- **N0:** um operador a mais, de controle, fora dos quatro de D5.
- **Pares do DeepEval:** os nomes deixam de andar com a posição; entra a medida do viés de nome.
