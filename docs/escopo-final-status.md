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
| 2. Juízes repetidos e anatomia | — | |
| 3. Comparação em pares | — | |
| 4. Relatório | — | |
| 5. Explicação dos juízes | — | |
| 6. Mutação na plataforma | — | |
| 7. Checklist | — | |
| 8. Apresentação | — | |
| 9. Rodada de análise e fechamento | — | |

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

```
set OLLAMA_BASE_URL=http://127.0.0.1:11434
backend\.venv\Scripts\python.exe tests\generate_answers.py --out tests\pergunta_unica\out\r.json ^
    --index md-v1 --local --goldens md=tests/deepeval/goldens/dataset_md.json ^
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
