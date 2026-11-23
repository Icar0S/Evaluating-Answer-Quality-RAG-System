# Escopo final: estado da implementação

**Plano:** [plano-escopo-final.md](../plano-escopo-final.md) · **Branch:** `feat/escopo-final` (um commit por
fase) · **Última atualização:** 23/11/2026

Este documento registra o que já foi feito do plano, fase por fase: o que mudou, onde, como usar, os testes que
rodaram e as decisões tomadas no caminho. Os números vêm das rodadas, com a pasta de cada uma.

## 0. Resumo

| Fase | Estado | O que entregou |
| :--- | :--- | :--- |
| 0. Preparação | Feita | As quatro premissas medidas (§2); D1 a D5 adotadas como propostas (§1) |
| 1. Protocolo das 5 respostas | — | |
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
