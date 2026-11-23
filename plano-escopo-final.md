# Plano: o escopo final da plataforma (uma pergunta, cinco respostas, três juízes e a comparação em pares)

**Criado em:** 23/11/2026, depois da apresentação para a turma e o orientador · **Regra:** um commit
por fase, CI verde em todas · **Estado:** proposta, com 7 decisões em aberto (seção 15)

## 0. Resumo

A apresentação deixou três melhorias e uma explicação a fazer:
1. **Uma pergunta só.** O gemma3:4b dá 5 respostas para ela, e o RAGAS, o DeepEval e o JevEval
   julgam as 5.
2. **Sem viés de conversa.** Toda resposta sai de um "chat limpo".
3. **Comparação em pares.** Um juiz recebe duas respostas à mesma pergunta e escolhe a melhor.
4. **O JevEval explicado.** Fica claro por que a nota dele não se lê como a dos outros dois.

Somam-se as duas ideias que já estavam na fila: a mutação dentro da plataforma (os PDFs e os
prompts mutados) e um checklist próprio para testes de RAG.

Este plano junta tudo numa história só, que é a que a plataforma conta no fim do doutorado:

1. **Uma pergunta, a do MT-Bench sobre os vieses do LLM-as-a-judge.** O RAG responde 5 vezes,
   sempre com o mesmo trecho recuperado e sempre com o servidor em estado limpo.
2. **Três juízes de natureza diferente julgam as 5 respostas, cada um 3 vezes.** Dá para separar o
   que varia no gerador do que varia no juiz.
3. **Os juízes comparam as respostas em pares, nas duas ordens.** Medimos nos próprios juízes os
   vieses que a resposta descreve: posição e verbosidade.
4. **Mutantes da mesma pergunta mostram quais juízes percebem um defeito injetado.** Um artigo
   removido, um documento desatualizado, o prompt sem ancoragem.
5. **Um checklist diz, item por item, o que está testado, o que falhou e o que ficou de fora.**
   É ele que fecha o escopo.
6. **Tudo sai num comando, num relatório e num vídeo.**

O que não está aqui fica de fora (seção 13).

## 1. De onde vem cada parte

| Origem | Pedido | Onde está neste plano |
| :--- | :--- | :--- |
| Turma e orientador | Focar em 1 pergunta, com 5 respostas do gemma, julgadas pelos 3 frameworks | §3, §4, §5 |
| Turma e orientador | Analisar o viés: sempre perguntar com o chat limpo | §2, §4 |
| Turma e orientador | Comparação em pares (pairwise) no LLM-as-a-Judge | §6 |
| Turma e orientador | Explicar a diferença do JevEval para entender os resultados | §7 |
| Ícaro | Mutação na plataforma, gerando os mutantes (PDFs) e os prompts | §8 |
| Ícaro | Checklist específico para testes de RAG | §9 |
| Verificação de 19/11/2026 | O que ficou aberto: PR #4, validação do Jev, README do JevEval | §11 |

## 2. O que o código faz hoje (conferido em 23/11/2026)

| Ponto | Hoje | O que isso muda no plano |
| :--- | :--- | :--- |
| Histórico do chat | O `POST /chat` recebe só `question` e `top_k` (`backend/app/schemas.py`). O frontend manda só `{ question }` (`frontend/components/chat.js`). O histórico fica no `localStorage`, só para a tela | **Perguntar no mesmo chat ou num chat novo não muda o que o modelo recebe.** O prompt é sempre o prompt de sistema mais uma mensagem, com a pergunta e os trechos |
| Estado do servidor | O cache de prefixo do Ollama faz a resposta depender do prompt anterior. Com temperatura 0 e semente 42, trocar a ordem das perguntas mudou 2 de 103 respostas (`docs/benchmark-status.md`, E0) | **É aqui que existe "chat sujo" nesta plataforma: no servidor, não na conversa.** O protocolo zera esse estado antes de cada resposta (§4) |
| Amostragem | `GENERATION_TEMPERATURE` e `GENERATION_SEED` vêm vazios (`backend/app/config.py`, `docker-compose.yml`): o gemma usa os padrões dele, sem semente | Cada resposta é uma amostra nova e não se repete. O protocolo fixa temperatura e semente |
| Semente por pergunta | Não existe. A semente vale para o servidor inteiro | Pela API, 5 respostas com a mesma semente sairiam iguais. É preciso semente por requisição (decisão D2) |
| Recuperação | No mesmo processo, a mesma pergunta traz os mesmos trechos. Entre processos, o grafo HNSW do Chroma muda o top-k de ~1 pergunta em 70 (E0, a c28) | O protocolo confere que as 5 respostas receberam os mesmos trechos e para se não receberam |
| Ruído do juiz | As mesmas 70 respostas, julgadas de novo pelo qwen3:8b, mudam de fidelidade em 7 a 9 casos, muitas vezes de 0 para 1 (E0) | Com 5 respostas, o ruído do juiz pode ser maior que a diferença entre elas. Cada resposta é julgada 3 vezes (§5) |
| Quem responde e quem julga | No Docker: gemma3:4b gera, qwen3:8b julga. No `.env.example` local, o contrário: qwen3:8b gera e o DeepEval julga com gemma3:4b | O protocolo fixa gemma3:4b gerando e qwen3:8b julgando, nos dois modos, e grava isso |
| Pares no DeepEval | `ArenaGEval` existe no 4.2.0 e no 4.3.1. Recebe N concorrentes e escolhe **um** vencedor, sem empate. Troca os nomes por nomes falsos e embaralha a ordem com `random.shuffle`, sem semente | Dá para usar, mas o teste precisa fixar a ordem para medir o viés de posição (§6) |
| Pares no RAGAS | O RAGAS 0.4.3 não tem métrica de pares | O RAGAS entra pela preferência implícita, derivada das notas pontuais (§6) |
| Pares no Jev | Um `Choice` com opções fechadas ("A", "B", "Empate") devolve a probabilidade de cada uma. O `TypeSafeModel.decide(state, questions)` aceita um estado qualquer | O Jev julga o par com probabilidade e com empate, coisa que o `ArenaGEval` não faz |
| Duração | Na rodada de 19/11, o DeepEval levou 26 min para 4 respostas (~6,5 min por resposta) | 5 respostas × 3 julgamentos ≈ 1h40 só no DeepEval. A rodada de análise não é a do vídeo (§10) |

## 3. A pergunta única

**Recomendação: `md-mtbench-vieses`** (`tests/deepeval/goldens/dataset_md.json`).

> Quais vieses e limitações do LLM-as-a-judge são examinados no trabalho do MT-Bench e Chatbot
> Arena, e qual a concordância do GPT-4 com as preferências humanas?
>
> **Referência:** os vieses de posição, de verbosidade e de autoaprimoramento (self-enhancement),
> além da capacidade de raciocínio limitada. Juízes fortes como o GPT-4 passam de 80% de
> concordância com as preferências humanas, o mesmo nível entre humanos.

Por quê:

- **A referência é curta e conferível:** três vieses, uma limitação e um número. A correção tem
  critério claro.
- **O tema é o próprio método.** A resposta descreve os vieses que o §6 mede nos juízes: posição e
  verbosidade. O autoaprimoramento é o que o projeto já evita, porque quem responde não julga. O
  artigo é o que popularizou a comparação em pares (Zheng et al., NeurIPS 2023).
- **O artigo está nos índices `md-v1`, `md-v2` e `pdf-v2`.** Não está no `pdf-v1`.

**Índice: `md-v1`.**

- **É o `backend-md` da stack do Docker**, sem configuração a mais.
- **Essa pergunta já funcionou nele.** Em 13/10/2026, o RAG recuperou o artigo, citou os três
  vieses (o de posição com o número, 75%), e o RAGAS deu fidelidade 1,0 e recall de contexto 1,0
  (`tests/ragas/results/20261013T160437Z.json`).
- **O `pdf-v2` não resolve as palavras grudadas.** O Markdown desse artigo tem esse problema (a
  anotação "grave" no golden), e o PDF dele no `pdf-v2` foi gerado a partir do mesmo Markdown
  (`data/conversions.json`).
- **O `pdf-v1` dá de graça a prévia do mutante C1.** Ele não tem o artigo, e o RAG respondeu mesmo
  assim, com trechos de outro artigo, em vez de se abster: fidelidade 0,0 no RAGAS
  (`tests/ragas/results/20261013T153809Z.json`).

**Alternativa:** `ragas-definicao` no `pdf-v1`, para ficar no acervo original. Ela perde a ligação
com o §6.

## 4. As 5 respostas e o chat limpo (melhorias 1 e 2)

**Protocolo:**

1. **Entrada fixa e gravada:** a pergunta, o índice, o `top_k`, a versão do prompt e o modelo
   (gemma3:4b, provider local; o remoto ignora temperatura e semente).
2. **Pergunta isolada.** Cada resposta é um `POST /chat` novo, sem histórico, como já é hoje.
3. **Servidor limpo.** Antes de cada resposta, o gemma é descarregado do Ollama
   (`POST /api/generate` com `keep_alive: 0`; no Docker, a porta 11435 da máquina). Nenhuma
   resposta herda o cache de prefixo da anterior. Recarregar custa segundos.
4. **Amostragem com semente:**
   - a temperatura do gerador fica fixa e gravada: a padrão do gemma3:4b, lida na fase 0 com
     `ollama show gemma3:4b --parameters`;
   - as sementes são 1 a 5;
   - as 5 respostas variam entre si e saem iguais quando a rodada se repete.
5. **Os mesmos trechos.** O script compara o hash das fontes das 5 respostas e para se algum
   diferir. Assim, a diferença entre as respostas vem só do gerador.
6. **Cada resposta grava:** semente, temperatura, estado ("frio"), latência, tokens e o hash dos
   trechos.

**O que muda no código:**

- **Semente por requisição (D2).** `ChatRequest` ganha `seed: int | None`, opcional, e o backend usa
  essa semente no lugar da do servidor, só naquela chamada. O frontend não manda o campo, então
  nada muda para quem usa o chat. Testes de API cobrem os dois casos.
- **`tests/generate_answers.py`.** Ganha `--only <golden>`, `--repeticoes 5`, `--sementes 1-5` e
  `--estado-limpo`, nos dois modos (`--api` e in-process). Cada resposta vira um caso com o nome
  `md-mtbench-vieses#1` … `#5`, no formato de arquivo de sempre, que os três frameworks já leem.

**Dois controles do viés de estado, didáticos e baratos:**

- **Quente × frio.** A mesma semente responde duas vezes: uma logo depois de outra pergunta
  (quente) e outra com o modelo recarregado (frio). O resultado mostra se o cache mudou a
  resposta, e vai para o relatório como evidência do item 2.
- **Chat limpo no E2E.** Um teste novo, `frontend/tests/e2e/chat-limpo.spec.ts`, faz duas perguntas
  na mesma sessão e confere que o corpo do segundo `POST /chat` só tem `question`. É a garantia de
  regressão de que o histórico nunca chega ao modelo. No vídeo, o palco mostra o que foi enviado.

## 5. Os três juízes, pontuais e repetidos

As 5 respostas vão para o RAGAS, o DeepEval e o JevEval pelo arquivo compartilhado, como hoje.

1. **Três julgamentos por resposta e por juiz, na rodada de análise.** A ordem das respostas é
   embaralhada com semente a cada repetição, porque o juiz também passa pelo cache do Ollama. O
   relatório mostra a média e o mínimo–máximo de cada métrica e separa as duas fontes de variação:
   - entre as 5 respostas, a do gerador;
   - entre os 3 julgamentos da mesma resposta, a do juiz.
2. **Os passos intermediários no export, para a "anatomia de uma nota" (§7):**
   - **RAGAS:** as afirmações e os vereditos. O export de hoje só guarda a nota, então entra um
     invólucro que guarda o que a métrica produziu.
   - **DeepEval:** `claims`, `truths`, `verdicts` e `reason`, que já ficam no objeto da métrica
     depois do `measure`.
   - **Jev:** o `score_breakdown` (probabilidade e confiança por pergunta), que já é gravado.
3. **A estabilidade do Jev com dados reais.** Os 15 julgamentos do Jev (5 respostas × 3) medem a
   variação de verdade e fecham a pendência de 12/11 (0,041 numa medição, contra a tolerância de
   0,02). A tolerância passa a vir do dado.

**Com 5 respostas, a leitura é descritiva:** médias, faixas e concordância, sem p-valor.

## 6. A comparação em pares (melhoria 3)

**Desenho:**

1. **5 respostas, 10 pares, as duas ordens.** Cada par é julgado como (A, B) e como (B, A): são 20
   julgamentos por juiz.
2. **Um critério, escrito antes de rodar (D3):** "Qual resposta responde melhor à pergunta,
   considerando a referência e sem afirmar nada que não esteja nos trechos?"
3. **Três juízes:**
   - **DeepEval:** `ArenaGEval` com o qwen3:8b, dois concorrentes por vez. O teste fixa a ordem e
     mantém os nomes falsos. Ele não aceita empate.
   - **Jev:** um `Choice` com as opções "A", "B" e "Empate", sobre um estado com a pergunta, os
     trechos, a referência e as duas respostas. Ele devolve P(A), P(B) e P(empate). O Jev só vê os
     nomes das opções, nunca os créditos.
   - **RAGAS, por preferência implícita:** A vence B se a média das notas pontuais de A passa a de
     B por mais de uma margem (D4, por exemplo 0,05); senão, empate. A mesma regra também vale
     para as notas pontuais do DeepEval e do Jev.
4. **Onde fica cada parte:**
   - o par do DeepEval, em `tests/deepeval/pairwise.py`;
   - o do Jev, em `tests/jeveval/pairwise.py`, cada um no venv e na versão do seu julgamento
     pontual;
   - a análise, em `tests/reports/`;
   - o export é comum a todos: o par, a ordem, o juiz, o vencedor, as probabilidades quando há, a
     justificativa e os segundos.

**O que se mede:**

- **Ranking das 5 respostas** pela taxa de vitória. O Bradley–Terry é opcional, porque com 5 itens
  a taxa basta.
- **Viés de posição:**
  - a porcentagem de pares com o mesmo vencedor nas duas ordens;
  - a taxa de vitória de quem aparece primeiro.

  É a medida do artigo do MT-Bench.
- **Viés de verbosidade:** a correlação de Spearman entre a diferença de tamanho das respostas e a
  preferência.
- **Concordância entre juízes:** a porcentagem de pares com o mesmo vencedor e o τ de Kendall entre
  os rankings.
- **Pontual × pares, dentro do mesmo framework:** se o DeepEval que dá a nota maior para A também
  escolhe A no par, e o mesmo para o Jev.

**Autoaprimoramento:** fica evitado pelo desenho, porque o gerador (gemma3:4b), o juiz generativo
(qwen3:8b) e o Jev são modelos diferentes. Medir esse viés (o gemma julgando as próprias
respostas) fica como opcional (§13).

## 7. Por que o JevEval dá notas diferentes (o PS)

### 7.1 Dois tipos de juiz

| | RAGAS e DeepEval: juiz generativo | JevEval: juiz de decisão |
| :--- | :--- | :--- |
| Quem julga | qwen3:8b, um LLM que gera texto, local, pelo Ollama | Jev (`jev-1.13.0`, TypeSafe AI), um modelo de decisão, hospedado |
| O que recebe | Um prompt com instruções e exemplos, um por etapa | O estado (os campos do caso: pergunta, resposta, trechos, referência) e perguntas fechadas que nós escrevemos |
| O que devolve | Texto (JSON): as afirmações extraídas, os vereditos e as justificativas | Uma probabilidade para cada opção de cada pergunta |
| Como vira nota | Uma contagem, por exemplo afirmações aprovadas ÷ afirmações | Uma média ponderada das probabilidades, numa conta fixa, feita fora do modelo |
| Chamadas por métrica | Várias: extrair, depois julgar cada afirmação | Uma |
| Julgar de novo | A nota muda: 7 a 9 de 70 respostas mudaram de fidelidade (E0) | A conta não muda. As probabilidades vêm da API e variaram 0,041 numa medição de 12/11 (o §5 mede de novo) |
| Quando não dá para julgar | Erro de formato ou timeout: a métrica sai como não medida | N/A: a pergunta sai da média. Se nenhuma se aplica, a nota vira 1,0, um cuidado já tratado em `tests/jeveval` |
| A explicação | Uma justificativa em texto (DeepEval) | A tabela por pergunta, com probabilidade e confiança |
| Custo e tempo | Sem custo em dinheiro; ~6,5 min de GPU por resposta no DeepEval (19/11) | US$ 0,042 por milhão de tokens de entrada; segundos |
| Os dados | Ficam na máquina | Saem para `api.typesafe.ai` (só o acervo público) |

### 7.2 Mesmo nome, outra pergunta: a fidelidade nos três

- **RAGAS:** uma afirmação só conta se "pode ser inferida diretamente" dos trechos (veredito 1).
  A nota é inferíveis ÷ total.
- **DeepEval:** uma afirmação só perde ponto se **contradiz** os trechos. O "idk" (os trechos não
  dizem) conta como fiel (`deepeval/metrics/faithfulness/faithfulness.py`, `_calculate_score`). A
  nota é não contraditas ÷ total.

  Ele é mais tolerante que o RAGAS: uma afirmação que os trechos não mencionam derruba o RAGAS e
  não derruba o DeepEval.
- **JevEval, como foi escrito em `tests/jeveval/_jev_shared.py`:** duas perguntas.
  - "Toda afirmação factual é sustentada pelos trechos?" Sim ou não, com peso 2.
  - "Quanto da resposta é fundamentado?" Em 4 níveis.

  O que não é mencionado conta como não sustentado. A nota é a média ponderada das
  probabilidades.

### 7.3 O que isso muda na leitura dos resultados

1. **A escala é outra.** 0,67 no RAGAS quer dizer 2 de 3 afirmações. 0,67 no Jev quer dizer que o
   Jev acha provável, mas não certo. As notas generativas saem em frações de poucas afirmações (0,
   0,33, 0,5…); as do Jev são contínuas.
2. **Compare posição, não número.** O que se compara entre juízes é:
   - o veredito (acima ou abaixo do limiar);
   - o ranking das 5 respostas;
   - a escolha no par.

   O relatório já calcula a concordância por dimensão.
3. **A correção só o JevEval mede.** É a comparação com a referência, com limiar 0,5.
4. **N/A não é 0, e erro não é 0.** Os dois ficam fora da média, como o placar já faz.
5. **Num juiz generativo, parte da diferença entre as 5 respostas pode ser do juiz.** Por isso há
   3 julgamentos por resposta.

### 7.4 Onde a explicação aparece

- **README:** a seção "Como cada juiz chega à nota", com a tabela 7.1 e os exemplos 7.2.
- **Relatório:** a "anatomia de uma nota". Para a resposta #1, os passos de cada juiz lado a lado:
  - as afirmações e os vereditos do RAGAS;
  - as afirmações, os vereditos e a justificativa do DeepEval;
  - as perguntas, as probabilidades e a conta do Jev.
- **Palco e vídeo:** um cartão com a diferença, antes do placar.
- **Glossário curto:** juiz generativo, juiz de decisão, N/A, confiança, limiar, preferência
  implícita.

## 8. Mutação na plataforma (ideia do Ícaro)

**Hoje:**
- 18 operadores em 5 camadas, em `tests/mutation/operators/catalog.yaml`: C (corpus), K (chunking),
  E (índice), R (recuperação) e P (prompt).
- O `apply.py` aplica e reverte os operadores sem resíduo.
- Os PDFs variantes ficam em `tests/mutation/corpus/variants/`, fora do git.
- Os oráculos O1 a O5 vivem no estudo; o O6 (Jev) vive em `tests/jeveval`.

**No escopo:** um comando da plataforma que, para a pergunta única:
1. gera o mutante e guarda o artefato gerado (o PDF alterado ou o prompt alterado, ao lado do
   original);
2. roda as 5 respostas com o mesmo protocolo do §4;
3. roda os três juízes;
4. diz qual juiz "matou" o mutante, isto é, reprovou ao menos uma das respostas que passavam no
   original.

**Quatro operadores, escolhidos pela relação com a pergunta (D5):**

| Operador | O que injeta | O comportamento certo |
| :--- | :--- | :--- |
| C1, remover documento | Tira o artigo do MT-Bench do índice | Abster-se ("não encontrei"). No `pdf-v1`, que não tem o artigo, o RAG não se absteve (§3) |
| C2, documento desatualizado | Gera um PDF com os fatos trocados | A resposta muda, e a correção deve cair |
| P1, sem ancoragem no contexto | Gera um prompt sem a ordem de usar só os trechos | A fidelidade deve cair |
| R1, top-k → 1 | Um trecho só | Uma resposta incompleta (menos vieses citados) |

Os operadores P e R não reindexam, então são baratos. O C1 e o C2 reindexam: o custo é medido na
fase 0, e se passar de alguns minutos, o C1 usa um índice temporário só com o acervo da pergunta.

**Saída:** a tabela mutante × juiz (morto ou vivo), com a nota média das 5 respostas antes e
depois.

**Fora do escopo:**
- a tela de mutação no frontend (opcional);
- a campanha completa, que é o artigo do IST e segue no próprio plano (`docs/estudo-mutacao-status.md`).

## 9. O checklist de testes de RAG (ideia do Ícaro)

**Base:** o CheckList (Ribeiro et al., ACL 2020), que organiza testes de comportamento por
capacidade e por tipo:
- **MFT:** funcionalidade mínima;
- **INV:** invariância;
- **DIR:** expectativa de direção.

Aqui, adaptado às etapas de um RAG e ao jeito de avaliar.

**Formato:**
- **Arquivo:** `tests/checklist/checklist.yaml`. Cada item tem a capacidade, o tipo, o que
  verifica, onde é testado (arquivo e teste) e o estado.
- **Relatório:** uma seção com ✓ (passou), ✗ (falhou) ou — (fora do escopo, com o motivo).
- **Fechamento:** todo item precisa terminar coberto, falhando com registro, ou fora do escopo
  declarado. É assim que o checklist fecha o escopo.

**Rascunho dos itens (a revisar com o orientador):**

| Etapa | Item | Tipo | Na pergunta única |
| :--- | :--- | :--- | :--- |
| Recuperação | O artigo certo chega no top-k | MFT | O MT-Bench nos trechos |
| Recuperação | A mesma pergunta traz os mesmos trechos | INV | O hash igual nas 5 respostas |
| Geração | A resposta traz os fatos da referência | MFT | Os 3 vieses e os 80% |
| Geração | Sem afirmação fora dos trechos | MFT | Fidelidade nos 3 juízes |
| Geração | A pergunta em inglês dá os mesmos fatos | INV | O `input_en` do golden |
| Geração | Sem o documento, abster-se | DIR | O mutante C1 |
| Geração | As 5 respostas concordam nos fatos | INV | A correção das 5 |
| Protocolo | O histórico do chat não chega ao modelo | MFT | `chat-limpo.spec.ts` |
| Protocolo | O estado do servidor é zerado antes de cada resposta | MFT | O controle quente × frio |
| Protocolo | Gerador e juiz são modelos diferentes | MFT | `judge-vs-generator.spec.ts` (já existe) |
| Juiz | O juiz acerta os controles com resposta conhecida | MFT | Os controles do RAGAS e do Jev (já existem) |
| Juiz | Trocar a ordem do par não muda o vencedor | INV | §6, consistência de posição |
| Juiz | A resposta mais longa não vence por ser longa | INV | §6, verbosidade |
| Juiz | Julgar de novo dá a mesma nota | INV | §5, 3 julgamentos |
| Juiz | O juiz percebe o defeito injetado | DIR | §8, mutantes mortos |
| Privacidade | Só o acervo público sai da máquina | MFT | O preflight do JevEval (já existe) |

## 10. A apresentação e o relatório

**`tests/apresentacao.py`.** O protocolo da pergunta única vira o padrão. As fases do palco:
1. sobe a aplicação;
2. os testes locais;
3. o E2E faz a pergunta num chat novo e mostra o que foi enviado;
4. as 5 respostas (um cartão por semente);
5. os três juízes (o placar por resposta, depois do cartão do §7);
6. os pares (a matriz 5 × 5 se preenchendo, com o viés de posição);
7. os mutantes;
8. o relatório e o checklist.

A rotina atual, com uma pergunta por acervo nos dois RAGs, continua disponível por opção.

**Duas rodadas:**
- **A de apresentação:** um julgamento por resposta, pares numa ordem só. Ela é gravada.
- **A de análise:** três julgamentos, pares nas duas ordens, os quatro mutantes. Leva horas, não é
  gravada, e é ela que vai para a tese.

**Versão curta do vídeo** (opcional): o mesmo MP4 acelerado nos trechos de espera, para caber num
slide.

**Relatório.** Seções novas:
- o protocolo (o que foi fixado e conferido);
- as 5 respostas lado a lado;
- os juízes, com a decomposição da variação;
- a anatomia de uma nota;
- os pares e os vieses;
- os mutantes;
- o checklist.

## 11. O que ficou aberto e o que fazer com cada coisa

| Pendência | Proposta |
| :--- | :--- |
| PR #4 (`plano-jeveval.md`, só documento) | Mergear, ao lado do `plano.md` e do `plano-benchmark.md` |
| README do JevEval com o estado de 16/10 ("nenhum número medido") | Atualizar: os goldens já rodaram com o Jev em 17/10, 12/11 e 19/11 |
| Estabilidade do Jev fora da tolerância (12/11) | Medir de novo no §5 e fixar a tolerância pelo dado |
| Validade de critério (247 pares) e O6 nunca rodaram com o Jev de verdade | Rodar na fase 9 (~US$ 1 pela estimativa do plano do JevEval) |
| E4 do benchmark (escada de acervo) não construído | Fora do escopo: trabalho futuro |
| HNSW muda o top-k entre processos | Fora do escopo: limitação documentada. O protocolo confere os trechos e não depende disso |
| Memória da VM do Docker (7,6 GB) tira notas do DeepEval | Subir para 12 GB, como o README pede, antes da rodada de análise (configuração sua) |
| Estudo de mutação (IST, submissão em 13/12/2026) | Este plano não mexe na campanha. A ordem das fases depende da agenda do artigo (D7) |

## 12. Fases

Cada fase é um commit, com testes herméticos (juízes falsos) para o CI continuar verde. O que
precisa de Ollama ou do Jev fica marcado como `live`.

| Fase | Entregas | Critério de pronto | Estimativa |
| :--- | :--- | :--- | :--- |
| **0. Preparação** | D1 a D5 decididas. A temperatura padrão do gemma3:4b. O `ArenaGEval` com ordem fixa. O `Choice` do Jev com o estado do par (cabe nos 32k?). O custo de reindexar no C1 e no C2 | Os 4 pontos medidos e anotados aqui; nenhuma premissa do plano sem conferir | 0,5–1 dia |
| **1. Protocolo das 5 respostas** | `seed` opcional no `/chat`, com testes de API. `generate_answers.py` com `--only`, `--repeticoes`, `--sementes` e `--estado-limpo`. A conferência dos trechos. O controle quente × frio. `chat-limpo.spec.ts` | 5 respostas diferentes entre si, iguais ao repetir; mesmos trechos; CI verde | 1–2 dias |
| **2. Juízes repetidos e anatomia** | `--repeticoes-juiz 3` com ordem embaralhada por semente. Os passos intermediários no export dos três | Os três exports com 15 julgamentos e os passos de cada nota | 1–2 dias |
| **3. Comparação em pares** | `tests/deepeval/pairwise.py`, `tests/jeveval/pairwise.py`, a preferência implícita e as métricas do §6, com juízes falsos nos testes | 20 julgamentos por juiz; posição, verbosidade e concordância calculadas | 2–3 dias |
| **4. Relatório** | As seções novas do §10 | `build_report.py` gera MD, PDF e XLSX com tudo; testes verdes | 1–2 dias |
| **5. Explicação dos juízes** | README, a anatomia no relatório, o cartão do palco, o glossário. README do JevEval atualizado | Alguém de fora da turma lê e explica por que o 0,67 do Jev não é o 0,67 do RAGAS | 1 dia |
| **6. Mutação na plataforma** | O comando, os 4 operadores, os artefatos gerados e a tabela mutante × juiz | Os 4 mutantes gerados, revertidos sem resíduo e julgados | 2–3 dias |
| **7. Checklist** | `tests/checklist/checklist.yaml`, a seção do relatório e cada item ligado ao seu teste | Nenhum item sem estado | 1–2 dias |
| **8. Apresentação** | O palco com o novo protocolo, o vídeo e a versão curta | Um comando grava o vídeo da pergunta única | 1–2 dias |
| **9. Rodada de análise e fechamento** | A rodada completa; o ARCHITECTURE com as decisões; o PR #4; a validade de critério e o O6 com o Jev | O checklist todo com estado; os números da tese gerados por um comando | 1 dia mais o tempo de máquina |

**Total:** de 12 a 19 dias de trabalho, sem contar o tempo de máquina das rodadas.

## 13. Fora do escopo

- **Conversa com várias trocas** (`ConversationalJevEval` e afins). A API é de pergunta única por
  desenho, e é isso que garante o chat limpo.
- **Mais perguntas no fluxo principal.** As 10 + 10 perguntas e as 103 do benchmark continuam no
  `run_all.py` e no benchmark, sem mudança.
- **O E4 do benchmark**, a correção do HNSW e a tela de mutação no frontend.
- **Medir o autoaprimoramento** (o gemma julgando a si mesmo). Opcional, se sobrar tempo.
- **Outros geradores ou juízes**, e o ajuste fino de juiz.

## 14. Riscos

| Risco | Efeito | O que fazer |
| :--- | :--- | :--- |
| O ruído do juiz é maior que a diferença entre as 5 respostas | As notas não separam as respostas | É resultado, não falha: o E0 já mostrou que o juiz varia mais que o gerador. Os 3 julgamentos tornam isso visível |
| As 5 respostas saem iguais | Nada para comparar | A fase 0 mede. Se saírem iguais, a temperatura sobe, gravada e justificada |
| O `ArenaGEval` não deixa fixar a ordem | Sem medida de viés de posição | Fixar o `random` no teste. Se não der, um par feito com o `GEval` e ordem explícita |
| O estado do par não cabe no limite do Jev | O par não roda | A fase 0 mede com os trechos reais. Se não couber, o par leva só os trechos citados |
| O DeepEval perde notas por falta de memória | Buracos na tabela | A VM com 12 GB; o erro já aparece como erro, não como 0 |
| As rodadas demoram | O vídeo fica longo | A rodada de análise fica fora do vídeo; a versão curta é opcional |
| A agenda do IST (13/12) | Atrasar o artigo | D7: as fases 0 a 5 cabem antes ou depois, conforme a sua escolha |

## 15. Decisões para você e o orientador

| | Decisão | Proposta |
| :--- | :--- | :--- |
| D1 | A pergunta e o índice | `md-mtbench-vieses` no `md-v1` |
| D2 | Semente por requisição na API (campo opcional, só os testes usam) | Sim |
| D3 | O critério do par e se há empate | O critério do §6; empate onde o juiz aceita (o Jev aceita, o `ArenaGEval` não) |
| D4 | O número de julgamentos e a margem da preferência implícita | 3 julgamentos; margem de 0,05 |
| D5 | Os operadores de mutação | C1, C2, P1, R1 |
| D6 | O que entra na tese e o que fica só no pacote de replicação | Os pares e a decomposição da variação na tese; os mutantes como demonstração |
| D7 | A ordem das fases em relação ao IST | Decidir com o orientador |
