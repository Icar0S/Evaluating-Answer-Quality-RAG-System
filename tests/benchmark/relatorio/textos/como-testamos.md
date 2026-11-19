---
titulo: Como o teste foi feito
rodadas:
  e1: E1_20261018T141628
  e0: E0_20261018T082340
figuras:
  exemplo:
    tipo: diferencas
    rodada: e1
    pares: [nomic-pt→bgem3-pt, nomic-pt→traducao]
    rotulos:
      nomic-pt→bgem3-pt: "embedding nomic → bge-m3"
      nomic-pt→traducao: "nomic → nomic com tradução"
    tabela: false
    legenda: >
      Exemplo, tirado do E1: o quanto a evidência nos trechos muda quando se troca o embedding (primeira linha) ou
      se traduz a pergunta (segunda), com perguntas em português.
    como_ler: >
      Cada linha compara dois braços. O ponto é a diferença (segundo menos primeiro); o traço em volta é o
      intervalo de confiança de 95%, a faixa onde a diferença verdadeira provavelmente está. A faixa cinza em volta
      do zero é a margem de ruído: o quanto a nota muda sozinha, rodando a mesma coisa duas vezes (E0). Um ponto
      dentro da faixa cinza não é diferença. Um traço inteiro de um lado do zero, com o ponto fora da faixa, é. À
      direita, o veredito, com um ícone além da cor.
---
Esta seção é para quem quer entender **como** a conclusão foi tirada, e não só qual foi. Ela vale para todos os
experimentos; cada seção seguinte só diz o que mudou.

### O assistente em quatro passos

1. **Ingestão.** Os 39 artigos (PDFs, quase todos em inglês) são cortados em trechos de ~800 tokens. Cada trecho
   vira um vetor de números pelo **embedding**, um modelo que põe textos parecidos perto uns dos outros.
2. **Busca.** A pergunta do aluno vira um vetor do mesmo jeito, e o sistema pega os 4 trechos mais perto dela
   (o `top_k`).
3. **Geração.** O gerador (`gemma3:4b`) lê os 4 trechos e responde. Se os trechos não têm a resposta, o prompt
   manda ele dizer "Não encontrei essa informação nos documentos fornecidos".
4. **Avaliação.** Um segundo modelo, o **juiz** (`qwen3:8b`), dá notas à resposta: ela se apoia nos trechos? Ela
   responde a pergunta?

Cada experimento mexe em **uma** dessas peças e deixa o resto fixo.

### O funil: medir barato em tudo, caro só no fim

O teste é feito em camadas. A camada de baixo mede sem o juiz e roda em segundos; a de cima precisa do juiz e
leva horas. Um braço que perde feio embaixo não precisa subir.

| Camada | A pergunta | Como mede | Custo |
|---|---|---|---|
| C0 · Acervo | O texto dos artigos está bom? | contagens no texto extraído | segundos |
| C1 · Recuperação | O trecho certo chega ao gerador? | a evidência do gabarito aparece nos 4 trechos? | segundos a minutos |
| C2 · Resposta | A resposta usa bem o que chegou? | o juiz (RAGAS) e a contagem de recusas | ~1 min por resposta |
| C4 · Custo | Quanto custa? | latência e tokens de cada resposta | de graça, vem junto |

A C1 decide sem juiz, de propósito: o juiz já se mostrou ruim em ordenar versões do índice, e a evidência nos
trechos é uma conta exata.

### As perguntas e o gabarito

O benchmark tem 103 perguntas (os "goldens"). **70 são de dentro**: a resposta está num dos artigos, e o gabarito
guarda o trecho exato que a sustenta. **30 são de fora**: a resposta não está no acervo, e o certo é o assistente
recusar. Três são ambíguas ("Qual é a melhor métrica?"), e o certo seria pedir esclarecimento. As perguntas
são em português; o acervo, em inglês.

### Quatro regras que valem para todo experimento

- **Uma variável por vez.** Cada braço é a referência com uma mudança. Se o braço ganha, foi aquela mudança.
- **As mesmas perguntas em todos os braços.** A comparação é pareada: cada pergunta é comparada com ela mesma
  no outro braço. Isso tira da conta a dificuldade de cada pergunta.
- **O ruído é medido antes do sinal.** Rodar a mesma coisa duas vezes não dá o mesmo número: o juiz é um LLM e
  varia. O E0 mede quanto, e essa é a **margem de ruído**: diferença menor que ela não conta.
- **A regra é escrita antes de rodar.** Cada experimento tem um **cartão**: a pergunta, a hipótese, a métrica
  que decide e o que não pode piorar (as **guardas**). O relatório mostra o cartão como foi escrito, para
  ninguém escolher a métrica depois de ver o resultado.

### Como ler um veredito

O veredito sai de duas coisas: o tamanho da diferença e a certeza sobre ela.

| Veredito | Quando |
|---|---|
| ✓ melhor / ✗ pior | a diferença passa da margem de ruído, e o intervalo de confiança não cruza o zero |
| ? inconclusivo | a diferença passa da margem, mas o intervalo cruza o zero: pode ser sorte, faltam perguntas |
| ≈ sem diferença relevante | a diferença cabe na margem de ruído, seja qual for o intervalo |

O intervalo de confiança é calculado sorteando **documentos**, e não perguntas, porque as perguntas de um mesmo
artigo erram juntas. Com 70 perguntas em 21 artigos, os intervalos são largos, e é por isso que tantos
vereditos saem inconclusivos.

[[figura exemplo]]

No exemplo, a troca do embedding sobe a evidência em {{dif e1 nomic-pt→bgem3-pt c1.evidence_at_k}}, muito acima
da margem de ruído da recuperação ({{ruido e0 c1 c1.evidence_at_k}}), e o intervalo inteiro fica do lado
positivo: veredito **{{veredito e1 nomic-pt→bgem3-pt}}**.
