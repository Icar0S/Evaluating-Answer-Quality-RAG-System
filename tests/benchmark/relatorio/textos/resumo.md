---
titulo: Resumo para quem mantém o RAG
quebra: false
rodadas:
  e9: E9_20261019T014602
---
O assistente responde perguntas sobre 39 artigos de teste de sistemas com LLM. Cada experimento deste relatório
responde a uma pergunta de engenharia sobre ele ("vale trocar o embedding?", "PDF ou Markdown?", "vale pôr um
reranker?") e termina num veredito e num ajuste. A tabela resume os experimentos na ordem em que aparecem. A
seção de cada um traz os gráficos e o porquê.

[[resumo]]

### O ajuste, em forma de `.env`

Os experimentos viram estas linhas no `.env` do backend. A coluna "De onde" diz qual experimento sustenta
cada uma. Uma linha que só confirma o padrão do repositório ("manter") também está aqui, porque o teste é o
que dá a ela um motivo.

[[ajustes]]

> **O problema que sobrava, e quem resolveu**
> Com a busca boa (E1), o `gemma3:4b` passa a responder perguntas de fora do acervo que são do mesmo tema
> (JaCoCo, TMMi, Lighthouse), às vezes dizendo que a resposta veio dos documentos. Nenhum limiar de
> similaridade barra essas perguntas, porque elas se parecem com o acervo (E6). O E9 mostrou que o problema é
> do gerador: com o mesmo prompt e a mesma busca, o `llama3.1:8b` recusa
> {{contagem e9 llama abstention.correct_abstention}} perguntas de fora, e o `gemma3:4b` recusa
> {{contagem e9 gemma abstention.correct_abstention}}.
