---
titulo: Glossário
---
Os termos do relatório, na ordem em que aparecem num experimento.

| Termo | O que é |
|---|---|
| RAG | o assistente que busca trechos de um acervo e responde com eles, em vez de responder só de memória |
| acervo | os 39 artigos que o assistente consulta |
| trecho (chunk) | um pedaço de ~800 tokens de um artigo; é a unidade que a busca devolve |
| embedding | o modelo que transforma um texto num vetor de números, de modo que textos parecidos ficam perto (`nomic-embed-text`, `bge-m3`) |
| `top_k` | quantos trechos a busca entrega ao gerador (4, no padrão) |
| gerador | o LLM que lê os trechos e escreve a resposta (`gemma3:4b`, `llama3.1:8b`) |
| juiz | o LLM que dá as notas das respostas (`qwen3:8b`, pelo RAGAS) |
| golden / gabarito | uma pergunta do benchmark com a resposta esperada e o trecho exato do artigo que a sustenta |
| pergunta de dentro / de fora | de dentro: a resposta está no acervo; de fora: não está, e o certo é recusar |
| braço | uma configuração testada: a referência com uma mudança |
| referência (baseline) | o braço contra o qual os outros são comparados |
| cartão | a ficha do experimento, escrita antes de rodar: pergunta, hipótese, braços, métrica que decide e guardas |
| guarda | uma métrica que não pode piorar além de uma tolerância, mesmo que a principal melhore |
| rodada A/A | rodar a mesma configuração duas vezes, para medir quanto o resultado muda sozinho |
| margem de ruído | o tamanho de uma diferença que aparece por acaso numa rodada A/A (o percentil 95); diferença menor que ela não conta |
| Δ (delta) | a diferença entre dois braços: o segundo menos o primeiro, nas mesmas perguntas |
| intervalo de confiança (IC 95%) | a faixa onde a diferença verdadeira provavelmente está; calculada sorteando artigos, porque as perguntas de um artigo erram juntas |
| evidência nos trechos | a fração das perguntas de dentro em que o trecho do gabarito chegou inteiro aos trechos entregues ao gerador |
| documento certo nos trechos (Hit@k) | a fração das perguntas em que pelo menos um trecho é do artigo da resposta |
| precisão dos trechos | quantos dos trechos entregues são do artigo certo ou têm a evidência |
| MRR | a posição do primeiro trecho do artigo certo na lista, como 1/posição: 1 se é o primeiro, ½ se é o segundo |
| fidelidade | a nota do juiz para "a resposta se apoia nos trechos?": a fração das afirmações da resposta que os trechos sustentam |
| relevância da resposta | a nota do juiz para "a resposta responde a pergunta?" |
| abstenção correta | a fração das perguntas de fora que o assistente recusa |
| recusa indevida | a fração das perguntas de dentro que o assistente recusa |
| recusa com / sem a evidência | a recusa indevida separada pelo que a busca entregou: com a evidência no prompt, recusar é cautela demais; sem ela, é o certo |
| atribuição errada | a resposta que cita uma fonte que não recebeu, ou que cita algo numa pergunta de fora |
| limiar de similaridade | o corte abaixo do qual o sistema nem tenta responder |
| AUC | o quanto uma nota separa dois grupos: 1 é separação perfeita, 0,5 é o acaso |
| MMR | um jeito de diversificar os trechos, trocando trechos parecidos entre si por trechos de outros lugares |
| BM25 | a busca por palavras, a de um buscador clássico |
| busca híbrida (RRF) | junta a lista do embedding e a do BM25 numa só |
| reranker | um modelo que relê cada par pergunta-trecho e reordena a lista |
| curva de Lorenz | o gráfico de quanto os trechos se concentram em poucos documentos |
| ímã | um documento que aparece nos trechos de perguntas sobre outros documentos muito mais do que o acaso daria |
| recorte de seleção / confirmação | as perguntas divididas em dois grupos: um escolhe os candidatos, o outro confirma |
