---
titulo: O que este relatório ainda não sustenta
---
Um benchmark honesto diz onde ele pode estar errado. Estas são as ressalvas que valem para todo o relatório, em
ordem de peso.

- **As perguntas são rascunho.** As 103 perguntas e os gabaritos foram escritos com ajuda de um LLM e ainda não
  passaram pela revisão humana que o plano exige. Uma pergunta com gabarito errado conta como erro do sistema.
  Revisadas as perguntas, todos os experimentos rodam de novo, e este relatório é gerado de novo pelo mesmo
  comando.
- **A regra de veredito é provisória.** A margem de ruído antes do intervalo de confiança é a regra usada; a
  seção do plano que a fixa ainda não chegou e pode mudar alguns vereditos de fronteira, sobretudo os
  inconclusivos.
- **Poucas perguntas, intervalos largos.** São 70 perguntas de dentro, em 21 artigos, e 30 de fora. Uma diferença
  de duas ou três perguntas quase nunca sai do inconclusivo. Os experimentos com recorte (E2, E6) usam metade
  disso em cada etapa.
- **Duas análises foram definidas depois de ver os dados.** A separação das recusas pela evidência recebida (E9)
  e o preço da citação, que compara rodadas diferentes (E3), não estavam nos cartões. Elas explicam o resultado,
  mas precisam de confirmação com perguntas novas antes de virar regra.
- **O detector de recusa tem um erro conhecido.** Uma resposta que escreve tudo e termina com a frase de recusa
  conta como recusa (o caso do script, no E9). Separar isso de uma recusa com explicação pede leitura caso a caso.
- **A recuperação varia um pouco entre processos.** Com menos de 1.000 trechos, o índice de busca é reconstruído
  a cada processo, e os empates mudam de ordem: cerca de uma pergunta de diferença entre duas rodadas iguais
  (E0). A margem de ruído da recuperação já conta isso.
- **As latências são desta máquina.** Elas servem para comparar os braços entre si, e não como promessa de
  tempo em outra máquina.
- **O E4 não rodou**, e a pergunta "quanto acervo um chatbot deve ter?" fica sem resposta neste relatório.
- **Os títulos dos artigos (E3) são rascunho**, tirados dos metadados dos PDFs, sem revisão.
