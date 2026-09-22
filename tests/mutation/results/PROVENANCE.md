# Proveniência dos logs de execução

Cada campanha deste estudo deixou um log. Alguns foram invalidados por defeitos de
instrumentação descobertos depois, e **nenhum foi apagado**: um log inválido é a
evidência da ameaça à validade que ele revelou, e o artigo o cita como tal.
Este arquivo diz o que cada um é, com que código foi produzido e por que foi
superado. É o índice do que vai para o Zenodo.

Os logs em si **não são versionados** (`.gitignore`: `tests/mutation/results/*`) —
são grandes e regeneráveis pelos comandos do README. O que o repositório versiona é
este índice, com os hashes, mais o código, a configuração, a suíte e a calibração.

Gerado em 2026-09-22T09:45
com o repositório em `4590c49`.

## `pilot/runs.jsonl`

- **Quando:** 2026-09-16  
- **Código:** `e3fae9e`  
- **O que é:** Piloto da semana 5: 3 casos (c01, c07, c16) x (baseline, R1, P2) x 2 repetições.  
- **Estado:** Rodou com o prompt TRUNCADO (ver §4.13) e com as perguntas anteriores de c01. Serve como evidência do piloto e da calibração do O4, não como medida do SUT.  
- **Conteúdo:** {'linhas': 18, 'casos': 3, 'mutantes': 3, 'erros': 0, 'wall_mediana_s': 17.8, 'wh_total': 5.9}  
- **sha256 (runs):** `8f8926d22ba90180328c48a87a7ab1de96d5cee65626852e5d30e4be29eac41e`  
- **Vereditos:** `pilot/verdicts.jsonl`, 45 linhas, sha256 `923afaeb596d958ebc502209cdfda9e0ac9cdba8ff2b844ef9b2156ed17497fe`  

## `archive_v1_truncated/runs_baseline_v1.jsonl`

- **Quando:** 2026-09-21 17:21–19:41  
- **Código:** `bfcd105`  
- **O que é:** Baseline v1 completo, 30 casos x 10 repetições.  
- **Estado:** INVÁLIDO: as 300 invocações rodaram sem `num_ctx` declarado e tiveram o prompt cortado em 2.050 tokens (288 avisos `truncating input prompt` no log do Ollama). O system prompt não chegou ao modelo. Mantido porque é a evidência do achado §4.13 e da ameaça do §9.  
- **Conteúdo:** {'linhas': 300, 'casos': 30, 'mutantes': 1, 'erros': 0, 'wall_mediana_s': 26.2, 'wh_total': 142.6}  
- **sha256 (runs):** `0e31dbdd018b77847a58e892d1564beb203ba04d212cbf7f4df523391404b141`  
- **Vereditos:** `archive_v1_truncated/verdicts_baseline_v1.jsonl`, 120 linhas, sha256 `395328019d1c55766bad27d159efbc1e39c257ca73bd27756671e41fc93d4ac6`  

## `archive_v1_truncated/runs_partial_v2.jsonl`

- **Quando:** 2026-09-21 20:22–20:29  
- **Código:** `859a8b9`  
- **O que é:** Tentativa parcial depois da reescrita de 8 perguntas, ainda sem `num_ctx`.  
- **Estado:** INVÁLIDA pelo mesmo motivo. 8 invocações.  
- **Conteúdo:** {'linhas': 231, 'casos': 30, 'mutantes': 1, 'erros': 0, 'wall_mediana_s': 23.0, 'wh_total': 103.5}  
- **sha256 (runs):** `f80680c6dc86838f54134e89b6cd84a724de9441035bd417ede7efc464c07240`  
- **Vereditos:** `archive_v1_truncated/verdicts_partial_v2.jsonl`, 88 linhas, sha256 `b9433f82a44c024ae182920deafa51a2c29178bc705a9fb0452a4287d2049b6a`  

## `archive_v2_fixed_order/runs.jsonl`

- **Quando:** 2026-09-21 21:08–23:23  
- **Código:** `ed20e49`  
- **O que é:** Baseline v2: primeiro baseline sem truncagem. 30 casos x 10 repetições.  
- **Estado:** VÁLIDO como medida, SUPERSEDIDO como amostra: os casos rodaram em ordem fixa, e a ordem fixa faz r2..r10 repetirem o mesmo predecessor no cache do servidor (respostas byte a byte iguais). |S| = O1 25, O4 23, O2 21, O5 21. É o par de comparação do v3.  
- **Conteúdo:** {'linhas': 300, 'casos': 30, 'mutantes': 1, 'erros': 0, 'wall_mediana_s': 22.6, 'wh_total': 108.7}  
- **sha256 (runs):** `5972acfc401fd48d8fdb0ae7241c0e01e521c63c526e37a9305000a95313965c`  
- **Vereditos:** `archive_v2_fixed_order/verdicts.jsonl`, 178 linhas, sha256 `77cbed8fa7ca45c2fb6d7e70e6720a15ac6d59978b47eac51e01c942f7fecc89`  

## `runs.jsonl`

- **Quando:** 2026-09-22 00:04 em diante  
- **Código:** `4590c49`  
- **O que é:** Baseline v3 (ordem embaralhada por repetição) e, na sequência, a campanha dos 18 mutantes.  
- **Estado:** EM CURSO. Duas anomalias já registradas, ambas tratadas no commit desta data: `baseline-c18-r1` atravessou uma suspensão da máquina (7,3 h de relógio de parede, 58,4 Wh medidos — custo inválido, resposta válida) e 6 execuções de c14 bateram o teto de geração com resposta vazia.  
- **Conteúdo:** {'linhas': 225, 'casos': 30, 'mutantes': 1, 'erros': 0, 'wall_mediana_s': 22.4, 'wh_total': 151.7}  
- **sha256 (runs):** `21d1465b3d79987936abe3a1ee646aa55da91cbce111c2bdc5f8e5ee762feb0b`  
- **Vereditos:** `verdicts.jsonl` ainda não gerado  

## Como refazer cada um

```bash
# v1 (truncado) — só para reproduzir o achado §4.13:
#   remova generation_num_ctx/generation_num_predict de config/study.yaml
#   e observe `truncating input prompt` no log do servidor Ollama.
# v2 (ordem fixa):
#   shuffle_cases_per_repetition: false em config/study.yaml
python -m tests.mutation.run_campaign --baseline
# v3 (ordem embaralhada, configuração atual):
python -m tests.mutation.run_campaign --baseline
python -m tests.mutation.run_campaign --campaign
python -m tests.mutation.evaluate
```

Cada etapa é retomável: `run_id` já presente em `runs.jsonl` é pulado.
