"""O oráculo de fatos da pergunta única: que fatos da referência uma resposta cita, por palavra-chave, sem juiz.

Usado pelo checklist (tests/checklist/avaliar.py) e pela análise dos mutantes (mutacao.py), que precisam saber se a
RESPOSTA mudou, e não só se um juiz reagiu. Só biblioteca padrão.
"""
from __future__ import annotations

import re

# Os fatos da referência de md-mtbench-vieses, por palavra-chave, em português e em inglês. "auto-elogio" e
# "autopromoção" entram porque o gemma traduz assim.
#
# "raciocínio" (a capacidade de raciocínio limitada) entrou em 23/11/2026, DEPOIS de ver a rodada 20261123T181001: está
# na referência desde o começo e tinha ficado de fora da lista (docs/escopo-final-status.md, §10.5). Nas 35 respostas
# daquela rodada, toda menção a raciocínio é sobre essa limitação.
FACTS: dict[str, tuple[str, ...]] = {
    "posição": (r"posi[çc][ãa]o", r"position"),
    "verbosidade": (r"verbos",),
    "autoaprimoramento": (r"self-?enhancement", r"auto-?aprimoramento", r"auto-?elogio", r"auto-?promo[çc][ãa]o",
                          r"auto-?aperfei[çc]oamento", r"auto-?favorecimento"),
    "raciocínio": (r"racioc[íi]nio", r"reasoning"),
    "80%": (r"80\s?%",),
}
# Os fatos que o mutante C2 põe no lugar (mutantes.py, C2_SUBSTITUTIONS): nenhum deles está na referência.
SWAPPED_FACTS: dict[str, tuple[str, ...]] = {
    "recência": (r"rec[êe]ncia", r"recency"),
    "formatação": (r"formata[çc][ãa]o", r"formatting"),
    "autoconsistência": (r"self-?consistency", r"auto-?consist[êe]ncia"),
    "capacidade multilíngue": (r"multil[íi]ng",),
    "60%": (r"60\s?%",),
}


def facts_in(text: str | None, facts: dict[str, tuple[str, ...]] = FACTS) -> set[str]:
    low = (text or "").lower()
    return {fact for fact, patterns in facts.items() if any(re.search(p, low) for p in patterns)}
