"""O oráculo de fatos da pergunta única: que fatos da referência uma resposta cita, por palavra-chave, sem juiz.

Usado pelo checklist (tests/checklist/avaliar.py) e pela análise dos mutantes (mutacao.py), que precisam saber se a
RESPOSTA mudou, e não só se um juiz reagiu. Só biblioteca padrão.
"""
from __future__ import annotations

import re

# Os fatos da referência de md-mtbench-vieses, por palavra-chave, em português e em inglês. "auto-elogio" e
# "autopromoção" entram porque o gemma traduz assim.
FACTS: dict[str, tuple[str, ...]] = {
    "posição": (r"posi[çc][ãa]o", r"position"),
    "verbosidade": (r"verbos",),
    "autoaprimoramento": (r"self-?enhancement", r"auto-?aprimoramento", r"auto-?elogio", r"auto-?promo[çc][ãa]o",
                          r"auto-?aperfei[çc]oamento", r"auto-?favorecimento"),
    "80%": (r"80\s?%",),
}
# Os fatos que o mutante C2 põe no lugar (mutantes.py, C2_SUBSTITUTIONS): nenhum deles está na referência.
SWAPPED_FACTS: dict[str, tuple[str, ...]] = {
    "recência": (r"rec[êe]ncia", r"recency"),
    "formatação": (r"formata[çc][ãa]o", r"formatting"),
    "autoconsistência": (r"self-?consistency", r"auto-?consist[êe]ncia"),
    "60%": (r"60\s?%",),
}


def facts_in(text: str | None, facts: dict[str, tuple[str, ...]] = FACTS) -> set[str]:
    low = (text or "").lower()
    return {fact for fact, patterns in facts.items() if any(re.search(p, low) for p in patterns)}
