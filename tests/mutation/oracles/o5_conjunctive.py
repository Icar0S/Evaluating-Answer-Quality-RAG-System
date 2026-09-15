"""O5 — conjunção O1 ∧ O2, formalizada no §7.1 do protocolo.

    O5(r, g) = O1(r, g) ∧ ⋀ᵢ O2⁽ⁱ⁾(r)

Uma violação determinística reprova o caso **independentemente** do escore
semântico. Isso NÃO é o mesmo que rodar dois oráculos em paralelo e comparar
resultados: em paralelo, um caso com cosseno alto e assertiva violada apareceria
como discordância entre oráculos; em conjunção, ele é uma reprovação. O protocolo
marca essa confusão como o erro a não repetir na figura de arquitetura, e a
implementação segue a mesma regra — O5 tem veredito próprio, não derivado da
comparação entre O1 e O2.

O5 herda a avaliabilidade dos componentes: se O1 não pôde medir (embedding fora
do ar), O5 não emite juízo. Reprovar por falta de medida inflaria a taxa de
mortes com falha de infraestrutura.
"""
from __future__ import annotations

from tests.mutation.oracles.base import Verdict


def evaluate(o1: Verdict, o2: Verdict) -> Verdict:
    if not o1.evaluable or not o2.evaluable:
        return Verdict(
            oracle="O5",
            verdict="pass",
            evaluable=False,
            detail=f"componente não avaliável (O1={o1.evaluable}, O2={o2.evaluable})",
        )

    failed = o1.verdict == "fail" or o2.verdict == "fail"
    reasons = []
    if o1.verdict == "fail":
        reasons.append(f"O1: {o1.detail}")
    if o2.verdict == "fail":
        reasons.append(f"O2: {o2.detail}")

    return Verdict(
        oracle="O5",
        verdict="fail" if failed else "pass",
        score=o1.score,
        detail=" | ".join(reasons),
        components={"o1": o1.verdict, "o2": o2.verdict},
    )
