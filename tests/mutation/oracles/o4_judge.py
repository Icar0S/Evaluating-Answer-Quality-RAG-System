"""O4 — LLM-as-a-judge: 5 julgamentos, maioria simples (§6.4).

Custo alto, roda só sobre a resposta modal das N execuções. Essa assimetria é
deliberada e é *parte do resultado da RQ3*, não um atalho escondido: reduz as
chamadas de juiz de 13.500 para 2.700 e o artigo reporta o número.

Duas mitigações, e o que sobra delas:
- família distinta da do SUT (qwen3 gera, gemma3 julga) contra viés de
  autopreferência;
- 5 julgamentos com maioria contra instabilidade do próprio juiz.
O viés residual depois das duas é justamente o que a RQ2 mede.

Sobre a temperatura do juiz: com temperature=0 (o padrão do study.yaml), os 5
julgamentos medem o não-determinismo residual do runtime, não variância de
amostragem. É uma escolha conservadora — se os 5 concordarem sempre, isso é um
dado sobre o juiz, e a maioria simples degenera em julgamento único sem custo
metodológico. Subir a temperatura do juiz mede outra coisa e precisaria ser
declarado no artigo.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import httpx

from tests.mutation.oracles.base import Verdict
from tests.mutation.runner import sut  # noqa: F401  (garante backend no sys.path)
from app.config import get_settings

# Causas que um juiz consegue enxergar olhando SÓ pergunta/referência/resposta.
# São sugestões para a classificação de coerência (§7.4), não o codebook: os
# códigos RET-* exigem olhar o que foi recuperado, o que o juiz não vê.
CAUSE_OPTIONS = (
    "missing_info",
    "wrong_value",
    "unsupported_claim",
    "no_abstention",
    "no_citation",
    "off_topic",
    "none",
)

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "cause": {"type": "string", "enum": list(CAUSE_OPTIONS)},
        "justification": {"type": "string"},
    },
    "required": ["verdict", "cause", "justification"],
}

SYSTEM_PROMPT = """Você avalia respostas de um assistente de perguntas e respostas sobre documentos técnicos.

Receberá a PERGUNTA, a RESPOSTA DE REFERÊNCIA (correta, escrita por um especialista) e a RESPOSTA SOB AVALIAÇÃO.

Decida se a RESPOSTA SOB AVALIAÇÃO é aceitável:
- "pass": transmite a mesma informação essencial da referência, sem contradizê-la e sem acrescentar afirmação não sustentada. Diferenças de redação, ordem ou detalhe não essencial não reprovam.
- "fail": omite informação essencial, contradiz a referência, inventa dado, responde outra pergunta, ou deixa de fazer o que a referência faz (por exemplo: a referência se abstém e ela responde, ou a referência cita a fonte e ela não cita).

Escolha também a causa predominante, entre: missing_info, wrong_value, unsupported_claim, no_abstention, no_citation, off_topic, none (use "none" quando o veredito for "pass").

Responda SOMENTE com o objeto JSON pedido, em português, com justificativa de no máximo duas frases."""


def _build_user_message(question: str, reference: str, answer: str) -> str:
    return (
        f"PERGUNTA:\n{question}\n\n"
        f"RESPOSTA DE REFERÊNCIA:\n{reference}\n\n"
        f"RESPOSTA SOB AVALIAÇÃO:\n{answer or '(resposta vazia)'}"
    )


def _judge_once(
    question: str,
    reference: str,
    answer: str,
    model: str,
    temperature: float,
    seed: int,
    base_url: str,
    timeout: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(question, reference, answer)},
        ],
        "stream": False,
        "format": JUDGE_SCHEMA,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192},
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(f"{base_url}/api/chat", json=payload)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")

    parsed = json.loads(content)
    if parsed.get("verdict") not in ("pass", "fail"):
        raise ValueError(f"veredito inválido do juiz: {parsed!r}")
    return parsed


def evaluate(
    question: str,
    reference: str,
    answer: str,
    model: str,
    judgments: int = 5,
    temperature: float = 0.0,
    base_url: str | None = None,
    timeout: int = 300,
) -> Verdict:
    settings = get_settings()
    # Juiz sempre no Ollama local, mesmo quando o SUT roda em outro provider:
    # trocar o juiz no meio da campanha invalidaria a comparação entre mutantes.
    url = base_url or settings.ollama_base_url

    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for index in range(judgments):
        try:
            results.append(
                _judge_once(question, reference, answer, model, temperature, index + 1, url, timeout)
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    if not results:
        return Verdict(
            oracle="O4",
            verdict="pass",
            evaluable=False,
            detail=f"juiz indisponível: {errors[:1]}",
        )

    verdicts = [r["verdict"] for r in results]
    counts = Counter(verdicts)
    majority, majority_count = counts.most_common(1)[0]
    causes = Counter(r.get("cause", "none") for r in results if r["verdict"] == "fail")
    top_cause = causes.most_common(1)[0][0] if causes else "none"
    justification = next(
        (r.get("justification", "") for r in results if r["verdict"] == majority),
        "",
    )

    return Verdict(
        oracle="O4",
        verdict=majority,
        score=round(counts.get("pass", 0) / len(results), 4),
        detail=f"{majority_count}/{len(results)} juízes; causa={top_cause}; {justification}",
        components={
            "judgments": len(results),
            "agreement": round(majority_count / len(results), 4),
            "cause": top_cause,
            "errors": len(errors),
            "model": model,
        },
    )
