"""Traduz uma vez as perguntas em português dos goldens, para o braço de tradução do E1 (plano-benchmark.md).

A recuperação lê a tradução gravada aqui (RETRIEVAL_TRANSLATION_CACHE), e a C1 e
a C2 usam a mesma. O tempo de cada tradução fica gravado: com o cache, a busca
não paga a chamada de LLM, e o custo do braço precisa somá-lo de volta.
Retomável: pergunta já traduzida não é traduzida de novo.

A tradução não é corrigida à mão: corrigida, ela viraria o `input_en` escrito à
mão, que é o braço de teto. Cada tradução ganha marcas automáticas de defeito
(`flags`), e o resumo conta quantas há.

    backend\\.venv\\Scripts\\python.exe tests\\benchmark\\translate_questions.py --modelo gemma3:4b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings, set_settings_overrides  # noqa: E402
from app.rag import translation  # noqa: E402

DEFAULT_GOLDENS = ROOT / "tests" / "benchmark" / "goldens" / "v2.json"
DEFAULT_OUT = ROOT / "tests" / "benchmark" / "goldens" / "v2_traducoes.json"
# Palavras que não existem em inglês: duas ou mais na tradução indicam um trecho que ficou em português.
PT_ONLY = {"da", "das", "dos", "com", "para", "foram", "foi", "qual", "quais", "não", "são", "pelo", "pela", "nas",
           "nos", "seu", "sua", "uma", "que", "ao", "aos", "entre", "sobre", "quantos", "quantas", "como", "em"}


def quality_flags(question: str, translated: str) -> list[str]:
    """Defeitos que dá para ver sem saber a tradução certa.

    - `curta`: menos da metade do tamanho da pergunta, o jeito de uma resposta ("H2O", "Brazil");
    - `sem_pergunta`: a pergunta termina em "?" e a tradução não tem nenhum;
    - `portugues`: duas ou mais palavras que só existem em português;
    - `prompt`: a tradução repete a instrução do tradutor (pergunta de injeção).
    """
    flags = []
    if len(translated) < 0.5 * len(question):
        flags.append("curta")
    if question.rstrip().endswith("?") and "?" not in translated:
        flags.append("sem_pergunta")
    if sum(word in PT_ONLY for word in re.findall(r"[a-zà-ú]+", translated.lower())) >= 2:
        flags.append("portugues")
    if "translation engine" in translated.lower() or "from portuguese into english" in translated.lower():
        flags.append("prompt")
    return flags


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Traduz as perguntas dos goldens uma vez, para o E1.")
    parser.add_argument("--goldens", type=Path, default=DEFAULT_GOLDENS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--modelo", default=None, help="modelo do Ollama local (padrão: o gerador)")
    args = parser.parse_args()

    if args.modelo:
        set_settings_overrides({"retrieval_translation_model": args.modelo})
    model = get_settings().retrieval_translation_model or get_settings().generation_model
    goldens = json.loads(args.goldens.read_text(encoding="utf-8"))
    data = json.loads(args.out.read_text(encoding="utf-8")) if args.out.is_file() else {}
    if data and (data.get("model"), data.get("prompt_version")) != (model, translation.PROMPT_VERSION):
        print(f"[ERRO] {args.out} foi traduzido com {data.get('model')} e o prompt {data.get('prompt_version')}, "
              f"não com {model} e o {translation.PROMPT_VERSION}. Use outro --out ou apague o arquivo.",
              file=sys.stderr)
        return 1
    translations: dict[str, str] = data.get("translations", {})
    timings: dict[str, float] = data.get("milliseconds", {})
    questions = [g["input"] for g in goldens if g.get("input") and g["input"] not in translations]
    print(f"{len(questions)} pergunta(s) a traduzir com {model} ({len(translations)} já traduzidas).", flush=True)
    if questions:
        # A primeira chamada carrega o modelo; medida, ela poria segundos a mais numa pergunta só.
        translation.translate_with_llm("Olá")
    flags = {question: quality_flags(question, text) for question, text in translations.items()}
    for i, question in enumerate(questions, start=1):
        t0 = time.perf_counter()
        translations[question] = translation.translate_with_llm(question)
        timings[question] = round((time.perf_counter() - t0) * 1000, 1)
        flags[question] = quality_flags(question, translations[question])
        mark = f" [{', '.join(flags[question])}]" if flags[question] else ""
        print(f"  [{i}/{len(questions)}] {timings[question]:.0f} ms{mark}: {translations[question][:90]}", flush=True)
        args.out.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "prompt_version": translation.PROMPT_VERSION,
            "goldens": args.goldens.relative_to(ROOT).as_posix() if args.goldens.is_relative_to(ROOT) else str(args.goldens),
            "goldens_sha256": hashlib.sha256(args.goldens.read_bytes()).hexdigest(),
            "translations": translations,
            "milliseconds": timings,
            "flags": {q: f for q, f in flags.items() if f},
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flagged = {q: f for q, f in flags.items() if f}
    print(f"\n{len(flagged)} de {len(translations)} tradução(ões) com marca de defeito.")
    for question, marks in flagged.items():
        print(f"  [{', '.join(marks)}] {question[:80]}\n      → {translations[question][:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
