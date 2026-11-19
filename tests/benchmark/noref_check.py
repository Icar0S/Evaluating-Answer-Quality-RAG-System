"""Antes do H5b (E5, versão pdf-v1-noref): nenhuma evidência dos goldens pode cair no bloco de referências cortado.

Para cada golden de dentro do pdf-v1, cada fragmento do `context` é casado (25-gramas, tests/benchmark/c1.py)
com o texto inteiro do documento e com o texto sem o bloco (ingestion.strip_references). Fragmento que casa no
inteiro e não casa sem o bloco seria evidência perdida: o corte mudaria a pergunta, e não só o ruído.
Sai com código 1 se achar algum. Roda no venv do backend (lê os PDFs, sem Ollama):

    backend\\.venv\\Scripts\\python.exe -m tests.benchmark.noref_check
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import PROJECT_ROOT, RAG_INDICES  # noqa: E402
from app.rag import ingestion  # noqa: E402
from tests.benchmark import c1  # noqa: E402

GOLDENS = ROOT / "tests" / "benchmark" / "goldens" / "v2.json"


def lost_evidence(goldens: list[dict], folder: Path, documents: dict[str, str]) -> tuple[int, list[dict]]:
    """(fragmentos conferidos, os que o corte perderia). `documents`: nome sem extensão → arquivo da versão."""
    checked, lost, texts = 0, [], {}
    for golden in goldens:
        source = documents.get(c1.doc_stem(golden.get("source_document")))
        if not source:
            continue
        if source not in texts:
            doc = ingestion.read_document(folder / source)
            texts[source] = (doc.text, ingestion.strip_references(doc)[0].text)
        full, stripped = texts[source]
        for fragment in c1.split_fragments(golden.get("context")):
            checked += 1
            if c1.evidence([fragment], [full])["evidence"] and not c1.evidence([fragment], [stripped])["evidence"]:
                lost.append({"golden": golden["name"], "document": source, "fragment": fragment[:120]})
    return checked, lost


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    spec = RAG_INDICES["pdf-v1-noref"]
    documents = {c1.doc_stem(name): name for name in ingestion.registered_documents("pdf-v1-noref") or {}}
    goldens = json.loads(GOLDENS.read_text(encoding="utf-8"))
    checked, lost = lost_evidence(goldens, PROJECT_ROOT / spec["source_dir"], documents)
    print(f"{checked} fragmento(s) de evidência conferidos; {len(lost)} cairiam no bloco cortado.")
    for item in lost:
        print(f"  {item['golden']} ({item['document'][:50]}): {item['fragment']}")
    return 1 if lost else 0


if __name__ == "__main__":
    sys.exit(main())
