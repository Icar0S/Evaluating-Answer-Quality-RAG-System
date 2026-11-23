"""Os mutantes da pergunta única (plano-escopo-final.md, §8): prepara cada um e guarda o que foi gerado.

Quatro operadores do catálogo do estudo de mutação (tests/mutation/operators/catalog.yaml), escolhidos pela
relação com a pergunta (D5):

| Operador | O que injeta | O comportamento certo |
| :--- | :--- | :--- |
| C1, remover documento | O artigo do MT-Bench sai do índice | Abster-se ("não encontrei") |
| C2, documento desatualizado | O artigo com os fatos trocados | A resposta muda, e a correção cai |
| P1, sem ancoragem no contexto | O prompt sem a regra de usar só os trechos | A fidelidade cai |
| R1, top-k -> 1 | Um trecho só | Uma resposta incompleta |

Os de corpus (C1, C2) não reindexam o acervo: o índice do mutante é uma cópia dos vetores do `md-v1` sem os
trechos do artigo (C1) ou com os trechos do artigo trocados e reembedados (C2), numa pasta da rodada. A coleção
`rag_md_v1` não é tocada, e nada fica em data/vector_store: o mutante é revertido apagando a pasta. Os de
configuração (P1, R1) são sobrescritas de Settings, em memória, no processo que gera as respostas.

Cada mutante grava, em <pasta>/artefatos/, o que foi gerado: o Markdown do artigo mutado e as substituições
(C2), o prompt de sistema antes e depois (P1), a lista do que saiu (C1), a configuração (R1). E grava
<pasta>/mutante.json, que o rodar.py lê para gerar as respostas com as sobrescritas certas.

Roda no venv do backend (importa o pipeline):
    backend\\.venv\\Scripts\\python.exe tests\\pergunta_unica\\mutantes.py preparar --op C2 --dir <pasta>
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _path in (ROOT / "backend", ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

BASE_INDEX = "md-v1"
DOCUMENT = "NeurIPS-2023-judging-llm-as-a-judge-with-mt-bench-and-chatbot-arena-Paper-Datasets_and_Benchmarks.md"
COLLECTION = "mutante"

# C2: os fatos da referência, trocados por outros plausíveis, onde o artigo os resume: o resumo (c0) e a
# introdução (c2). As seções que discutem cada viés em detalhe ficam como estão: o mutante é o documento
# desatualizado no que a pergunta pede, não um artigo inteiro reescrito. Cada padrão vale com e sem espaços,
# porque o Markdown deste artigo tem as palavras grudadas ("includingposition,verbosity,andself-enhancement").
C2_SUBSTITUTIONS: list[tuple[str, str]] = [
    (r"position,\s*verbosity,\s*and\s*self-enhancement", "recency, formatting, and self-consistency"),
    (r"position\s*bias,\s*verbosity\s*bias,\s*self-enhancement\s*bias",
     "recency bias, formatting bias, self-consistency bias"),
    (r"limited\s*reasoning\s*ability", "limited multilingual ability"),
    (r"80\s?%", "60%"),
]

OPERATORS: dict[str, dict[str, Any]] = {
    # Medido na fase 6: os outros 3 trechos do top-4 são do survey "Testing LLM-Based Systems", que repete os
    # fatos do MT-Bench (os vieses, os 75%, os 80% de Zheng et al.). Sem o artigo, o acervo ainda tem os fatos
    # por outra fonte: responder pelo survey não é falha. O que o C1 tira é a fonte primária.
    "C1": {"nome": "Remover documento", "defeito": "Fonte ausente: o artigo do MT-Bench sai do índice",
           "esperado": "abster-se, ou responder só com o que o survey repete (o acervo ainda tem os fatos por outra "
                       "fonte)", "camada": "corpus"},
    "C2": {"nome": "Documento desatualizado", "defeito": "O artigo indexado tem os fatos trocados (vieses, limitação e "
                                                        "a concordância de 80% para 60%)",
           "esperado": "a resposta muda, e a correção cai", "camada": "corpus"},
    "P1": {"nome": "Remover ancoragem no contexto", "defeito": "O prompt sem a regra de responder só com os trechos",
           "esperado": "a fidelidade cai", "camada": "prompt", "set": {"prompt_require_grounding": "false"}},
    "R1": {"nome": "top-k -> 1", "defeito": "Um trecho só chega ao gerador", "esperado": "uma resposta incompleta",
           "camada": "recuperação", "set": {"top_k": "1"}},
}


def substitute(text: str, substitutions: list[tuple[str, str]] = C2_SUBSTITUTIONS) -> tuple[str, dict[str, int]]:
    """Aplica as substituições do C2; devolve o texto e quantas vezes cada padrão trocou."""
    counts = {}
    for pattern, replacement in substitutions:
        text, n = re.subn(pattern, replacement, text, flags=re.IGNORECASE)
        counts[pattern] = n
    return text, counts


def copy_index(data: dict[str, list], dest: Path, keep: Callable[[dict], bool],
               replace: dict[int, tuple[str, list[float]]] | None = None) -> int:
    """Uma coleção nova em `dest` com os trechos de `data` que `keep` aceita; `replace` troca texto e vetor por posição.

    Mesmo espaço da coleção do backend (cosseno, app/rag/vector_store.py). Devolve quantos trechos entraram.
    """
    import chromadb

    replace = replace or {}
    dest.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(dest))
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001 - não existia
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    rows = [i for i, meta in enumerate(data["metadatas"]) if keep(meta)]
    for start in range(0, len(rows), 500):
        batch = rows[start:start + 500]
        collection.add(
            ids=[data["ids"][i] for i in batch],
            embeddings=[replace[i][1] if i in replace else data["embeddings"][i] for i in batch],
            documents=[replace[i][0] if i in replace else data["documents"][i] for i in batch],
            metadatas=[data["metadatas"][i] for i in batch],
        )
    return collection.count()


def index_overrides(dest: Path, embedding_model: str) -> dict[str, str]:
    """As sobrescritas que fazem o pipeline buscar no índice do mutante (e não no `md-v1`)."""
    return {"vector_store_dir": str(dest.resolve()), "vector_store_collection": COLLECTION,
            "embedding_model": embedding_model}


def prepare(op: str, out: Path) -> dict[str, Any]:
    from app.config import RAG_INDICES, get_settings, set_settings_overrides
    from app.rag import generation, ingestion, vector_store

    if op not in OPERATORS:
        raise ValueError(f"operador desconhecido: {op} (válidos: {', '.join(OPERATORS)})")
    spec = dict(OPERATORS[op])
    artifacts = out / "artefatos"
    artifacts.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result: dict[str, Any] = {"operador": op, **{k: v for k, v in spec.items() if k != "set"}, "indice_base": BASE_INDEX,
                              "documento": DOCUMENT, "artefatos": [], "set": {}, "usa_indice_base": True}

    def save(name: str, content: str) -> None:
        (artifacts / name).write_text(content, encoding="utf-8")
        result["artefatos"].append(f"artefatos/{name}")

    if spec["camada"] == "corpus":
        set_settings_overrides({"rag_index": BASE_INDEX})
        data = vector_store.get_all(include_embeddings=True)
        in_doc = [i for i, meta in enumerate(data["metadatas"]) if meta.get("document") == DOCUMENT]
        if not in_doc:
            raise ValueError(f"o artigo {DOCUMENT} não está no índice {BASE_INDEX}")
        embedding_model = RAG_INDICES[BASE_INDEX]["embedding_model"]
        dest = out / "indice"
        if op == "C1":
            count = copy_index(data, dest, keep=lambda meta: meta.get("document") != DOCUMENT)
            save("removido.txt", f"{DOCUMENT}\n{len(in_doc)} trechos fora do índice (de {len(data['ids'])})\n")
        else:
            changed, before_after, totals = {}, [], {p: 0 for p, _ in C2_SUBSTITUTIONS}
            for i in in_doc:
                text, counts = substitute(data["documents"][i])
                if text != data["documents"][i]:
                    changed[i] = text
                    before_after.append({"chunk_id": data["ids"][i], "antes": data["documents"][i], "depois": text})
                for pattern, n in counts.items():
                    totals[pattern] += n
            vectors = ingestion.embed_texts(list(changed.values()), model=embedding_model) if changed else []
            replace = {i: (text, vec) for (i, text), vec in zip(changed.items(), vectors)}
            count = copy_index(data, dest, keep=lambda meta: True, replace=replace)
            source = ROOT / RAG_INDICES[BASE_INDEX]["source_dir"] / DOCUMENT
            if source.is_file():
                mutated, file_counts = substitute(source.read_text(encoding="utf-8"))
                save(DOCUMENT, mutated)
                result["substituicoes_no_arquivo"] = file_counts
            save("substituicoes.json", json.dumps({"padroes": [{"de": p, "para": r} for p, r in C2_SUBSTITUTIONS],
                                                   "trocas_nos_trechos": totals}, ensure_ascii=False, indent=2))
            save("trechos_mutados.json", json.dumps(before_after, ensure_ascii=False, indent=2))
            result["trechos_mutados"] = len(changed)
            if not changed:
                raise ValueError("nenhuma substituição do C2 casou com os trechos do artigo: o mutante seria o original")
        result["indice"] = {"pasta": "indice", "colecao": COLLECTION, "trechos": count,
                            "trechos_do_artigo": len(in_doc), "embedding_model": embedding_model}
        result["set"] = index_overrides(dest, embedding_model)
        result["usa_indice_base"] = False
    else:
        result["set"] = dict(spec["set"])
        if op == "P1":
            set_settings_overrides({})
            before = generation.build_system_prompt()
            set_settings_overrides({"prompt_require_grounding": False})
            after = generation.build_system_prompt()
            save("prompt_original.txt", before)
            save("prompt_mutante.txt", after)
            save("prompt.diff", "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                               "prompt_original.txt", "prompt_mutante.txt")))
            if before == after:
                raise ValueError("o P1 não mudou o prompt: o mutante seria o original")
        else:
            set_settings_overrides({"rag_index": BASE_INDEX})
            save("configuracao.txt", f"top_k: {get_settings().top_k} -> 1 (índice {BASE_INDEX})\n")
    set_settings_overrides({})
    result["segundos"] = round(time.perf_counter() - t0, 2)
    (out / "mutante.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepara um mutante da pergunta única.")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("preparar", help="gera os artefatos e o índice do mutante")
    prep.add_argument("--op", required=True, choices=list(OPERATORS))
    prep.add_argument("--dir", type=Path, required=True, help="a pasta do mutante")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.op, args.dir)
    except ValueError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in result.items() if k != "set"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
