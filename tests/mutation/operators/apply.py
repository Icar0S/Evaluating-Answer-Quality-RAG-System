"""Aplicação e reversão dos operadores de mutação. Idempotente por construção.

O ponto de decisão GO/NO-GO da semana 4 é exatamente uma propriedade deste
módulo: *apply.py aplica e reverte os 18 operadores sem resíduo*. Por isso o
desenho evita o padrão frágil (guardar um backup e desfazer a operação inversa) e
usa o padrão robusto:

    revert  = reconstruir work/ do zero a partir de corpus/base/
    apply X = reconstruir work/ do zero e então aplicar as operações de X

Assim aplicar duas vezes tem o mesmo efeito que aplicar uma, aplicar B depois de
A não deixa resto de A, e uma campanha interrompida no meio não deixa o corpus
num estado intermediário: a próxima chamada reconstrói tudo.

Mutações de configuração não tocam disco nenhum — voltam como um dicionário de
overrides que o runner aplica em memória (runner/sut.py).

Uso:
    python -m tests.mutation.operators.apply list
    python -m tests.mutation.operators.apply apply R1
    python -m tests.mutation.operators.apply revert
    python -m tests.mutation.operators.apply verify      # o portão da semana 4
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

if __package__ in (None, ""):  # permite rodar como script solto
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.mutation.runner import paths
from tests.mutation.runner.console import get_logger

logger = get_logger("mutation.apply")


class OperatorError(RuntimeError):
    """Erro de definição ou de aplicação de operador — sempre acionável."""


@dataclass
class Operator:
    id: str
    layer: str
    name: str
    defect: str
    failure_point: str | None
    cuttable: bool
    action: dict[str, Any]
    rationale: str | None = None
    replicates_incident: bool = False

    @property
    def kind(self) -> str:
        return self.action.get("kind", "config")


@dataclass
class Catalog:
    version: str
    layers: dict[str, str]
    operators: list[Operator]
    by_id: dict[str, Operator] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.by_id = {op.id: op for op in self.operators}

    def get(self, operator_id: str) -> Operator:
        try:
            return self.by_id[operator_id]
        except KeyError as exc:
            known = ", ".join(sorted(self.by_id))
            raise OperatorError(f"Operador '{operator_id}' não existe. Conhecidos: {known}") from exc

    def active(self, include_cuttable: bool = True) -> list[Operator]:
        """Operadores da campanha. include_cuttable=False aplica a regra de corte da semana 6."""
        if include_cuttable:
            return list(self.operators)
        return [op for op in self.operators if not op.cuttable]


def load_catalog(path: Path | None = None) -> Catalog:
    catalog_path = path or paths.OPERATOR_CATALOG
    raw = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    operators = [
        Operator(
            id=item["id"],
            layer=item["layer"],
            name=item["name"],
            defect=item["defect"],
            failure_point=item.get("failure_point"),
            cuttable=bool(item.get("cuttable", False)),
            action=item["action"],
            rationale=item.get("rationale"),
            replicates_incident=bool(item.get("replicates_incident", False)),
        )
        for item in raw["operators"]
    ]
    catalog = Catalog(version=str(raw.get("version", "0")), layers=raw["layers"], operators=operators)
    _validate_catalog(catalog)
    return catalog


def _validate_catalog(catalog: Catalog) -> None:
    seen: set[str] = set()
    for op in catalog.operators:
        if op.id in seen:
            raise OperatorError(f"Operador duplicado no catálogo: {op.id}")
        seen.add(op.id)
        if op.layer not in catalog.layers:
            raise OperatorError(f"{op.id}: camada '{op.layer}' não declarada em `layers`")
        if op.kind not in ("config", "corpus"):
            raise OperatorError(f"{op.id}: kind '{op.kind}' desconhecido")
        if op.kind == "config" and not (op.action.get("overrides") or op.action.get("overrides_relative")):
            raise OperatorError(f"{op.id}: operador de configuração sem `overrides`")
        if op.kind == "corpus" and not op.action.get("ops"):
            raise OperatorError(f"{op.id}: operador de corpus sem `ops`")


# --------------------------------------------------------------------- manifesto


def load_manifest() -> dict[str, Any]:
    if not paths.CORPUS_MANIFEST.exists():
        raise OperatorError(
            f"{paths.CORPUS_MANIFEST} não existe. Rode primeiro:\n"
            "  python -m tests.mutation.corpus.build_corpus --from data/source_pdfs"
        )
    return json.loads(paths.CORPUS_MANIFEST.read_text(encoding="utf-8"))


def resolve_target(target: str, manifest: dict[str, Any]) -> list[str]:
    """Resolve `role:x` ou um nome de arquivo literal para uma lista de nomes de PDF."""
    if not target.startswith("role:"):
        return [target]
    role = target.split(":", 1)[1]
    roles = manifest.get("roles", {})
    if role not in roles:
        raise OperatorError(
            f"Papel '{role}' não está definido em corpus/manifest.json. "
            f"Papéis disponíveis: {sorted(roles)}"
        )
    value = roles[role]
    return list(value) if isinstance(value, list) else [value]


# ------------------------------------------------------------------ operações


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(src.glob("*.pdf")):
        shutil.copy2(pdf, dst / pdf.name)


def rebuild_work() -> None:
    """Reconstrói work/corpus e work/index_src a partir de corpus/base. É o revert."""
    if not paths.CORPUS_BASE.is_dir() or not any(paths.CORPUS_BASE.glob("*.pdf")):
        raise OperatorError(
            f"Corpus base vazio em {paths.CORPUS_BASE}. Rode corpus/build_corpus.py antes."
        )
    _copy_tree(paths.CORPUS_BASE, paths.WORK_CORPUS)
    _copy_tree(paths.CORPUS_BASE, paths.WORK_INDEX_SRC)


def _variant_path(document: str, variant: str) -> Path:
    stem = Path(document).stem
    candidate = paths.CORPUS_VARIANTS / f"{stem}__{variant}.pdf"
    if not candidate.exists():
        raise OperatorError(
            f"Variante '{variant}' de '{document}' não existe em {paths.CORPUS_VARIANTS}. "
            "Gere com: python -m tests.mutation.corpus.build_corpus --make-variants"
        )
    return candidate


def _apply_file_op(op: dict[str, Any], targets: list[str], directories: list[Path]) -> list[str]:
    """Aplica uma operação de arquivo em todos os diretórios pedidos. Devolve o log do que fez."""
    kind = op["op"]
    log: list[str] = []

    for document in targets:
        for directory in directories:
            target_path = directory / document

            if kind == "remove_document":
                if target_path.exists():
                    target_path.unlink()
                    log.append(f"removido {document} de {directory.name}/")

            elif kind == "use_variant":
                source = _variant_path(document, op["variant"])
                shutil.copy2(source, target_path)
                log.append(f"{document} substituído pela variante '{op['variant']}' em {directory.name}/")

            elif kind == "truncate_pages":
                keep = float(op.get("keep_fraction", 0.7))
                pages_before, pages_after = _truncate_pdf(target_path, keep)
                log.append(f"{document} truncado {pages_before}->{pages_after} páginas em {directory.name}/")

            elif kind == "add_variant_as_document":
                source = _variant_path(document, op["variant"])
                new_name = f"{Path(document).stem}{op.get('suffix', '__dup')}.pdf"
                shutil.copy2(source, directory / new_name)
                log.append(f"adicionado {new_name} em {directory.name}/")

            else:
                raise OperatorError(f"Operação de arquivo desconhecida: {kind}")

    return log


def _truncate_pdf(pdf_path: Path, keep_fraction: float) -> tuple[int, int]:
    import fitz  # PyMuPDF; import local porque só os operadores de corpus precisam

    if not pdf_path.exists():
        raise OperatorError(f"{pdf_path} não existe para truncar")
    temp_path = pdf_path.with_suffix(".pdf.tmp")
    with fitz.open(pdf_path) as doc:
        total = doc.page_count
        keep = max(1, int(round(total * keep_fraction)))
        if keep < total:
            doc.delete_pages(from_page=keep, to_page=total - 1)
        # Salva em arquivo temporário e troca: o PyMuPDF não regrava com segurança
        # o arquivo que ainda está aberto.
        doc.save(str(temp_path))
    temp_path.replace(pdf_path)
    return total, keep


def _resolve_override_value(value: Any, study_config: dict[str, Any] | None) -> Any:
    """Resolve referências '@models.x' contra o study.yaml."""
    if isinstance(value, str) and value.startswith("@"):
        if study_config is None:
            raise OperatorError(f"Referência '{value}' exige study.yaml carregado")
        node: Any = study_config
        for part in value[1:].split("."):
            if not isinstance(node, dict) or part not in node:
                raise OperatorError(f"Referência '{value}' não resolve em study.yaml")
            node = node[part]
        return node
    return value


def config_overrides(
    operator: Operator,
    baseline: dict[str, Any],
    study_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Overrides de configuração do operador, já resolvidos contra o baseline."""
    if operator.kind != "config":
        return {}

    overrides: dict[str, Any] = {}
    for key, value in (operator.action.get("overrides") or {}).items():
        overrides[key] = _resolve_override_value(value, study_config)

    for key, spec in (operator.action.get("overrides_relative") or {}).items():
        if key not in baseline:
            raise OperatorError(f"{operator.id}: override relativo sobre '{key}', ausente do baseline")
        current = baseline[key]
        if "multiply" in spec:
            overrides[key] = type(current)(current * spec["multiply"])
        elif "add" in spec:
            overrides[key] = type(current)(current + spec["add"])
        else:
            raise OperatorError(f"{operator.id}: override relativo '{spec}' não suportado")

    unchanged = [k for k, v in overrides.items() if k in baseline and baseline[k] == v]
    if unchanged:
        raise OperatorError(
            f"{operator.id}: override(s) {unchanged} são iguais ao baseline — o mutante seria "
            "equivalente por construção. Ajuste o baseline em config/study.yaml ou o catálogo."
        )
    return overrides


@dataclass
class AppliedMutant:
    operator_id: str
    layer: str
    kind: str
    config_overrides: dict[str, Any]
    file_log: list[str]

    @property
    def requires_reindex(self) -> bool:
        from tests.mutation.runner.sut import INDEX_AFFECTING_KEYS

        if self.kind == "corpus":
            return True
        return any(key in self.config_overrides for key in INDEX_AFFECTING_KEYS)


def apply(
    operator_id: str,
    catalog: Catalog | None = None,
    baseline: dict[str, Any] | None = None,
    study_config: dict[str, Any] | None = None,
) -> AppliedMutant:
    """Deixa work/ no estado do mutante e devolve os overrides de configuração.

    Sempre reconstrói work/ primeiro: é o que torna a chamada idempotente e
    independente do que estava aplicado antes.
    """
    catalog = catalog or load_catalog()
    operator = catalog.get(operator_id)
    rebuild_work()

    file_log: list[str] = []
    if operator.kind == "corpus":
        manifest = load_manifest()
        scope = operator.action.get("scope", "both")
        directories = [paths.WORK_INDEX_SRC] if scope == "index_only" else [paths.WORK_CORPUS, paths.WORK_INDEX_SRC]
        for file_op in operator.action["ops"]:
            targets = resolve_target(file_op["target"], manifest)
            file_log.extend(_apply_file_op(file_op, targets, directories))

    overrides = config_overrides(operator, baseline or {}, study_config)
    return AppliedMutant(
        operator_id=operator.id,
        layer=operator.layer,
        kind=operator.kind,
        config_overrides=overrides,
        file_log=file_log,
    )


def revert() -> None:
    """Volta work/ ao corpus base. Mutações de configuração morrem com o processo."""
    rebuild_work()


# ------------------------------------------------------------------------ CLI


def _directory_fingerprint(directory: Path) -> list[tuple[str, int, str]]:
    import hashlib

    result = []
    for pdf in sorted(directory.glob("*.pdf")):
        data = pdf.read_bytes()
        result.append((pdf.name, len(data), hashlib.sha256(data).hexdigest()[:12]))
    return result


def verify(catalog: Catalog | None = None) -> int:
    """Portão da semana 4: aplica e reverte os 18 operadores e confere que não sobrou resíduo."""
    catalog = catalog or load_catalog()
    study_raw = yaml.safe_load(paths.STUDY_CONFIG.read_text(encoding="utf-8"))
    baseline = study_raw["baseline_config"]

    revert()
    reference = _directory_fingerprint(paths.WORK_CORPUS)
    reference_index = _directory_fingerprint(paths.WORK_INDEX_SRC)

    failures: list[str] = []
    for operator in catalog.operators:
        try:
            mutant = apply(operator.id, catalog, baseline, study_raw)
        except OperatorError as exc:
            failures.append(f"{operator.id}: apply falhou — {exc}")
            continue

        if operator.kind == "corpus":
            changed = (
                _directory_fingerprint(paths.WORK_CORPUS) != reference
                or _directory_fingerprint(paths.WORK_INDEX_SRC) != reference_index
            )
            if not changed:
                failures.append(f"{operator.id}: operador de corpus não alterou nada em work/")
        elif not mutant.config_overrides:
            failures.append(f"{operator.id}: operador de configuração não produziu overrides")

        revert()
        if _directory_fingerprint(paths.WORK_CORPUS) != reference:
            failures.append(f"{operator.id}: resíduo em work/corpus depois do revert")
        if _directory_fingerprint(paths.WORK_INDEX_SRC) != reference_index:
            failures.append(f"{operator.id}: resíduo em work/index_src depois do revert")

    if failures:
        logger.error("VERIFY FALHOU (%d problema(s)):", len(failures))
        for failure in failures:
            logger.error("  - %s", failure)
        return 1

    logger.info("VERIFY OK: %d operadores aplicam e revertem sem resíduo.", len(catalog.operators))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aplica/reverte operadores de mutação do estudo RAG.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="lista o catálogo")
    apply_parser = sub.add_parser("apply", help="aplica um operador em work/")
    apply_parser.add_argument("operator_id")
    sub.add_parser("revert", help="volta work/ ao corpus base")
    sub.add_parser("verify", help="aplica e reverte todos, checando ausência de resíduo")

    args = parser.parse_args(argv)
    catalog = load_catalog()

    if args.command == "list":
        print(f"{'ID':<4} {'camada':<11} {'FP':<5} {'corte':<6} nome")
        for operator in catalog.operators:
            print(
                f"{operator.id:<4} {operator.layer:<11} {str(operator.failure_point or '-'):<5} "
                f"{'sim' if operator.cuttable else 'nao':<6} {operator.name}"
            )
        return 0

    if args.command == "apply":
        study_raw = yaml.safe_load(paths.STUDY_CONFIG.read_text(encoding="utf-8"))
        mutant = apply(args.operator_id, catalog, study_raw["baseline_config"], study_raw)
        print(json.dumps(mutant.__dict__ | {"requires_reindex": mutant.requires_reindex}, indent=2, ensure_ascii=False))
        return 0

    if args.command == "revert":
        revert()
        print("work/ reconstruído a partir de corpus/base/")
        return 0

    return verify(catalog)


if __name__ == "__main__":
    raise SystemExit(main())
