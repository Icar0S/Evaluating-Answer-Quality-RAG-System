"""C2/C4 no tests/generate_answers.py: o que cada resposta grava, o braço, a ordem e a recusa."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from app.config import set_settings_overrides
from tests.benchmark import abstention, arm

ROOT = Path(__file__).resolve().parents[2]
MARKER = "Não encontrei essa informação nos documentos fornecidos"
GOLDEN = {"name": "g1", "input": "Quantas classes?", "expected_output": "16.", "context": ["16 classes"],
          "source_document": "artigo"}
CHUNKS = [
    {"chunk_id": "artigo::p4::c0", "document": "artigo.pdf", "page": 4, "text": "16 classes", "similarity_score": 0.7},
    {"chunk_id": "outro::c3", "document": "outro.md", "page": None, "text": "nada", "similarity_score": 0.5},
]


@pytest.fixture
def generate(monkeypatch):
    spec = importlib.util.spec_from_file_location("generate_answers", ROOT / "tests" / "generate_answers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reply = {"answer": "São 16 classes.", "latency_ms": 1234.56, "prompt_tokens": 900, "completion_tokens": 40}
    monkeypatch.setattr(module.retrieval, "retrieve", lambda question, top_k=None: (CHUNKS, 87.65))
    monkeypatch.setattr(module.generation, "generate_answer", lambda question, chunks: dict(reply))
    module.reply = reply
    return module


def test_recusa_sem_acento_caixa_ou_pontuacao_ainda_e_recusa():
    assert abstention.is_abstention("Nao encontrei essa informacao nos documentos fornecidos.", MARKER)
    assert abstention.is_abstention("NÃO ENCONTREI ESSA INFORMAÇÃO NOS DOCUMENTOS FORNECIDOS!", MARKER)
    assert abstention.is_abstention("Desculpe: não encontrei essa informação\nnos documentos fornecidos.", MARKER)
    assert not abstention.is_abstention("Não encontrei o arquivo, mas são 16 classes.", MARKER)
    assert not abstention.is_abstention(None, MARKER)
    assert abstention.is_abstention("O contexto não traz esse dado.", MARKER, variants=("o contexto nao traz",))


def test_resposta_grava_custo_documentos_e_recusa(generate):
    record = generate.answer_record("v2", GOLDEN, GOLDEN["input"], 4)

    assert record["source_documents"] == ["artigo.pdf", "outro.md"]
    assert record["sources"] == ["artigo.pdf (pág. 4)", "outro.md"]
    assert record["chunk_ids"] == ["artigo::p4::c0", "outro::c3"]
    assert (record["retrieval_ms"], record["generation_ms"]) == (87.7, 1234.6)
    assert (record["prompt_tokens"], record["completion_tokens"]) == (900, 40)
    # Sem GENERATION_NUM_CTX não há limite conhecido, logo não há como dizer que truncou.
    assert (record["truncated"], record["abstained"], record["error"]) == (False, False, None)


def test_prompt_no_limite_do_ollama_e_truncagem(generate):
    set_settings_overrides({"generation_num_ctx": "4096", "generation_num_predict": "512"})
    generate.reply["prompt_tokens"] = 3583  # limite = 4096 - 512 = 3584

    assert generate.answer_record("v2", GOLDEN, GOLDEN["input"], 4)["truncated"] is True
    generate.reply["prompt_tokens"] = 3000
    assert generate.answer_record("v2", GOLDEN, GOLDEN["input"], 4)["truncated"] is False


def test_falha_vira_registro_depois_de_duas_tentativas(generate, monkeypatch):
    calls = []

    def broken(question, top_k=None):
        calls.append(question)
        raise TimeoutError("Ollama demorou")

    monkeypatch.setattr(generate.retrieval, "retrieve", broken)

    record = generate.answer_record("v2", GOLDEN, GOLDEN["input"], 4)

    assert len(calls) == 2
    assert record["actual_output"] is None and record["error"].startswith("TimeoutError")


def test_ordem_embaralhada_e_reproduzivel_e_o_braco_registra_as_sobrescritas(generate):
    assert generate.processing_order(5, None) == [0, 1, 2, 3, 4]
    assert generate.processing_order(5, 3) == generate.processing_order(5, 3) != [0, 1, 2, 3, 4]
    assert sorted(generate.processing_order(5, 3)) == [0, 1, 2, 3, 4]

    overrides = arm.parse_overrides(["generation_temperature=0", "GENERATION_SEED=42", "generation_think=false"])
    set_settings_overrides(overrides)
    config = arm.provenance(overrides)

    assert config["generation"]["temperature"] == 0.0
    assert (config["generation"]["seed"], config["generation"]["think"]) == (42, False)
    assert config["generation"]["provider"] == "local"
    assert config["overrides"] == {"generation_temperature": "0", "generation_seed": "42", "generation_think": "false"}
    with pytest.raises(SystemExit):
        arm.parse_overrides(["sem_igual"])
