"""Fixtures da camada JevEval.

Dois tipos de teste moram aqui, com regras opostas:

- herméticos (sem marcador): rodam com o Jev falso de _fake_jev.py, sem rede e
  sem chave, em todo PR no CI (.github/workflows/ci.yml, job `jeveval`). Validam
  as perguntas, os pesos, a agregação, o N/A e o formato do export;
- ao vivo (`@pytest.mark.live`): chamam o Jev de verdade na API da TypeSafe.
  Sem TYPESAFE_API_KEY eles se auto-pulam, como as suítes DeepEval e RAGAS se
  auto-pulam sem o Ollama. Nunca rodam no CI (o job usa `-m "not live"`).
"""
from __future__ import annotations

import pytest

from _jev_shared import JevSettings, build_jev, get_settings, jev_status


def pytest_configure(config: pytest.Config) -> None:
    # O pytest.ini da raiz usa --strict-markers: marcador não registrado é erro.
    config.addinivalue_line("markers", "live: chama o Jev de verdade (API da TypeSafe); pula sem TYPESAFE_API_KEY")


@pytest.fixture(scope="session")
def settings() -> JevSettings:
    return get_settings()


@pytest.fixture(scope="session")
def jev(settings: JevSettings):
    """O Jev de verdade. Só os testes ao vivo pedem esta fixture."""
    reason = jev_status(settings)
    if reason:
        pytest.skip(reason)
    return build_jev(settings)
