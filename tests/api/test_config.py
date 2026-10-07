"""Configuração: o .env.example precisa funcionar como veio.

O passo 2 do README é `copy .env.example .env`. Se o template não validar, o
backend morre no boot antes de qualquer outra mensagem útil.
"""
from __future__ import annotations

from app.config import PROJECT_ROOT, Settings


def test_env_example_carrega_sem_erro():
    settings = Settings(_env_file=str(PROJECT_ROOT / ".env.example"))

    # Campos vazios no template significam "não definido".
    assert settings.generation_seed is None
    assert settings.generation_temperature is None
    assert settings.generation_think is None


def test_campo_vazio_no_ambiente_vira_none(monkeypatch):
    monkeypatch.setenv("GENERATION_SEED", "")
    monkeypatch.setenv("GENERATION_THINK", "  ")

    settings = Settings(_env_file=None)

    assert settings.generation_seed is None
    assert settings.generation_think is None
