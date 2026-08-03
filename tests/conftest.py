"""Общая обвязка тестов.

Пакет лежит в `src/`, поэтому путь добавляется здесь: тесты должны запускаться
через `pytest` без установки пакета, иначе «проверить на записанной сессии»
превращается в ритуал.

Синтетический корпус генерируется один раз на весь прогон: он детерминирован по
сиду, а генерация — единственная заметно небыстрая операция в этих тестах.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Синтетическая сессия с известной истиной. Только для чтения."""
    from harness.corpus.synthetic import generate_session

    root = tmp_path_factory.mktemp("corpus") / "synthetic-0"
    generate_session(root, seed=7)
    return root


@pytest.fixture()
def session(corpus: Path):
    from harness.session import Session

    s = Session.open(corpus)
    yield s
    s.close()


@pytest.fixture()
def profile():
    from harness.core.profile import MILESTONE_0

    return MILESTONE_0
