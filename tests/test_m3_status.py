"""Список М3 в STATUS.md: три градации и число проверенных на живом.

`TASK-06`, часть 4. Смысл раздела — состояние вехи читается без чтения истории
коммитов. Смысл этого теста — раздел не расходится с реальностью молча.

Проверяется не текст, а данные: `tools/project_status.py` объявляет пункты М3, а
STATUS.md из них генерируется. Руками файл править бессмысленно — он перезапишется.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import project_status as ps  # noqa: E402


def test_every_m3_item_is_graded_and_explained() -> None:
    grades = dict(ps.GRADES)
    for item in ps.M3_ITEMS:
        assert item["grade"] in grades, f"{item['item']}: градация не из набора"
        assert item["how"], f"{item['item']}: не сказано, чем проверено"
        assert item["basis"], f"{item['item']}: не сказано, на чём"
        if item["grade"] == "measured":
            assert item["basis"] in ("синтетика", "живое"), (
                f"{item['item']}: у проверенного замером основание обязано быть "
                "синтетикой или живым — без этого «проверено» ничего не значит")


def test_the_live_count_is_computed_not_written() -> None:
    """Число проверенных на живом выводится из пунктов, а не пишется рядом.

    Написанное рядом расходится с таблицей при первой же правке — и расходится
    молча, потому что читают именно его.
    """
    s = ps.m3_summary()
    expected = sum(1 for i in ps.M3_ITEMS
                   if i["grade"] == "measured" and i["basis"] == "живое")
    assert s["measured_on_live"] == expected
    assert s["total"] == len(ps.M3_ITEMS)
    assert sum(s["by_grade"].values()) == s["total"], (
        "пункты потерялись между градациями")


def test_nothing_is_verified_on_a_live_screen_yet() -> None:
    """Пока это ноль — тест фиксирует именно ноль.

    Когда появится первая живая запись, тест упадёт, и это правильно: он заставит
    признать событие явно, а не оставить старую цифру в отчёте.
    """
    assert ps.m3_summary()["measured_on_live"] == 0, (
        "появилась живая запись — обновите пункты М3 и снимите этот тест "
        "(TASK-06, часть 4)")


def test_status_md_shows_the_live_count_prominently() -> None:
    """Число обязано быть в заголовке, а не выводиться читателем из таблицы."""
    text = (ROOT / "docs" / "STATUS.md").read_text(encoding="utf-8")
    assert "## М3: состояние по пунктам" in text
    head = re.search(r"### На живом экране проверено: \*\*(\d+) из (\d+)\*\*", text)
    assert head, "заголовка с числом проверенных на живом в STATUS.md нет"
    s = ps.m3_summary()
    assert int(head.group(1)) == s["measured_on_live"]
    assert int(head.group(2)) == s["total"]
    for _, title in ps.GRADES:
        assert title in text, f"градация «{title}» в отчёт не попала"


def test_acceptance_criteria_of_m3_are_present_as_items() -> None:
    """Критерии приёмки вехи из ROADMAP обязаны быть пунктами, а не подразумеваться.

    Их два: возврат в известное место не хуже шести из десяти и сужение sigma с
    опытом. Второй не измерен — и именно поэтому он должен стоять в списке: пункт,
    которого нет в таблице, невидим, а не выполнен.
    """
    joined = " ".join(i["item"] for i in ps.M3_ITEMS)
    assert "6 из 10" in joined
    assert "sigma" in joined and "сужается" in joined
