"""Счётчики срабатываний на ветках арбитража. Инвариант 26.

Проверяется то, из-за чего инвариант появился: мёртвая ветка и ветка,
заблокированная ненасыщаемым условием сверху, — разные болезни, и отчёт обязан их
различать.
"""
from __future__ import annotations

import pytest

from harness.core.branches import (STARVATION_BAR, Arbitration, BranchError,
                                   Ledger)

SPEC = (("первая", "срабатывает всегда"),
        ("вторая", "почти никогда"),
        ("третья", "никогда: её случай не встречается"))


def test_branch_outside_the_declared_list_is_an_error() -> None:
    """Ветка вне перечня не попала бы в отчёт о нулевых — а значит соврала бы."""
    arb = Arbitration.of("тест", SPEC)
    with pytest.raises(BranchError, match="мёртвый код снова окажется невидимым"):
        arb.hit("четвёртая")


def test_zero_hits_is_reported_as_a_defect() -> None:
    arb = Arbitration.of("тест", SPEC)
    for _ in range(10):
        arb.hit("первая")
    for _ in range(10):
        arb.hit("вторая")
    assert [b.name for b in arb.dead] == ["третья"]
    assert arb.has_defects
    assert "ни разу не сработали" in arb.report()
    assert "её случай не встречается" in arb.report(), "причина обязана печататься"


def test_starved_and_merely_dead_are_different_diagnoses() -> None:
    """То, из-за чего инвариант появился: 100 % наверху делает низ недостижимым."""
    arb = Arbitration.of("тест", SPEC)
    for _ in range(1500):
        arb.hit("первая")
    assert arb.dominant is not None and arb.dominant.name == "первая"
    # Обе нижние мертвы, но диагноз у них другой: их не пропускают сверху.
    assert {b.name for b in arb.starved()} == {"вторая", "третья"}
    assert "заблокированы сверху" in arb.report()
    assert f"{STARVATION_BAR:.0%}" in arb.report()


def test_no_dominant_means_dead_is_about_the_world() -> None:
    arb = Arbitration.of("тест", SPEC)
    for _ in range(50):
        arb.hit("первая")
    for _ in range(50):
        arb.hit("вторая")
    assert arb.dominant is None
    assert arb.starved() == [], "без доминирующей заблокированных нет"
    assert [b.name for b in arb.dead] == ["третья"]


def test_arbitration_never_called_is_also_a_defect() -> None:
    arb = Arbitration.of("тест", SPEC)
    assert "ни одного решения за прогон" in arb.report()


def test_ledger_collects_defects_across_arbitrations() -> None:
    ledger = Ledger()
    a = ledger.add(Arbitration.of("первый", SPEC))
    b = ledger.add(Arbitration.of("второй", (("одна", "всегда"), ("две", "нет"))))
    for _ in range(100):
        a.hit("первая")
    for _ in range(10):
        b.hit("одна")
    for _ in range(10):
        b.hit("две")
    found = ledger.defects()
    assert any("первый/вторая: заблокирована сверху" in x for x in found)
    assert not any("второй/" in x for x in found), "у второго дефектов нет"
    assert "дефектов арбитража: " in ledger.report()


def test_clean_run_says_so_without_hedging() -> None:
    ledger = Ledger()
    arb = ledger.add(Arbitration.of("тест", (("одна", "a"), ("две", "b"))))
    for _ in range(10):
        arb.hit("одна")
        arb.hit("две")
    assert not arb.has_defects
    assert "дефектов арбитража нет" in ledger.report()


def test_closing_explorer_declares_all_its_branches() -> None:
    """Каждая ветка разведки с замыканием объявлена и отмечается."""
    import ast
    import inspect

    from harness.behaviour.planner import (CLOSING_BRANCHES, choose_closing_probe,
                                           closing_arbitration)

    declared = {n for n, _ in CLOSING_BRANCHES}
    assert declared == {b.name for b in closing_arbitration()}

    # Все имена, отмечаемые в коде через hit(...), объявлены — и наоборот.
    tree = ast.parse(inspect.getsource(choose_closing_probe))
    marked = {node.args[0].value for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "hit" and node.args
              and isinstance(node.args[0], ast.Constant)}
    assert marked == declared, (
        f"отмечаются {sorted(marked - declared)}, объявлены "
        f"{sorted(declared - marked)}")


def test_counters_do_not_change_the_choice() -> None:
    """Счётчик, влияющий на выбор, измерял бы себя, а не выбор."""
    from harness.behaviour.planner import choose_closing_probe, closing_arbitration
    from harness.model.forward import ForwardModel, Outcome
    from harness.model.places import PlaceGraph
    from harness.core.profile import from_schema

    profile = from_schema("ТЕСТ-счётчики", capture_width=64, capture_height=64)
    graph = PlaceGraph.from_profile(profile)
    model = ForwardModel()
    model.transitions[("A", "OUT_0A11@200")] = [Outcome("B", 9, 0.2, 0.01)]
    outputs = ["OUT_0A11", "OUT_0B22", "OUT_0C33"]

    without = choose_closing_probe(model, graph, "A", outputs, hold_ms=200)
    arb = closing_arbitration()
    with_counter = choose_closing_probe(model, graph, "A", outputs, hold_ms=200,
                                        arb=arb)
    assert without == with_counter
    assert arb.decisions == 1, "решение отмечено ровно один раз"
