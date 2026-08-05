"""Навык как шаг плана: цепочка действий — один переход со своей статистикой.

Зачем это вообще. Планировщик считает надёжность цепочки произведением надёжностей
звеньев: два шага по 0.9 дают 0.81, три — 0.73. Для независимых шагов верно, но
цепочка, пройденная целиком двадцать раз и двадцать раз попавшая куда надо, — это одно
наблюдение с надёжностью 1.0, а не произведение догадок. И промежуточные места цепочки
могут быть склеены или неустойчивы, а цепочка как целое всё равно работает.

Что измерено (четыре сида, глубина поиска 6):

| Цепочки | Достижимо мест (медиана) | Нажатий в плане | Дошло | Рёбер в графе |
|---|---|---|---|---|
| нет | 12 | 6 | 96 % | 100 |
| до двух шагов | 17 | 10 | 96 % | 172 |
| до трёх шагов | 21 | 12 | 94 % | 324 |

То есть при той же глубине поиска планировщик дотягивается почти вдвое дальше, не
теряя доли дошедших. Цена — размер графа: рёбер втрое больше.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.behaviour.goals import Goal
from harness.behaviour.planner import Planner, execute
from harness.behaviour.skills import MacroRecorder, SkillError
from harness.core.action import (Action, ActionError, Reversibility, action_key,
                                 macro_key, output_id, parse_any_key, parse_macro_key)
from harness.core.profile import from_schema
from harness.model.beliefs import Origin, Provenance
from harness.model.forward import ForwardModel
from harness.model.places import PlaceGraph, Traversal
from harness.model.rebuild import BodyMap

A = action_key(output_id("OUT", 1), 200)
B = action_key(output_id("OUT", 2), 200)
C = action_key(output_id("OUT", 3), 200)


def _profile(**kw):
    base = dict(capture_width=64, capture_height=64)
    base.update(kw)
    return from_schema("ТЕСТ-макро", **base)


def _goal(target: str) -> Goal:
    return Goal(id="g", kind="reach_place", target=target, test=lambda: False,
                test_text="я в этом месте", budget_ticks=10,
                provenance=Provenance(Origin.EXPERIENCE, branch="t", seq=0),
                drive="curiosity", pressure=0.5)


# --- ключ макроса -----------------------------------------------------------


def test_macro_key_roundtrip() -> None:
    key = macro_key([A, B])
    assert parse_macro_key(key) == (A, B)
    assert parse_any_key(key) == (A, B)
    assert parse_any_key(A) == (A,)
    assert parse_any_key("не ключ") is None
    assert parse_macro_key(A) is None


def test_macro_of_one_step_is_refused() -> None:
    """Один шаг — это одиночное действие, и ключ у него уже есть.

    Два ключа на одно действие развели бы его статистику надвое, и оба выглядели бы
    менее подтверждёнными, чем есть.
    """
    with pytest.raises(ActionError, match="одиночное действие"):
        macro_key([A])
    with pytest.raises(ActionError, match="не ключ действия"):
        macro_key([A, "чепуха"])


# --- запись цепочек в граф --------------------------------------------------


def _frames(n: int, seed: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (64, 64), dtype=np.uint8) for _ in range(n)]


def test_recorder_writes_chains_of_two_and_three() -> None:
    g = PlaceGraph()
    rec = MacroRecorder(g, max_length=3, seconds_per_seq=0.1)
    views = _frames(4)
    seq = 0
    places = []
    for frame in views:
        seq += 1
        src = g.current
        dst = g.see(frame, seq, seconds_per_seq=0.1, mode=A)
        places.append(dst)
        if src is not None:
            rec.note(src, A, dst, seq)

    modes = {e.mode for e in g.edges.values()}
    assert any(parse_macro_key(m) is not None and len(parse_macro_key(m)) == 2
               for m in modes), modes
    assert any(parse_macro_key(m) is not None and len(parse_macro_key(m)) == 3
               for m in modes), modes
    assert rec.stats()["recorded"] > 0


def test_recorder_skips_chains_that_return_where_they_started() -> None:
    """Цепочка, вернувшая на место, — не переход, а его отсутствие.

    Ставить такую шагом плана бессмысленно: план из неё никуда не ведёт.
    """
    g = PlaceGraph()
    rec = MacroRecorder(g, max_length=3, seconds_per_seq=0.1)
    a, b = _frames(2, seed=5)
    seq = 0
    for frame in (a, b, a, b, a):
        seq += 1
        src = g.current
        dst = g.see(frame, seq, seconds_per_seq=0.1, mode=A)
        if src is not None:
            rec.note(src, A, dst, seq)
    for edge in g.edges.values():
        parsed = parse_macro_key(edge.mode)
        if parsed is not None:
            assert edge.src != edge.dst, (
                f"записана макро-петля {edge.src}→{edge.dst}: цепочка, вернувшая "
                "на место, не переход")


def test_recorder_refuses_length_below_two() -> None:
    with pytest.raises(SkillError, match="короче двух шагов"):
        MacroRecorder(PlaceGraph(), max_length=1)


def test_recorder_respects_max_length() -> None:
    g = PlaceGraph()
    rec = MacroRecorder(g, max_length=2, seconds_per_seq=0.1)
    seq = 0
    for frame in _frames(5, seed=7):
        seq += 1
        src = g.current
        dst = g.see(frame, seq, seconds_per_seq=0.1, mode=A)
        if src is not None:
            rec.note(src, A, dst, seq)
    lengths = {len(parse_macro_key(e.mode) or ()) for e in g.edges.values()}
    assert max(lengths) <= 2, lengths


# --- модель перехода и осторожность -----------------------------------------


def test_model_accepts_macro_transitions() -> None:
    model = ForwardModel()
    model.observe(Traversal("P1", "P2", macro_key([A, B]), mu_seconds=0.4, n=5))
    pred = model.predict("P1", macro_key([A, B]))
    assert pred is not None and pred.likely.dst == "P2" and pred.likely.n == 5


def test_macro_caution_is_the_worst_step_not_the_average() -> None:
    """Цепочка с одним неоткатываемым звеном — неоткатываемая цепочка.

    Усреднение спрятало бы ровно это: девять откатываемых нажатий и одно необратимое
    дали бы «в целом безопасно».
    """
    body = BodyMap()
    safe = body.fact(output_id("OUT", 1), 0)
    safe.reversibility = Reversibility(1.0, 0.0, 5)          # откатывается всегда
    risky = body.fact(output_id("OUT", 2), 0)
    risky.reversibility = Reversibility(0.0, 0.0, 5)         # не откатывается никогда
    model = ForwardModel(body=body)

    assert model.caution(A) < 0.5
    assert model.caution(B) > 0.9
    assert model.caution(macro_key([A, B])) == model.caution(B), (
        "осторожность цепочки не равна худшему звену")


# --- шаг плана --------------------------------------------------------------


def test_plan_step_from_a_macro_runs_every_action() -> None:
    """Шаг-навык исполняется целиком, а сверка одна — по итогу цепочки.

    Если бы `execute` жал только первое действие шага, план «сработал» бы, сделав
    половину. Это ровно та тихая поломка, которую проект запрещает.
    """
    model = ForwardModel()
    model.observe(Traversal("P1", "P2", macro_key([A, B, C]), mu_seconds=0.6, n=6))
    plan = Planner(_profile(), model).plan(_goal("P2"), "P1", "P2")
    assert plan is not None and plan.length == 1
    step = plan.steps[0]
    assert step.is_macro and step.key() == macro_key([A, B, C])
    assert len(step.actions()) == 3

    pressed: list[str] = []

    def act(action: Action) -> str:
        pressed.append(action_key(action.output or "", action.duration_ms))
        return "P2" if len(pressed) == 3 else "середина"

    ex = execute(plan, act=act)
    assert pressed == [A, B, C], pressed
    assert ex.goal_passed and ex.surprises == 0, ex.as_dict()


def test_plan_step_reports_how_many_presses_it_costs() -> None:
    model = ForwardModel()
    model.observe(Traversal("P1", "P2", macro_key([A, B]), mu_seconds=0.4, n=4))
    plan = Planner(_profile(), model).plan(_goal("P2"), "P1", "P2")
    assert plan is not None
    d = plan.steps[0].as_dict()
    assert d["is_macro"] and d["chain"] == [A, B]


def test_single_step_plans_are_unchanged() -> None:
    """Одиночный шаг остался одиночным: `chain` пуст, действие одно."""
    model = ForwardModel()
    model.observe(Traversal("P1", "P2", A, mu_seconds=0.2, n=4))
    plan = Planner(_profile(), model).plan(_goal("P2"), "P1", "P2")
    assert plan is not None
    step = plan.steps[0]
    assert not step.is_macro and step.chain == () and step.key() == A
    assert len(step.actions()) == 1


def test_macro_reaches_farther_at_the_same_search_depth() -> None:
    """То, ради чего всё это: при том же пределе длины плана дотягиваемся дальше.

    Цепочка из трёх нажатий — один шаг, поэтому в плане длиной два шага получается
    шесть нажатий. На интерактивном мире это измерено: достижимо 12 мест против 21.
    """
    chain = macro_key([A, B, C])
    with_macro = ForwardModel()
    without = ForwardModel()
    # Прямая дорога P1 → P2 → P3 одиночными шагами, плюс та же дорога цепочкой
    for model in (with_macro, without):
        model.observe(Traversal("P1", "P2", A, mu_seconds=0.2, n=4))
        model.observe(Traversal("P2", "P3", B, mu_seconds=0.2, n=4))
        model.observe(Traversal("P3", "P4", C, mu_seconds=0.2, n=4))
    with_macro.observe(Traversal("P1", "P4", chain, mu_seconds=0.6, n=4))

    goal = _goal("P4")
    limit = _profile(plan_max_length=2)
    assert Planner(limit, without).plan(goal, "P1", "P4") is None, (
        "без цепочки план в два шага дотянулся до трёх переходов — замер неверен")
    plan = Planner(limit, with_macro).plan(goal, "P1", "P4")
    assert plan is not None and plan.length == 1 and plan.steps[0].is_macro
