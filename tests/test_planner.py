"""Тесты модели перехода и планировщика.

Планировщик — то место, где легче всего незаметно соврать: достаточно придумать
стоимость неизвестного перехода или молча взять самый вероятный исход за
единственный, и план будет выглядеть разумным, оставаясь обещанием, которое нечем
сдержать. Поэтому проверяется прежде всего то, чего планировщик делать **не** должен.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.behaviour.babbling import Babbler, run_babbling
from harness.behaviour.goals import Goal
from harness.behaviour.planner import (Planner, choose_probe, execute, rehearse,
                                       travel)
from harness.core.action import Action, Reversibility, action_key, output_id
from harness.core.clocks import Clocks, Stamp
from harness.core.journal import Kind
from harness.core.profile import from_schema
from harness.corpus.world import InteractiveWorld
from harness.model.beliefs import Origin, Provenance
from harness.model.forward import ForwardModel, Outcome, build
from harness.core.action import parse_action_key
from harness.model.places import PlaceGraph, Traversal
from harness.model.rebuild import BodyMap


# Непрозрачные идентификаторы выходов: `OUT_xx` из скан-кода, как в настоящем теле.
# Придумать «OUT_1» нельзя — действие такой выход не примет, и это правильно.
OUT_1 = output_id("OUT", 2)
OUT_2 = output_id("OUT", 3)
OUT_3 = output_id("OUT", 4)
OUT_SAFE = output_id("OUT", 5)
OUT_RISK = output_id("OUT", 6)


def _profile(**kw: object):
    base: dict[str, object] = dict(capture_width=320, capture_height=180)
    base.update(kw)
    return from_schema("ТЕСТ-план", **base)


def _goal(target: str, test=lambda: False, budget: int = 40) -> Goal:
    return Goal(id="g1", kind="reach_place", target=target, test=test,
                test_text="я в этом месте", budget_ticks=budget,
                provenance=Provenance(Origin.EXPERIENCE, branch="b0", seq=0),
                drive="curiosity", pressure=0.5)


def _edge(src: str, dst: str, output: str, seconds: float = 1.0, n: int = 3,
          hold_ms: int = 200) -> Traversal:
    """Ребро графа. `mode` — полный ключ действия: длительность часть действия."""
    mode = output if parse_action_key(output) else action_key(output, hold_ms)
    e = Traversal(src, dst, mode)
    for _ in range(n):
        e.observe(seconds)
    return e


# --- модель перехода --------------------------------------------------------


def test_model_says_i_do_not_know_instead_of_guessing() -> None:
    """«Не знаю» — законный ответ, и он отличается от «ничего не будет».

    Правдоподобное значение вместо реального построило бы план, который нечем
    сдержать, а причина была бы неотличима от ошибки планировщика.
    """
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1))
    assert m.predict("A", action_key(OUT_1, 200)) is not None
    assert m.predict("A", action_key(OUT_2, 200)) is None, "модель придумала переход"
    assert m.predict("Z", action_key(OUT_1, 200)) is None
    assert m.unknown == 2 and m.unknown_rate > 0.0
    assert m.stats()["unknown_rate"] > 0.0, "доля незнания должна быть на виду"


def test_model_ignores_edges_whose_action_is_unknown() -> None:
    """Переход, про который неизвестно, чем он пройден, повторить нечем."""
    m = ForwardModel()
    m.observe(_edge("A", "B", "unknown"))
    m.observe(_edge("A", "C", ""))
    assert m.transitions == {}
    assert m.actions_from("A") == []


def test_model_keeps_all_outcomes_not_just_the_likely_one() -> None:
    """Один и тот же выход в одном и том же месте уводит по-разному.

    Отброшенная как «не самая вероятная» ветка бывает единственной, ведущей к цели:
    на замере из-за этого не находилось 893 маршрута из 3205.
    """
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1, n=3))
    m.observe(_edge("A", "C", OUT_1, n=1))
    pred = m.predict("A", action_key(OUT_1, 200))
    assert pred is not None
    assert pred.likely.dst == "B" and pred.total_n == 4
    assert [o.dst for o, _ in pred.outcomes()] == ["B", "C"]
    assert abs(sum(p for _, p in pred.outcomes()) - 1.0) < 1e-9
    assert not pred.certain, "переход с двумя исходами не может быть определённым"


def test_model_marks_single_observation_as_a_guess() -> None:
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1, n=1))
    pred = m.predict("A", action_key(OUT_1, 200))
    assert pred is not None and pred.is_guess
    m.observe(_edge("A", "B", OUT_1, n=1))
    assert not m.predict("A", action_key(OUT_1, 200)).is_guess


def test_model_without_body_map_is_maximally_cautious() -> None:
    """Незнание обратимости — это и есть максимальная осторожность (инвариант 9)."""
    m = ForwardModel()
    assert m.caution(action_key(OUT_1, 200)) == 1.0
    body = BodyMap()
    f = body.fact(OUT_1, 0)
    f.reversibility = Reversibility(1.0, 0.05, 5)
    m2 = ForwardModel(body=body)
    assert m2.caution(action_key(OUT_1, 200)) < 1.0
    assert m2.caution(action_key(output_id("OUT", 77), 200)) == 1.0, "про незнакомый выход осторожность полная"


def test_ablation_can_drop_an_output_without_a_ban_list() -> None:
    """`skip` — для честных ablation-прогонов, а не для списка запретов."""
    g = PlaceGraph()
    g.edges[("A", "B", action_key(OUT_1, 200))] = _edge("A", "B", OUT_1)
    g.edges[("A", "C", action_key(OUT_2, 200))] = _edge("A", "C", OUT_2)
    full = build(g)
    ablated = build(g, skip=[OUT_2])
    assert full.actions_from("A") == sorted([action_key(OUT_1, 200), action_key(OUT_2, 200)])
    assert ablated.actions_from("A") == [action_key(OUT_1, 200)]


# --- планировщик: чего он делать не должен ----------------------------------


def test_no_reward_anywhere_in_the_planner() -> None:
    """Валюта одна — ошибка предсказания. Планировщик не заводит вторую.

    Проверяются имена в коде, а не текст: в пояснениях слово «награда» законно.
    """
    import ast

    import harness.behaviour.planner as planner_mod
    import harness.model.forward as forward_mod

    for mod in (planner_mod, forward_mod):
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.arg):
                names.add(node.arg)
            elif isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
        forbidden = ("reward", "score", "points", "utility", "payoff")
        hits = sorted(n for n in names if any(f in n.casefold() for f in forbidden))
        assert not hits, f"в {mod.__name__} завелась вторая валюта: {hits}"


def test_planner_does_not_route_through_what_it_does_not_know() -> None:
    """Ветка «не знаю» не строится: план через незнание нечем сдержать."""
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1))
    # До C ведёт только выход, которого никто не пробовал в B.
    plan = Planner(_profile(), m).plan(_goal("C"), "A", "C")
    assert plan is None


def test_planner_uses_measured_seconds_not_invented_cost() -> None:
    """Стоимость плана — сумма измеренных времён, а не придуманная величина."""
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1, seconds=0.5))
    m.observe(_edge("B", "C", OUT_2, seconds=1.5))
    plan = Planner(_profile(), m).plan(_goal("C"), "A", "C")
    assert plan is not None
    assert abs(plan.seconds - 2.0) < 1e-6
    assert [s.output for s in plan.steps] == [OUT_1, OUT_2]


def test_planner_takes_the_faster_of_two_routes() -> None:
    m = ForwardModel()
    m.observe(_edge("A", "X", OUT_1, seconds=0.2))
    m.observe(_edge("X", "C", OUT_2, seconds=0.2))
    m.observe(_edge("A", "C", OUT_3, seconds=5.0))
    plan = Planner(_profile(), m).plan(_goal("C"), "A", "C")
    assert plan is not None
    assert plan.length == 2 and plan.seconds < 1.0, \
        "выбран длинный по времени маршрут вместо короткого"


def test_planner_prefers_a_route_it_can_undo() -> None:
    """Осторожность определяется как «я не умею это откатить» (инвариант 9).

    Это не запрет: если безопасного маршрута нет, план строится и помечается
    опасным.
    """
    body = BodyMap()
    safe = body.fact(OUT_SAFE, 0)
    safe.reversibility = Reversibility(1.0, 0.05, 6)
    risky = body.fact(OUT_RISK, 0)
    risky.reversibility = Reversibility(0.0, 0.05, 6)
    m = ForwardModel(body=body)
    m.observe(_edge("A", "C", OUT_RISK, seconds=0.1))
    m.observe(_edge("A", "B", OUT_SAFE, seconds=1.0))
    m.observe(_edge("B", "C", OUT_SAFE, seconds=1.0))

    plan = Planner(_profile(), m).plan(_goal("C"), "A", "C")
    assert plan is not None
    assert plan.outputs() == [OUT_SAFE, OUT_SAFE], \
        "выбран быстрый неоткатываемый путь вместо медленного откатываемого"
    assert not plan.risky

    # Единственный путь через неоткатываемое — план есть, но помечен.
    only = ForwardModel(body=body)
    only.observe(_edge("A", "C", OUT_RISK, seconds=0.1))
    risky_plan = Planner(_profile(), only).plan(_goal("C"), "A", "C")
    assert risky_plan is not None and risky_plan.risky
    assert "неоткатываемый" in risky_plan.reason


def test_planner_skips_steps_that_more_often_lead_elsewhere() -> None:
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1, n=1))      # 1 из 4 → p = 0.25
    m.observe(_edge("A", "Z", OUT_1, n=3))
    strict = _profile(plan_min_step_p=0.5, plan_min_step_n=1)
    assert Planner(strict, m).plan(_goal("B"), "A", "B") is None
    plan = Planner(_profile(plan_min_step_p=0.2, plan_min_step_n=1), m).plan(
        _goal("B"), "A", "B")
    assert plan is not None and abs(plan.confidence - 0.25) < 1e-9


def test_plan_confidence_is_the_product_of_step_probabilities() -> None:
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1, n=3))
    m.observe(_edge("A", "Z", OUT_1, n=3))      # p = 0.5
    m.observe(_edge("B", "C", OUT_2, n=3))
    m.observe(_edge("B", "Y", OUT_2, n=3))      # p = 0.5
    plan = Planner(_profile(plan_min_step_p=0.4, plan_min_step_n=1), m).plan(
        _goal("C"), "A", "C")
    assert plan is not None
    assert abs(plan.confidence - 0.25) < 1e-9, "где тоньше, там и рвётся"


def test_plan_is_a_guess_until_it_is_applied() -> None:
    """То же правило, что у навыка: применение — единственное свидетельство."""
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1))
    plan = Planner(_profile(), m).plan(_goal("B"), "A", "B")
    assert plan is not None and plan.is_guess
    execute(plan, act=lambda a: "B")
    assert not plan.is_guess and plan.worked == 1


def test_planner_is_interruptible_and_has_an_answer_at_any_moment() -> None:
    """Мир не ставится на паузу (инвариант 3): планировщик — генератор."""
    m = ForwardModel()
    for i in range(30):
        m.observe(_edge(f"P{i}", f"P{i + 1}", OUT_1, seconds=0.1))
    planner = Planner(_profile(plan_max_length=40), m)
    steps = 0
    got_answer_before_the_end = False
    for progress in planner.search(_goal("P30"), "P0", "P30"):
        steps += 1
        if progress.best is not None and not progress.done:
            got_answer_before_the_end = True
        assert progress.expansions <= planner.max_expansions
    assert steps > 1, "поиск не отдавал управление ни разу"
    assert got_answer_before_the_end or planner.best is not None


def test_planner_respects_its_expansion_budget() -> None:
    m = ForwardModel()
    for i in range(200):
        for j in range(3):
            m.observe(_edge(f"P{i}", f"P{i + 1}", output_id("OUT", 2 + j), seconds=0.1))
    planner = Planner(_profile(plan_max_expansions=25, plan_max_length=100), m)
    for _ in planner.search(_goal("P199"), "P0", "P199"):
        pass
    assert planner.expansions <= 25


# --- репетиция воображением -------------------------------------------------


def test_rehearsal_goes_through_the_breaker_and_writes_thoughts(tmp_path) -> None:
    """Продуманное отличимо от сделанного: `THOUGHT`, а не `ACTION` (инвариант 1)."""
    from harness.behaviour.imagination import Breaker, GuardedEffectors, Loop
    from harness.session import Recorder

    profile = _profile()
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1))
    m.observe(_edge("B", "C", OUT_2))
    plan = Planner(profile, m).plan(_goal("C"), "A", "C")
    assert plan is not None

    with Recorder(tmp_path / "s", profile=profile, source="test",
                  synthetic=True) as rec:
        breaker = Breaker()
        touched: list[Action] = []
        effectors = GuardedEffectors(breaker, touched.append)
        loop = Loop(execute=effectors, simulate=lambda a, mode: "воображено",
                    journal=rec.journal, breaker=breaker)
        ok, why = rehearse(plan, m, loop, "A",
                           stamp_of=lambda i: Stamp(i, i, None))
        assert ok, why
        assert touched == [], "репетиция дотянулась до мира"
        kinds = [e.kind for e in rec.journal if e.action is not None]
    assert kinds and all(k is Kind.THOUGHT for k in kinds)
    assert Kind.ACTION not in kinds


def test_rehearsal_catches_a_plan_the_model_no_longer_supports() -> None:
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1))
    plan = Planner(_profile(), m).plan(_goal("B"), "A", "B")
    assert plan is not None

    from harness.behaviour.imagination import Loop

    forgetful = ForwardModel()                      # модель без этого перехода
    loop = Loop(execute=lambda a: None, simulate=lambda a, mode: None)
    ok, why = rehearse(plan, forgetful, loop, "A")
    assert not ok and "разучилась" in why


# --- исполнение -------------------------------------------------------------


def test_execution_stops_at_the_first_surprise() -> None:
    """Дальше план опирается на состояние, которого нет. Идти вслепую нельзя."""
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1))
    m.observe(_edge("B", "C", OUT_2))
    plan = Planner(_profile(), m).plan(_goal("C"), "A", "C")
    assert plan is not None and plan.length == 2

    ex = execute(plan, act=lambda a: "НЕ_ТУДА")
    assert ex.steps_done == 1 and ex.surprises == 1
    assert ex.abandoned_at == 0 and not ex.goal_passed
    assert "брошен" in ex.reason


def test_execution_writes_each_step_agreement_to_the_journal(tmp_path) -> None:
    from harness.session import Recorder

    profile = _profile()
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1))
    plan = Planner(profile, m).plan(_goal("B"), "A", "B")
    assert plan is not None

    with Recorder(tmp_path / "s", profile=profile, source="test",
                  synthetic=True) as rec:
        execute(plan, act=lambda a: "B", journal=rec.journal,
                stamp_of=lambda i: Stamp(i, i, None))
        notes = [e for e in rec.journal if e.event.get("code") == "plan_step"]
    assert len(notes) == 1
    assert notes[0].event["agreed"] is True
    assert notes[0].event["expected"] == "B" and notes[0].event["observed"] == "B"


def test_goal_test_decides_success_not_the_model() -> None:
    """План может дойти по модели туда, где тест цели всё равно не проходит."""
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1))
    goal = _goal("B", test=lambda: False)
    plan = Planner(_profile(), m).plan(goal, "A", "B")
    assert plan is not None
    ex = execute(plan, act=lambda a: "B", goal=goal)
    assert ex.surprises == 0 and not ex.goal_passed
    assert "тест цели не прошёл" in ex.reason


# --- полный круг ------------------------------------------------------------


def test_travel_replans_after_a_surprise() -> None:
    """Расхождение с моделью не игнорируется: план брошен, строится новый."""
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1))
    m.observe(_edge("B", "C", OUT_2))
    m.observe(_edge("X", "C", OUT_3))

    place = {"here": "A"}
    surprised = {"done": False}

    def act(action: Action) -> str:
        out = action.outputs_touched()[0]
        if out == OUT_1 and not surprised["done"]:
            surprised["done"] = True
            place["here"] = "X"          # мир увёл не туда, куда обещала модель
        elif out == OUT_3:
            place["here"] = "C"
        elif out == OUT_1:
            place["here"] = "B"
        elif out == OUT_2:
            place["here"] = "C"
        return place["here"]

    goal = _goal("C", test=lambda: place["here"] == "C")
    j = travel(_profile(), goal, "C", where=lambda: place["here"], act=act,
               rebuild=lambda: m)
    assert j.arrived and j.replans == 1, j.as_dict()
    assert len(j.attempts) == 2


def test_travel_says_there_is_no_plan_instead_of_inventing_one() -> None:
    """Из места, из которого ни разу не уходили, вести некуда.

    Правильный ответ здесь — «идти исследовать», а не придуманный план.
    """
    m = ForwardModel()
    m.observe(_edge("B", "C", OUT_1))
    goal = _goal("C", test=lambda: False)
    j = travel(_profile(), goal, "C", where=lambda: "A",
               act=lambda a: "A", rebuild=lambda: m)
    assert not j.arrived
    assert "плана нет" in j.reason and "исследовать" in j.reason


def test_travel_gives_up_when_the_goal_budget_runs_out() -> None:
    """Бюджет цели кончился — отказ с причиной (правило 2 из goals.py)."""
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1))
    m.observe(_edge("B", "A", OUT_2))
    # Мир не двигается: каждый план даёт расхождение, и так до конца бюджета.
    goal = _goal("B", test=lambda: False, budget=6)
    j = travel(_profile(), goal, "B", where=lambda: "A", act=lambda a: "A",
               rebuild=lambda: m)
    assert not j.arrived and "бюджет" in j.reason
    assert j.ticks >= goal.budget_ticks


def test_travel_can_be_told_not_to_replan() -> None:
    m = ForwardModel()
    m.observe(_edge("A", "B", OUT_1))
    goal = _goal("B", test=lambda: False)
    j = travel(_profile(plan_replan_on_surprise=False), goal, "B",
               where=lambda: "A", act=lambda a: "ДРУГОЕ", rebuild=lambda: m)
    assert not j.arrived and j.replans == 0
    assert "перепланирование выключено" in j.reason


# --- сквозной прогон по интерактивному миру ---------------------------------


def test_planner_reaches_a_place_in_the_interactive_world() -> None:
    """Главная проверка: весь стек целиком закрывает цель «добраться до места».

    Лепет находит обратные пары → разведка ими возвращается и подтверждает переходы
    → модель становится пригодной для плана → планировщик строит цепочку → цепочка
    исполняется и тест цели проходит. Без единого названия клавиши и без правил мира:
    соответствие «выход → эффект» выведено из сида и лежит только в отладочном потоке.

    Каждое звено здесь необходимо, и это измерено. Разведка без возврата даёт за 2500
    шагов **ни одного** подтверждённого перехода и ни одного плана; с возвратом — 40
    планов, из которых доходят 40. Числа — в ARCHITECTURE-AGENT.md.
    """
    profile = _profile()
    hold = 200
    min_n = int(profile.parameters["plan_min_step_n"])

    # 1. Лепет: обратные пары. Своё знание «чем это откатывается», а не «назад».
    babble_world = InteractiveWorld(profile, seed=5, n_outputs=16)
    babbler = Babbler(profile, babble_world.outputs, rng_seed=5)
    run_babbling(babble_world, babbler, steps=1200, clocks=Clocks())
    inverse = dict(babbler.inverse_found)
    assert inverse, "лепет не нашёл ни одной обратной пары — возвращаться нечем"

    # 2. Разведка, которая подтверждает.
    world = InteractiveWorld(profile, seed=5, n_outputs=16)
    graph = PlaceGraph.from_profile(profile)
    state: dict[str, object] = {"seq": 0, "last": None}

    def step(output: str, duration_ms: int = 200) -> str:
        obs = world.step(Action.key(output, duration_ms), with_audio=False)
        state["seq"] = int(state["seq"]) + 1
        state["last"] = output
        return graph.see(obs.frame, int(state["seq"]), seconds_per_seq=1 / 30.0,
                         mode=action_key(output, duration_ms))

    graph.see(world.step(None, with_audio=False).frame, 0,
              seconds_per_seq=1 / 30.0, mode="start")
    model = ForwardModel.from_graph(graph)
    for i in range(1500):
        if i % 25 == 0:
            model = ForwardModel.from_graph(graph)
        out, ms = choose_probe(model, graph.current, world.outputs, hold_ms=hold,
                               min_n=min_n, inverse=inverse,
                               last_output=state["last"])
        step(out, ms)

    model = ForwardModel.from_graph(graph)
    confirmed = sum(1 for outs in model.transitions.values()
                    for o in outs if o.n >= min_n)
    assert confirmed, "разведка не подтвердила ни одного перехода"

    # 3. Цель — место, куда модель знает дорогу под своими же порогами.
    def model_reach(src: str, max_depth: int = 4) -> dict[str, int]:
        depth = {src: 0}
        frontier = [src]
        for d in range(max_depth):
            nxt = []
            for node in frontier:
                for key in model.actions_from(node):
                    pred = model.predict(node, key)
                    if pred is None:
                        continue
                    for outcome, p in pred.outcomes():
                        if p < 0.5 or outcome.n < min_n:
                            continue
                        if outcome.dst not in depth:
                            depth[outcome.dst] = d + 1
                            nxt.append(outcome.dst)
            frontier = nxt
        return {k: v for k, v in depth.items() if v > 0}

    here = graph.current
    assert here is not None
    reachable = model_reach(here)
    assert reachable, "стоим там, откуда модель дороги не знает"

    target = sorted(reachable)[-1]
    goal = _goal(target, test=lambda: graph.current == target)
    plan = Planner(profile, model).plan(goal, here, target)
    assert plan is not None, "план до места, куда модель знает дорогу, не найден"
    assert plan.is_guess and plan.min_step_n >= min_n

    # 4. Исполнение: каждый шаг сверяется с предсказанием.
    ex = execute(plan, act=lambda a: step(a.outputs_touched()[0], a.duration_ms),
                 goal=goal)
    assert ex.goal_passed, ex.as_dict()
    assert ex.surprises == 0
    assert not plan.is_guess and plan.worked == 1


def test_exploration_without_going_back_confirms_nothing() -> None:
    """Замер, из которого взялся возврат обратной парой.

    Незнакомое место соблазняет расширяться дальше, и разведка уходит в
    бесконечность. Это про мир и представление, а не про планировщик.
    """
    profile = _profile()
    world = InteractiveWorld(profile, seed=5, n_outputs=16)
    graph = PlaceGraph.from_profile(profile)
    state = {"seq": 0, "last": None}

    def step(output: str, duration_ms: int = 200) -> str:
        obs = world.step(Action.key(output, duration_ms), with_audio=False)
        state["seq"] += 1
        state["last"] = output
        return graph.see(obs.frame, state["seq"], seconds_per_seq=1 / 30.0,
                         mode=action_key(output, duration_ms))

    graph.see(world.step(None, with_audio=False).frame, 0,
              seconds_per_seq=1 / 30.0, mode="start")
    model = ForwardModel.from_graph(graph)
    for i in range(1200):
        if i % 25 == 0:
            model = ForwardModel.from_graph(graph)
        out, ms = choose_probe(model, graph.current, world.outputs, hold_ms=200,
                               min_n=2)                     # без обратных пар
        step(out, ms)
    model = ForwardModel.from_graph(graph)
    confirmed = sum(1 for outs in model.transitions.values()
                    for o in outs if o.n >= 2)
    assert confirmed == 0, (
        f"разведка без возврата неожиданно подтвердила {confirmed} переходов — "
        "тогда возврат не нужен, и это надо перезамерить")


def test_random_exploration_does_not_produce_a_plannable_model() -> None:
    """Замер, из которого взялась подтверждающая разведка.

    Случайная разведка даёт больше мест и меньше знания: переходы наблюдаются по
    одному разу, и планировать по ним нельзя. Это про мир и представление, а не про
    планировщик, и потому проверяется отдельным тестом.
    """
    profile = _profile()
    world = InteractiveWorld(profile, seed=5, n_outputs=16)
    graph = PlaceGraph.from_profile(profile)
    rng = np.random.default_rng(0)
    state = {"seq": 0}

    graph.see(world.step(None, with_audio=False).frame, 0,
              seconds_per_seq=1 / 30.0, mode="start")
    for _ in range(1200):
        out = world.outputs[int(rng.integers(len(world.outputs)))]
        obs = world.step(Action.key(out, 200), with_audio=False)
        state["seq"] += 1
        graph.see(obs.frame, state["seq"], seconds_per_seq=1 / 30.0,
                  mode=action_key(out, 200))

    model = ForwardModel.from_graph(graph)
    confirmed = sum(1 for key in model.transitions
                    for o in model.transitions[key] if o.n >= 2)
    total = sum(len(v) for v in model.transitions.values())
    assert total > 100
    assert confirmed / total < 0.25, (
        f"случайная разведка неожиданно подтвердила {confirmed}/{total} переходов — "
        "тогда подтверждающая разведка не нужна, и это надо перезамерить")


def test_place_grid_forks_the_journal() -> None:
    """Сетка отпечатка меняет смысл каждого места разом, значит форкает журнал."""
    a = _profile()
    b = _profile(place_grid=6)
    assert "place_grid" in a.structural
    assert a.structure_hash != b.structure_hash
