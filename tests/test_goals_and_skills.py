"""Цели и навыки.

Главный тест здесь — последний: замкнутый круг «давление драйва → цель → пробы →
тест пройден». Пока он не проходит, всё остальное — отдельно стоящие механизмы.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.behaviour.babbling import Babbler, run_babbling
from harness.behaviour.goals import (Candidate, Goal, GoalError, GoalStack,
                                     KIND_LEARN_OUTPUT, KIND_UNDO_OUTPUT, State,
                                     candidates_from_beliefs, candidates_from_body,
                                     candidates_from_places, choose)
from harness.behaviour.skills import Library, Skill, SkillError, Step, mine, try_undo, verify
from harness.core.action import Action, Reversibility
from harness.core.journal import Kind
from harness.core.profile import BABBLE, MILESTONE_0, from_schema
from harness.corpus.world import Effect, InteractiveWorld
# Testimony импортируется под другим именем: pytest пытается собрать класс,
# начинающийся на Test, как набор тестов.
from harness.model.beliefs import (BeliefStore, Origin, Provenance,
                                   Testimony as Claim, merge_testimony)
from harness.model.drives import Motivation
from harness.model.places import PlaceGraph, fingerprint
from harness.model.rebuild import BodyMap
from harness.session import Recorder, Session


# --- цели -------------------------------------------------------------------


def _cand(kind: str = KIND_LEARN_OUTPUT, target: str = "OUT_0A11",
          drive: str = "curiosity", gain: float = 1.0,
          test=lambda: False) -> Candidate:
    return Candidate(kind, target, drive, gain, test, f"проверка для {target}")


def test_goal_without_test_is_impossible() -> None:
    prov = Provenance(Origin.EXPERIENCE, "b0", 1)
    with pytest.raises(GoalError, match="без способа проверки"):
        Goal("G1", KIND_LEARN_OUTPUT, "OUT_0A11", test=None, test_text="есть",  # type: ignore[arg-type]
             budget_ticks=10, provenance=prov, drive="curiosity", pressure=1.0)
    with pytest.raises(GoalError, match="не описан словами"):
        Goal("G1", KIND_LEARN_OUTPUT, "OUT_0A11", test=lambda: True, test_text="",
             budget_ticks=10, provenance=prov, drive="curiosity", pressure=1.0)


def test_goal_needs_positive_budget() -> None:
    prov = Provenance(Origin.EXPERIENCE, "b0", 1)
    with pytest.raises(GoalError, match="бюджет"):
        Goal("G1", KIND_LEARN_OUTPUT, "OUT_0A11", test=lambda: True, test_text="есть",
             budget_ticks=0, provenance=prov, drive="curiosity", pressure=1.0)


def test_goal_from_testimony_must_be_marked_human() -> None:
    """Цель рождается из своего опыта. Пришедшая снаружи — вмешательство."""
    outside = Provenance(Origin.TESTIMONY, "b0", 1, "человек", 0.8)
    with pytest.raises(GoalError, match="вмешательство"):
        Goal("G1", KIND_LEARN_OUTPUT, "OUT_0A11", test=lambda: True, test_text="есть",
             budget_ticks=5, provenance=outside, drive="curiosity", pressure=1.0)
    ok = Goal("G2", KIND_LEARN_OUTPUT, "OUT_0A11", test=lambda: True, test_text="есть",
              budget_ticks=5, provenance=outside, drive="human", pressure=1.0)
    assert ok.drive == "human"


def test_goal_passes_when_test_passes(tmp_path: Path) -> None:
    done = [False]
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        stack = GoalStack(MILESTONE_0, journal=rec.journal)
        stack.push(_cand(test=lambda: done[0]), Motivation(MILESTONE_0), 1, "b0",
                   rec.clocks.stamp())
        assert stack.tick(rec.clocks.stamp()) is not None
        done[0] = True
        stack.tick(rec.clocks.stamp())
        assert stack.stats()["passed"] == 1
        assert stack.active is None

    with Session.open(tmp_path / "s") as s:
        s.journal.verify()
        codes = [e.event.get("code") for e in s.journal if e.kind is Kind.GOAL]
    assert "set" in codes and "passed" in codes


def test_goal_is_abandoned_when_budget_runs_out(tmp_path: Path) -> None:
    """Отказ по бюджету пишется в журнал с причиной, а не просто исчезает."""
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        stack = GoalStack(MILESTONE_0, journal=rec.journal)
        goal = stack.push(_cand(test=lambda: False), Motivation(MILESTONE_0), 1, "b0",
                          rec.clocks.stamp(), budget_ticks=3)
        for _ in range(5):
            stack.tick(rec.clocks.stamp())
        assert goal.state is State.ABANDONED
        assert goal.reason and "бюджет" in goal.reason
        assert stack.stats()["mean_ticks_to_abandon"] is not None

    with Session.open(tmp_path / "s") as s:
        entries = [e for e in s.journal if e.kind is Kind.GOAL
                   and e.event.get("code") == "abandoned"]
    assert len(entries) == 1 and entries[0].event["reason"]


def test_more_pressing_goal_defers_the_current_one() -> None:
    m = Motivation(MILESTONE_0)
    stack = GoalStack(MILESTONE_0)
    weak = stack.push(_cand(target="OUT_0001", gain=0.01), m, 1, "b0")
    strong = stack.push(_cand(target="OUT_0002", gain=100.0), m, 2, "b0")
    assert strong.state is State.ACTIVE
    assert weak.state is State.DEFERRED


def test_deferred_goal_resumes_after_the_active_one_finishes() -> None:
    m = Motivation(MILESTONE_0)
    stack = GoalStack(MILESTONE_0)
    first = stack.push(_cand(target="OUT_0001", gain=0.01, test=lambda: False), m, 1, "b0",
                       budget_ticks=2)
    second = stack.push(_cand(target="OUT_0002", gain=100.0, test=lambda: False), m, 2, "b0",
                        budget_ticks=2)
    assert second.state is State.ACTIVE and first.state is State.DEFERRED
    for _ in range(3):
        stack.tick()
    assert second.state is State.ABANDONED
    assert first.state is State.ACTIVE, "отложенная цель не вернулась в работу"


def test_rejecting_a_goal_is_an_intervention(tmp_path: Path) -> None:
    """Браковка целей исследователем видна отдельно от собственных решений."""
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        stack = GoalStack(MILESTONE_0, journal=rec.journal)
        goal = stack.push(_cand(), Motivation(MILESTONE_0), 1, "b0", rec.clocks.stamp())
        stack.reject(goal.id, rec.clocks.stamp())
        assert goal.state is State.REJECTED

    with Session.open(tmp_path / "s") as s:
        interventions = [e for e in s.journal if e.kind is Kind.INTERVENTION]
    assert len(interventions) == 1
    assert interventions[0].event["code"] == "goal_rejected"


def test_no_reward_anywhere_in_goals() -> None:
    """Награды нет: валюта одна — ошибка предсказания.

    Проверяются **имена в коде**, а не текст файла: в пояснениях слово «награда»
    законно — там сказано, почему её нет. А вот поле, функция или переменная с
    таким именем означала бы, что рядом с ошибкой предсказания завелась вторая
    валюта, и тогда непонятно, какая из них главная.
    """
    import ast

    import harness.behaviour.goals as mod

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
    assert not hits, f"в целях завелась вторая валюта: {hits}"


# --- откуда берутся кандидаты -----------------------------------------------


def test_candidates_from_body_prefer_the_unknown() -> None:
    body = BodyMap()
    known = body.fact("OUT_0001", 0)
    known.delivered, known.responded = 5, 5
    known.reversibility = Reversibility(1.0, 0.05, 5)
    unknown = body.fact("OUT_0002", 0)
    unknown.delivered, unknown.responded = 5, 5

    cands = candidates_from_body(body, ["OUT_0001", "OUT_0002", "OUT_0003"],
                                caution_threshold=0.72)
    kinds = {(c.kind, c.target) for c in cands}
    assert (KIND_LEARN_OUTPUT, "OUT_0003") in kinds, "непробованный выход не стал целью"
    assert (KIND_UNDO_OUTPUT, "OUT_0002") in kinds, "неизвестная обратимость не стала целью"
    assert all(c.target != "OUT_0001" for c in cands), "изученный выход снова стал целью"


def test_candidates_from_beliefs_target_hearsay() -> None:
    store = BeliefStore("b0")
    merge_testimony(store, [Claim("ENT_0001|dyn|светится", "вики", 0.9, "b0", 3)])
    cands = candidates_from_beliefs(store)
    assert cands and all(c.drive == "order" for c in cands)
    assert not cands[0].test(), "чужое слово уже считается проверенным"


def test_candidates_from_places_target_stale_ones() -> None:
    graph = PlaceGraph()
    rng = np.random.default_rng(0)
    for i in range(4):
        graph.observe(fingerprint(rng.integers(0, 255, (64, 96), dtype=np.uint8)),
                      i * 100)
    cands = candidates_from_places(graph, stale_after=150)
    assert cands, "давно не посещённые места не стали целями"
    assert all(c.drive == "curiosity" for c in cands)


def test_choose_orders_by_pressure_times_gain() -> None:
    m = Motivation(MILESTONE_0)
    for _ in range(20):
        m.update(error_mean=0.2, error_sigma=0.05, error_now=0.05,
                 unknown_reversibility=9)
    high = _cand(target="OUT_HIGH", drive=m.dominant().name, gain=1.0)
    low = _cand(target="OUT_LOW", drive=m.dominant().name, gain=0.01)
    assert choose([low, high], m, top=1)[0].target == "OUT_HIGH"


# --- навыки -----------------------------------------------------------------


def test_skill_shorter_than_two_steps_is_not_a_skill() -> None:
    prov = Provenance(Origin.EXPERIENCE, "b0", 1)
    with pytest.raises(SkillError, match="короче двух шагов"):
        Skill("S1", (Step("OUT_0A11", 100),), 1.0, 0.1, 3, prov)


def test_mine_finds_repeated_chain(tmp_path: Path) -> None:
    from harness.core.journal import Actor

    chain = [("OUT_0A11", 100), ("OUT_0B22", 200)]
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        for _ in range(4):
            for out, ms in chain:
                rec.journal.append(Kind.ACTION, rec.clocks.stamp(), Actor.AGENT,
                                   action=Action.key(out, ms),
                                   event={"code": "delivered", "responded": True,
                                          "device": "t"})
            # разрыв: цепочка кончилась
            rec.record_note("пауза")

    with Session.open(tmp_path / "s") as s:
        found = mine(s.journal, min_repeats=3)
    assert found, "повторяющаяся цепочка не найдена"
    best = found[0]
    assert [st.output for st in best.steps] == ["OUT_0A11", "OUT_0B22"]
    assert [st.duration_ms for st in best.steps] == [100, 200]
    assert best.n == 4 and best.is_guess


def test_mine_ignores_silent_and_masked(tmp_path: Path) -> None:
    """Макрос из молчащих шагов ничего не делает, сколько бы раз ни повторился."""
    from harness.core.journal import Actor

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        for _ in range(6):
            rec.journal.append(Kind.ACTION, rec.clocks.stamp(), Actor.AGENT,
                               action=Action.key("OUT_0A11", 100),
                               event={"code": "delivered", "responded": False,
                                      "device": "t"})
            rec.journal.append(Kind.ACTION, rec.clocks.stamp(), Actor.AGENT,
                               action=Action.key("OUT_0B22", 100).masked_as("mask:window"),
                               event={"code": "masked", "device": "t"})

    with Session.open(tmp_path / "s") as s:
        assert mine(s.journal, min_repeats=2) == []


def test_mine_prefers_longer_chain_over_its_prefix(tmp_path: Path) -> None:
    from harness.core.journal import Actor

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        for _ in range(4):
            for out in ("OUT_0A11", "OUT_0B22", "OUT_0C33"):
                rec.journal.append(Kind.ACTION, rec.clocks.stamp(), Actor.AGENT,
                                   action=Action.key(out, 100),
                                   event={"code": "delivered", "responded": True,
                                          "device": "t"})
            rec.record_note("пауза")

    with Session.open(tmp_path / "s") as s:
        found = mine(s.journal, min_repeats=4, max_length=3)
    lengths = {len(s.steps) for s in found}
    assert 3 in lengths, "длинная цепочка не найдена"
    assert 2 not in lengths, "короткий префикс не отброшен, хотя длинная цепочка та же"


def test_skill_stays_a_guess_until_applied(tmp_path: Path) -> None:
    from harness.core.journal import Actor

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        for _ in range(3):
            for out in ("OUT_0A11", "OUT_0B22"):
                rec.journal.append(Kind.ACTION, rec.clocks.stamp(), Actor.AGENT,
                                   action=Action.key(out, 100),
                                   event={"code": "delivered", "responded": True,
                                          "device": "t"})
            rec.record_note("пауза")

    with Session.open(tmp_path / "s") as s:
        lib = Library()
        lib.from_journal(s.journal, min_repeats=3)
    skill = lib.best_for()[0]
    assert skill.is_guess
    guess_confidence = skill.confidence
    skill.observe_use(True)
    assert not skill.is_guess
    assert skill.confidence > guess_confidence, "проверка не повысила надёжность"


def test_skill_verify_uses_objective_check() -> None:
    """Навык подтверждается изменением мира, а не тем, что его применили."""
    world = InteractiveWorld(BABBLE, seed=3)
    fwd = [o for o, e in world._effects.items() if e is Effect.FORWARD][0]
    back = [o for o, e in world._effects.items() if e is Effect.BACK][0]
    prov = Provenance(Origin.EXPERIENCE, "b0", 1)
    moving = Skill("S1", (Step(fwd, 200), Step(fwd, 200)), 1.0, 0.1, 3, prov)

    def moved(before, after) -> bool:
        return before.frame.shape == after.frame.shape and not np.array_equal(
            before.frame, after.frame)

    assert verify(moving, world, expect=moved)
    assert moving.verified == 1 and moving.verified_ok == 1

    silent_out = world.truth()["silent_outputs"][0]
    nothing = Skill("S2", (Step(silent_out, 200), Step(silent_out, 200)), 1.0, 0.1, 3, prov)
    world2 = InteractiveWorld(BABBLE, seed=3)

    def state_changed(before, after) -> bool:
        del before, after
        return world2.state.snapshot() != snapshot

    snapshot = world2.state.snapshot()
    assert not verify(nothing, world2, expect=state_changed)
    assert nothing.mu < 1.0, "неработающий навык остался с прежней оценкой"
    del back


def test_skill_reversibility_is_its_own_question() -> None:
    """Обратимость цепочки не выводится из обратимости шагов."""
    world = InteractiveWorld(BABBLE, seed=3)
    fwd = [o for o, e in world._effects.items() if e is Effect.FORWARD][0]
    back = [o for o, e in world._effects.items() if e is Effect.BACK][0]
    prov = Provenance(Origin.EXPERIENCE, "b0", 1)
    skill = Skill("S1", (Step(fwd, 200), Step(fwd, 200)), 1.0, 0.1, 3, prov)
    assert not skill.reversibility.is_known, "обратимость навыка известна заранее"

    # Сравнивать сырые кадры здесь нельзя, и это не мелочь: интерфейс живёт своей
    # жизнью — полосы анимированы, — поэтому кадр не совпадёт сам с собой даже
    # после идеального отката. Сравнивается только мировой слой. Ровно за этим и
    # нужно разделение себя и мира: без него «вернулось ли как было» неотличимо
    # от «пока я ходил, дёрнулась полоса на панели».
    world_mask = ~world.hud_mask()

    def same(before, after) -> bool:
        a = before.frame[world_mask]
        b = after.frame[world_mask]
        return bool((a == b).mean() > 0.999)

    undone = try_undo(skill, world, [Step(back, 200), Step(back, 200)], same_as=same)
    assert skill.reversibility.is_known
    assert undone, "откат двумя шагами назад не вернул мир"


def test_skill_mined_from_babbling_is_real(tmp_path: Path) -> None:
    """Навыки, найденные в настоящем прогоне лепета, проверяются в мире."""
    world = InteractiveWorld(BABBLE, seed=11)
    with Recorder(tmp_path / "s", profile=BABBLE, source="babble",
                  synthetic=True) as rec:
        babbler = Babbler(BABBLE, world.outputs, journal=rec.journal, rng_seed=1)
        run_babbling(world, babbler, steps=900, clocks=rec.clocks)

    with Session.open(tmp_path / "s") as s:
        lib = Library()
        lib.from_journal(s.journal, min_repeats=3)
    assert len(lib) >= 1, "в настоящем прогоне не нашлось ни одного макроса"
    assert all(sk.is_guess for sk in lib.skills.values()), (
        "найденное в журнале сразу считается умением — так учат случайность")

    # Каждый найденный навык состоит только из выходов, которые действительно
    # что-то делают: иначе поиск подобрал шум.
    live = set(world.truth()["live_outputs"])
    for sk in lib.skills.values():
        for step in sk.steps:
            assert step.output in live, f"в навык попал молчащий выход {step.output}"


# --- замкнутый круг ---------------------------------------------------------


def test_closed_loop_drive_to_goal_to_action_to_passed(tmp_path: Path) -> None:
    """Главный тест: давление драйва порождает цель, пробы её закрывают.

    Ни одна часть здесь не подыгрывает: цель ставится по давлению, проверяется
    объективным тестом по карте тела, а карта тела заполняется настоящими пробами
    в мире, который про цель ничего не знает.
    """
    profile = from_schema("круг", capture_width=320, capture_height=180,
                          babble_rate=0.9, babble_repeats=2, drive_horizon_s=60.0)
    world = InteractiveWorld(profile, seed=13)

    with Recorder(tmp_path / "s", profile=profile, source="loop",
                  synthetic=True) as rec:
        babbler = Babbler(profile, world.outputs, journal=rec.journal, rng_seed=2)
        motivation = Motivation(profile)
        stack = GoalStack(profile, journal=rec.journal)
        branch = rec.journal.meta.branch_id

        for round_no in range(14):
            motivation.update(
                error_mean=0.1, error_sigma=0.02, error_now=0.05,
                unknown_reversibility=babbler.progress()["unknown_reversibility"])

            if stack.active is None:
                cands = candidates_from_body(
                    babbler.body, world.outputs,
                    caution_threshold=profile.parameters["irreversibility_threshold"])
                if cands:
                    best = choose(cands, motivation, top=1)[0]
                    stack.push(best, motivation, rec.journal.seq, branch,
                               rec.clocks.stamp(), budget_ticks=12)

            run_babbling(world, babbler, steps=40, clocks=rec.clocks)
            stack.tick(rec.clocks.stamp())
            del round_no

        stats = stack.stats()

    assert stats["goals"] >= 2, f"цели не ставились: {stats}"
    assert stats["passed"] >= 1, (
        f"ни одна цель не прошла тест — круг не замкнулся: {stats}")

    with Session.open(tmp_path / "s") as s:
        s.journal.verify()
        goal_entries = [e for e in s.journal if e.kind is Kind.GOAL]
        codes = {e.event.get("code") for e in goal_entries}
    assert "set" in codes and "passed" in codes
    assert all(e.actor.value == "agent" for e in goal_entries), (
        "цель поставлена не агентом — значит пришла снаружи")


def test_closed_loop_leaves_the_journal_explaining_everything(tmp_path: Path) -> None:
    """Из журнала выводится всё: и карта тела, и то, какие цели ставились."""
    from harness.model.rebuild import rebuild_from_journal

    profile = from_schema("круг", capture_width=320, capture_height=180,
                          babble_repeats=2)
    world = InteractiveWorld(profile, seed=17)
    with Recorder(tmp_path / "s", profile=profile, source="loop",
                  synthetic=True) as rec:
        babbler = Babbler(profile, world.outputs, journal=rec.journal, rng_seed=3)
        stack = GoalStack(profile, journal=rec.journal)
        stack.push(_cand(test=lambda: bool(babbler.body.outputs)),
                   Motivation(profile), rec.journal.seq,
                   rec.journal.meta.branch_id, rec.clocks.stamp(), budget_ticks=50)
        run_babbling(world, babbler, steps=200, clocks=rec.clocks)
        stack.tick(rec.clocks.stamp())

    with Session.open(tmp_path / "s") as s:
        rebuilt = rebuild_from_journal(s.journal)
        goals = [e.event for e in s.journal if e.kind is Kind.GOAL]
    assert rebuilt.body.outputs, "карта тела не восстановилась из журнала"
    assert goals and goals[0]["kind"], "цели не видны в журнале"
    assert all("test" not in g for g in goals), (
        "текст теста утёк в журнал: он для исследователя, а не часть опыта")
