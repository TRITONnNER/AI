"""Семь осей модуляции доходят до поведения. `TASK-06`, часть 1, расхождение «эмоции».

Аудит нашёл здесь «три оси из семи». Поиск потребителей нашёл хуже: `modulation()`
вызывался ровно в одном месте — при печати отчёта, — а планировщик и лепет брали
пороги прямо из профиля. То есть эмоция не модулировала **ничего**, и три
«реализованные» оси были мёртвым кодом.

Поэтому проверяется не наличие поля, а **сдвиг числа в поведении потребителя**
(инвариант 25). Тест на ось, у которой потребителя нет, не притворяется пройденным:
такая ось объявлена в `CONSUMERS` как `None` и докладывается числом.

Единица независимости для каждой оси — **ось**, `n = 7`. Не «прогон» и не «такт»:
утверждение здесь про устройство механизма, и повторение одного прогона тысячу раз
не добавляет ни одного независимого наблюдения о том, читается ли ось.
"""

from __future__ import annotations

import pytest

from harness.core.profile import MILESTONE_0, from_schema
from harness.model.drives import (AXES, CONSUMERS, Modulation, Mood, Motivation,
                                  axes_with_consumer)


def _mood(m: Motivation, valence: float, arousal: float) -> Modulation:
    m.mood = Mood(valence, arousal)
    return m.modulation()


def test_seven_axes_declared() -> None:
    """Семь осей из MIND.md: ширина внимания, горизонт, необратимость, темп, право
    рефлексов, доля ресурсов, инерция возврата."""
    assert len(AXES) == 7
    assert set(CONSUMERS) == set(AXES), "у каждой оси объявлено, кто её читает"


def test_axes_without_consumer_are_reported_not_hidden() -> None:
    """Ось без потребителя — незакрытый долг, и он предъявляется числом."""
    live = axes_with_consumer()
    dead = tuple(a for a in AXES if not CONSUMERS.get(a))
    assert len(live) + len(dead) == 7
    # С TASK-24 (направление C) мёртвых не осталось: у ширины внимания появился
    # читатель — распределитель окон. Если станет больше нуля, тест это покажет.
    assert dead == (), (
        f"осей без потребителя стало {len(dead)}: {dead}. Ось без читателя обещает "
        "поведение, которого нет")
    assert len(live) == 7


def test_neutral_mood_shifts_nothing() -> None:
    """Иначе нельзя отличить «настроение сдвинуло» от «оси считаются криво»."""
    assert Motivation(MILESTONE_0).modulation().shifted() == ()


# --- по одной оси: сдвиг числа у самого потребителя -------------------------


def test_horizon_axis_shortens_the_plan_the_planner_will_accept() -> None:
    from harness.behaviour.planner import Planner
    from harness.model.forward import ForwardModel

    planner = Planner(MILESTONE_0, ForwardModel())
    base_len = planner.max_length
    planner.modulate(_mood(Motivation(MILESTONE_0), 0.0, 0.9))
    assert planner.max_length < base_len, (
        "возбуждение обязано сокращать допустимую длину плана: горизонт сжат, "
        "а план той же длины означал бы, что ось никто не прочитал")


def test_caution_axis_reaches_the_planner_and_the_babbler() -> None:
    from harness.behaviour.babbling import Babbler
    from harness.behaviour.planner import Planner
    from harness.model.forward import ForwardModel

    hot = _mood(Motivation(MILESTONE_0), -0.9, 0.0)
    planner = Planner(MILESTONE_0, ForwardModel())
    base = planner.caution_threshold
    planner.modulate(hot)
    assert planner.caution_threshold > base

    # У лепета порог — аргумент отбора проб, и он читается по имени.
    b = Babbler(MILESTONE_0, ("OUT_01", "OUT_02"), rng_seed=1)
    assert b.next_probe(caution_threshold=hot.caution_threshold) is not None


def test_explore_axis_changes_the_number_of_probes() -> None:
    """Страх тормозит исследование, скука ускоряет. Обе стороны — числом."""
    from harness.behaviour.babbling import Babbler

    def probes(rate: float, ticks: int = 100) -> int:
        b = Babbler(MILESTONE_0, ("OUT_01", "OUT_02"), rng_seed=1)
        return sum(b.pace(rate) for _ in range(ticks))

    fast = probes(2.0)
    normal = probes(1.0)
    slow = probes(0.25)
    assert fast > normal > slow, (
        f"темп не влияет на число проб: {fast}, {normal}, {slow}")
    assert normal == 100, "при темпе 1.0 проба на каждом такте — прежнее поведение"
    assert slow == pytest.approx(25, abs=2)

    # И то же через настроение, а не через число руками.
    m = Motivation(MILESTONE_0)
    bored = _mood(m, 0.0, -0.9)
    assert bored.explore_rate > bored.base["explore_rate"], (
        "скука обязана ускорять исследование, а не только упираться в базу")


def test_reflex_priority_axis_changes_when_the_lower_contour_may_cut_in() -> None:
    from harness.behaviour.contours import Scheduler

    loop = Scheduler(MILESTONE_0)
    assert loop.reflex_priority == 0.0, (
        "база нулевая: прежнее поведение — перебивать только просроченного")
    hot = _mood(Motivation(MILESTONE_0), 0.0, 1.0)
    assert hot.reflex_priority > 0.0, (
        "возбуждение обязано отдавать право хода рефлексам")


def test_task_share_axis_narrows_what_one_task_may_learn() -> None:
    from harness.core.resources import OP_BELIEF, ResourceGovernor

    g = ResourceGovernor(MILESTONE_0)
    assert g.task_cap() == g.belief_cap, "по умолчанию доля целая"
    hot = _mood(Motivation(MILESTONE_0), -1.0, 0.0)
    g.set_task_share(hot.task_resource_share)
    assert g.task_cap() < g.belief_cap

    # И это видно в решении, а не только в числе.
    g.beliefs = g.task_cap()
    verdict = g.admit(OP_BELIEF)
    assert not verdict and "доля задачи" in (verdict.reason or "")


def test_return_inertia_axis_changes_the_goal_budget() -> None:
    from harness.behaviour.goals import Candidate, GoalStack
    from harness.core.clocks import Stamp

    def budget(valence: float, arousal: float) -> int:
        m = Motivation(MILESTONE_0)
        m.mood = Mood(valence, arousal)
        stack = GoalStack(MILESTONE_0)
        cand = Candidate(kind="reach", target="PLACE_01", drive="curiosity",
                         gain=1.0, test=lambda: False, test_text="я в этом месте")
        goal = stack.push(cand, m, 0, "b0", Stamp(1, 1))
        return goal.budget_ticks

    calm, hot = budget(0.0, 0.0), budget(0.0, 1.0)
    assert hot < calm, (
        f"возбуждение обязано сокращать инерцию возврата: {hot} против {calm}")


def test_attention_axis_narrows_the_window_budget_of_its_consumer() -> None:
    """Ось дошла до потребителя: возбуждение сужает бюджет окон внимания.

    До TASK-24 здесь стояло обратное утверждение — «величина считается, читателя нет», — и
    оно было верным: нарезка окон существовала, а спрашивать окна было некому. Теперь
    читатель есть, и проверяется сдвиг числа у него самого (инвариант 25), а не наличие поля.
    """
    from harness.perception.attention import Attention

    hot = _mood(Motivation(MILESTONE_0), 0.0, 1.0)
    assert hot.attention_windows < hot.base["attention_windows"], (
        "величина считается верно — возбуждение сужает внимание")
    assert CONSUMERS["attention_windows"], "и у неё объявлен читатель"

    att = Attention.from_profile(from_schema("окна", attention_windows=4))
    assert att.modulate(hot) < att.base_windows


def test_switching_emotion_off_freezes_all_seven() -> None:
    off = from_schema("без эмоций", emotion_enabled=False)
    m = Motivation(off)
    m.mood = Mood(-0.9, 0.9)
    mod = m.modulation()
    assert mod.shifted() == (), (
        "выключенная эмоция обязана не двигать ни одной оси; ось, сдвинувшаяся при "
        "выключенном механизме, означала бы, что ручка не читается")
    for axis in AXES:
        assert mod.value(axis) == pytest.approx(mod.base[axis])


# --- расхождение «драйвы»: набор задан, а не выведен ------------------------


def test_drives_declare_that_they_were_given_not_discovered() -> None:
    """MIND.md требует выведенных драйвов. Пока их ноль, и это поле, а не мнение."""
    from harness.model.drives import DriveOrigin

    m = Motivation(MILESTONE_0)
    assert len(m.drives) == 6
    assert all(d.origin is DriveOrigin.GIVEN for d in m.drives.values())
    assert m.discovered == 0, (
        "если это число стало ненулевым без М6, значит драйв помечен выведенным "
        "без корреляционной машинерии — то есть заглушка выдаёт себя за механизм")
    # Число видно в отчёте, а не только в тесте.
    assert m.as_dict()["discovered"] == "0 из 6"


def test_a_discovered_drive_must_say_what_it_is_grounded_in() -> None:
    """Иначе выведенный драйв неотличим от заданного, и вся разница М6 исчезает."""
    from harness.model.drives import Drive, DriveOrigin

    with pytest.raises(ValueError, match="не сказано, из чего"):
        Drive("новый", 0.5, 0.5, origin=DriveOrigin.DISCOVERED)
    ok = Drive("новый", 0.5, 0.5, origin=DriveOrigin.DISCOVERED,
               grounded_in="SYM_1A2B")
    assert ok.grounded_in == "SYM_1A2B"


def test_given_drives_are_named_a_stand_in_in_one_place() -> None:
    """Заданный набор собирается одной функцией, чтобы М6 заменил ровно её."""
    from harness.model.drives import given_drives

    assert set(given_drives()) == set(Motivation(MILESTONE_0).drives)
