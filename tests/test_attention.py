"""Внимание как бюджет дефицитного ресурса. TASK-24, направление C.

Проверяется не наличие полей, а то, из чего складывается утверждение «внимание —
распределение дефицита»:

1. Бюджет читается из профиля и **сужается настроением** — иначе ось `attention_windows`
   осталась бы мёртвой в том же смысле, в каком были мёртвы горизонт и осторожность.
2. Претензия снизу перебивает выданное сверху, а не наоборот.
3. У перебивания есть **цена**: прерванное наблюдение не даёт ничего и считается отдельно.
4. Источник, ничего не давший, сам уходит вниз очереди — без списка «уже смотрел».
5. Каждая ветка арбитража имеет счётчик, и ветка без срабатываний — дефект (инвариант 26).

Единица независимости здесь — **утверждение об устройстве механизма**, а не такт и не
прогон: повторение одного и того же такта тысячу раз не добавляет ни одного независимого
наблюдения о том, кто кого перебивает. Числа с единицей «прогон» живут в
`tools/measure_attention.py`.

Окружения не требует: ни дисплея, ни записей, ни сети.
"""

from __future__ import annotations

import pytest

from harness.core.profile import MILESTONE_0, from_schema
from harness.perception.attention import (FROM_ABOVE, FROM_BELOW, SALIENCE,
                                          Attention, AttentionError, Claim, Source)


def _two(windows: int = 1, *, arbitrate: bool = True) -> Attention:
    prof = from_schema("ВНИМАНИЕ", attention_windows=windows)
    return Attention.from_profile(
        prof, [Source("AREA_A", sigma=1.0), Source("AREA_B", sigma=1.0)],
        arbitrate=arbitrate)


# --- бюджет: откуда берётся и что его сужает --------------------------------


def test_budget_comes_from_the_profile_and_nowhere_else() -> None:
    """Захардкоженного числа окон быть не должно: это настройка поведения."""
    assert _two(3).windows == 3
    assert Attention.from_profile(MILESTONE_0).windows == int(
        MILESTONE_0.parameters["attention_windows"])


def test_arousal_narrows_attention_and_the_consumer_reads_it() -> None:
    """Ось модуляции доходит до распределителя: сдвиг числа, а не наличие поля.

    До этой правки `CONSUMERS["attention_windows"]` был `None` — величина считалась и не
    читалась никем. Проверяется именно чтение: сузилось настроение — сузился бюджет.
    """
    from harness.model.drives import CONSUMERS, Mood, Motivation

    assert CONSUMERS["attention_windows"], "у оси объявлен читатель"
    att = _two(4)
    assert att.base_windows == 4
    m = Motivation(MILESTONE_0)
    m.mood = Mood(0.0, 1.0)
    got = att.modulate(m.modulation())
    assert got < att.base_windows, (
        f"возбуждение обязано сужать бюджет окон: {got} против {att.base_windows}")
    assert att.windows == got >= 1, "ноль окон — не сужение, а отключение восприятия"


def test_emotion_switched_off_leaves_the_budget_alone() -> None:
    """Выключенная эмоция не двигает бюджет: иначе ручка не читается."""
    from harness.model.drives import Mood, Motivation

    off = from_schema("без эмоций", emotion_enabled=False, attention_windows=4)
    m = Motivation(off)
    m.mood = Mood(-0.9, 0.9)
    att = Attention.from_profile(off)
    assert att.modulate(m.modulation()) == 4


# --- права претензий --------------------------------------------------------


def test_salience_preempts_a_window_held_from_above() -> None:
    """Прерывание снизу не спрашивает разрешения. И перебивает **уже выданное**.

    Ветка недостижима, если занятость не переживает такт: перебивать в начале такта было бы
    нечего. Первая редакция сбрасывала выданные окна каждый такт и давала ноль перебиваний
    на всех бюджетах — дефект по инварианту 26, найденный замером, а не чтением.
    """
    att = _two(1)
    att.step([Claim("AREA_A", FROM_ABOVE, "цель: дойти")])
    assert [g.source for g in att.granted] == ["AREA_A"]

    got = att.step([Claim("AREA_B", FROM_BELOW, SALIENCE[0])])
    assert [g.source for g in got] == ["AREA_B"], "заметность обязана выселить цель"
    assert att.preemptions == 1
    assert got[0].preempted == "AREA_A"


def test_a_planner_claim_never_preempts_salience() -> None:
    """Обратной силы у релевантности нет: иначе право перебивать можно перекупить."""
    att = _two(1)
    att.step([Claim("AREA_A", FROM_BELOW, SALIENCE[1])])
    got = att.step([Claim("AREA_B", FROM_ABOVE, "цель: дойти")])
    assert [g.source for g in got] == ["AREA_A"]
    assert att.preemptions == 0
    assert att.refused == 1, "отказ считается, а не теряется"


def test_two_interrupts_do_not_cancel_each_other() -> None:
    """Окна снизу не выселяются: иначе ни одно прерывание не обслужится."""
    att = _two(1)
    att.step([Claim("AREA_A", FROM_BELOW, SALIENCE[0])])
    got = att.step([Claim("AREA_B", FROM_BELOW, SALIENCE[2])])
    assert [g.source for g in got] == ["AREA_A"]
    assert att.preemptions == 0


def test_preemption_costs_the_interrupted_observation() -> None:
    """Прерванное наблюдение не даёт ничего, и это отдельное число.

    Без него право перебивать выглядело бы бесплатным, а оно не бесплатно: цель, на которую
    смотрели три такта, теряет всё, что успела набрать.
    """
    att = _two(1)
    att.step([Claim("AREA_A", FROM_ABOVE, "цель: дойти")])
    att.step([Claim("AREA_B", FROM_BELOW, SALIENCE[0])])
    assert att.cut_short == 1
    victim = next(g for g in att.history if g.source == "AREA_A")
    assert victim.cut_short and victim.gain is None
    # И наблюдатель узнаёт о прерывании единственным доступным способом: окна больше нет.
    with pytest.raises(AttentionError, match="не занято"):
        att.report("AREA_A", sigma_after=0.1)


def test_the_window_stays_busy_until_the_observation_reports() -> None:
    """Окно освобождает отчёт, а не наступление следующего такта."""
    att = _two(1)
    att.step([Claim("AREA_A", FROM_ABOVE, "цель: дойти")])
    held = att.step([])
    assert [g.source for g in held] == ["AREA_A"]
    assert held[0].ticks_held == 1, "занятость обязана переживать такт"
    att.report("AREA_A", sigma_after=0.5)
    assert att.step([]) == []


def test_a_second_claim_on_a_watched_source_is_not_a_second_window() -> None:
    """Смотрим одно наблюдение: удвоенная претензия — потраченный бюджет, не двойная польза."""
    att = _two(2)
    att.step([Claim("AREA_A", FROM_ABOVE, "цель: дойти")])
    att.step([Claim("AREA_A", FROM_BELOW, SALIENCE[0]),
              Claim("AREA_B", FROM_BELOW, SALIENCE[0])])
    assert att.busy_skipped == 1
    assert sorted(g.source for g in att.granted) == ["AREA_A", "AREA_B"]


# --- арбитраж: польза на стоимость ------------------------------------------


def test_score_is_expected_gain_over_cost_and_urgency_only_from_below() -> None:
    """Одно число и оно объяснимо. Срочность — множитель только у заметности."""
    att = _two(1)
    att.sources["AREA_A"].cost_s = 0.02
    calm = att.score(Claim("AREA_A", FROM_BELOW, SALIENCE[0], urgency=1.0))
    hasty = att.score(Claim("AREA_A", FROM_BELOW, SALIENCE[0], urgency=2.0))
    assert hasty == pytest.approx(2 * calm)
    # 0.5 · 1.0 (оптимизм до первого взгляда) · вес 1.0 / стоимость 0.02
    assert calm == pytest.approx(25.0)
    above = att.score(Claim("AREA_A", FROM_ABOVE, "цель"))
    assert above == pytest.approx(calm), "срочность сверху не считается"


def test_a_quantity_that_does_not_affect_the_decision_gives_no_benefit() -> None:
    """Огромный разброс при нулевом весе в решении — не польза, а любопытство."""
    att = _two(1)
    att.sources["AREA_A"].weight = 0.0
    att.sources["AREA_A"].sigma = 100.0
    assert att.score(Claim("AREA_A", FROM_BELOW, SALIENCE[0])) == 0.0


def test_a_source_that_gave_nothing_sinks_without_a_seen_list() -> None:
    """Память о просмотренном не нужна: ожидаемая польза считается по наблюдённой."""
    att = _two(1)
    att.step([Claim("AREA_A", FROM_BELOW, SALIENCE[0])])
    att.report("AREA_A", sigma_after=1.0)          # ничего не дал
    att.step([Claim("AREA_B", FROM_BELOW, SALIENCE[0])])
    att.report("AREA_B", sigma_after=0.4)          # дал много
    a = att.score(Claim("AREA_A", FROM_BELOW, SALIENCE[0]))
    b = att.score(Claim("AREA_B", FROM_BELOW, SALIENCE[0]))
    assert a < b
    assert att.sources["AREA_A"].empty and not att.sources["AREA_B"].empty
    assert att.summary()["empty_sources"] == ["AREA_A"]


def test_control_mode_is_a_field_not_a_patched_method() -> None:
    """Контроль живёт в боевом коде: подменённый в замере отличался бы не только задуманным."""
    att = _two(1, arbitrate=False)
    att.sources["AREA_A"].sigma = 100.0
    assert att.score(Claim("AREA_A", FROM_BELOW, SALIENCE[0], urgency=9.0)) == 1.0
    assert att.summary()["arbitrate"] is False


# --- отчётность и счётчики --------------------------------------------------


def test_gain_comes_from_outside_and_a_report_needs_an_open_window() -> None:
    """Внимание не знает, что даёт наблюдение, и не придумывает пользу за наблюдателя."""
    att = _two(1)
    with pytest.raises(AttentionError, match="неизвестному источнику"):
        att.report("AREA_НЕТ", sigma_after=0.0)
    with pytest.raises(AttentionError, match="не занято"):
        att.report("AREA_A", sigma_after=0.0)
    att.step([Claim("AREA_A", FROM_BELOW, SALIENCE[0])])
    assert att.report("AREA_A", sigma_after=0.25) == pytest.approx(0.75)
    assert att.curve == [(1, pytest.approx(0.75))]


def test_windows_to_gain_is_the_number_that_separates_the_orderings() -> None:
    """Итог насыщенного прогона от порядка не зависит; расход окон до цели — зависит.

    Вырождение записано в `MEASUREMENT.md`, 13.6, и здесь проверяется, что неразложимый
    показатель считается и возвращает `None`, когда цель не взята.
    """
    att = _two(1)
    for _ in range(3):
        att.step([Claim("AREA_A", FROM_BELOW, SALIENCE[0])])
        att.report("AREA_A", sigma_after=max(0.0, att.sources["AREA_A"].sigma - 0.3))
    assert att.windows_to_gain(0.5) == 2
    assert att.windows_to_gain(10.0) is None


def test_every_arbitration_branch_has_a_counter() -> None:
    """Инвариант 26: ветка без счётчика не докладывается, а ветка с нулём — дефект.

    Здесь проверяется наличие счётчиков и то, что каждый способен сдвинуться. Что все они
    сдвигаются на полном прогоне — числа замера: перебиваний 3, 35 и 9 на бюджетах 1, 2, 4.
    """
    keys = ("granted", "refused", "preemptions", "cut_short", "busy_skipped",
            "requests", "from_below", "from_above")
    att = _two(1)
    att.step([Claim("AREA_A", FROM_ABOVE, "цель")])
    att.step([Claim("AREA_B", FROM_BELOW, SALIENCE[0]),
              Claim("AREA_B", FROM_BELOW, SALIENCE[1]),
              Claim("AREA_A", FROM_ABOVE, "цель")])
    got = att.summary()
    for key in keys:
        assert key in got, f"ветка {key} не докладывается числом"
    assert got["preemptions"] and got["refused"] and got["cut_short"]
    assert got["busy_skipped"], "повторная претензия на занятый источник считается"
    assert got["unit"] == "окно на такт"


def test_claim_from_below_must_name_a_listed_salience_feature() -> None:
    """Признак заметности — из перечисленных: новый это правка схемы, а не строка на месте."""
    with pytest.raises(AttentionError, match="признаки заметности"):
        Claim("AREA_A", FROM_BELOW, "мне показалось интересным")
    with pytest.raises(AttentionError, match="нет такого источника претензии"):
        Claim("AREA_A", "изнутри", "цель")
    # Сверху причина свободная: это имя цели, а не признак из закрытого набора.
    assert Claim("AREA_A", FROM_ABOVE, "цель: дойти до SYM_7A3F").reason
