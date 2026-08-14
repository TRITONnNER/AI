"""Цель с внешним судьёй и модель себя в чужих глазах. TASK-32, направление D.

Главное здесь — защита инварианта 10. Тест такой цели есть **предсказание** чужой оценки,
значит агент мог бы «выполнять» цели, предсказывая одобрение, и никакое наблюдение его бы не
поправило. Проверяется, что этого пути в коде нет:

1. Предсказание не переводит цель в выполненную ни при какой уверенности.
2. Пока оценки нет, ответ — «не определено», а не «нет».
3. Оценка не того судьи или не о том предмете отвергается.
4. Закрытие без предсказания отвергается: калибровка считается по промаху, а промаха без
   предсказания не бывает.
5. Незнание не выдаётся за половину: `predict` даёт `None`, а не 0.5.

Единица независимости — **утверждение об устройстве механизма**; у калибровки — **оценка**.
Окружения не требует.
"""

from __future__ import annotations

import pytest

from harness.model.reflected import (Judgement, JudgedError, ReflectedSelf,
                                     judged_goal, pending)


def _goal(want: float = 0.7):
    return judged_goal(goal_id="g-judged", judge="SYM_JUDGE_1", aspect="SYM_ASPECT_A",
                       want=want, budget_ticks=100, branch="b0", seq=1)


# --- предсказание не закрывает цель ------------------------------------------


def test_prediction_never_marks_the_goal_passed() -> None:
    """Инвариант 10: слова о себе не становятся результатом."""
    r = ReflectedSelf()
    g = _goal()
    # Учим модель ожидать высокой оценки, и предсказываем максимум.
    for _ in range(5):
        r.expect("SYM_JUDGE_1", "SYM_ASPECT_A").observe(1.0, 1.0)
    assert g.predict(r) == pytest.approx(1.0)
    assert g.passed is None, "предсказание не бывает результатом"
    assert not g.settled
    # И способа закрыть цель предсказанием в объекте нет.
    assert not [n for n in dir(g) if "claim" in n or "assert_passed" in n]


def test_unsettled_is_not_a_failure() -> None:
    """«Судья ещё не ответил» и «судья отказал» — разные исходы."""
    g = _goal()
    assert g.passed is None
    assert pending([g]) == [g]

    r = ReflectedSelf()
    g.predict(r)
    g.predicted = 0.9
    g.settle(Judgement("SYM_JUDGE_1", "SYM_ASPECT_A", 0.2), r)
    assert g.passed is False, "теперь это отказ, и он отличается от ожидания"
    assert pending([g]) == []


def test_only_the_real_verdict_closes_the_goal() -> None:
    """Закрывает цель оценка, и только она."""
    r = ReflectedSelf()
    g = _goal(want=0.6)
    g.predicted = 0.5
    got = g.settle(Judgement("SYM_JUDGE_1", "SYM_ASPECT_A", 0.8), r)
    assert g.settled and g.passed is True
    assert got.n == 1 and got.calibration == pytest.approx(0.3)


# --- чужая оценка не подставляется -------------------------------------------


def test_a_verdict_from_another_judge_is_refused() -> None:
    """Иначе модель училась бы на данных, которых не предсказывала."""
    r = ReflectedSelf()
    g = _goal()
    g.predicted = 0.5
    with pytest.raises(JudgedError, match="не относится к цели"):
        g.settle(Judgement("SYM_JUDGE_2", "SYM_ASPECT_A", 0.9), r)
    with pytest.raises(JudgedError, match="не относится к цели"):
        g.settle(Judgement("SYM_JUDGE_1", "SYM_ASPECT_B", 0.9), r)


def test_settling_without_a_prediction_teaches_mu_and_no_miss() -> None:
    """Промах — разница предсказанного и полученного; без первого его нет.

    **Но цель всё равно закрывается**, и это исправление TASK-33 C. Прежняя редакция
    отвергала такую оценку целиком, и получался тупик: предсказание берётся из `mu`, `mu`
    растёт из оценок, а оценки не принимались без предсказания. Замер канала отметок дал
    ноль оценок из 2400 — механизм не мог начаться. Закрывает цель **ответ судьи**, а не
    согласие агента с ним; промах при этом не записывается, потому что его не существует.
    """
    r = ReflectedSelf()
    g = _goal()
    g.settle(Judgement("SYM_JUDGE_1", "SYM_ASPECT_A", 0.9), r)
    assert g.passed is True
    kept = r.expect("SYM_JUDGE_1", "SYM_ASPECT_A")
    assert kept.n == 1 and kept.misses == []
    assert r.calibration()["judged"] == 0, "промахов нет — значит калибровки ещё нет"
    assert r.calibration()["without_prediction"] == 1


# --- незнание не выдаётся за знание ------------------------------------------


def test_unknown_judge_gives_none_not_half() -> None:
    """«Жду половину» и «не знаю, чего ждать» ведут к разным решениям."""
    r = ReflectedSelf()
    assert r.predict("SYM_JUDGE_9", "SYM_ASPECT_Z") is None
    g = _goal()
    assert g.predict(r) is None


def test_calibration_says_whether_the_model_beats_nothing() -> None:
    """Модель, ошибающаяся на 0.5 из 1.0, хуже отсутствия модели — и это видно числом."""
    good, bad = ReflectedSelf(), ReflectedSelf()
    for actual in (0.8, 0.9, 0.7, 0.85):
        good.observe(Judgement("SYM_J", "SYM_A", actual), predicted=actual - 0.05)
        bad.observe(Judgement("SYM_J", "SYM_A", actual), predicted=1.0 - actual)
    g, b = good.calibration(), bad.calibration()
    assert g["unit"] == "оценка" and g["judged"] == 4
    assert g["miss_mean"] == pytest.approx(0.05)
    assert g["better_than_nothing"] is True
    assert b["miss_mean"] > g["miss_mean"]
    assert b["better_than_nothing"] is False, (
        f"промах {b['miss_mean']:.2f} — это ответ наугад, и модель обязана это признать")


def test_calibration_is_none_before_any_verdict() -> None:
    """Ни одной оценки — ни одного числа. Ноль здесь читался бы как точность."""
    empty = ReflectedSelf().calibration()
    assert empty["judged"] == 0
    assert empty["miss_mean"] is None and empty["better_than_nothing"] is None


# --- вид цели виден, и он не наблюдение --------------------------------------


def test_the_goal_declares_that_it_is_not_checked_by_observation() -> None:
    """Читатель журнала не должен искать наблюдение, которого не будет."""
    g = _goal()
    assert g.goal.kind == "judged"
    assert "наблюдением не проверяется" in g.goal.test_text
    assert g.goal.test() is False, (
        "наблюдательный тест такой цели всегда ложен: закрывает её только оценка")
    assert g.as_dict()["test_is"].startswith("предсказание")


def test_a_judged_goal_cannot_be_disguised_as_an_observed_one() -> None:
    """Вид цели обязан отличаться: по нему видно, что тест — предсказание."""
    from harness.behaviour.goals import Goal
    from harness.model.beliefs import Origin, Provenance
    from harness.model.reflected import JudgedGoal

    ordinary = Goal(id="g", kind="reach_place", target="PLACE_1", test=lambda: True,
                    test_text="я в этом месте", budget_ticks=10,
                    provenance=Provenance(Origin.EXPERIENCE, "b0", 1),
                    drive="curiosity", pressure=0.5)
    with pytest.raises(JudgedError, match="вид"):
        JudgedGoal(goal=ordinary, judge="SYM_J", aspect="SYM_A", want=0.5)


def test_the_regime_calls_such_a_goal_inapplicable_here() -> None:
    """Канала оценки нет ни у одного нынешнего домена, и это сказано, а не обойдено."""
    from harness.perception.regime import JUDGE, Evidence, Regime, applicable

    r = Regime()
    r.add(Evidence(JUDGE, None, 0, "канал оценки", "канала нет вовсе"))
    verdict = applicable("цель с внешним судьёй", r)
    assert verdict.applicable is None, (
        "«не определено», а не «нет»: механизм готов, а мерить его пока нечем")
