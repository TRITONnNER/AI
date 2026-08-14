"""Канал оценки от оператора. TASK-33, направление C.

Числа с единицами живут в `tools/measure_judge.py` и `docs/measurements/judge.json`.
Здесь — утверждения об устройстве, и каждое соответствует одной границе, которую нельзя
стереть:

1. Отметка **не ставит цель** и не порождает действие: путь «сказал → сделал» отсутствует
   в коде, а не запрещён соглашением (инвариант 21).
2. Отметку **не может поставить агент**: источника «агент» не существует, и `behaviour/`
   этот модуль не импортирует (тот же способ, которым охраняется отладочный канал).
3. Цель закрывает **ответ судьи**, а не согласие агента с ним (инвариант 10). В том числе
   когда предсказания не было: тогда промах не записывается, а цель закрывается.
4. Неотвеченный запрос — «судья не ответил», а не «судья отказал».
5. Доверие весит сдвиг ожидания и **не** весит промах: промах — факт о предсказании
   агента, и недоверие к судье его не уменьшает.
6. «Лучше наугад» имеет три исхода: у проверки нет запаса ровно на границе 0.5.

Единица независимости — **утверждение об устройстве**. Окружения не требует.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core.profile import from_schema
from harness.model.marks import Mark, MarkQueue, Source, value_of
from harness.model.reflected import (Judgement, JudgedError, ReflectedSelf,
                                     judged_goal)

ROOT = Path(__file__).resolve().parent.parent


def _prof(**kw):
    base = dict(capture_width=64, capture_height=48)
    base.update(kw)
    return from_schema("ОТМЕТКА", **base)


def _goal(goal_id: str = "g1", *, want: float = 0.6):
    return judged_goal(goal_id=goal_id, judge="SYM_J", aspect="SYM_A", want=want,
                       budget_ticks=50, branch="отметка", seq=1)


# --- 1 и 2: отметка не команда и не самоотчёт ---------------------------------


def test_the_mark_module_creates_no_action_and_no_goal() -> None:
    """Путь «оператор сказал → агент сделал» отсутствует, а не запрещён соглашением."""
    text = (ROOT / "src" / "harness" / "model" / "marks.py").read_text(encoding="utf-8")
    for forbidden in ("Action.key", "Action.button", "Action.mouse", "def act",
                      "judged_goal(", "Goal("):
        assert forbidden not in text, (
            f"в канале отметок есть {forbidden!r}: отметка обязана отвечать на вопрос "
            "«как вышло», а не ставить цель и не порождать действие")


def test_the_agent_is_not_a_possible_source() -> None:
    """Источника «агент» не существует: иначе самоотчёт стал бы оценкой судьи."""
    assert {str(s) for s in Source} == {"оператор", "домен", "другой экземпляр"}
    with pytest.raises(ValueError):
        Source("агент")


def test_behaviour_does_not_import_the_mark_channel() -> None:
    """Тот же сторож, что у отладочного канала: у слоёв поведения доступа нет.

    Слой поведения, дотянувшийся до канала оценки, смог бы поставить отметку сам, и
    инвариант 10 сломался бы не в замысле, а в одной строке импорта.
    """
    behaviour = ROOT / "src" / "harness" / "behaviour"
    for path in sorted(behaviour.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "model.marks" not in text and "from .marks" not in text, (
            f"{path.name} импортирует канал оценки")


# --- 3: цель закрывает ответ судьи --------------------------------------------


def test_a_verdict_without_a_prediction_still_closes_the_goal() -> None:
    """Первая встреча с судьёй: промаха нет, а цель закрывается.

    Обратное правило было первой редакцией и оказалось тупиком: предсказание берётся из
    `mu`, `mu` растёт из оценок, а оценки не принимались без предсказания. Замер канала дал
    ноль оценок из 2400 — механизм не мог начаться.
    """
    mind, goal = ReflectedSelf(), _goal()
    assert goal.predict(mind) is None, "судью ещё не видели: предсказывать нечем"
    goal.settle(Judgement("SYM_J", "SYM_A", 1.0), mind)
    assert goal.passed is True
    r = mind.expect("SYM_J", "SYM_A")
    assert r.n == 1 and r.misses == [], "промах без предсказания не существует"
    assert r.mu > 0.5, "ожидание обязано сдвинуться: ответ судьи — данные о мире"
    cal = mind.calibration()
    assert cal["answers"] == 1 and cal["judged"] == 0
    assert cal["without_prediction"] == 1, "разницу надо видеть числом, а не выводить"


def test_prediction_alone_never_closes_a_goal() -> None:
    """Сторож инварианта 10: уверенность агента не бывает выполнением."""
    mind = ReflectedSelf()
    mind.observe(Judgement("SYM_J", "SYM_A", 1.0), predicted=None)
    goal = _goal()
    assert goal.predict(mind) == pytest.approx(1.0)
    assert goal.passed is None, "предсказание не закрывает цель ни при какой уверенности"
    assert not goal.settled
    assert not hasattr(goal, "mark_passed"), "метода, закрывающего цель предсказанием, нет"


# --- 4: очередь и «не ответил» ------------------------------------------------


def test_an_unanswered_request_is_not_a_refusal() -> None:
    q = MarkQueue.from_profile(_prof())
    mind, goal = ReflectedSelf(), _goal()
    q.ask(goal, mind, at=10)
    assert q.stats()["pending"] == 1
    assert goal.passed is None, "ждём судью — это не «нет»"
    assert q.settle([goal], mind) == [], "неотвеченный запрос ничего не закрывает"
    assert goal.passed is None


def test_the_queue_records_the_prediction_at_the_moment_of_asking() -> None:
    """Иначе предсказание можно подставить задним числом, зная оценку."""
    q = MarkQueue.from_profile(_prof())
    mind = ReflectedSelf()
    mind.observe(Judgement("SYM_J", "SYM_A", 0.0), predicted=None)   # mu уехало вниз
    goal = _goal()
    req = q.ask(goal, mind, at=1)
    asked = req.predicted
    # Модель успевает измениться до ответа оператора.
    for _ in range(5):
        mind.observe(Judgement("SYM_J", "SYM_A", 1.0), predicted=0.0)
    q.answer(goal.goal.id, Mark.DONE, at=4)
    row = q.settle([goal], mind)[0]
    assert row["predicted"] == asked, (
        "промах обязан считаться по тому предсказанию, о котором спрашивали")
    assert row["miss"] == pytest.approx(abs(asked - 1.0))
    assert row["waited"] == 3, "задержка ответа видна в записи"


def test_a_mark_nobody_asked_for_is_kept_apart() -> None:
    """Оператор мог отметить то, о чём агент не спрашивал: это данные, а не мусор."""
    q = MarkQueue.from_profile(_prof())
    q.answer("незнакомая", Mark.PARTLY)
    assert q.stats()["unmatched"] == 1 and q.stats()["pending"] == 0


# --- 5: доверие ---------------------------------------------------------------


def test_trust_weighs_the_shift_but_not_the_miss() -> None:
    """Иначе агент улучшал бы свою калибровку, объявляя неудобного судью недостоверным."""
    full, weak = ReflectedSelf(), ReflectedSelf()
    for mind, trust in ((full, 1.0), (weak, 0.25)):
        mind.observe(Judgement("SYM_J", "SYM_A", 1.0, trust=trust), predicted=0.5)
    a, b = full.expect("SYM_J", "SYM_A"), weak.expect("SYM_J", "SYM_A")
    assert a.mu > b.mu, "доверие обязано весить сдвиг ожидания"
    assert a.misses == b.misses == [0.5], "промах весить нельзя: это факт о предсказании"


def test_trust_lives_in_the_schema_and_differs_by_source() -> None:
    """Инвариант 23: доверие к источнику — не литерал в коде канала."""
    q = MarkQueue.from_profile(_prof())
    assert q.trust_of(Source.OPERATOR) > q.trust_of(Source.DOMAIN) > q.trust_of(Source.PEER)
    tuned = MarkQueue.from_profile(_prof(mark_trust_domain=0.1))
    assert tuned.trust_of(Source.DOMAIN) == pytest.approx(0.1)


def test_the_partial_mark_is_a_setting_and_the_ends_are_not() -> None:
    """«Частично» — порог, от которого зависят все промахи калибровки."""
    assert value_of(Mark.DONE, partial=0.5) == 1.0
    assert value_of(Mark.FAILED, partial=0.5) == 0.0
    assert value_of(Mark.PARTLY, partial=0.3) == pytest.approx(0.3)
    with pytest.raises(JudgedError, match="вне"):
        value_of(Mark.PARTLY, partial=1.5)
    q = MarkQueue.from_profile(_prof(mark_partial_value=0.25))
    assert q.partial == pytest.approx(0.25)


# --- 6: «лучше наугад» с тремя исходами ---------------------------------------


def test_better_than_nothing_has_three_outcomes() -> None:
    """У проверки нет запаса ровно на границе 0.5, и это сказано, а не скрыто.

    На замере судья-монета давал промах 0.494 при пределе 0.500, и прежняя проверка
    объявляла «лучше наугад» на трёх сидах из пяти — то есть победу шумом.
    """
    mind = ReflectedSelf()
    # Промахи ровно по 0.5: модель отвечает наугад, и это обязано быть «не отличимо».
    for _ in range(20):
        mind.observe(Judgement("SYM_J", "SYM_A", 1.0), predicted=0.5)
    cal = mind.calibration()
    assert cal["miss_mean"] == pytest.approx(0.5)
    assert cal["better_than_nothing"] is None, "ровно наугад — не «лучше» и не «хуже»"

    good = ReflectedSelf()
    for _ in range(20):
        good.observe(Judgement("SYM_J", "SYM_A", 1.0), predicted=0.95)
    assert good.calibration()["better_than_nothing"] is True
    assert good.calibration()["margin"] is not None, "запас обязан быть числом"
