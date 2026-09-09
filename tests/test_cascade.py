"""Каскад восприятия: гейт, учёт, логарифмический буфер.

Здесь же обе ошибки гейта по инвариантам 31 и 32 — доля ложных тревог и доля
ложных подтверждений, — и объявлено, откуда взята истина.

**Истина берётся из построения пары, а не из другого детектора.** Заведомо
одинаковая пара — побитовая копия кадра: одинаковее не бывает, и всякое «изменилось»
на ней есть ложная тревога. Заведомо различная пара — тот же кадр с прямоугольником
объявленной площади, залитым сдвигом объявленной амплитуды: изменение внесено нами,
и всякое «не менялось» на ней есть ложное подтверждение.

Ложное подтверждение здесь опаснее ложной тревоги ровно по инварианту 32: тревога
стоит одного лишнего прохода ступеней, а пропущенное изменение агент не увидит
никогда — кадр не дойдёт выше нулевой ступени, и следа не останется.

Единица независимости — **пара кадров** (инвариант 22). Пары независимы по
построению: каждая берёт свой кадр из своего сида, а не соседний кадр одной записи,
где второй похож на первый потому, что мир не успел измениться.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.capture.base import UNCHANGED
from harness.core.action import Action
from harness.core.profile import from_schema
from harness.corpus.world import Effect, InteractiveWorld
from harness.perception.cascade import (CHANGE, FINGERPRINT, FLOW, MODEL, STAGES,
                                        Cascade, ChangeDetector, LogBuffer)
from harness.perception.describers import AskGate
from harness.perception.layers import ORDER, PerceptionStack


def _profile(**kw):
    base = dict(capture_width=320, capture_height=180)
    base.update(kw)
    return from_schema("ТЕСТ-каскад", **base)


def _frames(profile, n: int = 120, seed: int = 5) -> list[np.ndarray]:
    world = InteractiveWorld(profile, seed=seed, n_outputs=16)
    fwd = [o for o, e in world._effects.items() if e is Effect.FORWARD][0]
    return [world.step(Action.key(fwd, 200) if i % 3 else None,
                       with_audio=False).frame for i in range(n)]


def _cascade(**kw) -> Cascade:
    return Cascade.from_profile(_profile(**kw))


# --- ступень 0: изменение ---------------------------------------------------


def test_the_first_frame_counts_as_changed() -> None:
    """Сравнивать не с чем, и «не менялся» было бы утверждением о невиденном."""
    d = ChangeDetector()
    changed, fraction = d.look(np.zeros((64, 64), np.uint8))
    assert changed and fraction == 1.0


def test_a_repeated_frame_is_not_a_change() -> None:
    d = ChangeDetector()
    frame = np.random.default_rng(1).integers(0, 255, (180, 320), dtype=np.uint8)
    d.look(frame)
    changed, fraction = d.look(frame.copy())
    assert not changed and fraction == 0.0


def test_a_changed_shape_is_the_largest_change_there_is() -> None:
    """Разноразмерные копии сравнивать нечем, и «то же самое» было бы ложью."""
    d = ChangeDetector()
    d.look(np.zeros((180, 320), np.uint8))
    changed, fraction = d.look(np.zeros((90, 160), np.uint8))
    assert changed and fraction == 1.0


def test_forgetting_makes_the_next_frame_a_change() -> None:
    """После разрыва в записи сравнение мерило бы длину пропуска, а не экран."""
    d = ChangeDetector()
    frame = np.zeros((64, 64), np.uint8)
    d.look(frame)
    assert not d.look(frame)[0]
    d.forget()
    assert d.look(frame)[0]


def test_the_source_mark_costs_no_diff_at_all() -> None:
    """`UNCHANGED` от источника — ответ, а не пропуск, и проверять его нечем.

    Отметка доходит до восприятия и запирает всё выше нулевой ступени. До этой
    задачи она доходила только до журнала.
    """
    c = _cascade()
    verdict = c.admit(UNCHANGED)
    assert verdict.top == CHANGE and not verdict.changed
    assert verdict.change_fraction is None and not verdict.measured
    assert c.free_unchanged == 1 and c.measured_unchanged == 0
    assert not verdict.allows(FLOW)


def test_measured_and_free_stillness_are_counted_apart() -> None:
    """«Нам сказали» и «мы проверили» — разные основания, и экономию они дают разную."""
    c = _cascade()
    frame = np.zeros((180, 320), np.uint8)
    c.admit(frame)              # первый кадр — изменение по построению
    c.admit(frame.copy())       # проверено разностью
    c.admit(UNCHANGED)          # сказано источником
    assert c.measured_unchanged == 1 and c.free_unchanged == 1


# --- ошибка предсказания управляет расходом ---------------------------------


def test_a_predicted_frame_stops_at_the_flow_stage() -> None:
    """Совпало с предсказанием — ступени выше первой не запускаются."""
    c = _cascade(cascade_predicted_error=0.15)
    frames = _frames(_profile(), n=4)
    c.admit(frames[0], error=0.9)
    verdict = c.admit(frames[1], error=0.01)
    assert verdict.changed and verdict.surprised is False
    assert verdict.top == FLOW and c.quiet == 1


def test_an_unmeasured_error_does_not_lock_the_deep_stages() -> None:
    """`None` — «ещё не считали». Запереть на нехватке величины значило бы объявить
    мир предсказанным, ни разу его не предсказав."""
    c = _cascade()
    verdict = c.admit(_frames(_profile(), n=1)[0], error=None)
    assert verdict.surprised is None and verdict.top >= FINGERPRINT


def test_the_deep_stage_asks_through_the_existing_gate() -> None:
    """Порог ступени 3 не заводится заново: он уже есть в группе «Файрвол»."""
    gate = AskGate(0.35, min_gap_cycles=15)
    c = Cascade.from_profile(_profile(), ask_gate=gate)
    frames = _frames(_profile(), n=6)
    first = c.admit(frames[0], error=0.9, t_self=0)
    assert first.top == MODEL, "первый удививший кадр обязан дойти до модели"
    second = c.admit(frames[1], error=0.9, t_self=1)
    assert second.top == FINGERPRINT, "промежуток между запросами не соблюдён"
    assert gate.skipped_gap == 1


def test_a_disabled_cascade_lets_everything_through() -> None:
    """Контрольный прогон, а не режим «как было»: разность всё равно считается."""
    c = _cascade(cascade_enabled=False)
    frame = np.zeros((180, 320), np.uint8)
    c.admit(frame)
    verdict = c.admit(frame.copy())
    assert verdict.top == MODEL and not verdict.changed
    assert verdict.change_fraction == 0.0, "разность не посчитана — сравнивать не с чем"


# --- учёт стоимости ---------------------------------------------------------


def test_a_stage_never_called_costs_none_and_not_zero() -> None:
    c = _cascade()
    assert c.costs[MODEL].ms_per_call is None
    c.spend(MODEL, 12.0)
    assert c.costs[MODEL].ms_per_call == pytest.approx(12.0)


def test_blocked_stages_are_counted_as_blocked() -> None:
    c = _cascade()
    frame = np.zeros((180, 320), np.uint8)
    c.admit(frame)
    c.admit(frame.copy())
    assert c.costs[FLOW].blocked == 1 and c.costs[MODEL].blocked == 1
    assert c.costs[CHANGE].blocked == 0, "нулевая ступень запереть себя не может"


def test_the_ratio_answers_the_architectural_question() -> None:
    """Во сколько раз ступень 3 реже нулевой. Ожидание архитектуры — 100–300."""
    c = _cascade()
    assert c.ratio is None, "модель не звали — отношения нет, и это не ноль"
    frame = np.zeros((180, 320), np.uint8)
    c.admit(frame)                                  # дошло до модели
    for _ in range(199):
        c.admit(frame.copy())                       # заперто на нулевой
    assert c.costs[CHANGE].calls == 200
    assert c.ratio == pytest.approx(200.0)


def test_sigma_narrowing_is_attributed_to_the_deepest_stage() -> None:
    """Предельная величина, а не средняя: сколько купит проход на ступень глубже."""
    sigmas = iter([1.0, 0.6, 0.6, 0.6])
    c = Cascade.from_profile(_profile(), sigma_probe=lambda: next(sigmas))
    frame = _frames(_profile(), n=1)[0]
    verdict = c.admit(frame, error=0.9)
    c.spend(verdict.top, 4.0)
    c.settle()
    cost = c.costs[verdict.top]
    assert cost.sigma_frames == 1
    assert cost.sigma_narrowed == pytest.approx(0.4)
    assert cost.sigma_per_ms is not None and cost.sigma_per_ms > 0.0


def test_without_a_probe_the_narrowing_is_none_and_not_zero() -> None:
    """«Не измеряли» и «не сузилась» — разные утверждения."""
    c = _cascade()
    verdict = c.admit(_frames(_profile(), n=1)[0], error=0.9)
    c.spend(verdict.top, 4.0)
    c.settle()
    assert c.costs[verdict.top].sigma_per_ms is None


# --- логарифмический буфер --------------------------------------------------


def test_the_buffer_holds_decades_not_a_window() -> None:
    """Горизонт растёт, а память нет: три полосы по 30 покрывают сто секунд."""
    b = LogBuffer(capacity=30, tiers=3, decimation=10)
    for i in range(3000):
        b.put(i, i)
    held = sum(len(b.tier(i)) for i in range(3))
    assert held <= 90, f"буфер держит {held} кадров вместо девяноста"
    assert b.horizon() >= 2900, "сто секунд назад буфер не достаёт"
    assert b.span(0) == 29 and b.span(1) == 290


def test_the_buffer_answers_none_when_it_does_not_reach() -> None:
    """Честный отказ, а не ближайший имеющийся: подменять давность нельзя."""
    b = LogBuffer(capacity=5, tiers=1, decimation=10)
    for i in range(5):
        b.put(i, i)
    assert b.at_least(100) is None
    assert b.at_least(2) == (2, 2)


def test_the_buffer_keeps_the_far_past_the_recent_one_lost() -> None:
    """То, ради чего полосы: мелкая полоса кадр уже забыла, крупная ещё помнит."""
    b = LogBuffer(capacity=10, tiers=2, decimation=10)
    for i in range(200):
        b.put(i, i)
    assert all(at >= 190 for at, _ in b.tier(0)), "мелкая полоса держит старое"
    assert any(at <= 110 for at, _ in b.tier(1)), "крупная полоса не достаёт до старого"


# --- сдвиг числа (инвариант 25) ---------------------------------------------


def test_the_gate_shifts_the_number_of_updates() -> None:
    """Правка предъявляет сдвиг числа на одних и тех же кадрах.

    Единица здесь — **прогон**: кадры внутри одной записи зависимы, и доля,
    посчитанная по кадрам, была бы псевдорепликацией (инвариант 22). Сравниваются
    два прогона на **одном и том же** списке кадров, поэтому разница — свойство
    гейта, а не разных кадров.
    """
    profile = _profile()
    frames = _frames(profile, n=120)

    control = PerceptionStack.from_profile(_profile(cascade_enabled=False))
    for f in frames:
        control.feed(f)

    gated = PerceptionStack.from_profile(profile)
    for f in frames:
        gated.feed(f)

    before = {n: control.layers[n].updates for n in ORDER}
    after = {n: gated.layers[n].updates for n in ORDER}
    assert after["screen"] < before["screen"], (
        f"гейт ничего не запер: {before} → {after}. Либо в этих кадрах нет "
        "неподвижности, либо отметки снова не доходят до восприятия")
    assert sum(gated.blocked.values()) > 0
    assert gated.cascade.stats()["unchanged"]["measured"] > 0


def test_the_stack_does_not_look_at_a_source_mark() -> None:
    """Отметка `UNCHANGED` не кадр: смотреть нечего ни одному слою."""
    stack = PerceptionStack.from_profile(_profile())
    assert stack.feed(UNCHANGED) == []
    assert all(stack.layers[n].updates == 0 for n in ORDER)
    assert stack.blocked["screen"] == 1


# --- обе ошибки гейта (инварианты 31 и 32) ----------------------------------


def _pair(seed: int, *, area: float, amplitude: int) -> tuple[np.ndarray, np.ndarray]:
    """Пара кадров с **внесённым нами** изменением объявленной площади и амплитуды."""
    rng = np.random.default_rng(seed)
    before = rng.integers(0, 255, (180, 320), dtype=np.uint8)
    after = before.copy()
    if area > 0.0:
        side = max(1, int(round((area * before.size) ** 0.5)))
        top = int(rng.integers(0, max(1, 180 - side)))
        left = int(rng.integers(0, max(1, 320 - side)))
        patch = after[top:top + side, left:left + side].astype(np.int16) + amplitude
        after[top:top + side, left:left + side] = np.clip(patch, 0, 255).astype(np.uint8)
    return before, after


def _gate_says_changed(before: np.ndarray, after: np.ndarray) -> bool:
    d = ChangeDetector.from_profile(_profile())
    d.look(before)
    return d.look(after)[0]


def test_the_gate_declares_its_false_alarm_rate() -> None:
    """Доля ложных тревог на заведомо одинаковых парах. Истина — побитовая копия.

    n = 40 пар, единица независимости — пара. Ожидание — ноль: разность побитовой
    копии равна нулю тождественно, и всякая тревога здесь была бы дефектом самой
    разности, а не порога.
    """
    alarms = sum(_gate_says_changed(f, f.copy())
                 for f, _ in (_pair(s, area=0.0, amplitude=0) for s in range(40)))
    assert alarms == 0, f"ложных тревог {alarms} из 40 на побитовых копиях"


def test_the_gate_declares_its_false_confirmation_rate() -> None:
    """Доля ложных подтверждений на заведомо различных парах — та ошибка, что опаснее.

    Изменение внесено нами: 5 % площади кадра сдвинуты на 60 уровней, то есть
    заведомо выше обоих порогов (`cascade_change_pixel_delta` 8, доля 0.002).
    Пропуск такого изменения означает, что кадр не дойдёт выше нулевой ступени и
    агент его не увидит никогда.

    n = 40 пар, единица независимости — пара.
    """
    missed = sum(not _gate_says_changed(*_pair(s, area=0.05, amplitude=60))
                 for s in range(40))
    assert missed == 0, f"ложных подтверждений {missed} из 40 на внесённом изменении"


def test_the_gate_reports_where_it_stops_seeing() -> None:
    """Покрытие предъявляется отдельно от доли (инвариант 31): что гейт **не** ловит.

    Молчание проверки, видящей четверть системы, доказательством не является,
    поэтому здесь названа граница: изменение мельче порога доли гейт пропускает
    **по построению**, и это не дефект, а объявленная цена. Мигающий курсор —
    ровно такой случай.
    """
    tiny = sum(_gate_says_changed(*_pair(s, area=0.0005, amplitude=60))
               for s in range(40))
    assert tiny == 0, (
        f"гейт поймал {tiny} изменений из 40 мельче объявленного порога — значит "
        "порог доли не работает, и мигающий курсор будет будить весь каскад")
