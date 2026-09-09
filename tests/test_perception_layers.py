"""Четыре слоя восприятия: свои частоты, свои ответы, измеренная цена.

Главный тест здесь опровергает удобное убеждение, которое до замера выглядело
очевидным. «Разные частоты экономят работу» — верно в обновлениях и почти неверно во
времени: 56 % пропущенных обновлений дают 4.9 % экономии времени, потому что 93 % цены
приходится на экранный слой, а он идёт каждый кадр. Единица «обновление» здесь просто
неверная.

| Слой | Мс на обновление | Частота | Доля общей цены |
|---|---|---|---|
| экранный | 25.07 | 30 Гц (каждый кадр) | 93.5 % |
| ближний | 0.54 | 20 Гц | 2.0 % |
| средний | 1.00 | 2 Гц | 3.7 % |
| дальний | 0.21 | 0.2 Гц | 0.8 % |
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from harness.core.action import Action
from harness.core.profile import from_schema
from harness.core.symbols import SymbolError
from harness.corpus.world import Effect, InteractiveWorld
from harness.perception.layers import (FAR, MID, NEAR, ORDER, SCREEN, FarLayer,
                                       LayerAnswer, MidLayer, NearLayer,
                                       PerceptionLayer, PerceptionStack, ScreenLayer)


def _profile(**kw):
    base = dict(capture_width=320, capture_height=180)
    base.update(kw)
    return from_schema("ТЕСТ-слои", **base)


def _frames(profile, n: int = 90, seed: int = 5) -> list[np.ndarray]:
    world = InteractiveWorld(profile, seed=seed, n_outputs=16)
    fwd = [o for o, e in world._effects.items() if e is Effect.FORWARD][0]
    return [world.step(Action.key(fwd, 200) if i % 3 else None,
                       with_audio=False).frame for i in range(n)]


# --- состав и частоты -------------------------------------------------------


def test_stack_has_four_layers_from_close_to_far() -> None:
    stack = PerceptionStack.from_profile(_profile())
    assert list(stack.layers) == list(ORDER) == [SCREEN, NEAR, MID, FAR]
    levels = [stack.layers[n].level for n in ORDER]
    assert levels == sorted(levels), (
        f"уровни не по возрастанию дальности: {levels}. Порядок субсумпции — чем "
        "ближе, тем срочнее")


def test_frequencies_come_from_the_profile() -> None:
    stack = PerceptionStack.from_profile(
        _profile(layer_screen_hz=10.0, layer_near_hz=5.0, layer_mid_hz=1.0,
                 layer_far_hz=0.5))
    assert [stack.layers[n].hz for n in ORDER] == [10.0, 5.0, 1.0, 0.5]


def test_rare_layer_does_not_answer_until_it_looked() -> None:
    """Дальний слой молчит, пока не посмотрел, и это не ноль.

    На 90 кадрах при 0.2 Гц и 30 fps его срок не приходит ни разу. Ответ `None`
    отличим от «уровень 0, ориентиров нет», и путать их нельзя.
    """
    profile = _profile()
    stack = PerceptionStack.from_profile(profile)
    for frame in _frames(profile, n=90):
        stack.feed(frame)
    assert stack.answer(FAR) is None
    assert stack.layers[FAR].age(stack.frames) is None
    assert stack.answer(SCREEN) is not None and stack.answer(MID) is not None


def test_answer_carries_its_age() -> None:
    """Ответ редкого слоя бывает старым, и решение по нему без давности — по устаревшему."""
    profile = _profile(layer_mid_hz=1.0)
    stack = PerceptionStack.from_profile(profile)
    frames = _frames(profile, n=60)
    for frame in frames:
        stack.feed(frame)
    assert stack.layers[MID].updates < stack.frames, "средний слой шёл каждый кадр"
    assert stack.layers[MID].age(stack.frames) is not None
    assert stack.layers[SCREEN].age(stack.frames) == 0
    # Ещё один кадр — и ответ среднего слоя заведомо устарел хотя бы на кадр,
    # потому что его срок так часто не приходит.
    stale_before = stack.layers[MID].updates
    stack.feed(frames[-1])
    if stack.layers[MID].updates == stale_before:
        assert stack.layers[MID].age(stack.frames) >= 1


def test_updates_are_skipped_exactly_as_the_frequencies_say() -> None:
    """Частотный контракт — при **выключенном** каскаде, и это не обход теста.

    Каскад запирает ступени на неизменившемся кадре, поэтому при нём число
    обновлений отвечает уже на другой вопрос: «сколько раз слою было пора **и**
    было на что смотреть». Здесь проверяется первая половина, поэтому вторая
    выключена явно — `cascade_enabled=False` объявлен в схеме именно как
    контрольный прогон. Что даёт включённый каскад на тех же кадрах, проверяет
    `test_cascade.py::test_the_gate_shifts_the_number_of_updates`.
    """
    profile = _profile(cascade_enabled=False)
    stack = PerceptionStack.from_profile(profile)
    for frame in _frames(profile, n=120):
        stack.feed(frame)
    updates = {n: stack.layers[n].updates for n in ORDER}
    assert updates[SCREEN] == 120, updates          # 30 Гц при 30 fps — каждый кадр
    assert 75 <= updates[NEAR] <= 85, updates       # 20 Гц при 30 fps — каждый 1.5-й
    assert 6 <= updates[MID] <= 10, updates         # 2 Гц — каждый пятнадцатый
    assert updates[FAR] == 0, updates               # 0.2 Гц — каждый 150-й
    assert stack.clock.savings() > 0.5


# --- главный замер: обновления и время — разные единицы ---------------------


def test_saved_updates_are_not_saved_time() -> None:
    """56 % пропущенных обновлений дают около 5 % экономии времени.

    Проверяется не точное число, а вывод: экономия обновлений сильно больше экономии
    времени, потому что цена сосредоточена в самом частом слое. Если это перестанет
    быть правдой — значит экранный слой подешевел или поредел, и таблицу цен в
    описании модуля надо перезамерить.
    """
    profile = _profile()
    frames = _frames(profile, n=40)
    per: dict[str, float] = {}
    for name in ORDER:
        layer = PerceptionStack.from_profile(profile).layers[name]
        t0 = time.perf_counter()
        for i, frame in enumerate(frames, 1):
            layer.update(frame, i)
        per[name] = (time.perf_counter() - t0) / len(frames)

    fps = float(profile.parameters["capture_fps"])
    stack = PerceptionStack.from_profile(profile)
    full = sum(per.values()) * len(frames)
    own = sum(per[n] * len(frames) / max(1.0, fps / stack.layers[n].hz) for n in ORDER)
    saved_time = 1.0 - own / full

    for frame in frames:
        stack.feed(frame)
    saved_updates = stack.clock.savings()

    assert saved_updates > 0.4, saved_updates
    assert saved_time < 0.25, (
        f"экономия времени {saved_time:.1%} при экономии обновлений "
        f"{saved_updates:.1%}. Если время правда стало экономиться — перезамерьте "
        "таблицу цен в описании harness.perception.layers")
    assert per[SCREEN] > 3 * max(per[NEAR], per[MID], per[FAR]), (
        f"экранный слой перестал быть самым дорогим: {per}. Тогда весь вывод про "
        "единицу измерения надо переписать")


# --- слои как контуры -------------------------------------------------------


def test_layers_are_served_closest_first() -> None:
    """Нижний слой обслуживается раньше верхнего. Это и есть субсумпция.

    Внутри слоя работа пока не нарезается: одно обновление атомарно, и перебить его
    нельзя. Это ограничение названо в описании модуля, а не спрятано: срез сейчас
    задаёт порядок между слоями, а не внутри одного.
    """
    from harness.behaviour.contours import Scheduler

    profile = _profile()
    frames = _frames(profile, n=8)
    stack = PerceptionStack.from_profile(profile)
    box = {"i": 0}

    def next_frame():
        box["i"] = min(box["i"] + 1, len(frames) - 1)
        return frames[box["i"]]

    now = [0.0]
    sched = Scheduler(profile, now=lambda: now[0])
    for name, start in stack.contours(next_frame).items():
        sched.add(name, stack.layers[name].hz, start, level=stack.layers[name].level)

    now[0] = 100.0                     # всем сразу пришёл срок
    ran = [r.contour for r in sched.step()]
    assert ran, "ни один слой не обслужен"
    order = [n for n in ORDER if n in ran]
    assert ran == order, f"порядок обслуживания не от близкого к дальнему: {ran}"


def test_stack_summary_carries_no_readable_text() -> None:
    """Сводка слоёв уходит в журнал, значит проходит ту же проверку, что восприятие."""
    profile = _profile()
    stack = PerceptionStack.from_profile(profile)
    for frame in _frames(profile, n=40):
        stack.feed(frame)
    payload = stack.as_perception()
    assert payload["frames"] == 40
    assert payload["layers"][FAR] is None

    # Подсунуть надпись — и проверка обязана упасть
    stack.layers[MID].update(np.zeros((180, 320), dtype=np.uint8), 41)
    stack.layers[MID].answer().payload["place"] = "Каменистый берег"
    with pytest.raises(SymbolError):
        stack.as_perception()


def test_base_layer_refuses_to_pretend_it_sees() -> None:
    """Слой без реализации падает, а не возвращает пустой ответ."""
    layer = PerceptionLayer(_profile(), 1.0, level=0)
    with pytest.raises(NotImplementedError, match="не умеет смотреть"):
        layer.update(np.zeros((8, 8), dtype=np.uint8), 1)


def test_every_layer_answers_with_its_own_measured_quantity() -> None:
    """У каждого слоя своя величина, и ни одна не выдумана.

    Экранный — доля пикселей, признанных интерфейсом, и признак, которым это решено.
    Ближний — тау в кадрах. Средний — место и размер графа. Дальний — свет и грубый
    ориентир. Если поле исчезнет, слой начнёт молчать о том, зачем он есть.
    """
    profile = _profile()
    frame = _frames(profile, n=3)[-1]
    checks = {
        ScreenLayer: ("screen_fraction", "signal", "disagreement_fraction"),
        NearLayer: ("size", "growth", "tau_steps", "n"),
        MidLayer: ("place", "places", "leaving", "lost"),
        FarLayer: ("level", "contrast", "landmark"),
    }
    for cls, fields in checks.items():
        layer = cls(profile, 1.0)
        answer = layer.update(frame, 1)
        assert isinstance(answer, LayerAnswer)
        for f in fields:
            assert f in answer.payload, (cls.__name__, f, answer.payload)
