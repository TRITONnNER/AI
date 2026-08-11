"""Отпечаток вида: разрешение, устойчивость и обе ошибки узнавания. TASK-29, A.

Числа с единицами живут в `tools/measure_fingerprint.py` и `docs/measurements/fingerprint.json`.
Здесь — утверждения об устройстве, каждое из которых уже один раз оказалось неверным:

1. Допуск по уровню — часть **сравнения**, размытие — часть **снятия** отпечатка; оба
   структурные, оба форкают журнал.
2. Точное равенство требует не «тот же вид», а «то же положение относительно границ
   квантования», и это разные требования.
3. Разрешение функции мерится в пикселях мира и обязано быть **крупнее шага мира** —
   иначе каждое действие уводит вид в новое место и возврат невозможен.
4. Шум (дизеринг, яркость, блочные артефакты) отпечаток не путает — вывод «различает шум»
   был неверен, и здесь стоит проверка, которая его бы поймала.
5. У узнавания две ошибки, и меряются обе (инвариант 32).

Единица независимости — **утверждение об устройстве механизма**. Окружения не требует.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.core.profile import from_schema
from harness.corpus.world import InteractiveWorld, WorldState
from harness.model.places import (PlaceGraph, fingerprint, similarity, view)


def _prof(**kw):
    base = dict(capture_width=160, capture_height=120)
    base.update(kw)
    return from_schema("ОТПЕЧАТОК", **base)


def _views(prof, *, shifts=(0.0,), seed: int = 1):
    """Кадры одного мира при заданных сдвигах камеры. Истина — из состояния мира."""
    world = InteractiveWorld(prof, seed=seed)
    x0, y0 = world.state.cam_x, world.state.cam_y
    out = []
    for dx in shifts:
        world.state = WorldState(cam_x=x0 + dx, cam_y=y0)
        out.append(world.step(None, with_audio=False).frame.astype(np.float64))
    return out


# --- допуск и размытие: где они живут ---------------------------------------


def test_tolerance_is_part_of_comparison_not_of_the_print() -> None:
    """Одна и та же пара отпечатков сходится или нет в зависимости от допуска."""
    a = (0, 1, 2, 3)
    b = (0, 2, 2, 4)
    assert similarity(a, b, tolerance=0) == 0.5
    assert similarity(a, b, tolerance=1) == 1.0


def test_blur_is_part_of_taking_the_print() -> None:
    """Размытие меняет сам отпечаток, а не сравнение."""
    frame = np.zeros((32, 32))
    frame[::2, :] = 200.0            # частая полоска: ровно то, что размытие убирает
    sharp = fingerprint(frame, blur_px=0)
    smooth = fingerprint(frame, blur_px=9)
    assert sharp != smooth
    # И обе величины дошли до графа из профиля, а не из константы модуля.
    g = PlaceGraph.from_profile(_prof(place_blur_px=5, place_level_tolerance=1))
    assert g.blur_px == 5 and g.level_tolerance == 1


def test_both_are_structural_and_fork_the_journal() -> None:
    """Меняют смысл каждого места разом — значит форкают журнал (инвариант 11)."""
    base = _prof()
    for key, value in (("place_blur_px", 9), ("place_level_tolerance", 0),
                       ("place_levels", 4)):
        other = _prof(**{key: value})
        assert key in base.structural, f"{key} обязана быть структурной"
        assert base.structure_hash != other.structure_hash, key


# --- почему точное равенство было заменено ----------------------------------


def test_exact_equality_breaks_on_a_quantisation_boundary() -> None:
    """Значение у границы корзины перескакивает уровень от любого шевеления.

    Это и есть причина замены функции: точное равенство требует не «тот же вид», а «то же
    положение относительно границ квантования». Проверяется на построенной паре, а не на
    мире: утверждение о механизме квантования, и мир для него — лишний посредник.
    """
    a = (1, 2, 3, 4)
    boundary_jitter = (2, 2, 3, 4)     # одна ячейка ушла на соседний уровень
    assert similarity(a, boundary_jitter, tolerance=0) == 0.75
    assert similarity(a, boundary_jitter, tolerance=1) == 1.0


def test_resolution_must_be_coarser_than_the_step_of_the_world() -> None:
    """Разрешение функции против шага мира. Числа — из замера, здесь знак.

    Шаг мира — 6 px за 100 мс удержания. При старой функции 6 px уже уводили вид за порог
    того же места (0.70 против 0.82), значит **каждое** действие рождало новое место. При
    новой 6 px остаются тем же местом, а удержание разведки (12 px) — уже другим.
    """
    thr = float(_prof().parameters["place_same_similarity"])
    here, step, hold = _views(_prof(), shifts=(0.0, 6.0, 12.0))

    def sim(levels: int, tol: int, other) -> float:
        return similarity(fingerprint(here, levels=levels),
                          fingerprint(other, levels=levels), tolerance=tol)

    assert sim(4, 0, step) < thr, "старая функция обязана терять место на шаге мира"
    assert sim(8, 1, step) >= thr, "новая обязана узнавать место после шага мира"
    assert sim(8, 1, hold) < thr, (
        "и обязана различать места после удержания разведки — иначе карта склеится в одну "
        "точку, и планировать будет нечего")


# --- шум: вывод «различает шум» был неверен ---------------------------------


@pytest.mark.parametrize("kind", ["дизеринг", "яркость", "блоки"])
def test_noise_does_not_move_the_place(kind: str) -> None:
    """Дизеринг, смена яркости и блочные артефакты не делают из вида новое место."""
    prof = _prof()
    thr = float(prof.parameters["place_same_similarity"])
    levels = int(prof.structural["place_levels"])
    tol = int(prof.structural["place_level_tolerance"])
    (frame,) = _views(prof)
    rng = np.random.default_rng(1)
    if kind == "дизеринг":
        spoiled = frame + rng.integers(-4, 5, frame.shape)
    elif kind == "яркость":
        spoiled = frame + 20.0
    else:
        spoiled = frame.copy()
        for i in range(0, frame.shape[0], 8):
            for j in range(0, frame.shape[1], 8):
                spoiled[i:i + 8, j:j + 8] = frame[i:i + 8, j:j + 8].mean()
    got = similarity(fingerprint(frame, levels=levels),
                     fingerprint(spoiled, levels=levels), tolerance=tol)
    assert got >= thr, f"{kind} увёл вид в другое место: {got:.3f} против порога {thr}"


def test_recognition_reports_both_errors_against_known_truth() -> None:
    """Инвариант 32: и ложная тревога, и ложное подтверждение, обе против истины мира.

    Истина берётся из состояния мира: сдвиг ноль — заведомо то же место, сдвиг больше
    кадра — заведомо другое (общих пикселей не остаётся вовсе).
    """
    prof = _prof()
    thr = float(prof.parameters["place_same_similarity"])
    levels = int(prof.structural["place_levels"])
    tol = int(prof.structural["place_level_tolerance"])
    here, same, far = _views(prof, shifts=(0.0, 0.0, 200.0))

    def sim(other) -> float:
        return similarity(fingerprint(here, levels=levels),
                          fingerprint(other, levels=levels), tolerance=tol)

    assert sim(same) >= thr, "ложная тревога: мир не двигался, а место объявлено новым"
    assert sim(far) < thr, (
        "ложное подтверждение: мир уехал дальше кадра, а место объявлено тем же. Это "
        "опаснее ложной тревоги — карта скажет «я тут был», и следа не останется")


# --- граф: узнавание идёт функцией графа, а не глобальной --------------------


def test_the_graph_recognises_with_its_own_function() -> None:
    """Два графа с разными функциями на одних кадрах дают разное число мест.

    Иначе настройка объявлена, а узнавание идёт мимо неё — тот самый дефект, из-за
    которого `place_blur_px` пришлось протаскивать и в дальний слой, и в benchmark.
    """
    frames = _views(_prof(), shifts=tuple(float(i) for i in range(0, 60, 6)))
    counts = []
    for levels, tol in ((4, 0), (8, 1)):
        prof = _prof(place_levels=levels, place_level_tolerance=tol)
        g = PlaceGraph.from_profile(prof)
        for i, f in enumerate(frames):
            g.see(f, i, seconds_per_seq=1 / 30.0, mode="OUT_01@200")
        counts.append(len(g.places))
    assert counts[0] > counts[1], (
        f"мест при точном равенстве {counts[0]}, при полосе ±1 {counts[1]} — полоса "
        "обязана склеивать больше, иначе допуск не читается графом")


def test_far_layer_takes_the_print_the_same_way_as_the_place_graph() -> None:
    """Ориентир дальнего слоя и отпечаток места считаются одной функцией."""
    from harness.perception.layers import FarLayer

    prof = _prof(place_blur_px=9)
    layer = FarLayer(prof, hz=0.2)
    assert layer.blur_px == 9, (
        "дальний слой берёт размытие из профиля: разойдись он с графом мест, «тот же "
        "вид» означало бы у них разное")
    (frame,) = _views(prof)
    got = layer.look(frame, at_frame=0)
    expected = view(frame, grid=layer.grid, levels=layer.levels, blur_px=9)
    assert got["landmark"] == list(expected.cells)
