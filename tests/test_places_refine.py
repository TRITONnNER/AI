"""Граф мест: «нажал и остался» — факт, и место делится, когда врёт само себе.

Оба механизма появились из замера, а не из чтения кода, и оба теста написаны так,
чтобы падать ровно при возврате прежнего поведения.

Первый: раньше ребро появлялось только при смене места, а «нажал и остался» молча
выбрасывалось. Замер: на сиде 7 разведка сделала 1500 шагов, открыла одно место и
нажала один и тот же выход 1500 раз, потому что модель перехода собирается из рёбер,
и про действие, которое никуда не ведёт, она не знала ничего.

Второй: приведение к среднему — это допущение «яркость не важна». На сиде 5
переключатель света давал одно и то же место при разном состоянии мира, и
подтверждённый на двенадцати наблюдениях переход врал при исполнении: доходили 2
плана из 36. Теперь место делится само, когда его исходы расходятся, — и это тот же
единый механизм: ошибка предсказания правит карту.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.core.profile import from_schema
from harness.model.places import (LEVELS_KEPT, PlaceGraph, Traversal,
                                  fingerprint, place_id, similarity, view)


def _views(n: int, seed: int = 7, shape: tuple[int, int] = (64, 96)) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, shape, dtype=np.uint8) for _ in range(n)]


def _dark(frame: np.ndarray, factor: int = 45) -> np.ndarray:
    return (frame.astype(np.uint16) * factor // 100).astype(np.uint8)


# --- «нажал и остался» ------------------------------------------------------


def test_staying_put_is_recorded_as_an_edge() -> None:
    """Действие, не сменившее место, тоже попадает в граф — как петля.

    Иначе про такое действие модель перехода не знает ничего: не «знает, что оно
    бесполезно», а не знает вообще, и разведка запирается на одном нажатии.
    """
    g = PlaceGraph()
    frame = _views(1)[0]
    for seq in range(1, 5):
        g.observe(fingerprint(frame), seq, seconds_per_seq=0.1, mode="OUT_1A@200")
    loops = g.loops()
    assert len(loops) == 1, "петля не записалась"
    assert loops[0].src == loops[0].dst and loops[0].n == 3
    assert g.stats()["loops"] == 1 and g.stats()["leaving"] == 0


def test_loop_time_is_one_step_not_time_spent_here() -> None:
    """У петли своё время: длительность шага, а не «сколько я тут сижу».

    Накопленное время означало бы другой вопрос, и маршрут по таким рёбрам считался
    бы по числам, которых никто не измерял.
    """
    g = PlaceGraph()
    frame = _views(1)[0]
    for seq in range(1, 11):
        g.observe(fingerprint(frame), seq, seconds_per_seq=0.5, mode="OUT_1A@200")
    loop = g.loops()[0]
    assert loop.mu_seconds == pytest.approx(0.5), loop.mu_seconds


def test_route_ignores_loops() -> None:
    """Петля не участвует в маршруте: она никуда не ведёт."""
    g = PlaceGraph()
    a, b = _views(2)
    seq = 0
    for _ in range(3):
        for frame in (a, a, b):
            seq += 2
            g.observe(fingerprint(frame), seq, seconds_per_seq=0.1, mode="OUT_1A@200")
    ids = [p.id for p in g]
    assert g.loops(), "петли не появились — тест ничего не проверяет"
    route = g.route(ids[0], ids[1])
    assert route is not None
    assert all(e.src != e.dst for e in route)


# --- деление места ----------------------------------------------------------


def _split_setup() -> tuple[PlaceGraph, np.ndarray, np.ndarray, np.ndarray]:
    """Место, из которого одно и то же действие ведёт по-разному при разной яркости."""
    g = PlaceGraph(refine_margin=6.0, refine_min_n=2, refine_cell_min_n=0)
    here, there, elsewhere = _views(3, seed=11)
    return g, here, there, elsewhere


def test_place_splits_when_its_own_predictions_diverge() -> None:
    """Два исхода одного действия + разошедшиеся уровни = место склеило два состояния."""
    g, here, there, elsewhere = _split_setup()
    seq = 0
    # Светло: отсюда уходим в `there`. Темно: отсюда уходим в `elsewhere`.
    for _ in range(3):
        for bright, dst in ((True, there), (False, elsewhere)):
            src = here if bright else _dark(here)
            seq += 1
            g.see(src, seq, seconds_per_seq=0.1, mode="start")
            seq += 1
            g.see(dst if bright else _dark(dst), seq, seconds_per_seq=0.1,
                  mode="OUT_1A@200")

    report = g.refine()
    assert report, "место не разделилось, хотя исходы расходятся по яркости"
    assert report[0]["feature"] == "level"
    assert report[0]["gap"] >= 6.0
    assert report[0]["edges_dropped"] > 0, "статистика склеенного места осталась"
    assert g.stats()["refinements"] == 1


def test_split_throws_away_the_conflated_statistics() -> None:
    """Рёбра делимого места выбрасываются, а не делятся пополам.

    Разделить статистику склеенного узла значило бы придумать данные: она собрана
    про узел, которого больше нет.
    """
    g, here, there, elsewhere = _split_setup()
    seq = 0
    for _ in range(3):
        for bright, dst in ((True, there), (False, elsewhere)):
            src = here if bright else _dark(here)
            seq += 1
            g.see(src, seq, seconds_per_seq=0.1, mode="start")
            seq += 1
            g.see(dst if bright else _dark(dst), seq, seconds_per_seq=0.1,
                  mode="OUT_1A@200")
    before = len(g.edges)
    g.refine()
    split = next(iter(g.splits))
    assert len(g.edges) < before
    assert not [e for e in g.edges.values() if split in (e.src, e.dst)]


def test_no_split_when_nothing_separates_the_outcomes() -> None:
    """Расходятся исходы, но не по яркости — значит порог выдумывать нельзя.

    Это половина смысла механизма. Деление «на всякий случай» дало бы место,
    разделённое ни по чему, и разведка начала бы всё заново без причины.
    """
    g = PlaceGraph(refine_margin=6.0, refine_min_n=2, refine_cell_min_n=0)
    here, there, elsewhere = _views(3, seed=13)
    seq = 0
    for i in range(6):
        seq += 1
        g.see(here, seq, seconds_per_seq=0.1, mode="start")
        seq += 1
        g.see(there if i % 2 else elsewhere, seq, seconds_per_seq=0.1,
              mode="OUT_1A@200")
    pairs = {(e.src, e.mode) for e in g.edges.values()}
    assert any(len([e for e in g.edges.values() if (e.src, e.mode) == p]) >= 2
               for p in pairs), "исходы не разошлись — тест ничего не проверяет"
    assert g.refine() == []
    assert g.splits == {}


def test_split_needs_more_than_one_observation_of_each_outcome() -> None:
    """Один расходящийся проход — шум узнавания, а не открытие."""
    g = PlaceGraph(refine_margin=6.0, refine_min_n=3, refine_cell_min_n=0)
    here, there, elsewhere = _views(3, seed=17)
    seq = 0
    for bright, dst in ((True, there), (False, elsewhere)):
        src = here if bright else _dark(here)
        seq += 1
        g.see(src, seq, seconds_per_seq=0.1, mode="start")
        seq += 1
        g.see(dst if bright else _dark(dst), seq, seconds_per_seq=0.1,
              mode="OUT_1A@200")
    assert g.refine() == [], "разделилось по одному наблюдению на исход"


def test_place_can_split_by_more_than_one_feature() -> None:
    """Место делится столько раз, сколько разных признаков его склеивают.

    Один порог на место был первой версией. Он упирался в измеримое: состояние
    мира, различимое двумя признаками сразу, делилось только по первому, а второе
    расхождение оставалось неразделённым навсегда. Здесь два действия из одного
    места расходятся по разным признакам — по яркости и по одной ячейке.
    """
    rows = np.linspace(20, 235, 64, dtype=np.uint8)
    here = np.repeat(rows[:, None], 96, axis=1)
    marked = here.copy()
    marked[:8, :12] = 235
    dst = _views(4, seed=23)

    g = PlaceGraph(refine_margin=6.0, refine_min_n=2, refine_cell_min_n=2,
                   refine_max_tests=3)
    seq = 0
    for i in range(6):
        # Действие A расходится по яркости: светло → dst[0], темно → dst[1].
        for bright in (True, False):
            src = here if bright else _dark(here)
            seq += 1
            g.see(src, seq, seconds_per_seq=0.1, mode="start")
            seq += 1
            target = dst[0] if bright else _dark(dst[1])
            g.see(target, seq, seconds_per_seq=0.1, mode="OUT_1A@200")
        # Действие B расходится по ячейке: угол светлее → dst[2], иначе → dst[3].
        for corner in (True, False):
            seq += 1
            g.see(marked if corner else here, seq, seconds_per_seq=0.1, mode="start")
            seq += 1
            g.see(dst[2] if corner else dst[3], seq, seconds_per_seq=0.1,
                  mode="OUT_2B@200")

    tests: list[str] = []
    for _ in range(6):
        # Уточнение делает по одному делению на семью за вызов: рёбра выброшены, и
        # судить по ним второй раз нельзя. Второй признак приходит следующим кругом.
        for r in g.refine():
            tests.append(r["feature"])
        for i in range(6):
            for bright in (True, False):
                src = here if bright else _dark(here)
                seq += 1
                g.see(src, seq, seconds_per_seq=0.1, mode="start")
                seq += 1
                g.see(dst[0] if bright else _dark(dst[1]), seq, seconds_per_seq=0.1,
                      mode="OUT_1A@200")
            for corner in (True, False):
                seq += 1
                g.see(marked if corner else here, seq, seconds_per_seq=0.1, mode="start")
                seq += 1
                g.see(dst[2] if corner else dst[3], seq, seconds_per_seq=0.1,
                      mode="OUT_2B@200")

    assert len(tests) >= 2, f"признаков нашлось {tests}, а склейки две"
    assert len({t for t in tests}) >= 2, f"оба деления по одному признаку: {tests}"
    assert g.stats()["tests"] >= 2


def test_refine_never_repeats_the_same_feature() -> None:
    """Дважды делить по одному признаку нельзя: разряд уже есть, рёбра — зря."""
    g, here, there, elsewhere = _split_setup()
    seq = 0
    for _ in range(4):
        for bright, d in ((True, there), (False, elsewhere)):
            src = here if bright else _dark(here)
            seq += 1
            g.see(src, seq, seconds_per_seq=0.1, mode="start")
            seq += 1
            g.see(d if bright else _dark(d), seq, seconds_per_seq=0.1,
                  mode="OUT_1A@200")
    first = g.refine()
    assert first and first[0]["feature"] == "level"
    base = first[0]["base"]
    # Тот же признак второй раз не берётся, каким бы разрывом он ни разделял
    a = Traversal("x", "y", "OUT_1A@200", n=5, src_levels=[10.0] * 5)
    b = Traversal("x", "z", "OUT_1A@200", n=5, src_levels=[200.0] * 5)
    assert g._separating_threshold(a, b, existing=g.splits[base]) is None


def test_number_of_features_per_place_is_bounded() -> None:
    """Предел признаков обязателен: иначе место превращается в таблицу по пикселям."""
    g = PlaceGraph(refine_max_tests=1, refine_margin=6.0, refine_min_n=2,
                   refine_cell_min_n=2)
    here, there, elsewhere = _views(3, seed=29)
    seq = 0
    for _ in range(4):
        for bright, d in ((True, there), (False, elsewhere)):
            src = here if bright else _dark(here)
            seq += 1
            g.see(src, seq, seconds_per_seq=0.1, mode="start")
            seq += 1
            g.see(d if bright else _dark(d), seq, seconds_per_seq=0.1,
                  mode="OUT_1A@200")
    assert g.refine(), "первое деление не случилось — тест ничего не проверяет"
    base = next(iter(g.splits))
    assert len(g.splits[base]) == 1
    # Сколько бы расхождений ни нашлось дальше, второго признака не будет
    for _ in range(4):
        for bright, d in ((True, there), (False, elsewhere)):
            src = here if bright else _dark(here)
            seq += 1
            g.see(src, seq, seconds_per_seq=0.1, mode="start")
            seq += 1
            g.see(d if bright else _dark(d), seq, seconds_per_seq=0.1,
                  mode="OUT_2B@200")
        g.refine()
    assert len(g.splits[base]) == 1, g.splits[base]


def test_invariant_4_band_ids_stay_opaque() -> None:
    """Идентификатор полосы не рассказывает, что это «то же место, но ярче»."""
    fp = fingerprint(_views(1)[0])
    base, band = place_id(fp), place_id(fp, 1)
    assert base != band
    for pid in (base, band):
        assert pid.startswith("PLACE_") and len(pid) == len("PLACE_") + 4
        assert pid[6:].isalnum() and pid[6:].upper() == pid[6:]


def test_view_keeps_what_normalisation_discarded() -> None:
    """Уровень и контраст сохраняются, но в узнавании не участвуют."""
    frame = _views(1)[0]
    bright, dark = view(frame), view(_dark(frame))
    assert similarity(bright.cells, dark.cells) > 0.8, "место перестало узнаваться"
    assert bright.level > dark.level + 6.0, "уровень не отличается — делить будет нечем"


def test_edge_remembers_a_bounded_tail_of_levels() -> None:
    """Ребро помнит хвост уровней, а не всю историю: расти здесь нечему."""
    g = PlaceGraph()
    frame = _views(1)[0]
    for seq in range(1, LEVELS_KEPT * 3):
        g.see(frame, seq, seconds_per_seq=0.1, mode="OUT_1A@200")
    loop = g.loops()[0]
    assert len(loop.src_levels) == LEVELS_KEPT
    assert len(loop.src_cells) == LEVELS_KEPT


def test_refine_settings_come_from_the_profile() -> None:
    """Пороги деления — настройки поведения, а не константы в коде."""
    p = from_schema("ТЕСТ", place_refine_margin=11.5, place_refine_min_n=5,
                    place_refine_cell_min_n=9)
    g = PlaceGraph.from_profile(p)
    assert g.refine_margin == 11.5 and g.refine_min_n == 5
    assert g.refine_cell_min_n == 9


def test_cell_split_needs_more_observations_than_brightness() -> None:
    """Деление по одной ячейке требует больше наблюдений — и по измеренной причине.

    Ячеек 64, уровней 4: при двух наблюдениях в группе какая-нибудь ячейка разделит
    их почти всегда, и деление окажется по шуму. Поэтому у деления по ячейке свой,
    более высокий порог, а ноль его выключает совсем.
    """
    _, there, elsewhere = _views(3, seed=19)
    # Вид со структурой, а не шум: на шуме квантование ячеек неустойчиво, и «тот же
    # вид с одним отличием» получить нельзя. Градиент даёт устойчивые уровни.
    rows = np.linspace(20, 235, 64, dtype=np.uint8)
    here = np.repeat(rows[:, None], 96, axis=1)
    # Один угол кадра другой: так выглядит исчезнувшая панель или открытое окно.
    marked = here.copy()
    marked[:8, :12] = 235
    assert similarity(fingerprint(here), fingerprint(marked)) > 0.9

    def run(cell_min_n: int) -> list[dict]:
        g = PlaceGraph(refine_margin=1e9,          # яркостью делить запрещено
                       refine_min_n=2, refine_cell_min_n=cell_min_n)
        seq = 0
        for i in range(8):
            seq += 1
            g.see(here if i % 2 else marked, seq, seconds_per_seq=0.1, mode="start")
            seq += 1
            g.see(there if i % 2 else elsewhere, seq, seconds_per_seq=0.1,
                  mode="OUT_1A@200")
        return g.refine()

    assert run(0) == [], "деление по ячейке не выключается нулём"
    loose = run(2)
    assert loose and loose[0]["feature"].startswith("cell:"), (
        "при низком пороге деление по ячейке не сработало — тест ничего не проверяет")
