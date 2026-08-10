"""Драйвы из корреляции областей интерфейса с событиями. TASK-24, направление B, начало М6.

Проверяется механизм, а не числа: числа с единицами живут в `tools/measure_drives.py` и
`docs/measurements/drives.json`. Здесь — утверждения об устройстве, каждое из которых уже
один раз оказывалось неверным или чуть не оказалось:

1. Область — непрозрачное имя без координат (инвариант 4).
2. Уровень считается **по пикселям экранного слоя**: средняя по всей ячейке измеряет
   движение мира, а не состояние интерфейса.
3. Устойчивость ячейки определяется положением, а не тем, случилось ли за прогрев
   интересующее событие.
4. Окно корреляции включает тик самого события: смещение на единицу — причина, по которой
   первая редакция находила ноль связей там, где связь была заведомо.
5. Растущая связь драйвом не становится: драйв — дефицит, рост — награда, а награды нет.
6. Гипотеза уходит в хранилище с происхождением `experience` и с тестом в действиях агента.
7. Порог и минимальное число событий — из схемы, а не из кода анализа (инвариант 23).

Единица независимости — **утверждение об устройстве механизма**. Окружения не требует.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.core.profile import from_schema
from harness.model.beliefs import BeliefStore, Origin
from harness.model.drives import DriveOrigin
from harness.model.gauges import (Cell, GaugeError, Watcher, drive_of, find_links,
                                  grid_cells, ground, hypothesis_of)

PROF = from_schema("ДРАЙВЫ")


def _known_link(*, ticks: int = 240, period: int = 40, drop: float = 60.0,
                seed: int = 1) -> Watcher:
    """Мир уровней с одной заведомой связью: `AREA_0_0` убывает при `OUT_DROP`."""
    rng = np.random.default_rng(seed)
    ids = ["AREA_0_0", "AREA_0_1"]
    w = Watcher([Cell(0, i, 0, i, 1, 1) for i in range(len(ids))])
    w.stable = ids
    w.series = {cid: [] for cid in ids}
    level = {cid: 200.0 for cid in ids}
    for tick in range(ticks):
        cls = "OUT_DROP" if tick % period == 7 else "OUT_IDLE"
        if cls == "OUT_DROP":
            level["AREA_0_0"] = max(0.0, level["AREA_0_0"] - drop)
        for cid in ids:
            level[cid] = min(255.0, max(0.0, level[cid] + 1.5 + rng.normal(0.0, 1.0)))
        w.observe_levels({cid: level[cid] for cid in ids}, [cls])
    return w


# --- область: имя без координат ---------------------------------------------


def test_area_id_carries_no_coordinates() -> None:
    """Инвариант 4: агент ссылается на область, но не знает, где она на экране."""
    cell = Cell(3, 5, 240, 400, 80, 80)
    assert cell.id == "AREA_3_5"
    for forbidden in ("240", "400", "80"):
        assert forbidden not in cell.id


def test_grid_cells_are_equal_and_cover_the_frame() -> None:
    """Неравные ячейки вносили бы структуру, которой в пикселях нет."""
    cells = grid_cells((48, 64), rows=4, cols=8)
    assert len(cells) == 32
    assert len({(c.height, c.width) for c in cells}) == 1


# --- уровень: только по экранному слою --------------------------------------


def test_level_is_measured_on_screen_layer_pixels_only() -> None:
    """Средняя по всей ячейке измеряет мир, просвечивающий сквозь неё, а не интерфейс.

    Проверяется числом: та же ячейка, тот же кадр, разница только в маске. Без маски
    ответом становится яркость мира, с маской — яркость полоски.
    """
    frame = np.full((10, 10), 200.0)
    frame[0:2, :] = 20.0                       # тонкая полоска интерфейса
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:2, :] = True
    cell = Cell(0, 0, 0, 0, 10, 10)
    assert cell.value(frame, mask) == pytest.approx(20.0)
    assert cell.value(frame) == pytest.approx(164.0), (
        "без маски восемь строк мира перевешивают две строки интерфейса")


def test_a_cell_entirely_outside_the_screen_layer_gives_no_level() -> None:
    """Ячейка без пикселей экранного слоя — это мир, и уровня интерфейса у неё нет."""
    frame = np.full((4, 4), 100.0)
    assert Cell(0, 0, 0, 0, 4, 4).value(frame, np.zeros((4, 4), dtype=bool)) == 0.0


def test_use_mask_off_is_a_declared_control_not_a_concession() -> None:
    """Прогон без разделения слоёв — контроль на то, много ли разделение даёт."""
    frame = np.full((4, 4), 100.0)
    frame[0, :] = 0.0
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, :] = True
    cells = grid_cells((4, 4), rows=1, cols=1)
    with_mask, without = Watcher(cells, mask=mask), Watcher(cells, mask=mask,
                                                           use_mask=False)
    for w in (with_mask, without):
        w.stable = [cells[0].id]
        w.series = {cells[0].id: []}
        w.observe(frame, ["OUT_A"])
    assert with_mask.series["AREA_0_0"] == [0.0]
    assert without.series["AREA_0_0"] == [75.0]


# --- устойчивость: положение, а не наличие события за прогрев ---------------


def test_settle_keeps_a_thin_bar_that_never_changed_during_warmup() -> None:
    """Критерий, зависящий от того, случилось ли событие, работает против своей цели.

    Первая редакция требовала изменения уровня за прогрев и отсекала ровно те области, ради
    которых всё делается: полоска за прогрев не убывала ни разу, её уровень был константой.
    Чем реже событие, тем чаще такая проверка выбрасывала бы искомое.
    """
    mask = np.zeros((40, 40), dtype=bool)
    mask[0:8, :] = True                        # полоска в верхней строке ячеек
    cells = grid_cells((40, 40), rows=4, cols=4)
    kept = Watcher(cells).settle(mask)
    assert kept, "полоска обязана попасть в наблюдение"
    assert all(cid.startswith("AREA_0_") for cid in kept)


def test_min_screen_below_a_half_because_a_thin_bar_never_holds_a_majority() -> None:
    """Порог доли — 0.2, и это решение: требовать большинства значит видеть только толстое."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[0:3, :] = True                        # 3 из 10 строк ячейки
    cells = grid_cells((40, 40), rows=4, cols=4)
    assert Watcher(cells).settle(mask, min_screen=0.2)
    assert not Watcher(cells).settle(mask, min_screen=0.5)


def test_observe_levels_refuses_to_invent_a_missing_level() -> None:
    """Никаких молчаливых заглушек: не задан уровень — отказ, а не ноль."""
    w = Watcher(grid_cells((4, 4), rows=1, cols=1))
    w.stable, w.series = ["AREA_0_0"], {"AREA_0_0": []}
    with pytest.raises(GaugeError, match="подставить нечего"):
        w.observe_levels({}, ["OUT_A"])


# --- окно корреляции: смещение на единицу -----------------------------------


def test_the_correlation_window_includes_the_tick_of_the_event_itself() -> None:
    """Главный след события лежит в `deltas[t-1]`, и без этого связь не находится.

    Это тот самый дефект, который чуть не стал выводом «пиксельный путь М6 закрыт
    восприятием»: детектор находил ноль связей на мире с падением в 76 уровней яркости.
    Отсюда правило — утверждение о препятствии требует той же проверки, что об успехе.
    """
    links = find_links(_known_link(), profile=PROF)
    found = [l for l in links if l.area == "AREA_0_0" and l.event_class == "OUT_DROP"]
    assert found, "заведомая связь обязана находиться"
    assert found[0].falls and abs(found[0].effect) > 3.0
    assert found[0].n_events == 6 and found[0].as_dict()["unit"] == "событие класса"


def test_a_background_without_spread_is_skipped_not_called_infinite() -> None:
    """Делить не на что — пара пропускается, и это видно по отсутствию строки."""
    w = Watcher([Cell(0, 0, 0, 0, 1, 1)])
    w.stable, w.series = ["AREA_0_0"], {"AREA_0_0": []}
    for tick in range(60):
        w.observe_levels({"AREA_0_0": 100.0}, ["OUT_A" if tick % 10 == 3 else "OUT_B"])
    assert find_links(w, profile=PROF) == []


def test_thresholds_come_from_the_schema_so_they_reach_the_profile_hash() -> None:
    """Инвариант 23: порог, живущий в анализе, не сравним между прогонами."""
    w = _known_link()
    # Заведомая связь здесь силой d = 21.9, то есть выше потолка настройки (10.0); чтобы
    # порог мог её отсечь, ослабляется сам мир, а не порог выводится за схему.
    weak = _known_link(drop=8.0)
    assert find_links(w, profile=PROF) != []
    strict = from_schema("строгий", link_min_effect=10.0)
    assert find_links(weak, profile=PROF) != []
    assert find_links(weak, profile=strict) == []
    few = from_schema("мало событий", link_min_events=1000)
    assert find_links(w, profile=few) == []


def test_a_silent_class_yields_no_link_which_is_the_false_share_denominator() -> None:
    """Класс, который в мире не делает ничего: связь на нём ложная по построению.

    Это единственный доступный знаменатель для доли ложных связей (инвариант 31): мир без
    полоски контролем не является, потому что необратимое действие меняет область экрана
    само по себе.
    """
    w = _known_link()
    for tick in range(w.ticks):
        if tick % 17 == 0:                     # молчащий класс, вкраплённый в те же тики
            w.events[tick] = w.events[tick] + ("OUT_SILENT",)
    silent = [l for l in find_links(w, profile=PROF) if l.event_class == "OUT_SILENT"]
    assert silent == [], f"ложная связь на молчащем классе: {[l.claim() for l in silent]}"


# --- знак: дефицит против награды -------------------------------------------


def test_a_rising_link_is_recorded_but_never_becomes_a_drive() -> None:
    """Растущая связь — награда, а награды в проекте нет."""
    from harness.model.gauges import Link

    rising = Link(area="AREA_0_1", event_class="OUT_RISE", effect=+8.0, n_events=6,
                  mean_after=10.0, mean_background=1.0)
    assert not rising.falls and "растёт" in rising.claim()
    with pytest.raises(GaugeError, match="награда, а не дефицит"):
        drive_of(rising, series=[1.0, 2.0, 3.0], profile=PROF)


def test_a_discovered_drive_is_marked_discovered_and_says_what_grounds_it() -> None:
    """Иначе выведенный драйв неотличим от заданного, и вся разница М6 исчезает."""
    links = find_links(_known_link(), profile=PROF)
    link = next(l for l in links if l.falls)
    drive = drive_of(link, series=_known_link().series[link.area], profile=PROF)
    assert drive.origin is DriveOrigin.DISCOVERED
    assert drive.grounded_in == f"{link.area}@{link.event_class}"
    assert drive.name.startswith("дефицит:")


def test_the_drive_forecasts_instead_of_reporting_the_current_value() -> None:
    """Аллостаз: решение по прогнозу на горизонт, а не по тому, что сейчас.

    Проверяется знаком: серия падает — прогноз обязан быть ниже текущего значения.
    Гомеостаз реагировал бы на текущее и начинал бы действовать, когда уже поздно.
    """
    from harness.model.gauges import Link

    link = Link(area="AREA_0_0", event_class="OUT_DROP", effect=-9.0, n_events=6,
                mean_after=-30.0, mean_background=1.0)
    # Величина уже была на нуле раньше, потом восстановилась, теперь снова падает. Серия
    # именно такая нарочно: значение нормируется размахом, и на **историческом минимуме**
    # дефицит уже максимален — прогнозировать ниже там нечего, и это свойство постановки,
    # а не промах прогноза.
    falling = [0.0, 255.0, 200.0, 170.0, 140.0, 110.0]
    drive = drive_of(link, series=falling, profile=PROF)
    assert drive.forecast < drive.value, (
        f"падающая величина обязана прогнозироваться ниже: {drive.forecast} против "
        f"{drive.value}")
    rising = [255.0, 0.0, 60.0, 120.0, 180.0, 240.0]
    up = drive_of(link, series=rising, profile=PROF)
    assert up.forecast > up.value, "растущая — выше; иначе прогноз потерял знак"


# --- выход в конвейер убеждений ---------------------------------------------


def test_the_hypothesis_has_experience_provenance_and_a_test_in_agent_actions() -> None:
    """Утверждение без происхождения в хранилище не попадает (инвариант 7)."""
    link = next(l for l in find_links(_known_link(), profile=PROF) if l.falls)
    h = hypothesis_of(link, branch="b0", seq=3)
    assert h.provenance.origin is Origin.EXPERIENCE
    assert h.test and link.event_class in h.test and link.area in h.test
    assert "удержать" in h.test, "тест выражен в том, что агент умеет сделать сам"


def test_ground_puts_hypotheses_in_the_store_and_reports_numbers_with_a_unit() -> None:
    """Полный проход: связи → гипотезы → драйвы, и всё это числами для отчёта."""
    store = BeliefStore(branch="b0")
    got = ground(_known_link(), profile=PROF, store=store)
    assert got["links"] and got["drives"]
    assert got["unit"] == "связь (область × класс)"
    assert len(store.hypotheses) == len(got["links"])
    # Убеждением связь становится только после проверки действием — не здесь.
    assert all(not h.checked and h.outcome is None
               for h in store.hypotheses.values())


def test_a_claim_names_only_symbols_and_a_sign() -> None:
    """Ни одного смысла: «полоски здоровья» агент не знает и знать не может."""
    link = next(l for l in find_links(_known_link(), profile=PROF) if l.falls)
    claim = link.claim()
    assert claim.startswith("область AREA_")
    assert "убывает при событиях класса OUT_" in claim
    for meaning in ("здоровье", "health", "полоск", "урон"):
        assert meaning not in claim.lower()
