"""Обнаружение драйвов по корреляции: за сколько, сколько ложных, сколько пропущено.

TASK-24, направление B. Начало М6.

## Две части, и вторая — препятствие, а не успех

**Часть 1: механика корреляции против заведомо известной связи.** Мир — набор областей с
объявленной истиной: одна убывает при событиях класса X, одна растёт при событиях класса Y
(это не драйв, а награда, и драйва из неё делаться не должно), остальные шумят сами по себе.
Истина известна, значит измеримы **обе** ошибки: пропущенные связи и ложные.

Единица независимости — **связь** (пара «область × класс») для чисел о находках и **прогон**
для чисел о том, за сколько эпизодов связь находится: внутри прогона эпизоды зависимы, потому
что серия одна и та же.

**Часть 2: тот же детектор по настоящим пикселям синтетического мира.** Классов событий
три, и один из них — **контрольный**: выход, который в этом мире не делает ничего
(`silent_outputs` из истины мира). Связь, найденная на контрольном классе, ложная **по
построению**, и это единственный способ получить знаменатель для доли ложных: мир без
полоски контролем не является, потому что необратимое действие ломает панель интерфейса, то
есть меняет область экрана само по себе.

## Чуть не опубликованное препятствие

Первая редакция этого замера находила ноль связей и там, где связь заведомо была, и вывод
уже был написан: «разделитель слоёв объявляет заливку полоски миром, пиксельный путь М6
закрыт восприятием». Вывод был **неверен**. Причина нуля — смещение на единицу в окне
корреляции: `deltas[i]` есть изменение, наблюдаемое на тике `i+1`, а окно начиналось с
`deltas[t]` и главный след события пропускало целиком. После правки тот же детектор на тех
же пикселях находит связь с `d` больше десяти.

Отсюда правило, стоящее дороже самого замера: **утверждение о препятствии требует той же
проверки, что утверждение об успехе.** «Не нашлось» звучит скромно и потому проскакивает без
разбора, а стоит за ним ровно так же часто своя же ошибка.

Прогон: `python3 tools/measure_drives.py`. Результат — `docs/measurements/drives.json`.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.core.action import Action                                # noqa: E402
from harness.core.profile import from_schema                          # noqa: E402
from harness.corpus.world import InteractiveWorld                     # noqa: E402
from harness.model.beliefs import BeliefStore                         # noqa: E402
from harness.model.drives import DriveOrigin                          # noqa: E402
from harness.model.gauges import (Cell, Watcher, drive_of, find_links,  # noqa: E402
                                  grid_cells, ground)
from harness.vision.selfworld import SCREEN, LayerArbiter             # noqa: E402

#: Истина мира уровней: какая область как связана с каким классом. Объявлена здесь и в
#: механику не попадает — ровно как маска истины в замерах разделения слоёв.
TRUTH: dict[str, tuple[str, str]] = {
    "AREA_0_0": ("OUT_DROP", "убывает"),     # драйв: дефицит
    "AREA_0_1": ("OUT_RISE", "растёт"),      # награда: драйва быть не должно
}
DISTRACTORS = ("AREA_0_2", "AREA_0_3", "AREA_0_4", "AREA_0_5")
CLASSES = ("OUT_DROP", "OUT_RISE", "OUT_IDLE", "OUT_NOISE")

#: Сколько тиков между событиями класса. Событие обязано быть редким: если полоска не
#: успевает восстановиться, она стоит на нуле, и коррелировать становится не с чем.
PERIOD = 40
DROP = 0.30
REFILL = 0.008


def level_world(*, episodes: int, seed: int) -> Watcher:
    """Мир уровней с известной истиной. Кадров нет: уровни задаются прямо.

    Кадров нет намеренно. Часть 1 проверяет **механику корреляции**, и рисовать ради этого
    пиксели значило бы смешать её проверку с проверкой восприятия. Восприятие проверяется
    частью 2, на настоящих пикселях.
    """
    rng = np.random.default_rng(seed)
    ids = list(TRUTH) + list(DISTRACTORS)
    cells = [Cell(0, i, 0, i, 1, 1) for i in range(len(ids))]
    watcher = Watcher(cells)
    watcher.stable = ids
    watcher.series = {cid: [] for cid in ids}

    level = {cid: 1.0 for cid in ids}
    for tick in range(episodes):
        cls = "OUT_IDLE"
        if tick % PERIOD == 7:
            cls = "OUT_DROP"
        elif tick % PERIOD == 23:
            cls = "OUT_RISE"
        elif rng.random() < 0.05:
            cls = "OUT_NOISE"
        # Истинные связи.
        if cls == "OUT_DROP":
            level["AREA_0_0"] = max(0.0, level["AREA_0_0"] - DROP)
        if cls == "OUT_RISE":
            level["AREA_0_1"] = min(1.0, level["AREA_0_1"] + DROP)
        # Восстановление и собственный шум — у всех областей, включая связанные: связь
        # обязана находиться **на фоне** собственной динамики, а не в её отсутствие.
        for cid in ids:
            drift = REFILL if cid != "AREA_0_1" else -REFILL
            level[cid] = min(1.0, max(0.0, level[cid] + drift
                                      + rng.normal(0.0, 0.004)))
        watcher.observe_levels({cid: level[cid] * 255.0 for cid in ids}, [cls])
    return watcher


def part_one(profile: Any, *, seeds: tuple[int, ...], sizes: tuple[int, ...]
             ) -> dict[str, Any]:
    """Механика: за сколько эпизодов связь находится, сколько ложных, сколько пропущено."""
    rows: list[dict[str, Any]] = []
    for size in sizes:
        for seed in seeds:
            watcher = level_world(episodes=size, seed=seed)
            store = BeliefStore("замер-драйвов")
            got = ground(watcher, profile=profile, store=store)
            found = {(link["area"], link["event_class"], "убывает" if link["falls"]
                      else "растёт") for link in got["links"]}
            truth = {(area, cls, sign) for area, (cls, sign) in TRUTH.items()}
            hit = found & truth
            missed = truth - found
            false = found - truth
            drives = got["drives"]
            rows.append({
                "episodes": size, "seed": seed,
                "found": len(found), "true_found": len(hit),
                "missed": len(missed), "false": len(false),
                "false_list": sorted(f"{a}×{c}" for a, c, _ in false)[:4],
                "drives": len(drives),
                "drive_names": [d["name"] for d in drives],
                "hypotheses": len(store.hypotheses),
            })
    by_size: dict[str, Any] = {}
    for size in sizes:
        mine = [r for r in rows if r["episodes"] == size]
        by_size[str(size)] = {
            "n": len(mine), "unit": "прогон",
            "true_found_median": statistics.median(r["true_found"] for r in mine),
            "missed_median": statistics.median(r["missed"] for r in mine),
            "false_median": statistics.median(r["false"] for r in mine),
            "drives_median": statistics.median(r["drives"] for r in mine),
        }
    first_full = next((size for size in sizes
                       if by_size[str(size)]["true_found_median"] == len(TRUTH)), None)
    return {"rows": rows, "by_size": by_size,
            "episodes_to_find_all": first_full,
            "truth": {a: list(v) for a, v in TRUTH.items()},
            "unit_for_links": "связь (область × класс)",
            "unit_for_speed": "прогон"}


def part_two(profile_on: Any, profile_off: Any) -> dict[str, Any]:
    """Тот же детектор по настоящим пикселям, с контрольным классом событий.

    Контроль — молчащий выход: он ничем не подключён, и связь на нём ложная по построению.
    Мир без полоски контролем **не** является: необратимое действие ломает панель
    интерфейса, то есть меняет область экрана и без всякой полоски.
    """
    out: list[dict[str, Any]] = []
    for name, profile, gauge in (("полоска есть", profile_on, True),
                                 ("полоски нет", profile_off, False)):
        for use_mask in (True, False):
            world = InteractiveWorld(profile, seed=0)
            truth = world.truth()
            irreversible = truth["irreversible_outputs"][0]
            silent = (truth["silent_outputs"] or [None])[0]
            rng = np.random.default_rng(4)
            arbiter = LayerArbiter(profile)
            for _ in range(60):
                dx = int(rng.integers(-30, 31)) or 7
                obs = world.step(Action.mouse(dx, 0, 33), with_audio=False)
                arbiter.feed(obs.frame)
            mask = arbiter.result().pixel_mask(SCREEN)
            watcher = Watcher(grid_cells((90, 160), rows=12, cols=16), use_mask=use_mask)
            watcher.settle(mask)
            for tick in range(1200):
                if tick % 85 == 7:
                    act, cls = Action.key(irreversible, 100), irreversible
                elif silent is not None and tick % 85 == 44:
                    act, cls = Action.key(silent, 100), silent
                else:
                    act, cls = Action.nothing(100), "OUT_IDLE"
                obs = world.step(act, with_audio=False)
                watcher.observe(obs.frame, [cls])
            links = find_links(watcher, profile=profile)
            on_real = [link for link in links if link.event_class == irreversible]
            on_silent = [link for link in links if link.event_class == silent]
            on_idle = [link for link in links if link.event_class == "OUT_IDLE"]
            out.append({
                "world": name, "gauge": gauge, "by_screen_mask": use_mask,
                "areas_watched": len(watcher.stable),
                "links": len(links),
                "on_real_class": len(on_real),
                "on_silent_class": len(on_silent),
                "on_idle_class": len(on_idle),
                # Ложные — те, что на молчащем выходе: он ничего не делает, и любая связь
                # на нём есть ошибка детектора, а не свойство мира.
                "false_share": (len(on_silent) / len(links)) if links else None,
                "best_effect": max((abs(link.effect) for link in links), default=0.0),
            })
    return {"rows": out,
            "control": ("молчащий выход: связь на нём ложная по построению. Мир без "
                        "полоски контролем не является — необратимое действие ломает "
                        "панель и меняет область экрана само"),
            "note": ("первая редакция замера находила ноль связей из-за смещения на "
                     "единицу в окне корреляции и почти опубликовала это как препятствие "
                     "восприятия. Утверждение о препятствии требует той же проверки, что "
                     "утверждение об успехе"),
            "unit": "прогон"}


def main() -> int:
    profile = from_schema("ДРАЙВЫ", capture_width=160, capture_height=90)
    profile_on = from_schema("ДРАЙВЫ-П", capture_width=160, capture_height=90,
                             world_gauge=True, world_gauge_drop=DROP,
                             world_gauge_refill=REFILL)
    profile_off = from_schema("ДРАЙВЫ-К", capture_width=160, capture_height=90)

    print("часть 1: механика корреляции против известной истины")
    one = part_one(profile, seeds=(1, 2, 3), sizes=(80, 200, 400, 800))
    print(f"{'эпизодов':>9} {'найдено истинных':>17} {'пропущено':>10} {'ложных':>8} "
          f"{'драйвов':>8}")
    for size, v in one["by_size"].items():
        print(f"{size:>9} {v['true_found_median']:>17.1f} {v['missed_median']:>10.1f} "
              f"{v['false_median']:>8.1f} {v['drives_median']:>8.1f}")
    print(f"все истинные связи найдены начиная с {one['episodes_to_find_all']} эпизодов "
          f"(единица — прогон, сидов {len({r['seed'] for r in one['rows']})})")

    print()
    print("часть 2: тот же детектор по пикселям")
    two = part_two(profile_on, profile_off)
    print(f"{'мир':<14} {'по маске':>9} {'областей':>9} {'связей':>7} "
          f"{'на деле':>8} {'на молчащем':>12} {'ложных':>7} {'лучший d':>9}")
    for row in two["rows"]:
        false_share = ("—" if row["false_share"] is None
                       else f"{row['false_share']:.0%}")
        print(f"{row['world']:<14} {str(row['by_screen_mask']):>9} "
              f"{row['areas_watched']:>9} {row['links']:>7} {row['on_real_class']:>8} "
              f"{row['on_silent_class']:>12} {false_share:>7} "
              f"{row['best_effect']:>9.2f}")
    print()
    print("контроль: " + two["control"])
    print("оговорка: " + two["note"])

    data = {"part_one": one, "part_two": two,
            "thresholds": {"link_min_effect": profile.parameters["link_min_effect"],
                           "link_min_events": profile.parameters["link_min_events"]},
            "world": {"drop": DROP, "refill": REFILL, "period": PERIOD}}
    out = ROOT / "docs" / "measurements" / "drives.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nзаписано: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
