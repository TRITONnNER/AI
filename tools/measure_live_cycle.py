"""Замер живого цикла: конфабуляция как свойство расстановки частот. Начало М5.

Что здесь проверяется. `CLAUDE.md`, инвариант 13 предсказывает: «планировщик работает на
0.5 Гц и не имеет доступа к причинам действий рефлекса на 20 Гц, поэтому **он будет
конфабулировать структурно**… Это не чинится, это измеряется». Предсказание сделано из
устройства, а не из наблюдения, и до сих пор ни одного числа под ним не стояло.

Замер ставит вопрос так, чтобы ответ мог его опровергнуть: **если конфабуляция — свойство
расстановки частот, доля расхождений обязана следовать за отношением частот.** Если она
окажется постоянной при любом отношении, значит метрика мерит что-то другое, и предсказание
неверно в той форме, в которой записано.

Единица независимости — **прогон**: внутри прогона действия зависимы по построению (один мир,
один планировщик, одна цепочка). Сидов на точку несколько, и разброс по ним печатается рядом
со средним: доля 0.85 при разбросе 0.84–0.86 и та же доля при разбросе 0.4–1.0 — разные
утверждения.

Прогон: `python3 tools/measure_live_cycle.py`. Результат — `docs/measurements/live_cycle.json`.
"""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.behaviour.arena import reach_place                     # noqa: E402
from harness.core.profile import from_schema                        # noqa: E402
from harness.livecycle import run                                   # noqa: E402

#: Отношения частот рефлекса к частоте планировщика. Значения выбраны так, чтобы покрыть
#: три качественно разных случая, а не «побольше точек»: рефлекс много быстрее (как в
#: архитектуре), вдвое быстрее и **равен** планировщику. Если предсказание верно, доля
#: расхождений должна упасть от первого случая к третьему.
RATIOS: tuple[tuple[float, float], ...] = (
    (20.0, 0.5),      # архитектура: 40 к 1
    (8.0, 2.0),       # 4 к 1
    (2.0, 2.0),       # 1 к 1 — планировщик успевает столько же, сколько рефлекс
)

SEEDS = (1, 2, 3)
SECONDS = 2.5
HZ_SCALE = 20.0


def one(hz_reflex: float, hz_planner: float, seed: int) -> dict[str, Any]:
    profile = from_schema("М5-замер", capture_width=160, capture_height=90,
                          hz_reflex=hz_reflex, hz_planner=hz_planner)
    with tempfile.TemporaryDirectory(prefix="harness-m5-") as tmp:
        got = run(Path(tmp) / "s", scenario=reach_place(profile), profile=profile,
                  seconds=SECONDS, seed=seed, hz_scale=HZ_SCALE)
    conf = got.confabulation
    by = got.by_layer
    delivered = max(1, got.delivered)
    return {
        "hz_reflex": hz_reflex, "hz_planner": hz_planner,
        "ratio": hz_reflex / hz_planner, "seed": seed,
        "loops": got.loops, "world_ticks": got.world_ticks,
        "world_never_paused": got.world_never_paused,
        "layers_complete": got.layers_complete,
        "delivered": got.delivered,
        "reflex_share": by.get("reflex", 0) / delivered,
        "planner_share": by.get("planner", 0) / delivered,
        "preempted": got.preempted,
        "episodes": got.episodes.get("episodes", 0),
        "reached": got.episodes.get("by_outcome", {}).get("достигнута", 0),
        "confab_share": conf.get("share_by_explanation"),
        "confab_rate_by_episode": conf.get("rate"),
        # TASK-24, A: доля объяснённых действий и доля перехвата внутри ожидания. Первая
        # показывает, что объяснение накрывает уже **не всё**; вторая — то, что метрика
        # теперь мерит: как часто рефлекс перебивал команду, исполнения которой
        # планировщик ждал.
        "explained_share": got.as_dict()["explained_share"],
        "usurped_share": got.as_dict()["usurped_share"],
        "expectations": got.expectations,
        "confab_episodes": conf.get("n"),
        "confab_absent": conf.get("absent_reason", ""),
    }


def _median_of(rows: list[dict[str, Any]], key: str) -> float | None:
    """Медиана поля, если оно посчитано хоть где-то. `None` — не посчитано нигде."""
    got = [r[key] for r in rows if r.get(key) is not None]
    return None if not got else round(statistics.median(got), 4)


def main() -> int:
    rows: list[dict[str, Any]] = []
    for hz_reflex, hz_planner in RATIOS:
        for seed in SEEDS:
            got = one(hz_reflex, hz_planner, seed)
            rows.append(got)
            share = got["confab_share"]
            print(f"  рефлекс {hz_reflex:>4g} Гц / планировщик {hz_planner:g} Гц "
                  f"(1:{got['ratio']:.0f}), сид {seed}: "
                  f"конфабуляция "
                  + ("не измерена" if share is None else f"{share:.1%}")
                  + f", рефлекс вёл {got['reflex_share']:.1%} действий, "
                    f"перебиваний {got['preempted']}, эпизодов {got['episodes']}",
                  flush=True)

    by_ratio: dict[str, Any] = {}
    for hz_reflex, hz_planner in RATIOS:
        mine = [r for r in rows
                if r["hz_reflex"] == hz_reflex and r["hz_planner"] == hz_planner]
        shares = [r["confab_share"] for r in mine if r["confab_share"] is not None]
        reflex = [r["reflex_share"] for r in mine]
        key = f"1:{hz_reflex / hz_planner:.0f}"
        by_ratio[key] = {
            "hz_reflex": hz_reflex, "hz_planner": hz_planner,
            "n": len(mine), "unit": "прогон",
            "confab_median": None if not shares else round(statistics.median(shares), 4),
            "confab_min": None if not shares else round(min(shares), 4),
            "confab_max": None if not shares else round(max(shares), 4),
            "reflex_share_median": round(statistics.median(reflex), 4),
            "preempted_median": statistics.median([r["preempted"] for r in mine]),
            "explained_median": _median_of(mine, "explained_share"),
            "usurped_median": _median_of(mine, "usurped_share"),
        }

    # **Вырождение, найденное этим же замером.** Доля расхождений совпала с долей действий,
    # начатых рефлексом, до десятых процента на всех трёх отношениях. Это не совпадение и не
    # успех: в нынешней расстановке объяснение планировщика стоит до следующего объяснения и
    # приписывается **всем** действиям подряд, поэтому «объяснено не тем слоем» тождественно
    # «начато не планировщиком». Метрика равна уже известной величине, то есть не несёт
    # своей информации (инвариант 27).
    gap = max(abs((r["confab_share"] or 0.0) - r["reflex_share"]) for r in rows)
    # Порог вырождения — 1 п.п., и он объявлен здесь, а не подобран: две величины,
    # различающиеся меньше чем на процентный пункт на **всех** точках, для любого
    # практического вывода одно и то же число. Проверяется максимум разницы, а не среднее:
    # среднее спрятало бы совпадение на части точек.
    degenerate = gap < 0.01

    data = {
        "rows": rows, "by_ratio": by_ratio,
        "degenerate": degenerate,
        "max_gap_to_reflex_share": round(gap, 4),
        "degeneracy": (
            "доля расхождений тождественна доле действий рефлекса: объяснение "
            "планировщика приписывается всем действиям подряд, поэтому «объяснено не тем "
            "слоем» и «начато не планировщиком» — одно и то же. Метрика в этой "
            "расстановке не несёт информации, которой нет в распределении по слоям "
            "(вырождение по постановке, инвариант 27). Чинится не порогом: ссылка на "
            "объяснение должна ставиться там, где планировщик считает, что он и вызвал "
            "действие, а не на всё подряд"),
        "seconds_per_run": SECONDS, "hz_scale": HZ_SCALE,
        "seeds": list(SEEDS),
        "unit": "прогон",
        "claim": ("конфабуляция — свойство расстановки частот, а не честности агента: "
                  "доля расхождений следует за долей действий, начатых рефлексом"),
        "how_refuted": ("если доля расхождений не зависит от отношения частот, метрика "
                        "мерит не то, что заявлено, и предсказание инварианта 13 неверно "
                        "в записанной форме"),
        "world_never_paused": all(r["world_never_paused"] for r in rows),
        "layers_complete": all(r["layers_complete"] for r in rows),
    }
    out = ROOT / "docs" / "measurements" / "live_cycle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"{'отношение':>10} {'конфабуляция':>14} {'разброс':>15} "
          f"{'рефлекс вёл':>12} {'разница':>9} {'объяснено':>10}")
    for key, v in by_ratio.items():
        share = "—" if v["confab_median"] is None else f"{v['confab_median']:.1%}"
        rng = ("—" if v["confab_min"] is None
               else f"{v['confab_min']:.1%}…{v['confab_max']:.1%}")
        diff = ("—" if v["confab_median"] is None
                else f"{abs(v['confab_median'] - v['reflex_share_median']) * 100:.1f} п.п.")
        expl = ("—" if v["explained_median"] is None
                else f"{v['explained_median']:.1%}")
        print(f"{key:>10} {share:>14} {rng:>15} "
              f"{v['reflex_share_median']:>11.1%} {diff:>9} {expl:>10}")
    print()
    if degenerate:
        print()
        print("ВЫРОЖДЕНИЕ: доля расхождений совпала с долей действий рефлекса "
              f"(расхождение не больше {gap:.2%}). Метрика конфабуляции в этой "
              "расстановке — это доля рефлекса под другим именем, и своей информации она "
              "не несёт. Причина в постановке: объяснение планировщика приписывается всем "
              "действиям подряд. См. MEASUREMENT.md, раздел 13.")
    print()
    print(f"мир ни разу не останавливался: {data['world_never_paused']}; "
          f"слой-инициатор заполнен всегда: {data['layers_complete']} "
          f"(прогонов {len(rows)})")
    print(f"записано: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
