"""Внимание как бюджет: на что уходят окна и что они дают. TASK-24, направление C.

Три показателя названы задачей, и ни один из них не измерим без известной истины: надо
знать, какой источник **действительно** уточняет решение, а какой только мигает. Поэтому мир
здесь объявленный: часть источников снижает `sigma`, часть — нет, и обе части перечислены до
прогона.

## Что сравнивается

Три расстановки бюджета (`attention_windows` = 1, 2, 4) и **контроль**: тот же поток
претензий, но окна раздаются по порядку поступления, без арбитража по пользе. Контроль нужен
затем, что «арбитраж работает» без него означало бы «числа получились», а не «получились
лучше, чем без арбитража».

## Почему главное число — не «снижение sigma на окно»

Первая редакция сравнивала расстановки по среднему снижению `sigma` на окно и получила
отношение 1.00 на всех трёх бюджетах. Это не слабый эффект, а **вырожденная величина**:
числитель насыщается потолком мира (каждый источник отдаёт свой геометрический ряд и
кончается), знаменатель задан потоком претензий и бюджетом, и ни то, ни другое от порядка
раздачи не зависит. Итог 0.8866 против 0.8826 при **совпадающем до единицы** числе выданных
окон — сравнение числа с самим собой, инвариант 27. Занесено в `MEASUREMENT.md`, 13.6.

Арбитраж покупает не итог, а **скорость**: на 30-м такте те же прогоны дали 0.5266 против
0.3615. Поэтому главное число — **сколько окон потрачено до 60 % доступного снижения**;
потолок доступного объявлен ниже из истины мира, а не взят из результата. Меньше — лучше.

## Единица независимости и вид вывода

**Прогон**: окна внутри прогона зависимы (бюджет один и тот же, и выданное окно меняет
очередь и разброс источника). Сидов на точку девять.

Сравнение **парное**: поток претензий строится генератором до раздачи окон и от режима не
зависит, поэтому арбитраж и контроль на одном сиде видят ровно один и тот же поток. Вывод —
знаковый: арбитраж считается быстрее, только если он дешевле **на всех девяти** сидах.
Непарное правило «медиана вдвое лучше» пришлось выбросить: на парах контроль-против-контроля
оно срабатывало в 40–90 % случаев, то есть почти всегда врало (инвариант 31).

**Ложные срабатывания** считаются на контроле против контроля с перевёрнутым порядком
претензий: набор претензий тот же, порядок внутри такта другой, механизм раздачи одинаковый.
Сравнивать контроль с самим собой на том же сиде было бы вырожденным нулём — разница там ноль
по построению, и любая проверка прошла бы (инвариант 27).

Прогон: `python3 tools/measure_attention.py`. Результат — `docs/measurements/attention.json`.
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

from harness.core.profile import from_schema                          # noqa: E402
from harness.perception.attention import (FROM_ABOVE, FROM_BELOW,      # noqa: E402
                                          SALIENCE, Attention, Claim, Source)

#: Истина мира источников: сколько разброса источник снимает за одно окно. Ноль означает
#: «мигает и не даёт ничего» — именно на таких считается третий показатель задачи.
TRUTH_GAIN: dict[str, float] = {
    "AREA_ГРОМКАЯ": 0.0,      # мигает часто, не даёт ничего
    "AREA_ТИХАЯ": 0.20,       # редко просит, уточняет много
    "AREA_СРЕДНЯЯ": 0.06,
    "AREA_ПУСТАЯ": 0.0,
    "AREA_ЦЕЛЬ": 0.12,        # нужна планировщику
}
#: Как часто источник заявляет о себе снизу и сверху. «Громкая» кричит вдвое чаще всех.
CLAIM_RATE: dict[str, tuple[float, float]] = {
    "AREA_ГРОМКАЯ": (0.9, 0.0),
    "AREA_ТИХАЯ": (0.15, 0.0),
    "AREA_СРЕДНЯЯ": (0.4, 0.1),
    "AREA_ПУСТАЯ": (0.6, 0.0),
    "AREA_ЦЕЛЬ": (0.05, 0.8),
}
#: Сколько тактов длится наблюдение источника. Больше одного — окно занято и на следующем
#: такте, и именно такое окно может перебить заметность. Без источников с длинным
#: наблюдением ветка перебивания недостижима, а не редка: перебивать было бы нечего.
HOLD_TICKS: dict[str, int] = {
    "AREA_ГРОМКАЯ": 1,
    "AREA_ТИХАЯ": 2,
    "AREA_СРЕДНЯЯ": 1,
    "AREA_ПУСТАЯ": 1,
    "AREA_ЦЕЛЬ": 3,           # смотреть на цель долго; её и перебивают
}
#: Затухание отдачи источника: k-е наблюдение даёт `TRUTH_GAIN · DECAY^k`. Мир, где источник
#: отдаёт одно и то же вечно, отличал бы «пустой» от «исчерпанного» только словами.
DECAY = 0.7
#: Потолок доступного снижения: сумма геометрических рядов, ограниченная начальной `sigma`.
CEILING = sum(min(1.0, g / (1.0 - DECAY)) for g in TRUTH_GAIN.values())
#: Доля потолка, до которой считается расход окон. Объявлена здесь и попадает в запись; из
#: результатов не выбиралась — 0.6 взято как «больше половины доступного и достижимо на
#: любом из трёх бюджетов», что проверено отдельным числом `reached` в выводе.
TARGET_SHARE = 0.6
TICKS = 300


def one(*, windows: int, seed: int, arbitrate: bool,
        nuisance: bool = False) -> dict[str, Any]:
    """Один прогон. `arbitrate=False` — контроль: окна по порядку поступления.

    `nuisance=True` — тот же поток претензий с перевёрнутым порядком внутри такта. Помеха,
    которая не должна давать систематического выигрыша; на ней считается доля ложных
    срабатываний вывода. Генератор при этом трогается одинаково: перестановка идёт **после**
    `shuffle`, иначе сравнивались бы разные потоки, а не разные порядки одного.
    """
    profile = from_schema("ВНИМАНИЕ", attention_windows=windows)
    rng = np.random.default_rng(seed)
    attention = Attention.from_profile(
        profile, [Source(name, sigma=1.0) for name in TRUTH_GAIN],
        arbitrate=arbitrate)

    for _ in range(TICKS):
        claims: list[Claim] = []
        for name, (below_rate, above_rate) in CLAIM_RATE.items():
            if rng.random() < below_rate:
                claims.append(Claim(name, FROM_BELOW,
                                    SALIENCE[int(rng.integers(0, len(SALIENCE)))],
                                    urgency=1.0 + float(rng.random())))
            if rng.random() < above_rate:
                claims.append(Claim(name, FROM_ABOVE, "цель: дойти"))
        rng.shuffle(claims)
        if nuisance:
            claims.reverse()
        # step возвращает **занятые** окна: и начатые сейчас, и тянущиеся с прошлых тактов.
        # Наблюдение, которое прервали, из этого списка исчезает — так наблюдатель и узнаёт
        # о прерывании, и отчёта по нему не будет.
        for grant in attention.step(claims):
            if grant.ticks_held + 1 < HOLD_TICKS[grant.source]:
                continue          # наблюдение ещё идёт, окно занято
            source = attention.sources[grant.source]
            gain = TRUTH_GAIN[grant.source] * (DECAY ** (source.grants - 1))
            attention.report(grant.source, sigma_after=max(0.0, source.sigma - gain))

    got = attention.summary()
    got.pop("curve", None)        # кривая целиком не нужна в строке прогона
    got.update({"windows": windows, "seed": seed, "arbitrate": arbitrate,
                "nuisance": nuisance,
                "windows_to_target": attention.windows_to_gain(TARGET_SHARE * CEILING),
                "gain_at_30_windows": next(
                    (g for spent, g in reversed(attention.curve) if spent <= 30), 0.0)})
    return got


def _median_or_none(values: list[Any]) -> float | None:
    """Медиана по достигшим цели. `None` — цель не взята ни в одном прогоне."""
    got = [v for v in values if v is not None]
    return statistics.median(got) if got else None


def verdict(fast: list[Any], slow: list[Any]) -> dict[str, Any]:
    """Знаковый вывод по парным прогонам: «дешевле на всех сидах» или нет.

    Пара — один сид: поток претензий на нём одинаков у обоих режимов. Правило требует
    выигрыша во **всех** парах; при девяти парах случайное совпадение знаков имеет
    вероятность 1/512, и это единственная причина, по которой порог такой строгий.
    Промах при недостижении цели считается отдельно, а не молча выкидывается.
    """
    pairs = [(a, b) for a, b in zip(fast, slow) if a is not None and b is not None]
    wins = sum(1 for a, b in pairs if a < b)
    ratios = [b / a for a, b in pairs if a]
    return {"pairs": len(pairs), "unusable": len(fast) - len(pairs), "wins": wins,
            "fires": bool(pairs) and wins == len(pairs),
            "median_ratio": statistics.median(ratios) if ratios else None}


def main() -> int:
    seeds = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    rows: list[dict[str, Any]] = []
    for windows in (1, 2, 4):
        for arbitrate in (True, False):
            for seed in seeds:
                rows.append(one(windows=windows, seed=seed, arbitrate=arbitrate))
    # Прогоны для нуля: контроль против него же с перевёрнутым порядком претензий, на пяти
    # независимых блоках сидов. Одна проверка на бюджет давала бы «0 из 2» — доля, из которой
    # ничего не следует; пять блоков в обе стороны дают тридцать проверок.
    null_seeds = [tuple(range(101 + 9 * b, 110 + 9 * b)) for b in range(5)]
    null_rows: list[dict[str, Any]] = []
    for w in (1, 2, 4):
        for block in null_seeds:
            for s in block:
                null_rows.append(one(windows=w, seed=s, arbitrate=False))
                null_rows.append(one(windows=w, seed=s, arbitrate=False, nuisance=True))

    by_case: dict[str, Any] = {}
    for windows in (1, 2, 4):
        for arbitrate in (True, False):
            mine = [r for r in rows
                    if r["windows"] == windows and r["arbitrate"] is arbitrate]
            key = f"окон {windows}, {'арбитраж' if arbitrate else 'по порядку'}"
            by_case[key] = {
                "windows": windows, "arbitrate": arbitrate,
                "n": len(mine), "unit": "прогон",
                "below_share": statistics.median(r["below_share_of_grants"] for r in mine),
                "gain_per_grant": statistics.median(r["gain_per_grant"] for r in mine),
                "total_gain": statistics.median(r["total_gain"] for r in mine),
                "granted": statistics.median(r["granted"] for r in mine),
                "windows_to_target": _median_or_none(
                    [r["windows_to_target"] for r in mine]),
                "reached": sum(1 for r in mine if r["windows_to_target"] is not None),
                "gain_at_30_windows": statistics.median(
                    r["gain_at_30_windows"] for r in mine),
                "wasted_share": statistics.median(r["wasted_share"] for r in mine),
                "refused": statistics.median(r["refused"] for r in mine),
                "preemptions": statistics.median(r["preemptions"] for r in mine),
                "cut_short": statistics.median(r["cut_short"] for r in mine),
                "busy_skipped": statistics.median(r["busy_skipped"] for r in mine),
                "empty_found": statistics.median(len(r["empty_sources"]) for r in mine),
            }

    def spent(windows: int, *, arbitrate: bool, nuis: bool = False,
              where: list[dict[str, Any]] | None = None,
              only: tuple[int, ...] | None = None) -> list[Any]:
        src = rows if where is None else where
        return [r["windows_to_target"] for r in src
                if r["windows"] == windows and r["arbitrate"] is arbitrate
                and r["nuisance"] is nuis and (only is None or r["seed"] in only)]

    # Вывод по каждому бюджету и **доля ложных срабатываний того же вывода** (инвариант 31):
    # он же выносится на контроль против контроля с перевёрнутым порядком претензий. Всё, что
    # он там найдёт, — его собственная ошибка: механизм раздачи по обе стороны одинаков.
    verdicts: dict[str, Any] = {}
    false: dict[str, Any] = {}
    for windows in (1, 2, 4):
        verdicts[f"окон {windows}"] = verdict(spent(windows, arbitrate=True),
                                             spent(windows, arbitrate=False))
        fired = 0
        checks = 0
        wins: list[int] = []
        for block in null_seeds:
            a = spent(windows, arbitrate=False, where=null_rows, only=block)
            b = spent(windows, arbitrate=False, nuis=True, where=null_rows, only=block)
            # Ложное срабатывание — вердикт в любую сторону: помеха «выиграла» или
            # «проиграла» всухую. Обе стороны одинаково ложны, считать одну значило бы
            # вдвое занизить долю.
            forward, backward = verdict(a, b), verdict(b, a)
            fired += int(forward["fires"]) + int(backward["fires"])
            checks += 2
            wins.append(forward["wins"])
        false[f"окон {windows}"] = {
            "checks": checks, "fired": fired, "share": fired / checks,
            "blocks": len(null_seeds), "wins_per_block": wins,
            "pairs_per_check": len(null_seeds[0])}

    # Покрытие вывода (инвариант 31): какие ветки арбитража этот замер вообще задевает.
    # Молчание проверки, видящей четверть механизма, доказательством не является.
    branches = ("granted", "refused", "preemptions", "cut_short", "busy_skipped")
    hit = {b: sum(r[b] for r in rows) for b in branches}
    coverage = {
        "branches": hit,
        "fired": sum(1 for v in hit.values() if v),
        "declared": len(branches),
        "not_seen": ("Attention.modulate — ось настроения; отказ report по прерванному "
                     "окну — обе ветки покрыты тестами, а не этим замером"),
    }

    data = {
        "rows": rows, "null_rows": null_rows, "by_case": by_case,
        "verdicts": verdicts, "false_positives": false, "coverage": coverage,
        "ticks": TICKS, "seeds": list(seeds), "decay": DECAY,
        "truth_gain": TRUTH_GAIN, "hold_ticks": HOLD_TICKS,
        "claim_rate": {k: list(v) for k, v in CLAIM_RATE.items()},
        "ceiling": round(CEILING, 6), "target_share": TARGET_SHARE,
        "target": round(TARGET_SHARE * CEILING, 6),
        "unit": "прогон",
        "claim": ("арбитраж по пользе на стоимость набирает 60 % доступного снижения "
                  "sigma за меньшее число окон, и выигрыш тем больше, чем туже бюджет"),
        "how_refuted": ("если расход окон до цели у арбитража не ниже, чем у контроля, "
                        "арбитраж не работает, и «внимание как бюджет» — переименование"),
        "degenerate": ("снижение sigma на окно на насыщённом прогоне: итог упирается в "
                       "потолок мира, число окон задано потоком претензий, и отношение не "
                       "зависит от порядка раздачи. MEASUREMENT.md, 13.6"),
    }
    out = ROOT / "docs" / "measurements" / "attention.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"потолок доступного снижения {CEILING:.4f}, цель "
          f"{TARGET_SHARE:.0%} = {TARGET_SHARE * CEILING:.4f}\n")
    print(f"{'расстановка':<28} {'окон до цели':>13} {'взято':>6} {'снизу':>7} "
          f"{'впустую':>8} {'перебиваний':>12} {'прервано':>9}")
    for key, v in by_case.items():
        got = "—" if v["windows_to_target"] is None else f"{v['windows_to_target']:.0f}"
        print(f"{key:<28} {got:>13} {v['reached']:>3}/{v['n']:<2} "
              f"{v['below_share']:>6.0%} {v['wasted_share']:>8.0%} "
              f"{v['preemptions']:>12.0f} {v['cut_short']:>9.0f}")
    print()
    print("парный вывод: дешевле ли арбитраж на каждом сиде по отдельности")
    for windows in (1, 2, 4):
        v = verdicts[f"окон {windows}"]
        ratio = "—" if v["median_ratio"] is None else f"{v['median_ratio']:.2f}"
        print(f"  окон {windows}: выигрышей {v['wins']} из {v['pairs']}, медиана "
              f"отношения {ratio} — {'да' if v['fires'] else 'нет'}")
    print()
    print("вырожденная величина, для сверки: снижение sigma на окно")
    for windows in (1, 2, 4):
        a = by_case[f"окон {windows}, арбитраж"]
        b = by_case[f"окон {windows}, по порядку"]
        print(f"  окон {windows}: {a['gain_per_grant']:.4f} против "
              f"{b['gain_per_grant']:.4f} — итог {a['total_gain']:.4f} против "
              f"{b['total_gain']:.4f} при {a['granted']:.0f} и {b['granted']:.0f} окнах")
    print()
    print("доля ложных срабатываний вывода (контроль против него же с перевёрнутым "
          "порядком):")
    for key, v in false.items():
        print(f"  {key}: сработал {v['fired']} раз из {v['checks']} проверок "
              f"({v['share']:.0%}); выигрышей помехи по блокам "
              f"{v['wins_per_block']} из {v['pairs_per_check']}")
    print(f"покрытие: сработало ветвей {coverage['fired']} из {coverage['declared']} — "
          + ", ".join(f"{k} {v}" for k, v in coverage["branches"].items()))
    print(f"не видит: {coverage['not_seen']}")
    print()
    print(f"единица независимости: прогон, сидов {len(seeds)}, тиков на прогон {TICKS}")
    print(f"записано: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
