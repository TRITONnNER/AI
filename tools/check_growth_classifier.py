"""Своя доля ложных срабатываний у классификатора роста. TASK-24, D, инвариант 31.

Часовой замер (`tools/measure_hour.py`) объявляет класс каждой величины — линейный рост,
полка, падение, не растёт — и сверяет его с предрегистрированным ожиданием. Само это
объявление есть **проверка**, и по инварианту 31 она обязана предъявить два числа: сколько
раз она кричит на пустом месте и какую часть возможных случаев вообще видит. Без них
«три величины разошлись с ожиданием» может означать и находку, и собственную ошибку прибора.

## Как считается

Классификатор запускается на **сгенерированных** сериях, чей класс известен по построению:
константа, линейный рост, выходящая на полку, падающая. Шум добавляется на трёх уровнях,
потому что «не растёт» — суждение о величине шума, а не о его отсутствии; на каждый уровень
по несколько сидов.

- **Ложное срабатывание** — класс объявлен не тот, что заложен в серию. В обе стороны:
  «линейный вместо полки» и «полка вместо линейного» одинаково ложны.
- **Покрытие** — какая доля объявляемых классов вообще встретилась в контроле. Проверка,
  умеющая различать два класса из четырёх, молчанием ничего не доказывает.

Единица независимости — **серия**: точки внутри серии зависимы по построению.

Прогон: `python3 tools/check_growth_classifier.py`. Результат —
`docs/measurements/hour_classifier.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from measure_hour import classify, slope_and_plateau                  # noqa: E402

#: Серии с известным классом. Точек столько же, сколько отрезков в часовом прогоне (120 по
#: 30 секунд), чтобы проверялся классификатор в той же расстановке, а не в удобной.
POINTS = 120
#: Уровень шума как доля от размаха серии. Ноль тоже нужен: без него неизвестно, ошибается
#: классификатор от шума или сам по себе.
NOISE = (0.0, 0.02, 0.10)
SEEDS = (1, 2, 3, 4, 5)


def series(kind: str, n: int) -> np.ndarray:
    """Серия заданного класса без шума. Формулы объявлены здесь, а не подобраны."""
    t = np.arange(n, dtype=float)
    if kind == "не растёт":
        return np.full(n, 50.0)
    if kind == "линейный":
        return 10.0 + 2.0 * t
    if kind == "полка":
        # Насыщение: наклон последней трети заведомо меньше половины наклона первой.
        return 100.0 * (1.0 - np.exp(-t / (n / 12.0)))
    if kind == "падение":
        return 300.0 - 2.0 * t
    raise ValueError(f"нет такого класса: {kind}")


def main() -> int:
    rows: list[dict[str, Any]] = []
    kinds = ("не растёт", "линейный", "полка", "падение")
    xs = [float(i) for i in range(POINTS)]
    for kind in kinds:
        base = series(kind, POINTS)
        # Масштаб шума: размах серии, а для константы — её уровень. Иначе у константы
        # размах равен нулю, шум оказался бы абсолютным и заведомо крошечным, и «0 % ошибок
        # при шуме 10 %» относилось бы к шуму в одну пятидесятую процента.
        span = float(base.max() - base.min()) or float(abs(base.mean()))
        for noise in NOISE:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                ys = list(base + rng.normal(0.0, noise * span, POINTS))
                fit = slope_and_plateau(xs, ys)
                # `classify` сверяет с ожиданием по имени величины; здесь ожидание —
                # заложенный класс, поэтому имя подставляется через ту же таблицу.
                got, _ = classify(_name_for(kind), fit, ys=ys)
                rows.append({"kind": kind, "noise": noise, "seed": seed, "got": got,
                             "ok": got == kind})

    by_kind: dict[str, Any] = {}
    for kind in kinds:
        mine = [r for r in rows if r["kind"] == kind]
        wrong = [r for r in mine if not r["ok"]]
        by_kind[kind] = {
            "n": len(mine), "unit": "серия", "wrong": len(wrong),
            "share": len(wrong) / len(mine),
            "confusions": sorted({r["got"] for r in wrong}),
        }
    # Доля ложных **по уровню шума**, а не сводная. Сводная здесь обманывает: ошибки
    # сосредоточены на константе при большом шуме, и одно число на всё смешивает «проверка
    # надёжна» с «проверке нельзя верить», не давая различить, когда именно.
    by_noise: dict[str, Any] = {}
    for noise in NOISE:
        mine = [r for r in rows if r["noise"] == noise]
        wrong = [r for r in mine if not r["ok"]]
        by_noise[f"{noise:.0%}"] = {
            "n": len(mine), "unit": "серия", "wrong": len(wrong),
            "share": len(wrong) / len(mine),
            "wrong_kinds": sorted({r["kind"] for r in wrong})}
    seen = {r["got"] for r in rows}
    data = {
        "rows": rows, "by_kind": by_kind, "by_noise": by_noise,
        "points": POINTS,
        "noise_levels": list(NOISE), "seeds": list(SEEDS), "unit": "серия",
        "false_share": sum(1 for r in rows if not r["ok"]) / len(rows),
        "coverage": {"classes_declared": list(kinds),
                     "classes_seen": sorted(seen),
                     "share": len(seen & set(kinds)) / len(kinds)},
        "claim": ("классификатор роста в часовом замере различает четыре класса на сериях "
                  "с известным классом и не путает полку с линией"),
        "how_refuted": ("если доля ошибок велика, «класс разошёлся с ожиданием» в часовом "
                        "замере — собственная ошибка прибора, а не свойство прогона"),
    }
    out = ROOT / "docs" / "measurements" / "hour_classifier.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'заложено':<12} {'серий':>6} {'ошибок':>7} {'доля':>6}  путаница")
    for kind, v in by_kind.items():
        print(f"{kind:<12} {v['n']:>6} {v['wrong']:>7} {v['share']:>5.0%}  "
              f"{', '.join(v['confusions']) or '—'}")
    print()
    print(f"{'шум':<8} {'серий':>6} {'ошибок':>7} {'доля':>6}  на каких классах")
    for key, v in by_noise.items():
        print(f"{key:<8} {v['n']:>6} {v['wrong']:>7} {v['share']:>5.0%}  "
              f"{', '.join(v['wrong_kinds']) or '—'}")
    print()
    print(f"доля ложных срабатываний всего: {data['false_share']:.1%} "
          f"({len(rows)} серий: {len(kinds)} класса × {len(NOISE)} уровня шума × "
          f"{len(SEEDS)} сидов)")
    print(f"покрытие: объявлено классов {len(kinds)}, встретилось "
          f"{len(data['coverage']['classes_seen'])} — "
          f"{', '.join(data['coverage']['classes_seen'])}")
    print(f"записано: {out.relative_to(ROOT)}")
    return 0


def _name_for(kind: str) -> str:
    """Величина из часового замера, чей объявленный класс совпадает с проверяемым.

    Нужна затем, что `classify` берёт ожидание из таблицы `EXPECTED` по имени величины, а не
    аргументом. Подставить имя — честнее, чем дублировать логику классификатора здесь: копия
    расходится с оригиналом, и проверка начинает проверять копию.
    """
    from measure_hour import EXPECTED

    for name, spec in EXPECTED.items():
        if spec["class"] == kind:
            return name
    # «Падение» ни одной величиной не ожидается — это исход, а не ожидание. Берётся любая
    # величина с ожиданием «не растёт»: сверка всё равно идёт по возвращённому классу.
    if kind == "падение":
        return next(n for n, s in EXPECTED.items() if s["class"] == "не растёт")
    raise ValueError(f"в EXPECTED нет величины с классом {kind!r}")


if __name__ == "__main__":
    raise SystemExit(main())
