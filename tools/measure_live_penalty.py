#!/usr/bin/env python3
"""Насколько живой экран хуже синтетики — замером, а не на глаз. TASK-11, часть 1.

Зачем этот скрипт существует. `MEASUREMENT.md` §18 предрегистрирует ожидание по живому
IoU. Прежнее ожидание («около 0.6») было откалибровано против **0.97** — числа с одной
независимой единицей, — и против настоящего опорного числа синтетики (медиана 0.667 по
доменам) то же ожидание означает «ничего не изменится». Ожидание, которое нельзя
опровергнуть, — не ожидание, и заменять одно нарисованное рукой число другим
нарисованным рукой смысла нет (инвариант 23).

Что здесь делается вместо этого. Четыре причины, по которым живое обязано быть хуже,
уже названы в §18.2 словами. Каждая из них — **порча картинки**, и её можно нанести на
синтетический кадр:

1. компрессия и дизеринг → квантование плюс дрожание уровней между кадрами;
2. полупрозрачная панель → обрамление подмешивает то, что за ним;
3. частицы и эффекты → разреженное наложение, едущее с собственной скоростью;
4. курсор → мелкий объект, движущийся независимо от всего.

Кадры портятся, **истина не портится**: маски берутся у домена нетронутыми. Это ровно
то, что делает настоящий экран, — рисует то же самое хуже. Считает всё тот же
`bench_domain` (порча входит хуком `distort`), поэтому падение измеряется против
собственного числа того же кода на том же домене, а не против другой реализации.

Единица независимости — **домен**, `n` = 5 (инвариант 22). Сид меняет только шум порчи:
маска истины от сида не зависит (§2), поэтому сиды идут внутрь домена и докладываются
разбросом шума, а не числом наблюдений.

Чего этот замер **не** делает: он не предсказывает живой IoU. Он даёт границу сверху на
ожидание — «даже под названными четырьмя порчами синтетика даёт вот столько», — а живой
экран портит картинку и способами, которых в этом списке нет. Поэтому ожидание ставится
не выше полученного здесь.

    python3 tools/measure_live_penalty.py                    # все домены, 3 сида шума
    python3 tools/measure_live_penalty.py --seeds 6 --frames 240
    python3 tools/measure_live_penalty.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Четыре порчи, по одной на каждую названную причину
# ---------------------------------------------------------------------------
#
# Все амплитуды объявлены здесь константами с обоснованием, а не подобраны под
# результат. Подбирать их под желаемое падение означало бы получить ожидание, которое
# опять ничего не запрещает.

#: Компрессия и дизеринг. 6 бит на канал — это то, что остаётся от 8 после видеокодека
#: с невысоким битрейтом; ±2 уровня дрожания между кадрами — субпиксельное сглаживание
#: шрифтов и дизеринг градиентов. Главное здесь не «шумно», а то, что **одинаковые
#: пиксели перестают быть одинаковыми между кадрами**, а на этом стоит признак
#: неподвижности.
QUANT_BITS = 6
DITHER_LEVELS = 2

#: Полупрозрачная панель: непрозрачность 80 %, то есть 20 % того, что за ней. Число
#: взято из §18.2, где эта причина названа словами, а не выбрано здесь.
PANEL_ALPHA = 0.2

#: Частицы: доля пикселей кадра под наложением и скорость наложения в пикселях за кадр.
#: 1.5 % — редкий дым, не ливень: замер должен показать вклад причины, а не утопить
#: картинку.
PARTICLE_SHARE = 0.015
PARTICLE_SPEED = (3, 1)

#: Курсор: сторона квадрата в пикселях при кадре 320×180. Меньше — и он не попадёт ни в
#: одну оценку вовсе, то есть причина будет объявлена и не проверена.
CURSOR_SIDE = 7


def _compression(rng: np.random.Generator) -> Callable[[np.ndarray, int, Any], np.ndarray]:
    """Квантование плюс дрожание уровней. Портит признак неподвижности."""
    step = 1 << (8 - QUANT_BITS)

    def f(frame: np.ndarray, i: int, domain: Any) -> np.ndarray:
        q = (frame // step) * step
        jitter = rng.integers(-DITHER_LEVELS, DITHER_LEVELS + 1, size=frame.shape)
        return np.clip(q.astype(np.int16) + jitter, 0, 255).astype(np.uint8)

    return f


def _translucent(rng: np.random.Generator) -> Callable[[np.ndarray, int, Any], np.ndarray]:
    """Обрамление подмешивает то, что за ним.

    Где обрамление — берётся у домена. Это законно: порча описывает, как **мир рисует**
    панель, и знание о панели есть у мира. Проверяемый код маски не видит; истина, с
    которой сверяют ответ, тоже не меняется.
    """
    def f(frame: np.ndarray, i: int, domain: Any) -> np.ndarray:
        mask = domain.screen_mask()
        if mask is None or not mask.any():
            return frame
        # «То, что за панелью» — содержимое кадра, сдвинутое так, что под панель попадает
        # не она сама. Настоящего «за панелью» в двумерном кадре нет ни у кого, включая
        # настоящий экран после захвата: там оно тоже уже смешано.
        behind = np.roll(frame, shift=(11, 17), axis=(0, 1))
        out = frame.astype(np.float32)
        out[mask] = (1.0 - PANEL_ALPHA) * out[mask] + PANEL_ALPHA * behind[mask]
        return np.clip(out, 0, 255).astype(np.uint8)

    return f


def _particles(rng: np.random.Generator) -> Callable[[np.ndarray, int, Any], np.ndarray]:
    """Разреженное наложение со своей скоростью: ведёт себя как мир, миром не является."""
    made: dict[str, np.ndarray] = {}

    def f(frame: np.ndarray, i: int, domain: Any) -> np.ndarray:
        if "pts" not in made:
            n = max(1, int(frame.size * PARTICLE_SHARE))
            ys = rng.integers(0, frame.shape[0], size=n)
            xs = rng.integers(0, frame.shape[1], size=n)
            made["pts"] = np.stack([ys, xs])
            made["val"] = rng.integers(200, 256, size=n).astype(np.uint8)
        ys = (made["pts"][0] + PARTICLE_SPEED[1] * i) % frame.shape[0]
        xs = (made["pts"][1] + PARTICLE_SPEED[0] * i) % frame.shape[1]
        out = frame.copy()
        out[ys, xs] = made["val"]
        return out

    return f


def _cursor(rng: np.random.Generator) -> Callable[[np.ndarray, int, Any], np.ndarray]:
    """Мелкий объект, движущийся независимо от всего остального."""
    def f(frame: np.ndarray, i: int, domain: Any) -> np.ndarray:
        h, w = frame.shape[:2]
        # Траектория своя и ни с чем не совпадающая: шаги взаимно простые с размерами,
        # чтобы курсор не оказался неподвижным по модулю кадра.
        y = (13 * i) % max(1, h - CURSOR_SIDE)
        x = (7 * i) % max(1, w - CURSOR_SIDE)
        out = frame.copy()
        out[y:y + CURSOR_SIDE, x:x + CURSOR_SIDE] = 250
        return out

    return f


#: Порчи по именам причин из §18.2. Ключ — то, как причина названа в документе, чтобы
#: строку результата можно было сопоставить с обоснованием ожидания без догадок.
PENALTIES: dict[str, Callable[[np.random.Generator], Callable[..., np.ndarray]]] = {
    "компрессия и дизеринг": _compression,
    "полупрозрачная панель": _translucent,
    "частицы и эффекты": _particles,
    "курсор": _cursor,
}


def _all(rng: np.random.Generator) -> Callable[[np.ndarray, int, Any], np.ndarray]:
    """Все четыре сразу. Это и есть оценка живого материала: причины не по очереди."""
    parts = [make(np.random.default_rng(int(rng.integers(0, 1 << 31))))
             for make in PENALTIES.values()]

    def f(frame: np.ndarray, i: int, domain: Any) -> np.ndarray:
        for p in parts:
            frame = p(frame, i, domain)
        return frame

    return f


# ---------------------------------------------------------------------------
# Замер
# ---------------------------------------------------------------------------

def measure(*, seeds: list[int], frames: int) -> dict[str, Any]:
    from harness.benchmark import bench_domain
    from harness.corpus.domains import DOMAINS

    names = sorted(DOMAINS)
    rows: list[dict[str, Any]] = []
    for name in names:
        # Чистое число того же кода на том же домене — знаменатель падения. Берём его
        # здесь, а не из записанного в MEASUREMENT.md: сравнивать надо с тем, что этот
        # прогон даёт сам, иначе падение смешается с расхождением прогонов.
        clean_row = bench_domain(name, seed=seeds[0], frames=frames,
                                 babble_steps=1).as_dict()
        clean = clean_row["iou"]
        print(f"{name:<10} чистый IoU {clean} (признак: {clean_row['signal']})", flush=True)
        for label, make in list(PENALTIES.items()) + [("все четыре", _all)]:
            got = []
            for s in seeds:
                rng = np.random.default_rng(1000 + s)
                d = bench_domain(name, seed=seeds[0], frames=frames, babble_steps=1,
                                 distort=make(rng)).as_dict()
                got.append(d["iou"])
            vals = [v for v in got if v is not None]
            rows.append({
                "domain": name, "penalty": label, "clean_iou": clean,
                # Какой признак выиграл. Без этого поля рост IoU под порчей выглядел бы
                # чудом; с ним видно, что порча иногда просто меняет победивший признак.
                "clean_signal": clean_row["signal"], "signal": d["signal"],
                "iou": None if not vals else round(statistics.median(vals), 4),
                "noise_min": None if not vals else round(min(vals), 4),
                "noise_max": None if not vals else round(max(vals), 4),
                "noise_seeds": len(vals),
            })
            r = rows[-1]
            print(f"  {label:<24} IoU {r['iou']} "
                  f"(шум по {r['noise_seeds']} сидам: {r['noise_min']}…{r['noise_max']}; "
                  f"признак: {r['signal']})", flush=True)

    out: dict[str, Any] = {
        "unit": "домен", "n": len(names), "frames": frames,
        "noise_seeds": len(seeds),
        "rows": rows,
        "note": "единица независимости — домен; сиды меняют только шум порчи, потому что "
                "маска истины от сида не зависит (MEASUREMENT.md, §2)",
    }

    by_penalty: dict[str, Any] = {}
    for label in list(PENALTIES) + ["все четыре"]:
        mine = [r for r in rows if r["penalty"] == label]
        vals = [r["iou"] for r in mine if r["iou"] is not None]
        clean = [r["clean_iou"] for r in mine if r["clean_iou"] is not None]
        # Доля от чистого, а не разность, — и это не оформление. Чистое число этого
        # прогона (140 кадров) не равно записанному в MEASUREMENT.md (240 кадров): у
        # `bench_domain` результат зависит от числа кадров. Разность, снятая здесь,
        # переносилась бы на опорное число неверно; доля переносится.
        ratios = [r["iou"] / r["clean_iou"] for r in mine
                  if r["iou"] is not None and r["clean_iou"]]
        if not vals:
            continue
        by_penalty[label] = {
            "median_iou": round(statistics.median(vals), 4),
            "median_clean": round(statistics.median(clean), 4),
            "drop": round(statistics.median(clean) - statistics.median(vals), 4),
            "median_ratio": None if not ratios else round(statistics.median(ratios), 4),
            "ratio_min": None if not ratios else round(min(ratios), 4),
            "ratio_max": None if not ratios else round(max(ratios), 4),
            "n": len(vals), "unit": "домен",
            "min": round(min(vals), 4), "max": round(max(vals), 4),
        }
    out["by_penalty"] = by_penalty

    # Что из этого следует для ожидания. Перенос на опорное число — долей, и опорное
    # берётся из единственного источника, а не литералом.
    from harness.corpus.live import synthetic_reference

    ref = synthetic_reference()
    full = by_penalty.get("все четыре")
    if full and full["median_ratio"] is not None:
        out["expectation"] = {
            "reference": ref["value"],
            "reference_unit": ref["unit"], "reference_n": ref["n"],
            "ratio": full["median_ratio"],
            "ratio_range": [full["ratio_min"], full["ratio_max"]],
            "value": round(ref["value"] * full["median_ratio"], 3),
            "unit": "домен", "n": full["n"],
            "note": "ожидание = опорное число × доля, оставшаяся под всеми четырьмя "
                    "порчами сразу. Это граница сверху: живой экран портит картинку и "
                    "способами, которых в списке четырёх нет",
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3,
                    help="сколько сидов шума порчи на домен (единица — домен, не сид)")
    ap.add_argument("--frames", type=int, default=140)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    res = measure(seeds=list(range(a.seeds)), frames=a.frames)
    print()
    print(f"Единица независимости: {res['unit']}, n = {res['n']}. "
          f"Сидов шума на домен: {res['noise_seeds']}, кадров {res['frames']}.")
    print()
    print(f"{'порча':<24} {'медиана IoU':>12} {'чистый':>8} {'доля':>7}  доля по доменам")
    for label, v in res["by_penalty"].items():
        ratio = "—" if v["median_ratio"] is None else f"{v['median_ratio']:.3f}"
        rng = ("" if v["ratio_min"] is None
               else f"  {v['ratio_min']:.3f}…{v['ratio_max']:.3f}")
        print(f"{label:<24} {v['median_iou']:>12.3f} {v['median_clean']:>8.3f} "
              f"{ratio:>7}{rng}")
    exp = res.get("expectation")
    if exp:
        print()
        print(f"Ожидание по живому: **{exp['value']:.2f}** = опорное {exp['reference']:.3f} "
              f"({exp['reference_unit']}, n={exp['reference_n']}) × доля {exp['ratio']:.3f}, "
              f"оставшаяся под всеми четырьмя порчами сразу "
              f"(медиана по {exp['n']} доменам, единица — домен).")
        print("Это граница сверху: живой экран портит картинку и способами, которых в "
              "списке четырёх нет.")
    if a.json:
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nЗаписано: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
