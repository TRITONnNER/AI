"""Детектор режима среды против объявленной истины домена. TASK-32, направление A.

Кросс-доменный замер дал IoU 0.196–0.815 и читался как «метод где-то работает хуже».
Читался неверно: параллакс в мире без камеры **неприменим**, а не плох. Здесь детектор
режима определяет наблюдением четыре признака, сверяется с истиной каждого домена и
показывает, что делается с разбросом, когда неприменимые ответы не идут в сводку.

## Истина по доменам объявлена здесь, а не выведена

Иначе сверять было бы не с чем. Истина берётся из **кода** домена — какие выходы к чему
подключены и что делает `advance`, — а не из представления о том, «что это за мир».
Таблица `TRUTH` ниже; она была неверна дважды, и оба раза поправил прогон, а не чтение:
обрамление анимировано у всех пяти доменов, а перемотка видео translates картинку, то есть
даёт согласованный сдвиг от собственного действия.

Внешнего судьи нет ни у одного, и поэтому признак `внешний судья` **не проверяется**:
детектор, который всегда отвечает «не определено», не подтверждён ничем (инвариант 27).
Это сказано числом «покрытие 3 признака из 4», а не тишиной.

## Обе ошибки (инвариант 32)

По каждому признаку:

- **ложная тревога** — признак объявлен там, где его в устройстве домена нет;
- **ложное подтверждение** — признак объявлен отсутствующим там, где он есть. Опаснее:
  «параллакс неприменим» закрывает механизм, и никто больше не проверит, был ли он нужен.

Прогон: `python3 tools/measure_regime.py`. Результат — `docs/measurements/regime.json`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.core.profile import from_schema                           # noqa: E402
from harness.corpus.domains import DOMAINS, make_domain                # noqa: E402
from harness.perception.regime import (EGO_MOTION, JUDGE, NO_BASELINE,  # noqa: E402
                                       REVERSIBLE, WORLD_ALONE, detect, report)

#: Истина по устройству домена — **по коду домена**, а не по догадке о том, «что это за
#: мир». Первая редакция этой таблицы была неверна дважды, и оба раза поправил прогон:
#:
#: 1. «Мир сам не идёт» стояло у четырёх доменов из пяти. Неверно: у всех пяти обрамление
#:    анимировано на 10–20 % кадра. Признак поэтому определён относительно (изменение при
#:    бездействии против вызванного мной), и по этой мерке идёт сам только проигрыватель.
#: 2. «У видео камеры нет» стояло как «эго-движения нет». Неверно наблюдательно: у
#:    проигрывателя есть перемотка, и она translates всю картинку — согласованный сдвиг
#:    от собственного действия есть. Различить «двигаю себя» и «двигаю содержимое» нельзя
#:    наблюдением, и признак это честно отражает.
TRUTH: dict[str, dict[str, bool | None]] = {
    # выходы двигают камеру (`cam_x`), сам мир не идёт: `advance` не переопределён
    "game":     {EGO_MOTION: True, WORLD_ALONE: False, REVERSIBLE: True,  JUDGE: None},
    # прокрутка по одной оси; обратной пары у прокрутки нет — вниз есть, вверх нет
    "document": {EGO_MOTION: True, WORLD_ALONE: False, REVERSIBLE: False, JUDGE: None},
    # окна двигаются мышью; парного «вернуть» среди выходов нет
    "desktop":  {EGO_MOTION: True, WORLD_ALONE: False, REVERSIBLE: False, JUDGE: None},
    # перемотка даёт согласованный сдвиг, содержимое идёт само каждый кадр,
    # «перемотать назад» возвращает вид
    "video":    {EGO_MOTION: True, WORLD_ALONE: True,  REVERSIBLE: True,  JUDGE: None},
    # три плана двигаются с разной скоростью при движении наблюдателя
    "depth":    {EGO_MOTION: True, WORLD_ALONE: False, REVERSIBLE: True,  JUDGE: None},
}
SEEDS = (1, 2, 3)
#: Разброс IoU параллакса по доменам из кросс-доменного замера (`harness bench`). Числа
#: перенесены сюда, а не пересчитаны: пересчёт был бы вторым прогоном того же, а нужно
#: показать, что делает с **этим** разбросом отказ отвечать там, где вопроса нет.
BENCH_PARALLAX_IOU: dict[str, float] = {
    "game": 0.815, "document": 0.762, "desktop": 0.735, "video": 0.196, "depth": 0.417,
}


def one(domain: str, *, seed: int, reference: str = "criterion",
        with_baseline: bool = True) -> dict[str, Any]:
    """Один прогон детектора по домену.

    `with_baseline=False` — тот же домен, но **без записи неподвижности**: так проверяется
    главное требование TASK-33 B. Признак, оставшийся без опорного уровня, обязан отвечать
    «не определено», а не подбирать порог.
    """
    prof = from_schema("РЕЖИМ", capture_width=160, capture_height=120,
                       regime_view_reference=reference)
    world = make_domain(domain, prof, seed=seed)

    def step(action: Any) -> np.ndarray:
        obs = world.step(action, with_audio=False)
        return np.asarray(obs.frame, dtype=np.float64)

    regime = detect(step=step, outputs=list(world.outputs), profile=prof,
                    baseline=None if with_baseline else NO_BASELINE)
    got = report(regime)
    got.update({"domain": domain, "seed": seed, "reference": reference,
                "with_baseline": with_baseline})
    return got


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="детектор режима среды")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "regime.json")
    a = ap.parse_args(argv)

    rows = [one(d, seed=s) for d in DOMAINS for s in SEEDS]
    # Тот же набор доменов без записи неподвижности, и три отсчёта для «вид вернулся».
    blind = [one(d, seed=s, with_baseline=False) for d in DOMAINS for s in SEEDS]
    refs = {ref: [one(d, seed=s, reference=ref) for d in DOMAINS for s in SEEDS]
            for ref in ("background", "looser")}
    refs["criterion"] = rows

    # Признак считается определённым по домену большинством сидов: детектор — прибор, и
    # один прогон у прибора бывает шумным. Единица независимости — **домен**: три сида
    # одного домена проверяют один и тот же устройственный факт.
    by_domain: dict[str, Any] = {}
    for d in DOMAINS:
        mine = [r for r in rows if r["domain"] == d]
        feats: dict[str, Any] = {}
        for f in (EGO_MOTION, WORLD_ALONE, REVERSIBLE, JUDGE):
            vals = [r["features"][f]["value"] for r in mine]
            known = [v for v in vals if v is not None]
            feats[f] = (None if not known
                        else sum(1 for v in known if v) > len(known) / 2)
        by_domain[d] = {
            "features": feats, "truth": TRUTH[d], "n": len(mine), "unit": "домен",
            "applicable": mine[0]["applicable"],
            "inapplicable": mine[0]["inapplicable"],
            "undetermined": mine[0]["undetermined"],
        }

    # Обе ошибки по каждому признаку. Клетки с истиной `None` не участвуют: неизвестное
    # не бывает ни ложной тревогой, ни ложным подтверждением.
    errors: dict[str, Any] = {}
    for f in (EGO_MOTION, WORLD_ALONE, REVERSIBLE, JUDGE):
        checked = [(d, by_domain[d]["features"][f], TRUTH[d][f]) for d in DOMAINS
                   if TRUTH[d][f] is not None]
        # Доля считается по доменам, где признак **ответил**. Абстиненция — не ошибка, но
        # и не подтверждение: держать её в знаменателе значило бы улучшать долю тем, что
        # проверка промолчала (инвариант 31 — покрытие предъявляется отдельно от доли).
        answered = [(d, got, true) for d, got, true in checked if got is not None]
        silent = [d for d, got, true in checked if got is None]
        alarm = [d for d, got, true in answered if got and not true]
        confirm = [d for d, got, true in answered if got is False and true]
        errors[f] = {
            "checked": len(checked), "answered": len(answered), "unit": "домен",
            "undetermined_where": silent,
            "false_alarm": len(alarm), "false_alarm_where": alarm,
            "false_confirm": len(confirm), "false_confirm_where": confirm,
            "share_alarm": (len(alarm) / len(answered)) if answered else None,
            "share_confirm": (len(confirm) / len(answered)) if answered else None,
            "how_alarm": "признак объявлен там, где его в устройстве домена нет",
            "how_confirm": ("признак объявлен отсутствующим там, где он есть — опаснее: "
                            "закрывает механизм, и никто больше не проверит"),
        }
    # TASK-33 B: что делает отсутствие записи неподвижности. Обе ошибки этой проверки:
    #
    # - **ложная тревога** — признак ответил «да»/«нет» без опорного уровня, то есть
    #   сравнил с нулём или с чужим порогом. Опаснее для вывода: молча даёт число;
    #   - **ложное подтверждение** — признак сказал «не определено» там, где запись
    #   неподвижности есть и её хватает. Теряет информацию.
    needs_base = (EGO_MOTION, WORLD_ALONE, REVERSIBLE)
    answered_blind = [(r["domain"], f) for r in blind for f in needs_base
                      if r["features"][f]["value"] is not None]
    silent_with_base = [(r["domain"], f) for r in rows for f in needs_base
                        if r["features"][f]["value"] is None
                        and f != REVERSIBLE]
    baseline_check = {
        "cells_blind": len(blind) * len(needs_base),
        "cells_with_base": len(rows) * len(needs_base),
        "unit": "признак × прогон",
        "answered_without_baseline": len(answered_blind),
        "false_alarm_share": len(answered_blind) / max(1, len(blind) * len(needs_base)),
        "silent_with_baseline": len(silent_with_base),
        "false_confirm_share": (len(silent_with_base)
                                / max(1, len(rows) * len(needs_base))),
        "where_silent": sorted({f"{d}/{f}" for d, f in silent_with_base}),
        "how_alarm": "признак ответил без опорного уровня — то есть сравнил с нулём",
        "how_confirm": ("признак промолчал там, где запись неподвижности есть и её "
                        "хватает; обратимость исключена: у неё «не определено» бывает и "
                        "по другой причине — ни одно действие не изменило вид"),
    }

    # Три отсчёта для «вид вернулся», обе ошибки у каждого. Выбор объявлен структурной
    # настройкой, поэтому сравнение воспроизводимо, а не остаётся утверждением в прозе.
    references: dict[str, Any] = {}
    for ref, rr in refs.items():
        per_dom: dict[str, Any] = {}
        for d in DOMAINS:
            vals = [r["features"][REVERSIBLE]["value"] for r in rr if r["domain"] == d]
            known = [v for v in vals if v is not None]
            per_dom[d] = (None if not known
                          else sum(1 for v in known if v) > len(known) / 2)
        checked = [(d, per_dom[d], TRUTH[d][REVERSIBLE]) for d in DOMAINS
                   if TRUTH[d][REVERSIBLE] is not None]
        alarm = [d for d, got_v, true in checked if got_v and not true]
        confirm = [d for d, got_v, true in checked if got_v is False and true]
        references[ref] = {
            "by_domain": per_dom, "checked": len(checked), "unit": "домен",
            "false_alarm": len(alarm), "false_alarm_where": alarm,
            "false_confirm": len(confirm), "false_confirm_where": confirm,
        }

    coverage = {
        "features_declared": 4,
        "features_checked": sum(1 for f in errors if errors[f]["checked"]),
        "why": ("внешний судья не проверен ни на одном домене: канала оценки нет ни у "
                "одного, и детектор, всегда отвечающий «не определено», не подтверждён "
                "ничем (инвариант 27). Станет измеримым с доменом, у которого есть "
                "канал оценки — направление D"),
    }

    # Что делает с разбросом отказ отвечать там, где вопроса нет.
    where_ok = [d for d in DOMAINS
                if by_domain[d]["features"][EGO_MOTION]
                and by_domain[d]["features"][WORLD_ALONE] is False]
    all_iou = sorted(BENCH_PARALLAX_IOU.values())
    ok_iou = sorted(BENCH_PARALLAX_IOU[d] for d in where_ok)
    spread = {
        "all": {"domains": len(all_iou), "min": min(all_iou), "max": max(all_iou),
                "median": statistics.median(all_iou),
                "range": round(max(all_iou) - min(all_iou), 3)},
        "applicable_only": {"domains": len(ok_iou), "min": min(ok_iou),
                            "max": max(ok_iou), "median": statistics.median(ok_iou),
                            "range": round(max(ok_iou) - min(ok_iou), 3),
                            "which": where_ok},
        "unit": "домен",
    }

    print(f"{'домен':<10} {'эго-движение':>26} {'мир без меня':>26} "
          f"{'обратимость':>24} {'судья':>12}")
    for d in DOMAINS:
        cells = []
        for f in (EGO_MOTION, WORLD_ALONE, REVERSIBLE, JUDGE):
            got, true = by_domain[d]["features"][f], TRUTH[d][f]
            mark = {True: "да", False: "нет", None: "не опр."}[got]
            want = {True: "да", False: "нет", None: "—"}[true]
            cells.append(f"{mark} (истина {want})")
        print(f"{d:<10} {cells[0]:>26} {cells[1]:>26} {cells[2]:>24} {cells[3]:>12}")

    print()
    for d in DOMAINS:
        v = by_domain[d]
        print(f"{d:<10} применимо: {', '.join(v['applicable']) or '—'}")
        print(f"{'':<10} неприменимо: {', '.join(v['inapplicable']) or '—'}")
        print(f"{'':<10} не определено: {', '.join(v['undetermined']) or '—'}")

    print("\nбез записи неподвижности (TASK-33 B): признаков ответило "
          f"{baseline_check['answered_without_baseline']} из "
          f"{baseline_check['cells_blind']} — доля ложных срабатываний "
          f"{baseline_check['false_alarm_share']:.0%}; промолчало при снятом фоне "
          f"{baseline_check['silent_with_baseline']} из "
          f"{baseline_check['cells_with_base']} — доля ложных подтверждений "
          f"{baseline_check['false_confirm_share']:.0%}")
    # На чём держится ответ по домену: сколько прогонов из трёх вообще дали ответ. Без
    # этого числа «1 ошибка из 5 доменов» может стоять на одном прогоне из трёх, и это
    # ровно тот случай (инвариант 22: n считает единицы, а не события).
    print("\nустойчивость ответа по домену (обратимость), прогонов с ответом из "
          f"{len(SEEDS)}:")
    stability: dict[str, Any] = {}
    for d in DOMAINS:
        vals = [r["features"][REVERSIBLE]["value"] for r in rows if r["domain"] == d]
        known = [v for v in vals if v is not None]
        stability[d] = {"determined": len(known), "runs": len(vals),
                        "values": vals, "agree": len(set(known)) <= 1}
        mark = "" if (len(known) == len(vals) and stability[d]["agree"]) else "  !"
        print(f"  {d:<10} {len(known)} из {len(vals)}  ответы: "
              + ", ".join({True: "да", False: "нет", None: "не опр."}[v] for v in vals)
              + mark)

    print("\nотсчёт для «вид вернулся» (единица — домен, обратимость):")
    print(f"  {'отсчёт':<12} {'ложных тревог':>14} {'ложных подтверждений':>21}")
    for ref, v in references.items():
        print(f"  {ref:<12} {str(v['false_alarm']) + ' из ' + str(v['checked']):>14} "
              f"{str(v['false_confirm']) + ' из ' + str(v['checked']):>21}"
              + (f"   тревога: {', '.join(v['false_alarm_where'])}"
                 if v["false_alarm_where"] else "")
              + (f"   подтверждение: {', '.join(v['false_confirm_where'])}"
                 if v["false_confirm_where"] else ""))

    print("\nобе ошибки по признакам (единица — домен):")
    for f, e in errors.items():
        if not e["checked"]:
            print(f"  {f:<20} не проверен ни на одном домене")
            continue
        print(f"  {f:<20} ложных тревог {e['false_alarm']}/{e['answered']}, "
              f"ложных подтверждений {e['false_confirm']}/{e['answered']}"
              + (f", не определено на {len(e['undetermined_where'])} из {e['checked']}: "
                 + ", ".join(e["undetermined_where"])
                 if e["undetermined_where"] else "")
              + (f" — тревога на: {', '.join(e['false_alarm_where'])}"
                 if e["false_alarm_where"] else "")
              + (f" — подтверждение на: {', '.join(e['false_confirm_where'])}"
                 if e["false_confirm_where"] else ""))
    print(f"  покрытие: проверено признаков {coverage['features_checked']} из "
          f"{coverage['features_declared']}; доменов с ответом по каждому — в строках выше")

    print(f"\nразброс IoU параллакса по всем пяти доменам: "
          f"{spread['all']['min']:.3f}–{spread['all']['max']:.3f} "
          f"(размах {spread['all']['range']})")
    print(f"по тем, где параллакс применим ({', '.join(where_ok)}): "
          f"{spread['applicable_only']['min']:.3f}–"
          f"{spread['applicable_only']['max']:.3f} "
          f"(размах {spread['applicable_only']['range']})")

    data = {
        "rows": rows, "by_domain": by_domain, "errors": errors, "coverage": coverage,
        "baseline_check": baseline_check, "references": references,
        "reversible_stability": stability,
        "baseline_by_domain": {d: next(r["baseline"] for r in rows if r["domain"] == d)
                               for d in DOMAINS},
        "spread": spread, "truth": TRUTH, "seeds": list(SEEDS),
        "bench_parallax_iou": BENCH_PARALLAX_IOU, "unit": "домен",
        "claim": ("режим среды определяется наблюдением, и механизм, которому нужного "
                  "признака нет, докладывает «неприменимо» вместо числа; разброс IoU "
                  "0.196–0.815 объясняется неприменимостью, а не качеством метода"),
        "how_refuted": ("если детектор путает признаки на доменах с известным "
                        "устройством, режим определяется неверно, и гейт по нему сделает "
                        "хуже: закроет работающий механизм"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nзаписано: {a.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
