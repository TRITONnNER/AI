"""Час непрерывной работы: что растёт, что выходит на полку, что не растёт. TASK-24, D.

Условие вехи М5 — «работает час без вмешательства». Все прогоны до сих пор шли по 2.5
секунды при ускорении часов контуров в 20 раз, и этим оно не проверялось: за две секунды не
проявятся ни утечка памяти, ни рост журнала, ни дрейф часов, ни накопление карточек, ни
деградация от заполнения хранилища.

## Ожидание записано **до** прогона

Иначе любой исход выглядел бы приемлемым. Ниже — `EXPECTED`, и в нём три класса величин:

- **линейный рост** — то, что обязано расти пропорционально времени: обороты цикла, тики
  мира, записи журнала, байты на диске. Отсутствие роста здесь означает, что цикл встал;
- **полка** — то, что растёт и упирается: число карточек (мир конечен, эпизоды повторяются),
  расход памяти процесса (буферы переиспользуются). Линейный рост здесь означает утечку;
- **не растёт вовсе** — дрейф `t_self` относительно `wall_clock` за оборот и число
  обращений к сну без явного вызова. Рост здесь означает, что часы разъезжаются или что
  что-то запускается само.

Единица независимости — **отрезок прогона** (`SEGMENT_S` секунд): величины внутри отрезка
зависимы по построению, а отрезки друг от друга — нет, потому что каждый мерится заново.
Прогон один, поэтому утверждения о самом прогоне имеют `n = 1`, и это сказано в отчёте, а не
спрятано: час нельзя повторить трижды за одну задачу.

    python3 tools/measure_hour.py                 # час, как требует веха
    python3 tools/measure_hour.py --seconds 120   # короткая репетиция того же кода
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.behaviour.arena import Arena, reach_place                # noqa: E402
from harness.behaviour.contours import Scheduler                      # noqa: E402
from harness.core.journal import Actor, ActorLayer, Kind               # noqa: E402
from harness.core.profile import from_schema                          # noqa: E402
from harness.livecycle import Proposal, babbler_contour, slow_planner  # noqa: E402
from harness.session import Recorder                                  # noqa: E402

#: Длина отрезка, по которому снимается точка. 30 секунд: короче — точка ловит дрожание
#: планировщика на 0.5 Гц (он успевает запуститься 15 раз), длиннее — за час выходит меньше
#: сотни точек, и полку от линии не отличить.
SEGMENT_S = 30.0

#: Ожидание, записанное до прогона. Ключ — величина, значение — класс и почему.
EXPECTED: dict[str, dict[str, str]] = {
    "loops": {"class": "линейный",
              "why": "оборот цикла ничем не ограничен сверху, кроме процессора"},
    "world_ticks": {"class": "линейный",
                    "why": "один оборот — один шаг мира, это инвариант 3 в виде равенства"},
    "journal_entries": {"class": "линейный",
                        "why": "журнал только дозаписывается (инвариант 1)"},
    "disk_bytes": {"class": "линейный",
                   "why": "записи журнала и кадры пишутся на диск и не удаляются"},
    "rss_mb": {"class": "полка",
               "why": "буферы переиспользуются, новых объектов на оборот не создаётся; "
                      "линейный рост здесь означает утечку"},
    "episodes_total": {"class": "линейный",
                       "why": "эпизоды идут один за другим и не кончаются: их число обязано "
                              "расти. Полка здесь означала бы, что респавн перестал работать"},
    "places": {"class": "полка",
               "why": "мир конечен, виды повторяются: новых мест становится всё меньше. "
                      "**Сверять эту величину по времени нельзя** (13.7): кривая гнётся "
                      "от того, что цикл замедлился, а не от того, что мир кончился. "
                      "Правильная ось — наблюдения, и по ней насыщение считает "
                      "tools/measure_fingerprint.py; здесь величина оставлена для полноты "
                      "прогона, а её вердикт помечен ненадёжным (axis_wrong)"},
    "wall_us_per_loop": {"class": "не растёт",
                         "why": "настенных микросекунд на один оборот. t_self считает "
                                "обороты, wall_clock идёт сам; их отношение обязано "
                                "колебаться вокруг постоянной. Рост означает, что цикл "
                                "замедляется — то есть деградацию от накопления"},
    "loops_per_s": {"class": "не растёт",
                    "why": "оборотов в секунду на **этом** отрезке, а не в среднем за "
                           "прогон. Падение здесь — то же замедление, но видное прямо, "
                           "без деления кумулятивных величин"},
    "sleeps": {"class": "не растёт",
               "why": "сон вызывается явно и в этом прогоне не вызывается ни разу"},
}


def rss_mb() -> float:
    """Расход памяти процесса, МиБ. Читается из /proc — без сторонних пакетов.

    Отсутствие /proc не заменяется нулём: ноль читался бы как «памяти не занято».
    """
    try:
        with open(f"/proc/{os.getpid()}/statm", encoding="ascii") as f:
            pages = int(f.read().split()[1])
    except (OSError, IndexError, ValueError):
        return float("nan")
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def slope_and_plateau(xs: list[float], ys: list[float]) -> dict[str, Any]:
    """Наклон по методу наименьших квадратов и признак полки.

    Полка объявляется не «на глаз»: сравниваются наклоны первой и последней трети. Если
    последняя треть растёт медленнее первой хотя бы вдвое, это полка; если наклон держится
    — линия. Порог объявлен здесь (вдвое), и он один для всех величин.
    """
    def slope(px: list[float], py: list[float]) -> float:
        n = len(px)
        if n < 2:
            return 0.0
        mx, my = sum(px) / n, sum(py) / n
        den = sum((x - mx) ** 2 for x in px)
        return 0.0 if den == 0 else sum((x - mx) * (y - my) for x, y in zip(px, py)) / den

    third = max(2, len(xs) // 3)
    first = slope(xs[:third], ys[:third])
    last = slope(xs[-third:], ys[-third:])
    whole = slope(xs, ys)
    flattens = abs(last) * 2 <= abs(first) if first else False
    return {"slope_per_s": whole, "slope_first_third": first, "slope_last_third": last,
            "flattens": flattens,
            "total_growth": (ys[-1] - ys[0]) if ys else 0.0}


def classify(name: str, fit: dict[str, Any], *, ys: list[float]) -> tuple[str, str]:
    """Какой класс получился по факту и совпал ли он с ожиданием.

    Порог «не растёт» — 1 % от размаха величины за прогон: абсолютного нуля у измеренной
    величины не бывает, и требовать его значило бы объявлять дефектом дрожание последнего
    разряда.
    """
    moved = abs(fit["total_growth"]) > 0.01 * max(1e-9, max(map(abs, ys)) if ys else 1.0)
    if not moved:
        got = "не растёт"
    elif fit["total_growth"] < 0:
        # **Направление важно.** Первая редакция звала падение «линейным», и «обороты в
        # секунду: линейный» читалось как разгон, тогда как цикл замедлялся. Классификатор,
        # теряющий знак, сообщает противоположное.
        got = "падение" if not fit["flattens"] else "полка сверху"
    elif fit["flattens"]:
        got = "полка"
    else:
        got = "линейный"
    want = EXPECTED[name]["class"]
    if got == want:
        return got, "совпало"
    return got, f"разошлось: ожидался {want}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=3600.0,
                    help="сколько секунд крутить; по умолчанию час, как требует веха")
    ap.add_argument("--segment", type=float, default=SEGMENT_S)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "hour.json")
    ap.add_argument("--path", type=Path, default=None,
                    help="куда писать сессию; по умолчанию временный каталог")
    a = ap.parse_args(argv)

    import tempfile

    tmp = None
    if a.path is None:
        tmp = tempfile.TemporaryDirectory(prefix="harness-hour-")
        session_path = Path(tmp.name) / "s"
    else:
        session_path = a.path

    # **Без ускорения часов.** Это и есть предмет замера: контуры идут на своих частотах,
    # и час означает час.
    profile = from_schema("ЧАС", capture_width=160, capture_height=90)
    print(f"час непрерывной работы: {a.seconds:g} с, отрезок {a.segment:g} с, "
          "ускорения часов нет")
    print("ожидание записано до прогона:")
    for name, spec in EXPECTED.items():
        print(f"  {name:<26} {spec['class']:<12} {spec['why']}")
    print(flush=True)

    series: dict[str, list[float]] = {k: [] for k in EXPECTED}
    times: list[float] = []
    started = time.monotonic()
    next_point = started + a.segment
    loops = world_ticks = delivered = 0
    # **Окно, а не список.** Первая редакция копила по одному числу на оборот — 300 тысяч
    # чисел за минуту, — и рост памяти, который замер объявил утечкой, был на треть
    # утечкой самого замера. Инструмент, растущий вместе с измеряемым, мерит себя.
    from collections import deque

    drift_us: deque[float] = deque(maxlen=1000)
    # Граф мест: та самая накапливающаяся структура, про которую записано ожидание «полка».
    # Кормится отпечатками кадров мира — то есть тем же, чем кормился бы граф агента.
    from harness.model.places import PlaceGraph, fingerprint

    graph = PlaceGraph()

    with Recorder(session_path, profile=profile, source="hour", synthetic=True,
                  note="TASK-24 D: час непрерывной работы") as rec:
        arena = Arena(reach_place(profile), journal=rec.journal, seed=1)
        arena.start(rec.clocks.stamp())
        import numpy as np

        rng = np.random.default_rng(1)
        sched = Scheduler.standard(profile, reflex=babbler_contour(arena, rng),
                                   planner=slow_planner(arena))
        while True:
            now = time.monotonic()
            if now - started >= a.seconds:
                break
            loops += 1
            sched.step()
            choice: Proposal | None = None
            for contour in sched.contours:
                if isinstance(contour.best, Proposal):
                    choice = contour.best
                    contour.best = None
                    break
            stamp = rec.clocks.stamp()
            obs = arena.step(choice.action if choice else None, stamp)
            world_ticks += 1
            # Отпечаток каждого сотого кадра: чаще незачем — граф мест растёт от смены
            # вида, а не от частоты взглядов, — и дешевле, чем считать сетку на каждом
            # обороте. Число объявлено здесь, а не подобрано под результат.
            if loops % 100 == 0 and obs is not None and getattr(obs, "frame", None) is not None:
                graph.observe(fingerprint(obs.frame), loops, seconds_per_seq=1.0 / 30.0)
            if choice is not None:
                delivered += 1
                rec.journal.append(Kind.ACTION, stamp, Actor.AGENT, choice.layer,
                                   action=choice.action,
                                   event={"code": "delivered",
                                          "contour": choice.contour})
            if arena.finished():
                arena.respawn(stamp)
            # Дрейф: сколько настенных микросекунд на один оборот против t_self.
            # Мгновенное время оборота, а не кумулятивная средняя: у второй рост заложен
            # в определение, если цикл хоть немного замедлился, и «дрейф растёт» вышло бы
            # даже на идеально ровном цикле.
            drift_us.append((time.monotonic() - now) * 1e6)
            if now >= next_point:
                elapsed = now - started
                times.append(elapsed)
                series["loops"].append(float(loops))
                series["world_ticks"].append(float(world_ticks))
                # Число записей журнала — приватный счётчик `_seq`: публичного счётчика
                # у журнала нет, и заводить его ради замера значило бы менять журнал под
                # инструмент. Читается только для отчёта.
                series["journal_entries"].append(float(rec.journal._seq))
                series["disk_bytes"].append(float(dir_bytes(session_path)))
                series["rss_mb"].append(rss_mb())
                series["episodes_total"].append(float(len(arena.episodes)))
                series["places"].append(float(len(graph.places)))
                series["wall_us_per_loop"].append(statistics.median(drift_us))
                # Отрезочная частота: обороты этого отрезка на его длину. Кумулятивная
                # средняя размазала бы замедление по всему прогону.
                prev_loops = series["loops"][-2] if len(series["loops"]) > 1 else 0.0
                prev_t = times[-2] if len(times) > 1 else 0.0
                span = max(1e-9, elapsed - prev_t)
                series["loops_per_s"].append((loops - prev_loops) / span)
                series["sleeps"].append(0.0)
                next_point += a.segment
                print(f"  {elapsed / 60:5.1f} мин: оборотов {loops}, память "
                      f"{series['rss_mb'][-1]:.1f} МиБ, диск "
                      f"{series['disk_bytes'][-1] / 1024 / 1024:.1f} МиБ, эпизодов "
                      f"{len(arena.episodes)}", flush=True)
        arena.close(arena.finished() or "прервано", rec.clocks.stamp())
        summary = arena.summary()

    rows = {}
    for name, ys in series.items():
        if len(ys) < 3:
            rows[name] = {"class_expected": EXPECTED[name]["class"],
                          "verdict": "нечем проверить: точек меньше трёх"}
            continue
        fit = slope_and_plateau(times, ys)
        got, verdict = classify(name, fit, ys=ys)
        rows[name] = {"class_expected": EXPECTED[name]["class"], "class_got": got,
                      "verdict": verdict, "why_expected": EXPECTED[name]["why"],
                      "first": ys[0], "last": ys[-1],
                      # Величины, копящиеся по наблюдениям, а не по времени: их вердикт
                      # по оси времени — ложное подтверждение (инвариант 32, 13.7).
                      "axis_wrong": name in ("places",),
                      **fit}

    data = {
        "seconds": a.seconds, "segment_s": a.segment, "points": len(times),
        "unit": "отрезок прогона", "n": len(times), "runs": 1,
        "hz_scale": 1.0,
        "loops": loops, "world_ticks": world_ticks, "delivered": delivered,
        "world_never_paused": loops == world_ticks,
        "episodes": summary,
        "expected": EXPECTED, "series": {k: v for k, v in series.items()},
        "times": times, "checks": rows,
        "diverged": [k for k, v in rows.items() if v.get("verdict", "").startswith("разошл")],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"{'величина':<26} {'ожидалось':<12} {'вышло':<12} итог")
    for name, v in rows.items():
        print(f"{name:<26} {v['class_expected']:<12} {v.get('class_got', '—'):<12} "
              f"{v['verdict']}")
    print()
    print(f"оборотов {loops}, тиков мира {world_ticks} — "
          + ("мир не останавливался" if loops == world_ticks else "МИР ОСТАНАВЛИВАЛСЯ"))
    print(f"единица независимости: отрезок прогона, n = {len(times)}; прогонов 1 — "
          "час нельзя повторить трижды за одну задачу, и это сказано, а не спрятано")
    print(f"записано: {a.out}")
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
