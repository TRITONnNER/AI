"""Кривая баланса разведки. `TASK-06`, часть 2. Долг, перенесённый трижды.

Замер по предрегистрации из `MEASUREMENT.md`, раздел 15, записанной и закоммиченной
**до** прогона. Читать её надо до чисел: три переноса одного долга — это ровно та
ситуация, в которой любой полученный результат объявляется ожидаемым.

## Что меряется

Доля шагов разведки, отданная замыканию петель (`explore_closing_share`), на пяти
значениях и трёх сидах, в **ограниченном** мире — том, на котором граф мест впервые
подтверждён количественно (TASK-05).

Три величины: открытых узлов, узлов «и за 2, и за 3», пар мест с двумя маршрутами.
Вторая — та, ради которой всё затевалось: пока подтверждённый подграф остаётся цепью,
он двудолен, и цель, достижимая за два шага, никогда не достижима за три.

## Единица независимости

**Прогон** (мир, сид), `n = 3` на каждое значение доли. Не шаг и не место: внутри
прогона всё зависимо — второе место открыто потому, что первое куда-то вело. Та же
ошибка, что дала 1913 «планов» на 94 маршрутах, здесь дала бы 1500 «наблюдений» на
одном прогоне.

Запуск: `python3 tools/measure_balance_curve.py [--out FILE]`
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.behaviour.babbling import Babbler, run_babbling      # noqa: E402
from harness.behaviour.planner import probe_chooser               # noqa: E402
from harness.core.action import Action, action_key                # noqa: E402
from harness.core.clocks import Clocks                            # noqa: E402
from harness.core.profile import from_schema                      # noqa: E402
from harness.corpus.world import InteractiveWorld                 # noqa: E402
from harness.model.forward import ForwardModel                    # noqa: E402
from harness.model.places import PlaceGraph                       # noqa: E402

SHARES = (0.0, 0.15, 0.3, 0.5, 1.0)
SEEDS = (11, 12, 13)
#: Расширение по числу сидов. Предрегистрированный прогон — три сида; расширение
#: докладывается **отдельно** и не подменяет его. Подменять было бы подгонкой: выбор
#: объёма после взгляда на числа — это выбор того объёма, который дал нужный ответ.
MORE_SEEDS = (11, 12, 13, 14, 15, 16, 17, 18)
STEPS = 1500
BABBLE_STEPS = 600
OUTPUTS = 16

# Отметки, на которых снимается прирост открытых узлов. Нужны для проверки на
# вакуумность (инвариант 27): если фронтир исчерпан к сотому шагу, кривая говорит не
# про баланс разведки, а про то, что разведывать было нечего с самого начала.
CHECKPOINTS = (100, 400, 1500)


def run(share: float, seed: int) -> dict[str, Any]:
    profile = from_schema(
        f"баланс-{share}", capture_width=320, capture_height=180,
        babble_repeats=3, place_record_loops=True, macro_max_length=0,
        explore_closing_share=share,
        # Тот самый мир, на котором граф подтверждён: ограниченный, с переменной ценой.
        world_bounded=True, world_variable_cost=True)
    min_n = int(profile.parameters["plan_min_step_n"])

    bw = InteractiveWorld(profile, seed=seed, n_outputs=OUTPUTS)
    bab = Babbler(profile, bw.outputs, rng_seed=seed)
    run_babbling(bw, bab, steps=BABBLE_STEPS, clocks=Clocks())
    inverse = dict(bab.inverse_found)

    world = InteractiveWorld(profile, seed=seed, n_outputs=OUTPUTS)
    graph = PlaceGraph.from_profile(profile)
    first = world.step(None, with_audio=False)
    graph.see(first.frame, 0, seconds_per_seq=1 / 30.0, mode="start")
    st: dict[str, Any] = {"seq": 0, "last": None}

    chooser = probe_chooser(profile)
    model = ForwardModel.from_graph(graph, bab.body)
    opened: dict[int, int] = {}
    for i in range(STEPS):
        if i % 25 == 0:
            model = ForwardModel.from_graph(graph, bab.body)
        out, ms = chooser(model, graph, graph.current, world.outputs,
                          hold_ms=200, min_n=min_n, inverse=inverse,
                          last_output=st["last"])
        obs = world.step(Action.key(out, ms), with_audio=False)
        st["seq"] += 1
        st["last"] = out
        # Время мира берётся у мира: в мире с переменной ценой шаг больше не тик.
        graph.see(obs.frame, st["seq"], seconds_per_seq=1 / 30.0,
                  mode=action_key(out, ms))
        if i + 1 in CHECKPOINTS:
            opened[i + 1] = len(graph)

    conf: dict[str, set[str]] = {}
    for e in graph.edges.values():
        if e.n >= min_n and e.dst != e.src:
            conf.setdefault(e.src, set()).add(e.dst)

    def at_depth(src: str, d: int) -> set[str]:
        cur = {src}
        for _ in range(d):
            cur = {y for x in cur for y in conf.get(x, ())}
        return cur

    both = sum(1 for src in conf if (at_depth(src, 2) & at_depth(src, 3)) - {src})

    multi = 0
    for src in conf:
        for dst in conf[src]:
            ways = [e for e in graph.edges.values()
                    if e.src == src and e.dst == dst and e.n >= min_n]
            if len(ways) > 1:
                multi += 1

    undirected: dict[str, set[str]] = {}
    for e in graph.edges.values():
        if e.n >= min_n and e.dst != e.src:
            undirected.setdefault(e.src, set()).add(e.dst)
            undirected.setdefault(e.dst, set()).add(e.src)
    deg = Counter(len(v) for v in undirected.values())

    counts = getattr(chooser, "counts", {"closing": 0, "deep": STEPS})
    return {"share": share, "seed": seed, "places": len(graph),
            "conf_nodes": len(undirected), "deg": dict(sorted(deg.items())),
            "both_2_and_3": both, "multi_route_pairs": multi,
            "opened_at": opened, "walls_hit": getattr(world, "walls_hit", None),
            "closing_steps": counts["closing"], "deep_steps": counts["deep"]}


def summarise(rows: list[dict[str, Any]],
              seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for share in SHARES:
        mine = [r for r in rows if r["share"] == share]
        out[str(share)] = {
            "n_runs": len(mine),
            "places": [r["places"] for r in mine],
            "places_median": statistics.median(r["places"] for r in mine),
            "both_2_and_3": [r["both_2_and_3"] for r in mine],
            "both_median": statistics.median(r["both_2_and_3"] for r in mine),
            "multi_route_pairs": [r["multi_route_pairs"] for r in mine],
            "multi_median": statistics.median(r["multi_route_pairs"] for r in mine),
            # Доля, дошедшая до выбора: без неё «кривая плоская» неотличимо от
            # «ручка не читается» (инвариант 26).
            "closing_share_actual": [
                round(r["closing_steps"] / max(1, r["closing_steps"] + r["deep_steps"]), 3)
                for r in mine],
            "opened_at": [r["opened_at"] for r in mine],
            # Медиана на трёх прогонах с нулями бесполезна: если два прогона из трёх
            # дают ноль при любой доле, медиана равна нулю всегда — и это свойство
            # статистики, а не мира. Поэтому рядом сумма и число ненулевых прогонов.
            "both_total": sum(r["both_2_and_3"] for r in mine),
            "both_nonzero": sum(1 for r in mine if r["both_2_and_3"] > 0),
            "multi_total": sum(r["multi_route_pairs"] for r in mine),
            "multi_nonzero": sum(1 for r in mine if r["multi_route_pairs"] > 0),
        }
    return out


def verdict(s: dict[str, Any]) -> list[str]:
    """Сверка с предрегистрацией. Пишется до чисел и не подгоняется под них."""
    lines: list[str] = []
    # Сумма, а не медиана: см. комментарий в summarise.
    both = {k: v["both_total"] for k, v in s.items()}
    places = {k: v["places_median"] for k, v in s.items()}
    multi = {k: v["multi_total"] for k, v in s.items()}

    best = max(both, key=lambda k: (both[k], -float(k)))
    lines.append(f"Оптимум по «и за 2, и за 3»: доля {best} (сумма {both[best]} "
                 f"по {s[best]['n_runs']} прогонам, ненулевых "
                 f"{s[best]['both_nonzero']}).")
    # Главная проверка: различает ли замер доли вообще. Если ненулевой результат
    # приходит с одного и того же сида при всех долях, то меряется сид, а не доля.
    nonzero_runs = {k: v["both_nonzero"] for k, v in s.items()}
    if max(nonzero_runs.values()) <= 1:
        lines.append(
            "Ненулевой результат приходит не более чем от одного прогона на "
            "значение доли. При такой разреженности кривая не локализуется: "
            "различие между сидами больше различия между долями, и оптимум — "
            "это имя самого удачного сида, а не свойство доли.")

    # 1. Ожидание: открытых узлов монотонно падает с ростом доли.
    seq = [places[str(x)] for x in SHARES]
    falls = all(b <= a for a, b in zip(seq, seq[1:]))
    lines.append("Открытых узлов монотонно падает: "
                 + ("да, как ожидалось" if falls else f"нет — {seq}"))

    # 2. Ожидание: оптимум ближе к нулю, ориентировочно 0.15–0.3.
    if both[str(0.0)] == max(both.values()):
        lines.append("Оптимум на нуле: замыкание не дало ничего — против ожидания, "
                     "требует объяснения.")
    elif float(best) <= 0.3:
        lines.append("Оптимум в ожидаемой области 0.15–0.3.")
    elif float(best) == 1.0:
        lines.append("Оптимум на 1.0 — против ожидания. По предрегистрации самое "
                     "вероятное объяснение: фронтира нет с самого начала, и тогда "
                     "замер вакуумен по свойству мира, а не информативен. Проверка "
                     "ниже, по приросту открытых узлов после сотого шага.")
    else:
        lines.append(f"Оптимум на {best} — выше ожидаемой области 0.15–0.3.")

    # 3. Проверка на вакуумность: исчерпан ли фронтир к сотому шагу при доле 0.
    zero = s[str(0.0)]["opened_at"]
    growth = [(o[str(1500)] if str(1500) in o else o[1500])
              - (o[str(100)] if str(100) in o else o[100]) for o in zero]
    lines.append(f"Прирост открытых узлов при доле 0 со 100-го по 1500-й шаг: "
                 f"{growth}. "
                 + ("Фронтир исчерпан почти сразу — замер вакуумен по свойству мира "
                    "(инвариант 27)." if max(growth) <= 1 else
                    "Фронтир был — замер не вакуумен."))

    # 4. Плоская кривая: ручка не читается или мир не отвечает.
    flat = len(set(both.values())) == 1 and len(set(multi.values())) == 1
    if flat:
        actual = s[str(0.5)]["closing_share_actual"]
        lines.append(f"Все величины одинаковы при всех долях. Фактическая доля "
                     f"замыкающих шагов при 0.5: {actual}. "
                     + ("Доля до выбора не доходит — дефект, а не свойство мира."
                        if max(actual) < 0.1 else
                        "Доля доходит, значит одинаковость — свойство мира."))
    return lines


def report(s: dict[str, Any]) -> str:
    lines = ["# Кривая баланса разведки", "",
             f"Единица независимости — прогон (мир, сид), n = "
             f"{s[str(SHARES[0])]['n_runs']} на "
             f"значение доли. Бюджет {STEPS} шагов, мир ограниченный.", "",
             "| доля | открытых узлов, медиана | «и за 2, и за 3»: сумма / "
             "прогонов с ненулём | пар с двумя маршрутами: сумма / с ненулём |",
             "|---|---|---|---|"]
    for share in SHARES:
        v = s[str(share)]
        lines.append(
            f"| {share} | {v['places_median']:.0f} | "
            f"{v['both_total']} / {v['both_nonzero']} из {v['n_runs']} | "
            f"{v['multi_total']} / {v['multi_nonzero']} из {v['n_runs']} |")
    lines += ["", "## Сверка с предрегистрацией", ""]
    lines += [f"- {x}" for x in verdict(s)]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--more-seeds", action="store_true",
                    help="расширение до восьми сидов; докладывается отдельно")
    args = ap.parse_args()
    seeds = MORE_SEEDS if args.more_seeds else SEEDS
    rows = []
    for share in SHARES:
        for seed in seeds:
            r = run(share, seed)
            rows.append(r)
            print(f"доля {share:<5} сид {seed}: мест {r['places']:>3}, "
                  f"«и за 2, и за 3» {r['both_2_and_3']:>2}, "
                  f"пар {r['multi_route_pairs']:>2}, "
                  f"замыкающих шагов {r['closing_steps']}", flush=True)
    s = summarise(rows, seeds)
    print()
    print(report(s))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"runs": rows, "summary": s},
                                       ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\nчисла: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
