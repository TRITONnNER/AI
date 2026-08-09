"""Доля дошедших планов: по длине и по силе слабейшего ребра.

Зачем отдельный инструмент. Этот замер уже дважды давал неверный вывод, и оба
раза причина была не в мире и не в планировщике, а в **единице наблюдения**.

Первый неверный вывод — «планы от трёх шагов не доходят, это предел отпечатка
вида как представления места». Он получился потому, что цель выбиралась как самая
далёкая достижимая, поэтому длина плана и расстояние до цели были одной и той же
переменной, а трёхшаговые цели встречались только на концах цепочки, где разведка
была меньше всего.

Второй неверный вывод — «доходит не длина, а сила слабейшего ребра». Он получился
из корзин по силе, написанных рукой, и из подсчёта каждой попытки как независимой.
Обе ошибки видны здесь, и вторая интереснее первой:

    сид 19 PLACE_0FD2|OUT_11C4@200|PLACE_F0EA
        попыток 60, дошло 0, сила: первая 12, последняя 12
    сид  5 PLACE_29CE|OUT_1AEB@200|PLACE_B321
        попыток 30, дошло 30, сила: первая 12, последняя 41

**У ребра, на котором план рвётся, сила не растёт никогда.** План верит, что это
нажатие ведёт в `PLACE_F0EA`; нажатие уводит в другое место; наблюдение достаётся
другому ребру, а то, в которое план верил, остаётся с прежним `n`. Приход
инкрементирует силу, отказ — нет. Значит сила ребра — **следствие** прихода, и
корреляция «сильные рёбра доходят» построена задом наперёд. Оба дошедших ребра
стартовали с той же силы 12–13, что и замороженное.

Поэтому здесь считаются две таблицы, а не одна:

- **все попытки** — так, как считали раньше; годится только для сравнения с
  прежними числами;
- **первая попытка на различный маршрут** — единица, в которой обратной связи
  между приходом и силой нет по построению.

Замер на 80 сидах (`python3 tools/measure_plan_arrival.py`):

    всего попыток 1913, различных маршрутов 94 (20.4 попытки на маршрут)
    сидов, давших хотя бы один план: 32 из 80

    длина │ все попытки │ первая попытка на маршрут
        1 │ 845/965 88% │ 33/35 94%
        2 │ 304/355 86% │ 12/13 92%
        3 │     1/1     │   1/1
        4 │   59/59     │   2/2
        6 │ 502/533 94% │ 38/43 88%

    при равной силе 12–13: длина 1 → 33/35, длина 2 → 10/10, длина 6 → 30/34

**Не объясняет ничего ни длина, ни сила.** На первой попытке доля дошедших одна и
та же, около 91 %, и от силы не зависит нигде, кроме `n = 2`, где всего три
маршрута. Единственная отсечка по силе в коде — `plan_min_step_n`, по умолчанию 2,
и область ниже неё тавтологична: планов там нет по построению.

Отдельное ограничение, которое надо помнить при любом замере на этом мире: **43
сида из 80 дают мир, в котором агент не сдвигается с места** — одно место после
800 шагов лепета и 1500 проб. Планы получаются только на 32 сидах, и все числа
выше — про них.

Условия воспроизводят исходный замер: петли не записываются
(`place_record_loops=False`), макросы выключены (`macro_max_length=0`). С петлями
картина другая и лучше — см. `AUDIT.md`, часть 3.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness.behaviour.babbling import Babbler, run_babbling            # noqa: E402
from harness.behaviour.goals import Goal                                # noqa: E402
from harness.behaviour.planner import Planner, choose_probe, execute     # noqa: E402
from harness.core.action import Action, action_key                      # noqa: E402
from harness.core.clocks import Clocks                                  # noqa: E402
from harness.core.profile import from_schema                            # noqa: E402
from harness.corpus.world import InteractiveWorld                       # noqa: E402
from harness.model.beliefs import Origin, Provenance                    # noqa: E402
from harness.model.forward import ForwardModel                          # noqa: E402
from harness.model.places import PlaceGraph                             # noqa: E402

HOLD_MS = 200


def one_seed(seed: int, rows: list[dict[str, Any]], *, babble: int, explore: int,
             goals: int, depth: int, loops: bool) -> dict[str, int]:
    """Один сид: лепет, разведка, потом цели. Каждая попытка — строка в `rows`."""
    profile = from_schema("измерение доли дошедших", capture_width=320,
                          capture_height=180, babble_repeats=3,
                          place_record_loops=loops, macro_max_length=0)
    min_n = int(profile.parameters["plan_min_step_n"])

    # Лепет в отдельном мире: карта тела — про тело, а не про этот проход.
    bw = InteractiveWorld(profile, seed=seed, n_outputs=16)
    bab = Babbler(profile, bw.outputs, rng_seed=seed)
    run_babbling(bw, bab, steps=babble, clocks=Clocks())
    inverse = dict(bab.inverse_found)

    world = InteractiveWorld(profile, seed=seed, n_outputs=16)
    graph = PlaceGraph.from_profile(profile)
    graph.see(world.step(None, with_audio=False).frame, 0, seconds_per_seq=1 / 30.0,
              mode="start")
    state = {"seq": 0, "last": None}

    def step(out: str, ms: int = HOLD_MS) -> str:
        obs = world.step(Action.key(out, ms), with_audio=False)
        state["seq"] += 1
        state["last"] = out
        return graph.see(obs.frame, state["seq"], seconds_per_seq=1 / 30.0,
                         mode=action_key(out, ms))

    model = ForwardModel.from_graph(graph, bab.body)
    for i in range(explore):
        if i % 25 == 0:
            model = ForwardModel.from_graph(graph, bab.body)
        out, ms = choose_probe(model, graph.current, world.outputs, hold_ms=HOLD_MS,
                              min_n=min_n, inverse=inverse, last_output=state["last"])
        step(out, ms)

    def reach(src: str, max_depth: int) -> dict[str, int]:
        """Куда можно дойти подтверждёнными рёбрами. Петли шагом не считаются."""
        seen = {src: 0}
        frontier = [src]
        for d in range(max_depth):
            nxt: list[str] = []
            for node in frontier:
                for key in model.actions_from(node):
                    pred = model.predict(node, key)
                    if pred is None:
                        continue
                    for outcome, p in pred.outcomes():
                        if (p < 0.5 or outcome.n < min_n or outcome.dst == node
                                or outcome.dst in seen):
                            continue
                        seen[outcome.dst] = d + 1
                        nxt.append(outcome.dst)
            frontier = nxt
        return {k: v for k, v in seen.items() if v > 0}

    made = 0
    for g in range(goals):
        model = ForwardModel.from_graph(graph, bab.body)
        here = graph.current or ""
        reachable = reach(here, depth)
        if not reachable:
            out, ms = choose_probe(model, here, world.outputs, hold_ms=HOLD_MS,
                                  min_n=min_n, inverse=inverse,
                                  last_output=state["last"])
            step(out, ms)
            continue
        target = max(sorted(reachable), key=lambda k: reachable[k])
        goal = Goal(id=f"g{g}", kind="reach_place", target=target,
                    test=lambda t=target: graph.current == t,
                    test_text="я в этом месте", budget_ticks=40,
                    provenance=Provenance(Origin.EXPERIENCE, branch="b", seq=0),
                    drive="curiosity", pressure=0.5)
        plan = Planner(profile, model).plan(goal, here, target)
        if plan is None:
            continue
        made += 1
        weak = min(plan.steps, key=lambda s: s.n)
        route = "→".join(f"{s.key()}:{s.expected_place}" for s in plan.steps)
        ex = execute(plan, act=lambda a: step(a.outputs_touched()[0], a.duration_ms),
                     goal=goal)
        rows.append({
            "seed": seed, "length": plan.length, "weakest": plan.min_step_n,
            "arrived": bool(ex.goal_passed), "min_n_setting": min_n,
            "route": f"{seed}|{here}|{route}",
            "weak_edge": f"{here}|{weak.key()}|{weak.expected_place}",
        })
    return {"plans": made, "places": len(graph)}


def by_field(data: list[dict], field: str) -> dict[int, tuple[int, int]]:
    acc: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for r in data:
        acc[r[field]][0] += 1
        acc[r[field]][1] += r["arrived"]
    return {k: (v[0], v[1]) for k, v in acc.items()}


def print_table(data: list[dict], field: str, title: str) -> None:
    print(f"\n{title}")
    for k, (total, arrived) in sorted(by_field(data, field).items()):
        print(f"  {field}={k:>3}: {arrived:>4}/{total:<4} = {arrived / total:>4.0%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=80, help="сколько сидов (с 2-го)")
    ap.add_argument("--babble", type=int, default=800)
    ap.add_argument("--explore", type=int, default=1500)
    ap.add_argument("--goals", type=int, default=60)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--loops", action="store_true",
                    help="записывать «нажал и остался» ребром (не как в исходном замере)")
    args = ap.parse_args()

    seeds = list(range(2, 2 + args.seeds))
    rows: list[dict[str, Any]] = []
    inert = 0
    for seed in seeds:
        info = one_seed(seed, rows, babble=args.babble, explore=args.explore,
                        goals=args.goals, depth=args.depth, loops=args.loops)
        inert += int(info["places"] <= 1)
        print(f"  сид {seed:>3}: мест {info['places']:>3}, планов {info['plans']:>3}",
              flush=True)

    if not rows:
        print("\nни одного плана: сравнивать нечего. Это результат, а не ошибка "
              "прогона — на этом мире так бывает")
        return 1

    seen: set[str] = set()
    first: list[dict] = []
    for r in rows:
        if r["route"] not in seen:
            seen.add(r["route"])
            first.append(r)

    print(f"\nвсего попыток {len(rows)}, различных маршрутов {len(first)} "
          f"({len(rows) / len(first):.1f} попытки на маршрут)")
    print(f"сидов, давших хотя бы один план: {len({r['seed'] for r in rows})} "
          f"из {len(seeds)}; сидов с миром из одного места: {inert}")
    print(f"порог планировщика plan_min_step_n = {rows[0]['min_n_setting']}, "
          f"минимальная сила среди построенных планов "
          f"{min(r['weakest'] for r in rows)} — ниже порога планов нет по построению")

    print_table(rows, "length", "ВСЕ попытки, по длине плана (так считали раньше):")
    print_table(rows, "weakest", "ВСЕ попытки, по силе слабейшего ребра:")
    print_table(first, "length", "ПЕРВАЯ попытка на маршрут, по длине плана:")
    print_table(first, "weakest", "ПЕРВАЯ попытка на маршрут, по силе слабейшего ребра:")

    print("\nпервая попытка: сила против длины. Если длина что-то значит, "
          "строки разойдутся при одном столбце")
    bins = ((2, 5), (6, 11), (12, 13), (14, 10 ** 6))
    grid: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for r in first:
        for lo, hi in bins:
            if lo <= r["weakest"] <= hi:
                grid[(r["length"], (lo, hi))][0] += 1
                grid[(r["length"], (lo, hi))][1] += r["arrived"]
                break
    head = "".join(f"{f'{lo}–' + (str(hi) if hi < 10 ** 6 else '∞'):>12}"
                   for lo, hi in bins)
    print("  длина \\ сила" + head)
    for length in sorted({r["length"] for r in first}):
        cells = []
        for b in bins:
            total, arrived = grid[(length, b)]
            cells.append((f"{arrived}/{total}" if total else "—").rjust(12))
        print(f"  {length:>10}  " + "".join(cells))

    print("\nрастёт ли сила ребра, на котором план рвётся — вот вся причина, по "
          "которой прежний вывод был неверен:")
    hist: dict[tuple, list[tuple[int, bool]]] = defaultdict(list)
    for r in rows:
        hist[(r["seed"], r["weak_edge"])].append((r["weakest"], r["arrived"]))
    for (seed, edge), h in sorted(hist.items(), key=lambda kv: -len(kv[1]))[:6]:
        ns = [n for n, _ in h]
        print(f"  сид {seed:>3} {edge}")
        print(f"      попыток {len(h):>3}, дошло {sum(1 for _, a in h if a):>3}, "
              f"сила: первая {ns[0]}, последняя {ns[-1]}, "
              f"различных значений {len(set(ns))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
