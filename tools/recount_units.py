"""Пересчёт заявленных метрик проекта по независимым единицам.

Основание — `TASK-03-MEASUREMENT-LOOPS.md`, часть 0, и инвариант 22. Причина: в
замере доли дошедших планов 1913 «наблюдений» оказались 94 маршрутами —
псевдорепликация в двадцать раз, — и вывод из-за этого был неверен дважды подряд.
Ошибка такого рода не бывает локальной, поэтому здесь пересчитывается всё, что
проект заявляет числом.

Что именно проверяется. Не «какое число получилось», а **сколько независимых
наблюдений за ним стоит**. Для каждой метрики:

1. что считалось за единицу в прежнем замере;
2. независимы ли эти единицы между собой;
3. чему равно `n`, если считать независимые единицы, а не события.

Три уровня вложенности, которые здесь всё время путались:

- **пиксель** — соседние пиксели одного кадра почти всегда одного класса, и их
  миллионы. Никогда не единица;
- **кадр** — соседние кадры одной сессии различаются на доли процента. Не единица;
- **прогон** `(домен, сид)` — то, что действительно варьируется независимо внутри
  одного генератора;
- **домен** — разный генератор, разная логика обрамления. Единица для утверждений
  вида «метод доменно-независим»;
- **генератор целиком** — если детектор и генератор писались вместе, все прогоны
  разделяют одно допущение, и независимых единиц ровно **одна**. Это не
  преувеличение: именно это проверяет часть 2 живыми записями.

Запуск: `python3 tools/recount_units.py [--seeds 6] [--frames 240] [--json FILE]`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Разброс, посчитанный честно
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Spread:
    """Разброс по набору независимых наблюдений. `n` — число единиц, не событий."""

    n: int
    median: float | None
    lo: float | None
    hi: float | None
    sd: float | None

    @classmethod
    def of(cls, values: Iterable[float | None]) -> Spread:
        xs = [float(x) for x in values if x is not None]
        if not xs:
            return cls(0, None, None, None, None)
        return cls(len(xs), statistics.median(xs), min(xs), max(xs),
                   statistics.stdev(xs) if len(xs) > 1 else None)

    def text(self) -> str:
        if self.n == 0:
            return "n=0, считать нечего"
        if self.n == 1:
            return f"n=1, единственное значение {self.median:.3f}"
        sd = "—" if self.sd is None else f"{self.sd:.3f}"
        return (f"n={self.n}, медиана {self.median:.3f}, "
                f"мин {self.lo:.3f}, макс {self.hi:.3f}, sd {sd}")

    def as_dict(self) -> dict[str, Any]:
        return {"n": self.n, "median": self.median, "lo": self.lo, "hi": self.hi,
                "sd": self.sd}


# ---------------------------------------------------------------------------
# Замеры
# ---------------------------------------------------------------------------


def measure_layers(seeds: list[int], frames: int) -> dict[str, Any]:
    """IoU и точность разделения себя и мира по трём уровням единиц.

    Прежнее заявление — «IoU 0.97, точность 0.99». Что за ним стояло: одно число
    на прогон, посчитанное по накопленной за прогон маске, и шесть сидов на домен.
    Пикселей в накоплении миллионы, но независимых наблюдений — шесть на домен, и
    сиды одного генератора независимы только по шуму, а не по устройству мира.
    """
    from harness.benchmark import bench_domain
    from harness.corpus.domains import DOMAINS

    rows: list[dict[str, Any]] = []
    for name in sorted(DOMAINS):
        for seed in seeds:
            d = bench_domain(name, seed=seed, frames=frames,
                             babble_steps=400).as_dict()
            rows.append({"domain": name, "seed": seed, "iou": d["iou"],
                         "precision": d["precision"], "recall": d["recall"],
                         "frames": d["frames"],
                         "live": d["live"], "silent": d["silent"],
                         "undecided": d["undecided"]})
            print(f"  {name:<10} сид {seed}: IoU {d['iou']}, точность {d['precision']}, "
                  f"полнота {d['recall']}", flush=True)

    per_domain_iou = []
    per_domain: dict[str, Spread] = {}
    for name in sorted(DOMAINS):
        xs = [r["iou"] for r in rows if r["domain"] == name]
        per_domain[name] = Spread.of(xs)
        got = [x for x in xs if x is not None]
        if got:
            per_domain_iou.append(statistics.median(got))

    return {
        "rows": rows,
        "by_run_iou": Spread.of([r["iou"] for r in rows]).as_dict(),
        "by_run_precision": Spread.of([r["precision"] for r in rows]).as_dict(),
        "by_run_recall": Spread.of([r["recall"] for r in rows]).as_dict(),
        "by_seed_within_domain": {k: v.as_dict() for k, v in per_domain.items()},
        "by_domain_iou": Spread.of(per_domain_iou).as_dict(),
        "frames_per_run": rows[0]["frames"] if rows else 0,
    }


def measure_babbling(seeds: list[int], domains: list[str]) -> dict[str, Any]:
    """Лепет: живые выходы, молчащие, обратные пары.

    Прежнее заявление — «9/9 живых, 15/15 молчащих, 8/8 обратных пар». Это один
    прогон одного домена: шестнадцать выходов одного мира. Единица здесь —
    прогон, а не выход: выходы одного мира разделяют его проводку, и «9 из 9»
    означает «в одном мире всё найдено», а не девять независимых подтверждений.
    """
    from harness.benchmark import bench_domain

    rows: list[dict[str, Any]] = []
    for name in domains:
        for seed in seeds:
            d = bench_domain(name, seed=seed, frames=140,
                             babble_steps=700).as_dict()
            live_found, live_true = (int(x) for x in d["live"].split("/"))
            silent_found, silent_true = (int(x) for x in d["silent"].split("/"))
            rows.append({
                "domain": name, "seed": seed,
                "live_found": live_found, "live_true": live_true,
                "silent_found": silent_found, "silent_true": silent_true,
                "live_ok": live_true > 0 and live_found == live_true,
                "silent_ok": silent_true > 0 and silent_found == silent_true,
                "wrong_body": d["wrong_body"],
            })
            print(f"  {name:<10} сид {seed}: живых {d['live']}, молчащих {d['silent']}, "
                  f"ошибок тела {d['wrong_body']}", flush=True)

    runs_all_live = sum(1 for r in rows if r["live_ok"])
    runs_all_silent = sum(1 for r in rows if r["silent_ok"])
    return {
        "rows": rows,
        "runs": len(rows),
        "runs_all_live_found": runs_all_live,
        "runs_all_silent_found": runs_all_silent,
        # Доля найденных живых внутри прогона — по прогонам, а не по выходам.
        "live_share_by_run": Spread.of(
            [r["live_found"] / r["live_true"] if r["live_true"] else None
             for r in rows]).as_dict(),
        "domains": sorted({r["domain"] for r in rows}),
    }


def measure_goals(seeds: list[int]) -> dict[str, Any]:
    """Цели: сколько поставлено и сколько прошло тест.

    Прежнее заявление — «9 целей, 6 прошли, 2 брошены» с одного прогона на сорок
    раундов. Девять целей одного прогона не независимы: они выведены из одной и
    той же карты тела, в одном и том же мире, и вторая цель существует только
    потому, что первая что-то изменила. Единица — прогон.
    """
    from harness.behaviour.babbling import Babbler, run_babbling
    from harness.behaviour.goals import GoalStack, candidates_from_body, choose
    from harness.core.clocks import Clocks
    from harness.core.profile import from_schema
    from harness.corpus.world import InteractiveWorld
    from harness.model.drives import Motivation
    from harness.vision.predict import PredictionError

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        # Условия воспроизводят прежний замер: сорок раундов, цель ставится, когда
        # предыдущая закрылась, лепет идёт между раундами.
        profile = from_schema("пересчёт-цели", capture_width=160, capture_height=96,
                              babble_rate=0.9, babble_repeats=3)
        world = InteractiveWorld(profile, seed=seed, n_outputs=16)
        bab = Babbler(profile, world.outputs, rng_seed=seed)
        motivation = Motivation(profile)
        stack = GoalStack(profile)
        error = PredictionError(profile)
        clocks = Clocks()
        threshold = float(profile.parameters["irreversibility_threshold"])

        pushed = 0
        for round_no in range(40):
            summary = error.summary()
            motivation.update(error_mean=summary["mean"],
                              error_sigma=summary["sigma"],
                              error_now=summary["last"])
            if stack.active is None:
                cands = candidates_from_body(bab.body, world.outputs,
                                            caution_threshold=threshold)
                picked = choose(cands, motivation, top=1)
                if picked:
                    stack.push(picked[0], motivation, round_no, "b0",
                               budget_ticks=50)
                    pushed += 1
            run_babbling(world, bab, steps=40, clocks=clocks, error=error)
            stack.tick()
        st = stack.stats()
        rows.append({"seed": seed, "pushed": pushed,
                     "passed": int(st.get("passed", 0)),
                     "abandoned": int(st.get("abandoned", 0))})
        print(f"  сид {seed}: поставлено {pushed}, прошло {st.get('passed')}, "
              f"брошено {st.get('abandoned')}", flush=True)

    closed = [(r["passed"], r["passed"] + r["abandoned"]) for r in rows]
    return {
        "rows": rows,
        "runs": len(rows),
        "pass_share_by_run": Spread.of(
            [p / t if t else None for p, t in closed]).as_dict(),
        "goals_pushed_by_run": Spread.of(
            [float(r["pushed"]) for r in rows]).as_dict(),
    }


# ---------------------------------------------------------------------------
# Планы: образец методики — единица наблюдения есть различный маршрут
# ---------------------------------------------------------------------------
#
# Уже пересчитано в `TASK-02`, часть 0, и зафиксировано подробно в
# `tools/measure_plan_arrival.py`. Здесь то же самое короче и с обоими состояниями
# переключателя петель, чтобы все числа проекта лежали в одной таблице.


def arrival_by_route(seeds: list[int], *, loops: bool, goals: int = 40,
                     explore: int = 1200, depth: int = 6) -> dict[str, Any]:
    from harness.behaviour.babbling import Babbler, run_babbling
    from harness.behaviour.goals import Goal
    from harness.behaviour.planner import Planner, choose_probe, execute
    from harness.core.action import Action, action_key
    from harness.core.clocks import Clocks
    from harness.core.profile import from_schema
    from harness.corpus.world import InteractiveWorld
    from harness.model.beliefs import Origin, Provenance
    from harness.model.forward import ForwardModel
    from harness.model.places import PlaceGraph

    hold = 200
    attempts: list[dict[str, Any]] = []
    for seed in seeds:
        profile = from_schema("пересчёт-планы", capture_width=320, capture_height=180,
                              babble_repeats=3, place_record_loops=loops,
                              macro_max_length=0)
        min_n = int(profile.parameters["plan_min_step_n"])
        bw = InteractiveWorld(profile, seed=seed, n_outputs=16)
        bab = Babbler(profile, bw.outputs, rng_seed=seed)
        run_babbling(bw, bab, steps=600, clocks=Clocks())
        inverse = dict(bab.inverse_found)

        world = InteractiveWorld(profile, seed=seed, n_outputs=16)
        graph = PlaceGraph.from_profile(profile)
        graph.see(world.step(None, with_audio=False).frame, 0,
                  seconds_per_seq=1 / 30.0, mode="start")
        state = {"seq": 0, "last": None}

        def step(out: str, ms: int = hold) -> str:
            obs = world.step(Action.key(out, ms), with_audio=False)
            state["seq"] += 1
            state["last"] = out
            return graph.see(obs.frame, state["seq"], seconds_per_seq=1 / 30.0,
                            mode=action_key(out, ms))

        model = ForwardModel.from_graph(graph, bab.body)
        for i in range(explore):
            if i % 25 == 0:
                model = ForwardModel.from_graph(graph, bab.body)
            out, ms = choose_probe(model, graph.current, world.outputs, hold_ms=hold,
                                  min_n=min_n, inverse=inverse,
                                  last_output=state["last"])
            step(out, ms)

        def reach(src: str) -> dict[str, int]:
            seen = {src: 0}
            frontier = [src]
            for d in range(depth):
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

        for g in range(goals):
            model = ForwardModel.from_graph(graph, bab.body)
            here = graph.current or ""
            r = reach(here)
            if not r:
                out, ms = choose_probe(model, here, world.outputs, hold_ms=hold,
                                      min_n=min_n, inverse=inverse,
                                      last_output=state["last"])
                step(out, ms)
                continue
            target = max(sorted(r), key=lambda k: r[k])
            goal = Goal(id=f"g{g}", kind="reach_place", target=target,
                        test=lambda t=target: graph.current == t,
                        test_text="я в этом месте", budget_ticks=40,
                        provenance=Provenance(Origin.EXPERIENCE, branch="b", seq=0),
                        drive="curiosity", pressure=0.5)
            plan = Planner(profile, model).plan(goal, here, target)
            if plan is None:
                continue
            route = "→".join(f"{s.key()}:{s.expected_place}" for s in plan.steps)
            ex = execute(plan, act=lambda a: step(a.outputs_touched()[0],
                                                  a.duration_ms), goal=goal)
            attempts.append({"seed": seed, "length": plan.length,
                             "route": f"{seed}|{here}|{route}",
                             "arrived": bool(ex.goal_passed)})

    seen: set[str] = set()
    first: list[dict[str, Any]] = []
    for a in attempts:
        if a["route"] not in seen:
            seen.add(a["route"])
            first.append(a)

    def by_length(data: list[dict[str, Any]]) -> dict[str, str]:
        out: dict[str, list[int]] = {}
        for a in data:
            out.setdefault(str(a["length"]), [0, 0])
            out[str(a["length"])][0] += 1
            out[str(a["length"])][1] += a["arrived"]
        return {k: f"{v[1]}/{v[0]}" for k, v in sorted(out.items(),
                                                       key=lambda kv: int(kv[0]))}

    return {
        "loops": loops,
        "attempts": len(attempts),
        "routes": len(first),
        "pseudoreplication": round(len(attempts) / max(1, len(first)), 1),
        "arrived_by_attempt": sum(1 for a in attempts if a["arrived"]),
        "arrived_by_route": sum(1 for a in first if a["arrived"]),
        "by_length_attempts": by_length(attempts),
        "by_length_routes": by_length(first),
    }


# ---------------------------------------------------------------------------
# Печать
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--only", nargs="*", default=[],
                    help="layers / babbling / goals / plans")
    args = ap.parse_args()

    seeds = list(range(1, args.seeds + 1))
    want = set(args.only) or {"layers", "babbling", "goals", "plans"}
    out: dict[str, Any] = {"seeds": seeds, "frames": args.frames}

    if "layers" in want:
        print("── разделение себя и мира ──")
        out["layers"] = measure_layers(seeds, args.frames)
        L = out["layers"]
        print(f"\n  единица «прогон (домен, сид)»: IoU "
              f"{Spread(**L['by_run_iou']).text()}")
        print(f"  единица «прогон»: точность "
              f"{Spread(**L['by_run_precision']).text()}")
        print(f"  единица «домен» (медиана по сидам): IoU "
              f"{Spread(**L['by_domain_iou']).text()}")
        for name, sp in L["by_seed_within_domain"].items():
            print(f"    сиды внутри {name:<10} {Spread(**sp).text()}")

    if "babbling" in want:
        print("\n── лепет ──")
        out["babbling"] = measure_babbling(seeds, ["game", "document", "desktop",
                                                   "video"])
        B = out["babbling"]
        print(f"\n  прогонов {B['runs']}, из них все живые найдены в "
              f"{B['runs_all_live_found']}, все молчащие в "
              f"{B['runs_all_silent_found']}")
        print(f"  доля найденных живых по прогонам: "
              f"{Spread(**B['live_share_by_run']).text()}")

    if "goals" in want:
        print("\n── цели ──")
        out["goals"] = measure_goals(seeds)
        G = out["goals"]
        print(f"\n  прогонов {G['runs']}; доля прошедших среди закрытых: "
              f"{Spread(**G['pass_share_by_run']).text()}")
        print(f"  целей поставлено за прогон: "
              f"{Spread(**G['goals_pushed_by_run']).text()}")

    if "plans" in want:
        print("\n── планы (образец методики: единица — маршрут) ──")
        out["plans"] = {}
        for loops in (False, True):
            key = "с петлями" if loops else "без петель"
            print(f"  {key}…", flush=True)
            r = arrival_by_route(seeds, loops=loops)
            out["plans"][key] = r
            print(f"    попыток {r['attempts']}, маршрутов {r['routes']} "
                  f"(псевдорепликация ×{r['pseudoreplication']})")
            print(f"    дошло по попыткам {r['arrived_by_attempt']}/{r['attempts']}, "
                  f"по маршрутам {r['arrived_by_route']}/{r['routes']}")
            print(f"    по длинам, попытки: {r['by_length_attempts']}")
            print(f"    по длинам, маршруты: {r['by_length_routes']}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=1, default=str) + "\n",
            encoding="utf-8")
        print(f"\nмашинно: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
