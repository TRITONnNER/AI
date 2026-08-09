"""Счётчики срабатываний на ветках арбитража: где код мёртв. Инвариант 26.

`TASK-04-PLACEGRAPH.md`, часть 4. Основание — первая версия замыкания петли была
мёртвым кодом под ненасыщаемым условием, и установил это замер, а не чтение.

Запуск: `python3 tools/measure_branches.py [--seeds "3 5 7"] [--steps 1500]`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness.behaviour.babbling import Babbler, run_babbling
from harness.behaviour.planner import choose_closing_probe, closing_arbitration
from harness.core.action import Action, action_key
from harness.core.branches import Ledger
from harness.core.clocks import Clocks
from harness.core.profile import from_schema
from harness.corpus.world import InteractiveWorld
from harness.model.forward import ForwardModel
from harness.model.places import PlaceGraph


def run(seed: int, steps: int, arb) -> dict:
    profile = from_schema("ветки", capture_width=320, capture_height=180,
                          babble_repeats=3, place_record_loops=True,
                          macro_max_length=0, explore_closes_loops=True)
    min_n = int(profile.parameters["plan_min_step_n"])
    bw = InteractiveWorld(profile, seed=seed, n_outputs=16)
    bab = Babbler(profile, bw.outputs, rng_seed=seed)
    run_babbling(bw, bab, steps=600, clocks=Clocks())
    inverse = dict(bab.inverse_found)
    world = InteractiveWorld(profile, seed=seed, n_outputs=16)
    graph = PlaceGraph.from_profile(profile)
    graph.see(world.step(None, with_audio=False).frame, 0,
              seconds_per_seq=1 / 30.0, mode="start")
    st = {"seq": 0, "last": None}

    def step(out: str, ms: int = 200) -> str:
        obs = world.step(Action.key(out, ms), with_audio=False)
        st["seq"] += 1
        st["last"] = out
        return graph.see(obs.frame, st["seq"], seconds_per_seq=1 / 30.0,
                         mode=action_key(out, ms))

    model = ForwardModel.from_graph(graph, bab.body)
    for i in range(steps):
        if i % 25 == 0:
            model = ForwardModel.from_graph(graph, bab.body)
        out, ms = choose_closing_probe(model, graph, graph.current, world.outputs,
                                       hold_ms=200, min_n=min_n, inverse=inverse,
                                       last_output=st["last"], arb=arb)
        step(out, ms)
    return {"places": len(graph)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="3 5 7 11 13 17 19 23")
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()

    ledger = Ledger()
    arb = ledger.add(closing_arbitration())
    for seed in [int(x) for x in args.seeds.split()]:
        info = run(seed, args.steps, arb)
        print(f"  сид {seed:>3}: мест {info['places']:>3}", flush=True)
    print()
    print(ledger.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
