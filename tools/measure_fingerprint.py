"""Насыщается ли граф мест, и от чего это зависит. TASK-29, направление A.

Часовой прогон (TASK-24, D) дал 22 789 узлов графа мест и вывод «отпечаток различает шум».
**Вывод был неверен**, и здесь это устанавливается числами.

## Три части

1. **Разрешение функции отпечатка.** Тот же вид дважды, тот же вид с шумом, тот же вид,
   сдвинутый на k пикселей. Из кривой видно, при каком сдвиге вид перестаёт быть тем же
   местом, — это и есть **разрешение** функции в пикселях мира.
2. **Кривая насыщения.** Узлы против **наблюдений**, а не против времени (вырождение 13.7):
   по времени кривая гнётся от того, что цикл замедляется, а не от того, что мир кончается.
3. **Что от этого меняется в поведении.** Мест, подтверждённых переходов, разведка с
   возвратом и без — те же числа, что записаны в тестах планировщика.

## Почему первый вывод был неверен

Отпечаток снимался **с каждого сотого оборота**, за который мир уезжал на сотни пикселей, и
арена за час начала 14 338 эпизодов. Граф кормили видами из тысяч состояний, между которыми
нет пути, — «не насыщается» описывало **способ кормления**, а не функцию.

Что остаётся верным: замедление цикла шло именно от графа мест (выключение по частям, 1.02×
против 1.84×), и стоимость сравнения со всеми узлами растёт с их числом.

## Инвариант 32: обе ошибки, а не одна

- **ложная тревога** (ложное расщепление) — тот же вид объявлен новым местом. Граф пухнет,
  но след остаётся: узел виден и его можно пересчитать;
- **ложное подтверждение** (ложное слияние) — разные виды объявлены одним местом. Карта
  говорит «я тут был», планировщик верит, и следа не остаётся.

Обе — против известной истины из состояния мира (`cam_x`, `cam_y`), доступного только
исследователю.

Прогон: `python3 tools/measure_fingerprint.py`.
Результат — `docs/measurements/fingerprint.json`.
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

from harness.core.action import Action, action_key                     # noqa: E402
from harness.core.profile import from_schema                           # noqa: E402
from harness.corpus.world import InteractiveWorld, WorldState          # noqa: E402
from harness.model.forward import ForwardModel                         # noqa: E402
from harness.model.places import PlaceGraph, fingerprint, similarity    # noqa: E402

#: Две функции отпечатка: та, что была до TASK-29, и та, что стала.
FUNCTIONS: dict[str, dict[str, int]] = {
    "точное равенство, 4 уровня": {"levels": 4, "tolerance": 0, "blur": 0},
    "полоса ±1, 8 уровней": {"levels": 8, "tolerance": 1, "blur": 0},
}
THRESHOLDS = (0.70, 0.82, 0.90)
#: Сдвиги, на которых снимается разрешение. Шаг мира — 6 px за 100 мс удержания, удержание
#: разведки — 200 мс, то есть 12 px. Обе величины попадают в набор нарочно.
SHIFTS = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
WORLD_STEP_PX = 6.0
PROBE_HOLD_PX = 12.0


def _profile(**kw: Any):
    base: dict[str, Any] = dict(capture_width=160, capture_height=120)
    base.update(kw)
    return from_schema("ОТПЕЧАТОК", **base)


def resolution(fn: str, *, seeds: tuple[int, ...] = (1, 2, 3)) -> dict[str, Any]:
    """Кривая похожести против сдвига плюс обе доли ошибок (инвариант 32)."""
    cfg = FUNCTIONS[fn]
    prof = _profile(place_levels=cfg["levels"], place_level_tolerance=cfg["tolerance"],
                    place_blur_px=cfg["blur"])
    thr = float(prof.parameters["place_same_similarity"])
    tol = cfg["tolerance"]

    def fp(frame: np.ndarray) -> tuple[int, ...]:
        return fingerprint(frame, levels=cfg["levels"], blur_px=cfg["blur"])

    def sim(a: tuple[int, ...], b: tuple[int, ...]) -> float:
        return similarity(a, b, tolerance=tol)

    by_shift: dict[int, list[float]] = {s: [] for s in SHIFTS}
    repeat: list[float] = []
    noisy: dict[str, list[float]] = {"дизеринг ±1": [], "дизеринг ±4": [],
                                     "яркость +20": [], "блоки 8×8": []}
    for seed in seeds:
        world = InteractiveWorld(prof, seed=seed)
        x0, y0 = world.state.cam_x, world.state.cam_y

        def frame_at(dx: float) -> np.ndarray:
            world.state = WorldState(cam_x=x0 + dx, cam_y=y0)
            return world.step(None, with_audio=False).frame.astype(np.float64)

        base_frame = frame_at(0.0)
        base = fp(base_frame)
        repeat.append(sim(base, fp(frame_at(0.0))))
        rng = np.random.default_rng(seed)
        for _ in range(5):
            noisy["дизеринг ±1"].append(
                sim(base, fp(base_frame + rng.integers(-1, 2, base_frame.shape))))
            noisy["дизеринг ±4"].append(
                sim(base, fp(base_frame + rng.integers(-4, 5, base_frame.shape))))
        noisy["яркость +20"].append(sim(base, fp(base_frame + 20.0)))
        blocky = base_frame.copy()
        h, w = blocky.shape[:2]
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                blocky[i:i + 8, j:j + 8] = base_frame[i:i + 8, j:j + 8].mean()
        noisy["блоки 8×8"].append(sim(base, fp(blocky)))
        for s in SHIFTS:
            by_shift[s].append(sim(base, fp(frame_at(float(s)))))

    curve = {s: statistics.median(v) for s, v in by_shift.items()}
    crossed = [s for s in SHIFTS if s > 0 and curve[s] < thr]
    res = min(crossed) if crossed else None

    same_pairs = repeat + [v for vals in noisy.values() for v in vals]
    far = [v for s in SHIFTS if s >= 64 for v in by_shift[s]]
    false_alarm = sum(1 for v in same_pairs if v < thr)
    false_confirm = sum(1 for v in far if v >= thr)
    return {
        "function": fn, "levels": cfg["levels"], "tolerance": tol, "blur": cfg["blur"],
        "threshold": thr, "curve": {str(k): round(v, 4) for k, v in curve.items()},
        "repeat_median": round(statistics.median(repeat), 4),
        # Минимум рядом с медианой нарочно: на одном мире повторяемость 0.984, то есть
        # тот же вид теряет ячейку из 64 на собственной анимации интерфейса.
        "repeat_min": round(min(repeat), 4),
        "noise": {k: round(statistics.median(v), 4) for k, v in noisy.items()},
        "resolution_px": res,
        "world_step_px": WORLD_STEP_PX, "probe_hold_px": PROBE_HOLD_PX,
        "same_at_world_step": curve[6] >= thr,
        "differs_at_probe_hold": curve[12] < thr,
        "false_alarm": {"n": len(same_pairs), "fired": false_alarm,
                        "share": false_alarm / len(same_pairs) if same_pairs else None,
                        "unit": "пара видов",
                        "how": "мир не двигался (повтор, шум, яркость, блоки) — "
                               "объявлено другим местом"},
        "false_confirm": {"n": len(far), "fired": false_confirm,
                          "share": false_confirm / len(far) if far else None,
                          "unit": "пара видов",
                          "how": "мир уехал дальше кадра, общих пикселей нет — "
                                 "объявлено тем же местом"},
        "n": len(seeds), "unit": "мир (сид)",
    }


def saturation(fn: str, *, threshold: float, observations: int, every: int,
               seed: int = 1, width: int = 64, height: int = 48) -> dict[str, Any]:
    """Узлы против наблюдений. `every` — раз во сколько шагов мира снимается отпечаток."""
    cfg = FUNCTIONS[fn]
    prof = _profile(capture_width=width, capture_height=height,
                    place_levels=cfg["levels"], place_level_tolerance=cfg["tolerance"],
                    place_blur_px=cfg["blur"], place_same_similarity=threshold)
    world = InteractiveWorld(prof, seed=seed, n_outputs=16)
    graph = PlaceGraph.from_profile(prof)
    rng = np.random.default_rng(seed)
    curve: list[list[int]] = []
    seen = step = 0
    while seen < observations:
        out = world.outputs[int(rng.integers(len(world.outputs)))]
        obs = world.step(Action.key(out, 200), with_audio=False)
        step += 1
        if step % every:
            continue
        seen += 1
        graph.see(obs.frame, step, seconds_per_seq=1 / 30.0, mode=action_key(out, 200))
        if seen % 250 == 0:
            curve.append([seen, len(graph.places)])
    half = observations // 2
    at_half = next((p for o, p in curve if o >= half), len(graph.places))
    tail = (len(graph.places) - at_half) / max(1, observations - half)
    return {"function": fn, "threshold": threshold, "every": every,
            "observations": observations, "places": len(graph.places),
            "new_per_observation_tail": round(tail, 4), "curve": curve, "seed": seed,
            "capture": f"{width}×{height}", "unit": "прогон"}


def behaviour(fn: str, *, seed: int, steps: int = 1200,
              with_return: bool = True) -> dict[str, Any]:
    """Мест, подтверждённых переходов, ведущих куда-то. Те же числа, что в тестах."""
    from harness.behaviour.babbling import Babbler, run_babbling
    from harness.behaviour.planner import choose_probe
    from harness.core.clocks import Clocks

    cfg = FUNCTIONS[fn]
    prof = _profile(capture_width=320, capture_height=180,
                    place_levels=cfg["levels"], place_level_tolerance=cfg["tolerance"],
                    place_blur_px=cfg["blur"])
    inverse: dict[str, str] = {}
    if with_return:
        bw = InteractiveWorld(prof, seed=seed, n_outputs=16)
        b = Babbler(prof, bw.outputs, rng_seed=seed)
        run_babbling(bw, b, steps=1200, clocks=Clocks())
        inverse = dict(b.inverse_found)

    world = InteractiveWorld(prof, seed=seed, n_outputs=16)
    graph = PlaceGraph.from_profile(prof)
    state: dict[str, Any] = {"seq": 0, "last": None}
    graph.see(world.step(None, with_audio=False).frame, 0, seconds_per_seq=1 / 30.0,
              mode="start")
    model = ForwardModel.from_graph(graph)
    for i in range(steps):
        if i % 25 == 0:
            graph.refine()
            model = ForwardModel.from_graph(graph)
        out, ms = choose_probe(model, graph.current, world.outputs, hold_ms=200,
                              min_n=2, inverse=inverse or None,
                              last_output=state["last"])
        obs = world.step(Action.key(out, ms), with_audio=False)
        state["seq"] = int(state["seq"]) + 1
        state["last"] = out
        graph.see(obs.frame, int(state["seq"]), seconds_per_seq=1 / 30.0,
                  mode=action_key(out, ms))
    graph.refine()
    model = ForwardModel.from_graph(graph)
    confirmed = sum(1 for outs in model.transitions.values() for o in outs if o.n >= 2)
    leaving = sum(1 for (src, _), outs in model.transitions.items()
                  for o in outs if o.n >= 2 and o.dst != src)
    total = sum(len(v) for v in model.transitions.values())
    return {"function": fn, "seed": seed, "with_return": with_return,
            "places": len(graph.places), "confirmed": confirmed, "leaving": leaving,
            "outcomes": total, "confirmed_share": round(confirmed / max(1, total), 4),
            "unit": "прогон"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="разрешение и насыщение отпечатка места")
    ap.add_argument("--observations", type=int, default=3000)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "fingerprint.json")
    a = ap.parse_args(argv)

    print("часть 1: разрешение функции отпечатка\n")
    res = [resolution(fn) for fn in FUNCTIONS]
    head = "  ".join(f"{s:>5}" for s in SHIFTS)
    print(f"{'функция':<28} {'повтор':>13} {'шум±4':>7} | сдвиг, px: {head}")
    for r in res:
        row = "  ".join(f"{r['curve'][str(s)]:>5.2f}" for s in SHIFTS)
        rep = f"{r['repeat_median']:.3f}/{r['repeat_min']:.3f}"
        print(f"{r['function']:<28} {rep:>13} {r['noise']['дизеринг ±4']:>7.3f} | "
              f"{' ' * 11}{row}")
    print()
    for r in res:
        print(f"{r['function']:<28} разрешение {str(r['resolution_px']) + ' px':>7}; "
              f"шаг мира 6 px — {'то же место' if r['same_at_world_step'] else 'другое'}; "
              f"удержание 200 мс (12 px) — "
              f"{'другое место' if r['differs_at_probe_hold'] else 'то же'}")
    print("\nобе ошибки проверки «это то же место» (инвариант 32):")
    for r in res:
        fa, fc = r["false_alarm"], r["false_confirm"]
        print(f"  {r['function']:<28} ложных тревог {fa['fired']}/{fa['n']} "
              f"({fa['share']:.0%}), ложных подтверждений {fc['fired']}/{fc['n']} "
              f"({fc['share']:.0%})")

    print("\nчасть 2: кривая насыщения — узлы против наблюдений\n")
    sat = [saturation(fn, threshold=thr, observations=a.observations, every=1)
           for fn in FUNCTIONS for thr in THRESHOLDS]
    # Отдельно — то, как кормили граф в часовом прогоне: каждый сотый оборот.
    sat.append(saturation(next(iter(FUNCTIONS)), threshold=0.82,
                          observations=a.observations, every=100))
    print(f"{'функция':<28} {'порог':>6} {'каждый':>7} {'мест':>6} "
          f"{'новых мест на наблюдение (2-я половина)':>40}")
    for s in sat:
        print(f"{s['function']:<28} {s['threshold']:>6.2f} {s['every']:>7} "
              f"{s['places']:>6} {s['new_per_observation_tail']:>40.3f}")

    print("\nчасть 3: что меняется в поведении (сиды 5, 7, 13)\n")
    beh = [behaviour(fn, seed=seed, with_return=wr)
           for fn in FUNCTIONS for seed in (5, 7, 13) for wr in (True, False)]
    print(f"{'функция':<28} {'возврат':>8} {'мест':>16} {'подтверждено':>18} "
          f"{'ведущих куда-то':>18}")
    for fn in FUNCTIONS:
        for wr in (True, False):
            mine = [b for b in beh if b["function"] == fn and b["with_return"] is wr]
            print(f"{fn:<28} {'есть' if wr else 'нет':>8} "
                  f"{' / '.join(str(b['places']) for b in mine):>16} "
                  f"{' / '.join(str(b['confirmed']) for b in mine):>18} "
                  f"{' / '.join(str(b['leaving']) for b in mine):>18}")

    data = {
        "resolution": res, "saturation": sat, "behaviour": beh,
        "functions": FUNCTIONS, "thresholds": list(THRESHOLDS), "shifts": list(SHIFTS),
        "world_step_px": WORLD_STEP_PX, "probe_hold_px": PROBE_HOLD_PX,
        "unit": "мир (сид) для разрешения, прогон для насыщения и поведения",
        "claim": ("граф мест насыщается, если наблюдения идут подряд; «не насыщается» в "
                  "часовом прогоне описывало способ кормления графа, а не функцию"),
        "how_refuted": ("если кривая узлов не выходит на полку и при наблюдениях подряд, "
                        "и при обеих функциях, дело не в кормлении, и объяснение неверно"),
        "corrects": "MEASUREMENT.md, 30.4 — вывод «насыщения нет» и его причина",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nзаписано: {a.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
