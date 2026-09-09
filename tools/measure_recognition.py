"""Три средства узнавания места и разделение отпечатка карточки. SPEC-FULL, A2.

Предрегистрация — `MEASUREMENT.md`, раздел 39.

## Повкладно, а не вместе

22 789 узлов чинятся четырьмя разными правками, и включить их разом значило бы узнать,
что стало лучше, но не узнать, от чего. Поэтому здесь **все восемь сочетаний** трёх
средств, и выключенное сочетание — контрольная точка, воспроизводящая прежний граф мест.

## Истина берётся из уже объявленной, а не заводится вторая

`tools/measure_fingerprint.py` объявил истину так: два вида — одно место, если камера
сдвинулась меньше чем на 32 пикселя мира, и разные места, если на 64 и больше. Пары
между 32 и 64 **исключаются как неоднозначные**, а не приписываются к одной из сторон.
Здесь та же истина и те же числа: вторая копия разошлась бы с первой молча, и сравнить
разделы 31.2 и 39 стало бы нечем.

## Единица независимости — прогон

Доли ошибок считаются **внутри прогона**, а потом берётся медиана по прогонам. Пары
наблюдений внутри прогона зависимы: соседняя пара похожа на предыдущую оттого, что
камера ушла на один шаг. Считать `n` парами значило бы завысить уверенность в сотни
раз — та же псевдорепликация, что обесценила 1913 «планов» (инвариант 22).

## Узлы против наблюдений, а не против времени

Вырождение 13.7: по времени кривая гнётся оттого, что цикл замедляется, а не оттого,
что мир кончается. Часовой прогон уже один раз объявил «вышли на полку», и это было
ложное подтверждение из-за неверной оси.

Прогон: `python3 tools/measure_recognition.py`.
Результат — `docs/measurements/recognition.json`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.core.action import Action, action_key           # noqa: E402
from harness.core.profile import from_schema                 # noqa: E402
from harness.corpus.world import InteractiveWorld            # noqa: E402
from harness.model.identity import Quantized, Split          # noqa: E402
from harness.model.places import PlaceGraph                  # noqa: E402

#: Три средства A2. Ключ — настройка схемы, значение — короткое имя для отчёта.
MEANS = {
    "place_sequence_matching": "отрезок",
    "place_adaptive_threshold": "фон",
    "place_novelty_hypothesis": "новизна",
}
SEEDS = (11, 12, 13)
OBSERVATIONS = 1500

#: Истина, объявленная в `measure_fingerprint.py` и повторённая здесь без изменений.
SAME_PLACE_PX = 32.0        # ближе — заведомо одно место
OTHER_PLACE_PX = 64.0       # дальше — заведомо разные; между ними неоднозначно

#: Сколько пар брать на прогон. Все пары — это n², и на 1500 наблюдениях это миллион
#: сравнений на прогон при 24 прогонах. Выборка фиксированным генератором даёт то же
#: число с тем же разбросом и не превращает замер в ночной.
PAIRS_PER_RUN = 4000


def _combination(bits: tuple[bool, ...]) -> dict[str, bool]:
    return dict(zip(MEANS, bits))


def _label(on: dict[str, bool]) -> str:
    names = [MEANS[k] for k, v in on.items() if v]
    return "+".join(names) if names else "контроль"


def _run(on: dict[str, bool], seed: int, *, observations: int) -> dict[str, Any]:
    """Один прогон: разведка по интерактивному миру, узлы и обе ошибки против истины."""
    profile = from_schema("ЗАМЕР-узнавание", capture_width=64, capture_height=48,
                          place_background_min_gap=60, place_background_min_n=40,
                          **on)
    world = InteractiveWorld(profile, seed=seed, n_outputs=16)
    graph = PlaceGraph.from_profile(profile)
    rng = np.random.default_rng(seed)

    assigned: list[str] = []
    truth: list[tuple[float, float]] = []
    curve: list[list[int]] = []
    for step in range(1, observations + 1):
        out = world.outputs[int(rng.integers(len(world.outputs)))]
        obs = world.step(Action.key(out, 200), with_audio=False)
        node = graph.see(obs.frame, step, seconds_per_seq=1 / 30.0,
                         mode=action_key(out, 200))
        assigned.append(node)
        # Истина доступна только исследователю: она из состояния мира, а не из кадра.
        truth.append((float(world.state.cam_x), float(world.state.cam_y)))
        if step % 250 == 0:
            curve.append([step, len(graph.places)])

    pairs = _pair_errors(assigned, truth, rng)
    half = observations // 2
    at_half = next((p for o, p in curve if o >= half), len(graph.places))
    tail = (len(graph.places) - at_half) / max(1, observations - half)

    threshold = None
    if graph.recognizer is not None:
        threshold = graph.recognizer.threshold().as_dict()

    return {
        "means": dict(on), "label": _label(on), "seed": seed,
        "observations": observations, "places": len(graph.places),
        "new_per_observation_tail": round(tail, 5),
        "curve": curve,
        "threshold": threshold,
        "rejected_by_novelty": (None if graph.recognizer is None
                                else graph.recognizer.rejected_by_novelty),
        **pairs,
    }


def _pair_errors(assigned: list[str], truth: list[tuple[float, float]],
                 rng: np.random.Generator) -> dict[str, Any]:
    """Обе ошибки против истины. Неоднозначные пары исключаются, а не приписываются."""
    n = len(assigned)
    same_total = same_split = 0          # заведомо одно место
    other_total = other_merged = 0       # заведомо разные места
    ambiguous = 0
    for _ in range(PAIRS_PER_RUN):
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i == j:
            continue
        dx = truth[i][0] - truth[j][0]
        dy = truth[i][1] - truth[j][1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < SAME_PLACE_PX:
            same_total += 1
            if assigned[i] != assigned[j]:
                same_split += 1
        elif dist >= OTHER_PLACE_PX:
            other_total += 1
            if assigned[i] == assigned[j]:
                other_merged += 1
        else:
            ambiguous += 1
    return {
        # Ложная тревога: одно место объявлено двумя. Граф пухнет, но след остаётся.
        "false_split": {"n": same_total, "fired": same_split,
                        "share": (same_split / same_total) if same_total else None},
        # Ложное подтверждение: разные места объявлены одним. Карта говорит «я тут
        # был», планировщик верит, и следа не остаётся. Опаснее (инвариант 32).
        "false_merge": {"n": other_total, "fired": other_merged,
                        "share": (other_merged / other_total) if other_total else None},
        "ambiguous_pairs": ambiguous,
    }


def _cards_doubled(seed: int = 11, things: int = 12, repaints: int = 4) -> dict[str, Any]:
    """A2.4: доля карточек, задвоенных сменой свойства. Контроль — прежний отпечаток.

    Мир объявлен здесь: `things` предметов, у каждого своя форма и положение, и каждый
    перекрашивается `repaints` раз. Истина известна по построению — предметов ровно
    `things`, — и в механику она не попадает.
    """
    rng = np.random.default_rng(seed)
    shapes = [(round(float(rng.uniform(0, 10)), 3), round(float(rng.uniform(0, 10)), 3))
              for _ in range(things)]
    colours = [round(float(c), 3) for c in rng.uniform(0, 10, repaints)]

    plain = Quantized(bucket=0.1)
    split = Split(inner=Quantized(bucket=0.1))
    plain_cards: set[str] = set()
    split_cards: set[str] = set()
    seq = 0
    for shape, position in shapes:
        for colour in colours:
            seq += 1
            plain_cards.add(plain([shape, position, colour]))
            got, _ = split.observe(
                {"shape": shape, "position": position, "colour": colour}, seq=seq)
            split_cards.add(got)

    return {
        "true_things": things, "repaints": repaints,
        "plain_cards": len(plain_cards), "split_cards": len(split_cards),
        # Доля задвоенных: лишние карточки сверх истины, отнесённые к истине.
        "plain_doubled_share": round((len(plain_cards) - things) / things, 4),
        "split_doubled_share": round((len(split_cards) - things) / things, 4),
        "property_events": len(split.changes),
        "split_state": split.state(),
        "unit": "прогон (мир объявлен здесь, истина по построению)",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--observations", type=int, default=OBSERVATIONS)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "recognition.json")
    a = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for bits in itertools.product((False, True), repeat=len(MEANS)):
        on = _combination(bits)
        for seed in SEEDS:
            rows.append(_run(on, seed, observations=a.observations))
            r = rows[-1]
            fs = r["false_split"]["share"]
            fm = r["false_merge"]["share"]
            print(f"  {r['label']:<26} сид {seed}: узлов {r['places']:5d}, "
                  f"хвост {r['new_per_observation_tail']:.4f}, "
                  f"ложных расщеплений {'—' if fs is None else f'{fs:.3f}'}, "
                  f"ложных слияний {'—' if fm is None else f'{fm:.3f}'}")

    by_label: dict[str, Any] = {}
    for label in {r["label"] for r in rows}:
        got = [r for r in rows if r["label"] == label]
        places = [r["places"] for r in got]
        tails = [r["new_per_observation_tail"] for r in got]
        fs = [r["false_split"]["share"] for r in got
              if r["false_split"]["share"] is not None]
        fm = [r["false_merge"]["share"] for r in got
              if r["false_merge"]["share"] is not None]
        adaptive = [r["threshold"]["value"] for r in got
                    if r["threshold"] and r["threshold"]["adaptive"]]
        by_label[label] = {
            "n_runs": len(got), "unit": "прогон",
            "places_median": statistics.median(places),
            "places_range": [min(places), max(places)],
            "tail_median": round(statistics.median(tails), 5),
            "false_split_median": round(statistics.median(fs), 4) if fs else None,
            "false_merge_median": round(statistics.median(fm), 4) if fm else None,
            "runs_with_adaptive_threshold": len(adaptive),
            "adaptive_threshold_median": (round(statistics.median(adaptive), 4)
                                          if adaptive else None),
        }

    control = by_label["контроль"]
    best = min(by_label.items(), key=lambda kv: kv[1]["places_median"])
    cards = _cards_doubled()

    data = {
        "rows": rows,
        "by_combination": by_label,
        "control": control,
        "fewest_nodes": {"label": best[0], **best[1]},
        "cards": cards,
        "truth": {"same_place_px": SAME_PLACE_PX, "other_place_px": OTHER_PLACE_PX,
                  "source": "состояние мира (cam_x, cam_y), отладочный канал",
                  "same_as": "tools/measure_fingerprint.py, раздел 31.2"},
        "means": MEANS, "seeds": list(SEEDS),
        "observations_per_run": a.observations,
        "pairs_per_run": PAIRS_PER_RUN,
        "unit": "прогон (сочетание средств, сид)",
        "claim": ("вклад каждого из трёх средств узнавания места измерен отдельно на "
                  "восьми сочетаниях, узлы считаются против наблюдений, и обе ошибки "
                  "предъявлены против истины из состояния мира"),
        "how_refuted": ("если ни одно сочетание не уменьшает число узлов против "
                        "контроля, средства не лечат ненасыщение, и менять надо "
                        "функцию отпечатка, а не узнавание"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    shown = a.out.relative_to(ROOT) if a.out.is_relative_to(ROOT) else a.out
    print(f"\nзаписано: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
