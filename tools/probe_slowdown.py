"""Почему цикл замедляется за час: что именно накапливается. TASK-24, D, разбор находки.

Часовой прогон показал, что оборотов в секунду падает в девять раз — с 4086 до 440 за час, —
хотя ожидание было «не растёт». Ожидание разошлось, и вопрос «из-за чего» отдельный от вопроса
«разошлось ли». Здесь он и решается: тот же цикл крутится короткими отрезками, каждый раз с
выключенной одной частью, и сравниваются наклоны.

Выключается по одной части за прогон, потому что две сразу не различить:

- **как есть** — то, что мерил час;
- **без графа мест** — отпечаток каждого сотого кадра не считается. Граф растёт, и сравнение
  нового отпечатка со всеми узлами дорожает вместе с ним;
- **без журнала** — действия не дозаписываются. Запись с `flush` на каждое действие стоит
  постоянно, но проверить это надо числом, а не рассуждением;
- **без обхода каталога** — не считается размер сессии. Обход дерева файлов делается на
  границе отрезка, и его время попадает в **следующий** отрезок, то есть **прибор** способен
  изобразить замедление цикла;
- **пусто** — ни того, ни другого, ни третьего: остаётся только мир и планировщик.

Единица независимости — **прогон** (отрезки внутри прогона зависимы: одно и то же накопление).
Наклон считается по отрезкам первой и последней трети, как в самом часовом замере.

    python3 tools/probe_slowdown.py --seconds 90
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np                                                    # noqa: E402

from harness.behaviour.arena import Arena, reach_place                 # noqa: E402
from harness.behaviour.contours import Scheduler                       # noqa: E402
from harness.core.journal import Actor, Kind                           # noqa: E402
from harness.core.profile import from_schema                           # noqa: E402
from harness.livecycle import Proposal, babbler_contour, slow_planner   # noqa: E402
from harness.model.places import PlaceGraph, fingerprint               # noqa: E402
from harness.session import Recorder                                   # noqa: E402
from measure_hour import dir_bytes, slope_and_plateau                  # noqa: E402

CASES = ("как есть", "без графа мест", "без журнала", "без обхода каталога", "пусто")
SEGMENT_S = 10.0


def one(case: str, *, seconds: float, root: Path) -> dict[str, Any]:
    """Один прогон с выключенной одной частью. Возвращает отрезочные частоты."""
    use_graph = case in ("как есть", "без журнала", "без обхода каталога")
    use_journal = case in ("как есть", "без графа мест", "без обхода каталога")
    use_walk = case in ("как есть", "без графа мест", "без журнала")

    profile = from_schema("ЧАС", capture_width=64, capture_height=48)
    session_path = root / f"probe-{CASES.index(case)}"
    rng = np.random.default_rng(1)
    graph = PlaceGraph()
    rates: list[float] = []
    times: list[float] = []

    with Recorder(session_path, profile=profile, source="probe", synthetic=True,
                  note="TASK-24 D: разбор замедления") as rec:
        arena = Arena(reach_place(profile), journal=rec.journal, seed=1)
        arena.start(rec.clocks.stamp())
        sched = Scheduler.standard(profile, reflex=babbler_contour(arena, rng),
                                   planner=slow_planner(arena))
        started = time.monotonic()
        next_point = SEGMENT_S
        loops = prev_loops = 0
        prev_t = 0.0
        while True:
            now = time.monotonic()
            if now - started >= seconds:
                break
            loops += 1
            sched.step()
            choice: Proposal | None = None
            for contour in sched.contours:
                if isinstance(contour.best, Proposal):
                    choice, contour.best = contour.best, None
                    break
            stamp = rec.clocks.stamp()
            obs = arena.step(choice.action if choice else None, stamp)
            if use_graph and loops % 100 == 0 and getattr(obs, "frame", None) is not None:
                graph.observe(fingerprint(obs.frame), loops, seconds_per_seq=1.0 / 30.0)
            if choice is not None and use_journal:
                rec.journal.append(Kind.ACTION, stamp, Actor.AGENT, choice.layer,
                                   action=choice.action, event={"проба": "разбор"})
            if arena.finished():
                arena.close(arena.finished(), rec.clocks.stamp())
                arena.start(rec.clocks.stamp())
            if now - started >= next_point:
                elapsed = now - started
                if use_walk:
                    dir_bytes(session_path)
                rates.append((loops - prev_loops) / max(1e-9, elapsed - prev_t))
                times.append(elapsed)
                prev_loops, prev_t = loops, elapsed
                next_point += SEGMENT_S
        arena.close(arena.finished() or "прервано", rec.clocks.stamp())

    fit = slope_and_plateau(times, rates)
    return {"case": case, "loops": loops, "places": len(graph.places),
            "rates": rates, "times": times, "unit": "прогон",
            "first": rates[0] if rates else 0.0, "last": rates[-1] if rates else 0.0,
            "drop": (rates[0] / rates[-1]) if rates and rates[-1] else None,
            "slope_per_s": fit["slope_per_s"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="разбор замедления цикла")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "slowdown.json")
    a = ap.parse_args(argv)

    import tempfile
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="probe-slowdown-") as tmp:
        for case in CASES:
            rows.append(one(case, seconds=a.seconds, root=Path(tmp)))
            r = rows[-1]
            drop = "—" if r["drop"] is None else f"{r['drop']:.2f}×"
            print(f"{case:<22} оборотов {r['loops']:>9}  "
                  f"первый отрезок {r['first']:>7.0f}/с  последний {r['last']:>7.0f}/с  "
                  f"падение {drop:>7}  мест {r['places']}", flush=True)

    data = {"rows": rows, "seconds": a.seconds, "segment_s": SEGMENT_S,
            "unit": "прогон", "cases": list(CASES),
            "claim": ("замедление цикла за час вызвано накоплением в одной из названных "
                      "частей, и выключение этой части убирает падение"),
            "how_refuted": ("если падение одинаково во всех расстановках, включая «пусто», "
                            "причина не в них, и искать надо в мире или планировщике")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nзаписано: {a.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
