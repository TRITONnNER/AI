"""Смерть контура: планировщик недоступен, агент продолжает и глупеет. TASK-29, D2.

`model_fallback` в схеме есть и **читается** (`perception/describers.py`, выбор описателя),
что подтверждает детектор мёртвых параметров: дефектных 0. Но ручка отвечает на вопрос «кого
спрашивать, если названного нет», а не на вопрос «что будет, если верхнего контура не станет
вовсе». Второе — про цикл, и здесь оно измеряется.

Планировщик может пропасть по обычным причинам: сервис не отвечает, кончился бюджет запросов,
машина занята. Требование одно: агент **не останавливается**, продолжает на рефлексах, и
падение видно **объективной величиной**, а не самоотчётом (инвариант 10).

## Что измеряется и в каких единицах

Единица независимости — **прогон**: обороты внутри прогона зависимы по построению (один мир,
одна цепочка). Три точки смерти × несколько сидов; печатается разброс.

- доля действий, за которые планировщик считает себя причиной, **до** и **после** смерти;
- продолжил ли агент действовать вовсе (действий после смерти);
- сколько раз мир остановился (обязан ноль: инвариант 3 смертью контура не отменяется).

## Инвариант 32: обе ошибки проверки «заметил»

Проверка «падение замечено» может ошибиться в две стороны, и обе считаются:

- **ложная тревога** — «заметил» объявлено там, где замечать было нечего: планировщик и до
  смерти не действовал ни разу. Контроль — прогон, где контур снимается на первом обороте;
- **ложное подтверждение** — «заметил» объявлено, а планировщик продолжает действовать.
  Контроль — прогон **без** смерти контура: там проверка обязана молчать.

Прогон: `python3 tools/measure_contour_death.py`.
Результат — `docs/measurements/contour_death.json`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.behaviour.arena import reach_place                        # noqa: E402
from harness.core.profile import from_schema                           # noqa: E402
from harness.livecycle import run                                      # noqa: E402

#: Обороты, на которых снимается верхний контур. Ранняя смерть проверяет, что агент вообще
#: способен идти без него; поздняя — что падение видно на фоне уже накопленной работы.
KILL_AT = (1, 200, 1000)
SEEDS = (4, 5, 6)
SECONDS = 0.8
HZ_SCALE = 40.0


def one(*, kill_at: int | None, seed: int) -> dict[str, Any]:
    prof = from_schema("СМЕРТЬ КОНТУРА", capture_width=64, capture_height=48)
    with tempfile.TemporaryDirectory(prefix="contour-death-") as tmp:
        got = run(Path(tmp) / "s", scenario=reach_place(prof), profile=prof,
                  seconds=SECONDS, seed=seed, hz_scale=HZ_SCALE,
                  kill_planner_at=kill_at)
    d = got.as_dict()
    d.update({"kill_at": kill_at, "seed": seed})
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="смерть верхнего контура")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "contour_death.json")
    a = ap.parse_args(argv)

    rows = [one(kill_at=k, seed=s) for k in KILL_AT for s in SEEDS]
    alive = [one(kill_at=None, seed=s) for s in SEEDS]        # контроль: контур жив

    by_kill: dict[str, Any] = {}
    for k in KILL_AT:
        mine = [r for r in rows if r["kill_at"] == k]
        by_kill[str(k)] = {
            "n": len(mine), "unit": "прогон",
            "before": statistics.median(r["planner_share_before"] or 0.0 for r in mine),
            "after": statistics.median(r["planner_share_after"] or 0.0 for r in mine),
            "delivered_after": statistics.median(r["delivered_after_death"] for r in mine),
            "noticed": sum(1 for r in mine if r["noticed_by_expectations"]),
            "world_never_paused": all(r["world_never_paused"] for r in mine),
        }

    # Обе ошибки проверки «заметил» (инвариант 32). Способ объявлен вместе с проверкой.
    early = [r for r in rows if r["kill_at"] == 1]
    false_alarm = sum(1 for r in early
                      if r["noticed_by_expectations"] and not r["planner_share_before"])
    false_confirm = sum(1 for r in alive if r["noticed_by_expectations"])
    errors = {
        "false_alarm": {
            "n": len(early), "fired": false_alarm,
            "share": false_alarm / len(early) if early else None, "unit": "прогон",
            "how": "контур снят на первом обороте: планировщик не успел подействовать ни "
                   "разу, замечать нечего — «заметил» здесь было бы ложной тревогой"},
        "false_confirm": {
            "n": len(alive), "fired": false_confirm,
            "share": false_confirm / len(alive) if alive else None, "unit": "прогон",
            "how": "контур жив весь прогон: планировщик действует, и «заметил» здесь было "
                   "бы ложным подтверждением — опаснее тревоги, потому что закрывает вопрос"},
    }

    print(f"{'смерть на обороте':>18} {'доля планировщика до':>21} "
          f"{'после':>7} {'действий после':>15} {'заметил':>9} {'мир не стоял':>13}")
    for k, v in by_kill.items():
        print(f"{k:>18} {v['before']:>21.2f} {v['after']:>7.2f} "
              f"{v['delivered_after']:>15.0f} {v['noticed']:>6}/{v['n']:<2} "
              f"{'да' if v['world_never_paused'] else 'НЕТ':>13}")
    print()
    ctl = statistics.median(r["by_layer"].get("planner", 0) for r in alive)
    print(f"контроль (контур жив): действий планировщика {ctl:.0f}, "
          f"«заметил» сработал {false_confirm} раз из {len(alive)}")
    print(f"ложных тревог {false_alarm}/{len(early)}, "
          f"ложных подтверждений {false_confirm}/{len(alive)}")
    print(f"\nединица независимости: прогон, сидов {len(SEEDS)}, "
          f"точек смерти {len(KILL_AT)}")

    data = {"rows": rows, "alive": alive, "by_kill": by_kill, "errors": errors,
            "kill_at": list(KILL_AT), "seeds": list(SEEDS), "unit": "прогон",
            "seconds": SECONDS, "hz_scale": HZ_SCALE,
            "claim": ("при недоступном верхнем контуре агент продолжает действовать на "
                      "рефлексах, а падение видно объективной величиной — доля действий, "
                      "за которые планировщик считает себя причиной, обращается в ноль"),
            "how_refuted": ("если после смерти контура действий нет вовсе, агент "
                            "останавливается вместе с планировщиком, и «продолжает на "
                            "рефлексах» неверно; если доля не падает — величина мерит не то")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"записано: {a.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
