"""Оживает ли цель с внешним судьёй от канала оценки. TASK-33, направление C.

`JudgedGoal` был мёртв не по устройству, а по отсутствию судьи: детектор режима отвечал
«внешний судья не определён» на всех пяти доменах, и калибровка делила на ноль. Здесь
проверяется, что канал отметок (`model/marks.py`) её оживляет, и **чему именно** он даёт
измерить.

## Судья синтетический, и это сказано вслух

Оператора изображает правило, известное исследователю и неизвестное агенту: доля `p`
отметок «получилось», остальное «нет». Это законный стенд для механизма — как `Undoable` в
замере обратимости, — но он **ничего не говорит о живом операторе**: живой судья
нестационарен, а этот стационарен по построению. Что из этого следует, сказано в разделе
«чего этот замер не показывает» (`MEASUREMENT.md`, 36.4).

## Предрегистрация: до какого числа может дойти калибровка

`ReflectedSelf` не имеет признаков: он помнит `mu` по паре «судья × предмет» и ничего
больше. Значит лучшее возможное предсказание — базовая доля `p`, и средний промах при таком
предсказании считается заранее:

    E|p − оценка| = p(1 − p) + (1 − p)p = 2p(1 − p)

Отсюда три ожидания, записанные **до** прогона:

| судья | p | лучший возможный промах | лучше наугад (< 0.5) |
|---|---|---|---|
| уверенный | 0.9 | 0.18 | да |
| неровный | 0.7 | 0.42 | да |
| монета | 0.5 | 0.50 | **нет** |

Третья строка — не провал механизма, а его граница: судью, который отвечает наугад,
предсказать нечем, и `better_than_nothing` обязан стать ложью. Если он окажется истиной,
значит калибровка считается неверно.

## Обе ошибки проверки «цель выполнена» (инварианты 31 и 32)

Метка — это **предсказание** агента, сравненное с порогом успеха цели:

- **ложная тревога** — предсказал успех, судья отказал;
- **ложное подтверждение** — предсказал неудачу, судья одобрил. Опаснее: агент не станет
  делать то, что на самом деле оценили бы хорошо, и никто об этом не узнает.

Покрытие предъявляется отдельно: у первой оценки по каждой паре «судья × предмет»
предсказания нет вовсе, и она в доли не входит.

## Единица независимости

**Оценка** (одна пара «предсказал / получил»). Оценки одного судьи об одной работе
зависимы, разные работы независимы. `n` считает оценки, а не эпизоды.

Прогон: `python3 tools/measure_judge.py`. Результат — `docs/measurements/judge.json`.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.core.profile import from_schema                                # noqa: E402
from harness.model.marks import Mark, MarkQueue, Source                     # noqa: E402
from harness.model.reflected import ReflectedSelf, judged_goal              # noqa: E402

SEEDS = (1, 2, 3, 4, 5)
EPISODES = 120
JUDGES = (("уверенный", 0.9), ("неровный", 0.7), ("монета", 0.5))
WANT = 0.6


def best_possible(p: float) -> float:
    """Лучший возможный средний промах для модели без признаков."""
    return 2.0 * p * (1.0 - p)


def one(*, seed: int, p: float, trust: float, source: Source = Source.OPERATOR,
        episodes: int = EPISODES) -> dict[str, Any]:
    """Один прогон: `episodes` эпизодов, после каждого отметка оператора."""
    prof = from_schema("СУДЬЯ", capture_width=64, capture_height=48,
                       mark_trust_operator=trust)
    queue = MarkQueue.from_profile(prof)
    mind = ReflectedSelf()
    rng = random.Random(seed)
    judge, aspect = "SYM_JUDGE_1", "SYM_ASPECT_1"

    misses: list[float] = []
    predictions: list[tuple[float | None, float]] = []
    settled_rows: list[dict[str, Any]] = []
    closed_without_judgement = 0
    pending_seen = 0
    goals = []
    for ep in range(episodes):
        goal = judged_goal(goal_id=f"g{ep}", judge=judge, aspect=aspect, want=WANT,
                           budget_ticks=100, branch="судья", seq=ep)
        goals.append(goal)
        queue.ask(goal, mind, at=ep)
        # Сторож инварианта 10: пока судья не ответил, цель не выполнена и не провалена.
        if goal.passed is None:
            pending_seen += 1
        # Оператор отвечает не сразу: очередь асинхронна, задержка от одного до трёх тактов.
        mark = Mark.DONE if rng.random() < p else Mark.FAILED
        queue.answer(goal.goal.id, mark, source=source, at=ep + rng.randint(1, 3))
        for row in queue.settle(goals, mind):
            if row["settled"]:
                settled_rows.append(row)
                if row.get("miss") is not None:
                    misses.append(row["miss"])
                    predictions.append((row["predicted"], row["value"]))
            elif row.get("why", "").startswith("цель"):
                closed_without_judgement += 1

    cal = mind.calibration()
    graded = [(pred, val) for pred, val in predictions if pred is not None]
    alarms = [(pred, val) for pred, val in graded if pred >= WANT]
    calms = [(pred, val) for pred, val in graded if pred < WANT]
    return {
        "seed": seed, "p": p, "trust": trust, "source": str(source),
        "episodes": episodes,
        "unit": "оценка",
        "judged": cal["judged"],
        "miss_mean": cal["miss_mean"],
        "miss_last30": (statistics.mean(misses[-30:]) if len(misses) >= 30 else None),
        "better_than_nothing": cal["better_than_nothing"],
        "best_possible": best_possible(p),
        "mu": mind.expect(judge, aspect).mu,
        "mu_error": abs(mind.expect(judge, aspect).mu - p),
        "closed_without_judgement": closed_without_judgement,
        "pending_seen": pending_seen,
        # Покрытие — доля закрытых целей, у которых предсказание было. У первой встречи с
        # судьёй его нет по построению, и держать её в долях значило бы называть промахом
        # отсутствие предсказания.
        "settled": len(settled_rows),
        "coverage": (len(graded) / len(settled_rows)) if settled_rows else None,
        "false_alarms": sum(1 for pred, val in alarms if val < WANT),
        "false_alarm_share": (sum(1 for pred, val in alarms if val < WANT) / len(alarms)
                              if alarms else None),
        "false_confirms": sum(1 for pred, val in calms if val >= WANT),
        "false_confirm_share": (sum(1 for pred, val in calms if val >= WANT) / len(calms)
                                if calms else None),
        "queue": queue.stats(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="канал оценки от оператора")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "judge.json")
    a = ap.parse_args(argv)

    rows = [one(seed=s, p=p, trust=1.0) for _, p in JUDGES for s in SEEDS]
    # Доверие: та же расстановка при доверии 0.25. Ожидание — ожидание судьи движется
    # медленнее, промах при этом не обязан улучшиться.
    low = [one(seed=s, p=0.9, trust=0.25) for s in SEEDS]

    def med(rs: list[dict[str, Any]], key: str) -> float | None:
        vals = [r[key] for r in rs if r[key] is not None]
        return statistics.median(vals) if vals else None

    cases: dict[str, Any] = {}
    for name, p in JUDGES:
        mine = [r for r in rows if r["p"] == p]
        cases[name] = {
            "p": p, "n": len(mine), "unit": "прогон (оценок в прогоне "
                                            f"{mine[0]['judged']})",
            "judged": mine[0]["judged"],
            "best_possible": best_possible(p),
            "miss_mean": med(mine, "miss_mean"),
            "miss_last30": med(mine, "miss_last30"),
            "mu_error": med(mine, "mu_error"),
            "better_than_nothing": sum(1 for r in mine
                                       if r["better_than_nothing"] is True),
            "indistinguishable": sum(1 for r in mine
                                     if r["better_than_nothing"] is None),
            "false_alarms": sum(r["false_alarms"] for r in mine),
            "false_confirms": sum(r["false_confirms"] for r in mine),
            "false_alarm_share": med(mine, "false_alarm_share"),
            "false_confirm_share": med(mine, "false_confirm_share"),
            "coverage": med(mine, "coverage"),
        }
    cases["уверенный, доверие 0.25"] = {
        "p": 0.9, "n": len(low), "unit": "прогон", "judged": low[0]["judged"],
        "best_possible": best_possible(0.9),
        "miss_mean": med(low, "miss_mean"), "miss_last30": med(low, "miss_last30"),
        "mu_error": med(low, "mu_error"),
        "better_than_nothing": sum(1 for r in low
                                   if r["better_than_nothing"] is True),
        "indistinguishable": sum(1 for r in low if r["better_than_nothing"] is None),
        "false_alarms": sum(r["false_alarms"] for r in low),
        "false_confirms": sum(r["false_confirms"] for r in low),
        "false_alarm_share": med(low, "false_alarm_share"),
        "false_confirm_share": med(low, "false_confirm_share"),
        "coverage": med(low, "coverage"),
    }

    def show(x: float | None, digits: int = 3) -> str:
        return "—" if x is None else f"{x:.{digits}f}"

    print(f"{'судья':<26} {'оценок':>7} {'промах':>8} {'лучший возможный':>18} "
          f"{'ошибка mu':>10} {'лучше наугад':>22}")
    for name, v in cases.items():
        verdict_col = (f"{v['better_than_nothing']} из {v['n']}"
                       + (f", не отл. {v['indistinguishable']}"
                          if v["indistinguishable"] else ""))
        print(f"{name:<26} {v['judged']:>7} {show(v['miss_mean']):>8} "
              f"{show(v['best_possible']):>18} {show(v['mu_error']):>10} "
              f"{verdict_col:>22}")

    print()
    print(f"{'судья':<26} {'ложных тревог':>14} {'ложных подтверждений':>21} "
          f"{'покрытие':>9}")
    for name, v in cases.items():
        def pct(x: float | None) -> str:
            return "—" if x is None else f"{x:.0%}"
        print(f"{name:<26} "
              f"{pct(v['false_alarm_share']) + ' (' + str(v['false_alarms']) + ')':>14} "
              f"{pct(v['false_confirm_share']) + ' (' + str(v['false_confirms']) + ')':>21} "
              f"{pct(v['coverage']):>9}")

    # Сторож инварианта 10 — главное число этого замера, и оно обязано быть нулём.
    closed = sum(r["closed_without_judgement"] for r in rows + low)
    waited = sum(r["pending_seen"] for r in rows + low)
    verdict: dict[str, Any] = {
        "alive": all(c["judged"] > 0 for c in cases.values()),
        "closed_without_judgement": closed,
        "pending_seen": waited,
        "coin_is_not_better_than_nothing": (cases["монета"]["better_than_nothing"] == 0),
        "coin_indistinguishable": cases["монета"]["indistinguishable"],
        "steady_is_better_than_nothing": (cases["уверенный"]["better_than_nothing"]
                                         == len(SEEDS)),
        "trust_slows_the_shift": (
            (cases["уверенный, доверие 0.25"]["mu_error"] or 0.0)
            > (cases["уверенный"]["mu_error"] or 0.0)),
    }
    print()
    print(f"закрыто целей без оценки судьи: {closed} (обязан быть ноль — сторож "
          "инварианта 10)")
    print(f"целей, побывавших в состоянии «судья не ответил»: {waited}")
    print("монета не объявлена «лучше наугад»: "
          f"{'да' if verdict['coin_is_not_better_than_nothing'] else 'НЕТ — калибровка объявляет победу шумом'}"
          f" (не отличимо от наугад: {verdict['coin_indistinguishable']} из {len(SEEDS)})")
    print("доверие 0.25 замедляет сдвиг ожидания: "
          f"{'да' if verdict['trust_slows_the_shift'] else 'НЕТ — параметр не читается'}")
    print(f"\nединица независимости: оценка; прогонов {len(rows) + len(low)}, "
          f"эпизодов в прогоне {EPISODES}")

    data = {"rows": rows + low, "cases": cases, "verdict": verdict,
            "seeds": list(SEEDS), "episodes": EPISODES, "want": WANT,
            "unit": "оценка",
            "claim": ("канал отметок оживляет цель с внешним судьёй: калибровка считается "
                      "и доходит до предрегистрированного предела 2p(1−p), а сторож "
                      "инварианта 10 остаётся нулём"),
            "how_refuted": ("если промах не сходится к 2p(1−p), калибровка считается не "
                            "то; если монета оказывается «лучше наугад», считается "
                            "неверно; если хоть одна цель закрылась без оценки судьи, "
                            "нарушен инвариант 10")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"записано: {a.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
