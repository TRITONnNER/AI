"""Каскад восприятия: доля кадров по ступеням, отношение и цена. SPEC-FULL, A3.

Предрегистрация — `MEASUREMENT.md`, раздел 37. Она записана до этого прогона, и
менять её задним числом нельзя: смысл был именно в том, чтобы «получилось много»
нельзя было объявить ожидаемым.

## Единица независимости — прогон, а не кадр

`n` считает прогоны (домен, сид, доля бездействия), потому что кадры внутри прогона
зависимы: соседний похож на предыдущий ровно потому, что мир не успел измениться.
Доля, посчитанная по кадрам, дала бы `n` в шестьсот раз больше и вывод во столько же
раз увереннее, чем он есть, — та самая псевдорепликация, что обесценила 1913 «планов»
(инвариант 22).

## Доля бездействия — ось, а не настройка

Сколько кадров дойдёт до глубоких ступеней, определяется прежде всего тем, сколько
агент стоит. Замер при одном значении измерял бы выбор экспериментатора, поэтому доля
бездействия здесь ось: 0.0, 0.25, 0.5, 0.8. При 0.0 запирать нечего по построению, и
гейт обязан это показать, а не сгладить.

## Квантили ошибки печатаются рядом с порогом

Порог `cascade_predicted_error` сравнивается с величиной, чей рабочий диапазон заранее
неизвестен. Детектор режима трижды сравнивал признак с нулём вместо фона домена
(раздел 35), и здесь та же ловушка: если весь диапазон ошибки лежит ниже порога, замер
вакуумен, а не отрицателен. Отличить одно от другого можно только по распределению,
поэтому оно в выводе.

Прогон: `python3 tools/measure_cascade.py`. Результат — `docs/measurements/cascade.json`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.core.action import Action                       # noqa: E402
from harness.core.profile import from_schema                 # noqa: E402
from harness.corpus.domains import DOMAINS, make_domain      # noqa: E402
from harness.perception.cascade import (CHANGE, FINGERPRINT, FLOW, MODEL,  # noqa: E402
                                        STAGE_NAMES, STAGES, Cascade)
from harness.perception.describers import AskGate            # noqa: E402
from harness.vision.predict import PredictionError           # noqa: E402

#: Оси замера. Объявлены здесь и повторены в предрегистрации.
DOMAIN_NAMES = ("game", "document", "desktop", "video", "depth")
SEEDS = (11, 12, 13)
IDLE_SHARES = (0.0, 0.25, 0.5, 0.8)
FRAMES = 600

#: Ожидание из предрегистрации, 37.3. Печатается рядом с полученным.
EXPECTED_MODEL_SHARE = (0.02, 0.05)
EXPECTED_RATIO = (100.0, 300.0)
#: Потолок из критерия запуска A4, пункт 5.
LAUNCH_CEILING = 0.10


def _run(domain_name: str, seed: int, idle: float, *, frames: int,
         profile: Any) -> dict[str, Any]:
    """Один прогон. Возвращает доли по ступеням, цену и квантили ошибки."""
    domain = make_domain(domain_name, profile, seed=seed)
    outputs = list(domain.outputs)
    rng = np.random.default_rng(seed * 1000 + int(idle * 100))

    error = PredictionError(profile)
    gate = AskGate(float(profile.parameters["model_min_novelty"]),
                   min_gap_cycles=int(profile.parameters["model_min_gap_cycles"]))
    cascade = Cascade.from_profile(profile, ask_gate=gate)

    values: list[float] = []
    for t in range(frames):
        act = None if rng.random() < idle else Action.key(
            outputs[int(rng.integers(0, len(outputs)))], 200)
        frame = domain.step(act, with_audio=False).frame
        sample = error.feed(frame)
        value = None if sample is None else sample.value
        if value is not None:
            values.append(value)
        verdict = cascade.admit(frame, error=value, t_self=t)
        # Ступени 1–3 здесь не исполняются: замеряется гейт и его собственная цена,
        # а не работа слоёв. Цена слоёв измерена отдельно, в `test_perception_layers`,
        # и подставлять её сюда значило бы смешать измеренное с заявленным.
        del verdict

    stats = cascade.stats()
    qs = ([round(float(np.quantile(values, q)), 6) for q in (0.05, 0.5, 0.95)]
          if values else [None, None, None])
    return {
        "domain": domain_name, "seed": seed, "idle": idle, "frames": cascade.frames,
        "share_reaching": stats["share_reaching"],
        "unchanged": stats["unchanged"],
        "quiet": stats["quiet"],
        "ratio": stats["ratio_change_to_model"],
        "stage0_ms_per_call": stats["stages"][CHANGE]["ms_per_call"],
        "error_quantiles": {"p05": qs[0], "p50": qs[1], "p95": qs[2]},
        "error_n": len(values),
    }


def _stage_costs(profile: Any, *, frames: int = 120, seed: int = 11) -> dict[str, Any]:
    """Цена каждой ступени на вызов, на этой машине.

    Ступени 1 и 2 исполняются настоящими слоями, а не имитацией. Ступень 3 не
    исполняется вовсе: большой модели за файрволом здесь нет, и подставить сюда
    правдоподобное число было бы худшим, что можно сделать, — оно молча вошло бы в
    бюджет. Поэтому её цена остаётся `null`, и рядом сказано почему.
    """
    from harness.perception.layers import FarLayer, MidLayer, NearLayer, ScreenLayer

    domain = make_domain("game", profile, seed=seed)
    outputs = list(domain.outputs)
    rng = np.random.default_rng(seed)
    frames_list = [domain.step(Action.key(outputs[int(rng.integers(0, len(outputs)))], 200),
                               with_audio=False).frame for _ in range(frames)]

    p = profile.parameters
    made = {
        FLOW: [ScreenLayer(profile, float(p["layer_screen_hz"])),
               NearLayer(profile, float(p["layer_near_hz"]))],
        FINGERPRINT: [MidLayer(profile, float(p["layer_mid_hz"])),
                      FarLayer(profile, float(p["layer_far_hz"]))],
    }
    out: dict[str, Any] = {}
    for stage, layers in made.items():
        started = time.perf_counter()
        for i, frame in enumerate(frames_list):
            for layer in layers:
                layer.update(frame, i)
        ms = (time.perf_counter() - started) * 1000.0 / len(frames_list)
        out[STAGE_NAMES[stage]] = round(ms, 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "cascade.json")
    a = ap.parse_args()

    profile = from_schema("ЗАМЕР-каскад", capture_width=320, capture_height=180)
    unknown = sorted(set(DOMAIN_NAMES) - set(DOMAINS))
    if unknown:
        raise SystemExit(f"нет доменов: {unknown}")

    rows: list[dict[str, Any]] = []
    for idle in IDLE_SHARES:
        for name in DOMAIN_NAMES:
            for seed in SEEDS:
                rows.append(_run(name, seed, idle, frames=a.frames, profile=profile))
                r = rows[-1]
                print(f"  {name:9s} сид {seed} бездействие {idle:.2f}: "
                      f"до модели {r['share_reaching']['model']:.4f}, "
                      f"неизменившихся {r['unchanged']['share']:.3f}, "
                      f"отношение {r['ratio']}")

    #: Сводка **по прогонам**, а не по кадрам: единица независимости — прогон.
    by_idle: dict[str, Any] = {}
    for idle in IDLE_SHARES:
        got = [r for r in rows if r["idle"] == idle]
        shares = [r["share_reaching"]["model"] for r in got]
        ratios = [r["ratio"] for r in got if r["ratio"] is not None]
        still = [r["unchanged"]["share"] for r in got]
        by_idle[str(idle)] = {
            "n_runs": len(got),
            "unit": "прогон",
            "model_share_median": round(statistics.median(shares), 6),
            "model_share_range": [round(min(shares), 6), round(max(shares), 6)],
            "runs_over_launch_ceiling": sum(s > LAUNCH_CEILING for s in shares),
            "ratio_median": (round(statistics.median(ratios), 1) if ratios else None),
            "runs_never_reaching_model": len(got) - len(ratios),
            "unchanged_share_median": round(statistics.median(still), 4),
        }

    all_shares = [r["share_reaching"]["model"] for r in rows]
    p50 = [r["error_quantiles"]["p50"] for r in rows
           if r["error_quantiles"]["p50"] is not None]
    p95 = [r["error_quantiles"]["p95"] for r in rows
           if r["error_quantiles"]["p95"] is not None]
    flow_gate = float(profile.parameters["cascade_predicted_error"])
    model_gate = float(profile.parameters["model_min_novelty"])

    # Вакуумность проверяется **по домену и по обоим порогам**, а не по максимуму
    # across прогонов и одному порогу. Первая редакция этой проверки сверяла разброс
    # ошибки только с `cascade_predicted_error` и брала максимум по всем прогонам —
    # и ответила «не вакуумно» там, где для четырёх доменов из пяти ступень 3
    # недостижима по построению. Это было ложное подтверждение собственной проверки
    # (инвариант 32), и поймал его не код, а разбор прогонов по доменам.
    per_domain: dict[str, Any] = {}
    for name in DOMAIN_NAMES:
        got = [r for r in rows if r["domain"] == name]
        q50 = [r["error_quantiles"]["p50"] for r in got
               if r["error_quantiles"]["p50"] is not None]
        q95 = [r["error_quantiles"]["p95"] for r in got
               if r["error_quantiles"]["p95"] is not None]
        top = max(q95) if q95 else None
        per_domain[name] = {
            "p50_median": round(statistics.median(q50), 6) if q50 else None,
            "p95_max": round(top, 6) if top is not None else None,
            "reaches_flow_gate": bool(top is not None and top >= flow_gate),
            "reaches_model_gate": bool(top is not None and top >= model_gate),
            "model_calls_median": statistics.median(
                [round(r["share_reaching"]["model"] * r["frames"]) for r in got]),
        }
    dead = sorted(n for n, v in per_domain.items() if not v["reaches_model_gate"])
    vacuous = len(dead) == len(DOMAIN_NAMES)

    costs = _stage_costs(profile)
    costs_row = {"change": round(statistics.median(
        [r["stage0_ms_per_call"] for r in rows if r["stage0_ms_per_call"] is not None]), 4)}
    costs_row.update(costs)
    costs_row["model"] = None       # модели за файрволом здесь нет; см. `_stage_costs`

    data = {
        "rows": rows,
        "by_idle": by_idle,
        "stage_ms_per_call": costs_row,
        "stage_cost_note": ("цена ступени 3 не измерена: большой модели за файрволом в "
                            "этом окружении нет. Правдоподобное число здесь молча вошло "
                            "бы в бюджет, поэтому стоит null"),
        "error_scale": {
            "flow_gate": flow_gate,
            "model_gate": model_gate,
            "p50_across_runs": round(statistics.median(p50), 6) if p50 else None,
            "p95_max_across_runs": round(max(p95), 6) if p95 else None,
            "per_domain": per_domain,
            "domains_below_model_gate": dead,
            "vacuous_by_scale": vacuous,
            "vacuous_domains_share": f"{len(dead)}/{len(DOMAIN_NAMES)}",
            "why": ("если рабочий диапазон ошибки домена лежит ниже порога, ступень "
                    "недостижима по построению, и «доля ниже потолка» есть ложное "
                    "подтверждение, а не выполненный критерий: порог сравнивается не "
                    "с тем, с чем надо (та же ошибка, что в разделе 35)"),
        },
        "preregistered": {
            "model_share": list(EXPECTED_MODEL_SHARE),
            "ratio": list(EXPECTED_RATIO),
            "launch_ceiling": LAUNCH_CEILING,
            "expected_first_run_higher": True,
            "where": "MEASUREMENT.md, раздел 37",
        },
        "outcome": {
            "model_share_median_all": round(statistics.median(all_shares), 6),
            "runs": len(rows),
            "runs_over_launch_ceiling": sum(s > LAUNCH_CEILING for s in all_shares),
            # Критерий A4.5 «ниже 10 %» формально выполнен. Засчитывать его целиком
            # нельзя: на доменах из `domains_below_model_gate` ступень 3 недостижима
            # по построению, и там ноль означает «механизм не работал», а не
            # «механизм уложился». Ровно такое «сошлось» инвариант 32 называет
            # опаснее ложной тревоги: тревогу проверяют, а совпадение закрывают.
            "launch_ceiling_met_but_vacuous_on": dead,
            "runs_calling_model_more_than_once": sum(
                round(r["share_reaching"]["model"] * r["frames"]) > 1 for r in rows),
            "stage_cost_monotonic": (
                costs_row["change"] <= costs_row["flow"] <= costs_row["fingerprint"]),
        },
        "domains": list(DOMAIN_NAMES), "seeds": list(SEEDS),
        "idle_shares": list(IDLE_SHARES), "frames_per_run": a.frames,
        "unit": "прогон (домен, сид, доля бездействия)",
        "claim": ("каскад запирает ступени выше нулевой на неизменившемся кадре и на "
                  "кадре, совпавшем с предсказанием; доля кадров, доходящих до "
                  "большой модели, и отношение нулевой ступени к третьей измерены "
                  "против предрегистрации раздела 37"),
        "how_refuted": ("если доля, доходящая до ступени 3, выше 10 %, критерий "
                        "запуска A4 (пункт 5) не выполнен и бюджет не сходится; если "
                        "доля не зависит от доли бездействия, гейт не подключён к "
                        "тому, что меняется"),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    shown = a.out.relative_to(ROOT) if a.out.is_relative_to(ROOT) else a.out
    print(f"\nзаписано: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
