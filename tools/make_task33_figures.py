"""Изображения по TASK-33. Инвариант 28: одна картинка — одно утверждение.

40. **A, цена ошибки в разведке.** Счётчики нажатий необратимого выхода разошлись при
    работающем пороге и при снятом — на всех пяти сидах, — и обе ошибки метки «дорого»
    предъявлены рядом.

Запуск: `python3 tools/make_task33_figures.py [каталог]`. Числа — из
`docs/measurements/reversibility.json`, то есть из `tools/measure_reversibility.py`.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

WIDTH_PX = 1200
DPI = 100
TITLE_PT = 15
LABEL_PT = 14
TICK_PT = 12

#: Цвет только как значение, и в подписи сказано какое.
GATE = "#b42318"         # порог осторожности работает / ложные срабатывания
OFF = "#30414f"          # порог снят / ложные подтверждения
NEUTRAL = "#30414f"


def _fig(height_px: int, ncols: int = 1):
    fig, axes = plt.subplots(1, ncols, figsize=(WIDTH_PX / DPI, height_px / DPI), dpi=DPI)
    axes = [axes] if ncols == 1 else list(axes)
    if len(axes) > 4:
        raise ValueError("панелей больше четырёх — значит утверждений несколько")
    for ax in axes:
        ax.tick_params(labelsize=TICK_PT)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig, axes


def _stamp(fig, *, n: str, unit: str, compares: str = "") -> None:
    fig.text(0.01, 0.048, f"n = {n}   ·   единица независимости: {unit}",
             fontsize=LABEL_PT - 1, color=NEUTRAL)
    if compares:
        fig.text(0.01, 0.012, f"сравнивается: {compares}",
                 fontsize=LABEL_PT - 2, color=NEUTRAL)


def _save(fig, path: Path, title: str) -> Path:
    if not title.endswith((".", "!", "?")) and ":" not in title:
        raise ValueError(f"заголовок «{title}» — тема, а не предложение")
    if len(title) > 96:
        raise ValueError(f"заголовок длиной {len(title)} символов обрежется")
    fig.suptitle(title, fontsize=TITLE_PT, x=0.01, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0.115, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"нарисовано: {path.name}")
    return path


def fig_error_cost(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: цена ошибки вошла в порядок проб, и это видно по счётчикам."""
    counts = data["verdict"]["counts_by_seed"]
    seeds = sorted(counts, key=int)
    gate = [counts[s]["порог"] for s in seeds]
    off = [counts[s]["снят"] for s in seeds]
    fig, (ax1, ax2) = _fig(660, ncols=2)

    xs = list(range(len(seeds)))
    ax1.bar([x - 0.18 for x in xs], gate, width=0.34, color=GATE,
            label="порог осторожности работает")
    ax1.bar([x + 0.18 for x in xs], off, width=0.34, color=OFF,
            label="порог снят (контроль)")
    for x, (g, n) in enumerate(zip(gate, off)):
        mark = "=" if g == n else "≠"
        ax1.annotate(f"{g} {mark} {n}", (x, max(g, n)), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=LABEL_PT - 4, color=NEUTRAL)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"сид {s}" for s in seeds], fontsize=LABEL_PT - 3)
    ax1.set_ylim(0, max(gate + off) * 1.35)
    ax1.set_ylabel("нажатий по-настоящему необратимого выхода", fontsize=LABEL_PT - 2)
    same = len(data["verdict"]["identical_counts_seeds"])
    ax1.set_xlabel(
        f"среда: необратимое не замолкает; различие только в пороге\n"
        f"совпадений {same} из {len(seeds)}; до правки 5 из 5\n"
        "направление расхождения разное: цена меняет\nповедение, но риск не снижает",
        fontsize=LABEL_PT - 3, loc="left")
    ax1.legend(fontsize=LABEL_PT - 4, frameon=False, loc="upper left")

    # Панель 2: обе ошибки метки «дорого» как предсказания следующей попытки отката.
    pred = data["verdict"]["predicted"]
    envs = list(pred)
    ys = list(range(len(envs)))
    alarms = [(pred[e]["false_alarm_share"] or 0.0) * 100 for e in envs]
    calms = [(pred[e]["false_confirmation_share"] or 0.0) * 100 for e in envs]
    ax2.barh([y + 0.18 for y in ys], alarms, height=0.32, color=GATE,
             label="ложные срабатывания, %")
    ax2.barh([y - 0.18 for y in ys], calms, height=0.32, color=OFF,
             label="ложные подтверждения, %")
    for y, e in zip(ys, envs):
        p = pred[e]
        ax2.annotate(f"{alarms[ys.index(y)]:.0f} % ({p['false_alarms']})",
                     (alarms[ys.index(y)], y + 0.18), textcoords="offset points",
                     xytext=(6, -4), fontsize=LABEL_PT - 4, color=NEUTRAL)
        ax2.annotate(f"{calms[ys.index(y)]:.0f} % ({p['false_confirmations']})",
                     (calms[ys.index(y)], y - 0.18), textcoords="offset points",
                     xytext=(6, -4), fontsize=LABEL_PT - 4, color=NEUTRAL)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(envs, fontsize=LABEL_PT - 3)
    ax2.set_xlim(0, max(alarms + calms) * 1.6 + 5)
    total = sum(pred[e]["attempts"] for e in envs)
    cov = min(pred[e]["coverage"] or 0.0 for e in envs)
    ax2.set_xlabel(
        f"доля попыток отката, % (в скобках — сколько)\n"
        "метка утверждает: «следующий откат не удастся»\n"
        f"попыток {total}, покрытие не ниже {cov:.0%}\n"
        "против истины мира — "
        f"{data['verdict']['label']['false_alarm_share']:.0%} ложных: другой вопрос",
        fontsize=LABEL_PT - 3, loc="left")
    ax2.legend(fontsize=LABEL_PT - 4, frameon=False, loc="lower right")

    _stamp(fig, n=f"{len(data['rows'])} прогонов, {total} попыток отката",
           unit="прогон (справа — попытка отката внутри прогона)",
           compares="счётчик нажатий необратимого при пороге против снятого; "
                    "предсказание метки против исхода попытки")
    return _save(fig, out / "40-cena-oshibki.png",
                 "Цена ошибки вошла в порядок проб: счётчики разошлись на всех пяти сидах.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "reversibility.json"
    if not src.exists():
        print("пропущено: reversibility.json — сначала "
              "python3 tools/measure_reversibility.py", file=sys.stderr)
        return 1
    made = [fig_error_cost(base, json.loads(src.read_text(encoding="utf-8")))]
    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
