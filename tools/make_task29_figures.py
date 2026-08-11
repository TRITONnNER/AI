"""Изображения по TASK-29. Инвариант 28: одна картинка — одно утверждение.

36. **A, отпечаток.** Граф мест рос от способа кормления и от разрешения функции, а не от
    шума: кривая насыщения по наблюдениям плюс разрешение против шага мира.

Запуск: `python3 tools/make_task29_figures.py [каталог]`. Числа — из
`docs/measurements/fingerprint.json`.
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
NEW = "#b42318"          # функция после TASK-29
OLD = "#30414f"          # функция, что была
MUTED = "#8a9199"        # способ кормления графа, а не функция
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


def fig_fingerprint(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: рост графа давал способ кормления и разрешение, а не шум.

    Две панели: кривая насыщения (узлы против наблюдений — ось наблюдений, не времени,
    иначе 13.7) и разрешение функции против шага мира. Панели «шум» нет нарочно: ноль
    ложных тревог из 39 пар рисовать нечем, и это сказано подписью.
    """
    fig, (ax1, ax2) = _fig(680, ncols=2)
    thr = data["resolution"][0]["threshold"]
    old_name, new_name = list(data["functions"])

    def curve(fn: str, every: int) -> list[list[int]]:
        for s in data["saturation"]:
            if s["function"] == fn and s["every"] == every and \
                    abs(s["threshold"] - 0.82) < 1e-9:
                return s["curve"]
        return []

    lines = (
        (curve(old_name, 100), MUTED, "каждый сотый оборот — так кормили в часовом прогоне"),
        (curve(old_name, 1), OLD, "наблюдения подряд, функция как была"),
        (curve(new_name, 1), NEW, "наблюдения подряд, функция после TASK-29"),
    )
    for pts, color, label in lines:
        if pts:
            ax1.plot([p[0] for p in pts], [p[1] for p in pts], color=color,
                     linewidth=2.0, label=label)
    ax1.set_xlabel("наблюдений\nпорог того же места 0.82; ось — наблюдения, а не время:\n"
                   "по времени кривая гнётся от замедления цикла (13.7)",
                   fontsize=LABEL_PT - 2)
    ax1.set_ylabel("узлов графа мест", fontsize=LABEL_PT - 1)
    ax1.legend(fontsize=LABEL_PT - 4, frameon=False, loc="upper left")

    tails = {(s["function"], s["every"]): s["new_per_observation_tail"]
             for s in data["saturation"] if abs(s["threshold"] - 0.82) < 1e-9}
    ax1.annotate("новых узлов на наблюдение в конце:\n"
                 f"{tails.get((old_name, 100), 0):.3f} · "
                 f"{tails.get((old_name, 1), 0):.3f} · {tails.get((new_name, 1), 0):.3f}",
                 (0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                 fontsize=LABEL_PT - 4, color=NEUTRAL)

    shifts = [s for s in data["shifts"] if s <= 32]
    for r, color in ((data["resolution"][0], OLD), (data["resolution"][1], NEW)):
        ys = [r["curve"][str(s)] for s in shifts]
        ax2.plot(shifts, ys, marker="o", markersize=6, color=color, linewidth=2.0,
                 label=f"{r['function']} — разрешение {r['resolution_px']} px")
    ax2.axhline(thr, color=NEUTRAL, linewidth=1.0, linestyle=(0, (4, 3)))
    ax2.annotate(f"порог того же места {thr}", (max(shifts), thr),
                 textcoords="offset points", xytext=(-4, 6), ha="right",
                 fontsize=LABEL_PT - 4, color=NEUTRAL)
    ax2.axvline(data["world_step_px"], color=NEUTRAL, linewidth=1.0,
                linestyle=(0, (1, 2)))
    ax2.annotate("шаг мира 6 px", (data["world_step_px"], 0.06),
                 textcoords="offset points", xytext=(6, 0), fontsize=LABEL_PT - 4,
                 color=NEUTRAL)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("сдвиг камеры, пикселей\nпри разрешении 4 px каждое действие мира\n"
                   "уводило вид в новое место — возврат был невозможен",
                   fontsize=LABEL_PT - 2)
    ax2.set_ylabel("похожесть отпечатков", fontsize=LABEL_PT - 1)
    ax2.legend(fontsize=LABEL_PT - 4, frameon=False, loc="lower left")

    fa = data["resolution"][0]["false_alarm"]
    fc = data["resolution"][0]["false_confirm"]
    fig.text(0.01, 0.086,
             f"шум отпечаток не путает: ложных тревог {fa['fired']} из {fa['n']} пар "
             f"(дизеринг ±1 и ±4, яркость +20, блоки 8×8), ложных подтверждений "
             f"{fc['fired']} из {fc['n']}",
             fontsize=LABEL_PT - 2, color=NEUTRAL)
    _stamp(fig, n=f"{data['resolution'][0]['n']} мира для разрешения, "
                 f"{len(data['saturation'])} прогонов для насыщения",
           unit="мир (сид) и прогон",
           compares="способ кормления графа против функции отпечатка, при одном пороге")
    return _save(fig, out / "36-otpechatok-nasyshchenie.png",
                 "Граф мест рос от способа кормления и разрешения функции, а не от шума.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "fingerprint.json"
    if not src.exists():
        print("нет docs/measurements/fingerprint.json — сначала "
              "python3 tools/measure_fingerprint.py", file=sys.stderr)
        return 2
    made = [fig_fingerprint(base, json.loads(src.read_text(encoding="utf-8")))]
    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
