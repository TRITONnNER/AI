"""Изображение по TASK-16. Инвариант 28: одна картинка — одно утверждение.

Утверждение одно: **молчащая запись теряла не кадры, а оператора.** Ни одна из четырёх
величин не про качество захвата — все четыре про то, что человек видит и что у него
остаётся, и именно из-за них у проекта нет живого корпуса.

Запуск: `python3 tools/make_task16_figures.py [каталог]`. Числа — из
`docs/measurements/record_feedback.json`, ни одно не вписано рукой.
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

BEFORE = "#b8860b"      # до правок
AFTER = "#1a7f37"       # после правок
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


def fig_feedback(out: Path, data: dict[str, Any]) -> Path:
    """Одно утверждение: терялся оператор, а не кадры.

    Все величины приведены к доле, потому что мерят они разное — строки, прогоны,
    попытки, — и в одной шкале сравнимы только доли. Что именно взято за целое, сказано
    в подписи каждой строки: доля без названного знаменателя — не число.
    """
    a, b = data["summary"]["до"], data["summary"]["после"]
    stopped_a = int(a["из них прерванных"]) or 1
    stopped_b = int(b["из них прерванных"]) or 1
    runs = int(b["прогонов"]) or 1
    per_min = float(data["lines_per_minute_after"]) or 1.0
    defect = data["verify_defect"]

    rows = [
        (f"Оператор видит, что запись идёт\n(строк за минуту записи из {per_min:.0f})",
         float(data["lines_per_minute_before"]) / per_min, 1.0),
        (f"Прерванная запись помечена «прервано на N из M»\n"
         f"(прогонов из {stopped_b})",
         int(a["прерванных с отметкой"]) / stopped_a,
         int(b["прерванных с отметкой"]) / stopped_b),
        (f"Повторный запуск говорит, что делать\n(попыток из {runs})",
         int(a["повторный запуск даёт команду"]) / runs,
         int(b["повторный запуск даёт команду"]) / runs),
        (f"Запись со статикой проходит verify\n"
         f"(прерванных прогонов из {stopped_b})",
         int(defect["до"]["прерванных проходят verify"]) / stopped_a,
         int(defect["после"]["прерванных проходят verify"]) / stopped_b),
    ]

    fig, (ax,) = _fig(600)
    ys = range(len(rows))
    # Ось перевёрнута, поэтому «до» идёт на `y - 0.19`: при `y + 0.19` полоса рисуется
    # ниже, и подпись «верхняя — до правок» говорила бы обратное картинке. Ровно эта
    # ошибка была в изображении 22 и переворачивала его вывод.
    ax.barh([y - 0.19 for y in ys], [r[1] for r in rows], 0.34, color=BEFORE)
    ax.barh([y + 0.19 for y in ys], [r[2] for r in rows], 0.34, color=AFTER)
    for y, (_label, before, after) in enumerate(rows):
        ax.text(before + 0.015, y - 0.19, f"{before:.0%}", va="center",
                fontsize=LABEL_PT - 2, color=BEFORE if before else NEUTRAL)
        ax.text(after + 0.015, y + 0.19, f"{after:.0%}", va="center",
                fontsize=LABEL_PT - 2, color=AFTER)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=LABEL_PT - 3)
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel(
        "доля от названного в строке целого\n"
        "верхняя полоса — до правок, нижняя — после.\n"
        "Нижняя строка — дефект, не относящийся к правке: он был в обоих плечах,\n"
        "и его нашёл замер", fontsize=LABEL_PT)
    ax.invert_yaxis()

    _stamp(fig, n=f"{runs} прогонов на плечо, из них прерванных {stopped_b}",
           unit="прогон записи",
           compares="одно и то же прерывание в двух плечах: прежний цикл и нынешний")
    return _save(fig, out / "24-molchashchaya-zapis.png",
                 "Молчащая запись теряла не кадры, а оператора.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "record_feedback.json"
    if not src.exists():
        print("нет docs/measurements/record_feedback.json — сначала "
              "python3 tools/measure_record_feedback.py", file=sys.stderr)
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    made = [fig_feedback(base, data)]
    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
