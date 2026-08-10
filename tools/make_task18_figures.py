"""Изображение по TASK-18. Инвариант 28: одна картинка — одно утверждение.

Утверждение одно: **6.5 и 246 кадр/с — это два разных экрана, а не две машины.** Стоимость
кадра задаётся тем, насколько меняется экран, и меняется на порядок: от 4 КиБ у
неподвижного до 2026 КиБ у несжимаемого. Оператор снял два прогона на одной машине, и без
доли изменения они выглядели загадкой.

Запуск: `python3 tools/make_task18_figures.py [каталог]`. Числа — из
`docs/measurements/record_cost.json`, ни одно не вписано рукой.
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

DELTA = "#b8860b"        # разность кадров
COMPRESS = "#30414f"     # zlib
WRITE = "#1a7f37"        # файл
LIMIT = "#b42318"        # бюджет кадра и отметка оператора
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


def fig_change(out: Path, data: dict[str, Any]) -> Path:
    """Одно утверждение: стоимость кадра задаёт экран.

    Полосы составные — разность, сжатие, файл, — потому что на несжимаемом содержимом
    растут **две** стадии сразу, и одним числом это не видно. Бюджет кадра стоит линией:
    без него «82 мс» не число, а впечатление.
    """
    rows = data["change_scan"]
    budget = float(data["frame_period_ms_at_30fps"])

    fig, (ax,) = _fig(620)
    ys = list(range(len(rows)))
    d = [r["delta_ms"] for r in rows]
    c = [r["compress_ms"] for r in rows]
    w = [r["write_ms"] for r in rows]
    ax.barh(ys, d, 0.55, color=DELTA)
    ax.barh(ys, c, 0.55, left=d, color=COMPRESS)
    ax.barh(ys, w, 0.55, left=[a + b for a, b in zip(d, c)], color=WRITE)
    for y, r in enumerate(rows):
        total = r["delta_ms"] + r["compress_ms"] + r["write_ms"]
        ax.text(total + 1.2, y, f"{total:.0f} мс · {r['kib_per_frame']:.0f} КиБ · "
                                f"{r['max_fps']:.0f} кадр/с",
                va="center", fontsize=LABEL_PT - 3, color=NEUTRAL)

    labels = []
    for r in rows:
        head = r["meaning"].split(":")[0]
        labels.append(f"{head}\nизменилось {r['change_measured']:.0%} пикселей")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=LABEL_PT - 3)

    ax.axvline(budget, color=LIMIT, linewidth=1.4)
    # Подпись бюджета — **над** полосами, в пустом поле над первой строкой. Повёрнутая
    # вдоль линии, она ложилась поверх чисел четырёх строк из пяти: линия проходит через
    # всю картинку, и вдоль неё свободного места нет нигде.
    top = max(a + b + c2 for a, b, c2 in zip(d, c, w))
    ax.set_xlim(0, max(top * 1.45, budget * 1.2))
    ax.set_ylim(len(rows) - 0.45, -1.25)
    ax.text(budget + 1.0, -0.95,
            f"бюджет кадра при 30 кадр/с — {budget:.0f} мс",
            fontsize=LABEL_PT - 3, color=LIMIT, va="center")

    ax.set_xlabel(
        "миллисекунд на кадр 1920×1080, gray8, уровень сжатия 6\n"
        "жёлтый — разность кадров, тёмный — сжатие zlib, зелёный — запись в файл.\n"
        "Нижняя строка — несжимаемый предел, а не экран: у оператора вышло "
        f"{data['operator_kib_per_frame'] / 1024:.2f} МиБ\n"
        "на кадр, то есть его экран сжимался почти как шум", fontsize=LABEL_PT)

    _stamp(fig, n=f"{data['frames_per_setting']} кадров на случай, случаев {len(rows)}",
           unit="кадр",
           compares="стоимость кадра при разной доле изменения экрана")
    return _save(fig, out / "26-stoimost-i-ekran.png",
                 "Стоимость кадра задаёт экран, а не машина.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "record_cost.json"
    if not src.exists():
        print("нет docs/measurements/record_cost.json — сначала "
              "python3 tools/measure_record_cost.py", file=sys.stderr)
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    if "change_scan" not in data:
        print("в замере нет разреза по доле изменения — перезапустите "
              "tools/measure_record_cost.py", file=sys.stderr)
        return 2
    made = [fig_change(base, data)]
    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
