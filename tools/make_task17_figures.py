"""Изображение по TASK-17. Инвариант 28: одна картинка — одно утверждение.

Утверждение одно: **дорогой была не запись на диск.** Файл забирает 0.1–0.3 мс на кадр
1080p при любой настройке, а кадр съедали две вычислительные стадии в основном потоке:
разность, записанная через int16, и сжатие. Прерывания оператора приходились на
`zlib.compress` не случайно — в основном потоке прерывание падает туда, где код проводит
больше всего времени.

Запуск: `python3 tools/make_task17_figures.py [каталог]`. Числа — из
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
LIMIT = "#b42318"        # период кадра при 30 кадр/с
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


def fig_cost(out: Path, data: dict[str, Any]) -> Path:
    """Одно утверждение: время кадра съедали вычисления, а не диск.

    Полосы составные: разность, сжатие, файл. Так видно не только «сколько всего», но и
    **что именно** — а именно это и было неизвестно, когда узкое место называли «записью
    на диск».
    """
    rows: list[tuple[str, float, float, float]] = []
    forms = data["delta_forms"]
    cur = data["current"]
    # Первая строка — как было: та же настройка, но разность через int16. Число измерено в
    # этом же прогоне на этих же кадрах, а не взято из истории.
    rows.append(("было: уровень 6,\nразность через int16",
                 forms["int16_ms"], cur["compress_ms"], cur["write_ms"]))
    for r in sorted((x for x in data["rows"] if x["keyframe_interval"] == 30),
                    key=lambda x: x["compress_level"]):
        rows.append((f"уровень сжатия {r['compress_level']}",
                     r["delta_ms"], r["compress_ms"], r["write_ms"]))

    fig, (ax,) = _fig(620)
    ys = list(range(len(rows)))
    d = [r[1] for r in rows]
    c = [r[2] for r in rows]
    w = [r[3] for r in rows]
    ax.barh(ys, d, 0.55, color=DELTA)
    ax.barh(ys, c, 0.55, left=d, color=COMPRESS)
    ax.barh(ys, w, 0.55, left=[a + b for a, b in zip(d, c)], color=WRITE)
    for y, (_l, a, b, cc) in enumerate(rows):
        ax.text(a + b + cc + 0.6, y, f"{a + b + cc:.1f} мс", va="center",
                fontsize=LABEL_PT - 2, color=NEUTRAL)

    period = float(data["frame_period_ms_at_30fps"])
    ax.axvline(period, color=LIMIT, linewidth=1.4)
    # Подпись **вдоль** линии и внутри поля: справа от линии она обрезалась краем
    # картинки, то есть на изображении оставалась красная черта без объяснения.
    ax.text(period - 0.7, len(rows) - 0.6,
            f"период кадра при 30 кадр/с — {period:.0f} мс",
            fontsize=LABEL_PT - 3, color=LIMIT, rotation=90,
            ha="right", va="bottom")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=LABEL_PT - 3)
    # Предел оси накрывает период кадра: без этого линия «33 мс» оказывалась за границей
    # и не рисовалась вовсе, оставляя подпись без линии — то есть подпись про то, чего на
    # картинке нет.
    ax.set_xlim(0, max(max(a + b + c2 for _l, a, b, c2 in rows) * 1.22, period * 1.12))
    ax.set_xlabel(
        "миллисекунд на кадр 1920×1080, gray8\n"
        "жёлтый — разность кадров, тёмный — сжатие zlib, зелёный — запись в файл.\n"
        "Захват здесь не мерится: дисплея в контейнере нет, и эта стадия печатается\n"
        "самой harness record на машине оператора", fontsize=LABEL_PT)
    ax.invert_yaxis()

    _stamp(fig, n=f"{data['frames_per_setting']} кадров на настройку",
           unit="кадр",
           compares="стадии одного кадра при разных настройках профиля")
    return _save(fig, out / "25-stoimost-kadra.png",
                 "Дорогой была не запись на диск, а вычисления перед ней.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "record_cost.json"
    if not src.exists():
        print("нет docs/measurements/record_cost.json — сначала "
              "python3 tools/measure_record_cost.py", file=sys.stderr)
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    made = [fig_cost(base, data)]
    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
