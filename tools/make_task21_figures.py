"""Изображения по TASK-21. Инвариант 28: одна картинка — одно утверждение.

Два утверждения, две картинки:

1. **Уменьшить область дешевле, чем понизить сжатие, и дешевле по обеим величинам сразу.**
   Уровень платит диском, размер — нет.
2. **Стоимость кадра падает ровно по пикселям.** Ни быстрее, ни медленнее, и это надо было
   проверить, а не предположить.

Запуск: `python3 tools/make_task21_figures.py [каталог]`. Числа — из
`docs/measurements/capture_area.json`, ни одно не вписано рукой.
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

TIME = "#30414f"         # время на кадр
DISK = "#b8860b"         # байты на кадр
LIMIT = "#b42318"        # опора сравнения и бюджет
GOOD = "#1a7f37"         # рычаг, выигрывающий по обеим величинам
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


def fig_levers(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: область выигрывает по обеим величинам, уровень платит диском.

    Две шкалы на одной картинке — время и байты, — потому что утверждение именно про две
    величины сразу. Одна шкала показала бы «оба рычага работают» и скрыла бы плату.
    """
    rows = data["levers"]
    base = rows[0]
    fig, (ax,) = _fig(620)

    ys = list(range(len(rows)))
    time_share = [r["total_ms"] / base["total_ms"] for r in rows]
    disk_share = [r["kib_per_frame"] / base["kib_per_frame"] for r in rows]
    # Шкала логарифмическая, потому что величины различаются в сто раз: 0.25 у области и
    # 35 у «без сжатия». На линейной шкале байты «без сжатия» растягивали ось так, что
    # всё остальное сливалось в одну черту — то есть картинка показывала одну строку из
    # пяти при утверждении про все пять.
    #
    # И потому же здесь **точки с поводком от опоры, а не полосы**: у логарифмической оси
    # нет нуля, и длина полосы от левого края не значит ничего. Длина поводка от единицы
    # значит ровно то, что утверждается: насколько рычаг отошёл от опоры.
    ax.set_xscale("log")
    h = 0.20
    for y, (ts, ds) in enumerate(zip(time_share, disk_share)):
        ax.plot([1.0, ts], [y + h, y + h], color=TIME, linewidth=1.6, zorder=2)
        ax.plot([1.0, ds], [y - h, y - h], color=DISK, linewidth=1.6, zorder=2)
    ax.scatter(time_share, [y + h for y in ys], s=90, color=TIME, zorder=3,
               label="время на кадр")
    ax.scatter(disk_share, [y - h for y in ys], s=90, color=DISK, zorder=3,
               label="байты на кадр")
    for y, (r, ts, ds) in enumerate(zip(rows, time_share, disk_share)):
        # Подпись правее и опоры, и обеих точек: иначе она перечёркивала красную черту,
        # то есть ложилась поверх того самого, относительно чего всё и сравнивается.
        ax.text(max(1.0, ts, ds) * 1.25, y,
                f"{1 / ts:.1f}× быстрее · {r['kib_per_frame']:.0f} КиБ",
                va="center", fontsize=LABEL_PT - 3, color=NEUTRAL)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=LABEL_PT - 2)
    ax.axvline(1.0, color=LIMIT, linewidth=1.4)
    top = max(max(time_share), max(disk_share))
    ax.set_xlim(0.05, top * 6.0)
    ax.set_ylim(-0.75, len(rows) - 0.2)
    ax.text(1.12, len(rows) - 0.45, "опора: полный кадр, уровень 6",
            fontsize=LABEL_PT - 3, color=LIMIT, va="center")
    ax.invert_yaxis()
    ax.legend(fontsize=LABEL_PT - 3, loc="lower right", frameon=False)

    area = next(r for r in rows if r["lever"] == "область")
    weak = next(r for r in rows if r["lever"] == "сжатие" and r["label"].endswith("1"))
    ax.set_xlabel(
        "доля от опоры, шкала логарифмическая: левее красной черты — дешевле\n"
        f"область (окно 960×540): {base['total_ms'] / area['total_ms']:.1f}× по времени "
        f"и {base['kib_per_frame'] / area['kib_per_frame']:.1f}× по диску — оба меньше\n"
        f"уровень (1080p, 6→1): {base['total_ms'] / weak['total_ms']:.1f}× по времени, "
        f"но диск {weak['kib_per_frame'] / base['kib_per_frame']:.1f}× — это плата\n"
        "«без сжатия» быстро только здесь: 2 МиБ на кадр пишутся в файловую систему "
        "контейнера",
        fontsize=LABEL_PT)

    _stamp(fig, n=f"{data['frames_per_size']} кадров на случай, случаев {len(rows)}",
           unit="кадр",
           compares="время и байты на кадр при уменьшении области против понижения сжатия")
    return _save(fig, out / "29-dva-rychaga.png",
                 "Уменьшение области дешевле по времени и по диску; сжатие платит диском.")


def fig_pixels(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: стоимость падает ровно по пикселям.

    Обе оси — доли от полного кадра, и диагональ нарисована: без неё «падает по пикселям»
    нечем проверить глазом, а именно это и утверждается.
    """
    rows = data["sizes"]
    fig, (ax,) = _fig(600)

    xs = [r["pixel_share"] for r in rows]
    ys = [r["cost_share"] for r in rows]
    ax.plot([0, 1], [0, 1], color=LIMIT, linewidth=1.2)
    ax.scatter(xs, ys, s=110, color=TIME, zorder=3)
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(f"{r['width']}×{r['height']}\n{r['cheaper_times']:.1f}× дешевле",
                    (x, y), textcoords="offset points", xytext=(12, -6),
                    fontsize=LABEL_PT - 3, color=NEUTRAL)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel(
        "доля пикселей от кадра 1920×1080\n"
        "красная линия — «стоимость падает ровно по пикселям». Точки лежат на ней "
        "и чуть ниже:\n"
        f"при {rows[-1]['pixel_share']:.1%} пикселей стоимость "
        f"{rows[-1]['cost_share']:.1%} — то есть немного дешевле, чем линейно",
        fontsize=LABEL_PT)
    ax.set_ylabel("доля стоимости кадра", fontsize=LABEL_PT)

    _stamp(fig, n=f"{data['frames_per_size']} кадров на размер, размеров {len(rows)}",
           unit="кадр",
           compares="доля пикселей против доли стоимости, доля изменения экрана полная")
    return _save(fig, out / "30-stoimost-po-pikselyam.png",
                 "Стоимость кадра падает по пикселям: вчетверо меньше площади — "
                 "вчетверо дешевле.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "capture_area.json"
    if not src.exists():
        print("нет docs/measurements/capture_area.json — сначала "
              "harness cost --sizes --levels --frames 30 --json > "
              "docs/measurements/capture_area.json", file=sys.stderr)
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    if not data.get("levers"):
        print("в замере нет разреза по рычагам — перезапустите с --levels",
              file=sys.stderr)
        return 2
    made = [fig_levers(base, data), fig_pixels(base, data)]
    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
