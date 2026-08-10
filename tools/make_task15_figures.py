"""Изображение по TASK-15. Инвариант 28: одна картинка — одно утверждение.

Утверждение одно: **ложно-зелёный бывает двух видов, и один из них замер не видит.**
Тест, чей исход зависит от машины, находится прогоном в другом окружении. Тест, чьё
утверждение стоит под условием, недостижимым в этом окружении, зелен во **всех**
окружениях сразу — исход не меняется, потому что проверки не было ни в одном, — и его
находит только статический разбор. Два инструмента, и слепые пятна не перекрываются.

Запуск: `python3 tools/make_task15_figures.py [каталог]`. Числа — из
`docs/measurements/env_dependence.json`, ни одно не вписано рукой.
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

#: Цвет здесь — **инструмент**, которым дефект найден, и в подписи оси это сказано.
BY_RUN = "#b8860b"        # прогон набора в другом объявленном окружении
BY_STATIC = "#30414f"     # статический разбор тестов
FIXED = "#1a7f37"         # столько осталось после правок
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


def fig_false_green(out: Path, data: dict[str, Any]) -> Path:
    """Одно утверждение: два вида ложно-зелёного, и каждый находит свой инструмент.

    Одна панель. Числа берутся из замера: сколько дефектов дал прогон в подменённых
    окружениях до правок, сколько осталось после, и сколько пустых тестов нашёл
    статический разбор — тех, которые не меняли исхода ни в одном из пяти окружений и
    потому для замера были невидимы.
    """
    before = data["before"]["в подменённых окружениях"]
    blind = data["blind_spot"]
    total = int(data["coverage"]["tests_run"])
    envs = int(data["coverage"]["environments"])
    after = len(data["flipped"])

    rows = [
        ("Исход зависел от машины:\nтест проверял окружение, а не поведение",
         int(before["упало"]) - 1, BY_RUN),
        ("Отказ с трассировкой вместо внятного:\nдефект кода, не теста",
         1, BY_RUN),
        ("Утверждение под недостижимым условием:\nзелено во всех пяти окружениях",
         int(blind["сколько"]), BY_STATIC),
        (f"Осталось после правок:\nменяют исход в {envs} окружениях",
         after, FIXED),
    ]

    fig, (ax,) = _fig(560)
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], 0.52, color=[r[2] for r in rows])
    # Число подписывается цветом своей строки: у нижней строки столбец нулевой, и без
    # этого зелёный, названный в подписи оси, на картинке не появлялся бы вовсе.
    for y, (_label, value, colour) in enumerate(rows):
        ax.text(value + 0.09, y, str(value), va="center", fontsize=LABEL_PT,
                color=colour)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=LABEL_PT - 3)
    ax.set_xlim(0, max(r[1] for r in rows) + 1.0)
    ax.set_xticks(range(0, max(r[1] for r in rows) + 2))
    # Подписи короткие по строкам: длинная строка при ширине 1200 px обрезается справа, и
    # обрезается именно та часть, где сказано, что каким цветом обозначено.
    ax.set_xlabel(
        f"тестов из {total}, зелёных без проверки\n"
        "цвет — чем найдено: жёлтый — прогоном в другом окружении,\n"
        "тёмный — статическим разбором, зелёный — осталось после правок.\n"
        "Верхние две строки замер видит, третью не видит устройством",
        fontsize=LABEL_PT)
    ax.invert_yaxis()

    _stamp(fig, n=f"{total} тестов × {envs} окружений",
           unit="тест",
           compares="исход одного теста между окружениями; "
                    "для третьей строки — достижимость")
    return _save(fig, out / "23-lozhno-zelenye.png",
                 "Ложно-зелёный бывает двух видов, и один из них замер не видит.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "env_dependence.json"
    if not src.exists():
        print("нет docs/measurements/env_dependence.json — сначала "
              "python3 tools/measure_env_dependence.py", file=sys.stderr)
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    made = [fig_false_green(base, data)]
    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
