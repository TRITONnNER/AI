"""Изображения по TASK-14. Инвариант 28: одна картинка — одно утверждение.

Одно утверждение, одна картинка: **направление ошибки тождества задаёт функция
отпечатка, а не механика.** Тот же мир под огрублением слипается, под точным хешем
дробится, и пересмотр помогает ровно там, где источник умеет ответить «это близко».

Запуск: `python3 tools/make_task14_figures.py [каталог]`. Числа — из
`docs/measurements/identity.json`.
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

TRUTH = "#30414f"       # истина мира
BEFORE = "#b8860b"      # до пересмотра
AFTER = "#1a7f37"       # после пересмотра
BAD = "#b42318"         # ошибка выросла
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


def fig_identity(out: Path, data: dict[str, Any]) -> Path:
    """Одно утверждение: направление ошибки задаёт отпечаток, а не механика.

    Одна панель, а не две. Первая редакция рисовала рядом «карточек до и после» и
    «ошибку до и после» — то есть одно и то же дважды, потому что ошибка и есть
    расстояние от числа карточек до истины. Подписи при этом наползали друг на друга, и
    читать было нечего. Число карточек ушло в подпись строки, где оно и нужно: как
    пояснение к величине ошибки, а не как отдельное утверждение.
    """
    cases = data["cases"]
    truth = int(cases[0]["true_entities"])
    fig, (ax,) = _fig(560)

    rows = sorted(cases, key=lambda c: (c["source"], c["case"]))
    labels = [f"{c['source']} · {c['case']}\n"
              f"карточек {c['before']} → {c['after']}" for c in rows]
    before = [c["error_before"] for c in rows]
    after = [c["error_after"] for c in rows]
    ys = range(len(rows))

    # Смещения со знаком минус у «до» — потому что ось перевёрнута (`invert_yaxis`), и
    # `y + 0.19` рисуется **ниже**, а не выше. Первая редакция ставила «до» на `y + 0.19`
    # при подписи «верхняя полоса — до пересмотра», то есть подпись говорила ровно
    # обратное тому, что на картинке: читатель видел «ошибка 1 → 6» там, где было 6 → 1.
    ax.barh([y - 0.19 for y in ys], before, 0.34, color=BEFORE)
    ax.barh([y + 0.19 for y in ys], after, 0.34,
            color=[AFTER if a <= b else BAD for a, b in zip(after, before)])
    for y, (b, a) in enumerate(zip(before, after)):
        ax.text(b + 0.12, y - 0.19, str(b), va="center", fontsize=LABEL_PT - 2,
                color=BEFORE)
        ax.text(a + 0.12, y + 0.19, str(a), va="center", fontsize=LABEL_PT - 2,
                color=AFTER if a <= b else BAD)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=LABEL_PT - 3)
    ax.set_xlim(0, max(before + after) + 1.2)
    ax.set_xlabel(f"ошибка тождества: |карточек − истина|, истина {truth} сущностей\n"
                  "верхняя полоса — до пересмотра, нижняя — после\n"
                  "красная означала бы, что пересмотр сделал хуже", fontsize=LABEL_PT)
    ax.invert_yaxis()

    # Подпись короче на две трети строки: прежняя обрезалась справа на слове «источник
    # отпечатка», то есть единица независимости на картинке была не дочитываема, хотя
    # инвариант 28 требует её присутствия.
    _stamp(fig, n=f"{len(cases)} прогонов, 2 источника отпечатков",
           unit="карточка; о методе — источник отпечатка",
           compares="ошибка тождества до и после пересмотра, по источникам")
    return _save(fig, out / "22-tozhdestvo.png",
                 "Направление ошибки тождества задаёт отпечаток, а не механика.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "identity.json"
    if not src.exists():
        print("нет docs/measurements/identity.json — сначала "
              "python3 tools/measure_identity.py", file=sys.stderr)
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    made = [fig_identity(base, data)]
    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
