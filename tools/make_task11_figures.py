"""Изображения по TASK-11, части 1. Инвариант 28: одна картинка — одно утверждение.

Две картинки, два утверждения:

1. **Прежние границы стояли не там.** Пол 0.4 попадал внутрь разброса синтетики, потолок
   0.85 — выше любого синтетического домена, а ожидание 0.6 отличалось от опорного 0.667
   меньше, чем домены отличаются друг от друга.
2. **Порча картинки не даёт однонаправленного падения.** На двух доменах из пяти те же
   четыре порчи IoU подняли, и «живое обязано быть хуже» тем самым опровергнуто.

Запуск: `python3 tools/make_task11_figures.py [каталог]`. Числа — из
`docs/measurements/live_penalty.json` и из `harness.corpus.live`, ни одно не вписано рукой.
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
sys.path.insert(0, str(ROOT / "src"))

WIDTH_PX = 1200
DPI = 100
TITLE_PT = 15
LABEL_PT = 14
TICK_PT = 12

DOMAIN = "#30414f"       # синтетический домен
OLD = "#b42318"          # прежняя граница или ожидание
NEW = "#1a7f37"          # выведенная граница или ожидание
NEUTRAL = "#30414f"
GREW = "#b8860b"         # доля выше единицы: IoU вырос


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


def fig_bounds(out: Path) -> Path:
    """Утверждение: прежние границы стояли внутри разброса синтетики и выше любого домена.

    Одна ось — само значение IoU, потому что весь спор был про то, где на этой оси стоят
    четыре числа. Домены точками: без них «0.4» выглядит низким порогом, а не серединой
    измеренного разброса.
    """
    from harness.corpus.live import (LIVE_EXPECTATION, SYNTHETIC_IOU,
                                     synthetic_reference)

    ref = synthetic_reference()
    fig, (ax,) = _fig(560)

    names = sorted(SYNTHETIC_IOU, key=lambda k: SYNTHETIC_IOU[k])
    xs = [SYNTHETIC_IOU[k] for k in names]
    ax.scatter(xs, [1.0] * len(xs), s=110, color=DOMAIN, zorder=3)
    # Подписи ступеньками: `document` 0.667 и `desktop` 0.730 стоят близко, и на одной
    # высоте их имена наезжали друг на друга — читалось «document desktop» одним словом.
    for i, (k, x) in enumerate(zip(names, xs)):
        ax.annotate(f"{k}\n{x:.3f}", (x, 1.0), textcoords="offset points",
                    xytext=(0, 14 + 30 * (i % 2)), ha="center",
                    fontsize=LABEL_PT - 3, color=DOMAIN)

    ax.plot([min(xs), max(xs)], [1.0, 1.0], color=DOMAIN, linewidth=1.2, zorder=2)

    # Прежние числа — вниз, выведенные — вверх. Разделение по стороне, а не по цвету:
    # цвет здесь и так означает «прежнее / выведенное», и два кодирования одного смысла
    # не добавляют сведений, а мешают.
    old_marks = [(0.4, "прежний пол 0.4"), (0.6, "прежнее ожидание 0.6"),
                 (0.85, "прежний потолок 0.85")]
    for x, label in old_marks:
        ax.plot([x, x], [0.60, 0.97], color=OLD, linewidth=1.4)
        ax.annotate(label, (x, 0.58), ha="center", va="top",
                    fontsize=LABEL_PT - 3, color=OLD)

    new_marks = [(ref["value"], f"опорное {ref['value']:.3f}"),
                 (LIVE_EXPECTATION["value"], f"ожидание {LIVE_EXPECTATION['value']:.2f}"),
                 (max(xs), f"потолок {max(xs):.3f}")]
    # Зелёные подписи выше подписей доменов: на прежней высоте «потолок 0.815» задевал
    # приподнятое имя `desktop`. Ступенька между зелёными оставлена — «опорное 0.667» и
    # «потолок 0.815» тоже стоят близко.
    for i, (x, label) in enumerate(new_marks):
        top = 1.62 + 0.12 * (i % 2)
        ax.plot([x, x], [1.03, top], color=NEW, linewidth=1.4)
        ax.annotate(label, (x, top + 0.015), ha="center", va="bottom",
                    fontsize=LABEL_PT - 3, color=NEW)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.30, 1.95)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(
        "IoU разделения себя и мира. Тёмные точки — пять синтетических доменов "
        "(единица усреднения — домен).\n"
        "Красное снизу — прежние границы и ожидание, нарисованные в прозе. Зелёное сверху "
        "— выведенные из этих точек.\n"
        f"Пол 0.4 стоял между `depth` {SYNTHETIC_IOU['depth']:.3f} и "
        f"`video` {SYNTHETIC_IOU['video']:.3f}; потолок 0.85 — выше всех пяти",
        fontsize=LABEL_PT)

    _stamp(fig, n=f"{ref['n']} доменов, {len(SYNTHETIC_IOU) * 6} прогонов",
           unit=ref["unit"],
           compares="где на оси IoU стоят прежние границы и где — измеренные домены")
    return _save(fig, out / "27-granicy-ne-tam.png",
                 "Прежний пол стоял внутри разброса синтетики, а потолок — выше всех доменов.")


def fig_penalty(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: порча не даёт однонаправленного падения.

    Полосы — доля от чистого числа того же домена, а не сам IoU: домены отличаются друг от
    друга сильнее, чем порча меняет каждый, и на абсолютных значениях этого не видно вовсе.
    Линия единицы — то, относительно чего утверждение и формулируется.
    """
    rows = [r for r in data["rows"] if r["penalty"] == "все четыре"]
    rows.sort(key=lambda r: r["iou"] / r["clean_iou"])
    ratios = [r["iou"] / r["clean_iou"] for r in rows]

    fig, (ax,) = _fig(560)
    ys = list(range(len(rows)))
    colors = [GREW if x > 1.0 else DOMAIN for x in ratios]
    ax.barh(ys, ratios, 0.55, color=colors)
    for y, (r, x) in enumerate(zip(rows, ratios)):
        ax.text(x + 0.03, y, f"×{x:.2f}   {r['clean_iou']:.3f} → {r['iou']:.3f}",
                va="center", fontsize=LABEL_PT - 3, color=NEUTRAL)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["domain"] for r in rows], fontsize=LABEL_PT - 1)
    ax.axvline(1.0, color=OLD, linewidth=1.4)
    ax.set_xlim(0, max(ratios) * 1.32)
    # Свободное поле — над верхней полосой; внизу подпись линии ложилась ровно на ось.
    ax.set_ylim(-0.55, len(rows) - 0.05)
    ax.text(1.02, len(rows) - 0.42, "единица: порча ничего не изменила",
            fontsize=LABEL_PT - 3, color=OLD, va="center")

    grew = sum(1 for x in ratios if x > 1.0)
    # Звёздочки разметки в подписи картинки печатались буквально, а третья строка уезжала
    # за правый край: подпись — не markdown и не резиновая.
    ax.set_xlabel(
        "доля IoU, оставшаяся под всеми четырьмя названными порчами сразу\n"
        "(компрессия, полупрозрачная панель, частицы, курсор — все вместе на один кадр).\n"
        f"Жёлтым — домены, где IoU вырос: их {grew} из {len(rows)}.\n"
        "Значит «живое обязано быть хуже» неверно как общее утверждение",
        fontsize=LABEL_PT)

    _stamp(fig, n=f"{data['n']} доменов, шума по {data['noise_seeds']} сидам, "
                 f"{data['frames']} кадров",
           unit=data["unit"],
           compares="испорченный кадр против чистого на том же домене и том же коде")
    return _save(fig, out / "28-porcha-ne-vniz.png",
                 "Порча картинки уронила IoU на трёх доменах из пяти и подняла на двух.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "live_penalty.json"
    if not src.exists():
        print("нет docs/measurements/live_penalty.json — сначала "
              "python3 tools/measure_live_penalty.py "
              "--json docs/measurements/live_penalty.json", file=sys.stderr)
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    made = [fig_bounds(base), fig_penalty(base, data)]
    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
