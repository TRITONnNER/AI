"""Изображения по TASK-12. Инвариант 28: одна картинка — одно утверждение.

Два утверждения, две картинки:

- **20** — два различения, которые размывались сами: подтверждённый пересказ становился
  опытом (40 из 40), а вопросы считались очередью дел (17 вместо 5);
- **21** — переоткрытие отложенного работает и промахивается: 6 вопросов из 13 вернулись
  и разрешились, треть предложений оказалась ложной, покрытие 69 % (инвариант 31).

Запуск: `python3 tools/make_task12_figures.py [каталог]`. Числа — из
`docs/measurements/beliefs.json`, то есть из прогона, а не из текста.
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

# Цвет только как значение, и роль названа в подписи там, где встречается.
WRONG = "#b42318"       # как было: различение потеряно
RIGHT = "#1a7f37"       # как стало: различение сохранено
BLIND = "#8a8a8a"       # не объявил нехватку либо ложное предложение
NEUTRAL = "#30414f"


def _fig(height_px: int, ncols: int = 1, nrows: int = 1):
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(WIDTH_PX / DPI, height_px / DPI), dpi=DPI)
    axes = [axes] if ncols * nrows == 1 else (
        list(axes.ravel()) if hasattr(axes, "ravel") else list(axes))
    if len(axes) > 4:
        raise ValueError(f"панелей {len(axes)}: больше четырёх — значит утверждений "
                         "несколько, значит картинок несколько (инвариант 28)")
    for ax in axes:
        ax.tick_params(labelsize=TICK_PT)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig, axes


def _stamp(fig, *, n: str, unit: str, compares: str = "") -> None:
    if not n or not unit:
        raise ValueError("на изображении обязательны n и единица независимости "
                         "(инвариант 28)")
    fig.text(0.01, 0.048, f"n = {n}   ·   единица независимости: {unit}",
             fontsize=LABEL_PT - 1, color=NEUTRAL)
    if compares:
        fig.text(0.01, 0.012, f"сравнивается: {compares}",
                 fontsize=LABEL_PT - 2, color=NEUTRAL)


def _save(fig, path: Path, title: str) -> Path:
    if not title.endswith((".", "!", "?")) and ":" not in title:
        raise ValueError(f"заголовок «{title}» — тема, а не предложение (инвариант 28)")
    if len(title) > 96:
        raise ValueError(
            f"заголовок длиной {len(title)} символов обрежется по краю при 1200 px")
    fig.suptitle(title, fontsize=TITLE_PT, x=0.01, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0.115, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"нарисовано: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 20. Два различения, которые размывались сами
# ---------------------------------------------------------------------------


def fig_distinctions(out: Path, data: dict[str, Any]) -> Path:
    """Одно утверждение: оба различения терялись без всякой попытки их убрать."""
    r = data["свидетельство_не_опыт"]
    q = data["очередь_не_перечень"]
    fig, (ax1, ax2) = _fig(580, ncols=2)

    # Панель 1: сколько пересказов меняло происхождение.
    old = int(r["сменили_происхождение_по_прежнему_правилу"])
    new = int(r["сменили_по_новому"])
    total = int(r["пересказов"])
    ax1.bar([0, 1], [old, new], color=[WRONG, RIGHT], width=0.55)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["прежнее правило", "теперь"], fontsize=LABEL_PT)
    ax1.set_ylabel("пересказов, ставших «опытом»", fontsize=LABEL_PT)
    ax1.set_ylim(0, total * 1.15)
    for x, v in ((0, old), (1, new)):
        ax1.text(x, v + total * 0.02, str(v), ha="center", fontsize=LABEL_PT,
                 color=NEUTRAL)
    # Опорной линии «все 40» здесь нет намеренно: столбик прежнего правила ей равен,
    # и подпись легла бы поверх его значения, ничего не добавив.
    ax1.set_xlabel("красный — различение потеряно\nзелёный — сохранено",
                   fontsize=LABEL_PT)

    # Панель 2: очередь дел против перечня непонятого.
    old_q = int(q["ожидает_проверки_по_прежнему_условию"])
    now_q = int(q["ожидает_проверки_теперь"])
    deferred = int(q["отложено_теперь"])
    ax2.bar([0], [old_q], color=WRONG, width=0.55)
    ax2.bar([1], [now_q], color=RIGHT, width=0.55)
    ax2.bar([1], [deferred], bottom=[now_q], color=BLIND, width=0.55)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["«ожидает проверки»\nпрежним условием",
                         "очередь дел + отложенное"], fontsize=LABEL_PT - 1)
    ax2.set_ylabel("гипотез", fontsize=LABEL_PT)
    ax2.set_ylim(0, old_q * 1.2)
    ax2.set_yticks(range(0, old_q + 3, 3))     # гипотезы целые: дроби на оси лгут
    ax2.text(0, old_q + old_q * 0.03, str(old_q), ha="center", fontsize=LABEL_PT,
             color=NEUTRAL)
    ax2.text(1, now_q / 2, f"{now_q} в работе", ha="center", va="center",
             fontsize=LABEL_PT - 1, color="white")
    ax2.text(1, now_q + deferred / 2, f"{deferred} отложено", ha="center", va="center",
             fontsize=LABEL_PT - 1, color="white")
    ax2.set_xlabel("серый — перечень непонятого, который считался очередью",
                   fontsize=LABEL_PT)

    _stamp(fig, n=f"{total} пересказов и {old_q} гипотез",
           unit="утверждение слева, вопрос справа",
           compares="прежнее правило против нового на одном и том же наборе")
    return _save(fig, out / "20-dva-razlicheniya.png",
                 "Подтверждённый пересказ становился опытом в 40 случаях из 40.")


# ---------------------------------------------------------------------------
# 21. Переоткрытие: работает и промахивается
# ---------------------------------------------------------------------------


def fig_reopening(out: Path, data: dict[str, Any]) -> Path:
    """Одно утверждение: механизм возвращает загадки, но видит не все и ошибается."""
    p = data["переоткрытие"]
    asked = int(p["asked"])
    resolved = int(p["resolved"])
    blind = int(p["blind"])
    refused = int(p["offers_refused"])
    open_declared = asked - resolved - blind
    fig, (ax1, ax2) = _fig(560, ncols=2)

    # Панель 1: что стало с каждым вопросом.
    labels = ["разрешено\nзадним числом", "открыто,\nнехватка объявлена",
              "слепые: нехватка\nне объявлена"]
    vals = [resolved, open_declared, blind]
    ax1.bar(range(3), vals, color=[RIGHT, NEUTRAL, BLIND], width=0.6)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels, fontsize=LABEL_PT - 2)
    ax1.set_ylabel("вопросов", fontsize=LABEL_PT)
    ax1.set_ylim(0, max(vals) * 1.25)
    ax1.set_yticks(range(0, max(vals) + 2))    # вопросы целые
    for i, v in enumerate(vals):
        ax1.text(i, v + max(vals) * 0.04, str(v), ha="center", fontsize=LABEL_PT,
                 color=NEUTRAL)
    ax1.set_xlabel(f"всего задано {asked}\nсерые не вернутся никогда, и это видно "
                   "числом", fontsize=LABEL_PT)

    # Панель 2: два обязательных числа проверки — доля ложных и покрытие.
    cov = float(p["coverage"])
    fo = float(p["false_offer_share"])
    ax2.barh([1, 0], [cov, fo], color=[RIGHT, BLIND], height=0.45)
    ax2.set_yticks([1, 0])
    ax2.set_yticklabels(["покрытие", "ложные\nпредложения"], fontsize=LABEL_PT)
    ax2.set_xlim(0, 1.05)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(["0", "25 %", "50 %", "75 %", "100 %"], fontsize=TICK_PT)
    ax2.text(cov + 0.02, 1, f"{cov:.0%}", va="center", fontsize=LABEL_PT, color=NEUTRAL)
    ax2.text(fo + 0.02, 0, f"{fo:.0%} ({refused} из {int(p['offers'])})", va="center",
             fontsize=LABEL_PT, color=NEUTRAL)
    ax2.set_xlabel("зелёный — вопросы, объявившие нехватку\nсерый — предложения без "
                   "построенного теста", fontsize=LABEL_PT)

    _stamp(fig, n=f"{asked} вопросов, {int(p['offers'])} предложений",
           unit="вопрос",
           compares="исход каждого вопроса; доля ложных и покрытие по инварианту 31")
    return _save(fig, out / "21-pereotkrytie.png",
                 f"Переоткрытие вернуло {resolved} вопросов из {asked}, "
                 f"{fo:.0%} предложений ложны.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "beliefs.json"
    if not src.exists():
        print("нет docs/measurements/beliefs.json — сначала "
              "python3 tools/measure_beliefs.py", file=sys.stderr)
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))

    made = [fig_distinctions(base, data), fig_reopening(base, data)]

    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
