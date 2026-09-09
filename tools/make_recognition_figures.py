"""Изображения по узнаванию места и разделению отпечатка. SPEC-FULL, A2. Инвариант 28.

45. **Ни одно из трёх средств не уменьшило граф.** Гипотеза нового места не сработала
    ни разу, адаптивный порог добавил 3 %, сопоставление отрезков увеличило граф в
    1.6 раза. Единственный выигрыш — ложные слияния вчетверо реже, то есть опасная
    ошибка падает ценой безопасной.
46. **Разделение отпечатка убрало задвоение карточек.** 48 карточек против 12 при
    истине 12: перекраска стала событием, а не новым предметом.

Заголовки проверяются по данным перед рисованием: перестанет число подтверждать —
инструмент упадёт, а не нарисует подпись, которой данные больше не отвечают.

Запуск: `python3 tools/make_recognition_figures.py [каталог]`. Числа — из
`docs/measurements/recognition.json`.
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
WORSE = "#b42318"        # хуже контроля
SAME = "#8a97a3"         # не отличается от контроля
BETTER = "#1f6f43"       # лучше контроля
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


def fig_means(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: ни одно средство не уменьшило граф, а отрезок увеличил в 1.6 раза."""
    combos = data["by_combination"]
    control = combos["контроль"]["places_median"]
    order = sorted(combos, key=lambda k: combos[k]["places_median"])
    best = min(combos.values(), key=lambda v: v["places_median"])["places_median"]
    worst_label = max(combos, key=lambda k: combos[k]["places_median"])
    worst = combos[worst_label]["places_median"] / control

    if best < control:
        raise ValueError(f"какое-то сочетание уменьшило граф ({best} < {control}) — "
                         "заголовок неверен, средство надо включать по умолчанию")
    seq = combos["отрезок"]["places_median"] / control
    if not 1.5 <= seq <= 1.8:
        raise ValueError(f"отрезок даёт {seq:.2f}×, а в заголовке «в 1.6 раза»")

    fig, (ax1, ax2) = _fig(680, ncols=2)

    xs = list(range(len(order)))
    values = [combos[k]["places_median"] for k in order]
    colors = [SAME if v == control else (WORSE if v > control else BETTER)
              for v in values]
    ax1.barh(xs, values, color=colors, height=0.62)
    ax1.axvline(control, color=NEUTRAL, ls="--", lw=1.4)
    for x, v in zip(xs, values):
        ax1.annotate(f"{v:.0f}  ({v / control:.2f}×)", (v, x),
                     textcoords="offset points", xytext=(6, -4),
                     fontsize=LABEL_PT - 4, color=NEUTRAL)
    ax1.set_yticks(xs)
    ax1.set_yticklabels(order, fontsize=LABEL_PT - 4)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(values) * 1.28)
    ax1.set_xlabel("узлов графа мест, медиана по трём сидам\n"
                   "пунктир — контроль (все средства выключены)",
                   fontsize=LABEL_PT - 3, loc="left")

    # Панель 2: обе ошибки. Единственный выигрыш виден только здесь.
    split = [combos[k]["false_split_median"] * 100 for k in order]
    merge = [combos[k]["false_merge_median"] * 100 for k in order]
    ax2.barh([x - 0.19 for x in xs], split, height=0.34, color=SAME,
             label="ложное расщепление (безопасная)")
    ax2.barh([x + 0.19 for x in xs], merge, height=0.34, color=WORSE,
             label="ложное слияние (опасная)")
    for x, m in zip(xs, merge):
        ax2.annotate(f"{m:.2f} %", (m, x + 0.19), textcoords="offset points",
                     xytext=(6, -4), fontsize=LABEL_PT - 5, color=NEUTRAL)
    ax2.set_yticks(xs)
    ax2.set_yticklabels([""] * len(xs))
    ax2.invert_yaxis()
    ax2.set_xscale("symlog", linthresh=0.1)
    ax2.set_xlim(0, 300)
    ax2.legend(fontsize=LABEL_PT - 5, frameon=False, loc="upper right",
               bbox_to_anchor=(1.0, 1.12), ncols=2)
    ax2.set_xlabel("доля пар с ошибкой, %; шкала логарифмическая\n"
                   "отрезок пути снижает опасную ошибку вчетверо",
                   fontsize=LABEL_PT - 3, loc="left")

    _stamp(fig, n="24 прогона (8 сочетаний × 3 сида), 1500 наблюдений в каждом",
           unit="прогон",
           compares="каждое сочетание против контроля на тех же сидах")
    return _save(fig, out / "45-sredstva-ne-umenshili-graf.png",
                 f"Ни одно из трёх средств не уменьшило граф мест: отрезок пути "
                 f"увеличил его в {seq:.1f} раза.")


def fig_cards(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: разделение отпечатка убрало задвоение карточек."""
    c = data["cards"]
    if c["split_cards"] != c["true_things"]:
        raise ValueError(f"разделение дало {c['split_cards']} карточек при истине "
                         f"{c['true_things']} — заголовок неверен")
    if c["plain_cards"] <= c["true_things"]:
        raise ValueError("контроль больше не задваивает — сравнивать не с чем")

    fig, (ax,) = _fig(520)
    labels = ["прежний отпечаток\n(контроль)", "разделённый отпечаток\n(A2.4)", "истина"]
    values = [c["plain_cards"], c["split_cards"], c["true_things"]]
    colors = [WORSE, BETTER, NEUTRAL]
    xs = list(range(3))
    ax.bar(xs, values, color=colors, width=0.55)
    for x, v in zip(xs, values):
        ax.annotate(f"{v}", (x, v), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=LABEL_PT - 2, color=NEUTRAL)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=LABEL_PT - 3)
    ax.set_ylabel("карточек сущностей", fontsize=LABEL_PT - 2)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_xlabel(
        f"мир объявлен замером: {c['true_things']} предметов, каждый перекрашен "
        f"{c['repaints']} раза\n"
        f"перекраска стала {c['property_events']} событиями смены свойства вместо "
        f"{c['plain_cards'] - c['true_things']} лишних карточек",
        fontsize=LABEL_PT - 3, loc="left")

    _stamp(fig, n="1 прогон, истина по построению", unit="прогон",
           compares="разделённый отпечаток против прежнего на одном и том же мире")
    return _save(fig, out / "46-razdelenie-otpechatka.png",
                 f"Разделение отпечатка убрало задвоение карточек: "
                 f"{c['plain_cards']} против {c['split_cards']} при истине "
                 f"{c['true_things']}.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "recognition.json"
    if not src.exists():
        print(f"нет {src.name} — сначала python3 tools/measure_recognition.py",
              file=sys.stderr)
        return 1
    data = json.loads(src.read_text(encoding="utf-8"))
    made = [fig_means(base, data), fig_cards(base, data)]

    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
