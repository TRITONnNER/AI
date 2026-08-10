"""Изображение по TASK-22 и началу М5. Инвариант 28: одна картинка — одно утверждение.

Утверждение одно, и оно про прибор, а не про агента: **метрика конфабуляции в нынешней
расстановке равна доле действий рефлекса.** Предсказание инварианта 13 подтвердилось — доля
следует за отношением частот, — но подтвердилось вырожденной формой: две величины совпадают
до десятых процента, значит вторая не несёт своей информации.

Запуск: `python3 tools/make_task22_figures.py [каталог]`. Числа — из
`docs/measurements/live_cycle.json`.
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

CONF = "#b42318"         # доля расхождений
REFLEX = "#30414f"       # доля действий рефлекса
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


def fig_degenerate(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: две величины совпадают, значит одна из них лишняя.

    Обе величины на одной оси и в одних единицах — иначе «совпадают» проверить нечем. Точки
    парами: если бы метрика несла своё, пары расходились бы хоть где-то.
    """
    keys = list(data["by_ratio"])
    rows = [data["by_ratio"][k] for k in keys]
    fig, (ax,) = _fig(600)

    ys = list(range(len(rows)))
    conf = [r["confab_median"] for r in rows]
    reflex = [r["reflex_share_median"] for r in rows]
    for y, (c, r) in enumerate(zip(conf, reflex)):
        ax.plot([min(c, r), max(c, r)], [y, y], color=NEUTRAL, linewidth=1.0, zorder=2)
    ax.scatter(reflex, ys, s=130, color=REFLEX, zorder=3,
               label="доля действий, начатых рефлексом")
    ax.scatter(conf, ys, s=60, color=CONF, zorder=4,
               label="доля объяснений «не тем слоем»")
    for y, (c, r) in enumerate(zip(conf, reflex)):
        # Подпись уходит влево, когда точка правее середины: справа места нет, и первая
        # редакция обрезала «0.02 п.п.» краем картинки.
        right = max(c, r) < 0.55
        ax.annotate(f"{c:.1%} против {r:.1%} — разница {abs(c - r) * 100:.2f} п.п.",
                    (max(c, r) if right else min(c, r), y),
                    textcoords="offset points", xytext=(14 if right else -14, -4),
                    ha="left" if right else "right",
                    fontsize=LABEL_PT - 3, color=NEUTRAL)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"рефлекс/планировщик {k}\n"
                        f"{r['hz_reflex']:g} и {r['hz_planner']:g} Гц"
                        for k, r in zip(keys, rows)], fontsize=LABEL_PT - 3)
    ax.set_xlim(0, 1.0)
    # Пустая строка сверху — место под легенду. Без неё легенда ложится либо на подпись
    # нижней строки (снизу), либо на подпись верхней (сверху).
    ax.set_ylim(-1.5, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.legend(fontsize=LABEL_PT - 3, loc="upper center", frameon=False, ncol=2)
    ax.set_xlabel(
        "доля от доставленных действий\n"
        f"наибольшее расхождение двух величин по всем прогонам — "
        f"{data['max_gap_to_reflex_share'] * 100:.2f} п.п.\n"
        "предсказание инварианта 13 подтвердилось: доля следует за отношением частот.\n"
        "Но подтвердилось вырожденной формой: метрика равна уже известной величине",
        fontsize=LABEL_PT)

    _stamp(fig, n=f"{len(data['rows'])} прогонов ({len(keys)} отношения × "
                 f"{len(data['seeds'])} сида)",
           unit=data["unit"],
           compares="доля объяснений «не тем слоем» против доли действий рефлекса")
    return _save(fig, out / "31-konfabulyaciya-vyrozhdena.png",
                 "Метрика конфабуляции равна доле действий рефлекса: своей информации нет.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "live_cycle.json"
    if not src.exists():
        print("нет docs/measurements/live_cycle.json — сначала "
              "python3 tools/measure_live_cycle.py", file=sys.stderr)
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    made = [fig_degenerate(base, data)]
    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
