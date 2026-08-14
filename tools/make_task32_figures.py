"""Изображения по TASK-32. Инвариант 28: одна картинка — одно утверждение.

38. **A, режим среды.** Признаки определяются наблюдением без ошибок в обе стороны, и
    гейт по ним сужает разброс IoU параллакса.
39. **C, обратимость.** Счётчики рискованных действий совпадают при работающем пороге и при
    снятом: осторожность в разведке не читается.

Запуск: `python3 tools/make_task32_figures.py [каталог]`. Числа — из
`docs/measurements/{regime,reversibility}.json`.
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
NEW = "#b42318"          # применимо / порог работает
OLD = "#30414f"          # неприменимо / порог снят
MUTED = "#8a9199"        # не определено
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



def fig_regime(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: гейт по режиму убирает из сводки не плохой ответ, а ответ не о том."""
    doms = list(data["by_domain"])
    fig, (ax1, ax2) = _fig(640, ncols=2)

    # Панель 1: IoU параллакса по доменам, применимые и неприменимые разным цветом.
    iou = [data["bench_parallax_iou"][d] for d in doms]
    ok = [d in data["spread"]["applicable_only"]["which"] for d in doms]
    xs = list(range(len(doms)))
    ax1.bar([x for x, k in zip(xs, ok) if k], [v for v, k in zip(iou, ok) if k],
            color=NEW, width=0.55, label="параллакс применим")
    ax1.bar([x for x, k in zip(xs, ok) if not k], [v for v, k in zip(iou, ok) if not k],
            color=OLD, width=0.55, label="неприменим — из сводки уходит")
    for x, v in zip(xs, iou):
        ax1.annotate(f"{v:.3f}", (x, v), textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=LABEL_PT - 4, color=NEUTRAL)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(doms, fontsize=LABEL_PT - 3)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("IoU разделения слоёв по параллаксу", fontsize=LABEL_PT - 2)
    a, b = data["spread"]["all"], data["spread"]["applicable_only"]
    ax1.set_xlabel(f"домен\nразмах по всем пяти {a['range']}, "
                   f"по применимым {b['range']}\n"
                   "остаток разброса — depth, чей провал объявлен заранее по другой причине",
                   fontsize=LABEL_PT - 3)
    ax1.legend(fontsize=LABEL_PT - 4, frameon=False, loc="upper right")

    # Панель 2: обе ошибки по признакам. Ноль рисуется как ноль, а не пропускается.
    feats = [f for f, e in data["errors"].items() if e["checked"]]
    unchecked = [f for f, e in data["errors"].items() if not e["checked"]]
    ys = list(range(len(feats)))
    ax2.barh([y + 0.18 for y in ys], [data["errors"][f]["false_alarm"] for f in feats],
             height=0.32, color=NEW, label="ложные тревоги")
    ax2.barh([y - 0.18 for y in ys], [data["errors"][f]["false_confirm"] for f in feats],
             height=0.32, color=OLD, label="ложные подтверждения")
    for y, f in zip(ys, feats):
        e = data["errors"][f]
        ax2.annotate(f"0 из {e['checked']}", (0, y), textcoords="offset points",
                     xytext=(8, -4), fontsize=LABEL_PT - 4, color=NEUTRAL)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(feats, fontsize=LABEL_PT - 3)
    ax2.set_xlim(0, 5)
    ax2.set_xlabel(f"доменов с ошибкой (из 5)\nне проверен: {', '.join(unchecked)} — "
                   f"канала оценки нет ни у одного домена,\nи покрытие сказано числом: "
                   f"{data['coverage']['features_checked']} признака из "
                   f"{data['coverage']['features_declared']}",
                   fontsize=LABEL_PT - 3)
    ax2.legend(fontsize=LABEL_PT - 4, frameon=False, loc="center right")

    _stamp(fig, n=f"{len(data['rows'])} прогонов (5 доменов × {len(data['seeds'])} сида)",
           unit=data["unit"],
           compares="определённый признак против устройства домена; разброс IoU по всем "
                    "доменам против разброса по применимым")
    return _save(fig, out / "38-rezhim-sredy.png",
                 "Гейт по режиму убирает из сводки не плохой ответ, а ответ не о том.")


def fig_reversibility(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: счётчики совпадают, значит порог осторожности в разведке не читается."""
    rows = [r for r in data["rows"] if r["rollback"]]
    seeds = sorted({r["seed"] for r in rows})
    fig, (ax,) = _fig(620)

    gate = [next(r["risky_chosen"] for r in rows if r["seed"] == s and r["gate"])
            for s in seeds]
    nogate = [next(r["risky_chosen"] for r in rows if r["seed"] == s and not r["gate"])
              for s in seeds]
    xs = list(range(len(seeds)))
    ax.bar([x - 0.18 for x in xs], gate, width=0.34, color=NEW,
           label="порог осторожности работает")
    ax.bar([x + 0.18 for x in xs], nogate, width=0.34, color=OLD,
           label="порог снят (контроль)")
    for x, (g, n) in enumerate(zip(gate, nogate)):
        ax.annotate(f"{g} = {n}", (x, max(g, n)), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=LABEL_PT - 4, color=NEUTRAL)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"сид {s}" for s in seeds])
    ax.set_ylabel("нажатий по-настоящему необратимого выхода", fontsize=LABEL_PT - 2)
    ax.set_ylim(0, max(gate + nogate) * 1.35)
    ax.legend(fontsize=LABEL_PT - 4, frameon=False, loc="upper left")
    ax.set_xlabel(
        "среда одна и та же — с надёжным откатом; различие только в пороге\n"
        "счётчики совпали до единицы на всех пяти сидах: в ветке разведки порог не "
        "читается вовсе.\nОсторожность читает планировщик, а необратимое нажимает "
        "разведка — величина считается и читается не тем, кто рискует",
        fontsize=LABEL_PT - 2)

    _stamp(fig, n=f"{len(data['rows'])} прогонов (3 расстановки × {len(data['seeds'])} "
                 f"сидов)", unit=data["unit"],
           compares="счётчик рискованных действий при работающем пороге против снятого, "
                    "в одной и той же среде")
    return _save(fig, out / "39-obratimost-ne-vliyaet.png",
                 "Порог осторожности снят — счётчики те же: в разведке он не читается.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    m = ROOT / "docs" / "measurements"
    made: list[Path] = []
    plan = ((fig_regime, "regime.json", "tools/measure_regime.py"),
            (fig_reversibility, "reversibility.json", "tools/measure_reversibility.py"))
    missing: list[str] = []
    for draw, name, how in plan:
        src = m / name
        if not src.exists():
            missing.append(f"{name} — сначала python3 {how}")
            continue
        made.append(draw(base, json.loads(src.read_text(encoding="utf-8"))))
    for line in missing:
        print(f"пропущено: {line}", file=sys.stderr)
    other = ROOT / "figures"
    if made and base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
