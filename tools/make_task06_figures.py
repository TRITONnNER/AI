"""Изображения по TASK-06. Инвариант 28: одна картинка — одно утверждение.

Три утверждения, три картинки:

- **кривая баланса** — что замер не разрешился, и почему это видно, а не выведено;
- **таблица расхождений** — что стало с пятью расхождениями аудита;
- **состояние М3** — сколько пунктов проверено замером и сколько на живом экране.

Правила соблюдаются механически, как в `make_task05_figures.py`: заголовок —
предложение и не длиннее 96 символов, `n` и единица независимости обязательны,
панелей не больше четырёх, ширина 1200 px, ничего декоративного, цвет только как
значение с названием в подписи.

Запуск: `python3 tools/make_task06_figures.py [каталог]`. По умолчанию
`docs/figures/`, копия в `figures/` — обе раскладки отслеживаются в репозитории.
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
sys.path.insert(0, str(ROOT / "tools"))

WIDTH_PX = 1200
DPI = 100
TITLE_PT = 15
LABEL_PT = 14
TICK_PT = 12

# Цвет только как значение. Каждая роль названа в подписи там, где встречается.
PROVEN = "#1a7f37"      # проверено замером
PARTIAL = "#b8860b"     # есть, но проверенного числа нет
ALARM = "#b42318"       # нет совсем, либо дефект
VACUUM = "#8a8a8a"      # замер не различает
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
            f"заголовок длиной {len(title)} символов обрежется по краю при 1200 px. "
            "Утверждение должно влезать целиком, иначе картинка сообщает половину")
    fig.suptitle(title, fontsize=TITLE_PT, x=0.01, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0.115, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"нарисовано: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 14. Кривая баланса: замер не разрешился
# ---------------------------------------------------------------------------


def fig_balance(out: Path, three: dict[str, Any], eight: dict[str, Any]) -> Path:
    """Одно утверждение: величина слишком разрежена, чтобы кривая локализовалась."""
    fig, (ax1, ax2) = _fig(560, ncols=2)
    shares = ["0.0", "0.15", "0.3", "0.5", "1.0"]
    xs = range(len(shares))

    # Панель 1: сколько прогонов из восьми дали ненулевую целевую величину.
    nonzero = [eight[s]["both_nonzero"] for s in shares]
    runs = eight[shares[0]]["n_runs"]
    # Один цвет, а не три по величине: столбик нулевой высоты цвета не показывает,
    # и подпись «красный — ноль прогонов» описывала бы невидимое. Ноль называется
    # числом и цветом текста, а не цветом столбика, которого нет.
    ax1.bar(xs, nonzero, color=VACUUM)
    ax1.axhline(runs, color=NEUTRAL, linestyle="--", linewidth=1)
    ax1.text(-0.45, runs - 0.55, f"все {runs} прогонов",
             fontsize=LABEL_PT - 2, color=NEUTRAL)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(shares, fontsize=TICK_PT)
    ax1.set_ylabel("прогонов с ненулевым результатом", fontsize=LABEL_PT)
    ax1.set_ylim(0, runs + 0.5)
    for i, v in enumerate(nonzero):
        ax1.text(i, v + 0.12, str(v), ha="center", fontsize=LABEL_PT,
                 color=ALARM if v == 0 else NEUTRAL,
                 fontweight="bold" if v == 0 else "normal")
    ax1.set_xlabel("доля шагов на замыкание\nкрасный ноль — ни одного прогона "
                   "с результатом", fontsize=LABEL_PT)

    # Панель 2: мощность. Сколько прогонов нужно, чтобы вопрос стал разрешимым.
    need = {8: 0.01, 12: 0.05, 20: 0.29, 30: 0.47, 40: 0.76, 60: 0.97}
    ns = list(need)
    ax2.plot(ns, [need[n] for n in ns], marker="o", color=NEUTRAL, linewidth=1.5)
    ax2.axhline(0.8, color=PROVEN, linestyle="--", linewidth=1)
    ax2.text(9, 0.83, "мощность 0.80", fontsize=LABEL_PT - 2, color=PROVEN)
    ax2.scatter([runs], [need[8]], s=90, color=ALARM, zorder=5)
    ax2.annotate("сделано", (runs, need[8]), textcoords="offset points",
                 xytext=(8, 10), fontsize=LABEL_PT - 1, color=ALARM)
    ax2.set_ylabel("мощность теста Фишера", fontsize=LABEL_PT)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("прогонов на класс доли\nсделанного объёма хватает на мощность "
                   "0.01", fontsize=LABEL_PT)

    _stamp(fig, n=f"{runs} прогонов на класс доли, 5 классов",
           unit="прогон (мир, сид)",
           compares="доля 0 против четырёх ненулевых долей, p = 0.19")
    return _save(fig, out / "14-krivaya-balansa.png",
                 "Кривая баланса не локализуется: величина ненулевая в 1 прогоне "
                 "из 6")


# ---------------------------------------------------------------------------
# 15. Расхождения аудита: что стало с каждым
# ---------------------------------------------------------------------------


def fig_divergences(out: Path) -> Path:
    """Одно утверждение: все пять расхождений закрыты, три — в этой задаче."""
    # (расхождение, где найдено, где закрыто, исход)
    rows = [
        ("Формат записи, actor_layer", 1, 2, "документ прав"),
        ("Hypothesis без теста как вопрос", 1, 2, "документ прав"),
        ("Текст: пространства имён", 1, 6, "документ прав"),
        ("Эмоции: семь осей", 1, 6, "документ прав"),
        ("Драйвы: набор дан, не выведен", 1, 6, "оба описывают разное"),
    ]
    # Цвет — исход, положение по оси — когда закрыто. Обе величины настоящие:
    # столбик постоянной длины не кодировал ничего и был украшением.
    outcome_color = {"документ прав": PROVEN, "оба описывают разное": PARTIAL}
    fig, (ax,) = _fig(470)
    ys = list(range(len(rows)))
    for y, (_, found, closed, outcome) in zip(ys, rows):
        ax.plot([found, closed], [y, y], color=VACUUM, linewidth=1.5, zorder=1)
        ax.scatter([found], [y], s=70, facecolors="white", edgecolors=VACUUM,
                   linewidths=1.5, zorder=3)
        ax.scatter([closed], [y], s=110, color=outcome_color[outcome], zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=LABEL_PT)
    ax.invert_yaxis()
    ax.set_xticks([1, 2, 6])
    ax.set_xticklabels(["TASK-01\nнайдено", "TASK-02\nзакрыто",
                        "TASK-06\nзакрыто"], fontsize=TICK_PT)
    ax.set_xlim(0.5, 7.2)
    for name, color in outcome_color.items():
        ax.scatter([], [], s=110, color=color, label=f"исход: {name}")
    ax.scatter([], [], s=70, facecolors="white", edgecolors=VACUUM,
               linewidths=1.5, label="пустой круг — где найдено")
    # Легенда над полем, а не внутри: внутри она накрыла две последние точки —
    # то есть заслонила ровно то, о чём картинка.
    ax.legend(fontsize=LABEL_PT - 3, frameon=False, ncol=3,
              loc="lower left", bbox_to_anchor=(-0.28, 1.01),
              handletextpad=0.4, columnspacing=1.4)
    ax.spines["left"].set_visible(False)

    _stamp(fig, n="5 расхождений имени с механизмом",
           unit="расхождение",
           compares="три исхода: код прав, документ прав, оба описывают разное")
    return _save(fig, out / "15-rashozhdeniya.png",
                 "Пять расхождений аудита закрыты решениями, три из них — в этой "
                 "задаче.")


# ---------------------------------------------------------------------------
# 16. Состояние М3
# ---------------------------------------------------------------------------


def fig_m3(out: Path, m3: dict[str, Any]) -> Path:
    """Одно утверждение: замерено почти всё, на живом экране — ничего."""
    fig, (ax1, ax2) = _fig(520, ncols=2)

    # Панель 1: три градации.
    titles = {"measured": "проверено\nзамером", "unmeasured": "есть, но\nне мерено",
              "absent": "нет\nсовсем"}
    order = ["measured", "unmeasured", "absent"]
    vals = [m3["by_grade"][g] for g in order]
    ax1.bar(range(3), vals, color=[PROVEN, PARTIAL, ALARM])
    ax1.set_xticks(range(3))
    ax1.set_xticklabels([titles[g] for g in order], fontsize=TICK_PT)
    ax1.set_ylabel("пунктов М3", fontsize=LABEL_PT)
    ax1.set_ylim(0, m3["total"] + 0.8)
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.15, str(v), ha="center", fontsize=LABEL_PT)

    # Панель 2: на чём именно проверено. Здесь и стоит ноль.
    synth = sum(1 for i in m3["items"]
                if i["grade"] == "measured" and i["basis"] == "синтетика")
    live = m3["measured_on_live"]
    ax2.bar([0, 1], [synth, live], color=[VACUUM, ALARM])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["на синтетике", "на живом экране"], fontsize=TICK_PT)
    ax2.set_ylabel("пунктов, проверенных замером", fontsize=LABEL_PT)
    ax2.set_ylim(0, m3["total"] + 0.8)
    ax2.text(0, synth + 0.15, str(synth), ha="center", fontsize=LABEL_PT)
    ax2.text(1, live + 0.15, str(live), ha="center", fontsize=LABEL_PT,
             color=ALARM, fontweight="bold")
    # Подпись под осью, а не заголовком панели: заголовком она налезала на
    # собственную подпись оси Y соседней панели.
    ax2.set_xlabel("серый — мир писался вместе с проверяемым кодом",
                   fontsize=LABEL_PT - 2, color=NEUTRAL)

    _stamp(fig, n=f"{m3['total']} пунктов вехи М3 из ROADMAP.md",
           unit="пункт вехи",
           compares="синтетический мир против живого экрана")
    return _save(fig, out / "16-sostoyanie-m3.png",
                 f"М3: замером проверено {m3['by_grade']['measured']} пунктов из "
                 f"{m3['total']}, на живом экране — {m3['measured_on_live']}")


# ---------------------------------------------------------------------------
# 17. Расход места: измеренный против оценённого (TASK-08)
# ---------------------------------------------------------------------------


def fig_storage(out: Path, rows: list[dict[str, Any]]) -> Path:
    """Одно утверждение: три захода, и разница была в содержимом, а не в конвейере."""
    fig, (ax1, ax2) = _fig(540, ncols=2)
    order = ["неподвижный экран", "экран с движением", "рабочий стол", "шум"]
    by = {r["kind"]: r for r in rows}
    vals = [by[k]["kib_per_frame"] for k in order if k in by]
    # Подписи короткие: длинные при четырёх столбцах налезают друг на друга.
    names = ["неподвижный", "нарисованный", "рабочий стол", "шум"]
    colors = [VACUUM, ALARM, PROVEN, NEUTRAL]

    ax1.bar(range(len(vals)), vals, color=colors[:len(vals)])
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(vals)))
    ax1.set_xticklabels(names[:len(vals)], fontsize=TICK_PT)
    ax1.set_ylabel("КиБ на кадр, 1080p", fontsize=LABEL_PT)
    for i, v in enumerate(vals):
        ax1.text(i, v * 1.3, f"{v:g}", ha="center", fontsize=LABEL_PT)
    # Живое число — линией: то, с чем всё сравнивается.
    ax1.axhline(49.3, color=PROVEN, linestyle="--", linewidth=1.5)
    ax1.text(-0.45, 60, "живой экран: 49.3", fontsize=LABEL_PT - 2, color=PROVEN)
    ax1.set_xlabel("красный — синтетика, по которой оценка ошиблась в 14 раз\n"
                   "зелёный — синтетика, доведённая до похожести",
                   fontsize=LABEL_PT - 2, color=NEUTRAL)

    # Панель 2: три захода на одно число против живого замера.
    passes = [("TASK-07\nиз головы", 90.0, ALARM),
              ("TASK-08\nнарисованный", 3.6, ALARM),
              ("TASK-09\nживой замер", 49.3, PROVEN)]
    ax2.barh(range(3), [p[1] for p in passes], color=[p[2] for p in passes],
             height=0.55)
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([p[0] for p in passes], fontsize=LABEL_PT - 1)
    ax2.invert_yaxis()
    ax2.set_xlabel("КиБ на кадр, которые считали верными", fontsize=LABEL_PT)
    for i, p in enumerate(passes):
        ax2.text(p[1] + 2, i, f"{p[1]:g}", va="center", fontsize=LABEL_PT,
                 color=p[2])
    ax2.set_xlim(0, 100)

    _stamp(fig, n=f"{len(vals)} вида содержимого, по 200 кадров, плюс живой замер",
           unit="вид содержимого",
           compares="три оценки одной величины против замера на настоящем экране")
    return _save(fig, out / "17-raskhod-mesta.png",
                 "Расход зависит от содержимого: нарисованный экран сжимается в 14 раз "
                 "лучше живого.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    made = []

    three = json.loads((ROOT / "docs" / "measurements" / "balance_curve.json")
                       .read_text(encoding="utf-8"))["summary"]
    eight = json.loads((ROOT / "docs" / "measurements" /
                        "balance_curve_8seeds.json").read_text(encoding="utf-8"))
    made.append(fig_balance(base, three, eight["summary"]))
    made.append(fig_divergences(base))

    import project_status as ps
    made.append(fig_m3(base, ps.m3_summary()))

    rate = json.loads((ROOT / "docs" / "measurements" / "storage_rate.json")
                      .read_text(encoding="utf-8"))
    made.append(fig_storage(base, rate))

    # Обе раскладки рисунков отслеживаются в репозитории и уже расходились однажды.
    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
