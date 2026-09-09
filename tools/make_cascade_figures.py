"""Изображения по каскаду восприятия. SPEC-FULL, A3. Инвариант 28.

43. **Ступень 3 открылась в одном домене из пяти.** Потолок A4.5 («ниже 10 %»)
    формально выполнен во всех 60 прогонах, но на четырёх доменах ступень недостижима
    по построению: рабочий диапазон ошибки предсказания лежит целиком ниже порога.
    Ноль там означает «механизм не работал», а не «механизм уложился».
44. **Цена ступени с глубиной не растёт.** Поток дороже отпечатка в двенадцать раз,
    то есть дорогое стоит перед дешёвым, и порядок ступеней по цене неверен.

Заголовок каждой картинки — утверждение, и оно **проверяется по данным** перед
рисованием: если число перестанет его подтверждать, инструмент падает, а не рисует
подпись, которой данные больше не отвечают.

Запуск: `python3 tools/make_cascade_figures.py [каталог]`. Числа — из
`docs/measurements/cascade.json`.
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
OPEN = "#1f6f43"         # ступень достижима / порог перекрыт
SHUT = "#b42318"         # ступень недостижима / порог не перекрыт
NEUTRAL = "#30414f"
MUTED = "#8a97a3"        # величина не измерена


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


def fig_deep_stage(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: потолок A4.5 выполнен, но на четырёх доменах — вакуумно."""
    scale = data["error_scale"]
    per = scale["per_domain"]
    domains = data["domains"]
    dead = scale["domains_below_model_gate"]

    # Заголовок обязан отвечать данным, а не памяти о прогоне.
    if not (len(dead) == 4 and len(domains) == 5):
        raise ValueError(f"данные больше не подтверждают заголовок: недостижимо на "
                         f"{len(dead)} доменах из {len(domains)}")
    if data["outcome"]["runs_over_launch_ceiling"] != 0:
        raise ValueError("потолок A4.5 больше не выполнен — заголовок неверен")

    fig, (ax1, ax2) = _fig(660, ncols=2)

    # Панель 1: сколько раз за прогон вообще звалась большая модель.
    calls = [per[d]["model_calls_median"] for d in domains]
    colors = [SHUT if d in dead else OPEN for d in domains]
    xs = list(range(len(domains)))
    ax1.bar(xs, calls, color=colors, width=0.6)
    for x, c in zip(xs, calls):
        ax1.annotate(f"{c:.0f}", (x, c), textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=LABEL_PT - 3, color=NEUTRAL)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(domains, fontsize=LABEL_PT - 3, rotation=20, ha="right")
    ax1.set_ylabel("вызовов большой модели за прогон, медиана", fontsize=LABEL_PT - 2)
    ax1.set_ylim(0, max(calls) * 1.3)
    ax1.set_xlabel(
        "красный — ступень недостижима по построению;\nзелёный — открывается\n"
        "единица — вызов на прогон в 600 кадров",
        fontsize=LABEL_PT - 3, loc="left")

    # Панель 2: почему. Порог сравнивается с рабочим диапазоном ошибки домена.
    p95 = [per[d]["p95_max"] or 0.0 for d in domains]
    ax2.barh(xs, p95, color=colors, height=0.6)
    ax2.axvline(scale["model_gate"], color=NEUTRAL, ls="--", lw=1.4)
    ax2.set_yticks(xs)
    ax2.set_yticklabels(domains, fontsize=LABEL_PT - 3)
    ax2.invert_yaxis()
    # Подпись порога ставится после инверсии оси: до неё «верх» — это низ картинки,
    # и первая редакция увела её за край.
    ax2.annotate(f"порог ступени 3 = {scale['model_gate']}",
                 (scale["model_gate"], -0.55), annotation_clip=False,
                 textcoords="offset points", xytext=(-4, 0), ha="right",
                 fontsize=LABEL_PT - 4, color=NEUTRAL)
    ax2.set_xlabel("ошибка предсказания, 95-й процентиль по домену\n"
                   "(доля полного диапазона яркости)", fontsize=LABEL_PT - 3, loc="left")

    _stamp(fig, n="60 прогонов (5 доменов × 3 сида × 4 доли бездействия)",
           unit="прогон",
           compares="полученное против порога, объявленного в схеме до прогона")
    return _save(fig, out / "43-stupen-3-vakuumna.png",
                 "Ступень 3 открылась в одном домене из пяти: на остальных потолок "
                 "выполнен вакуумно.")


def fig_stage_cost(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: цена с глубиной не растёт — поток дороже отпечатка."""
    costs = data["stage_ms_per_call"]
    order = ("change", "flow", "fingerprint", "model")
    names = {"change": "0. изменение", "flow": "1. поток",
             "fingerprint": "2. отпечаток", "model": "3. большая модель"}

    if costs["flow"] <= costs["fingerprint"]:
        raise ValueError("поток больше не дороже отпечатка — заголовок неверен")
    times = costs["flow"] / costs["fingerprint"]
    if not 10.0 <= times <= 14.0:
        raise ValueError(f"отношение {times:.1f}, а в заголовке сказано «в двенадцать»")

    fig, (ax,) = _fig(560)
    xs = list(range(len(order)))
    values = [costs[k] for k in order]
    colors = [SHUT if k == "flow" else (MUTED if costs[k] is None else NEUTRAL)
              for k in order]
    drawn = [0.0 if v is None else v for v in values]
    ax.bar(xs, drawn, color=colors, width=0.6)
    for x, (k, v) in enumerate(zip(order, values)):
        if v is None:
            ax.annotate("не измерена:\nмодели за файрволом\nв этом окружении нет",
                        (x, 0), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=LABEL_PT - 4, color=MUTED)
        else:
            ax.annotate(f"{v:.2f} мс", (x, v), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=LABEL_PT - 3,
                        color=NEUTRAL)
    ax.set_xticks(xs)
    ax.set_xticklabels([names[k] for k in order], fontsize=LABEL_PT - 3)
    ax.set_ylabel("миллисекунд на вызов", fontsize=LABEL_PT - 2)
    ax.set_ylim(0, max(drawn) * 1.25)
    ax.set_xlabel(
        f"красный — ступень, стоящая не на своём месте по цене: она в {times:.0f} раз "
        f"дороже следующей за ней\n"
        "цена снята на процессоре этой машины, кадр 320×180; на другой машине числа "
        "другие, порядок — тот же",
        fontsize=LABEL_PT - 3, loc="left")

    _stamp(fig, n="120 кадров на ступень, домен game",
           unit="вызов ступени",
           compares="ступени между собой на одних и тех же кадрах")
    return _save(fig, out / "44-cena-stupeni-ne-rastet.png",
                 "Цена ступени с глубиной не растёт: поток дороже отпечатка в "
                 "двенадцать раз.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "cascade.json"
    if not src.exists():
        print(f"нет {src.name} — сначала python3 tools/measure_cascade.py",
              file=sys.stderr)
        return 1
    data = json.loads(src.read_text(encoding="utf-8"))
    made = [fig_deep_stage(base, data), fig_stage_cost(base, data)]

    other = ROOT / "figures"
    if base.resolve() != other.resolve():
        other.mkdir(parents=True, exist_ok=True)
        for p in made:
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other.name}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
