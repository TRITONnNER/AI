"""Изображения по TASK-10. Инвариант 28: одна картинка — одно утверждение.

Два утверждения, две картинки:

- **18** — пять из семи «дефектов» третьей редакции оказались ошибкой модели путей, а
  не кода. Это главный результат задачи: детектор врал о коде, пока ему не объявили
  применимость, и признать это числом важнее, чем показать конечный ноль;
- **19** — счётчики в прогоне читают 36 настроек из 155, поэтому путь устанавливает
  только статика, и молчание прогона доказательством не является.

Правила соблюдаются механически, как в `make_task06_figures.py`. Запуск:
`python3 tools/make_task10_figures.py [каталог]`.
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
REAL = "#b42318"        # настоящий дефект кода
FALSE = "#8a8a8a"       # ошибка модели путей детектора
DONE = "#1a7f37"        # прочитано, проверено
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
# 18. Детектор врал о коде, пока ему не объявили применимость
# ---------------------------------------------------------------------------


def fig_redactions(out: Path, data: dict[str, Any]) -> Path:
    """Одно утверждение: почти все «дефекты» были ошибкой детектора, а не кода."""
    reds = data["редакции"]
    total = int(data["всего_настроек"])
    fig, (ax1, ax2) = _fig(580, ncols=2)

    names = [str(r["название"]).split(":")[0] for r in reds]
    claimed = [int(r["объявлено"]) for r in reds]
    real = [int(r["настоящих"]) for r in reds]
    false = [c - t for c, t in zip(claimed, real)]
    xs = range(len(reds))

    ax1.bar(xs, real, color=REAL, label="настоящий дефект кода")
    ax1.bar(xs, false, bottom=real, color=FALSE,
            label="ошибка модели путей детектора")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(names, fontsize=TICK_PT)
    ax1.set_ylabel("объявлено дефектными, настроек", fontsize=LABEL_PT)
    for i, c in enumerate(claimed):
        ax1.text(i, c + 2.5, str(c), ha="center", fontsize=LABEL_PT, color=NEUTRAL)
    ax1.legend(fontsize=LABEL_PT - 2, frameon=False, loc="upper right")
    ax1.set_xlabel("редакция модели путей\nкод между 1 и 4 не менялся",
                   fontsize=LABEL_PT)

    # Панель 2: доля ложных срабатываний по редакциям — то же число иначе, потому что
    # именно доля решает, будут ли в детектор смотреть.
    share = [f / c if c else 0.0 for f, c in zip(false, claimed)]
    ax2.plot(list(xs), share, marker="o", color=NEUTRAL, linewidth=1.5)
    for i, s in enumerate(share):
        ax2.text(i, s + 0.035, f"{s:.0%}", ha="center", fontsize=LABEL_PT,
                 color=FALSE if s > 0 else DONE)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(names, fontsize=TICK_PT)
    ax2.set_ylim(-0.05, 1.12)
    ax2.set_ylabel("доля ложных срабатываний", fontsize=LABEL_PT)
    ax2.set_xlabel("редакция модели путей\nпри 99 % в детектор перестают смотреть",
                   fontsize=LABEL_PT)

    _stamp(fig, n=f"{total} настроек схемы, 4 редакции модели путей",
           unit="настройка",
           compares="объявлено дефектными против подтверждённых чтением кода")
    return _save(fig, out / "18-model-putey.png",
                 "Из 7 «дефектов» третьей редакции 5 были ошибкой детектора, "
                 "а не кода.")


# ---------------------------------------------------------------------------
# 19. Почему нужны оба способа
# ---------------------------------------------------------------------------


def fig_methods(out: Path, data: dict[str, Any]) -> Path:
    """Одно утверждение: прогон покрывает малую долю схемы, путь даёт только статика."""
    total = int(data["всего_настроек"])
    run = data["прогон"]
    read = int(run["прочитано_ключей"])
    fig, (ax1, ax2) = _fig(560, ncols=2)

    # Панель 1: покрытие. Статика видит все объявленные настройки, счётчик — только
    # прочитанные в этом прогоне.
    ax1.barh([1, 0], [total, read], color=[DONE, FALSE], height=0.5)
    ax1.set_yticks([1, 0])
    ax1.set_yticklabels(["статика (AST)", f"прогон, {run['шагов']} шагов"],
                        fontsize=LABEL_PT)
    for y, v in ((1, total), (0, read)):
        ax1.text(v + 2, y, str(v), va="center", fontsize=LABEL_PT, color=NEUTRAL)
    ax1.set_xlim(0, total * 1.12)
    ax1.set_xlabel("настроек, попавших в поле зрения способа\nсерый — прогон видит "
                   f"{read / total:.0%} схемы", fontsize=LABEL_PT)

    # Панель 2: что каждый способ умеет и чего не умеет. Не декорация: именно из этого
    # следует, что диагноз ставит статика, а счётчик её проверяет.
    rows = [("знает путь чтения", 1, 0),
            ("видит вычисленный ключ", 0, 1),
            ("видит непройденный путь", 1, 0),
            ("отличает печать от чтения", 1, 0)]
    ys = range(len(rows))
    ax2.barh([y + 0.18 for y in ys], [r[1] for r in rows], height=0.32, color=DONE)
    ax2.barh([y - 0.18 for y in ys], [r[2] for r in rows], height=0.32, color=FALSE)
    ax2.set_yticks(list(ys))
    ax2.set_yticklabels([r[0] for r in rows], fontsize=LABEL_PT - 1)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["нет", "да"], fontsize=TICK_PT)
    # Легенды нет намеренно: полоса идёт во всю ширину панели, и любая рамка легенды
    # легла бы поверх значения. Роли названы в подписи оси — это и есть требование
    # инварианта 28 «цвет только как значение, и в подписи сказано какое».
    ax2.set_xlabel("верхняя полоса зелёная — статика\nнижняя серая — счётчики в прогоне",
                   fontsize=LABEL_PT)

    _stamp(fig, n=f"{total} настроек схемы, 1 прогон на {run['шагов']} шагов",
           unit="настройка",
           compares=f"покрытие двух способов; расхождений «статика слепа» "
                    f"{run['статика_слепа']}, «путь не пройден» "
                    f"{run['путь_не_пройден']}")
    return _save(fig, out / "19-dva-sposoba.png",
                 f"Счётчики в прогоне видят {read} настроек из {total}: путь даёт "
                 "только статика.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    src = ROOT / "docs" / "measurements" / "params_audit.json"
    if not src.exists():
        print("нет docs/measurements/params_audit.json — сначала "
              "python3 tools/measure_params_audit.py", file=sys.stderr)
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))

    made = [fig_redactions(base, data), fig_methods(base, data)]

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
