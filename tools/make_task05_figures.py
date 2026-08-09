"""Изображения по TASK-05. Инвариант 28: одна картинка — одно утверждение.

Правила, которые здесь соблюдаются механически, а не по памяти:

- **заголовок — предложение, а не тема.** Не «Сверка mu», а «mu двух маршрутов
  согласуются: 9 из 10 пар при z<2»;
- **на каждой обязательны `n` и единица независимости.** Проверяется функцией
  `_stamp`, и без них картинка не сохраняется;
- **не более четырёх панелей.** Больше — значит утверждений несколько;
- **значащие разряды по смыслу величины:** доля — два знака, время — один;
- **читаемо на телефоне:** ширина 1200 px, подписи от 14 px, оси от 12 px;
- **ничего декоративного:** без градиентов, свечения, теней, объёма. Цвет только как
  значение, и в подписи сказано, какое.

Запуск: `python3 tools/make_task05_figures.py [каталог]`. По умолчанию
`docs/figures/`, и туда же кладётся копия в `figures/` — обе раскладки отслеживаются
в репозитории.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

WIDTH_PX = 1200
DPI = 100
TITLE_PT = 15
LABEL_PT = 14
TICK_PT = 12

# Цвет только как значение. Три роли, и каждая названа в подписи там, где встречается.
PROVEN = "#1a7f37"      # замер дал число
VACUUM = "#8a8a8a"      # замер вакуумен: величина не несёт информации
ALARM = "#b42318"       # дефект или отказ
NEUTRAL = "#30414f"


def _fig(height_px: int, ncols: int = 1, nrows: int = 1):
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(WIDTH_PX / DPI, height_px / DPI), dpi=DPI)
    if ncols * nrows == 1:
        axes = [axes]
    else:
        axes = list(axes.ravel()) if hasattr(axes, "ravel") else list(axes)
    if len(axes) > 4:
        raise ValueError(
            f"панелей {len(axes)}: больше четырёх. Больше панелей — значит "
            "утверждений несколько, значит картинок несколько (инвариант 28)")
    for ax in axes:
        ax.tick_params(labelsize=TICK_PT)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig, axes


def _stamp(fig, *, n: str, unit: str, compares: str = "") -> None:
    """Штамп с `n` и единицей независимости. Без него картинка не сохраняется."""
    if not n or not unit:
        raise ValueError("на изображении обязательны n и единица независимости "
                         "(инвариант 28)")
    text = f"n = {n}   ·   единица независимости: {unit}"
    fig.text(0.01, 0.048, text, fontsize=LABEL_PT - 1, color=NEUTRAL)
    if compares:
        # Второй строкой, а не в конце первой: длинная строка обрезается по краю
        # изображения, и тогда `n` вида «41 пар (все непересекающи…» перестаёт быть
        # проверяемым — а именно его на картинке и надо читать.
        fig.text(0.01, 0.012, f"сравнивается: {compares}",
                 fontsize=LABEL_PT - 2, color=NEUTRAL)


def _save(fig, path: Path, title: str) -> Path:
    if not title.endswith((".", "!", "?")) and ":" not in title:
        raise ValueError(f"заголовок «{title}» — тема, а не предложение "
                         "(инвариант 28)")
    if len(title) > 96:
        raise ValueError(
            f"заголовок длиной {len(title)} символов обрежется по краю при 1200 px. "
            "Утверждение должно влезать целиком, иначе картинка сообщает половину")
    fig.suptitle(title, fontsize=TITLE_PT, x=0.01, ha="left", y=0.985)
    fig.tight_layout(rect=(0, 0.115, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"нарисовано: {path.name}")
    return path


# ---------------------------------------------------------------------------
# 10. Сверка mu: контроль против нового мира
# ---------------------------------------------------------------------------


def fig_mu(out: Path, data: dict[str, dict[str, Any]]) -> Path:
    """Одно утверждение: на прежнем мире сверить нечего, на новом — сходится."""
    order = ["control", "nojitter", "new"]
    names = {"control": "прежний мир\n(контроль)",
             "nojitter": "переменная цена,\nразброс 0",
             "new": "переменная цена\n+ ограниченный"}
    fig, (ax1, ax2) = _fig(520, ncols=2)

    # Панель 1: сколько пар вообще можно сверить.
    measurable, unmeasurable = [], []
    for key in order:
        r = data[key]["one_step"]
        measurable.append(r["scorable"])
        unmeasurable.append(r["unmeasured_spread"])
    xs = range(len(order))
    ax1.bar(xs, unmeasurable, color=VACUUM, label="sigma нулевая: сверять нечем")
    ax1.bar(xs, measurable, bottom=unmeasurable, color=PROVEN,
            label="разброс измерен: сверка возможна")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([names[k] for k in order], fontsize=TICK_PT)
    ax1.set_ylabel("пар мест с двумя рёбрами", fontsize=LABEL_PT)
    ax1.legend(fontsize=LABEL_PT - 2, frameon=False, loc="upper left")
    for i, (m, u) in enumerate(zip(measurable, unmeasurable)):
        ax1.text(i, m + u + 0.4, f"{m} из {m + u}", ha="center", fontsize=LABEL_PT)

    # Панель 2: доля согласившихся там, где сверка возможна.
    shares, labels, colors = [], [], []
    for key in order:
        r = data[key]["one_step"]
        if r["agreed_share"] is None:
            shares.append(0.0)
            labels.append("замер\nне поставлен")
            colors.append(VACUUM)
        else:
            shares.append(r["agreed_share"])
            labels.append(f"{r['agreed']}/{r['scorable']}")
            colors.append(PROVEN if r["agreed_share"] >= 0.8 else ALARM)
    ax2.bar(xs, shares, color=colors)
    ax2.axhline(0.8, color=NEUTRAL, linestyle="--", linewidth=1)
    ax2.text(-0.42, 0.74, "порог согласия 0.80, объявлен до прогона",
             fontsize=LABEL_PT - 2, ha="left", color=NEUTRAL)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([names[k] for k in order], fontsize=TICK_PT)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("доля пар с z < 2", fontsize=LABEL_PT)
    for i, (v, lab) in enumerate(zip(shares, labels)):
        ax2.text(i, v + 0.03, lab, ha="center", fontsize=LABEL_PT)

    new = data["new"]["one_step"]
    title = (f"На прежнем мире сверять нечем, на новом mu сходятся: "
             f"{new['agreed']} из {new['scorable']} пар при z<2")
    _stamp(fig, n=f"{new['pairs_independent']} пар (все непересекающиеся по рёбрам)",
           unit="пара мест с двумя маршрутами",
           compares="три конфигурации мира, восемь сидов каждая")
    return _save(fig, out / "10-svertka-mu.png", title)


# ---------------------------------------------------------------------------
# 11. Вырождение: время против топологии
# ---------------------------------------------------------------------------


def fig_degeneracy(out: Path, data: dict[str, dict[str, Any]]) -> Path:
    """Одно утверждение: в прежнем мире mu выводится из числа шагов, в новом — нет."""
    fig, axes = _fig(480, ncols=2)
    ax1, ax2 = axes

    for ax, key, name in ((ax1, "control", "прежний мир"),
                          (ax2, "new", "новый мир")):
        r = data[key]["one_step"]
        mus = [e["a"]["mu"] for e in r["examples"]] + [e["b"]["mu"]
                                                        for e in r["examples"]]
        sig = [e["a"]["sigma"] for e in r["examples"]] + [e["b"]["sigma"]
                                                          for e in r["examples"]]
        if not mus:
            ax.text(0.5, 0.5, "пар нет", ha="center", transform=ax.transAxes,
                    fontsize=LABEL_PT)
            continue
        # Миллисекунды, а не секунды: sigma здесь порядка сотых секунды, и один
        # знак после запятой дал бы «0.0 с» — ровно то ненастоящее число, которое
        # правила части 5 запрещают. Значащие разряды по смыслу величины.
        mus_ms = [x * 1000 for x in mus]
        sig_ms = [x * 1000 for x in sig]
        ax.scatter(mus_ms, sig_ms, s=48, color=PROVEN if any(sig) else VACUUM)
        ax.set_xlabel("mu ребра, мс", fontsize=LABEL_PT)
        ax.set_ylabel("sigma ребра, мс", fontsize=LABEL_PT)
        span = (f"{min(mus_ms):.0f}…{max(mus_ms):.0f}" if max(mus) > min(mus)
                else f"{mus_ms[0]:.0f}")
        ax.set_title(f"{name}: mu {span} мс, sigma "
                     + ("0 у всех" if not any(sig) else
                        f"{min(sig_ms):.0f}…{max(sig_ms):.0f} мс"),
                     fontsize=LABEL_PT, loc="left")
        ax.set_ylim(-2, max(10.0, max(sig_ms) * 1.3))
        ax.set_xlim(min(mus_ms) - 10, max(mus_ms) + 10)

    title = ("Время перестало быть тождественным топологии: sigma рёбер была "
             "нулевой у всех, стала ненулевой")
    _stamp(fig, n="по 8 примеров рёбер из каждой конфигурации",
           unit="ребро графа мест",
           compares="прежний мир против переменной цены с разбросом")
    return _save(fig, out / "11-vremya-i-topologiya.png", title)


# ---------------------------------------------------------------------------
# 12. Ветки арбитража
# ---------------------------------------------------------------------------


def fig_branches(out: Path, counts: dict[str, int]) -> Path:
    """Одно утверждение: одна ветка не срабатывает ни разу, и это дефект."""
    fig, (ax,) = _fig(430)
    names = list(counts)
    vals = [counts[k] for k in names]
    total = sum(vals)
    colors = [ALARM if v == 0 else NEUTRAL for v in vals]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=TICK_PT)
    ax.invert_yaxis()
    ax.set_xlabel("срабатываний за прогон", fontsize=LABEL_PT)
    for i, v in enumerate(vals):
        ax.text(v + total * 0.01, i, f"{v}  ({v / total:.0%})",
                va="center", fontsize=LABEL_PT)
    ax.set_xlim(0, max(vals) * 1.25)
    ax.text(0.99, 0.04, "красным — ноль срабатываний, докладывается как дефект",
            transform=ax.transAxes, ha="right", fontsize=LABEL_PT - 2, color=ALARM)

    dead = [k for k, v in counts.items() if v == 0]
    title = (f"Ветка «{dead[0]}» мертва: 0 срабатываний из {total} решений"
             if dead else
             f"Все ветки арбитража срабатывают: дефектов нет на {total} решениях")
    _stamp(fig, n=f"{total} решений на 4 сидах", unit="прогон разведки",
           compares="ветки одного арбитража между собой")
    return _save(fig, out / "12-vetki-arbitrazha.png", title)


# ---------------------------------------------------------------------------
# 13. Мощность замера по длинам планов
# ---------------------------------------------------------------------------


def fig_power(out: Path, observed_pp: float, detectable: dict[int, float]) -> Path:
    """Одно утверждение: наблюдаемый эффект меньше различимого во столько-то раз."""
    fig, (ax,) = _fig(430)
    ns = sorted(detectable)
    vals = [detectable[n] for n in ns]
    ax.plot(ns, vals, marker="o", color=NEUTRAL, linewidth=2)
    ax.axhline(observed_pp, color=ALARM, linewidth=2)
    ax.text(ns[-1], observed_pp + 2,
            f"наблюдаемая разница между длинами: {observed_pp:.0f} п.п.",
            ha="right", fontsize=LABEL_PT, color=ALARM)
    ax.axhline(10, color=PROVEN, linestyle="--", linewidth=1)
    ax.text(ns[0], 12, "порог запуска из TASK-05: 10 п.п.",
            fontsize=LABEL_PT - 2, color=PROVEN)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], fontsize=TICK_PT)
    ax.set_xlabel("маршрутов на класс длины", fontsize=LABEL_PT)
    ax.set_ylabel("различимая разница, п.п.", fontsize=LABEL_PT)
    for n, v in zip(ns, vals):
        ax.text(n, v + 2, f"{v:.0f}", ha="center", fontsize=LABEL_PT)

    need = next((n for n in ns if detectable[n] <= 10), None)
    title = (f"Замер по длинам не запускается: различимо {vals[0]:.0f} п.п. "
             f"при наблюдаемых {observed_pp:.0f}")
    _stamp(fig, n=f"расчёт мощности, точный тест Фишера, порог 0.05",
           unit="маршрут",
           compares="различимая разница против наблюдаемой")
    return _save(fig, out / "13-moshchnost-dlin.png", title)


def main(argv: Sequence[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else Path("docs/figures")
    scratch = Path(argv[2]) if len(argv) > 2 else Path(
        "/tmp/claude-0/-home-user-AI/a1c8b332-6d27-5416-a8ea-6af499fe0527/scratchpad")

    made: list[Path] = []
    files = {"control": scratch / "t5_control.json",
             "nojitter": scratch / "t5_nojitter.json",
             "new": scratch / "t5_new.json"}
    if all(p.exists() for p in files.values()):
        data = {k: json.loads(p.read_text(encoding="utf-8"))
                for k, p in files.items()}
        made.append(fig_mu(base, data))
        made.append(fig_degeneracy(base, data))
    else:
        missing = [str(p.name) for p in files.values() if not p.exists()]
        print(f"пропущены рисунки сверки mu: нет {missing}", file=sys.stderr)

    made.append(fig_branches(base, {
        "нет модели здесь": 2081, "подтвердить здесь": 123,
        "непробованное здесь": 1533, "идти в недозамкнутое": 332,
        "непробованное всё же": 1931, "прежняя разведка": 0}))

    # Мощность: пересчитывается здесь же, чтобы литералов в подписи не было.
    from math import comb

    def fisher(a: int, b: int, c: int, d: int) -> float:
        n, row1, row2, col1 = a + b + c + d, a + b, c + d, a + c
        def prob(x: int) -> float:
            return comb(row1, x) * comb(row2, col1 - x) / comb(n, col1)
        obs = prob(a)
        lo, hi = max(0, col1 - row2), min(row1, col1)
        return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= obs + 1e-12)

    detectable: dict[int, float] = {}
    for n in (10, 20, 50, 100, 200):
        best = 100.0
        for k1 in range(n, n // 2, -1):
            for k2 in range(k1):
                if fisher(k1, n - k1, k2, n - k2) < 0.05:
                    best = min(best, (k1 - k2) / n * 100)
        detectable[n] = best
    made.append(fig_power(base, observed_pp=6.0, detectable=detectable))

    # Обе раскладки рисунков отслеживаются в репозитории и уже расходились однажды.
    other = Path("figures") if base.name == "figures" and False else Path("figures")
    if base.resolve() != other.resolve():
        for p in made:
            other.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, other / p.name)
        print(f"скопировано в {other}/: {len(made)} файлов")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
