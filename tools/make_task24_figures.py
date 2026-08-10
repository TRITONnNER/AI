"""Изображения по TASK-24. Инвариант 28: одна картинка — одно утверждение.

Четыре направления, четыре утверждения, по одной картинке на каждое:

32. **A, конфабуляция.** После переноса ссылки доля расхождений перестала совпадать с долей
    действий рефлекса: разрыв вырос с 0.12 п.п. до 40.9 п.п.
33. **B, драйвы.** Заведомая связь находится, и на классе, который в мире не делает ничего,
    не находится ничего.
34. **C, внимание.** Арбитраж дешевле контроля ровно там, где бюджет тесен, и не даёт ничего,
    когда бюджет перестал быть дефицитом.
35. **D, час работы.** Цикл замедлился в девять раз, и это граф мест, а не утечка памяти;
    полка узлов по оси времени была ложным совпадением.

Запуск: `python3 tools/make_task24_figures.py [каталог]`. Числа — из
`docs/measurements/{live_cycle,drives,attention,hour}.json`. Отсутствие часового замера не
мешает остальным трём: печатается предупреждение, а не отказ.
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
NOW = "#b42318"          # величина после правки / арбитраж / истинная связь
BEFORE = "#30414f"       # величина до правки / контроль / фон
MUTED = "#8a9199"        # то, что находиться не должно
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


# --- A: конфабуляция перестала быть переименованием --------------------------

#: Разрыв до правки. Число из того же замера предыдущей редакции (`31-…png`): ссылка на
#: объяснение ставилась на каждое доставленное действие, и метрика совпадала с долей
#: рефлекса до 0.12 п.п. Хранится здесь, потому что старая запись перезаписана новой, а
#: утверждение «стало иначе» без прежнего числа не проверяемо.
GAP_BEFORE_PP = 0.12


def fig_confab(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: после переноса ссылки две величины разошлись, и разрыв — числом."""
    keys = list(data["by_ratio"])
    rows = [data["by_ratio"][k] for k in keys]
    fig, (ax,) = _fig(620)

    ys = list(range(len(rows)))
    conf = [r["confab_median"] for r in rows]
    reflex = [r["reflex_share_median"] for r in rows]
    for y, (c, r) in enumerate(zip(conf, reflex)):
        ax.plot([min(c, r), max(c, r)], [y, y], color=NEUTRAL, linewidth=1.0, zorder=2)
    ax.scatter(reflex, ys, s=130, color=BEFORE, zorder=3,
               label="доля действий, начатых рефлексом")
    ax.scatter(conf, ys, s=60, color=NOW, zorder=4,
               label="доля расхождений там, где планировщик ждал исполнения")
    for y, (c, r) in enumerate(zip(conf, reflex)):
        right = max(c, r) < 0.55
        ax.annotate(f"разрыв {abs(c - r) * 100:.1f} п.п.",
                    (max(c, r) if right else min(c, r), y),
                    textcoords="offset points", xytext=(14 if right else -14, -4),
                    ha="left" if right else "right",
                    fontsize=LABEL_PT - 3, color=NEUTRAL)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"рефлекс/планировщик {k}\n{r['hz_reflex']:g} и "
                        f"{r['hz_planner']:g} Гц" for k, r in zip(keys, rows)],
                       fontsize=LABEL_PT - 3)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1.5, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.legend(fontsize=LABEL_PT - 3, loc="upper center", frameon=False, ncol=1)
    ax.set_xlabel(
        "доля от действий, за которые планировщик считает себя причиной\n"
        f"наибольший разрыв двух величин — {data['max_gap_to_reflex_share'] * 100:.1f} "
        f"п.п. против {GAP_BEFORE_PP:.2f} п.п. до переноса ссылки.\n"
        "Метрика больше не пересказывает распределение по слоям другими словами",
        fontsize=LABEL_PT)

    _stamp(fig, n=f"{len(data['rows'])} прогонов ({len(keys)} отношения × "
                 f"{len(data['seeds'])} сида)",
           unit=data["unit"],
           compares="доля расхождений против доли действий рефлекса, на тех же трёх "
                    "отношениях частот")
    return _save(fig, out / "32-konfabulyaciya-razoshlas.png",
                 "Ссылка переставлена — и конфабуляция перестала быть долей рефлекса.")


# --- B: связь находится, и на молчащем классе не находится ------------------


def fig_drives(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: детектор находит заведомую связь и молчит там, где нечего находить.

    Две панели, а не одна: «нашлось» и «не нашлось лишнего» — два разных числа, и
    складывать их на одну ось значило бы сравнивать находки с их отсутствием.
    """
    one, two = data["part_one"], data["part_two"]
    fig, (ax1, ax2) = _fig(620, ncols=2)

    sizes = sorted(int(k) for k in one["by_size"])
    found = [one["by_size"][str(s)]["true_found_median"] for s in sizes]
    # По ложным связям берётся **худший** прогон, а не медиана: медиана здесь равна нулю
    # на всех размерах и скрывает, что на 800 эпизодах один прогон из трёх дал три ложные
    # связи. Доля ложных, показанная медианой, — обещание, которого замер не давал.
    worst = [max(r["false"] for r in one["rows"] if r["episodes"] == s) for s in sizes]
    xs = list(range(len(sizes)))
    ax1.plot(xs, found, marker="o", markersize=9, color=NOW, linewidth=2.0,
             label="истинных связей найдено (из 2), медиана")
    ax1.plot(xs, worst, marker="s", markersize=7, color=MUTED, linewidth=1.5,
             label="ложных связей, худший прогон")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(s) for s in sizes])
    ax1.set_ylim(-0.3, 3.5)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.legend(fontsize=LABEL_PT - 3, frameon=False, loc="center right")
    ax1.set_xlabel(f"эпизодов в прогоне\nобе связи найдены с "
                   f"{one['episodes_to_find_all']} эпизодов;\n"
                   f"на 800 один прогон из трёх дал {max(worst)} ложные",
                   fontsize=LABEL_PT - 1)
    ax1.set_ylabel("связей", fontsize=LABEL_PT - 1)

    rows = [r for r in two["rows"] if r["gauge"]]
    labels, real, silent = [], [], []
    for r in rows:
        labels.append("по экранному слою" if r["by_screen_mask"] else "по всей ячейке")
        real.append(r["on_real_class"])
        silent.append(r["on_silent_class"])
    ys = list(range(len(rows)))
    ax2.barh([y + 0.18 for y in ys], real, height=0.32, color=NOW,
             label="на классе, который в мире действует")
    ax2.barh([y - 0.18 for y in ys], silent, height=0.32, color=MUTED,
             label="на молчащем классе — ложные по построению")
    for y, v in zip(ys, real):
        ax2.annotate(str(v), (v, y + 0.18), textcoords="offset points", xytext=(6, -4),
                     fontsize=LABEL_PT - 3, color=NEUTRAL)
    for y, v in zip(ys, silent):
        ax2.annotate(str(v), (v, y - 0.18), textcoords="offset points", xytext=(6, -4),
                     fontsize=LABEL_PT - 3, color=NEUTRAL)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(labels, fontsize=LABEL_PT - 2)
    ax2.set_xlim(0, max(real) + 1.6)
    ax2.legend(fontsize=LABEL_PT - 3, frameon=False, loc="lower right")
    ax2.set_xlabel("связей на настоящих пикселях\nразделение слоёв добавляет одну связь,\n"
                   "силу главной не меняет", fontsize=LABEL_PT - 1)

    _stamp(fig, n=f"{len(one['rows'])} прогонов механики, {len(rows)} прогона по пикселям",
           unit=one["unit_for_links"],
           compares="находки против пропусков и против ложных на контрольном классе")
    return _save(fig, out / "33-svyaz-nahoditsya.png",
                 "Связь находится с 200 эпизодов; ложная — один раз из двенадцати прогонов.")


# --- C: арбитраж платит ровно там, где бюджет тесен -------------------------


def fig_attention(out: Path, data: dict[str, Any]) -> Path:
    """Утверждение: выигрыш арбитража кончается вместе с дефицитом.

    По оси — расход окон до цели, меньше лучше. Показывать «снижение sigma на окно» здесь
    нельзя: на насыщённом прогоне эта величина от порядка раздачи не зависит вовсе
    (`MEASUREMENT.md`, 13.6), и картинка утверждала бы обратное тому, что в записи.
    """
    budgets = sorted({r["windows"] for r in data["rows"]})
    fig, (ax,) = _fig(620)

    arb = [data["by_case"][f"окон {w}, арбитраж"]["windows_to_target"] for w in budgets]
    ctl = [data["by_case"][f"окон {w}, по порядку"]["windows_to_target"] for w in budgets]
    xs = list(range(len(budgets)))
    ax.plot(xs, ctl, marker="s", markersize=9, color=BEFORE, linewidth=2.0,
            label="окна по порядку поступления (контроль)")
    ax.plot(xs, arb, marker="o", markersize=9, color=NOW, linewidth=2.0,
            label="арбитраж по пользе на стоимость")
    for x, w in zip(xs, budgets):
        v = data["verdicts"][f"окон {w}"]
        ratio = "" if not v["median_ratio"] else f"{v['median_ratio']:.2f}×"
        verdict = "дешевле" if v["fires"] else "не установлено"
        if v["wins"] == 0:
            verdict = "выигрыша нет"
        # Подпись у последнего бюджета уходит влево: там кривые сходятся, и подпись по
        # центру легла бы прямо на точку.
        last = x == len(budgets) - 1
        ax.annotate(f"{ratio}\n{verdict}\nвыигрышей {v['wins']} из {v['pairs']}",
                    (x, max(arb[x], ctl[x])), textcoords="offset points",
                    xytext=(-14 if last else 0, 46 if last else 24),
                    ha="right" if last else "center",
                    fontsize=LABEL_PT - 3, color=NEUTRAL)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{w} окно" if w == 1 else f"{w} окна" for w in budgets])
    ax.set_xlim(-0.4, len(budgets) - 0.6)
    ax.set_ylim(0, max(ctl) * 1.75)
    ax.legend(fontsize=LABEL_PT - 3, frameon=False, loc="lower right")
    ax.set_ylabel("окон потрачено до цели", fontsize=LABEL_PT - 1)
    fired = sum(v["fired"] for v in data["false_positives"].values())
    checks = sum(v["checks"] for v in data["false_positives"].values())
    ax.set_xlabel(
        f"бюджет окон на такт (attention_windows)\n"
        f"цель — {data['target_share']:.0%} доступного снижения sigma "
        f"({data['target']:.2f} из {data['ceiling']:.2f}), взята во всех прогонах.\n"
        f"Вывод знаковый по парным прогонам, ложных срабатываний {fired} из {checks}",
        fontsize=LABEL_PT)

    _stamp(fig, n=f"{len(data['rows'])} прогонов ({len(budgets)} бюджета × "
                 f"{len(data['seeds'])} сидов), плюс {len(data['null_rows'])} прогонов нуля",
           unit=data["unit"],
           compares="расход окон до цели у арбитража против раздачи по порядку, "
                    "на одном и том же потоке претензий")
    return _save(fig, out / "34-arbitrazh-platit-pri-deficite.png",
                 "Арбитраж вдвое дешевле при одном окне и не даёт ничего при четырёх.")


# --- D: час работы, по классам ожидания -------------------------------------


def fig_hour(out: Path, data: dict[str, Any], probe: dict[str, Any] | None = None
             ) -> Path:
    """Утверждение: цикл замедлился в девять раз, и виноват граф мест, а не память.

    Три панели, и третья — не «мест по времени», а **мест на наблюдение**. Разница
    принципиальна: по времени кривая мест выходит на полку и сверка объявляет «совпало», но
    выходит она потому, что замедлился сам цикл и наблюдений в секунду стало меньше. По
    наблюдениям видно, что насыщения нет вовсе. Ось времени здесь превращает находку в
    подтверждение (`MEASUREMENT.md`, 13.7).
    """
    minutes = [t / 60.0 for t in data["times"]]
    checks = data["checks"]
    series = data["series"]
    fig, (ax1, ax2, ax3) = _fig(720, ncols=3)

    # 1. Обороты в секунду: то, что разошлось с ожиданием.
    rate = series["loops_per_s"]
    ax1.plot(minutes[:len(rate)], rate, color=NOW, linewidth=2.0)
    ax1.set_ylabel("оборотов цикла в секунду", fontsize=LABEL_PT - 2)
    ax1.set_ylim(0, max(rate) * 1.1)
    ax1.set_xlabel(
        f"минут работы\nожидалось: не растёт\nвышло: падение "
        f"{rate[0] / rate[-1]:.1f}× ({rate[0]:.0f} → {rate[-1]:.0f})\nразошлось",
        fontsize=LABEL_PT - 3)

    # 2. Память: то, что совпало. Полка, а не утечка.
    rss = series["rss_mb"]
    ax2.plot(minutes[:len(rss)], rss, color=BEFORE, linewidth=2.0)
    ax2.set_ylabel("память процесса, МиБ", fontsize=LABEL_PT - 2)
    ax2.set_ylim(0, max(rss) * 1.15)
    ax2.set_xlabel(
        f"минут работы\nожидалось: полка\nвышло: полка "
        f"({rss[0]:.0f} → {rss[-1]:.0f} МиБ)\nсовпало",
        fontsize=LABEL_PT - 3)

    # 3. Мест на наблюдение: то, что сверка по времени объявила совпавшим напрасно.
    loops, places = series["loops"], series["places"]
    obs = [n / 100.0 for n in loops]           # отпечаток считается каждый сотый оборот
    per_obs = [(places[i] - places[i - 1]) / max(1e-9, obs[i] - obs[i - 1])
               for i in range(1, len(places))]
    ax3.plot(obs[1:], per_obs, color=NOW, linewidth=2.0)
    ax3.set_ylabel("новых мест на одно наблюдение", fontsize=LABEL_PT - 2)
    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel(
        f"наблюдений, тысяч\nожидалось: полка\n"
        f"вышло: {per_obs[0]:.2f} → {per_obs[-1]:.2f}, узлов {places[-1]:.0f}\n"
        f"насыщения нет; по времени — скрыто",
        fontsize=LABEL_PT - 3)
    ax3.set_xticks([5000, 15000, 25000])
    ax3.set_xticklabels(["5", "15", "25"])

    diverged = data["diverged"]
    cause = ""
    if probe:
        by = {r["case"]: r for r in probe["rows"]}
        as_is = by.get("как есть", {}).get("drop")
        without = by.get("без графа мест", {}).get("drop")
        if as_is and without:
            cause = (f"причина найдена выключением по одной части: без графа мест падения "
                     f"нет ({without:.2f}× против {as_is:.2f}×); журнал и обход каталога "
                     f"не при чём")
    fig.text(0.01, 0.105,
             f"классов ожидания разошлось {len(diverged)} из {len(checks)} — и это одно "
             f"явление, а не шесть: пять из них накопленные, они согнулись вслед за частотой",
             fontsize=LABEL_PT - 2, color=NEUTRAL)
    if cause:
        fig.text(0.01, 0.083, cause, fontsize=LABEL_PT - 2, color=NEUTRAL)
    _stamp(fig, n=f"{data['n']} отрезков по {data['segment_s']:g} с "
                 f"({data['seconds'] / 60:.0f} минут, прогон один)",
           unit=data["unit"],
           compares="наблюдённый класс роста против предрегистрированного ожидания")
    return _save(fig, out / "35-chas-raboty.png",
                 "Час работы: цикл замедлился в девять раз, и это граф мест, а не утечка памяти.")


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else ROOT / "docs" / "figures"
    made: list[Path] = []
    m = ROOT / "docs" / "measurements"
    plan = ((fig_confab, "live_cycle.json", "tools/measure_live_cycle.py"),
            (fig_drives, "drives.json", "tools/measure_drives.py"),
            (fig_attention, "attention.json", "tools/measure_attention.py"),
            (fig_hour, "hour.json", "tools/measure_hour.py --seconds 3600"))
    probe_path = m / "slowdown.json"
    probe = (json.loads(probe_path.read_text(encoding="utf-8"))
             if probe_path.exists() else None)
    missing: list[str] = []
    for draw, name, how in plan:
        src = m / name
        if not src.exists():
            missing.append(f"{name} — сначала python3 {how}")
            continue
        payload = json.loads(src.read_text(encoding="utf-8"))
        made.append(draw(base, payload, probe) if draw is fig_hour
                    else draw(base, payload))
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
