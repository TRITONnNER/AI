#!/usr/bin/env python3
"""Картинки для отчёта: что видит агент, что он из этого выводит и что открыл.

Всё рисуется по настоящим прогонам, а не по придуманным числам. Каждая картинка
подписана тем, откуда взяты данные, чтобы её нельзя было принять за иллюстрацию.

    python3 tools/make_figures.py [каталог]
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness.behaviour.babbling import Babbler, run_babbling      # noqa: E402
from harness.core.action import Action                            # noqa: E402
from harness.core.clocks import Clocks                            # noqa: E402
from harness.core.profile import BABBLE, MILESTONE_0              # noqa: E402
from harness.corpus.synthetic import generate_session, load_hud_mask  # noqa: E402
from harness.corpus.world import Effect, InteractiveWorld         # noqa: E402
from harness.model.places import PlaceGraph, fingerprint          # noqa: E402
from harness.session import Session                               # noqa: E402
from harness.vision.predict import CopyPredictor, PredictionError, ShiftPredictor  # noqa: E402
from harness.vision.selfworld import SCREEN, SelfWorldSeparator, UNDECIDED, WORLD  # noqa: E402

plt.rcParams.update({
    "figure.facecolor": "#14181d", "axes.facecolor": "#1b2027",
    "text.color": "#e6e9ee", "axes.labelcolor": "#e6e9ee",
    "xtick.color": "#9aa4b2", "ytick.color": "#9aa4b2",
    "axes.edgecolor": "#39424f", "grid.color": "#2b3340",
    "font.size": 9, "axes.titlesize": 10, "figure.dpi": 130,
})
AGENT = "#4fc3d9"
HUMAN = "#e0a94f"
ALARM = "#e05a4a"
PROOF = "#5fc98a"


def _out(base: Path, name: str, fig) -> Path:
    path = base / name
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("нарисовано:", path)
    return path


# --- 1. Что видит агент и как мир отвечает ---------------------------------


def fig_world(base: Path) -> Path:
    world = InteractiveWorld(MILESTONE_0, seed=11)
    fwd = [o for o, e in world._effects.items() if e is Effect.FORWARD][0]
    light = [o for o, e in world._effects.items() if e is Effect.TOGGLE_LIGHT][0]
    brk = [o for o, e in world._effects.items() if e is Effect.BREAK_PANEL][0]

    shots = []
    shots.append(("кадр как он есть\nинтерфейс внизу и справа", world.step(None, with_audio=False).frame))
    for _ in range(6):
        world.step(Action.key(fwd, 400), with_audio=False)
    shots.append(("после движения вперёд\nмир уехал, интерфейс на месте",
                  world.step(None, with_audio=False).frame))
    world.step(Action.key(light, 100), with_audio=False)
    before_break = world.step(None, with_audio=False).frame
    shots.append(("свет выключен\nмир потемнел, интерфейс нет", before_break))
    world.step(Action.key(brk, 100), with_audio=False)
    after_break = world.step(None, with_audio=False).frame

    # Обводка того, что исчезло. Без неё разница между двумя последними кадрами
    # почти не видна глазом, и подпись обещала бы больше, чем показывает картинка.
    #
    # Прямоугольник берётся из истины мира, а не из разницы пикселей: разница
    # включает шевеление анимированных полос интерфейса и потому охватывает почти
    # весь кадр. Это картинка для исследователя, ему истина доступна — агенту нет.
    broken = np.dstack([after_break] * 3)
    if world.state.broken:
        rect = world.scene.hud[sorted(world.state.broken)[0]]
        pad = 2
        y0 = max(0, rect.top - pad)
        x0 = max(0, rect.left - pad)
        y1 = min(broken.shape[0] - 1, rect.top + rect.height + pad)
        x1 = min(broken.shape[1] - 1, rect.left + rect.width + pad)
        for yy in (y0, y1):
            broken[yy, x0:x1 + 1] = [224, 90, 74]
        for xx in (x0, x1):
            broken[y0:y1 + 1, xx] = [224, 90, 74]
    shots.append(("панель сломана — необратимо\nвернуть её нельзя ничем", broken))

    fig, axes = plt.subplots(1, 4, figsize=(13, 2.6))
    for ax, (title, img) in zip(axes, shots):
        if img.ndim == 3:
            ax.imshow(img, interpolation="nearest")
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Интерактивный мир: единственное, что агент получает — эти пиксели. "
                 "Ни координат, ни имён, ни правил.", fontsize=10, y=1.12)
    return _out(base, "1-mir.png", fig)


# --- 2. Разделение себя и мира ---------------------------------------------


def fig_selfworld(base: Path, session_root: Path) -> Path:
    with Session.open(session_root) as s:
        sep = SelfWorldSeparator(s.profile)
        frames = []
        for _, img in s:
            sep.feed(img)
            frames.append(img)
        res = sep.result()
    truth = load_hud_mask(session_root)
    found = res.pixel_mask(SCREEN)

    inter = int((found & truth).sum())
    union = int((found | truth).sum())
    recall = inter / int(truth.sum())
    precision = inter / max(1, int(found.sum()))

    overlay = np.zeros((*truth.shape, 3), dtype=np.uint8)
    overlay[..., :] = np.dstack([frames[len(frames) // 2]] * 3) // 3
    overlay[found & truth] = [95, 201, 138]       # верно найденный интерфейс
    overlay[found & ~truth] = [224, 90, 74]       # ложно принято за интерфейс
    overlay[~found & truth] = [224, 169, 79]      # пропущено

    labels = np.zeros((*res.labels.shape, 3), dtype=np.uint8)
    labels[res.labels == SCREEN] = [79, 195, 217]
    labels[res.labels == WORLD] = [60, 70, 85]
    labels[res.labels == UNDECIDED] = [130, 130, 130]

    fig, axes = plt.subplots(1, 4, figsize=(13, 2.7))
    axes[0].imshow(frames[len(frames) // 2], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("кадр из записи", fontsize=8.5)
    axes[1].imshow(labels, interpolation="nearest")
    axes[1].set_title("что решил алгоритм\nголубое — экранный слой", fontsize=8.5)
    axes[2].imshow(truth, cmap="gray", interpolation="nearest")
    axes[2].set_title("истина из отладочного потока\n(алгоритм её не видел)", fontsize=8.5)
    axes[3].imshow(overlay, interpolation="nearest")
    axes[3].set_title(f"сверка: IoU {inter / union:.2f}\nполнота {recall:.2f}, "
                      f"точность {precision:.2f}", fontsize=8.5)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("0.6 — отделить «моё» от «мирского» по параллаксу. "
                 "Ни одной захардкоженной координаты, ни одного знания об игре.",
                 fontsize=10, y=1.1)
    fig.text(0.5, -0.06,
             f"зелёное — интерфейс найден верно ({inter} px) · красное — принято за "
             f"интерфейс напрасно ({int((found & ~truth).sum())} px) · "
             f"жёлтое — пропущено ({int(((~found) & truth).sum())} px). "
             "Ореол вокруг панелей ушёл, когда к параллаксу добавилось условие "
             "статичности экранного слоя.",
             ha="center", fontsize=8, color="#9aa4b2")
    return _out(base, "2-sebya-i-mira.png", fig)


# --- 3. Ошибка предсказания ------------------------------------------------


def fig_prediction(base: Path) -> Path:
    world = InteractiveWorld(MILESTONE_0, seed=5)
    fwd = [o for o, e in world._effects.items() if e is Effect.FORWARD][0]
    brk = [o for o, e in world._effects.items() if e is Effect.BREAK_PANEL][0]

    shift = PredictionError(MILESTONE_0, predictor=ShiftPredictor())
    copy = PredictionError(MILESTONE_0, predictor=CopyPredictor())
    vals_s, vals_c, acting, spikes = [], [], [], []
    for i in range(160):
        act = Action.key(fwd, 300) if (i // 12) % 2 == 0 else None
        if i == 100:
            act = Action.key(brk, 100)
        frame = world.step(act, with_audio=False).frame
        a = shift.feed(frame)
        b = copy.feed(frame)
        if a is not None:
            vals_s.append(a.value)
            vals_c.append(b.value if b else 0.0)
            acting.append(act is not None)
            if a.is_spike:
                spikes.append(len(vals_s) - 1)

    fig, ax = plt.subplots(figsize=(12, 3.4))
    x = np.arange(len(vals_s))
    inside = np.array(acting)
    ax.fill_between(x, 0, max(vals_c) * 1.05, where=inside, color=HUMAN, alpha=0.10,
                    step="mid", label="агент действует")
    ax.plot(x, vals_c, color="#6b7787", lw=1.0, label="предсказание «будет то же» (опорный уровень)")
    ax.plot(x, vals_s, color=AGENT, lw=1.4, label="предсказание постоянной скоростью")
    ax.scatter(spikes, [vals_s[i] for i in spikes], s=26, color=ALARM, zorder=5,
               label=f"всплеск: мир повёл себя не так, как я думал ({len(spikes)})")
    ax.axvline(99, color=PROOF, ls="--", lw=1.0)
    ax.annotate("здесь агент необратимо\nсломал панель", (99, max(vals_s) * 0.8),
                xytext=(108, max(vals_s) * 0.95), fontsize=8, color=PROOF,
                arrowprops=dict(color=PROOF, arrowstyle="->", lw=0.8))
    ax.set_xlabel("цикл восприятия агента (t_self)")
    ax.set_ylabel("ошибка предсказания, доля")
    ax.set_title("Ошибка предсказания — единая валюта: она же обновляет карту, "
                 "направляет внимание и решает, что запоминать")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.15, loc="upper left")
    fig.text(0.5, -0.10, f"всплески приходятся на смену режима движения — именно то, "
             f"что должно удивлять предсказатель постоянной скорости. "
             f"Постоянная скорость точнее копии в "
             f"{np.mean(vals_c) / max(1e-9, np.mean(vals_s)):.1f} раза.",
             ha="center", fontsize=8, color="#9aa4b2")
    return _out(base, "3-oshibka-predskazaniya.png", fig)


# --- 4. Лепет: как открывается тело ---------------------------------------


def fig_babbling(base: Path) -> Path:
    world = InteractiveWorld(BABBLE, seed=11)
    babbler = Babbler(BABBLE, world.outputs, rng_seed=1)

    history = []
    clocks = Clocks()
    for _ in range(60):
        run_babbling(world, babbler, steps=25, clocks=clocks)
        st = babbler.progress()
        history.append((st["probes_done"], st["live"], st["silent"],
                        len(babbler.body.undoable())))

    truth = world.truth()
    live_true = set(truth["live_outputs"])
    irr_true = set(truth["irreversible_outputs"])

    fig = plt.figure(figsize=(13, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])

    ax = fig.add_subplot(gs[0, 0])
    probes = [h[0] for h in history]
    ax.plot(probes, [h[1] for h in history], color=AGENT, lw=1.6, label="признал живыми")
    ax.plot(probes, [h[2] for h in history], color="#6b7787", lw=1.4, label="признал молчащими")
    ax.plot(probes, [h[3] for h in history], color=PROOF, lw=1.6,
            label="нашёл способ откатить")
    ax.axhline(len(live_true), color=AGENT, ls=":", lw=1.0)
    ax.axhline(len(truth["silent_outputs"]), color="#6b7787", ls=":", lw=1.0)
    ax.axhline(8, color=PROOF, ls=":", lw=1.0)
    ax.set_xlabel("сделано проб")
    ax.set_ylabel("выходов")
    ax.set_title("Лепет: тело открывается с нуля\nпунктир — истина, которой агент не видел")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.15, loc="center right")

    ax2 = fig.add_subplot(gs[0, 1])
    outs = sorted(world.outputs)
    states = []
    colors = []
    for o in outs:
        f = babbler.body.outputs.get(o)
        state = f.state if f else "untried"
        undoable = f.undo_method is not None if f else False
        states.append(state)
        if state == "live" and undoable:
            colors.append(PROOF)
        elif state == "live":
            colors.append(ALARM)
        elif state == "silent":
            colors.append("#4a5361")
        else:
            colors.append("#2b3340")
    grid = np.arange(len(outs))
    ax2.barh(grid, [1] * len(outs), color=colors, height=0.78)
    ax2.set_yticks(grid)
    ax2.set_yticklabels(
        [f"{o[-4:]}  {'· необратим' if o in irr_true else ''}" for o in outs],
        fontsize=7, family="monospace")
    ax2.set_xticks([])
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    ax2.set_title("Карта тела после лепета\n"
                  "зелёный — умею откатить · красный — не умею · серый — молчит")
    for spine in ax2.spines.values():
        spine.set_visible(False)

    fig.text(0.5, -0.04,
             f"итог: живых {len(set(babbler.body.by_state('live')) & live_true)} из "
             f"{len(live_true)} верно, ложных "
             f"{len(set(babbler.body.by_state('live')) - live_true)} · "
             f"обратных пар {len(babbler.body.undoable())} из 8 · "
             f"необратимый выход распознан и остался единственным красным",
             ha="center", fontsize=8.5, color="#9aa4b2")
    return _out(base, "4-lepet.png", fig)


# --- 5. Граф мест -----------------------------------------------------------


def fig_places(base: Path) -> Path:
    world = InteractiveWorld(MILESTONE_0, seed=7)
    fwd = [o for o, e in world._effects.items() if e is Effect.FORWARD][0]
    back = [o for o, e in world._effects.items() if e is Effect.BACK][0]
    graph = PlaceGraph()

    plan = [fwd] * 8 + [back] * 8 + [fwd] * 8 + [back] * 8 + [fwd] * 4 + [back] * 4
    for i, out in enumerate(plan * 3):
        frame = world.step(Action.key(out, 400), with_audio=False).frame
        graph.observe(fingerprint(frame), i, seconds_per_seq=1 / 30,
                      mode="вперёд" if out == fwd else "назад")

    places = list(graph)
    rng = np.random.default_rng(0)
    pos = {p.id: rng.normal(size=2) * 3 for p in places}
    # Раскладка силами: положение узлов произвольно, и это принципиально —
    # в графе мест нет геометрии, только время прохода.
    for _ in range(220):
        for e in graph.edges.values():
            if e.src not in pos or e.dst not in pos:
                continue
            d = pos[e.dst] - pos[e.src]
            dist = max(1e-6, float(np.hypot(*d)))
            want = 0.6 + e.mu_seconds * 6.0
            pull = (dist - want) / dist * 0.08
            pos[e.src] = pos[e.src] + d * pull
            pos[e.dst] = pos[e.dst] - d * pull
        ids = list(pos)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                d = pos[b] - pos[a]
                dist = max(1e-6, float(np.hypot(*d)))
                if dist < 1.4:
                    push = (1.4 - dist) / dist * 0.05
                    pos[a] = pos[a] - d * push
                    pos[b] = pos[b] + d * push

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for e in graph.edges.values():
        if e.src not in pos or e.dst not in pos:
            continue
        x = [pos[e.src][0], pos[e.dst][0]]
        y = [pos[e.src][1], pos[e.dst][1]]
        ax.plot(x, y, color=AGENT, alpha=0.25 + 0.6 * e.confidence,
                lw=0.6 + 2.4 * e.confidence, zorder=1)
    visits = np.array([p.visits for p in places], dtype=float)
    ax.scatter([pos[p.id][0] for p in places], [pos[p.id][1] for p in places],
               s=40 + 220 * visits / visits.max(), color="#2b3340",
               edgecolors=AGENT, linewidths=1.2, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    st = graph.stats()
    ax.set_title(f"Граф мест: {st['places']} узлов, {st['edges']} рёбер\n"
                 "узел — отпечаток вида, ребро — время прохода с разбросом")
    fig.text(0.5, 0.02, "Это не карта. Положение узлов произвольно: в графе мест нет "
             "координат вообще —\nтолько то, как место выглядит и сколько секунд до "
             "соседнего. Толщина ребра — уверенность.",
             ha="center", fontsize=8, color="#9aa4b2")
    return _out(base, "5-graf-mest.png", fig)


# --- 6. Замкнутый круг: драйв → цель → пробы → тест ------------------------


def fig_loop(base: Path) -> Path:
    from harness.behaviour.goals import GoalStack, candidates_from_body, choose
    from harness.core.profile import from_schema
    from harness.model.drives import Motivation

    profile = from_schema("КРУГ-1", capture_width=320, capture_height=180,
                          babble_rate=0.9, babble_repeats=3, drive_horizon_s=60.0)
    world = InteractiveWorld(profile, seed=23)
    caution = float(profile.parameters["irreversibility_threshold"])
    babbler = Babbler(profile, world.outputs, rng_seed=23)
    motivation = Motivation(profile)
    stack = GoalStack(profile)
    error = PredictionError(profile)

    rounds, passed, abandoned, pressure_hist, live_hist, goal_marks = [], [], [], [], [], []
    clocks = Clocks()
    for rnd in range(40):
        s = error.summary()
        motivation.update(error_mean=s["mean"], error_sigma=s["sigma"],
                          error_now=s["last"],
                          unknown_reversibility=babbler.progress()["unknown_reversibility"])
        if stack.active is None:
            cands = candidates_from_body(babbler.body, world.outputs,
                                        caution_threshold=caution)
            if cands:
                stack.push(choose(cands, motivation, top=1)[0], motivation, rnd, "b0",
                           budget_ticks=10)
                goal_marks.append(rnd)
        run_babbling(world, babbler, steps=40, clocks=clocks, error=error)
        stack.tick()
        st = stack.stats()
        rounds.append(rnd)
        passed.append(st["passed"])
        abandoned.append(st["abandoned"])
        pressure_hist.append(max(motivation.goal_pressure().values()))
        live_hist.append(babbler.progress()["live"])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True,
                                  gridspec_kw={"height_ratios": [1.25, 1]})
    for r in goal_marks:
        ax.axvline(r, color="#39424f", lw=0.8, zorder=0)
    ax.plot(rounds, passed, color=PROOF, lw=1.8, label="целей прошло тест")
    ax.plot(rounds, abandoned, color=ALARM, lw=1.8, label="целей брошено по бюджету")
    ax.plot(rounds, live_hist, color=AGENT, lw=1.3, ls="--",
            label="отвечающих выходов найдено")
    ax.set_ylabel("штук")
    ax.set_title("Замкнутый круг: давление драйва ставит цель, пробы её закрывают. "
                 "Серые линии — момент постановки цели.")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.15, loc="center right")

    ax2.plot(rounds, pressure_hist, color=HUMAN, lw=1.6)
    ax2.fill_between(rounds, 0, pressure_hist, color=HUMAN, alpha=0.12)
    ax2.set_ylabel("давление\nведущего драйва")
    ax2.set_xlabel("раунд (40 проб в каждом)")
    ax2.grid(alpha=0.25)

    st = stack.stats()
    fig.text(0.5, -0.05,
             f"итог: целей {st['goals']}, прошло {st['passed']}, брошено "
             f"{st['abandoned']} · до успеха в среднем {st['mean_ticks_to_pass']} такта, "
             f"до отказа {st['mean_ticks_to_abandon']} · награды в этой схеме нет: "
             "цель либо проходит свой тест, либо нет",
             ha="center", fontsize=8.5, color="#9aa4b2")
    return _out(base, "6-zamknutyy-krug.png", fig)


# --- 8. Четыре домена и два признака разделения слоёв -----------------------


def fig_domains(base: Path) -> Path:
    """Один и тот же код по четырём разным мирам, и чем именно он там справился."""
    from harness.benchmark import bench_domain
    from harness.core.action import Action as _Action
    from harness.core.profile import from_schema
    from harness.corpus.domains import make_domain
    from harness.vision.selfworld import (COUPLING, LayerArbiter, MERGED, PARALLAX,
                                          SCREEN as _SCREEN, STILLNESS)

    names = ("game", "document", "desktop", "video")
    titles = {"game": "игра: камера панорамирует",
              "document": "документ: прокрутка по вертикали",
              "desktop": "рабочий стол: едут отдельные окна",
              "video": "видео: содержимое идёт само"}
    profile = from_schema("ФИГУРА-домены", capture_width=320, capture_height=180)

    fig, axes = plt.subplots(3, 4, figsize=(13.5, 7.6),
                             gridspec_kw={"height_ratios": [1.0, 1.0, 0.75],
                                          "wspace": 0.45, "hspace": 0.3})
    results = []
    for col, name in enumerate(names):
        dom = make_domain(name, profile, seed=3)
        arb = LayerArbiter(profile)
        rng = np.random.default_rng(103)
        frame = None
        for i in range(120):
            act = None
            if i % 3 != 0:
                dx = int(rng.integers(-40, 41))
                dy = int(rng.integers(-26, 27))
                if dx or dy:
                    act = _Action.mouse(dx, dy, duration_ms=33)
            frame = dom.step(act, with_audio=False).frame
            arb.feed(frame)
        verdict = arb.result()
        res = bench_domain(name, seed=3, frames=120, babble_steps=500)
        results.append(res)

        ax = axes[0][col]
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
        ax.set_title(titles[name], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        # Что нашёл выбранный признак, поверх истины.
        ax = axes[1][col]
        truth = dom.screen_mask()
        found = verdict.pixel_mask(_SCREEN)
        # Кадр остаётся видимым под разметкой: без него панель из одних цветных
        # пятен не даёт понять, о каком месте кадра идёт речь.
        rgb = np.dstack([frame.astype(np.float32) / 255.0] * 3) * 0.35
        rgb[truth & found] = (0.37, 0.79, 0.54)      # верно найдено
        rgb[truth & ~found] = (0.55, 0.60, 0.68)     # пропущено
        rgb[~truth & found] = (0.88, 0.35, 0.29)     # ложно
        ax.imshow(np.clip(rgb, 0, 1))
        sig = {PARALLAX: "параллакс", STILLNESS: "неподвижность",
               COUPLING: "связь с движением", MERGED: "три признака вместе",
               "none": "все признаки промолчали"}[verdict.signal]
        ax.set_title(f"{sig}\nIoU {res.iou if res.iou is None else round(res.iou, 2)}, "
                     f"точность {res.precision if res.precision is None else round(res.precision, 2)}",
                     fontsize=8.5,
                     color=AGENT if verdict.signal in (MERGED, COUPLING) else HUMAN)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[2][col]
        vals = [res.coupling_decided, res.parallax_decided, res.stillness_decided]
        ax.barh([2, 1, 0], vals, color=[PROOF, AGENT, HUMAN], height=0.6)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["связь", "параллакс", "неподвижн."], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("доля решённых пикселей", fontsize=8)
        for i, v in enumerate(vals):
            inside = v > 0.2
            ax.text(v - 0.04 if inside else v + 0.04, len(vals) - 1 - i, f"{v:.0%}",
                    va="center", ha="right" if inside else "left", fontsize=7.5,
                    color="#14181d" if inside else "#9aa4b2")
        ax.grid(axis="x", alpha=0.2)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Один и тот же код по четырём мирам. Зелёное — обрамление найдено, "
                 "светло-серое — пропущено, красное — ложно принято за обрамление.",
                 fontsize=10.5, y=0.98)
    body = "; ".join(f"{r.domain} {r.live_found}/{r.live_true} живых" for r in results)
    fig.text(0.5, -0.02,
             "Нижний ряд: сколько смог решить каждый из трёх признаков по отдельности. "
             "Где нет глобального сдвига, первые два молчат — отвечает неподвижность.\n"
             f"Тело по доменам: {body}. Ответ собирается из трёх признаков с "
             "происхождением у каждого пикселя; расхождения считаются отдельно и не "
             "усредняются.",
             ha="center", fontsize=8.5, color="#9aa4b2")
    return _out(base, "8-chetyre-domena.png", fig)


# --- 9. Планировщик: от лепета до дошедшего плана ---------------------------


def fig_planner(base: Path) -> Path:
    """Планировщик: от чего зависит, дойдёт план или нет. Всё из прогона."""
    import collections

    from harness.behaviour.babbling import Babbler as _Babbler, run_babbling as _run
    from harness.behaviour.goals import Goal
    from harness.behaviour.planner import Planner, choose_probe, execute
    from harness.core.action import Action as _Action, action_key
    from harness.core.profile import from_schema
    from harness.model.beliefs import Origin, Provenance
    from harness.model.forward import ForwardModel
    from harness.model.places import PlaceGraph

    hold = 200

    def babble(seed):
        """Лепет своим сидом на каждый мир.

        Раньше карта тела и обратные пары брались из одного прогона на сиде 5 и
        переиспользовались для остальных миров. Раскладка «выход → эффект»
        выводится из сида, поэтому чужая карта тела — это чужая обратимость и
        чужие обратные пары: на сиде 11 из-за этого не находилось ни одного плана,
        и картинка показывала провал там, где его нет.
        """
        profile = from_schema("ФИГУРА-план", capture_width=320, capture_height=180,
                              babble_repeats=3)
        world = InteractiveWorld(profile, seed=seed, n_outputs=16)
        b = _Babbler(profile, world.outputs, rng_seed=seed)
        _run(world, b, steps=1200, clocks=Clocks())
        return b.body, dict(b.inverse_found)

    plans: list[tuple[int, int, bool]] = []      # (слабейший шаг, длина, дошёл)
    summary: dict[str, dict[str, int]] = {}
    long_plans: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])

    def trial(name, *, use_inverse, min_n, seed=5, steps=1500, goals=40,
              random_walk=False, loops=True, refine=True, collect=True,
              max_depth=6):
        body, inverse = babble(seed)
        profile = from_schema("ФИГУРА-план", capture_width=320, capture_height=180,
                              babble_repeats=3, plan_min_step_n=min_n,
                              place_record_loops=loops)
        world = InteractiveWorld(profile, seed=seed, n_outputs=16)
        graph = PlaceGraph.from_profile(profile)
        rng = np.random.default_rng(seed)
        st = {"seq": 0, "last": None}

        def step(out, ms=hold):
            obs = world.step(_Action.key(out, ms), with_audio=False)
            st["seq"] += 1
            st["last"] = out
            return graph.see(obs.frame, st["seq"], seconds_per_seq=1 / 30.0,
                             mode=action_key(out, ms))

        def probe(model, here):
            if random_walk:
                return world.outputs[int(rng.integers(len(world.outputs)))], hold
            return choose_probe(model, here, world.outputs, hold_ms=hold,
                                min_n=min_n,
                                inverse=inverse if use_inverse else None,
                                last_output=st["last"])

        graph.see(world.step(None, with_audio=False).frame, 0,
                  seconds_per_seq=1 / 30.0, mode="start")
        model = ForwardModel.from_graph(graph, body)
        refinements: list[dict] = []
        for i in range(steps):
            if i % 25 == 0:
                # Уточнение карты — часть разведки: место, склеившее два состояния
                # мира, делится по расхождению своих же предсказаний.
                if refine:
                    refinements += graph.refine()
                model = ForwardModel.from_graph(graph, body)
            out, ms = probe(model, graph.current)
            step(out, ms)

        def reach(model, src):
            depth = {src: 0}
            frontier = [src]
            for d in range(max_depth):
                nxt = []
                for node in frontier:
                    for key in model.actions_from(node):
                        pred = model.predict(node, key)
                        if pred is None:
                            continue
                        for outcome, p in pred.outcomes():
                            if p < 0.5 or outcome.n < min_n:
                                continue
                            if outcome.dst == node:
                                continue      # петля никуда не ведёт
                            if outcome.dst not in depth:
                                depth[outcome.dst] = d + 1
                                nxt.append(outcome.dst)
                frontier = nxt
            return {k: v for k, v in depth.items() if v > 0}

        found = arrived = 0
        for i in range(goals):
            if refine:
                refinements += graph.refine()
            model = ForwardModel.from_graph(graph, body)
            here = graph.current
            targets = reach(model, here or "")
            if not targets:
                out, ms = probe(model, here)
                step(out, ms)
                continue
            target = sorted(targets, key=lambda k: (-targets[k], k))[0]
            goal = Goal(id=f"g{i}", kind="reach_place", target=target,
                        test=lambda t=target: graph.current == t,
                        test_text="я в этом месте", budget_ticks=40,
                        provenance=Provenance(Origin.EXPERIENCE, branch="fig", seq=0),
                        drive="curiosity", pressure=0.5)
            plan = Planner(profile, model).plan(goal, here, target)
            if plan is None:
                continue
            found += 1
            ex = execute(plan,
                         act=lambda a: step(a.outputs_touched()[0], a.duration_ms),
                         goal=goal)
            arrived += ex.goal_passed
            if collect:
                plans.append((plan.min_step_n, plan.length, bool(ex.goal_passed)))
            long_plans[name][0] += plan.length >= 3
            long_plans[name][1] += plan.length >= 3 and bool(ex.goal_passed)
        confirmed = sum(1 for outs in ForwardModel.from_graph(graph, body)
                        .transitions.values() for o in outs if o.n >= 2)
        gs = graph.stats()
        acc = summary.setdefault(name, {"places": 0, "confirmed": 0, "found": 0,
                                        "arrived": 0, "splits": 0, "loops": 0,
                                        "leaving": 0})
        acc["places"] += len(graph)
        acc["confirmed"] += confirmed
        acc["found"] += found
        acc["arrived"] += arrived
        acc["splits"] += len(refinements)
        acc["loops"] += gs["loops"]
        acc["leaving"] += gs["leaving"]

    # Рабочая настройка на трёх сидах. Сравнение с отключёнными петлями и без
    # деления карты здесь не рисуется намеренно: на двух-трёх сидах разница между
    # ступенями тонет в разбросе, и картинка соврала бы. Это сравнение измерено на
    # восьми сидах и лежит в docs/ARCHITECTURE-AGENT.md, «Карта, которая исправляет
    # себя»: 192 из 235 без деления против 224 из 239 с ним.
    for seed in (5, 11, 13):
        trial(f"сид {seed}", use_inverse=True, min_n=2, seed=seed)

    by_n: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    by_len: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    for weak, length, ok in plans:
        k = 1 if weak < 2 else (2 if weak < 5 else (5 if weak < 15 else 15))
        by_n[k][0] += 1
        by_n[k][1] += ok
        j = min(length, 3)
        by_len[j][0] += 1
        by_len[j][1] += ok

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8),
                             gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    names = list(summary)
    x = np.arange(len(names))
    ax.bar(x - 0.2, [summary[n]["leaving"] for n in names], width=0.4,
           color=PROOF, label="рёбер, ведущих куда-то")
    ax.bar(x + 0.2, [summary[n]["loops"] for n in names], width=0.4,
           color=AGENT, label="петель «нажал и остался»")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7.5)
    for i, n in enumerate(names):
        ax.text(i - 0.2, summary[n]["leaving"] + 0.4,
                str(summary[n]["leaving"]), ha="center", fontsize=8)
        ax.text(i + 0.2, summary[n]["loops"] + 0.4,
                str(summary[n]["loops"]), ha="center", fontsize=8)
    ax.set_title("Из чего состоит карта", fontsize=9)
    ax.legend(fontsize=7.5, facecolor="#1b2027", edgecolor="#39424f",
              loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.2)

    labels_n = {1: "1 раз", 2: "2–4", 5: "5–14", 15: "15+"}
    ax = axes[1]
    keys = sorted(by_n)
    rates = [by_n[k][1] / by_n[k][0] for k in keys]
    colors = [ALARM if r < 0.9 else PROOF for r in rates]
    ax.bar([labels_n[k] for k in keys], rates, color=colors, width=0.55)
    for i, k in enumerate(keys):
        ax.text(i, rates[i] + 0.03, f"{by_n[k][1]}/{by_n[k][0]}", ha="center",
                fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("доля дошедших")
    ax.set_xlabel("слабейший шаг наблюдён")
    ax.set_title("Шаг, увиденный один раз, врёт;\nподтверждённый — нет", fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[2]
    keys = sorted(by_len)
    rates = [by_len[k][1] / by_len[k][0] for k in keys]
    colors = [ALARM if r < 0.9 else PROOF for r in rates]
    ax.bar([f"{k}{'+' if k == 3 else ''}" for k in keys], rates, color=colors,
           width=0.55)
    for i, k in enumerate(keys):
        ax.text(i, rates[i] + 0.03, f"{by_len[k][1]}/{by_len[k][0]}", ha="center",
                fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("доля дошедших")
    ax.set_xlabel("шагов в плане")
    ax.set_title("Длина плана: провал остался там,\nгде карта ещё склеена", fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle("Планировщик: цель «добраться до места» закрывается цепочкой, "
                 "выбранной по модели из журнала.", fontsize=10.5, y=1.05)
    fig.text(0.5, -0.2,
             "Модель перехода строится только из наблюдённых «(место, действие) → "
             "место», и «не знаю» для неё — законный ответ. Слева: «нажал и остался» "
             "— тоже ребро;\nбез этих петель модель не знает про бесполезное "
             "действие ничего, и разведка запирается на одном нажатии. В середине: "
             "порог «наблюдено дважды»\nвзят не из осторожности, а отсюда. Справа: "
             "длинные планы доходят после того, как место научилось делиться по "
             "расхождению своих же предсказаний;\nостаток провалов приходится на "
             "планы в два шага — именно такой длины они и получаются там, где карта "
             "всё ещё склеивает два состояния мира.",
             ha="center", fontsize=8.5, color="#9aa4b2")
    return _out(base, "9-planirovshchik.png", fig)


# --- 7. Что сделано и что нет ----------------------------------------------


def fig_status(base: Path) -> Path:
    """Что сделано и что нет — из того же источника, что docs/STATUS.md.

    Раньше здесь лежала рукописная таблица литералов, и это дало два расхождения,
    найденные аудитом (`AUDIT.md`, часть 4): полоса «Планировщик по модели» была
    зелёной с подписью «40 из 40 планов доходят», тогда как детальная фигура в этом
    же файле показывала 0/3 на трёхшаговых планах; а «6 из 8 целей прошли» считалось
    по закрытым целям, тогда как деталь говорила про девять всего. Плюс три числа
    устарели молча: настроек стало 140 вместо 104, а IoU 0.97 после пересчёта по
    независимым единицам оказался характеристикой одного генератора (n=1), а не
    метода: по пяти доменам медиана 0.667. См. MEASUREMENT.md.

    Причина у всех пяти одна: сводка была прозой, а деталь — замером. Теперь сводка
    берётся из `tools/project_status.py`, то есть из кода и тестов, и **не может
    позеленеть поверх заявленного отсутствующего**: строка с непустым `absent`
    ограничена сверху.
    """
    import importlib.util
    import sys as _sys

    tools = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("project_status",
                                                tools / "project_status.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("нет tools/project_status.py: сводку считать нечем")
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["project_status"] = mod
    spec.loader.exec_module(mod)
    data = mod.collect(with_tests=False)

    rows = []
    for part in data["parts"]:
        measured, absent = part["measured"], part["absent"]
        if measured and not absent:
            value = 1.0
        elif measured and absent:
            # Замер есть, но что-то заявлено отсутствующим. Зеленеть нельзя.
            value = 0.6
        elif absent:
            value = 0.25
        else:
            value = 0.0
        note = measured or absent or "замера нет"
        if measured and absent:
            note = f"{measured}; но: {absent}"
        rows.append((part["name"], value, note[:96]))
    for item in data["missing"]:
        rows.append((item["what"], 0.0, item["why"][:96]))

    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    notes = [r[2] for r in rows]
    colors = [PROOF if v >= 0.99 else (HUMAN if v >= 0.5 else ALARM) for v in vals]

    fig, ax = plt.subplots(figsize=(12.5, 0.34 * len(rows) + 1.6))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.55)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["нет", "", "половина", "", "готово и проверено"], fontsize=8)
    for i, (v, note) in enumerate(zip(vals, notes)):
        ax.text(v + 0.03, i, note, va="center", fontsize=7.5, color="#9aa4b2")
    ax.grid(axis="x", alpha=0.2)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Что сделано и что нет. Собрано из кода и тестов, а не написано "
                 "руками.", fontsize=10.5, pad=14)
    fig.text(0.5, -0.02,
             "Зелёное — замер есть и ничего не заявлено отсутствующим. Жёлтое — замер "
             "есть, но часть чего-то не умеет: сводка не имеет права зеленеть поверх "
             "известного провала.\nКрасное — отсутствует, чаще всего невыполнимо "
             "здесь: нет дисплея, звука, игры, ffmpeg и доступа к модели.",
             ha="center", fontsize=8.5, color="#9aa4b2")
    return _out(base, "7-chto-sdelano.png", fig)


def main(argv: list[str]) -> int:
    base = Path(argv[1]) if len(argv) > 1 else Path("figures")
    base.mkdir(parents=True, exist_ok=True)
    session = base / "session-for-figures"
    if not session.exists():
        generate_session(session, seed=7)

    fig_world(base)
    fig_selfworld(base, session)
    fig_prediction(base)
    fig_babbling(base)
    fig_places(base)
    fig_loop(base)
    fig_domains(base)
    fig_planner(base)
    fig_status(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
