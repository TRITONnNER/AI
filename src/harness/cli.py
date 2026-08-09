"""Командная строка харнесса.

    harness backends                     что доступно на этой машине
    harness profile                      профиль вехи 0 и его хеши
    harness gen-corpus PATH [--seed N]   записать синтетическую сессию
    harness verify PATH                  проверить целостность записи
    harness journal PATH                 сводка по журналу
    harness report PATH                  показатели и самоотчёт агента рядом
    harness replay PATH [--at N]         пройти запись покадрово
    harness selfworld PATH               разделить экранный и мировой слои (0.6)
    harness record PATH                  запись с живого экрана

Команда `record` на машине без дисплея честно отказывается, а не пишет чёрные
кадры: см. `harness backends`.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from pathlib import Path


def _print_json(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def cmd_backends(_: argparse.Namespace) -> int:
    from .capture.base import describe_backends

    rows = describe_backends()
    width = max(len(k) for k in rows)
    for name, info in rows.items():
        mark = "есть" if info["available"] else "нет "
        print(f"  [{mark}] {name.ljust(width)}  {info['why']}")
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    from .core import settings as sch
    from .core.profile import MILESTONE_0, short

    if getattr(args, "schema", False):
        rows = sch.table(getattr(args, "group", None))
        if not rows:
            print(f"нет группы {args.group!r}; есть: {', '.join(sch.GROUPS)}",
                  file=sys.stderr)
            return 2
        widths = {k: max(len(str(r[k])) for r in rows) for k in
                  ("key", "unit", "default", "range")}
        group = None
        for r in rows:
            if r["group"] != group:
                group = r["group"]
                print(f"\n=== {group} ===")
            print(f"  [{r['block']}] {r['key']:<{widths['key']}}  "
                  f"{str(r['default']):>{widths['default']}} {r['unit']:<{widths['unit']}}  "
                  f"{r['range']:<{widths['range']}}  {r['note']}")
        print(f"\nвсего настроек: {len(rows)}; [А] крутится на ходу, "
              "[Б] форкает журнал")
        return 0

    p = MILESTONE_0
    print(f"профиль {p.name}")
    print(f"  profile_hash   {short(p.profile_hash)}  ({p.profile_hash})")
    print(f"  structure_hash {short(p.structure_hash)}  ({p.structure_hash})")
    print("\n  Блок А · параметры (меняются на ходу):")
    for k, v in sorted(p.parameters.items()):
        print(f"    {k:28} {v}")
    print("\n  Блок Б · структурные (форкают журнал):")
    for k, v in sorted(p.structural.items()):
        print(f"    {k:28} {v}")
    return 0


def cmd_gen_corpus(args: argparse.Namespace) -> int:
    from .corpus.synthetic import DEFAULT_TIMELINE, Segment, generate_session

    timeline = DEFAULT_TIMELINE
    if args.frames:
        # Растянуть или обрезать раскладку до нужного числа кадров, сохранив
        # чередование «стоим / едем»: без него 0.6 нечего разделять.
        total = sum(s.frames for s in DEFAULT_TIMELINE)
        k = args.frames / total
        timeline = tuple(Segment(max(2, round(s.frames * k)), s.vx, s.vy, s.acting)
                         for s in DEFAULT_TIMELINE)

    path = generate_session(args.path, seed=args.seed, timeline=timeline,
                            width=args.width, height=args.height,
                            with_audio=not args.no_audio,
                            note=args.note)
    print(f"записана синтетическая сессия: {path}")
    from .session import Session
    with Session.open(path) as s:
        print(f"  кадров {len(s)}, записей журнала {len(s.journal)}, "
              f"кадры на диске {s.frames.bytes_on_disk() / 1024:.0f} КиБ")
    return 0


def cmd_devices(_: argparse.Namespace) -> int:
    from .core.profile import MILESTONE_0
    from .devices import default_registry

    reg = default_registry(MILESTONE_0)
    _print_json(reg.summary())
    print("\nчто из этого видит агент:")
    _print_json(reg.for_agent())
    return 0


def cmd_resources(_: argparse.Namespace) -> int:
    from .core.clocks import Stamp
    from .core.profile import MILESTONE_0
    from .core.resources import ProcMeasurer, ResourceGovernor

    m = ProcMeasurer()
    gov = ResourceGovernor(MILESTONE_0, measurer=m)
    breaches = gov.check(Stamp(0, 0))
    _print_json(gov.report())
    if m.is_peak_only:
        print("\nвнимание: /proc/self/statm недоступен, потребление измеряется "
              "пиковым getrusage — по нему не видно, помогло ли вытеснение")
    if breaches:
        print("\nупоры:")
        _print_json([b.as_dict() for b in breaches])
    return 0


def cmd_babble(args: argparse.Namespace) -> int:
    """Лепет по интерактивному миру: агент открывает своё тело с нуля."""
    from .behaviour.babbling import Babbler, run_babbling
    from .core.profile import BABBLE
    from .corpus.world import InteractiveWorld
    from .session import Recorder
    from .vision.predict import PredictionError

    profile = BABBLE
    world = InteractiveWorld(profile, seed=args.seed)
    with Recorder(args.path, profile=profile, source=f"babble:seed={args.seed}",
                  synthetic=True, note=args.note) as rec:
        babbler = Babbler(profile, world.outputs, journal=rec.journal,
                          rng_seed=args.seed)
        result = run_babbling(world, babbler, steps=args.steps, clocks=rec.clocks,
                              error=PredictionError(profile))
    print("итог лепета:")
    _print_json(result)

    if args.compare_truth:
        truth = world.truth()
        live_true = set(truth["live_outputs"])
        found = set(babbler.body.by_state("live"))
        pairs = babbler.body.undoable()
        print("\nсверка с истиной мира (только для исследователя):")
        print(f"  живых по истине     {len(live_true)}")
        print(f"  найдено живыми      {len(found)}, верно {len(found & live_true)}, "
              f"ложно {len(found - live_true)}")
        print(f"  обратных пар найдено {len(pairs)}")
        print(f"  необратимо по истине {[o for o in truth['irreversible_outputs']]}")
        print(f"  не умеет откатить    {result['not_undoable']}")
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    """Замкнутый круг: давление драйва → цель → пробы → тест цели.

    Ни одна часть здесь не подыгрывает другой: цель ставится по давлению драйвов,
    проверяется объективным тестом по карте тела, а карта тела заполняется
    настоящими пробами в мире, который про цель ничего не знает.
    """
    from .behaviour import selfreport
    from .behaviour.babbling import Babbler, run_babbling
    from .behaviour.goals import GoalStack, candidates_from_body, choose
    from .behaviour.skills import Library
    from .core.profile import from_schema
    from .corpus.world import InteractiveWorld
    from .model.drives import Motivation
    from .session import Recorder, Session
    from .vision.predict import PredictionError

    profile = from_schema("КРУГ-1", capture_width=320, capture_height=180,
                          babble_repeats=3, drive_horizon_s=60.0)
    world = InteractiveWorld(profile, seed=args.seed)
    # Порог осторожности больше не берётся здесь: его отдаёт модуляция на каждом
    # круге. Значение из профиля живо — оно её база.

    with Recorder(args.path, profile=profile, source=f"loop:seed={args.seed}",
                  synthetic=True, note=args.note) as rec:
        babbler = Babbler(profile, world.outputs, journal=rec.journal, rng_seed=args.seed)
        motivation = Motivation(profile)
        stack = GoalStack(profile, journal=rec.journal)
        error = PredictionError(profile)
        branch = rec.journal.meta.branch_id

        for _ in range(args.rounds):
            summary = error.summary()
            motivation.update(
                error_mean=summary["mean"], error_sigma=summary["sigma"],
                error_now=summary["last"],
                unknown_reversibility=babbler.progress()["unknown_reversibility"])
            # Модуляция берётся один раз на круг и отдаётся потребителям. Без этой
            # строки оси считаются и не читаются: именно так три из них и оказались
            # мёртвым кодом, а отчёт при этом печатал их сдвиги.
            mod = motivation.modulation(
                error_high=summary["last"] > summary["mean"])
            if stack.active is None:
                cands = candidates_from_body(babbler.body, world.outputs,
                                             caution_threshold=mod.caution_threshold)
                if cands:
                    stack.push(choose(cands, motivation, top=1)[0], motivation,
                               rec.journal.seq, branch, rec.clocks.stamp(),
                               budget_ticks=args.budget)
            run_babbling(world, babbler, steps=args.steps_per_round,
                         clocks=rec.clocks, error=error,
                         explore_rate=mod.explore_rate,
                         caution_threshold=mod.caution_threshold)
            stack.tick(rec.clocks.stamp())

            # Отчёт о себе — в конце каждого круга. Он никуда не возвращается и
            # ничем не читается: круг после него идёт точно так же, как без него.
            # Проверить это можно, убрав следующие три строки — числа не изменятся.
            stamp = rec.clocks.stamp()
            report = selfreport.compose(
                stamp=stamp, body=babbler.body, goals=stack, motivation=motivation,
                error=error, outputs=len(world.outputs),
                did={"babble": babbler.progress()["probes_done"]},
                error_high=summary["last"] > summary["mean"])
            selfreport.journal_report(rec.journal, report, stamp)

        goal_stats = stack.stats()
        body_stats = babbler.progress()

    print("цели:")
    _print_json(goal_stats)
    print("\nмотивация:")
    _print_json({"mood": motivation.mood.as_dict(),
                 "ведущий драйв": motivation.dominant().name,
                 "давление": motivation.goal_pressure(),
                 "модуляция порогов": motivation.modulation().as_dict()})
    print("\nтело:")
    _print_json({k: body_stats[k] for k in
                 ("outputs_total", "live", "silent", "untried", "probes_done",
                  "inverse_pairs", "not_undoable")})

    with Session.open(args.path) as s:
        library = Library()
        library.from_journal(s.journal, min_repeats=args.min_repeats)
        print("\nнавыки, найденные в журнале:")
        _print_json(library.stats())
        for skill in library.best_for()[:5]:
            mark = "догадка" if skill.is_guess else "проверен"
            print(f"  {skill.id}  n={skill.n:<3} {mark:<8} {skill.signature()}")

    if args.compare_truth:
        truth = world.truth()
        live = set(truth["live_outputs"])
        found = set(babbler.body.by_state("live"))
        print("\nсверка с истиной мира (только для исследователя):")
        print(f"  живых верно {len(found & live)} из {len(live)}, "
              f"ложных {len(found - live)}")
        print(f"  необратимо по истине {truth['irreversible_outputs']}")
        print(f"  не умеет откатить    {body_stats['not_undoable']}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    from .session import Session

    with Session.open(args.path) as s:
        report = s.verify()
    _print_json(report)
    return 0 if report["ok"] else 1


def cmd_journal(args: argparse.Namespace) -> int:
    from .core.journal import branch_chain
    from .session import Session

    with Session.open(args.path) as s:
        _print_json(s.journal.stats())
        chain = branch_chain(Path(args.path) / "journal")
        if len(chain) > 1:
            print("\nветки:")
            for b in chain:
                parent = f" ← {b.parent}" if b.parent else ""
                print(f"  {b.branch_id}{parent}  причина: {b.fork_reason}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    from .session import Session
    from .vision.selfworld import frame_change

    with Session.open(args.path) as s:
        if args.at is not None:
            c = s.seek(args.at)
            img = s.image(c.index)
            print(f"кадр {c.index}: запись {c.entry_seq}, {c.stamp}, форма {img.shape}, "
                  f"средняя яркость {img.mean():.1f}")
            return 0
        prev = None
        acted = {e.stamp.t_world for e, _ in s.actions()}
        for c, img in s:
            ch = "—" if prev is None else f"{frame_change(prev, img):.4f}"
            mark = "действие" if c.stamp.t_world in acted else ""
            print(f"  {c.index:4d}  {c.stamp}  изменилось {ch:>7}  {mark}")
            prev = img
        print(f"\nвсего кадров: {len(s)}")
    return 0


def cmd_selfworld(args: argparse.Namespace) -> int:
    import numpy as np

    from .session import Session
    from .vision.selfworld import SCREEN, SelfWorldSeparator, background_level

    with Session.open(args.path) as s:
        sep = SelfWorldSeparator(s.profile)
        acted = {e.stamp.t_world for e, _ in s.actions()}
        pairs = []
        prev = None
        for c, img in s:
            sep.feed(img)
            if prev is not None:
                pairs.append((prev, img, c.stamp.t_world in acted))
            prev = img
        res = sep.result()
        bg = background_level(pairs)

        print("разделение слоёв:")
        _print_json(res.summary())
        print("\nфон и превышение над фоном:")
        _print_json(bg.as_dict())

        # Истина есть только у синтетики и только в отладочном потоке.
        try:
            from .corpus.synthetic import load_hud_mask
            truth = load_hud_mask(args.path)
        except Exception as e:
            print(f"\nистинной маски нет ({e.__class__.__name__}): "
                  "сверять не с чем, это живая запись")
            return 0

        found = res.pixel_mask(SCREEN)
        inter = int((found & truth).sum())
        union = int((found | truth).sum())
        iou = inter / union if union else 0.0
        print("\nсверка с истиной из отладочного потока:")
        print(f"  IoU по пикселям        {iou:.3f}")
        print(f"  найдено интерфейса     {int(found.sum())} px")
        print(f"  истинного интерфейса   {int(truth.sum())} px")
        print(f"  ложно принято за него  {int((found & ~truth).sum())} px")
        print(f"  пропущено              {int((~found & truth).sum())} px")
        if args.dump_mask:
            np.save(args.dump_mask, found)
            print(f"  маска сохранена: {args.dump_mask}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Состояние проекта, посчитанное по коду и тестам.

    Не текст, который кто-то написал и забыл обновить: инварианты берутся из тестов,
    настройки из схемы, части из наличия модулей и тестов, а «чего нет» — из списка,
    в котором у каждой строки обязана быть причина.
    """
    import importlib.util
    import sys as _sys
    from pathlib import Path as _Path

    tools = _Path(__file__).resolve().parent.parent.parent / "tools"
    spec = importlib.util.spec_from_file_location("project_status",
                                                 tools / "project_status.py")
    if spec is None or spec.loader is None:
        print("не нашёл tools/project_status.py: отчёт считается им", file=sys.stderr)
        return 2
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["project_status"] = mod
    spec.loader.exec_module(mod)

    data = mod.collect(with_tests=args.tests)
    if args.json:
        _print_json(data)
    else:
        print(mod.render_text(data))
    if args.md:
        args.md.write_text(mod.render_markdown(data), encoding="utf-8")
        print(f"\nзаписано: {args.md}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Две колонки: объективные показатели и слова агента о себе.

    Рядом, но не вместе — и это главное в этой команде. Слева то, что посчитано по
    журналу и ни от чьих слов не зависит. Справа то, что агент сказал о себе, и оно
    ни на что не влияет: последняя запись `SELF_REPORT` просто читается из журнала
    как есть.

    Расхождение колонок — не ошибка, а самое интересное в отчёте. «Я научился» при
    компетентности 0.2 значит, что агент считает освоенным то, что тестами целей не
    подтверждается; сравнить это можно только глядя на обе колонки одновременно.
    """
    from .core.journal import Kind
    from .model import confabulation, vitals
    from .session import Session

    with Session.open(args.path) as s:
        v = vitals.from_journal(s.journal, profile=s.profile,
                                skip=() if args.beliefs else ("beliefs",))
        reports = [e for e in s.journal if e.kind is Kind.SELF_REPORT]
        # Метрика конфабуляции — третья величина в отчёте, и она про **разницу**
        # между колонками: заявленный слой против настоящего инициатора. Считается
        # тут же по журналу, потому что контроллеру сопоставлять нельзя.
        conf = confabulation.measure(s.journal)
        layers = confabulation.layer_histogram(s.journal)

    if args.json:
        _print_json({"vitals": v.as_dict(),
                     "confabulation": conf.as_dict(),
                     "actor_layers": layers,
                     "self_report": reports[-1].event if reports else None,
                     "self_reports": len(reports)})
        return 0

    print("показатели (посчитаны по журналу)")
    print(v.render_text())

    absent = v.absent()
    if absent:
        print(f"\nчего в журнале нет ({len(absent)}): "
              + ", ".join(x.code for x in absent))
        print("  причины — в json-выводе; это не заготовки, а честные пропуски")

    print("\nатрибуция действий")
    print(f"  {conf.line()}")
    if layers:
        print("  записей по слою-инициатору: "
              + ", ".join(f"{k} {n}" for k, n in layers.items()))
        print("  доля расхождений без этого распределения обманчива: 20 % при "
              "девяноста процентах записей от планировщика")
    for m in conf.examples[:3]:
        print(f"    запись {m.seq}: заявлен {m.claimed}, начал {m.actual}")

    print(f"\nсамоотчёты агента: {len(reports)}")
    if not reports:
        print("  ни одного. Отчёт пишется агентской стороной, а не этой командой")
        return 0
    last = reports[-1]
    print(f"  последний: запись {last.seq}, {last.stamp}")
    for line in last.event.get("lines", []):
        code = str(line.get("code"))
        value = line.get("value")
        prov = str(line.get("provenance") or "?")
        print(f"    {code:<22} {str(value):<28} происхождение: {prov}")
    print("\nслова справа ни на что не влияют: пересборка записи SELF_REPORT "
          "пропускает (инвариант 10)")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Весь стек до плана: лепет → разведка → модель → план → исполнение.

    Ничего не записывает в сессию: это демонстрация и замер, а не запись.
    """
    import numpy as np

    from .behaviour.babbling import Babbler, run_babbling
    from .behaviour.goals import Goal
    from .behaviour.planner import Planner, choose_probe, execute
    from .behaviour.skills import MacroRecorder
    from .core.action import Action, action_key
    from .core.clocks import Clocks
    from .core.profile import from_schema
    from .corpus.world import InteractiveWorld
    from .model.beliefs import Origin, Provenance
    from .model.forward import ForwardModel
    from .model.places import PlaceGraph

    profile = from_schema("ПЛАН-CLI", capture_width=320, capture_height=180,
                          babble_repeats=3)
    min_n = int(profile.parameters["plan_min_step_n"])
    hold = args.hold

    print("1. Лепет: какие выходы что делают и чем откатываются")
    babble_world = InteractiveWorld(profile, seed=args.seed, n_outputs=16)
    babbler = Babbler(profile, babble_world.outputs, rng_seed=args.seed)
    run_babbling(babble_world, babbler, steps=args.babble, clocks=Clocks())
    inverse = dict(babbler.inverse_found)
    prog = babbler.progress()
    print(f"   живых {prog['live']}, молчащих {prog['silent']}, "
          f"неясных {prog['unclear']}, обратных пар {len(inverse)}")

    print("2. Разведка, которая подтверждает переходы")
    world = InteractiveWorld(profile, seed=args.seed, n_outputs=16)
    graph = PlaceGraph.from_profile(profile)
    state: dict[str, Any] = {"seq": 0, "last": None}

    macro_len = int(profile.structural["macro_max_length"])
    macros = (MacroRecorder(graph, max_length=macro_len, seconds_per_seq=1 / 30.0)
              if macro_len >= 2 else None)

    def step(output: str, duration_ms: int = hold) -> str:
        src = graph.current
        obs = world.step(Action.key(output, duration_ms), with_audio=False)
        state["seq"] += 1
        state["last"] = output
        dst = graph.see(obs.frame, state["seq"], seconds_per_seq=1 / 30.0,
                        mode=action_key(output, duration_ms))
        # Цепочка, пройденная целиком, — тоже наблюдение, и своё: у неё своя
        # надёжность, а не произведение надёжностей звеньев.
        if macros is not None and src is not None:
            macros.note(src, action_key(output, duration_ms), dst, state["seq"])
        return dst

    graph.see(world.step(None, with_audio=False).frame, 0,
              seconds_per_seq=1 / 30.0, mode="start")
    # Карта тела из лепета передаётся модели: без неё осторожность у всех выходов
    # максимальна, и любой план оказывается «опасным» — верно по букве инварианта 9,
    # но бесполезно, потому что откатываемое от неоткатываемого уже отличимо.
    body = babbler.body
    model = ForwardModel.from_graph(graph, body)
    refinements: list[dict[str, Any]] = []
    for i in range(args.explore):
        if i % 25 == 0:
            # Уточнение карты перед пересборкой модели: если место склеило два
            # состояния мира, модель, собранная до деления, унаследует расхождение.
            refinements += graph.refine()
            model = ForwardModel.from_graph(graph, body)
        out, ms = choose_probe(model, graph.current, world.outputs, hold_ms=hold,
                               min_n=min_n, inverse=inverse,
                               last_output=state["last"])
        step(out, ms)
    refinements += graph.refine()
    model = ForwardModel.from_graph(graph, body)
    confirmed = sum(1 for outs in model.transitions.values()
                    for o in outs if o.n >= min_n)
    gs = graph.stats()
    print(f"   мест {len(graph)}, пар (место, действие) {len(model.transitions)}, "
          f"подтверждённых исходов {confirmed}")
    print(f"   рёбер {gs['edges']}, из них петель «нажал и остался» {gs['loops']}")
    if macros is not None:
        print(f"   макро-рёбер записано {macros.stats()['recorded']}: цепочка до "
              f"{macro_len} шагов может стать одним шагом плана")

    def model_reach(src: str, max_depth: int = 4) -> dict[str, int]:
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
                            # Петля никуда не ведёт: считать её шагом значит
                            # обещать достижимость, которой нет. Пока петли не
                            # записывались, этого случая не существовало.
                            continue
                        if outcome.dst not in depth:
                            depth[outcome.dst] = d + 1
                            nxt.append(outcome.dst)
            frontier = nxt
        return {k: v for k, v in depth.items() if v > 0}

    # Досягаемость при той же глубине поиска — то, ради чего макросы и нужны.
    # Замер: без макросов достижимо 12 мест (медиана), с цепочками до трёх — 21.
    here0 = graph.current or ""
    reach_all = model_reach(here0)
    singles_only = {k: v for k, v in reach_all.items()}
    print(f"   достижимо мест при глубине 4: {len(reach_all)}")

    print("3. Планы до мест, куда модель знает дорогу")
    del singles_only
    found = arrived = 0
    for _ in range(args.goals):
        refinements += graph.refine()
        model = ForwardModel.from_graph(graph, body)
        here = graph.current
        reach = model_reach(here or "")
        if not reach:
            out, ms = choose_probe(model, here, world.outputs, hold_ms=hold,
                                   min_n=min_n, inverse=inverse,
                                   last_output=state["last"])
            step(out, ms)
            continue
        target = sorted(reach)[-1]
        goal = Goal(id=f"g{found}", kind="reach_place", target=target,
                    test=lambda t=target: graph.current == t,
                    test_text="я в этом месте", budget_ticks=40,
                    provenance=Provenance(Origin.EXPERIENCE, branch="cli", seq=0),
                    drive="curiosity", pressure=0.5)
        plan = Planner(profile, model).plan(goal, here, target)
        if plan is None:
            continue
        found += 1
        ex = execute(plan, act=lambda a: step(a.outputs_touched()[0], a.duration_ms),
                     goal=goal)
        arrived += ex.goal_passed
        if found <= args.show:
            chains = sum(1 for s in plan.steps if s.is_macro)
            presses = sum(len(s.chain) or 1 for s in plan.steps)
            print(f"   план {plan.length} шагов ({presses} нажатий, "
                  f"навыков {chains}), {plan.seconds:.2f} с, "
                  f"слабое звено {plan.min_step_n}, опасный {plan.risky} → "
                  f"{'дошёл' if ex.goal_passed else 'нет: ' + ex.reason[:50]}")
    if found:
        print(f"   планов {found}, дошли {arrived} ({arrived / found:.0%})")
    else:
        print("   ни одного плана: модель не знает ни одной подтверждённой дороги. "
              "Это честный ответ, а не поломка — надо разведывать дольше")

    if refinements:
        print(f"\n4. Карта уточнилась по расхождению предсказаний: разделено мест "
              f"{len(refinements)}")
        for r in refinements[:4]:
            print(f"     {r['place']} по признаку {r['feature']} "
                  f"(разрыв {r['gap']}), выброшено рёбер {r['edges_dropped']} — "
                  f"место склеивало два состояния мира")
        print("   Статистика склеенного места выброшена, а не поделена: она была "
              "собрана про узел, которого больше нет")
    else:
        # «Не разделилось» и «нечего делить» — разные вещи, и путать их нельзя.
        # Расхождение, случившееся один раз, — шум узнавания, и деления по нему не
        # будет по построению (`place_refine_min_n`). Поэтому здесь печатается,
        # сколько расхождений вообще видно.
        diverging = sum(1 for outs in model.transitions.values() if len(outs) > 1)
        twice = sum(1 for outs in model.transitions.values()
                    if sum(1 for o in outs if o.n >= 2) > 1)
        print(f"\n4. Карта не делилась. Пар с расходящимися исходами {diverging}, "
              f"из них повторившихся дважды {twice}")
        print("   Одно расхождение — шум узнавания, а не открытие: делить по нему "
              "значило бы плодить места, которых нет")
    return 0 if found and arrived == found else 1


def cmd_describers(args: argparse.Namespace) -> int:
    """Что из бесплатных сервисов описания настроено на этой машине.

    Показывает не «что бывает вообще», а что реально доступно здесь и сейчас, и по
    какой причине недоступно остальное. Ни одного запроса к сервису при этом не
    делается: у сетевых проверяется ключ, у локальных — что узел отвечает.
    """
    from .perception.describers import PROVIDERS, probe_all

    rows = probe_all()
    width = max(len(r["name"]) for r in rows)
    print(f"{'сервис'.ljust(width)}  готов  лимиты        модель")
    print(f"{'-' * width}  -----  ------------  ------")
    for r in rows:
        limits = ("локально" if r["key_env"] is None else
                  f"{r['rpm'] or '?'}/мин {r['rpd'] or '?'}/сут")
        mark = " да  " if r["configured"] else " нет "
        print(f"{r['name'].ljust(width)}  {mark}  {limits:<12}  {r['model']}")
    print()
    for r in rows:
        if not r["configured"]:
            print(f"{r['name']}: {r['why']}")
    print("\nчисла лимитов справочные (август 2026) и требуют перепроверки: "
          "тарифы меняются чаще, чем код.")
    if args.verbose:
        print()
        for p in PROVIDERS:
            print(f"{p.name}: {p.note}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Кросс-доменный замер: одни и те же модули по четырём разным мирам.

    Главная проверка требования «если решение работает только в Minecraft, оно
    неправильное». Ничего не записывает: это замер, а не сессия.
    """
    from .benchmark import EXPECTATION, run_all

    rep = run_all(seed=args.seed, frames=args.frames,
                  babble_steps=args.babble_steps, motion_px=args.motion,
                  domains=args.domains or None)
    print(rep.table())
    print()
    for r in rep.results:
        known = rep.expected_failure(r)
        mark = "OK " if rep.passed(r) else ("НЕТ" if not known else "НЕТ*")
        print(f"{mark} {r.domain}: {r.verdict}")
        if known and not rep.passed(r):
            print(f"      * провал объявлен: {known}")
        print(f"      ожидаемо: {EXPECTATION.get(r.domain, '—')}")
        print(f"      признак:  {r.signal_reason}")
    ok = rep.as_dict()["ok"]
    print("\nитог:", "все домены прошли" if ok else "есть домены с провалом")
    if args.json:
        _print_json(rep.as_dict())
    return 0 if ok else 1


def cmd_record(args: argparse.Namespace) -> int:
    """Запись с живого экрана. С `--plan` только печатает, что записать.

    `--actor human` ставит в записи `Actor.HUMAN` и `ActorLayer.HUMAN`: в
    демонстрационной сессии действует оператор, и в журнале это должно быть видно.
    Иначе живой корпус окажется неотличим по атрибуции от синтетического, где
    кадры не начаты никем.
    """
    from .capture.base import BackendUnavailable
    from .capture.screen import ScreenCapture
    from .core.journal import Actor, ActorLayer
    from .core.profile import MILESTONE_0
    from .corpus.live import plan_text
    from .session import Recorder

    if args.plan:
        print(plan_text())
        return 0

    human = args.actor == "human"
    actor = Actor.HUMAN if human else Actor.NONE
    layer = ActorLayer.HUMAN if human else ActorLayer.NONE

    cap = ScreenCapture(gray=MILESTONE_0.structural["frame_format"] == "gray8")
    try:
        cap.start()
    except BackendUnavailable as e:
        print(f"захват недоступен: {e}", file=sys.stderr)
        print("\nчто доступно:", file=sys.stderr)
        cmd_backends(args)
        return 2

    written = 0
    try:
        with Recorder(args.path, profile=MILESTONE_0, source=cap.name,
                      synthetic=False, note=args.note) as rec:
            while written < args.frames:
                frame = cap.read()
                if frame is None:
                    rec.record_gap("source_ended", {"after_frames": written})
                    break
                rec.record_frame(frame.image, t_world=frame.t_world,
                                 actor=actor, actor_layer=layer)
                written += 1
    finally:
        cap.stop()
    print(f"записано кадров: {written} → {args.path}")
    print(f"слой-инициатор: {layer}. Дальше: harness ingest {args.path} --kind ВИД")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Принять запись с чужой машины в живой корпус. Ничего не пересчитывая."""
    from .corpus.live import KINDS, LiveError, ingest

    if args.list:
        for name, k in KINDS.items():
            print(f"  {name:<12} {k['title']} — {k['duration']}")
        return 0
    try:
        got = ingest(args.path, args.corpus, kind=args.kind, copy=not args.in_place)
    except LiveError as e:
        print(f"не принято: {e}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(got.as_dict())
    else:
        print(got.text())
    return 0


def cmd_mark(args: argparse.Namespace) -> int:
    """Отметить область обрамления на живой записи. Истина исследователя.

    Без этого IoU по живой записи не считается вовсе: на настоящем экране истины
    нет, а сравнить ответ метода с маской, полученной тем же методом, значит
    сравнить его с самим собой.
    """
    from .corpus.live import Region, read_regions, write_regions

    regions = list(read_regions(args.path))
    for quad in args.screen or []:
        top, left, height, width = (int(x) for x in quad)
        regions.append(Region(top, left, height, width, "screen"))
    path = write_regions(args.path, regions, note=args.note or "")
    print(f"разметка: {len(regions)} областей → {path}")
    print("лежит в debug/: это истина исследователя, агентскому код у она недоступна")
    return 0


def cmd_bench_live(args: argparse.Namespace) -> int:
    """Прогнать по живым записям тот же код разделения себя и мира."""
    from pathlib import Path as _Path

    from .corpus.live import LiveError, bench_live, compare_to_synthetic

    root = _Path(args.path)
    sessions = ([root] if (root / "session.json").exists()
                else sorted(p for p in root.iterdir()
                            if (p / "session.json").exists()))
    if not sessions:
        print(f"в {root} нет ни одной записи харнесса", file=sys.stderr)
        return 2

    from .corpus.live import METHOD_COMPARABLE, METHODS

    methods = [args.method] if args.method != "both" else list(METHODS)
    out: dict[str, Any] = {}
    for method in methods:
        scores = []
        for path in sessions:
            try:
                score = bench_live(path, method=method)
            except LiveError as e:
                print(f"{path.name}: {e}", file=sys.stderr)
                continue
            scores.append(score)
            print(score.text())
            print()
        # Синтетическая медиана своя у каждого пути: сравнивать арбитра с числом,
        # снятым одиночным разделителем, значило бы сравнивать разное.
        baseline = (args.synthetic_median if method == "arbiter"
                    else args.synthetic_corpus_median)
        cmp = compare_to_synthetic(scores, synthetic_median=baseline)
        print(f"путь «{method}»: {METHOD_COMPARABLE[method]}")
        print(f"  {cmp['verdict']}")
        if cmp.get("unscored"):
            print(f"  без разметки, в сравнение не вошли: {cmp['unscored']}")
        print()
        out[method] = {"scores": [s.as_dict() for s in scores], "comparison": cmp}
    if args.json:
        _print_json(out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="harness", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("backends", help="что доступно на этой машине").set_defaults(fn=cmd_backends)
    pr = sub.add_parser("profile", help="профиль и схема настроек")
    pr.add_argument("--schema", action="store_true", help="вся схема настроек таблицей")
    pr.add_argument("--group", default=None, help="только одна группа настроек")
    pr.set_defaults(fn=cmd_profile)

    sub.add_parser("devices", help="устройства ввода-вывода").set_defaults(fn=cmd_devices)
    sub.add_parser("resources", help="ресурсы и пределы").set_defaults(fn=cmd_resources)

    bb = sub.add_parser("babble", help="лепет: агент открывает своё тело")
    bb.add_argument("path", type=Path)
    bb.add_argument("--seed", type=int, default=0)
    bb.add_argument("--steps", type=int, default=1500)
    bb.add_argument("--note", default=None)
    bb.add_argument("--compare-truth", action="store_true",
                    help="сверить с истиной мира из отладочного потока")
    bb.set_defaults(fn=cmd_babble)

    g = sub.add_parser("gen-corpus", help="записать синтетическую сессию")
    g.add_argument("path", type=Path)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--frames", type=int, default=0, help="примерное число кадров")
    g.add_argument("--width", type=int, default=320)
    g.add_argument("--height", type=int, default=180)
    g.add_argument("--no-audio", action="store_true")
    g.add_argument("--note", default=None)
    g.set_defaults(fn=cmd_gen_corpus)

    lp = sub.add_parser("loop", help="замкнутый круг: драйв → цель → пробы → тест")
    lp.add_argument("path", type=Path)
    lp.add_argument("--seed", type=int, default=23)
    lp.add_argument("--rounds", type=int, default=40)
    lp.add_argument("--steps-per-round", type=int, default=40)
    lp.add_argument("--budget", type=int, default=10, help="бюджет цели в тактах")
    lp.add_argument("--min-repeats", type=int, default=3,
                    help="сколько повторов делает цепочку навыком")
    lp.add_argument("--note", default=None)
    lp.add_argument("--compare-truth", action="store_true")
    lp.set_defaults(fn=cmd_loop)

    v = sub.add_parser("verify", help="проверить целостность записи")
    v.add_argument("path", type=Path)
    v.set_defaults(fn=cmd_verify)

    j = sub.add_parser("journal", help="сводка по журналу")
    j.add_argument("path", type=Path)
    j.set_defaults(fn=cmd_journal)

    r = sub.add_parser("replay", help="пройти запись покадрово")
    r.add_argument("path", type=Path)
    r.add_argument("--at", type=int, default=None, help="показать один кадр")
    r.set_defaults(fn=cmd_replay)

    sw = sub.add_parser("selfworld", help="разделить экранный и мировой слои (0.6)")
    sw.add_argument("path", type=Path)
    sw.add_argument("--dump-mask", type=Path, default=None)
    sw.set_defaults(fn=cmd_selfworld)

    rp = sub.add_parser("report", help="показатели по журналу рядом со словами "
                                       "агента о себе")
    rp.add_argument("path", type=Path)
    rp.add_argument("--beliefs", action="store_true",
                    help="пересобрать убеждения и карту тела (медленнее)")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(fn=cmd_report)

    stt = sub.add_parser("status", help="что готово, что нет и почему — по коду")
    stt.add_argument("--tests", action="store_true",
                     help="прогнать тесты и показать их итог")
    stt.add_argument("--md", type=Path, default=None, help="записать markdown")
    stt.add_argument("--json", action="store_true")
    stt.set_defaults(fn=cmd_status)

    pn = sub.add_parser("plan", help="весь стек до плана: лепет, разведка, "
                                     "модель, план, исполнение")
    pn.add_argument("--seed", type=int, default=5)
    pn.add_argument("--babble", type=int, default=1200, help="шагов лепета")
    pn.add_argument("--explore", type=int, default=1500, help="шагов разведки")
    pn.add_argument("--goals", type=int, default=20, help="сколько целей поставить")
    pn.add_argument("--hold", type=int, default=200, help="удержание при пробе, мс")
    pn.add_argument("--show", type=int, default=5, help="сколько планов напечатать")
    pn.set_defaults(fn=cmd_plan)

    ds = sub.add_parser("describers",
                        help="бесплатные сервисы описания: что настроено")
    ds.add_argument("--verbose", action="store_true", help="с пояснениями по каждому")
    ds.set_defaults(fn=cmd_describers)

    bn = sub.add_parser("bench", help="кросс-доменный замер: игра, документ, "
                                     "рабочий стол, видео")
    bn.add_argument("--seed", type=int, default=3)
    bn.add_argument("--frames", type=int, default=140)
    bn.add_argument("--babble-steps", type=int, default=700)
    bn.add_argument("--motion", type=int, default=40,
                    help="размах движения мыши за шаг, px")
    bn.add_argument("--domains", nargs="*", default=None,
                    help="только эти домены")
    bn.add_argument("--json", action="store_true", help="ещё и машинно читаемо")
    bn.set_defaults(fn=cmd_bench)

    rec = sub.add_parser("record", help="запись с живого экрана")
    rec.add_argument("path", type=Path, nargs="?", default=Path("."))
    rec.add_argument("--frames", type=int, default=300)
    rec.add_argument("--note", default=None)
    rec.add_argument("--actor", choices=("human", "none"), default="none",
                     help="кто действует: human для демонстрации оператором")
    rec.add_argument("--plan", action="store_true",
                     help="только напечатать, что записать, и ничего не писать")
    rec.set_defaults(fn=cmd_record)

    ing = sub.add_parser("ingest", help="принять запись с чужой машины в корпус")
    ing.add_argument("path", type=Path, nargs="?", default=Path("."))
    ing.add_argument("--corpus", type=Path, default=Path("corpus/live"))
    ing.add_argument("--kind", default="play",
                     help="вид записи; --list покажет все")
    ing.add_argument("--in-place", action="store_true",
                     help="не копировать, только проверить на месте")
    ing.add_argument("--list", action="store_true", help="какие виды записей нужны")
    ing.add_argument("--json", action="store_true")
    ing.set_defaults(fn=cmd_ingest)

    mk = sub.add_parser("mark", help="отметить область обрамления на живой записи")
    mk.add_argument("path", type=Path)
    mk.add_argument("--screen", nargs=4, action="append", metavar=("ВЕРХ", "ЛЕВО", "ВЫСОТА", "ШИРИНА"),
                    help="прямоугольник экранного слоя; можно указать несколько раз")
    mk.add_argument("--note", default=None)
    mk.set_defaults(fn=cmd_mark)

    bl = sub.add_parser("bench-live", help="прогнать тот же код по живым записям")
    bl.add_argument("path", type=Path, default=Path("corpus/live"), nargs="?")
    bl.add_argument("--method", choices=("arbiter", "parallax", "both"),
                    default="both",
                    help="какой из двух существующих путей прогнать")
    bl.add_argument("--synthetic-median", type=float, default=0.667,
                    help="медиана IoU арбитра по пяти доменам, n=5 (MEASUREMENT.md)")
    bl.add_argument("--synthetic-corpus-median", type=float, default=0.9696,
                    help="медиана IoU одиночного разделителя на корпусе, n=1")
    bl.add_argument("--json", action="store_true")
    bl.set_defaults(fn=cmd_bench_live)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
