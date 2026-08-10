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
    harness panel                        панель в браузере: запись, ход, итог

Команда `record` на машине без дисплея честно отказывается, а не пишет чёрные
кадры: см. `harness backends`.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from pathlib import Path


def _record_kinds() -> tuple[str, ...]:
    """Виды записи для `--kind`. Из одного места (`corpus.live.KINDS`), а не списком.

    Второй список здесь разошёлся бы с первым молча — ровно так же, как разошлись три
    копии знания о Wayland (TASK-15).
    """
    from .corpus.live import KINDS

    return tuple(sorted(KINDS))


def _path_arg(raw: str) -> Path:
    """Путь из командной строки: тильда раскрыта, буквальный `~` внутри — отказ.

    Стоит **типом аргумента**, а не проверкой в каждой команде: раскрытие, забытое в одной
    команде из семнадцати, — это ровно то, как тильда доехала до `ingest` во второй раз.
    Здесь его забыть нельзя: аргумент пути без этого типа не создаётся.
    """
    import argparse as _ap

    from .paths import PathError, resolve_input

    try:
        return resolve_input(raw)
    except PathError as e:
        raise _ap.ArgumentTypeError(str(e)) from None


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
        motivation = Motivation(profile)
        stack = GoalStack(profile, journal=rec.journal)
        error = PredictionError(profile)

        def state_extra() -> dict[str, Any]:
            """Чем дополнить срез состояния каждой записи действия.

            Драйвы, настроение и активная цель считались в этом цикле и раньше, но в
            записи не попадали: срез оставался пустым, и все показатели, построенные на
            `StateSnapshot`, докладывали «не измерялось» по прогону, где всё было
            измерено. Числа были в оперативной памяти и умирали вместе с процессом.
            """
            active = stack.active
            return {"drives": {name: {"value": round(d.value, 6),
                                      "forecast": round(d.forecast, 6)}
                               for name, d in motivation.drives.items()},
                    "mood": (round(motivation.mood.valence, 6),
                             round(motivation.mood.arousal, 6)),
                    "goal_id": None if active is None else active.id}

        babbler = Babbler(profile, world.outputs, journal=rec.journal,
                          rng_seed=args.seed, state_extra=state_extra)
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
        conf = confabulation.measure(
            s.journal,
            min_episodes=int(s.profile.parameters["confab_min_episodes"]))
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

    macros = MacroRecorder.from_profile(graph, profile, seconds_per_seq=1 / 30.0)

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
    from .capture.base import UNCHANGED, BackendUnavailable
    from .capture.screen import AudioOverflow
    from .core.journal import Actor, ActorLayer
    from .core.profile import MILESTONE_0
    from .corpus.live import plan_text
    from .capture.record_loop import STOP_REASONS, TURNS_MEAN
    from .cost import verdict
    from .paths import PathError, resolve_input, show
    from .progress import Progress
    from .recording import Plan, RecordRefused, record
    from .session import describe_existing

    if args.plan:
        print(plan_text(getattr(args, "plan_set", "minimal")))
        return 0

    # Тильду раскрываем сами: `cmd.exe` и `powershell` этого не делают, и `~` доехал бы
    # до нас именем каталога. Оператор так уже получил каталог `~` внутри проекта.
    try:
        target = resolve_input(args.path)
    except PathError as e:
        print(str(e), file=sys.stderr)
        return 2
    args.path = target

    # **`--seconds` — это секунды.** До TASK-17 они переводились в кадры и запись ждала
    # этого числа кадров: на 6.5 кадр/с «десять секунд» превращались в 46, молча. Теперь
    # срок обрывает запись, а недобор частоты печатается числом. `--frames` остаётся тем,
    # чем был: ровно столько кадров, сколько названо, сколько бы это ни заняло.
    fps = float(MILESTONE_0.parameters["capture_fps"])
    seconds: float | None = None
    frames: int | None = None
    if args.turns is not None:
        frames = args.turns
        if args.seconds is not None:
            print("указаны и --turns, и --seconds; беру --turns: это счёт оборотов, "
                  "а не срок", file=sys.stderr)
        print(f"{frames} оборотов при цели {fps:g} кадр/с — это около "
              f"{frames / fps:.0f} с, а на медленной машине больше")
        print(f"  {TURNS_MEAN}")
    else:
        seconds = args.seconds if args.seconds is not None else 10.0
        # **Предел оборотов при срочной записи не применяется вовсе.** TASK-17 сделал срок
        # в печати, но оставил предел в условии выхода: `--seconds 10` кончилась за секунду
        # с причиной «набрано заданное число кадров», потому что 29 кадров и 271 отметка
        # «без изменений» дали 300 оборотов = 10 × 30. Совмещать срок с пределом нельзя: на
        # записи неподвижности отметок почти сто процентов, и минута кончается за секунды.
        print(f"{seconds:g} с — срок, и только он: предел оборотов не применяется. "
              f"При цели {fps:g} кадр/с ожидается около {int(seconds * fps)} оборотов")

    # Занятый каталог проверяется **до** открытия захвата: иначе оператор ждёт выбора
    # механизма, первого кадра и звукового входа, чтобы получить отказ о том, что было
    # видно с самого начала.
    if target.exists() and any(target.iterdir()):
        print(f"{show(target)} не пуст.", file=sys.stderr)
        print(describe_existing(target), file=sys.stderr)
        return 2

    # Дальше — общий путь записи (`recording.record`), тот же, которым пишет панель.
    # Второй цикл записи в сервере разошёлся бы с этим через месяц, как трижды разошлось
    # знание о Wayland (TASK-15). Печать остаётся здесь: модуль записи ничего не печатает.
    from .capture.source import pairing

    plan = Plan(path=target, kind=getattr(args, "kind", None), seconds=seconds,
                turns=frames, actor_human=args.actor == "human",
                with_audio=not args.no_audio, audio_device=args.audio_device,
                note=args.note, progress_channel=getattr(args, "progress", None),
                source_kind=getattr(args, "source_kind", "display"),
                window=getattr(args, "window", "") or "",
                region=(tuple(args.region) if getattr(args, "region", None) else None))

    def say_ready(o: Any) -> None:
        print(f"механизм: {o.mechanism}")
        if o.mechanism_caveat:
            print(f"ВНИМАНИЕ: {o.mechanism_caveat}")
        print(f"кадр {o.frame[0]}×{o.frame[1]}: профиль записи построен по нему, "
              f"а не по значению из схемы")
        # Источник печатается **до** записи вместе с тем, откуда взялась рамка: узнать
        # после записи, что снималось не то, значит потерять запись целиком.
        print(f"источник: {o.source_why}")
        # Сочетание источника и области ввода объявляется **до** записи: «невидимая рука»
        # — дефект постановки, и узнать о нём после записи значит записать зря.
        pair = pairing(o.source_kind, str(MILESTONE_0.parameters["input_scope"]))
        head = "ВНИМАНИЕ: " if not pair["declared"] else ""
        print(f"{head}ввод и кадр — «{pair['kind']}»: {pair['text']}")
        if o.source_kind != "display":
            print(f"  запись попадёт в отдельную ветку журнала (structure_hash "
                  f"{o.structure_hash[:8]}): записи «{o.source_kind}» и «display» "
                  f"несравнимы и в одно число не сводятся")
        if o.audio_note.startswith("без звука ("):
            print(f"звук не пишется: {o.audio_note[11:-1]}", file=sys.stderr)
            print("это не мешает записи; настроить вход поможет harness doctor",
                  file=sys.stderr)

    prog = Progress.from_profile(
        MILESTONE_0, total_turns=frames, path=target, seconds=seconds,
        kind=getattr(args, "kind", None), channel=getattr(args, "progress", None))
    hello = prog.open()
    if hello:
        print(hello)
    try:
        out = record(plan, progress=prog, on_ready=say_ready)
    except RecordRefused as e:
        print(e.why, file=sys.stderr)
        if e.hint:
            print(e.hint, file=sys.stderr)
        return 2
    turns, cost = out.turns, out.cost
    written, unchanged = turns.written, turns.unchanged
    audio_blocks, interrupted = turns.audio_blocks, turns.interrupted
    audio_note, layer, lid = out.audio_note, ("human" if plan.actor_human else "none"), \
        out.lineage_id
    profile = MILESTONE_0

    # Итог печатается **один раз** и печатается здесь: до TASK-17 строку писал и `finish`,
    # и этот `print`, и оператор видел её дважды.
    print(out.final_line)
    print(f"почему кончилось: {STOP_REASONS[turns.stop_reason]}")
    # Недобор частоты — свойство машины, и он печатается числом, а не растягиванием
    # записи. Молчать об этом нельзя: по числу кадров, делённому на заявленную частоту,
    # длительность записи вышла бы 10 с там, где прошло 46.
    st = turns.stages()
    print(f"частота: {st['achieved_fps']:.1f} кадр/с достигнуто при цели "
          f"{profile.parameters['capture_fps']:g}; оборотов {turns.turns}, "
          f"ожидалось к этому сроку {turns.expected_turns}")
    # Холостая доля: сколько цикл **ждал**, чтобы не обогнать цель. Без неё не отличить
    # «машина едва успевает» от «машина ждёт три четверти времени», а это разные машины.
    print(f"из них ждал, чтобы не обогнать цель: {st['idle_share']:.0%} времени "
          f"({st['idle_ms']:.1f} мс на оборот)")
    if turns.expected_turns and turns.turns < turns.expected_turns * 0.9:
        print(f"машина не успевала за целью: недобор "
              f"{100 * (1 - turns.turns / turns.expected_turns):.0f} %. Запись годна, "
              f"но заявленной частоте не верьте — верьте этой строке")
    for line in _cost_lines(turns, cost,
                            fps=float(profile.parameters["capture_fps"])):
        print(line)
    if interrupted:
        print("прервано вами; запись закрыта и годна к приёму — отметка о прерывании "
              "лежит в журнале")
    if unchanged and not written:
        print("ни одного изменившегося кадра: если это была запись неподвижности — "
              "так и должно быть", file=sys.stderr)
    if audio_blocks:
        print(f"блоков звука: {audio_blocks}")
    print(f"слой-инициатор: {layer}, линия {lid} (своя, не контейнерная)")
    corpus = target.resolve().parent / "corpus"
    print(f"Дальше: harness ingest {show(target)} --corpus {show(corpus)} --kind ВИД")
    print(f"        какие бывают виды: harness ingest --list")
    return 0


def _cost_lines(turns: Any, cost: dict[str, Any], *, fps: float) -> list[str]:
    """Разбивка времени по стадиям и приговор по **бюджету кадра**. `TASK-18`, пункт 3.

    Приговор выносится сравнением с бюджетом `1000 / capture_fps`, а не сравнением стадий
    между собой. Прежний вывод сравнивал стадии («ждал кадр 3.1, писал 0.8 → узкое место в
    захвате») и потому находил узкое место **всегда**: у двух чисел одно непременно больше.
    3.1 мс при бюджете 33 мс — десятикратный запас, а не узкое место.
    """
    from .cost import verdict as _verdict

    st = turns.stages()
    if not st["turns"]:
        return []
    frames_cost = cost.get("frames") or {}
    rows = [
        f"на что ушло время (мс на оборот из {st['turns']}): "
        f"ждал кадр {st['capture_ms']:.1f}, писал {st['record_ms']:.1f}",
        "  " + _verdict(st, fps=fps).text(),
    ]
    if frames_cost.get("calls"):
        rows.append(
            f"  внутри записи кадра: разность {frames_cost['delta_ms_per_call']:.1f}, "
            f"сжатие {frames_cost['compress_ms_per_call']:.1f}, "
            f"файл {frames_cost['write_ms_per_call']:.1f}"
            + (f"; сжатие в {frames_cost['ratio']:.1f} раза"
               if frames_cost.get("ratio") else ""))
        rows.append(
            f"  ключевых кадров {frames_cost['keyframes']} из {frames_cost['calls']}")
        # Совет про настройку даётся только тогда, когда бюджет **превышен** и превышен
        # записью, а не ожиданием кадра. Иначе совет крутить сжатие приходит на запись с
        # десятикратным запасом — и это не совет, а шум.
        v = _verdict(st, fps=fps)
        inside = (frames_cost["delta_ms_per_call"] + frames_cost["compress_ms_per_call"]
                  + frames_cost["write_ms_per_call"])
        if (not v.fits and st["record_ms"] >= st["capture_ms"] and inside > 0
                and frames_cost["compress_ms_per_call"] / inside > 0.5):
            rows.append(
                f"  сжатие — {frames_cost['compress_ms_per_call'] / inside:.0%} времени "
                "записи кадра. Дешевле всего frame_compress_level "
                "(замер: docs/measurements/record_cost.json). Это меняет profile_hash")
    if st["change_share"] is not None:
        # Доля изменившихся кадров — то, чего не хватало, чтобы понять чужую запись.
        # 6.5 кадр/с и 246 кадр/с сняты на разных экранах, и без этой доли они несравнимы.
        rows.append(
            f"  изменившихся кадров {st['change_share']:.0%} оборотов: "
            + ("экран почти не менялся, и стоимость кадра здесь не показательна — "
               "для стоимости нужен движущийся экран (harness cost)"
               if st["change_share"] < 0.5 else
               "экран менялся, стоимость кадра показательна"))
    return rows


def _frame_size_for_cost(width: int | None, height: int | None) -> tuple[int, int, str]:
    """Размер кадра для замера стоимости — и **откуда он взят**.

    Порядок: сказанное ключами, затем настоящий экран этой машины, затем 1920×1080 как
    заведомо крупный случай. Из профиля не берётся вовсе: там 320×180 — размер
    синтетического мира, и посчитанная по нему стоимость меньше настоящей в 36 раз, потому
    что сжатие линейно по числу пикселей. Это ровно та ложь профиля о кадре, из-за которой
    расчёт места ошибался в тридцать шесть раз (TASK-10, часть 4).
    """
    if width and height:
        return int(width), int(height), "задано ключами"
    from .capture.select import open_screen
    from .core.profile import MILESTONE_0
    from .machine import detect as detect_machine

    try:
        choice = open_screen(detect_machine(),
                             gray=MILESTONE_0.structural["frame_format"] == "gray8")
        src = choice.source
        if src is not None:
            try:
                frame = src.read()
                for _ in range(5):
                    if frame is not None and not isinstance(frame, bool) and \
                            getattr(frame, "image", None) is not None:
                        break
                    frame = src.read()
                if frame is not None and getattr(frame, "image", None) is not None:
                    h, w = frame.image.shape[:2]
                    return (int(width or w), int(height or h),
                            f"снято с вашего экрана механизмом {choice.chosen.name}")
            finally:
                src.stop()
    except Exception:
        # Дисплея нет или захват не открылся — это не повод не считать: считаем на
        # заведомо крупном кадре и **говорим**, что размер не с этой машины.
        pass
    return int(width or 1920), int(height or 1080), "дисплея нет, взят монитор 1920×1080"


def cmd_panel(args: argparse.Namespace) -> int:
    """Поднять панель и открыть браузер. Одна команда на всё. `TASK-19`.

    Дальше оператору не нужен терминал: запись запускается с экрана 8, ход виден там же,
    «Прервать» закрывает сессию тем же путём, что `Ctrl+C`, а фикстура после записи
    пересобирается сама.
    """
    from .panelserver import serve

    return serve(port=args.port, open_browser=not args.no_browser,
                 live_root=args.live_root, verbose=args.verbose)


def cmd_cost(args: argparse.Namespace) -> int:
    """Сколько стоит кадр **на этой машине** при разной доле изменения экрана.

    `TASK-18`, пункт 4. Загадка 6.5 кадр/с была обойдена, а не решена: во втором прогоне
    оператора экран почти не менялся — 271 отметка «без изменений» из 300 оборотов, — и 14
    КиБ на кадр не имеют отношения к 1.37 МиБ из первого. Ждать, когда условия повторятся
    сами, незачем: доля изменения задаётся здесь и объявляется в таблице.

    Дисплей не нужен: мерится не захват, а то, что делают с кадром после него. Поэтому
    команда работает и там, где записывать нечего.
    """
    from .cost import scan
    from .core.profile import MILESTONE_0

    p = MILESTONE_0.parameters
    fps = args.fps or float(p["capture_fps"])
    # Размер кадра берётся **у экрана**, а не из профиля. В профиле 320×180 — это размер
    # синтетического мира, и стоимость кадра, посчитанная по нему, к живой записи отношения
    # не имеет: сжатие линейно по числу пикселей, а разница здесь в 36 раз.
    w, h, whence = _frame_size_for_cost(args.width, args.height)
    level = args.compress_level if args.compress_level is not None else \
        int(p["frame_compress_level"])
    keyframe = args.keyframe or int(p["frame_keyframe_interval"])

    # Разрезы по размеру и по рычагам печатают свой заголовок: у них другие столбцы.
    # Общий заголовок здесь печатался бы вторым и противоречил бы их первому.
    if getattr(args, "sizes", False) or getattr(args, "levels", False):
        return _cost_by_area(args, fps=fps, level=level, keyframe=keyframe)
    # При `--json` на выходе **только** json: заголовок перед ним делает вывод
    # непарсимым, а ключ `--json` заводится ровно для того, чтобы его парсили.
    if not args.json:
        print(f"кадр {w}×{h} ({whence}), уровень сжатия {level}, ключевой каждые "
              f"{keyframe}, бюджет кадра при {fps:g} кадр/с — {1000 / fps:.1f} мс")
        print(f"кадров на точку: {args.frames}; экран синтетический (рабочий стол), "
              "и это сказано потому, что на своём экране цифры будут свои")
    rows = scan(width=w, height=h, frames=args.frames, level=level, keyframe=keyframe,
                fps=fps)
    if args.json:
        _print_json({"width": w, "height": h, "fps": fps, "compress_level": level,
                     "keyframe_interval": keyframe, "rows": rows})
        return 0
    print()
    print("  изменилось   разность  сжатие   файл    всего   потолок   на кадр   что это")
    for r in rows:
        mark = "" if r["fits_budget"] else "  ← бюджет превышен"
        print(f"  {r['change_measured']:>8.0%}   {r['delta_ms']:7.1f}  "
              f"{r['compress_ms']:6.1f}  {r['write_ms']:5.1f}  {r['total_ms']:7.1f}  "
              f"{r['max_fps']:6.1f}/с  {r['kib_per_frame']:7.1f} КиБ   "
              f"{r['meaning']}{mark}")
    worst = max(rows, key=lambda r: r["total_ms"])
    print()
    print(f"худший случай — {worst['meaning']}: {worst['total_ms']:.1f} мс на кадр, "
          f"потолок {worst['max_fps']:.1f} кадр/с, {worst['kib_per_frame']:.1f} КиБ/кадр")
    if not worst["fits_budget"]:
        print(f"в бюджет {1000 / fps:.1f} мс он не влезает: на таком экране запись "
              f"будет медленнее цели, и это свойство машины, а не поломка")
    else:
        print(f"в бюджет {1000 / fps:.1f} мс влезает даже он")
    print("\nСравнить с живой записью: строка «на что ушло время» в конце harness record")
    return 0


def _cost_by_area(args: argparse.Namespace, *, fps: float, level: int,
                  keyframe: int) -> int:
    """Стоимость по размеру области и сравнение двух рычагов. TASK-21, часть 3.

    Отдельной функцией, а не ветками внутри `cmd_cost`: у этих таблиц другие столбцы и
    другой вывод. Ветки в одной печати уже дали бы третий формат, склеенный из двух.
    """
    from .cost import LEVER_CASES, levers, scan_sizes

    rows = scan_sizes(frames=args.frames, level=level, keyframe=keyframe, fps=fps)
    lever_rows = levers(frames=args.frames, keyframe=keyframe) \
        if getattr(args, "levels", False) else []
    if args.json:
        _print_json({"fps": fps, "compress_level": level, "keyframe_interval": keyframe,
                     "frames_per_size": args.frames, "sizes": rows,
                     "levers": lever_rows, "lever_cases": list(LEVER_CASES)})
        return 0

    print(f"уровень сжатия {level}, ключевой каждые {keyframe}, бюджет кадра при "
          f"{fps:g} кадр/с — {1000 / fps:.1f} мс")
    print(f"кадров на размер: {args.frames}; экран синтетический (рабочий стол), доля "
          "изменения полная — размеры сравниваются там, где бюджет и трещит")
    print()
    print("       кадр   пикселей   стоимость   дешевле    на кадр   потолок   что это")
    for r in rows:
        print(f"  {r['width']:>5}×{r['height']:<5} {r['pixel_share']:>7.1%}   "
              f"{r['cost_share']:>8.1%}   {r['cheaper_times']:>6.2f}×  "
              f"{r['kib_per_frame']:>7.1f} КиБ  {r['max_fps']:>6.0f}/с   {r['meaning']}")
    win = next((r for r in rows if r["width"] == 960), rows[-1])
    print()
    print(f"окно {win['width']}×{win['height']} дешевле полного кадра в "
          f"{win['cheaper_times']:.2f} раза при {win['pixel_share']:.0%} пикселей — то "
          "есть стоимость падает по пикселям, а не быстрее и не медленнее")
    if lever_rows:
        big = lever_rows[0]
        print()
        print("Два рычага на одной опоре: уменьшить область или понизить сжатие.")
        print("       рычаг                 мс/кадр   дешевле      на кадр   потолок")
        for r in lever_rows:
            print(f"  {r['label']:<24} {r['total_ms']:>7.2f}  {r['cheaper_times']:>6.2f}×  "
                  f"{r['kib_per_frame']:>8.1f} КиБ  {r['max_fps']:>6.0f}/с")
        area = next((r for r in lever_rows if r["lever"] == "область"), None)
        lvl = next((r for r in lever_rows if r["lever"] == "сжатие"), None)
        if area and lvl:
            print()
            print(f"Уменьшение области выигрывает по обеим величинам сразу: "
                  f"{area['cheaper_times']:.2f}× по времени и "
                  f"{big['kib_per_frame'] / area['kib_per_frame']:.2f}× по диску. "
                  f"Понижение уровня даёт {lvl['cheaper_times']:.2f}× по времени и "
                  f"**платит** диском: {lvl['kib_per_frame'] / big['kib_per_frame']:.2f}× "
                  "байт на кадр")
    print()
    print("Переносимы **отношения**, а не сами миллисекунды: на своём содержимом сжатие "
          "будет своё (у оператора 2.4× против ~35× здесь).")
    if lever_rows:
        print("Отдельно про «без сжатия»: в контейнере он выходит самым быстрым, потому "
              "что запись идёт в файловую систему контейнера. На диске 2025 КиБ на кадр "
              "при 30 кадр/с — это 59 МиБ/с непрерывной записи, и там рычаг развернётся. "
              "Время стадии «запись», снятое здесь, к вашему диску отношения не имеет.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Осмотр машины. Код возврата: 0 — записывать можно, 1 — нельзя.

    Ненулевой код именно при «нельзя», а не при ошибке самой команды: по нему можно
    поставить проверку в сценарий, и «машина не готова» будет отличимо от «команда
    сломалась» (для второго кода 2, как у остальных).
    """
    from . import doctor

    rep = doctor.run(path=args.path)
    if args.json:
        _print_json(rep.as_dict())
    else:
        print(rep.render_text())
    return 0 if rep.can_record else 1


def cmd_selftest(args: argparse.Namespace) -> int:
    """Десять секунд захвата и проверка результата. 0 — всё прошло, 1 — сломалось."""
    import tempfile

    from . import selftest

    root = args.path
    tmp = None
    if root is None:
        tmp = tempfile.mkdtemp(prefix="harness-selftest-")
        root = Path(tmp) / "session"
    res = selftest.run(Path(root), seconds=args.seconds,
                       with_audio=not args.no_audio,
                       expect_change=not args.static)
    if args.json:
        _print_json(res.as_dict())
    else:
        print(res.render_text())
        if tmp and res.ok:
            print(f"\nПробная запись во временном каталоге: {root}\n"
                  f"Её можно удалить: rm -rf {tmp}")
    return 0 if res.ok else 1


def cmd_params(args: argparse.Namespace) -> int:
    """Мёртвые и однобокие параметры. 0 — дефектов нет, 1 — есть.

    Ненулевой код при найденных дефектах, а не при ошибке команды: по нему ставится
    проверка в сценарий. Число дефектов не должно расти незамеченным (TASK-10, часть 3),
    а незамеченным оно растёт ровно тогда, когда его никто не спрашивает.
    """
    import tempfile

    from . import params

    runtime = None
    if args.run:
        with tempfile.TemporaryDirectory(prefix="harness-params-") as d:
            runtime = params.runtime_probe(steps=args.steps, root=Path(d) / "session")
        print(f"прогон со счётчиками: прочитано ключей {len(runtime)} "
              f"за {args.steps} шагов лепета\n")
    findings = params.audit(runtime=runtime)
    if args.json:
        _print_json({"findings": [f.as_dict() for f in findings],
                     "counts": params.counts(findings),
                     "defects": params.defect_count(findings)})
    else:
        print(params.render_table(findings, only_defects=not args.all))
    return 0 if params.defect_count(findings) == 0 else 1


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

    from .corpus.live import METHODS, PARALLAX_REFERENCE, method_comparable

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
        # `None` означает «взять из единственного источника», а не «нет числа»:
        # `compare_to_synthetic` сам сходит за опорным. Подстановка литерала здесь
        # вернула бы второе место, где это число написано.
        baseline = (args.synthetic_median if method == "arbiter"
                    else (args.synthetic_corpus_median
                          if args.synthetic_corpus_median is not None
                          else PARALLAX_REFERENCE["value"]))
        cmp = compare_to_synthetic(scores, synthetic_median=baseline)
        print(f"путь «{method}»: {method_comparable(method)}")
        print(f"  {cmp['verdict']}")
        # Исход предрегистрированной проверки — по каждой записи и словами. Он лежал
        # только в `--json`, то есть оператор, читающий терминал, видел разницу с опорным
        # числом и **не** видел, считается ли она опровержением.
        if cmp.get("expectation_text"):
            print(f"  {cmp['expectation_text']}")
        for check in cmp.get("checks", ()):
            print(f"  {check['kind']}: {check['outcome']} — {check['why']}")
        if cmp.get("unscored"):
            print(f"  без разметки, в сравнение не вошли: {cmp['unscored']}")
        print()
        out[method] = {"scores": [s.as_dict() for s in scores], "comparison": cmp}
    if args.json:
        _print_json(out)
    return 0


def _synthetic_help() -> str:
    """Опорное число арбитра для строки помощи. Читается, а не пишется."""
    from .corpus.live import synthetic_text

    return synthetic_text()


def _parallax_help() -> str:
    """Опорное число одиночного разделителя — с единицей и `n`, как и всякое другое."""
    from .corpus.live import PARALLAX_REFERENCE

    r = PARALLAX_REFERENCE
    return (f"IoU {r['value']:.4f} (единица усреднения — {r['unit']}, n={r['n']}; "
            f"опорным для ожидания не является)")


def _source_kinds() -> tuple[str, ...]:
    """Виды источников — из одного места (`capture.source.KINDS`), а не списком здесь.

    Второй список тех же значений расходится с первым молча: ровно так трижды разошлось
    знание о Wayland между реестром механизмов и самим механизмом.
    """
    from .capture.source import KINDS

    return KINDS


def build_parser() -> argparse.ArgumentParser:
    """Разбор аргументов отдельно от запуска: иначе список команд не проверить.

    Нужно для теста, который сверяет `SETUP.md` с действительностью: документ,
    обещающий несуществующую команду, отправляет оператора читать код — то есть
    делает ровно то, от чего этот документ должен избавлять.
    """
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
    bb.add_argument("path", type=_path_arg)
    bb.add_argument("--seed", type=int, default=0)
    bb.add_argument("--steps", type=int, default=1500)
    bb.add_argument("--note", default=None)
    bb.add_argument("--compare-truth", action="store_true",
                    help="сверить с истиной мира из отладочного потока")
    bb.set_defaults(fn=cmd_babble)

    g = sub.add_parser("gen-corpus", help="записать синтетическую сессию")
    g.add_argument("path", type=_path_arg)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--frames", type=int, default=0, help="примерное число кадров")
    g.add_argument("--width", type=int, default=320)
    g.add_argument("--height", type=int, default=180)
    g.add_argument("--no-audio", action="store_true")
    g.add_argument("--note", default=None)
    g.set_defaults(fn=cmd_gen_corpus)

    lp = sub.add_parser("loop", help="замкнутый круг: драйв → цель → пробы → тест")
    lp.add_argument("path", type=_path_arg)
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
    v.add_argument("path", type=_path_arg)
    v.set_defaults(fn=cmd_verify)

    j = sub.add_parser("journal", help="сводка по журналу")
    j.add_argument("path", type=_path_arg)
    j.set_defaults(fn=cmd_journal)

    r = sub.add_parser("replay", help="пройти запись покадрово")
    r.add_argument("path", type=_path_arg)
    r.add_argument("--at", type=int, default=None, help="показать один кадр")
    r.set_defaults(fn=cmd_replay)

    sw = sub.add_parser("selfworld", help="разделить экранный и мировой слои (0.6)")
    sw.add_argument("path", type=_path_arg)
    sw.add_argument("--dump-mask", type=_path_arg, default=None)
    sw.set_defaults(fn=cmd_selfworld)

    rp = sub.add_parser("report", help="показатели по журналу рядом со словами "
                                       "агента о себе")
    rp.add_argument("path", type=_path_arg)
    rp.add_argument("--beliefs", action="store_true",
                    help="пересобрать убеждения и карту тела (медленнее)")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(fn=cmd_report)

    stt = sub.add_parser("status", help="что готово, что нет и почему — по коду")
    stt.add_argument("--tests", action="store_true",
                     help="прогнать тесты и показать их итог")
    stt.add_argument("--md", type=_path_arg, default=None, help="записать markdown")
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
    rec.add_argument("path", type=_path_arg, nargs="?", default=Path("."))
    # `--turns` — точное имя: предел считает **обороты**, то есть взгляды на экран, а не
    # изменившиеся кадры. `--frames` оставлен синонимом, потому что он уже написан в
    # SETUP.md и в чужих заметках, но имя, расходящееся со смыслом, — это то, на чём
    # спотыкаются дважды (TASK-17, пункт 1).
    rec.add_argument("--turns", "--frames", dest="turns", type=int, default=None,
                     help="сколько оборотов записать (взглядов на экран, включая "
                          "отметки «без изменений»); срок задаётся --seconds")
    # Секунды, а не только кадры: «3 мин» приходилось умножать на частоту в голове,
    # и это ровно то место, где человек ошибается на порядок и получает запись на
    # четыре секунды вместо трёх минут.
    rec.add_argument("--seconds", type=float, default=None,
                     help="сколько секунд записывать (удобнее, чем кадры)")
    rec.add_argument("--note", default=None)
    rec.add_argument("--actor", choices=("human", "none"), default="none",
                     help="кто действует: human для демонстрации оператором")
    # Вид записи нужен **до** приёма, а не только при `ingest`: от него зависит, куда
    # идёт ход записи. На записи неподвижности мигающая строка в терминале попадёт в
    # кадр как изменение, а изменение там — измеряемая величина.
    rec.add_argument("--kind", default=None, choices=tuple(_record_kinds()),
                     help="что записываете; от вида зависит канал хода записи "
                          "(для stillness ход идёт в файл, а не на экран)")
    # Источник захвата. Вид — структурный переключатель профиля, поэтому запись окна
    # попадает в **другую ветку** журнала, чем запись экрана, и их числа не сводятся в
    # одну медиану (TASK-21, часть 1).
    rec.add_argument("--source", dest="source_kind", default="display",
                     choices=_source_kinds(),
                     help="что попадает в кадр: display — экран целиком, window — одно "
                          "окно (границы приходят от системы, то есть агент получает "
                          "готовую рамку), region — прямоугольник от исследователя. "
                          "Меняет structure_hash: записи разных источников несравнимы")
    rec.add_argument("--window", default="",
                     help="часть заголовка окна для --source window. Заголовок нужен, "
                          "чтобы указать окно, и в кадр агента он не попадает")
    rec.add_argument("--region", nargs=4, type=int, default=None,
                     metavar=("ЛЕВО", "ВЕРХ", "ШИРИНА", "ВЫСОТА"),
                     help="рамка для --source region в экранных координатах")
    rec.add_argument("--progress", default=None, choices=("line", "file", "none"),
                     help="куда печатать ход: строкой на месте, в файл рядом с "
                          "сессией или никуда. По умолчанию — из профиля")
    rec.add_argument("--plan", action="store_true",
                     help="только напечатать, что записать, и ничего не писать")
    rec.add_argument("--set", dest="plan_set", default="minimal",
                     choices=("minimal", "full"),
                     help="какой набор записей печатать при --plan: minimal без "
                          "игры (по умолчанию) или full с игрой")
    rec.add_argument("--audio-device", default=None,
                     help="петлевое устройство звука; harness doctor покажет, какое")
    rec.add_argument("--no-audio", action="store_true",
                     help="не писать звук вовсе (записи он не блокирует)")
    rec.set_defaults(fn=cmd_record)

    pa = sub.add_parser("panel", help="открыть панель в браузере и записывать из неё")
    pa.add_argument("--port", type=int, default=8765,
                    help="порт на 127.0.0.1; наружу ничего не открывается")
    pa.add_argument("--no-browser", action="store_true",
                    help="не открывать браузер самому")
    pa.add_argument("--live-root", type=_path_arg, default=None,
                    help="куда класть записи; по умолчанию ~/harness-live")
    pa.add_argument("--verbose", action="store_true",
                    help="печатать обращения браузера")
    pa.set_defaults(fn=cmd_panel)

    co = sub.add_parser("cost", help="сколько стоит кадр на этой машине")
    co.add_argument("--frames", type=int, default=30,
                    help="кадров на каждую долю изменения")
    co.add_argument("--width", type=int, default=None)
    co.add_argument("--height", type=int, default=None)
    co.add_argument("--fps", type=float, default=None,
                    help="целевая частота, по ней считается бюджет кадра")
    co.add_argument("--compress-level", type=int, default=None,
                    dest="compress_level")
    co.add_argument("--keyframe", type=int, default=None,
                    help="интервал ключевых кадров")
    # Второй разрез той же стоимости: по размеру области, а не по доле изменения. Нужен
    # затем, что захват окна — рычаг бюджета, и «во сколько раз дешевле» имеет численный
    # ответ (TASK-21, часть 3).
    co.add_argument("--sizes", action="store_true",
                    help="разрез по размеру кадра: во сколько раз дешевле окно, чем "
                         "экран целиком. Доля изменения при этом полная — размеры "
                         "сравниваются там, где бюджет и трещит")
    co.add_argument("--levels", action="store_true",
                    help="сравнить два рычага на одной опоре: уменьшить область против "
                         "понизить уровень сжатия")
    co.add_argument("--json", action="store_true")
    co.set_defaults(fn=cmd_cost)

    doc = sub.add_parser("doctor", help="что на этой машине мешает записывать")
    doc.add_argument("--path", type=_path_arg, default=None,
                     help="куда собираетесь писать: по нему считается место")
    doc.add_argument("--json", action="store_true")
    doc.set_defaults(fn=cmd_doctor)

    st = sub.add_parser("selftest", help="десять секунд захвата с проверкой результата")
    st.add_argument("path", type=_path_arg, nargs="?", default=None,
                    help="куда положить пробную запись; по умолчанию во временный каталог")
    st.add_argument("--seconds", type=float, default=10.0)
    st.add_argument("--no-audio", action="store_true",
                    help="без звука: он записи не блокирует")
    st.add_argument("--static", action="store_true",
                    help="сцена статична (как в записи «неподвижность»): совпадение "
                         "кадров здесь правильный ответ, а не отказ")
    st.add_argument("--json", action="store_true")
    st.set_defaults(fn=cmd_selftest)

    pm = sub.add_parser("params", help="мёртвые и однобокие параметры (инвариант 30)")
    pm.add_argument("--all", action="store_true",
                    help="все настройки, а не только дефектные")
    pm.add_argument("--run", action="store_true",
                    help="второй способ: короткий прогон со счётчиками чтений")
    pm.add_argument("--steps", type=int, default=40,
                    help="шагов лепета в прогоне со счётчиками")
    pm.add_argument("--json", action="store_true")
    pm.set_defaults(fn=cmd_params)

    ing = sub.add_parser("ingest", help="принять запись с чужой машины в корпус")
    ing.add_argument("path", type=_path_arg, nargs="?", default=Path("."))
    ing.add_argument("--corpus", type=_path_arg, default=Path("corpus/live"))
    ing.add_argument("--kind", default="play",
                     help="вид записи; --list покажет все")
    ing.add_argument("--in-place", action="store_true",
                     help="не копировать, только проверить на месте")
    ing.add_argument("--list", action="store_true", help="какие виды записей нужны")
    ing.add_argument("--json", action="store_true")
    ing.set_defaults(fn=cmd_ingest)

    mk = sub.add_parser("mark", help="отметить область обрамления на живой записи")
    mk.add_argument("path", type=_path_arg)
    mk.add_argument("--screen", nargs=4, action="append", metavar=("ВЕРХ", "ЛЕВО", "ВЫСОТА", "ШИРИНА"),
                    help="прямоугольник экранного слоя; можно указать несколько раз")
    mk.add_argument("--note", default=None)
    mk.set_defaults(fn=cmd_mark)

    bl = sub.add_parser("bench-live", help="прогнать тот же код по живым записям")
    bl.add_argument("path", type=_path_arg, default=Path("corpus/live"), nargs="?")
    bl.add_argument("--method", choices=("arbiter", "parallax", "both"),
                    default="both",
                    help="какой из двух существующих путей прогнать")
    # Оба опорных числа — из `corpus.live`, а не литералами здесь. Литерал в разборе
    # аргументов был бы третьим местом, где живёт «IoU на синтетике», а двух уже хватило,
    # чтобы одно имя означало два разных утверждения (TASK-11, часть 1).
    bl.add_argument("--synthetic-median", type=float, default=None,
                    help=f"опорное число арбитра; по умолчанию {_synthetic_help()}")
    bl.add_argument("--synthetic-corpus-median", type=float, default=None,
                    help=f"опорное число одиночного разделителя; по умолчанию "
                         f"{_parallax_help()}")
    bl.add_argument("--json", action="store_true")
    bl.set_defaults(fn=cmd_bench_live)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
