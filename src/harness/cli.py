"""Командная строка харнесса.

    harness backends                     что доступно на этой машине
    harness profile                      профиль вехи 0 и его хеши
    harness gen-corpus PATH [--seed N]   записать синтетическую сессию
    harness verify PATH                  проверить целостность записи
    harness journal PATH                 сводка по журналу
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


def cmd_record(args: argparse.Namespace) -> int:
    from .capture.base import BackendUnavailable
    from .capture.screen import ScreenCapture
    from .core.profile import MILESTONE_0
    from .session import Recorder

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
                rec.record_frame(frame.image, t_world=frame.t_world)
                written += 1
    finally:
        cap.stop()
    print(f"записано кадров: {written} → {args.path}")
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

    rec = sub.add_parser("record", help="запись с живого экрана")
    rec.add_argument("path", type=Path)
    rec.add_argument("--frames", type=int, default=300)
    rec.add_argument("--note", default=None)
    rec.set_defaults(fn=cmd_record)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
