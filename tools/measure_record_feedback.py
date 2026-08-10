"""Замер по TASK-16: что оператор видит во время записи и что остаётся после Ctrl+C.

Правка поведения обязана предъявить сдвиг числа (инвариант 25). Здесь два плеча, и оба
прогоняются **на одном и том же поддельном источнике кадров**, без дисплея:

- `до` — цикл в том виде, в каком он был: без хода записи и без перехвата прерывания.
  Воспроизводится здесь функцией `old_run_turns`, а не берётся из истории: сравнивать
  надо поведение, а не коммиты, и старое поведение должно быть видно рядом с новым;
- `после` — `capture.record_loop.run_turns`, то есть текущий код.

**Единица независимости — прогон записи.** Не оборот: обороты внутри одного прогона
зависят друг от друга (прерывание на пятом обороте определяет судьбу всех следующих). Не
утверждение: утверждений четыре, и каждое считается по своим прогонам.

Четыре величины, и каждая отвечает на свой вопрос оператора:

1. **строк о ходе за минуту записи** — «идёт ли запись вообще»;
2. **доля прерванных прогонов с отметкой «прервано на N из M»** — «что у меня осталось»;
3. **доля прерванных прогонов, которые открываются и проходят verify** — «годно ли это»;
4. **доля повторных запусков, получивших команду вместо «каталог не пуст»** — «что делать
   дальше».

Запуск: `python3 tools/measure_record_feedback.py`. Идёт около минуты. Результат —
`docs/measurements/record_feedback.json`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                          # noqa: E402

from harness.capture.base import UNCHANGED, Frame           # noqa: E402
from harness.capture.record_loop import run_turns           # noqa: E402
from harness.core.journal import Actor, ActorLayer          # noqa: E402
from harness.core.profile import from_schema                # noqa: E402
from harness.progress import Progress                       # noqa: E402
from harness.session import Recorder, Session, describe_existing   # noqa: E402

W, H = 96, 64

#: Дефект, найденный **этим замером** и не относящийся ни к одному из плеч: `verify`
#: считал отметку «без изменений» кадром хранилища. Текущим кодом не воспроизводится —
#: он исправлен, — поэтому числа записаны здесь.
#:
#: Важное в нём то, что он не про прерывание вовсе: полная запись со статикой тоже
#: объявлялась сломанной. Первой такой была бы «неподвижность» — опорная запись
#: минимального набора, где отметок больше, чем кадров.
VERIFY_DEFECT: dict[str, Any] = {
    "что": "verify сравнивал число записей журнала вида FRAME с числом блоков в "
           "хранилище кадров. Отметка «без изменений» — запись журнала, ссылающаяся на "
           "уже лежащий блок: нового блока она не добавляет",
    "кем найдено": "этот замер: 17 из 19 прерванных прогонов не проходили verify в "
                   "**обоих** плечах, то есть дефект не имел отношения к правке",
    "до": {"прерванных проходят verify": 2, "прерванных всего": 19,
           "полная запись со статикой": "не проходила: «кадров в журнале 30, в "
                                        "хранилище 25» при 5 отметках"},
    "после": {"прерванных проходят verify": 19, "прерванных всего": 19,
              "полная запись со статикой": "проходит: 49 новых кадров, 11 отметок, "
                                           "49 блоков"},
}


class Screen:
    """Поддельный экран: кадры, статика и прерывание в заданный оборот.

    Захват он не изображает и на вопрос «работает ли захват» не отвечает вовсе. Он
    отвечает на другой вопрос — что делает цикл записи, когда экран отвечает так или
    иначе, — и только так этот цикл проверяется без дисплея.
    """

    name = "замер-экран"

    def __init__(self, *, interrupt_at: int | None, still_every: int = 5) -> None:
        self.interrupt_at = interrupt_at
        self.still_every = still_every
        self.reads = 0

    def first(self) -> Frame:
        return Frame(np.full((H, W), 11, dtype=np.uint8), 0, 0)

    def read(self):
        self.reads += 1
        if self.interrupt_at is not None and self.reads >= self.interrupt_at:
            raise KeyboardInterrupt
        if self.still_every and self.reads % self.still_every == 0:
            return UNCHANGED
        return Frame(np.full((H, W), (self.reads * 17) % 251, dtype=np.uint8),
                     self.reads, 0)

    def stop(self) -> None: ...


class Clock:
    """Часы под управлением замера: настоящие сделали бы число строк гадательным."""

    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.turn = 0

    def __call__(self) -> float:
        return self.turn / self.fps


def old_run_turns(rec: Any, *, source: Any, frames: int, first: Any) -> dict[str, Any]:
    """Цикл в том виде, в каком он был до TASK-16. Ход не печатается, Ctrl+C не ловится.

    Прерывание здесь **выходит наружу** — ровно как выходило у оператора из середины
    `zlib.compress`. Вызывающий его перехватывает, чтобы замер не умер, но записать в
    журнал уже нечего: `with` закрыл хранилища на выходе.
    """
    written = unchanged = 0
    while written + unchanged < frames:
        frame = first if written + unchanged == 0 else source.read()
        if frame is UNCHANGED:
            if written == 0:
                rec.record_gap("unchanged_before_first_frame", {"turns": unchanged + 1})
                unchanged += 1
                continue
            rec.record_unchanged(actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN)
            unchanged += 1
            continue
        if frame is None:
            rec.record_gap("source_ended", {"after_frames": written})
            break
        rec.record_frame(frame.image, t_world=frame.t_world,
                         actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN)
        written += 1
    return {"written": written, "unchanged": unchanged}


def one_run(root: Path, *, arm: str, frames: int, interrupt_at: int | None,
            fps: float = 30.0) -> dict[str, Any]:
    """Один прогон записи. Возвращает, что оператор увидел и что осталось на диске."""
    profile = from_schema("ЗАМЕР-запись", capture_width=W, capture_height=H)
    screen = Screen(interrupt_at=interrupt_at)
    lines: list[str] = []
    interrupted = False

    if arm == "после":
        clock = Clock(fps)
        prog = Progress.from_profile(profile, total_turns=frames, path=root,
                                     channel="none", now=clock)

        real_update = prog.update

        def update(**kw):
            # Часы двигает замер, а не настоящее время: иначе число строк зависело бы от
            # того, насколько занята машина, и сравнивать плечи было бы нечем.
            clock.turn = int(kw.get("written", 0)) + int(kw.get("unchanged", 0))
            got = real_update(**kw)
            if got is not None:
                lines.append(got)
            return got

        prog.update = update            # type: ignore[method-assign]
        with Recorder(root, profile=profile, source=screen.name, synthetic=False,
                      lineage_id="lin-замер") as rec:
            turns = run_turns(rec, source=screen, frames=frames, first=screen.first(),
                              actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN,
                              progress=prog)
        interrupted = turns.interrupted
        got = {"written": turns.written, "unchanged": turns.unchanged}
    else:
        try:
            with Recorder(root, profile=profile, source=screen.name, synthetic=False,
                          lineage_id="lin-замер") as rec:
                got = old_run_turns(rec, source=screen, frames=frames,
                                    first=screen.first())
        except KeyboardInterrupt:
            interrupted = True
            got = {"written": None, "unchanged": None}

    out: dict[str, Any] = {"arm": arm, "interrupt_at": interrupt_at,
                           "interrupted": interrupted, "progress_lines": len(lines),
                           **got}

    # Что осталось на диске: открывается ли запись, цела ли она, есть ли отметка.
    try:
        with Session.open(root) as s:
            out["readable"] = True
            out["verify_ok"] = bool(s.verify()["ok"])
            out["has_mark"] = s.interrupted is not None
            out["entries"] = len(list(s.journal))
    except Exception as e:
        out["readable"] = False
        out["verify_ok"] = False
        out["has_mark"] = False
        out["entries"] = 0
        out["why_unreadable"] = f"{type(e).__name__}: {e}"

    # Повторный запуск в тот же каталог: что получит оператор.
    if arm == "после":
        text = describe_existing(root)
    else:
        # Прежний отказ: одна строка про непустой каталог и ничего больше.
        text = f"{root} не пуст. Сессии не дописываются поверх чужих"
    out["retry_names_a_command"] = "harness record" in text
    out["retry_says_what_is_there"] = ("прервана" in text or "дописана" in text)
    return out


def measure(runs: int, *, seconds: float, fps: float, seed: int) -> dict[str, Any]:
    frames = int(seconds * fps)
    rng = random.Random(seed)
    # Точки прерывания одни и те же для двух плеч: сравниваются плечи, а не жребий.
    points = [None] + [rng.randrange(3, frames - 1) for _ in range(runs - 1)]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="record-feedback-") as tmp:
        base = Path(tmp)
        for arm in ("до", "после"):
            for i, point in enumerate(points):
                root = base / f"{arm}-{i}"
                rows.append(one_run(root, arm=arm, frames=frames,
                                    interrupt_at=point, fps=fps))
    return {"rows": rows, "frames": frames, "seconds": seconds, "fps": fps,
            "seed": seed}


def summarise(data: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in ("до", "после"):
        rows = [r for r in data["rows"] if r["arm"] == arm]
        stopped = [r for r in rows if r["interrupted"]]
        whole = [r for r in rows if not r["interrupted"]]
        out[arm] = {
            "прогонов": len(rows),
            "из них прерванных": len(stopped),
            "строк о ходе за запись": (max(r["progress_lines"] for r in rows)
                                       if rows else 0),
            "прерванных с отметкой": len([r for r in stopped if r["has_mark"]]),
            "прерванных читаются": len([r for r in stopped if r["readable"]]),
            "прерванных проходят verify": len([r for r in stopped if r["verify_ok"]]),
            "повторный запуск даёт команду": len(
                [r for r in rows if r["retry_names_a_command"]]),
            "повторный запуск говорит, что лежит": len(
                [r for r in rows if r["retry_says_what_is_there"]]),
            "целых прогонов без отметки": len([r for r in whole if not r["has_mark"]]),
        }
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=20, help="прогонов на плечо")
    ap.add_argument("--seconds", type=float, default=2.0,
                    help="длина записи в секундах (кадры считаются по fps)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "record_feedback.json")
    args = ap.parse_args(argv[1:])

    data = measure(args.runs, seconds=args.seconds, fps=args.fps, seed=args.seed)
    summary = summarise(data)

    # Строки о ходе за минуту записи считаются отдельно и **арифметикой из замера**, а не
    # прогоном на минуту: частота печати измерена (строк на N секунд), и растягивать её до
    # шестидесяти секунд — умножение, а не новое наблюдение. Выдавать умножение за замер
    # значило бы приписать себе минуту, которой не было.
    per_run = summary["после"]["строк о ходе за запись"]
    lines_per_minute = round(per_run * 60.0 / args.seconds, 1) if args.seconds else 0.0

    payload = {
        "unit": "прогон записи",
        "why_unit": "обороты внутри прогона зависят друг от друга: прерывание на пятом "
                    "обороте определяет судьбу всех следующих. Утверждений четыре, и "
                    "каждое считается по своим прогонам",
        "runs_per_arm": args.runs,
        "seconds": args.seconds, "fps": args.fps, "frames": data["frames"],
        "summary": summary,
        "verify_defect": VERIFY_DEFECT,
        "lines_per_minute_after": lines_per_minute,
        "lines_per_minute_before": 0.0,
        "arithmetic_note": "строк за минуту — умножение измеренной частоты на 60/секунды "
                           "записи, а не отдельный прогон",
        "rows": data["rows"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"прогонов на плечо: {args.runs}, запись {args.seconds:g} с "
          f"({data['frames']} оборотов), единица независимости: прогон записи")
    for arm in ("до", "после"):
        s = summary[arm]
        print(f"\n{arm}:")
        for k, v in s.items():
            print(f"  {k:38} {v}")
    print(f"\nстрок о ходе за минуту записи: {payload['lines_per_minute_before']:g} → "
          f"{lines_per_minute:g} (арифметика по измеренной частоте)")
    print(f"записано: {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
