"""Одна живая запись: подготовка, прогон, итог. Без печати и без командной строки.

`TASK-19`. Записывать оператор должен из панели, а не из `cmd.exe`, — значит у записи
появился **второй вызывающий**. Дальше есть только два пути, и один из них негодный:
завести в сервере панели свой цикл записи (тогда через месяц он разойдётся с командой, как
трижды разошлось знание о Wayland) — или вынуть запись из `cmd_record` целиком, чтобы
команда и сервер звали одно и то же.

Здесь второй. Модуль ничего не печатает и ничего не решает за вызывающего: он **отказывает
значением** (`Refusal`) и **возвращает числа** (`Outcome`). Печать — дело командной строки,
показ — дело панели.

Что осталось у вызывающих:

- `cli.cmd_record` печатает подготовку, ход и итог в терминал;
- `panelserver` кладёт те же числа в состояние и отдаёт их панели по HTTP.

Прерывание работает в обе стороны: `Ctrl+C` в терминале и кнопка «Прервать» в панели ведут
в одно и то же место — отметку `record_interrupted` в журнале, — потому что оба означают
одно: **человек решил остановиться**, и запись после этого годна.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .capture.base import UNCHANGED
from .capture.record_loop import STOP_REASONS, run_turns
from .core.journal import Actor, ActorLayer
from .core.profile import MILESTONE_0
from .paths import show
from .progress import Progress, dir_bytes
from .session import Recorder, describe_existing


class RecordRefused(RuntimeError):
    """Записывать нельзя, и сказано почему. Отказ — значение, а не печать в поток.

    `code` нужен вызывающим, а не для красоты: командная строка возвращает его кодом
    выхода, панель показывает по нему разное. `hint` — что делать, готовой строкой.
    """

    def __init__(self, why: str, *, code: str, hint: str = "") -> None:
        super().__init__(why)
        self.why = why
        self.code = code
        self.hint = hint

    def as_dict(self) -> dict[str, Any]:
        return {"refused": True, "code": self.code, "why": self.why, "hint": self.hint}


@dataclass(frozen=True, slots=True)
class Plan:
    """Что записывать. Ровно то, что оператор выбирает, и ничего сверх.

    `seconds` и `turns` — взаимоисключающие: срок или счёт оборотов, и совмещать их
    нельзя (TASK-18). Проверяется здесь, а не у каждого вызывающего.
    """

    path: Path
    kind: str | None = None
    seconds: float | None = None
    turns: int | None = None
    actor_human: bool = True
    with_audio: bool = True
    audio_device: str | None = None
    note: str | None = None
    progress_channel: str | None = None
    #: Что попадает в кадр (TASK-21, часть 1). Вид — структурный переключатель профиля;
    #: рамка и заголовок окна — условия одной записи, в хеш не входят.
    source_kind: str = "display"
    window: str = ""
    region: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.seconds is None and self.turns is None:
            raise ValueError("в плане записи нет ни срока, ни числа оборотов")
        from .capture.source import KINDS

        if self.source_kind not in KINDS:
            raise ValueError(f"нет такого источника: {self.source_kind!r}; есть {list(KINDS)}")


@dataclass(slots=True)
class Outcome:
    """Что получилось. Числа те же, что видел оператор, плюс путь и разбивка."""

    path: Path
    mechanism: str
    mechanism_caveat: str
    audio_note: str
    lineage_id: str
    profile_hash: str
    frame: tuple[int, int]
    turns: Any = None                      # capture.record_loop.Turns
    cost: dict[str, Any] = field(default_factory=dict)
    final_line: str = ""
    interrupted: bool = False
    #: Источник: вид, откуда рамка и хеш структуры. Печатается до записи (TASK-21, ч. 1).
    source_kind: str = "display"
    source_why: str = ""
    structure_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        st = self.turns.stages() if self.turns is not None else {}
        return {
            "path": str(self.path), "shown_path": show(self.path),
            "mechanism": self.mechanism, "mechanism_caveat": self.mechanism_caveat,
            "audio": self.audio_note, "lineage_id": self.lineage_id,
            "profile_hash": self.profile_hash,
            "frame": {"width": self.frame[0], "height": self.frame[1]},
            "written": getattr(self.turns, "written", 0),
            "unchanged": getattr(self.turns, "unchanged", 0),
            "stop_reason": getattr(self.turns, "stop_reason", ""),
            "stop_reason_text": STOP_REASONS.get(
                getattr(self.turns, "stop_reason", ""), ""),
            "interrupted": self.interrupted,
            "source": {"kind": self.source_kind, "why": self.source_why,
                       "structure_hash": self.structure_hash},
            "stages": st, "cost": self.cost, "final_line": self.final_line,
        }


def check_free(path: Path) -> None:
    """Каталог занят — отказ **до** открытия захвата.

    Иначе оператор ждёт выбора механизма, первого кадра и звукового входа, чтобы услышать
    про каталог, который был занят с самого начала.
    """
    if path.exists() and any(path.iterdir()):
        raise RecordRefused(f"{show(path)} не пуст", code="busy",
                            hint=describe_existing(path))


def record(plan: Plan, *,
           progress: Progress | None = None,
           on_progress: Callable[[Any], None] | None = None,
           should_stop: Callable[[], bool] | None = None,
           on_ready: Callable[[Outcome], None] | None = None) -> Outcome:
    """Записать по плану. Отказы — исключением `RecordRefused`, результат — `Outcome`.

    `on_ready` вызывается, когда механизм выбран, первый кадр получен и профиль построен, —
    то есть когда уже известно, **что** записывается, но записывать ещё не начали. Панели
    это нужно, чтобы показать механизм и размер кадра сразу, а не после записи.
    """
    from .capture.base import BackendUnavailable
    from .capture.select import open_screen
    from .capture.source import (Region, SourceError, WindowFollowing, resolve,
                                 system_windows)
    from .core.lineage import lineage_id
    from .machine import detect

    check_free(plan.path)

    machine = detect()
    # Что снимать, решается **до** открытия механизма: рамка нужна самому механизму, а
    # отказ «окна такого нет» должен прийти раньше, чем захват что-нибудь запишет.
    try:
        region, target, region_why = resolve(
            plan.source_kind,
            region=Region(*plan.region) if plan.region else None,
            window=plan.window)
    except (SourceError, BackendUnavailable) as e:
        raise RecordRefused(f"источник задан неверно: {e}", code="bad_source",
                            hint="виды источников: harness record --help") from e
    choice = open_screen(machine,
                         gray=MILESTONE_0.structural["frame_format"] == "gray8",
                         region=region.as_tuple() if region else None)
    if choice.source is None:
        raise RecordRefused(f"захват недоступен: {choice.why_text()}", code="no_capture",
                            hint="что именно чинить и какой командой: harness doctor")
    cap = choice.source
    if target is not None:
        # Захват окна — это слежение за окном, а не съёмка прямоугольника, где окно
        # однажды было. Обёртка перепрашивает рамку каждый оборот и превращает
        # закрытие окна в наблюдение, а не в разрыв (часть 6).
        cap = WindowFollowing(cap, target, system_windows())

    audio_src = None
    audio_note = "без звука"
    if plan.with_audio:
        from .capture.screen import LoopbackAudio

        try:
            audio_src = LoopbackAudio(
                rate=int(MILESTONE_0.parameters["audio_rate"]),
                channels=int(MILESTONE_0.structural["audio_channels"]),
                block_ms=float(MILESTONE_0.parameters["audio_block_ms"]),
                device=plan.audio_device)
            audio_src.start()
            audio_note = "со звуком"
        except Exception as e:                 # sounddevice бросает своё
            # Звук не блокирует запись, но и не пропадает молча: доктор велел оператору
            # настроить стерео, и запись без звука после этого выглядела бы как
            # выполненная настройка.
            audio_src = None
            audio_note = f"без звука ({e})"

    try:
        # Первый кадр — до открытия записи: по нему строится профиль. Заявить 320×180 над
        # кадрами 1080p значит записать ложь о себе, и посчитанное по такому профилю
        # ошибётся в тридцать шесть раз.
        first = cap.read()
        while first is UNCHANGED:
            first = cap.read()
        if first is None:
            raise RecordRefused("захват не отдал ни одного кадра", code="no_frame",
                                hint="harness doctor скажет, что с экраном")
        # Профиль записи объявляет **действительный** источник: иначе запись, снятая с
        # окна, лежала бы в той же ветке журнала, что запись экрана, и их числа сводились
        # бы в одну медиану (TASK-21, часть 1). Вид меняет `structure_hash`, то есть
        # форкает журнал, — это и означает «несравнимы».
        profile = MILESTONE_0.for_frame(first.image)
        if plan.source_kind != profile.structural["capture_source"]:
            profile = profile.with_structural(capture_source=plan.source_kind)
        h, w = first.image.shape[:2]

        prog = progress or Progress.from_profile(
            profile, total_turns=plan.turns, path=plan.path, seconds=plan.seconds,
            kind=plan.kind, channel=plan.progress_channel)
        if on_progress is not None:
            prog.on_sample = on_progress

        out = Outcome(path=plan.path, mechanism=choice.why_text(),
                      mechanism_caveat=choice.caveat or "", audio_note=audio_note,
                      lineage_id="", profile_hash=profile.profile_hash[:8],
                      frame=(int(w), int(h)),
                      source_kind=plan.source_kind, source_why=region_why,
                      structure_hash=profile.structure_hash)
        lid = lineage_id(MILESTONE_0, seed=None,
                         note=f"живая запись: {machine.os_name}/{machine.session}")
        out.lineage_id = lid
        if on_ready is not None:
            on_ready(out)

        actor = Actor.HUMAN if plan.actor_human else Actor.NONE
        layer = ActorLayer.HUMAN if plan.actor_human else ActorLayer.NONE
        with Recorder(plan.path, profile=profile, source=cap.name, synthetic=False,
                      note=plan.note, lineage_id=lid) as rec:
            # Рамка и её происхождение — в журнал записью об источниках. Не в профиль:
            # координаты окна зависят от того, куда его подвинули мышью, и форкать журнал
            # при каждом перетаскивании было бы бессмысленно.
            rec.record_source(kind=plan.source_kind,
                              region=region.as_dict() if region else None,
                              why=region_why,
                              window=target.id if target is not None else "")
            turns = run_turns(rec, source=cap, frames=plan.turns, first=first,
                              actor=actor, actor_layer=layer, audio=audio_src,
                              progress=prog, seconds=plan.seconds,
                              expected_fps=float(profile.parameters["capture_fps"]),
                              should_stop=should_stop,
                              disk_bytes=lambda: dir_bytes(plan.path))
            out.cost = rec.cost()
            rec.record_intervention("cost", {**out.cost, "stages": turns.stages()})
        out.turns = turns
        out.interrupted = turns.interrupted
        out.final_line = prog.finish(
            written=turns.written, unchanged=turns.unchanged,
            disk_bytes=lambda: dir_bytes(plan.path),
            interrupted=turns.interrupted, audio_note=audio_note)
        return out
    finally:
        cap.stop()
        if audio_src is not None:
            audio_src.stop()
