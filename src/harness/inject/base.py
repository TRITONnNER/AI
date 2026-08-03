"""Инъекция ввода.

Из 0.2: `press(key, duration_ms, modifiers)`, `move_mouse(dx, dy)`, сторожевой
таймер, глобальный СТОП, срабатывающий за один кадр.

Здесь же исправлено то, что в дизайне пульта сделано неверно (см.
`docs/DESIGN-REVIEW-CONSOLE.md`, пункт 1): **заглушённая маской попытка всё
равно попадает в журнал**, с `masked=True` и причиной. Иначе агент завёл бы
убеждение «этот выход молчит», а причины этого убеждения в журнале не было бы —
и пересборка убеждений из журнала дала бы другую карту тела, чем та, по которой
агент жил. Что забыть при пересборке, решает слой убеждений; вырезать из
источника истины нельзя (инвариант 1).

Ни одна функция в этом модуле не знает названий клавиш. Действие адресует
непрозрачный выход (`OUT_2C`); соответствие выход → скан-код живёт в конкретном
устройственном backend'е рядом с отладочным потоком (инвариант 4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from ..core.action import Action, Kind
from ..core.clocks import Stamp
from ..core.journal import Actor, Journal, Kind as EntryKind


class InjectionUnavailable(RuntimeError):
    """Устройство ввода недоступно. Текст обязан говорить, что сделать."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """Что случилось с попыткой. Пишется в журнал вместе с действием."""

    delivered: bool          # дошло ли до устройства
    masked: bool             # заглушено маской
    stopped: bool            # оборвано СТОПом
    reason: str | None       # почему не дошло
    latency_ms: float        # от вызова до конца удержания

    @property
    def code(self) -> str:
        if self.delivered:
            return "delivered"
        if self.stopped:
            return "stopped"
        if self.masked:
            return "masked"
        return "failed"


@runtime_checkable
class DeviceSink(Protocol):
    """Тот, кто реально шевелит устройством. Ниже него — только ОС."""

    name: str

    def key_down(self, output: str) -> None: ...
    def key_up(self, output: str) -> None: ...
    def mouse_move(self, dx: int, dy: int) -> None: ...
    def close(self) -> None: ...


class NullDevice:
    """Устройства нет: ввод никуда не идёт.

    Это не заглушка, подменяющая результат: `Injector` знает, что устройство
    пустое, и пишет в журнал `delivered=False` с причиной `no_device`. Нужен для
    офлайн-прогонов по записи, где инъекция не имеет смысла, и для тестов.
    """

    name = "null"

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def key_down(self, output: str) -> None:
        self.calls.append(("down", output))

    def key_up(self, output: str) -> None:
        self.calls.append(("up", output))

    def mouse_move(self, dx: int, dy: int) -> None:
        self.calls.append(("move", (dx, dy)))

    def close(self) -> None:
        pass


class InjectionSink:
    """Обёртка над устройством: удержание нарезано на срезы, чтобы СТОП успевал.

    Удержание 340 мс — это не `sleep(0.34)`. Между `key_down` и `key_up` спим
    срезами по `slice_ms` и после каждого среза спрашиваем СТОП. Поэтому
    зажатая клавиша отпускается не позже одного среза после нажатия СТОПа, а не
    когда закончится удержание.
    """

    def __init__(self, device: DeviceSink, *, slice_ms: float = 5.0,
                 sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], float] = time.monotonic) -> None:
        self.device = device
        self.slice_ms = float(slice_ms)
        self._sleep = sleep
        self._now = now

    @property
    def name(self) -> str:
        return self.device.name

    def hold(self, outputs: list[str], duration_ms: int,
             abort: Callable[[], bool]) -> tuple[bool, float]:
        """Зажать выходы на время, отпуская досрочно при `abort()`.

        Возвращает (дошло ли до конца, сколько мс держали).
        """
        t0 = self._now()
        held: list[str] = []
        try:
            for out in outputs:
                if abort():
                    return False, (self._now() - t0) * 1000.0
                self.device.key_down(out)
                held.append(out)
            remaining = duration_ms / 1000.0
            step = self.slice_ms / 1000.0
            completed = True
            while remaining > 0:
                if abort():
                    completed = False
                    break
                dt = min(step, remaining)
                self._sleep(dt)
                remaining -= dt
            return completed, (self._now() - t0) * 1000.0
        finally:
            for out in reversed(held):
                self.device.key_up(out)

    def move(self, dx: int, dy: int) -> None:
        self.device.mouse_move(dx, dy)

    def close(self) -> None:
        self.device.close()


class Injector:
    """Единственная дорога от действия к устройству.

    Всё, что здесь проходит, попадает в журнал — дошло, заглушено маской или
    оборвано СТОПом. Молчаливого пропуска нет ни в одной ветке.
    """

    def __init__(self, sink: InjectionSink, journal: Journal, *, mask=None, stop=None,
                 actor: Actor = Actor.AGENT) -> None:
        self.sink = sink
        self.journal = journal
        self.mask = mask
        self.stop = stop
        self.actor = actor

    def submit(self, action: Action, stamp: Stamp, *, scope: str = "window") -> Outcome:
        outputs = list(action.outputs_touched())

        # 1. СТОП. Проверяется первым: он важнее всего остального.
        if self.stop is not None and self.stop.is_engaged:
            return self._record(action, stamp, Outcome(
                False, True, True, f"stop:{self.stop.reason or 'engaged'}", 0.0))

        # 2. Маска ввода. Заглушённая попытка пишется в журнал наравне с дошедшей.
        if self.mask is not None:
            blocked = self.mask.blocked_outputs(outputs, scope=scope)
            if blocked:
                return self._record(action, stamp, Outcome(
                    False, True, False, f"mask:{scope}:{','.join(sorted(blocked))}", 0.0))

        # 3. Устройство.
        if isinstance(self.sink.device, NullDevice):
            return self._record(action, stamp, Outcome(
                False, False, False, "no_device", 0.0))

        abort = (lambda: self.stop.is_engaged) if self.stop is not None else (lambda: False)
        if action.kind is Kind.MOUSE_MOVE:
            self.sink.move(action.dx, action.dy)
            completed, latency = True, 0.0
        elif action.kind is Kind.NOTHING:
            completed, latency = self.sink.hold([], action.duration_ms, abort)
        else:
            completed, latency = self.sink.hold(outputs, action.duration_ms, abort)

        stopped = not completed
        return self._record(action, stamp, Outcome(
            completed, False, stopped,
            None if completed else f"stop:{getattr(self.stop, 'reason', None) or 'aborted'}",
            latency))

    def _record(self, action: Action, stamp: Stamp, outcome: Outcome) -> Outcome:
        recorded = action.masked_as(outcome.reason) if outcome.masked else action
        self.journal.append(
            EntryKind.ACTION, stamp, self.actor, action=recorded,
            event={"code": outcome.code, "reason": outcome.reason,
                   "latency_ms": round(outcome.latency_ms, 3), "device": self.sink.name},
        )
        return outcome
