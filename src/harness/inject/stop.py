"""Глобальный СТОП.

Из 0.2: «мгновенно обрывающий инъекцию. Работает всегда, в том числе когда всё
остальное зависло». Критерий готовности — срабатывание за один кадр.

Как это обеспечено:

- Состояние — `threading.Event`, а не поле объекта: его выставляет любой поток,
  включая слушателя горячей клавиши, и для этого ему не нужен ни планировщик, ни
  основной цикл.
- Удержание клавиши нарезано на срезы (`InjectionSink.hold`), и после каждого
  среза спрашивается `is_engaged`. При срезе 5 мс и кадре 33 мс отпускание
  происходит внутри одного кадра с запасом.
- Освобождение зажатых выходов идёт в `finally`, то есть случается даже если
  выше по стеку всё рухнуло.

Из инварианта 3: СТОП не ставит мир на паузу и не останавливает журнал. Он
отключает эффекторы — и только. Запись продолжается, потому что происходящее
после стопа тоже опыт.
"""

from __future__ import annotations

import threading
from typing import Callable

from ..core.clocks import Stamp
from ..core.journal import Actor, Journal, Kind as EntryKind


class StopSwitch:
    """Флаг «эффекторы отключены». Ставится откуда угодно, снимается вручную."""

    def __init__(self, journal: Journal | None = None) -> None:
        self._event = threading.Event()
        self._reason: str | None = None
        self._journal = journal
        self._lock = threading.Lock()
        self._listeners: list[Callable[[bool, str | None], None]] = []

    @property
    def is_engaged(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def engage(self, reason: str, stamp: Stamp | None = None, *,
               actor: Actor = Actor.HUMAN) -> None:
        """Оборвать инъекцию. Повторный вызов не считается новым событием."""
        with self._lock:
            if self._event.is_set():
                return
            self._reason = reason
            self._event.set()
        if self._journal is not None and stamp is not None:
            self._journal.append(EntryKind.STOP, stamp, actor,
                                 event={"code": "stop", "reason": reason})
        self._notify()

    def release(self, stamp: Stamp | None = None, *, actor: Actor = Actor.HUMAN) -> None:
        with self._lock:
            if not self._event.is_set():
                return
            was = self._reason
            self._reason = None
            self._event.clear()
        if self._journal is not None and stamp is not None:
            self._journal.append(EntryKind.RESUME, stamp, actor,
                                 event={"code": "resume", "after": was})
        self._notify()

    def wait(self, timeout: float | None = None) -> bool:
        """Заблокироваться до стопа. Для потока-наблюдателя, не для основного цикла."""
        return self._event.wait(timeout)

    def on_change(self, fn: Callable[[bool, str | None], None]) -> None:
        self._listeners.append(fn)

    def _notify(self) -> None:
        for fn in list(self._listeners):
            fn(self.is_engaged, self._reason)


class HotkeyListener:
    """Слушатель горячей клавиши СТОПа.

    Реализация платформенная и на этой машине непроверяема: без графической
    сессии перехватывать глобальную горячую клавишу нечем. Поэтому здесь честный
    отказ, а не тихо не работающий поток — иначе исследователь будет думать, что
    СТОП есть, а его не будет.

    Пока живого backend'а нет, СТОП вызывается программно —
    `StopSwitch.engage(reason, stamp)` — из любого потока, и это работает и
    проверено тестом. Команды `harness stop` нет: она потребовала бы канала к
    уже запущенному процессу записи, а обещать в документации то, чего нет,
    хуже, чем не обещать.
    """

    def __init__(self, switch: StopSwitch, combo: str = "esc") -> None:
        self.switch = switch
        self.combo = combo

    def start(self) -> None:
        raise NotImplementedError(
            "перехват глобальной горячей клавиши не реализован: нужна графическая "
            "сессия и платформенный backend (evdev-grab на Linux, RegisterHotKey на "
            "Windows). До этого СТОП вызывается программно — StopSwitch.engage() — "
            "и через CLI. Молча ничего не слушать было бы хуже: исследователь "
            "решил бы, что аварийная остановка есть"
        )
