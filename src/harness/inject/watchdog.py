"""Сторожевой таймер.

Из 0.2: «если N секунд состояние экрана не меняется или процесс игры пропал —
остановить инъекцию и записать событие. Без этого агент проведёт ночь в меню
настроек».

Время берётся из переданной функции, а не из `time.monotonic` напрямую: иначе
таймер нельзя проверить офлайн по записи, а по правилам проекта каждый модуль
обязан тестироваться на записанных сессиях без запуска игры.

Порог считается по доле изменившихся пикселей, а не по средней разнице яркости:
средняя разница «размазывается» на большом кадре, и мигающий курсор в углу
выглядит как полная неподвижность. Доля изменившихся пикселей ловит и мелкое
локальное шевеление.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..core.clocks import Stamp
from ..core.journal import Actor, Journal, Kind as EntryKind
from .stop import StopSwitch


@dataclass(frozen=True, slots=True)
class Trip:
    """Срабатывание. `code` — почему, `detail` — что показать человеку."""

    code: str
    detail: dict[str, object]


class Watchdog:
    def __init__(self, *, still_seconds: float, threshold: float,
                 stop: StopSwitch | None = None, journal: Journal | None = None,
                 now: Callable[[], float] = time.monotonic,
                 process_alive: Callable[[], bool] | None = None,
                 pixel_delta: int = 8) -> None:
        self.still_seconds = float(still_seconds)
        self.threshold = float(threshold)
        self.stop = stop
        self.journal = journal
        self._now = now
        self._process_alive = process_alive
        self.pixel_delta = int(pixel_delta)
        self._prev: np.ndarray | None = None
        self._still_since: float | None = None
        self._tripped: Trip | None = None
        self.last_changed_fraction: float | None = None

    @property
    def tripped(self) -> Trip | None:
        return self._tripped

    def reset(self) -> None:
        """После снятия стопа таймер начинает считать заново."""
        self._prev = None
        self._still_since = None
        self._tripped = None
        self.last_changed_fraction = None

    def changed_fraction(self, frame: np.ndarray) -> float | None:
        """Доля пикселей, изменившихся больше чем на `pixel_delta`."""
        if self._prev is None or self._prev.shape != frame.shape:
            return None
        diff = np.abs(frame.astype(np.int16) - self._prev.astype(np.int16))
        if diff.ndim == 3:
            diff = diff.max(axis=2)
        return float((diff > self.pixel_delta).mean())

    def feed(self, frame: np.ndarray, stamp: Stamp) -> Trip | None:
        """Дать таймеру кадр. Возвращает срабатывание, если оно случилось сейчас."""
        if self._tripped is not None:
            return None  # уже сработал; до reset() молчим, чтобы не спамить журнал

        if self._process_alive is not None and not self._process_alive():
            return self._trip("process_gone", {"still_seconds": None}, stamp)

        frac = self.changed_fraction(frame)
        self._prev = frame.copy()
        self.last_changed_fraction = frac
        now = self._now()

        if frac is None:
            return None
        if frac > self.threshold:
            self._still_since = None
            return None

        if self._still_since is None:
            self._still_since = now
            return None
        still_for = now - self._still_since
        if still_for >= self.still_seconds:
            return self._trip("screen_still",
                              {"still_seconds": round(still_for, 3),
                               "changed_fraction": round(frac, 6),
                               "threshold": self.threshold}, stamp)
        return None

    def _trip(self, code: str, detail: dict[str, object], stamp: Stamp) -> Trip:
        trip = Trip(code, detail)
        self._tripped = trip
        if self.journal is not None:
            self.journal.append(EntryKind.WATCHDOG, stamp, Actor.NONE,
                                event={"code": code, **detail})
        if self.stop is not None:
            self.stop.engage(f"watchdog:{code}", stamp)
        return trip
