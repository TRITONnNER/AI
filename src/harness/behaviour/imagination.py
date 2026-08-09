"""Мысль — это действие с отключёнными эффекторами.

Из архитектуры: «воображение, планирование, воспоминание и контрфактика — один и
тот же цикл через размыкатель. Пиши один раз».

Здесь размыкатель и есть. Один и тот же код действия исполняется либо в мире,
либо в модели мира — разница только в положении `Breaker`. Ни одна ветка не
дублируется: если бы воображение было отдельной функцией, оно бы разошлось с
реальным действием на второй неделе, и агент воображал бы то, чего не умеет.

Что обязательно, чтобы это не испортило журнал:

- **Воображаемое пишется отдельным видом записи** (`THOUGHT`, не `ACTION`). Иначе
  из журнала нельзя отличить сделанное от продуманного, и пересборка убеждений
  выучит воображаемые последствия как настоящие.
- **Воображаемое не обновляет убеждения о мире.** `rebuild_from_journal`
  пропускает `THOUGHT`. Мысль — опыт думания, а не опыт мира.
- **Размыкатель нельзя забыть закрыть.** Он контекстный менеджер и возвращает
  эффекторы на место даже при исключении.

Четыре режима — один механизм:

| Режим | Откуда берутся последствия |
|---|---|
| Действие | из мира |
| Воображение | из модели мира вперёд от текущего состояния |
| Воспоминание | из журнала, назад |
| Контрфактика | из модели мира от состояния в прошлом |
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Iterator

from ..core.action import Action
from ..core.clocks import Stamp
from ..core.journal import Actor, ActorLayer, Journal, Kind as EntryKind


class Mode(StrEnum):
    ACT = "act"                    # эффекторы подключены, последствия из мира
    IMAGINE = "imagine"            # вперёд от текущего состояния
    REMEMBER = "remember"          # назад по журналу
    COUNTERFACT = "counterfact"    # вперёд от состояния в прошлом


@dataclass(slots=True)
class Breaker:
    """Размыкатель эффекторов. Одна переменная, от которой зависит всё.

    Специально не булев флаг, а объект с режимом: «отключено» бывает по разным
    причинам, и в журнале должно быть видно, по какой именно.
    """

    mode: Mode = Mode.ACT

    @property
    def connected(self) -> bool:
        return self.mode is Mode.ACT

    def as_dict(self) -> dict[str, Any]:
        return {"mode": str(self.mode), "connected": self.connected}


@dataclass(slots=True)
class Step:
    """Один шаг цикла — реальный или воображаемый. Форма одна и та же."""

    action: Action
    mode: Mode
    consequence: Any = None
    predicted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"action": self.action.as_dict(), "mode": str(self.mode),
                "predicted": self.predicted}


class Loop:
    """Единый цикл действия. Куда пойдут эффекты — решает размыкатель.

    `execute` — как действовать в мире, `simulate` — как действовать в модели.
    Второе обязано быть дешевле первого и не иметь права трогать мир; за этим
    следит сам `Loop`, а не дисциплина вызывающего.
    """

    def __init__(self, *, execute: Callable[[Action], Any],
                 simulate: Callable[[Action, Mode], Any],
                 journal: Journal | None = None,
                 breaker: Breaker | None = None) -> None:
        self.execute = execute
        self.simulate = simulate
        self.journal = journal
        self.breaker = breaker or Breaker()
        self.acted = 0
        self.imagined = 0
        self.history: list[Step] = []

    # --- размыкатель --------------------------------------------------------

    @contextmanager
    def disconnected(self, mode: Mode = Mode.IMAGINE) -> Iterator[Breaker]:
        """Отключить эффекторы на время. Возвращает их на место всегда."""
        if mode is Mode.ACT:
            raise ValueError("режим ACT — это не отключение эффекторов")
        was = self.breaker.mode
        self.breaker.mode = mode
        try:
            yield self.breaker
        finally:
            self.breaker.mode = was

    # --- шаг ----------------------------------------------------------------

    def step(self, action: Action, stamp: Stamp | None = None, *,
             actor_layer: ActorLayer = ActorLayer.PLANNER) -> Step:
        """Сделать шаг. Один и тот же вызов и для действия, и для мысли.

        Слой по умолчанию planner: воображение — это и есть работа планировщика,
        и почти всякий его вызов идёт оттуда. Рефлекс и навык, если им понадобится
        прокрутить действие мысленно, передают свой слой сами. Значение по
        умолчанию здесь допустимо потому, что оно не «неизвестно», а «обычный
        случай», и оно указано явно — в отличие от `none`, которое означало бы
        «ничей».
        """
        mode = self.breaker.mode
        if mode is Mode.ACT:
            consequence = self.execute(action)
            self.acted += 1
            kind = EntryKind.ACTION
            predicted = False
        else:
            consequence = self.simulate(action, mode)
            self.imagined += 1
            kind = EntryKind.THOUGHT
            predicted = True

        step = Step(action, mode, consequence, predicted)
        self.history.append(step)
        if self.journal is not None and stamp is not None:
            self.journal.append(kind, stamp, Actor.AGENT, actor_layer, action=action,
                                event={"code": str(mode), "predicted": predicted,
                                       "device": "loop"})
        return step

    # --- готовые режимы -----------------------------------------------------

    def imagine(self, actions: list[Action], stamp_of: Callable[[int], Stamp] | None = None,
                mode: Mode = Mode.IMAGINE, *,
                actor_layer: ActorLayer = ActorLayer.PLANNER) -> list[Step]:
        """Прокрутить цепочку действий, не трогая мир. Это и есть планирование."""
        out: list[Step] = []
        with self.disconnected(mode):
            for i, a in enumerate(actions):
                out.append(self.step(a, stamp_of(i) if stamp_of else None,
                                     actor_layer=actor_layer))
        return out

    def counterfactual(self, actions: list[Action],
                       stamp_of: Callable[[int], Stamp] | None = None, *,
                       actor_layer: ActorLayer = ActorLayer.PLANNER) -> list[Step]:
        """«А если бы я тогда сделал иначе». Тот же цикл, другой режим."""
        return self.imagine(actions, stamp_of, mode=Mode.COUNTERFACT,
                            actor_layer=actor_layer)

    def remember(self, actions: list[Action],
                 stamp_of: Callable[[int], Stamp] | None = None, *,
                 actor_layer: ActorLayer = ActorLayer.PLANNER) -> list[Step]:
        """Проиграть прошлое как мысль. Тот же цикл."""
        return self.imagine(actions, stamp_of, mode=Mode.REMEMBER,
                            actor_layer=actor_layer)

    def stats(self) -> dict[str, Any]:
        return {"acted": self.acted, "imagined": self.imagined,
                "breaker": self.breaker.as_dict(),
                "imagination_ratio": round(
                    self.imagined / max(1, self.acted + self.imagined), 4)}


class GuardedEffectors:
    """Обёртка, которая физически не даёт воображению дотянуться до мира.

    Нужна потому, что «не вызывать эффекторы при отключённом размыкателе» — это
    договорённость, а договорённости ломаются. Здесь попытка исполнить действие
    при разомкнутом размыкателе кончается исключением, а не тихим нажатием.
    """

    def __init__(self, breaker: Breaker, submit: Callable[[Action], Any]) -> None:
        self._breaker = breaker
        self._submit = submit
        self.blocked = 0

    def __call__(self, action: Action) -> Any:
        if not self._breaker.connected:
            self.blocked += 1
            raise RuntimeError(
                f"эффекторы разомкнуты (режим {self._breaker.mode}), а действие "
                f"{action} попыталось дойти до мира. Воображение обязано идти "
                "через simulate, иначе мысль станет поступком")
        return self._submit(action)
