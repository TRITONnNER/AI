"""Контуры с разными частотами и субсумпция.

Из архитектуры:

    Драйвы и цели        ~0.01 Гц   что вообще делать
    Планировщик (VLM)    ~0.5 Гц    план на минуту
    Библиотека навыков   ~2 Гц      готовые макросы
    Реактивная политика  ~20 Гц     прицел, тайминг, уклонение
    Экран / ввод         60 Гц      единственный интерфейс

«Нижние контуры имеют право перебивать верхние (субсумпция). Пока планировщик
думает, рефлексы работают.» И инвариант 3: «мир никогда не ставится на паузу.
Планировщик обязан быть прерываемым и в любой момент отдавать лучший ответ на
текущий момент. Никаких блокирующих вызовов в основном цикле.»

Как это сделано так, чтобы оно было правдой, а не намерением.

**Контур — генератор, а не функция.** Функцию нельзя прервать: она либо
вернулась, либо нет. Генератор отдаёт управление на каждом `yield`, и его
состояние сохраняется между вызовами. Поэтому долгая работа планировщика
нарезается автоматически, без потоков и без блокировок.

**Лучший ответ в любой момент.** Контур на каждом `yield` может отдать частичный
результат. Планировщик, прерванный на середине, отдаёт то, что успел. Спросить
«что делать сейчас» можно всегда, и ответ будет — возможно, хуже, но будет.

**Субсумпция.** Порядок обслуживания — от нижнего контура к верхнему. Если во
время работы верхнего подошёл срок нижнего, верхний приостанавливается на
границе `yield`. Никакого приоритетного вытеснения внутри шага: шаг и так
короткий, а вытеснение посреди вычисления оставило бы порванное состояние.

Часы передаются снаружи. Иначе планировщик нельзя проверить офлайн по записи, а
по правилам проекта каждый модуль обязан так проверяться.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Generator

from ..core.profile import Profile

# Работа контура: генератор, который отдаёт частичные результаты.
Work = Generator[Any, None, Any]
Start = Callable[[], Work]


class ContourError(RuntimeError):
    pass


@dataclass(slots=True)
class Contour:
    """Один контур: частота, что он делает, и его лучший ответ на сейчас."""

    name: str
    hz: float
    start: Start
    level: int = 0                  # 0 — самый нижний (рефлексы), выше — медленнее
    best: Any = None                # лучший ответ на текущий момент
    runs: int = 0
    slices: int = 0
    preempted: int = 0
    overruns: int = 0               # сколько раз не успел в свой период
    _work: Work | None = None
    _next_due: float = 0.0
    _started_at: float = 0.0

    @property
    def period(self) -> float:
        return 1.0 / max(1e-9, self.hz)

    @property
    def busy(self) -> bool:
        return self._work is not None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "hz": self.hz, "level": self.level,
                "runs": self.runs, "slices": self.slices,
                "preempted": self.preempted, "overruns": self.overruns,
                "busy": self.busy, "has_answer": self.best is not None}


@dataclass(slots=True)
class Ran:
    """Что произошло за один шаг планировщика — для журнала и для пульта."""

    contour: str
    slices: int
    finished: bool
    preempted: bool
    partial: Any = None


class Scheduler:
    """Обслуживает контуры по срокам. Не блокирует, не спит, не ставит мир на паузу."""

    def __init__(self, profile: Profile, *, now: Callable[[], float] = time.monotonic) -> None:
        p = profile.parameters
        self.slice_seconds = float(p["contour_slice_ms"]) / 1000.0
        self.max_yields = int(p["contour_max_yields"])
        self.subsumption = bool(profile.structural["subsumption_enabled"])
        # Пятая ось модуляции: насколько рано нижний контур имеет право перебить
        # верхний. Ноль — только когда он уже просрочен, и это прежнее поведение,
        # поэтому база в профиле нулевая. Единица — за целый свой период до срока.
        self.reflex_priority = float(profile.parameters["emotion_reflex_priority_base"])
        self._now = now
        self.contours: list[Contour] = []
        self.steps = 0

    # --- состав -------------------------------------------------------------

    def add(self, name: str, hz: float, start: Start, *, level: int) -> Contour:
        if any(c.name == name for c in self.contours):
            raise ContourError(f"контур {name!r} уже есть")
        c = Contour(name, hz, start, level=level, _next_due=self._now())
        self.contours.append(c)
        # Нижние обслуживаются первыми — это и есть субсумпция в одну строку.
        self.contours.sort(key=lambda x: (x.level, x.name))
        return c

    @classmethod
    def standard(cls, profile: Profile, *, now: Callable[[], float] = time.monotonic,
                 reflex: Start | None = None, skills: Start | None = None,
                 planner: Start | None = None, drives: Start | None = None) -> Scheduler:
        """Четыре контура из архитектуры, с частотами из профиля."""
        s = cls(profile, now=now)
        p = profile.parameters
        if reflex is not None:
            s.add("reflex", float(p["hz_reflex"]), reflex, level=0)
        if skills is not None:
            s.add("skills", float(p["hz_skills"]), skills, level=1)
        if planner is not None:
            s.add("planner", float(p["hz_planner"]), planner, level=2)
        if drives is not None:
            s.add("drives", float(p["hz_drives"]), drives, level=3)
        return s

    def get(self, name: str) -> Contour:
        for c in self.contours:
            if c.name == name:
                return c
        raise ContourError(f"нет контура {name!r}")

    # --- шаг ----------------------------------------------------------------

    def step(self) -> list[Ran]:
        """Обслужить всё, чему пришёл срок. Возвращает, что успело поработать.

        Ни одного блокирующего вызова: если работы нет, шаг возвращается пустым и
        вызывающий волен заняться чем угодно — например, взять следующий кадр.
        """
        self.steps += 1
        now = self._now()
        out: list[Ran] = []

        due = [c for c in self.contours if now >= c._next_due or c.busy]
        for contour in due:
            ran = self._serve(contour, now)
            if ran is not None:
                out.append(ran)
                if ran.preempted:
                    # Верхний контур отложен: ниже него на этом шаге ничего нет,
                    # а выше — тем более. Возвращаемся, чтобы мир успел идти.
                    break
        return out

    def _serve(self, contour: Contour, now: float) -> Ran | None:
        if contour._work is None:
            if now < contour._next_due:
                return None
            contour._work = contour.start()
            contour.runs += 1
            contour._started_at = now
        deadline = now + self.slice_seconds
        slices = 0
        finished = False
        while True:
            try:
                partial = next(contour._work)
            except StopIteration as done:
                if done.value is not None:
                    contour.best = done.value
                finished = True
                break
            slices += 1
            contour.slices += 1
            if partial is not None:
                # Частичный результат — это и есть «лучший ответ на текущий момент».
                contour.best = partial
            if self._now() >= deadline or slices >= self.max_yields:
                # Второе условие — предохранитель. Разрешение часов может быть
                # грубее среза (на Windows это обычное дело), и тогда дедлайн не
                # срабатывает вообще, а верхний контур съедает весь шаг вместе с
                # рефлексами. Ограничение по числу шагов от часов не зависит.
                break
            if self.subsumption and self._lower_is_due(contour):
                break

        if finished:
            contour._work = None
            spent = self._now() - contour._started_at
            contour._next_due = max(now, contour._next_due) + contour.period
            if spent > contour.period:
                contour.overruns += 1
            return Ran(contour.name, slices, True, False, contour.best)

        contour.preempted = contour.preempted + 1
        return Ran(contour.name, slices, False, True, contour.best)

    def _lower_is_due(self, contour: Contour) -> bool:
        """Есть ли внизу тот, кому пора. `reflex_priority` решает, насколько «пора».

        Право хода — не «да/нет», а насколько рано. Запас отсчитывается от периода
        самого нижнего контура, а не от абсолютных секунд: у рефлекса на 20 Гц и у
        планировщика на 0.5 Гц «немного раньше» — это разные величины, и абсолютное
        число здесь означало бы разное на разных частотах.
        """
        now = self._now()
        return any(c.level < contour.level
                   and (now >= c._next_due - self.reflex_priority * c.period
                        or c.busy)
                   for c in self.contours)

    # --- опрос --------------------------------------------------------------

    def best_answer(self, name: str) -> Any:
        """Лучший ответ контура на сейчас. Работает и когда контур ещё думает."""
        return self.get(name).best

    def run_for(self, seconds: float, *, tick: Callable[[], None] | None = None,
                max_steps: int = 1_000_000) -> list[Ran]:
        """Покрутить планировщик заданное время. `tick` — то, что делает мир.

        Мир идёт в `tick`, и он вызывается на каждом шаге независимо от того,
        успел ли кто-то из контуров: это и есть «мир не ставится на паузу».
        """
        started = self._now()
        log: list[Ran] = []
        steps = 0
        while self._now() - started < seconds and steps < max_steps:
            log.extend(self.step())
            if tick is not None:
                tick()
            steps += 1
        return log

    def stats(self) -> dict[str, Any]:
        return {"steps": self.steps, "subsumption": self.subsumption,
                "slice_ms": round(self.slice_seconds * 1000, 3),
                "contours": [c.as_dict() for c in self.contours]}


# --- вспомогательное: как выглядит прерываемый контур -----------------------


def budgeted(steps: int, produce: Callable[[int], Any]) -> Start:
    """Сделать прерываемый контур из пошаговой работы.

    `produce(i)` вызывается на каждом шаге и возвращает частичный результат.
    Так и должен выглядеть планировщик: не «подумать и вернуть план», а
    «улучшать план шагами, отдавая лучшее из имеющегося».
    """
    def start() -> Work:
        best = None
        for i in range(steps):
            best = produce(i)
            yield best
        return best
    return start


def fixed(result: Any) -> Start:
    """Контур, который отвечает сразу. Годится для рефлексов: им думать нечем."""
    def start() -> Work:
        yield result
        return result
    return start
