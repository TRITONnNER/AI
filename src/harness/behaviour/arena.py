"""Арена: короткие сценарии с мгновенным респавном. Начало М5.

Из вехи М5: «Арена: короткие сценарии с мгновенным респавном для набора данных». Смысл в
объёме опыта: чтобы выучить последствия действия, его надо попробовать много раз, а мир, в
котором одна ошибка стоит десяти минут восстановления, даёт единицы проб в час вместо сотен.

## Что здесь считается эпизодом

Эпизод — это отрезок опыта с **объявленным началом** и одним из трёх исходов: цель
достигнута, бюджет тиков вышел, мир стал непригодным (необратимое действие всё сломало).
Исходов три, а не два, по той же причине, что и у конвейера убеждений: «не успел» и «сломал»
требуют разного, и слить их значило бы потерять то, ради чего арена и нужна.

## Мгновенный респавн — не тайная перезагрузка

Мир между эпизодами пересоздаётся, и это **видно в журнале**: границы эпизодов пишутся
записями со своим `actor_layer`. Три вещи, которые здесь легко испортить и которые
испорчены не будут:

1. **Мир не ставится на паузу** (инвариант 3). Респавн — такой же шаг мира, как остальные:
   часы `t_world` продолжают идти, и в журнале нет разрыва.
2. **Возрождение — действие агента, а не механика снаружи** (инвариант 17: «агент жмёт
   кнопку возрождения сам, и у этого действия есть `actor_layer`»). Поэтому у респавна есть
   инициатор, и по умолчанию это **не** человек: `ActorLayer.DRIVE`, потому что решение
   «начать заново» приходит от драйва, а не от плана.
3. **Смерть считается в общей валюте.** Отдельного «страха смерти» нет; исход эпизода
   записывается вместе с тем, сколько тиков и проб потеряно, и это те же тики и пробы, что
   в остальном журнале.

## Чего здесь нет

**Награды.** Эпизод не «оценивается»: он либо привёл туда, где тест цели проходит, либо нет.
Ни очков, ни успеха в долях. Сравнивать эпизоды между собой можно по времени и по числу
проб — величинам, которые кто-то измерял.

**Учебного плана.** Арена не устраивает сценарии «от простого к сложному»: это была бы
подсказка о структуре задачи, а её агент должен добывать сам. Сценарий задаёт мир и цель,
и ничего о том, как цель достигать.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.clocks import Stamp
from ..core.journal import Actor, ActorLayer, Journal, Kind as EntryKind


class ArenaError(RuntimeError):
    pass


#: Чем кончился эпизод. Набор закрыт: новый исход — это изменение схемы записи.
OUTCOMES: tuple[str, ...] = ("достигнута", "бюджет", "мир сломан", "прервано")


@dataclass(slots=True)
class Scenario:
    """Короткий сценарий: как построить мир, что считать целью, сколько дать тиков.

    `build` возвращает мир; `goal` по миру отдаёт функцию-тест и её описание словами.
    Тест — объективный (по истине мира, недоступной агенту), описание — для исследователя.
    Так же, как у целей: непроверяемая цель в работе не участвует.
    """

    name: str
    build: Callable[[int], Any]
    goal: Callable[[Any], tuple[Callable[[], bool], str]]
    budget_ticks: int = 200
    why: str = ""

    def __post_init__(self) -> None:
        if self.budget_ticks <= 0:
            raise ArenaError(f"сценарий {self.name}: бюджет должен быть положительным")
        if not self.why:
            raise ArenaError(
                f"сценарий {self.name}: не сказано, что он проверяет. Сценарий без "
                "объявленного смысла набирает данные ни о чём")


@dataclass(slots=True)
class Episode:
    """Один эпизод: чем кончился, сколько стоил и что в нём происходило."""

    index: int
    scenario: str
    seed: int
    outcome: str = ""
    ticks: int = 0
    tries: int = 0                     # сколько действий отправлено
    changed: int = 0                   # сколько из них изменили мир
    t_world_start: int = 0
    t_world_end: int = 0
    why: str = ""

    @property
    def useful_share(self) -> float | None:
        """Доля проб, изменивших мир. Единица независимости — проба."""
        return None if not self.tries else self.changed / self.tries

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "scenario": self.scenario, "seed": self.seed,
                "outcome": self.outcome, "ticks": self.ticks, "tries": self.tries,
                "changed": self.changed,
                "useful_share": (None if self.useful_share is None
                                 else round(self.useful_share, 4)),
                "t_world": [self.t_world_start, self.t_world_end], "why": self.why}


@dataclass(slots=True)
class Arena:
    """Сцепка эпизодов: строит мир, ведёт границы, пишет их в журнал.

    Не цикл и не планировщик: арена **не решает, что делать**. Она отвечает на два
    вопроса — «в каком мире мы сейчас» и «эпизод кончился?» — и всё. Кто действует, решают
    контуры; поэтому арену можно прогнать и с лепетом, и с планировщиком, и с человеком за
    рулём, не меняя её код.
    """

    scenario: Scenario
    journal: Journal | None = None
    seed: int = 0
    #: Кто инициирует респавн. По умолчанию драйв: решение «начать заново» приходит от
    #: драйва, а не от плана (инвариант 17). Человек — законный вариант для демонстраций,
    #: и тогда это видно в журнале.
    actor: Actor = Actor.AGENT
    actor_layer: ActorLayer = ActorLayer.DRIVE
    episodes: list[Episode] = field(default_factory=list)
    world: Any = None
    current: Episode | None = None
    _test: Callable[[], bool] | None = None
    _test_text: str = ""

    # --- эпизоды ------------------------------------------------------------

    def start(self, stamp: Stamp | None = None) -> Episode:
        """Начать эпизод. Мир строится заново, часы мира **продолжают** идти."""
        index = len(self.episodes)
        seed = self.seed + index
        self.world = self.scenario.build(seed)
        self._test, self._test_text = self.scenario.goal(self.world)
        ep = Episode(index=index, scenario=self.scenario.name, seed=seed,
                     t_world_start=int(getattr(self.world, "t_world", 0)))
        self.episodes.append(ep)
        self.current = ep
        self._write("episode_start", stamp, {"index": index, "seed": seed,
                                             "budget_ticks": self.scenario.budget_ticks,
                                             "test": self._test_text})
        return ep

    def step(self, action: Any, stamp: Stamp | None = None) -> Any:
        """Один шаг мира. Возвращает наблюдение мира как есть.

        Здесь **нет** ни оценки, ни награды: наблюдение отдаётся тому, кто действовал, а
        решение о конце эпизода принимается отдельным вызовом `finished`.
        """
        if self.current is None or self.world is None:
            raise ArenaError("эпизод не начат: сначала start()")
        obs = self.world.step(action)
        ep = self.current
        ep.ticks += 1
        if action is not None:
            ep.tries += 1
            if getattr(self.world, "last_action_changed", None):
                ep.changed += 1
        ep.t_world_end = int(getattr(self.world, "t_world", ep.t_world_end))
        return obs

    def finished(self) -> str:
        """Кончился ли эпизод и чем. Пустая строка — продолжается.

        Порядок проверок не случаен: сначала цель, потом поломка мира, потом бюджет. Иначе
        эпизод, в котором цель достигнута последним тиком бюджета, записался бы как
        «бюджет», и данные о достижимости цели уехали бы в сторону пессимизма.
        """
        if self.current is None:
            return ""
        if self._test is not None and self._test():
            return "достигнута"
        if self._broken():
            return "мир сломан"
        if self.current.ticks >= self.scenario.budget_ticks:
            return "бюджет"
        return ""

    def _broken(self) -> bool:
        """Мир стал непригоден для этой цели: необратимое действие всё сломало.

        Спрашивается **у мира**, а не выводится по наблюдению: непригодность — свойство
        истины, и агенту она не сообщается. Мир, не умеющий отвечать, считается целым:
        придумывать поломку за него нельзя.
        """
        broken = getattr(self.world, "unusable", None)
        return bool(broken()) if callable(broken) else bool(broken)

    def close(self, outcome: str, stamp: Stamp | None = None, *, why: str = "") -> Episode:
        """Закрыть эпизод объявленным исходом. Пишется в журнал со своим инициатором."""
        if self.current is None:
            raise ArenaError("закрывать нечего: эпизод не начат")
        if outcome not in OUTCOMES:
            raise ArenaError(f"нет такого исхода: {outcome!r}; есть {list(OUTCOMES)}")
        ep = self.current
        ep.outcome = outcome
        ep.why = why
        self._write("episode_end", stamp, {**ep.as_dict()})
        self.current = None
        return ep

    def respawn(self, stamp: Stamp | None = None) -> Episode:
        """Закрыть эпизод его исходом и **немедленно** начать следующий.

        Мгновенность здесь означает «без остановки мира», а не «без записи»: обе границы
        попадают в журнал, и по ним видно, сколько эпизод стоил.
        """
        outcome = self.finished() or "прервано"
        self.close(outcome, stamp)
        return self.start(stamp)

    # --- сводка -------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Числа по эпизодам. Единица независимости объявлена при каждом.

        Доли считаются по **эпизодам**, а не по тикам: эпизод — независимая попытка, тик —
        нет (тики внутри одного эпизода зависимы по построению, это один и тот же мир).
        """
        done = [e for e in self.episodes if e.outcome]
        by_outcome = {o: sum(1 for e in done if e.outcome == o) for o in OUTCOMES}
        ticks = [e.ticks for e in done]
        tries = sum(e.tries for e in done)
        changed = sum(e.changed for e in done)
        return {
            "episodes": len(done),
            "unit": "эпизод",
            "by_outcome": by_outcome,
            "reached_share": (None if not done
                              else round(by_outcome["достигнута"] / len(done), 4)),
            "mean_ticks": (None if not ticks else round(sum(ticks) / len(ticks), 2)),
            "ticks_total": sum(ticks),
            "tries": tries, "changed": changed,
            "useful_share": None if not tries else round(changed / tries, 4),
            "unit_for_useful": "проба",
            "scenario": self.scenario.name,
            "why": self.scenario.why,
        }

    def _write(self, code: str, stamp: Stamp | None, detail: dict[str, Any]) -> None:
        if self.journal is None or stamp is None:
            return
        # Границы эпизодов — записи вида `GOAL`: эпизод и есть проживание цели, и заводить
        # для него отдельный вид записи значило бы расширять схему журнала там, где
        # подходящий вид уже есть.
        self.journal.append(EntryKind.GOAL, stamp, self.actor, self.actor_layer,
                            event={"code": code, "scenario": self.scenario.name, **detail})


# ---------------------------------------------------------------------------
# Сценарии
# ---------------------------------------------------------------------------
#
# Набор маленький нарочно. Каждый сценарий отвечает на свой вопрос о том, чему агент может
# научиться за короткий эпизод, и «ещё один похожий» ничего не добавляет.


def reach_place(profile: Any, *, distance_px: float = 120.0) -> Scenario:
    """Дойти оттуда, где стоишь, туда, где не был. Проверяет связь действия и смещения.

    Тест объективный: пройденное расстояние по истине мира. Агент истины не видит и обязан
    добывать связь «удержание выхода → смещение» пробами.
    """
    from ..corpus.world import InteractiveWorld

    def build(seed: int) -> Any:
        return InteractiveWorld(profile, seed=seed)

    def goal(world: Any) -> tuple[Callable[[], bool], str]:
        start = (float(world.state.cam_x), float(world.state.cam_y))

        def test() -> bool:
            now = (float(world.state.cam_x), float(world.state.cam_y))
            moved = ((now[0] - start[0]) ** 2 + (now[1] - start[1]) ** 2) ** 0.5
            return moved >= distance_px

        return test, f"сместиться от начальной точки на {distance_px:g} пикселей истины"

    return Scenario("дойти", build, goal, budget_ticks=200,
                    why="есть ли у агента связь между удержанием выхода и смещением мира")


def flip_switch(profile: Any) -> Scenario:
    """Переключить обратимый эффект. Проверяет, отличает ли агент обратимое от прочего.

    Взят обратимый эффект, а не необратимый, и это часть постановки: на необратимом
    эпизод кончался бы с первой попытки, и данных о повторяемости не было бы вовсе.
    """
    from ..corpus.world import InteractiveWorld

    def build(seed: int) -> Any:
        return InteractiveWorld(profile, seed=seed)

    def goal(world: Any) -> tuple[Callable[[], bool], str]:
        # Свет читается из среза состояния мира (`state.light`), а не из выдуманного ключа:
        # первая редакция спрашивала `truth()["light_on"]`, которого в истине нет, и тест
        # молча возвращал бы «нет» всегда — то есть цель была бы недостижима по опечатке.
        was = bool(world.state.light)

        def test() -> bool:
            return bool(world.state.light) != was

        return test, "переключить свет: истина мира изменила состояние света"

    return Scenario("переключить", build, goal, budget_ticks=120,
                    why="отличает ли агент выход, меняющий состояние, от выхода без эффекта")


SCENARIOS: dict[str, Callable[[Any], Scenario]] = {
    "дойти": reach_place,
    "переключить": flip_switch,
}
