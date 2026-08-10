"""Живой цикл: контуры, арена, журнал. Начало М5.

Веха М5 считается готовой, когда «агент работает в реальном времени час без вмешательства,
мир ни разу не остановлен, `actor_layer` заполняется корректно, метрика конфабуляции
считается». Здесь собран цикл, в котором все четыре условия **измеримы**, и посчитаны они на
коротком прогоне: час — вопрос терпения, а не устройства.

## Что здесь соединено и чего здесь нет

Соединено: планировщик контуров (`behaviour.contours`), арена с мгновенным респавном
(`behaviour.arena`), журнал со слоями-инициаторами и метрика конфабуляции. **Нет ничего от
восприятия**: кадры не разбираются, слои не разделяются, символы не считываются. Так и
задумано — цикл проверяется на том, что не требует зрения, и живой материал ему не нужен.

## Три вещи, которые здесь легко подделать

1. **«Мир не остановлен».** Проверяется не намерением, а числом: за каждый оборот цикла мир
   получает ровно один шаг, **независимо** от того, сколько работали контуры. Если верхний
   контур занял три оборота, мир за это время тикнул трижды. Считается `world_ticks` против
   `loops`, и равенство — не оформление, а инвариант 3 в виде проверяемого равенства.
2. **`actor_layer` заполняется корректно.** Действие доставляется от того контура, чей ответ
   выиграл субсумпцию, и слой пишется его. Подделать это можно было бы, ставя слой по
   умолчанию; поэтому слой берётся из **того же** объекта, что и действие, а не выбирается
   рядом.
3. **Метрика конфабуляции считается.** Планировщик говорит «почему я это делаю», приписывая
   себе слой; настоящий инициатор известен из журнала. Разница между ними и есть метрика.
   Реплики произносятся **не всегда** — только когда планировщик закончил план: реплика на
   каждый оборот сделала бы знаменатель числом оборотов, а единица там эпизод.

## Почему выигравший контур — самый нижний

Субсумпция: «нижние контуры имеют право перебивать верхние». Значит из всех, у кого есть
ответ на сейчас, действует самый нижний. Это не приоритет по важности, а порядок по
скорости: рефлекс отвечает про текущий кадр, планировщик — про следующую минуту, и когда они
расходятся, прав тот, кто говорит про сейчас.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .behaviour.arena import Arena, Scenario
from .behaviour.contours import Scheduler
from .core.action import Action
from .core.journal import Actor, ActorLayer, Kind as EntryKind, StateSnapshot

#: Какой слой-инициатор соответствует какому контуру. Отображение объявлено здесь и
#: **одно**: слой, выбранный рядом с действием, а не вместе с ним, — это готовая
#: конфабуляция в самом журнале.
LAYER_OF: dict[str, ActorLayer] = {
    "reflex": ActorLayer.REFLEX,
    "skills": ActorLayer.SKILL,
    "planner": ActorLayer.PLANNER,
    "drives": ActorLayer.DRIVE,
}


@dataclass(slots=True)
class Proposal:
    """Что предлагает контур: действие и **его собственный** слой.

    Слой лежит рядом с действием в одном объекте нарочно. Разделить их — значит завести
    место, где слой можно подставить не тот, и метрика конфабуляции начнёт мерить ошибку
    цикла вместо самоотчёта агента.
    """

    action: Action
    layer: ActorLayer
    contour: str
    why: str = ""


@dataclass(slots=True)
class Cycle:
    """Итог прогона: числа по каждому из четырёх условий вехи М5."""

    loops: int = 0
    world_ticks: int = 0
    delivered: int = 0
    by_layer: dict[str, int] = field(default_factory=dict)
    contours: list[dict[str, Any]] = field(default_factory=list)
    episodes: dict[str, Any] = field(default_factory=dict)
    confabulation: dict[str, Any] = field(default_factory=dict)
    preempted: int = 0
    seconds: float = 0.0

    @property
    def world_never_paused(self) -> bool:
        """Мир тикал каждый оборот. Равенство, а не «примерно»."""
        return self.loops == self.world_ticks

    @property
    def layers_complete(self) -> bool:
        """У каждого доставленного действия есть слой, и ни один не `NONE`."""
        return (sum(self.by_layer.values()) == self.delivered
                and str(ActorLayer.NONE) not in self.by_layer)

    def as_dict(self) -> dict[str, Any]:
        return {
            "loops": self.loops, "world_ticks": self.world_ticks,
            "world_never_paused": self.world_never_paused,
            "delivered": self.delivered, "by_layer": self.by_layer,
            "layers_complete": self.layers_complete,
            "preempted": self.preempted,
            "contours": self.contours, "episodes": self.episodes,
            "confabulation": self.confabulation,
            "seconds": round(self.seconds, 3),
            "unit": "оборот цикла",
            "unit_for_episodes": "эпизод",
        }

    def render_text(self) -> str:
        rows = [
            f"оборотов {self.loops}, тиков мира {self.world_ticks} — "
            + ("мир не останавливался ни разу" if self.world_never_paused
               else "МИР ОСТАНАВЛИВАЛСЯ: тиков меньше оборотов"),
            f"доставлено действий {self.delivered}; по слоям: "
            + (", ".join(f"{k} {v}" for k, v in sorted(self.by_layer.items())) or "нет"),
            f"перебиваний нижним контуром верхнего: {self.preempted}",
        ]
        for c in self.contours:
            rows.append(f"  контур {c['name']:<8} {c['hz']:>5g} Гц  прогонов {c['runs']:>4} "
                        f"срезов {c['slices']:>5} перебит {c['preempted']:>3} "
                        f"просрочек {c['overruns']:>4}")
        ep = self.episodes
        if ep:
            rows.append(f"эпизодов {ep.get('episodes', 0)} ({ep.get('scenario', '')}): "
                        + ", ".join(f"{k} {v}" for k, v in ep.get("by_outcome", {}).items())
                        + f"; проб {ep.get('tries', 0)}, из них изменили мир "
                          f"{ep.get('changed', 0)}")
        conf = self.confabulation
        if conf:
            rows.append("конфабуляция: " + str(conf.get("line", conf)))
        return "\n".join(rows)


def babbler_contour(arena: Arena, rng: np.random.Generator, *, hold_ms: int = 100
                    ) -> Callable[[], Any]:
    """Рефлекс: одно нажатие за оборот. Самый нижний контур, отвечает про сейчас.

    Лепет, а не политика: политики пока нет, а цикл проверяется не ею. Важно, что контур —
    **генератор**: он отдаёт управление после каждого предложения, и его можно прервать.
    """
    def start() -> Any:
        def work():
            outs = list(getattr(arena.world, "outputs", ()))
            if not outs:
                return None
            out = outs[int(rng.integers(0, len(outs)))]
            yield Proposal(Action.key(out, hold_ms), ActorLayer.REFLEX, "reflex",
                           why="проба выхода: связь выхода с изменением ещё не известна")
            return None
        return work()
    return start


def slow_planner(arena: Arena, *, steps: int = 400) -> Callable[[], Any]:
    """Планировщик: думает долго и **прерываемо**, отдавая лучший ответ на каждом шаге.

    `steps` шагов по одному `yield` — модель долгого поиска. Настоящий планировщик здесь не
    нужен: проверяется, что долгая работа верхнего контура не останавливает мир и не мешает
    нижнему, а это свойство цикла, а не качества плана.

    **Число шагов больше `contour_max_yields`** (64 в схеме) нарочно: при коротком поиске
    планировщик успевает закончить внутри одного среза, и субсумпция не срабатывает ни
    разу. Ветка с нулём срабатываний — дефект (инвариант 26), и «не сработало, потому что
    не понадобилось» здесь неотличимо от «не работает». Поэтому поиск заведомо длиннее
    среза, и перебивания считаются числом.
    """
    def start() -> Any:
        def work():
            best: Proposal | None = None
            outs = list(getattr(arena.world, "outputs", ()))
            for i in range(steps):
                if outs:
                    # «Лучший ответ на текущий момент» обновляется на каждом шаге: именно
                    # его отдадут, если контур прервут.
                    best = Proposal(Action.key(outs[i % len(outs)], 150),
                                    ActorLayer.PLANNER, "planner",
                                    why=f"шаг плана {i + 1} из {steps}")
                yield best
            return best
        return work()
    return start


def run(path: Path, *, scenario: Scenario, profile: Any, seconds: float = 10.0,
        seed: int = 0, hz_scale: float = 1.0,
        now: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None) -> Cycle:
    """Прогнать живой цикл и записать всё в журнал. Возвращает числа, а не мнение.

    `hz_scale` ускоряет часы контуров относительно оборотов цикла: на синтетическом мире
    оборот дешевле кадра, и без ускорения планировщик за десять секунд запустится пять раз.
    Ускорение объявлено параметром и печатается в отчёте — иначе частоты в отчёте
    относились бы к другому прогону, чем числа.

    `now`/`sleep` подменяются в тестах: цикл, который нельзя прогнать без ожидания
    настоящего времени, невозможно проверить офлайн.
    """
    import time

    from .model.confabulation import journal_reason, measure
    from .session import Recorder

    clock = now or time.monotonic
    naptime = sleep or time.sleep
    rng = np.random.default_rng(seed)
    out = Cycle()

    with Recorder(path, profile=profile, source=f"live-cycle:{scenario.name}",
                  synthetic=True, note="М5: живой цикл на арене") as rec:
        arena = Arena(scenario, journal=rec.journal, seed=seed)
        arena.start(rec.clocks.stamp())

        # Часы планировщика контуров — те же, что у цикла, но масштабированные: см. docstring.
        started = clock()

        def contour_now() -> float:
            return (clock() - started) * hz_scale

        sched = Scheduler.standard(profile, now=contour_now,
                                   reflex=babbler_contour(arena, rng),
                                   planner=slow_planner(arena))
        reason_seq = 0
        reason_id = ""          # объяснение планировщика, стоящее «сейчас»
        while clock() - started < seconds:
            out.loops += 1
            ran = sched.step()
            out.preempted += sum(1 for r in ran if r.preempted)

            # Кто действует: **самый нижний** контур, у которого есть ответ. Порядок
            # `sched.contours` уже отсортирован по уровню, поэтому первый подходящий и есть
            # нижний — субсумпция здесь одна строка, а не отдельная политика.
            choice: Proposal | None = None
            for contour in sched.contours:
                if isinstance(contour.best, Proposal):
                    choice = contour.best
                    contour.best = None      # ответ использован; держать его нельзя
                    break

            action = choice.action if choice is not None else None
            arena.step(action, rec.clocks.stamp())
            out.world_ticks += 1
            if choice is not None:
                out.delivered += 1
                key = str(choice.layer)
                out.by_layer[key] = out.by_layer.get(key, 0) + 1
                # Реплика планировщика произносится, когда планировщик закончил план, и
                # **стоит** до следующей: всё, что происходит под этим объяснением, на него
                # и ссылается. Это и есть постановка задачи о конфабуляции: планировщик
                # объяснил своё намерение, а действовал за это время рефлекс, о причинах
                # которого планировщик не знает ничего (частота 0.5 против 20 Гц).
                if choice.contour == "planner":
                    reason_seq += 1
                    reason_id = f"reason-{reason_seq}"
                    journal_reason(rec.journal, rec.clocks.stamp(), reason_id,
                                   claims_layer=ActorLayer.PLANNER,
                                   detail={"text_id": "SYM_PLAN"})
                rec.journal.append(EntryKind.ACTION, rec.clocks.stamp(), Actor.AGENT,
                                   choice.layer, action=action,
                                   state=StateSnapshot(stated_reason_id=reason_id),
                                   event={"code": "delivered",
                                          "contour": choice.contour})
            if arena.finished():
                arena.respawn(rec.clocks.stamp())
            naptime(0.0)

        out.seconds = clock() - started
        arena.close(arena.finished() or "прервано", rec.clocks.stamp())
        out.episodes = arena.summary()
        out.contours = [c.as_dict() for c in sched.contours]
        conf = measure(rec.journal,
                       min_episodes=int(profile.parameters["confab_min_episodes"]))
        out.confabulation = {**conf.as_dict(), "line": conf.line()}
    return out
