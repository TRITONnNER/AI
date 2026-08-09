"""Цели: из давления драйвов, с обязательным тестом и с бюджетом.

Чего здесь нет и не будет: награды, счёта, очков. Цель не «даёт» ничего — она
либо проходит свой тест, либо не проходит, либо от неё отказываются по бюджету.
Награда завела бы отдельную валюту рядом с ошибкой предсказания, а валюта должна
быть одна.

Четыре правила, каждое проверяемое:

1. **Цель без теста не цель.** Тест — объективная проверка по состоянию мира или
   собственной памяти: «обратимость этого выхода стала известна», «ошибка
   предсказания на этой сущности упала ниже X», «я в этом месте». Формулировка
   вида «стало лучше» невыразима, потому что «лучше» надо откуда-то взять.
2. **У цели есть бюджет в собственных циклах.** Кончился — отказ, и отказ
   пишется в журнал с причиной. Иначе агент будет год добиваться недостижимого,
   и это не будет видно.
3. **Цель знает, откуда она взялась.** Тот же `Provenance`, что у убеждений:
   какой драйв её породил и на какой записи журнала. Цель без происхождения —
   это цель, поставленная снаружи, а такие бывают только от человека и тогда
   помечаются как вмешательство.
4. **Давление выбирает, но не назначает.** Из давления драйвов получается
   *порядок* кандидатов, а сами кандидаты берутся из того, что известно: какие
   выходы не изучены, какие убеждения держатся на чужих словах, какие места
   давно не посещались. Драйв не может потребовать того, о чём агент не знает.

Отказ от цели — не провал. Это тоже опыт, и он же источник компетентности: по
доле целей, прошедших тест, видно, растёт агент или ходит по кругу.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Iterable

from ..core.clocks import Stamp
from ..core.journal import Actor, ActorLayer, Journal, Kind as EntryKind
from ..core.profile import Profile
from ..model.beliefs import Origin, Provenance
from ..model.drives import Motivation
from ..model.rebuild import BodyMap


class GoalError(ValueError):
    pass


class State(StrEnum):
    ACTIVE = "active"
    DEFERRED = "deferred"      # отложена: место занято более срочной
    PASSED = "passed"          # тест пройден
    ABANDONED = "abandoned"    # бюджет кончился
    REJECTED = "rejected"      # забракована исследователем — вмешательство


# Что вообще бывает целью. Набор закрыт: новый вид цели означает новый способ
# её проверить, а не свободную строку.
KIND_LEARN_OUTPUT = "learn_output"       # выяснить, что делает выход
KIND_UNDO_OUTPUT = "undo_output"         # найти, чем откатить выход
KIND_VERIFY_HEARSAY = "verify_hearsay"   # проверить своими руками чужое слово
KIND_REACH_PLACE = "reach_place"         # добраться до места
KIND_REDUCE_ERROR = "reduce_error"       # понять то, что удивляет


@dataclass(slots=True)
class Goal:
    """Одна цель. `test` — функция, которая либо да, либо нет."""

    id: str
    kind: str
    target: str                          # непрозрачный идентификатор
    test: Callable[[], bool]
    test_text: str                       # как проверка звучит для исследователя
    budget_ticks: int
    provenance: Provenance
    drive: str                           # какой драйв её породил
    pressure: float                      # давление на момент постановки
    spent_ticks: int = 0
    state: State = State.ACTIVE
    reason: str | None = None            # почему брошена или забракована

    def __post_init__(self) -> None:
        if not callable(self.test):
            raise GoalError(
                f"цель {self.id} без способа проверки. Цель, которую нельзя "
                "проверить объективно, не участвует в работе")
        if not self.test_text:
            raise GoalError(f"цель {self.id}: тест не описан словами для исследователя")
        if self.budget_ticks <= 0:
            raise GoalError(f"цель {self.id}: бюджет должен быть положительным")
        if self.provenance.origin is not Origin.EXPERIENCE and self.drive != "human":
            raise GoalError(
                "цель рождается из собственного опыта; цель от человека помечается "
                "драйвом 'human' и считается вмешательством")

    @property
    def left_ticks(self) -> int:
        return max(0, self.budget_ticks - self.spent_ticks)

    @property
    def overdue(self) -> bool:
        return self.spent_ticks >= self.budget_ticks

    @property
    def finished(self) -> bool:
        return self.state in (State.PASSED, State.ABANDONED, State.REJECTED)

    def check(self) -> bool:
        return bool(self.test())

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "target": self.target,
                "state": str(self.state), "drive": self.drive,
                "pressure": round(self.pressure, 6),
                "test": self.test_text, "budget_ticks": self.budget_ticks,
                "spent_ticks": self.spent_ticks, "left_ticks": self.left_ticks,
                "reason": self.reason, "provenance": self.provenance.as_dict()}


@dataclass(slots=True)
class Candidate:
    """Кандидат в цели: что можно было бы сделать и насколько это востребовано."""

    kind: str
    target: str
    drive: str
    gain: float                          # ожидаемая польза: сколько неизвестного снимет
    test: Callable[[], bool]
    test_text: str

    def priority(self, pressure: dict[str, float]) -> float:
        """Давление драйва, помноженное на ожидаемую пользу.

        Названо приоритетом, а не оценкой: «оценка» в этом месте читалась бы как
        награда, а награды в проекте нет — валюта одна, ошибка предсказания.
        Приоритет решает только порядок кандидатов и ничего не начисляет.

        Ни то, ни другое по отдельности не годится: давление без пользы даёт цель,
        которая ничего не выяснит, а польза без давления — цель, которая никому не
        нужна прямо сейчас.
        """
        return pressure.get(self.drive, 0.0) * self.gain


class GoalStack:
    """Стек целей: одна активная, остальные отложены. Всё пишется в журнал."""

    def __init__(self, profile: Profile, *, journal: Journal | None = None) -> None:
        p = profile.parameters
        self.profile = profile
        self.journal = journal
        # Бюджет цели — в собственных циклах агента, а не в секундах стены:
        # у агента свои часы, и мерить его терпение чужими было бы неверно.
        self.default_budget = max(1, int(float(p["drive_horizon_s"]) * float(p["hz_skills"])))
        # Может ли человеческая речь ставить агенту цели. По умолчанию нет, и это
        # не осторожность, а инвариант 10: реакция только на поведение и объективные
        # величины. Ручка структурная — прогон, в котором человек ставит цели, это
        # другой эксперимент, и смешивать его с чистым нельзя.
        self.human_speech_affects_goals = bool(
            profile.structural["human_speech_affects_goals"])
        self.goals: list[Goal] = []
        self._counter = 0
        self.passed = 0
        self.abandoned = 0
        self.rejected = 0
        self._spent_on_passed: list[int] = []
        self._spent_on_abandoned: list[int] = []

    # --- постановка ---------------------------------------------------------

    def _next_id(self) -> str:
        self._counter += 1
        return f"GOAL_{self._counter:04d}"

    @property
    def active(self) -> Goal | None:
        """Активной может быть только одна цель.

        Субсумпция здесь не нужна: это уровень «что вообще делать», и делать
        одновременно два дела нельзя не по техническим причинам, а по смыслу.
        Прерываемость живёт ниже — в контурах.
        """
        for g in self.goals:
            if g.state is State.ACTIVE:
                return g
        return None

    def push(self, candidate: Candidate, motivation: Motivation, seq: int,
             branch: str, stamp: Stamp | None = None, *,
             budget_ticks: int | None = None) -> Goal:
        if candidate.drive == "human" and not self.human_speech_affects_goals:
            raise GoalError(
                "цель от человеческой речи запрещена этим профилем "
                "(`human_speech_affects_goals=False`). Инвариант 10: реакция только "
                "на поведение и объективные величины. Если это нужно замерить — "
                "включите ручку, и журнал форкнется")
        pressure = motivation.goal_pressure()
        goal = Goal(
            id=self._next_id(), kind=candidate.kind, target=candidate.target,
            test=candidate.test, test_text=candidate.test_text,
            budget_ticks=budget_ticks or self.default_budget,
            provenance=Provenance(Origin.EXPERIENCE, branch, seq),
            drive=candidate.drive, pressure=candidate.priority(pressure))

        current = self.active
        if current is not None:
            if goal.pressure > current.pressure * 1.5:
                # Новая цель заметно важнее — старая откладывается, а не
                # выбрасывается: к ней можно вернуться, и её бюджет цел.
                current.state = State.DEFERRED
                self._journal(current, "deferred", stamp,
                              reason=f"вытеснена целью {goal.id}")
            else:
                goal.state = State.DEFERRED

        self.goals.append(goal)
        self._journal(goal, "set", stamp)
        return goal

    # --- ход ----------------------------------------------------------------

    def tick(self, stamp: Stamp | None = None, *, ticks: int = 1) -> Goal | None:
        """Один такт: списать бюджет активной цели, проверить тест, решить судьбу.

        Возвращает активную цель после такта — возможно, другую, если предыдущая
        закончилась.
        """
        goal = self.active
        if goal is None:
            self._promote(stamp)
            return self.active

        goal.spent_ticks += ticks
        if goal.check():
            goal.state = State.PASSED
            self.passed += 1
            self._spent_on_passed.append(goal.spent_ticks)
            self._journal(goal, "passed", stamp)
            self._promote(stamp)
        elif goal.overdue:
            goal.state = State.ABANDONED
            goal.reason = (f"бюджет {goal.budget_ticks} циклов кончился, "
                           "тест не пройден")
            self.abandoned += 1
            self._spent_on_abandoned.append(goal.spent_ticks)
            self._journal(goal, "abandoned", stamp, reason=goal.reason)
            self._promote(stamp)
        return self.active

    def _promote(self, stamp: Stamp | None) -> None:
        """Поднять самую востребованную из отложенных."""
        deferred = [g for g in self.goals if g.state is State.DEFERRED]
        if not deferred:
            return
        best = max(deferred, key=lambda g: (g.pressure, -g.spent_ticks, g.id))
        best.state = State.ACTIVE
        self._journal(best, "resumed", stamp)

    def reject(self, goal_id: str, stamp: Stamp | None = None, *,
               reason: str = "забракована исследователем") -> Goal:
        """Браковка целей исследователем.

        Пишется видом записи INTERVENTION, а не GOAL: это вмешательство, и прогон
        с вмешательствами нельзя сравнивать с чистым как равный (см. разбор пульта,
        пункт 6).
        """
        goal = next((g for g in self.goals if g.id == goal_id), None)
        if goal is None:
            raise GoalError(f"нет цели {goal_id}")
        goal.state = State.REJECTED
        goal.reason = reason
        self.rejected += 1
        if self.journal is not None and stamp is not None:
            self.journal.append(EntryKind.INTERVENTION, stamp, Actor.HUMAN,
                                ActorLayer.HUMAN,
                                event={"code": "goal_rejected", "goal": goal.id,
                                       "kind": goal.kind, "reason": reason})
        self._promote(stamp)
        return goal

    def _journal(self, goal: Goal, code: str, stamp: Stamp | None,
                 reason: str | None = None) -> None:
        if self.journal is None or stamp is None:
            return
        event = {"code": code, **goal.as_dict()}
        if reason:
            event["reason"] = reason
        event.pop("test", None)         # текст теста — для исследователя, не для журнала
        # Переход цели начинает контур драйвов: цель существует потому, что
        # какой-то драйв вне коридора, и никакой другой слой её не заводит.
        # Забракованная исследователем цель пишется отдельным вызовом со слоем
        # human — там инициатор действительно другой.
        self.journal.append(EntryKind.GOAL, stamp, Actor.AGENT, ActorLayer.DRIVE,
                            event=event)

    # --- сводка -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        total = self.passed + self.abandoned
        return {
            "goals": len(self.goals),
            "active": self.active.id if self.active else None,
            "deferred": sum(1 for g in self.goals if g.state is State.DEFERRED),
            "passed": self.passed,
            "abandoned": self.abandoned,
            "rejected": self.rejected,
            "pass_rate": round(self.passed / total, 4) if total else 0.0,
            # «Время до отказа от цели» — одна из шести метрик из пульта.
            "mean_ticks_to_abandon": round(
                sum(self._spent_on_abandoned) / len(self._spent_on_abandoned), 2)
            if self._spent_on_abandoned else None,
            "mean_ticks_to_pass": round(
                sum(self._spent_on_passed) / len(self._spent_on_passed), 2)
            if self._spent_on_passed else None,
        }


# ---------------------------------------------------------------------------
# Откуда берутся кандидаты
# ---------------------------------------------------------------------------


def candidates_from_body(body: BodyMap, outputs: Iterable[str], *,
                         caution_threshold: float) -> list[Candidate]:
    """Что можно узнать про своё тело. Драйв — любопытство и целостность.

    Ожидаемая польза считается из того, чего не знаем: непробованный выход
    снимает больше неизвестного, чем выход с неизвестной обратимостью, а тот —
    больше, чем уже изученный.
    """
    out: list[Candidate] = []
    for output in outputs:
        f = body.outputs.get(output)
        if f is None or f.state == "untried":
            out.append(Candidate(
                KIND_LEARN_OUTPUT, output, "curiosity", gain=1.0,
                test=(lambda o=output: body.outputs.get(o) is not None
                      and body.outputs[o].delivered > 0),
                test_text=f"выход {output} хотя бы раз доехал до устройства"))
            continue
        if f.state == "live" and not f.reversibility.is_known:
            out.append(Candidate(
                KIND_UNDO_OUTPUT, output, "integrity", gain=0.8,
                test=(lambda o=output: body.outputs[o].reversibility.is_known),
                test_text=f"обратимость {output} стала известна"))
    del caution_threshold          # осторожность влияет на порядок проб, не на цели
    return out


def candidates_from_beliefs(store, *, limit: int = 8) -> list[Candidate]:
    """Что стоит проверить своими руками. Драйв — порядок.

    Берутся убеждения, держащиеся только на чужих словах. Польза тем выше, чем
    больше на этом утверждении висит: пересказ, на который опираются, опаснее
    пересказа, о котором забыли.
    """
    out: list[Candidate] = []
    for belief in sorted(store.hearsay(), key=lambda b: (-b.n, b.claim))[:limit]:
        claim = belief.claim

        def passed(c=claim) -> bool:
            for b in store.beliefs():
                if b.claim == c:
                    return not b.is_hearsay
            return False

        out.append(Candidate(
            KIND_VERIFY_HEARSAY, claim, "order", gain=min(1.0, 0.3 + 0.1 * belief.n),
            test=passed,
            test_text=f"утверждение {claim} проверено своими руками"))
    return out


def candidates_from_places(graph, *, stale_after: int = 200) -> list[Candidate]:
    """Куда стоит вернуться. Драйв — любопытство.

    Место, которое давно не видели, могло измениться, и модель про него устарела.
    Польза тем выше, чем давнее там были и чем меньше от него известно рёбер.
    """
    out: list[Candidate] = []
    if not graph.places:
        return out
    newest = max(p.last_seq for p in graph.places.values())
    for place in graph.places.values():
        age = newest - place.last_seq
        if age < stale_after:
            continue
        edges = len(graph.neighbours(place.id))
        out.append(Candidate(
            KIND_REACH_PLACE, place.id, "curiosity",
            gain=min(1.0, 0.2 + 0.1 * age / max(1, stale_after) + 0.1 * (3 - min(3, edges))),
            test=(lambda pid=place.id: graph.current == pid),
            test_text=f"я снова в месте {place.id}"))
    return out


def candidates_from_error(error, entity_ids: Iterable[str], *,
                          threshold: float | None = None) -> list[Candidate]:
    """Что удивляет и потому требует понимания. Драйв — компетентность.

    Порог берётся из самого предсказателя (`novelty_threshold` профиля), а не
    задаётся здесь: иначе появилась бы вторая, расходящаяся с первой, мера
    новизны.
    """
    limit = error.novelty_threshold if threshold is None else threshold
    out: list[Candidate] = []
    for ent in entity_ids:
        out.append(Candidate(
            KIND_REDUCE_ERROR, ent, "competence", gain=0.6,
            test=(lambda: error.summary()["mean"] < limit),
            test_text=f"средняя ошибка предсказания опустилась ниже {limit:.2f}"))
    return out


def choose(candidates: list[Candidate], motivation: Motivation, *,
           top: int = 1) -> list[Candidate]:
    """Отобрать кандидатов по давлению драйвов и ожидаемой пользе."""
    pressure = motivation.goal_pressure()
    ranked = sorted(candidates, key=lambda c: (-c.priority(pressure), c.kind, c.target))
    return ranked[:top]
