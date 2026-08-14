"""Модель меня в чужих глазах и цель с внешним судьёй. TASK-32, направление D.

Есть цели, чей тест — наблюдение: «я в этом месте», «полоска полна». И есть цели, качество
которых определяют **другие**: научная работа, текст, рисунок, поступок на людях. Без второго
рода целей недоступен целый класс задач, и подменить его первым нельзя: наблюдением проверить
«хорошо ли это» невозможно в принципе.

## Тест такой цели — предсказание, и это опасное место

Тест цели с судьёй — предсказание чужой оценки. Значит агент мог бы «выполнять» такие цели,
предсказывая одобрение, и никакое наблюдение его бы не поправило. Это ровно та порча, от
которой стоит инвариант 10: самоотчёты ни на что не влияют автоматически.

Поэтому здесь **два разных состояния**, и путать их нельзя ни в одном месте кода:

- `predicted` — агент считает, что судья одобрит. Годится для планирования и ни для чего
  больше;
- `settled` — судья ответил. Только это закрывает цель, и только это идёт в статистику.

Предсказание **никогда** не переводит цель в выполненную. Метод, который мог бы это сделать,
отсутствует, а не запрещён соглашением.

## Что измеряется

Не «сколько целей выполнено», а **калибровка**: расхождение предсказанной оценки с
полученной. Единица независимости — **оценка** (одна пара «предсказал / получил»): оценки
одного судьи по одной работе зависимы, а разные работы независимы.

Калибровка — это и есть польза `ReflectedSelf`: модель, которая ошибается на 0.5 из 1.0,
хуже отсутствия модели, и это видно числом, а не мнением.

## Чего здесь нет

**Судьи.** Канал чужой оценки приходит извне (речь оператора, разметка домена, отзыв), и
выдумывать его нельзя: детектор режима на всех нынешних доменах отвечает «внешний судья не
определён» (`perception/regime.py`), и цель с судьёй объявлена там **неприменимой**. Механизм
готов, измерять его пока нечем, и это сказано числом покрытия, а не тишиной.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from ..behaviour.goals import Goal, GoalError
from .beliefs import Origin, Provenance


class JudgedError(GoalError):
    pass


@dataclass(frozen=True, slots=True)
class Judgement:
    """Одна оценка: кто, о чём, какая. Всё непрозрачно, кроме числа.

    `judge` и `aspect` — непрозрачные символы (`SYM_*`, `JUDGE_*`): кто именно судит и что
    именно оценивает, агент знает как ярлык, а не как имя. Значение — доля в [0,1], потому
    что сравнивать предсказание с ответом надо в одной шкале, а шкалу судьи никто не знает;
    доля — самое слабое допущение, какое здесь возможно.
    """

    judge: str
    aspect: str
    value: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise JudgedError(f"оценка вне [0,1]: {self.value}")
        if not self.judge or not self.aspect:
            raise JudgedError("оценка без судьи или без предмета не хранится")

    @property
    def key(self) -> tuple[str, str]:
        return (self.judge, self.aspect)


@dataclass(slots=True)
class Reflected:
    """Что я думаю о том, как меня оценят по одной паре «судья × предмет»."""

    judge: str
    aspect: str
    mu: float = 0.5              # ожидаемая оценка; 0.5 — «не знаю», а не «средне»
    sigma: float = 1.0
    n: int = 0
    #: Абсолютные промахи предсказаний, по одному на полученную оценку. Хранится список, а
    #: не среднее: калибровка считается по промахам, и среднее их бы уже смешало.
    misses: list[float] = field(default_factory=list)

    @property
    def known(self) -> bool:
        return self.n > 0

    def observe(self, predicted: float, actual: float) -> "Reflected":
        """Судья ответил. Обновляет и ожидание, и промах."""
        self.misses.append(abs(predicted - actual))
        self.n += 1
        self.mu = ((self.mu * (self.n - 1)) + actual) / self.n
        spread = sum((m - (sum(self.misses) / len(self.misses))) ** 2
                     for m in self.misses) / len(self.misses)
        self.sigma = spread ** 0.5 if self.n > 1 else 0.5
        return self

    @property
    def calibration(self) -> float | None:
        """Средний промах предсказания. `None` — ни одной полученной оценки."""
        return (sum(self.misses) / len(self.misses)) if self.misses else None

    def as_dict(self) -> dict[str, Any]:
        return {"judge": self.judge, "aspect": self.aspect,
                "mu": round(self.mu, 4), "sigma": round(self.sigma, 4), "n": self.n,
                "calibration": (None if self.calibration is None
                                else round(self.calibration, 4)),
                "unit": "оценка"}


@dataclass(slots=True)
class ReflectedSelf:
    """Модель меня в чужих глазах: `mu`, `sigma`, `n` по каждой паре судья × предмет.

    Отдельно от модели других нарочно: «я моделирую судью» и «я моделирую, как судья
    моделирует меня» — разные утверждения, и без второго нет ни предвидения последствий
    поступка, ни репутации, ни осторожности в словах.
    """

    by_key: dict[tuple[str, str], Reflected] = field(default_factory=dict)

    def expect(self, judge: str, aspect: str) -> Reflected:
        key = (judge, aspect)
        got = self.by_key.get(key)
        if got is None:
            got = Reflected(judge, aspect)
            self.by_key[key] = got
        return got

    def predict(self, judge: str, aspect: str) -> float | None:
        """Ожидаемая оценка. `None` — этого судью по этому предмету ещё не видели.

        `None`, а не 0.5: «жду половину» и «не знаю, чего ждать» ведут к разным решениям,
        и подменять второе первым значило бы выдать незнание за знание.
        """
        got = self.by_key.get((judge, aspect))
        return got.mu if got is not None and got.known else None

    def observe(self, judgement: Judgement, *, predicted: float) -> Reflected:
        """Получена оценка. Единственный вход, меняющий модель."""
        return self.expect(judgement.judge, judgement.aspect).observe(
            predicted, judgement.value)

    def calibration(self) -> dict[str, Any]:
        """Насколько модель ошибается. Главное число этого механизма.

        Модель, ошибающаяся на 0.5 из 1.0, хуже отсутствия модели — это видно числом.
        Единица независимости — **оценка**: две оценки одного судьи по одной работе
        зависимы, разные работы независимы.
        """
        misses = [m for r in self.by_key.values() for m in r.misses]
        return {
            "pairs": len(self.by_key),
            "judged": len(misses),
            "unit": "оценка",
            "miss_mean": (sum(misses) / len(misses)) if misses else None,
            "miss_worst": max(misses) if misses else None,
            "better_than_nothing": (None if not misses
                                    else (sum(misses) / len(misses)) < 0.5),
            "why": ("промах 0.5 из 1.0 — это ответ наугад: модель хуже отсутствия модели, "
                    "и решать по ней нельзя"),
        }

    def as_dict(self) -> dict[str, Any]:
        return {"reflected": [r.as_dict() for r in self.by_key.values()],
                "calibration": self.calibration()}


@dataclass(slots=True)
class JudgedGoal:
    """Цель, чей тест — предсказание чужой оценки, а не наблюдение.

    Обёртка над обычной `Goal`, а не её подвид: у обычной цели тест **проверяет мир**, и
    делать вид, что предсказание — тоже проверка, значило бы стереть единственное
    различие, которое здесь важно.

    Закрывает цель только `settle`, то есть полученная оценка. Предсказание закрывать цель
    не умеет: такого метода нет.
    """

    goal: Goal
    judge: str
    aspect: str
    want: float                          # какая оценка считается успехом
    predicted: float | None = None       # что агент ожидает; None — не предсказывал
    verdict: Judgement | None = None     # что ответил судья; None — ещё не отвечал

    def __post_init__(self) -> None:
        if self.goal.kind != "judged":
            raise JudgedError(
                f"цель {self.goal.id} обёрнута как судимая, но её вид «{self.goal.kind}». "
                "Вид обязан отличаться: по нему читатель понимает, что тест здесь — "
                "предсказание, а не наблюдение")
        if not 0.0 <= self.want <= 1.0:
            raise JudgedError(f"порог успеха вне [0,1]: {self.want}")

    @property
    def settled(self) -> bool:
        return self.verdict is not None

    @property
    def passed(self) -> bool | None:
        """Выполнена ли цель. `None` — судья ещё не ответил, и это **не** «нет».

        Здесь и стоит защита инварианта 10: пока оценки нет, ответа нет. Никакое
        предсказание не делает `passed` истинным — ни своё, ни уверенное.
        """
        if self.verdict is None:
            return None
        return self.verdict.value >= self.want

    def predict(self, reflected: ReflectedSelf) -> float | None:
        """Предсказать оценку по модели себя в чужих глазах. Годится для планирования."""
        self.predicted = reflected.predict(self.judge, self.aspect)
        return self.predicted

    def settle(self, judgement: Judgement, reflected: ReflectedSelf) -> Reflected:
        """Судья ответил: цель закрывается, модель учится на промахе.

        Оценка не того судьи или не о том предмете отвергается: подставить чужую оценку
        значило бы научить модель на данных, которых она не предсказывала.
        """
        if judgement.key != (self.judge, self.aspect):
            raise JudgedError(
                f"оценка от {judgement.judge} о {judgement.aspect} не относится к цели "
                f"{self.goal.id}, которая ждёт {self.judge} о {self.aspect}")
        if self.predicted is None:
            raise JudgedError(
                f"цель {self.goal.id} закрывается оценкой, но предсказания не было. "
                "Тогда нечему учиться: калибровка считается по промаху, а промах — это "
                "разница между предсказанным и полученным")
        self.verdict = judgement
        return reflected.observe(judgement, predicted=self.predicted)

    def as_dict(self) -> dict[str, Any]:
        return {"goal": self.goal.id, "kind": self.goal.kind,
                "judge": self.judge, "aspect": self.aspect, "want": self.want,
                "predicted": self.predicted,
                "verdict": (None if self.verdict is None else self.verdict.value),
                "settled": self.settled, "passed": self.passed,
                "test_is": "предсказание чужой оценки, а не наблюдение"}


def judged_goal(*, goal_id: str, judge: str, aspect: str, want: float,
                budget_ticks: int, branch: str, seq: int,
                drive: str = "competence", pressure: float = 0.5) -> JudgedGoal:
    """Собрать цель с внешним судьёй. Тест — предсказание, и он назван так вслух.

    `test` обычной цели здесь возвращает **всегда `False`**, и это не заглушка: наблюдением
    такая цель не проверяется никогда, а `Goal` требует функцию. Единственный законный
    способ её закрыть — `settle`, и `test_text` говорит об этом прямо, чтобы читатель
    журнала не искал наблюдение, которого не будет.
    """
    goal = Goal(
        id=goal_id, kind="judged", target=aspect,
        test=lambda: False,
        test_text=(f"наблюдением не проверяется: оценку даёт {judge} о {aspect}; "
                   f"успех — оценка не ниже {want}"),
        budget_ticks=budget_ticks,
        provenance=Provenance(Origin.EXPERIENCE, branch, seq),
        drive=drive, pressure=pressure)
    return JudgedGoal(goal=goal, judge=judge, aspect=aspect, want=want)


def pending(goals: Iterable[JudgedGoal]) -> list[JudgedGoal]:
    """Цели, ждущие оценки. Отдельный список: ожидание — не провал и не успех."""
    return [g for g in goals if not g.settled]
