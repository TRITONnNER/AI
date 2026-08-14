"""Канал оценки от оператора: «получилось / частично / нет». TASK-33, направление C.

`JudgedGoal` был мёртв — не потому, что плохо устроен, а потому что судьи не было ни у
одного домена. Детектор режима отвечал «внешний судья не определён» на всех пяти, цель с
судьёй объявлялась неприменимой, а калибровка делила на ноль. Простейший честный канал:
**после эпизода оператор ставит отметку**, и она входит в модель как оценка судьи.

## Три отметки, а не две

`получилось`, `частично`, `нет`. Третья не «серединка для удобства»: без неё оператор
вынужден врать в одну из сторон, а промах предсказания — величина, по которой считается
калибровка, и врать в ней значит портить единственное честное число этого механизма. Во
что превращается `частично`, объявлено настройкой `mark_partial_value`: это не «примерно
половина», а порог, от которого зависят все числа калибровки (инвариант 23).

## Отметка — показание, а не команда

Три границы, и ни одну нельзя стереть:

1. **Отметка не ставит и не меняет цель.** Она отвечает на вопрос «как вышло», заданный
   уже поставленной целью. Путь «оператор сказал → агент сделал» отсутствует здесь, как и
   везде (инвариант 21): в этом модуле нет ни одной функции, порождающей действие.
2. **Отметку не может поставить агент.** У отметки есть источник, и `Source.AGENT` не
   существует. Проверяется тестом: `behaviour/` не импортирует этот модуль — тот же
   способ, которым охраняется отладочный канал (инвариант 12).
3. **Отметка не закрывает цель сама.** Она превращается в `Judgement`, а закрывает цель
   `JudgedGoal.settle`, который требует, чтобы предсказание уже существовало. Значит
   промах всегда определён, и «выполнено» никогда не выводится из собственной уверенности
   агента (инвариант 10).

## Доверие к источнику

У отметки есть `trust` — как у показания (`Testimony`), и по той же причине: оператор
бывает невнимателен, а разметка домена — устаревшей. Доверие **весит обновление модели**:
отметка с доверием 0.25 сдвигает ожидание вчетверо слабее. Ноль доверия означает «принял к
сведению и не учился», и такая отметка всё равно закрывает цель — потому что закрывает её
факт ответа судьи, а не согласие агента с ним.

## Очередь, а не блокирующий диалог

Оператор отвечает, когда отвечает (инвариант 19). Поэтому канал — очередь: эпизод
закончился, запрос встал в очередь, агент продолжает жить. Неотвеченный запрос остаётся
`None` — «судья не ответил», а не «судья отказал». Смешать эти два — значит объявить
провалом всё, о чём оператор просто не успел сказать.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

from .reflected import Judgement, JudgedError, JudgedGoal, ReflectedSelf


class Mark(StrEnum):
    """Что оператор отметил. Ровно три исхода."""

    DONE = "получилось"
    PARTLY = "частично"
    FAILED = "нет"


class Source(StrEnum):
    """Кто поставил отметку. Агента здесь нет и быть не может.

    `OPERATOR` — человек за пультом. `DOMAIN` — разметка самого домена (файл ответов,
    счётчик уровня): тоже внешняя оценка, но доверие к ней своё, потому что она бывает
    устаревшей. `PEER` — другой экземпляр агента, и это **не** «я сам»: у него своя
    история и свои ошибки, поэтому его отметка — показание, а не самоотчёт.
    """

    OPERATOR = "оператор"
    DOMAIN = "домен"
    PEER = "другой экземпляр"


def value_of(mark: Mark, *, partial: float) -> float:
    """Во что превращается отметка. `partial` — из профиля, а не из кода.

    Края шкалы не настраиваются: «получилось» — это единица, «нет» — ноль, иначе шкала
    перестаёт быть долей и сравнивать предсказание с ответом станет нечем.
    """
    if not 0.0 <= partial <= 1.0:
        raise JudgedError(f"значение «частично» вне [0,1]: {partial}")
    return {Mark.DONE: 1.0, Mark.PARTLY: float(partial), Mark.FAILED: 0.0}[mark]


@dataclass(frozen=True, slots=True)
class Request:
    """Запрос оценки: эпизод кончился, оператора спросили, ответа ещё нет.

    `predicted` записывается **в момент запроса**, а не при ответе. Иначе предсказание
    можно было бы подставить задним числом, зная оценку, и калибровка стала бы отчётом об
    удачливости, а не о модели.
    """

    goal_id: str
    judge: str
    aspect: str
    predicted: float | None
    asked_at: int                        # t_self, когда спросили

    def as_dict(self) -> dict[str, Any]:
        return {"goal": self.goal_id, "judge": self.judge, "aspect": self.aspect,
                "predicted": self.predicted, "asked_at": self.asked_at}


@dataclass(frozen=True, slots=True)
class Answer:
    """Ответ оператора на запрос. Отметка плюс доверие к источнику."""

    goal_id: str
    mark: Mark
    source: Source
    trust: float
    answered_at: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.trust <= 1.0:
            raise JudgedError(f"доверие вне [0,1]: {self.trust}")

    def as_dict(self) -> dict[str, Any]:
        return {"goal": self.goal_id, "mark": str(self.mark),
                "source": str(self.source), "trust": self.trust,
                "answered_at": self.answered_at}


@dataclass(slots=True)
class MarkQueue:
    """Очередь запросов оценки. Не блокирует ни одного цикла.

    Хранит и отвеченные тоже: неотвеченный запрос — это открытый вопрос, и терять его
    нельзя (та же логика, что у отложенных гипотез в конвейере убеждений).
    """

    partial: float
    trust_operator: float
    trust_domain: float
    trust_peer: float
    pending: list[Request] = field(default_factory=list)
    answered: list[tuple[Request, Answer]] = field(default_factory=list)
    #: Отметки, пришедшие ни на один запрос. Не выбрасываются: оператор мог отметить то,
    #: о чём агент не спрашивал, и это данные о нём, а не мусор.
    unmatched: list[Answer] = field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: Any) -> "MarkQueue":
        p = profile.parameters
        return cls(partial=float(p["mark_partial_value"]),
                   trust_operator=float(p["mark_trust_operator"]),
                   trust_domain=float(p["mark_trust_domain"]),
                   trust_peer=float(p["mark_trust_peer"]))

    def trust_of(self, source: Source) -> float:
        return {Source.OPERATOR: self.trust_operator,
                Source.DOMAIN: self.trust_domain,
                Source.PEER: self.trust_peer}[source]

    # --- сторона агента: спросить -------------------------------------------

    def ask(self, goal: JudgedGoal, reflected: ReflectedSelf, *, at: int) -> Request:
        """Спросить оценку. Предсказание делается здесь же и запоминается.

        Предсказание может быть `None` — этого судью по этому предмету ещё не видели. Тогда
        ответ придёт, но цель им закрыть будет нельзя: промах без предсказания не
        существует. Это не потеря — это первая встреча с судьёй, и она честно ничему не
        учит модель, кроме самого `mu`.
        """
        predicted = goal.predict(reflected)
        req = Request(goal.goal.id, goal.judge, goal.aspect, predicted, int(at))
        self.pending.append(req)
        return req

    # --- сторона оператора: ответить ----------------------------------------

    def answer(self, goal_id: str, mark: Mark, *, source: Source = Source.OPERATOR,
               at: int = 0, trust: float | None = None) -> Answer:
        """Оператор поставил отметку. Ничего в поведении агента при этом не меняется."""
        got = Answer(goal_id, Mark(mark), Source(source),
                     self.trust_of(Source(source)) if trust is None else float(trust),
                     int(at))
        for i, req in enumerate(self.pending):
            if req.goal_id == goal_id:
                self.pending.pop(i)
                self.answered.append((req, got))
                return got
        self.unmatched.append(got)
        return got

    # --- сведение: отметка становится оценкой судьи -------------------------

    def settle(self, goals: Iterable[JudgedGoal],
               reflected: ReflectedSelf) -> list[dict[str, Any]]:
        """Провести все полученные ответы через `JudgedGoal.settle`.

        Возвращает по строке на каждую пару «запрос — ответ»: что предсказали, что
        получили, закрылась ли цель и почему нет, если нет. Строки идут в отчёт
        исследователю; поведение агента от них не зависит.
        """
        by_id = {g.goal.id: g for g in goals}
        out: list[dict[str, Any]] = []
        rest: list[tuple[Request, Answer]] = []
        for req, ans in self.answered:
            goal = by_id.get(req.goal_id)
            row: dict[str, Any] = {"goal": req.goal_id, "mark": str(ans.mark),
                                   "source": str(ans.source), "trust": ans.trust,
                                   "predicted": req.predicted,
                                   "value": value_of(ans.mark, partial=self.partial),
                                   "waited": ans.answered_at - req.asked_at}
            if goal is None:
                row["settled"] = False
                row["why"] = "цели с таким идентификатором нет среди переданных"
                out.append(row)
                continue
            judgement = Judgement(req.judge, req.aspect, row["value"], trust=ans.trust)
            try:
                # Предсказание берётся из запроса, а не из цели: цель могла успеть
                # предсказать заново, и тогда промах считался бы по более свежему
                # предсказанию, чем то, о котором спрашивали.
                goal.predicted = req.predicted
                goal.settle(judgement, reflected)
                row["settled"] = True
                row["passed"] = goal.passed
                row["miss"] = (None if req.predicted is None
                               else abs(req.predicted - row["value"]))
            except JudgedError as e:
                row["settled"] = False
                row["why"] = str(e)
            out.append(row)
        self.answered = rest
        return out

    # --- сводка -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        waits = [a.answered_at - r.asked_at for r, a in self.answered]
        return {
            "pending": len(self.pending),
            "answered": len(self.answered),
            "unmatched": len(self.unmatched),
            "unit": "оценка",
            "wait_mean": (sum(waits) / len(waits)) if waits else None,
            "partial_value": self.partial,
            "why_pending_is_not_failure": (
                "неотвеченный запрос — «судья не ответил», а не «судья отказал». "
                "Смешать эти два значит объявить провалом всё, о чём оператор не успел "
                "сказать"),
        }
