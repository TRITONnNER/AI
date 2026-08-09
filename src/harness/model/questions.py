"""Открытые вопросы: перечень непонятого и механизм его переоткрытия.

Третий исход конвейера убеждений (инвариант 16) существует ради одной вещи: чтобы агент
**не забывал свои загадки**. Отказ по перерасходу выбрасывает очередь дел — и это
правильно, — но вместе с ней выбросил бы и всё, чего агент не понял, если бы непонятое
лежало в той же очереди. Поэтому `deferred` — отдельное множество, оно не тратит бюджет и
не удаляется.

Само по себе отдельное множество бесполезно. Нужен механизм, который **возвращает** из
него в работу, иначе перечень непонятого превращается в кладбище: агент аккуратно
записывает загадки и никогда к ним не возвращается, что ровно то же, что забыть.

## Как работает переоткрытие

У отложенной гипотезы хранится не только «почему теста нет» (`deferred_reason`), но и
**чего не хватало** (`reopens_on`). При изменении множества доступного — новое место,
новый навык, умение читать, встреча с новым типом сущностей, новое свидетельство — реестр
проверяет, не появилось ли недостающее, и предлагает построить тест.

**Предлагает, а не строит.** Построение теста — работа агента, а не бухгалтерии: реестр
знает, что возможность появилась, но не знает, годится ли она. Поэтому `offer` принимает
функцию построения и уважает её отказ.

## Доля ложных и покрытие (инвариант 31)

У этой проверки два числа, и без них она не считается сделанной.

**Доля ложных** — сколько раз реестр решил, что недостающее появилось, а тест всё равно
не построился. Такое предложение не бесплатно: агент тратит на него внимание, и проверка,
ошибающаяся часто, начнёт игнорироваться.

**Покрытие** — доля отложенных, которые вообще объявили, чего им не хватает. Вопрос с
пустым `reopens_on` этим механизмом не переоткроется **никогда**, и это законное
состояние («не знаю даже того, чего мне не хватает»), но оно обязано быть видно числом.
Иначе реестр, переоткрывающий три вопроса из трёх объявленных при сорока слепых, выглядит
работающим на сто процентов.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Iterable, Mapping

from .beliefs import BeliefError, Hypothesis, Stage


class Missing(StrEnum):
    """Чего может не хватать для построения теста. Набор закрыт.

    Закрыт по той же причине, по которой закрыт набор единиц независимости: «чего мне не
    хватает» — это утверждение о структуре мира и о собственных возможностях, и новый вид
    нехватки должен быть обоснован здесь, а не заведён строкой в вызывающем коде. Иначе
    множество отложенного наберёт сорок разных написаний одной и той же нехватки, и
    сопоставить появившуюся возможность с ней станет нечем.
    """

    PLACE = "место"              # нужно попасть туда, где это наблюдаемо
    SKILL = "навык"              # нужно уметь то, чего ещё не умею
    READING = "чтение"           # нужно разбирать символы, а не только видеть их
    ENTITY_KIND = "вид"          # нужно встретить сущность такого рода
    TESTIMONY = "свидетельство"  # нужно, чтобы кто-то рассказал
    TIME = "время"               # нужно просто дождаться


def token(kind: Missing, name: str = "") -> str:
    """Метка нехватки или возможности: `вид` либо `вид:имя`.

    Без имени — «любое такое»: «нужно новое место, всё равно какое». С именем — ровно это
    место. Оба вида нужны: агент часто знает род нехватки и не знает её адреса.
    """
    return f"{kind}:{name}" if name else str(kind)


def matches(need: str, have: Iterable[str]) -> str:
    """Что из появившегося закрывает эту нехватку. Пустая строка — ничего.

    Правило одно и объявлено здесь: нехватка без имени закрывается любой возможностью
    того же рода, нехватка с именем — только точным совпадением. Обратной поддавки нет:
    возможность без имени **не** закрывает именованную нехватку, потому что «появилось
    какое-то место» не означает «появилось то самое место».
    """
    need_kind = need.split(":", 1)[0]
    named = ":" in need
    for got in have:
        if got == need:
            return got
        if not named and got.split(":", 1)[0] == need_kind:
            return got
    return ""


@dataclass(slots=True)
class Question:
    """Один открытый вопрос и его история: когда задан, когда переоткрыт, чем кончился.

    Время держится в номерах записей журнала, а не в секундах. Секунды — свойство машины,
    номера записей — свойство опыта: вопрос, заданный на пятой записи и разрешённый на
    трёхсотой, ждал 295 записей независимо от того, шёл прогон на быстрой машине или на
    медленной.
    """

    hypothesis: Hypothesis
    asked_seq: int
    reopened_seq: int | None = None
    reopened_by: str = ""
    resolved_seq: int | None = None
    outcome: bool | None = None
    offers_refused: int = 0          # реестр предложил, тест не построился

    @property
    def claim(self) -> str:
        return self.hypothesis.claim

    @property
    def open(self) -> bool:
        return self.resolved_seq is None

    @property
    def blind(self) -> bool:
        """Не объявил, чего ему не хватает. Переоткрыт не будет никогда."""
        return not self.hypothesis.reopens_on

    @property
    def waited(self) -> int | None:
        """Сколько записей журнала прошло от постановки до разрешения."""
        if self.resolved_seq is None:
            return None
        return max(0, self.resolved_seq - self.asked_seq)

    def as_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "stage": str(self.hypothesis.stage),
                "asked_seq": self.asked_seq, "reopened_seq": self.reopened_seq,
                "reopened_by": self.reopened_by, "resolved_seq": self.resolved_seq,
                "outcome": self.outcome, "waited": self.waited,
                "blind": self.blind, "offers_refused": self.offers_refused,
                "because": self.hypothesis.deferred_reason,
                "needs": list(self.hypothesis.reopens_on)}


@dataclass(slots=True)
class Reopened:
    """Одно переоткрытие: какой вопрос, чем закрылась нехватка, какой тест построен."""

    claim: str
    by: str
    test: str
    seq: int

    def as_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "by": self.by, "test": self.test, "seq": self.seq}


class OpenQuestions:
    """Реестр отложенного. Ничего не хранит сверх выводимого из журнала.

    Реестр — производный слой, как и всё, кроме журнала: вопрос заводится записью
    `HYPOTHESIS`, и пересборка восстанавливает реестр целиком. Здесь он живёт для того,
    чтобы прогон не перечитывал журнал на каждое предложение возможности.
    """

    def __init__(self) -> None:
        self.questions: dict[str, Question] = {}
        self.offered = 0                 # сколько раз нехватка сочлась закрытой
        self.reopened: list[Reopened] = []

    # --- наполнение ---------------------------------------------------------

    def ask(self, h: Hypothesis, seq: int) -> Question:
        """Завести вопрос. Принимается только то, у чего теста нет.

        Гипотеза с тестом сюда не попадает — и это не придирка к типам. Стоит один раз
        пустить в перечень непонятого гипотезу, до которой просто не дошли руки, и
        показатель «открытых вопросов» начнёт мерить длину очереди дел, а не объём
        непонятого. Различение размывается именно так: по одной уступке.
        """
        if h.stage is not Stage.DEFERRED:
            raise BeliefError(
                f"в реестр отложенного попала гипотеза {h.claim!r} в положении "
                f"{h.stage}: у неё есть тест, значит она в работе, а не непонята. "
                "Перечень непонятого — не очередь дел")
        q = self.questions.get(h.claim)
        if q is not None:
            return q
        q = Question(h, asked_seq=int(seq))
        self.questions[h.claim] = q
        return q

    def resolve(self, claim: str, *, outcome: bool, seq: int,
                arb: Any | None = None) -> Question:
        """Вопрос разрешился. Считается разрешённым задним числом, если был переоткрыт.

        `arb` — те же счётчики переходов, что у `offer`. Первая редакция их сюда не
        передавала, и замер показал два мёртвых перехода из трёх: `hypothesis→verified` и
        `hypothesis→refuted` не отмечались ни разу, хотя исполнялись. Инвариант 26 ловит
        именно это — не отсутствие кода, а отсутствие учёта, из-за которого мёртвый
        переход неотличим от неучтённого.
        """
        q = self.questions.get(claim)
        if q is None:
            raise BeliefError(f"нет открытого вопроса {claim!r}")
        if q.reopened_seq is None:
            raise BeliefError(
                f"вопрос {claim!r} разрешён, минуя переоткрытие. Разрешить его можно "
                "только после того, как нашёлся тест: иначе исход взялся неизвестно "
                "откуда, и «разрешено задним числом» перестанет что-либо значить")
        q.resolved_seq = int(seq)
        q.outcome = bool(outcome)
        q.hypothesis = q.hypothesis.resolved(outcome, arb=arb)
        return q

    # --- переоткрытие -------------------------------------------------------

    def offer(self, opportunities: Iterable[str], *, seq: int,
              build_test: Callable[[Question, str], str | None],
              arb: Any | None = None) -> list[Reopened]:
        """Появилось новое доступное. Что из отложенного возвращается в работу.

        `build_test(question, by)` строит проверку или отказывается (`None`). Отказ —
        нормальный исход, и он считается: доля отказов и есть доля ложных срабатываний
        этой проверки (инвариант 31).
        """
        have = list(opportunities)
        out: list[Reopened] = []
        for q in sorted(self.questions.values(), key=lambda x: x.asked_seq):
            if not q.open or q.reopened_seq is not None:
                continue
            for need in q.hypothesis.reopens_on:
                by = matches(need, have)
                if not by:
                    continue
                self.offered += 1
                test = build_test(q, by)
                if not test:
                    q.offers_refused += 1
                    break
                q.hypothesis = q.hypothesis.with_test(test, arb=arb)
                q.reopened_seq = int(seq)
                q.reopened_by = by
                rec = Reopened(q.claim, by, test, int(seq))
                self.reopened.append(rec)
                out.append(rec)
                break
        return out

    # --- показатели ---------------------------------------------------------

    def open_questions(self) -> list[Question]:
        return [q for q in self.questions.values() if q.open]

    def resolved_questions(self) -> list[Question]:
        return [q for q in self.questions.values() if not q.open]

    @property
    def refused(self) -> int:
        return sum(q.offers_refused for q in self.questions.values())

    def false_offer_share(self) -> float | None:
        """Доля предложений, после которых тест так и не построился."""
        return None if not self.offered else self.refused / self.offered

    def coverage(self) -> float | None:
        """Доля вопросов, объявивших, чего им не хватает. Остальные слепы навсегда."""
        if not self.questions:
            return None
        declared = sum(1 for q in self.questions.values() if not q.blind)
        return declared / len(self.questions)

    def resolved_share(self) -> float | None:
        """Доля разрешённых задним числом от всех когда-либо заданных."""
        if not self.questions:
            return None
        return len(self.resolved_questions()) / len(self.questions)

    def mean_wait(self) -> float | None:
        """Среднее время от постановки до разрешения, в записях журнала.

        Это и есть скорость, с которой мир отвечает на вопросы, которых агент сам закрыть
        не мог. Считается только по разрешённым: включать в среднее ещё не разрешённые
        значило бы занижать его тем, что вопрос всё ещё открыт.
        """
        waits = [q.waited for q in self.resolved_questions() if q.waited is not None]
        return None if not waits else sum(waits) / len(waits)

    def stats(self) -> dict[str, Any]:
        return {
            "asked": len(self.questions),
            "open": len(self.open_questions()),
            "resolved": len(self.resolved_questions()),
            "resolved_share": self.resolved_share(),
            "mean_wait_entries": self.mean_wait(),
            "blind": sum(1 for q in self.questions.values() if q.blind),
            "coverage": self.coverage(),
            "offers": self.offered,
            "offers_refused": self.refused,
            "false_offer_share": self.false_offer_share(),
        }

    def report(self) -> str:
        """Строки для отчёта. Отсутствие числа называется словами, а не нулём."""
        s = self.stats()
        if not s["asked"]:
            return ("открытых вопросов нет: ни одна гипотеза не оказалась без теста. "
                    "Это не «всё понятно» — это отсутствие замера")
        rows = [f"вопросов задано {s['asked']}, открыто {s['open']}, "
                f"разрешено задним числом {s['resolved']}"]
        if s["mean_wait_entries"] is not None:
            rows.append(f"  среднее ожидание {s['mean_wait_entries']:.0f} записей "
                        "журнала от постановки до разрешения")
        else:
            rows.append("  среднее ожидание не считается: ни один вопрос не разрешён")
        cov = s["coverage"]
        rows.append(f"  покрытие {cov:.0%}: {s['blind']} вопросов не объявили, чего им "
                    "не хватает, и переоткрыты не будут никогда"
                    if cov is not None else "  покрытие неизвестно")
        fo = s["false_offer_share"]
        rows.append(f"  ложных предложений {fo:.0%} ({s['offers_refused']} из "
                    f"{s['offers']}): нехватка сочлась закрытой, тест не построился"
                    if fo is not None else
                    "  предложений не было: множество доступного не менялось")
        return "\n".join(rows)


def from_journal(journal: Any) -> OpenQuestions:
    """Собрать реестр по журналу. Единственный законный способ его получить.

    Живой реестр — кэш; истина в журнале (инвариант 1). Расхождение между собранным здесь
    и живым — это дрейф памяти, и он измеряется, а не предполагается отсутствующим.
    """
    from ..core.journal import Kind as EntryKind

    out = OpenQuestions()
    for e in journal.entries([EntryKind.HYPOTHESIS]):
        code = str(e.event.get("code", ""))
        claim = str(e.event.get("claim", ""))
        if not claim:
            raise BeliefError(f"запись {e.seq}: гипотеза без утверждения")
        if code == "asked":
            out.ask(Hypothesis.question(
                claim, float(e.event.get("prior", 0.5)),
                _prov(e), because=str(e.event.get("because", "не сказано")),
                reopens_on=tuple(str(x) for x in e.event.get("needs", ()))), e.seq)
        elif code == "reopened":
            q = out.questions.get(claim)
            if q is None:
                continue
            q.hypothesis = q.hypothesis.with_test(str(e.event.get("test", "?")))
            q.reopened_seq = e.seq
            q.reopened_by = str(e.event.get("by", ""))
            out.reopened.append(Reopened(claim, q.reopened_by,
                                         str(e.event.get("test", "?")), e.seq))
        elif code == "resolved":
            out.resolve(claim, outcome=bool(e.event.get("outcome", False)), seq=e.seq)
        elif code == "offer_refused":
            q = out.questions.get(claim)
            if q is not None:
                q.offers_refused += 1
            out.offered += 1
    # Предложения, кончившиеся переоткрытием, считаются по самим переоткрытиям: писать
    # их отдельной записью значило бы держать в журнале два счётчика одного события.
    out.offered += len(out.reopened)
    return out


def _prov(entry: Any) -> Any:
    """Происхождение вопроса из записи. Без него вопрос в реестр не попадает."""
    from .beliefs import Origin, Provenance

    src = str(entry.event.get("source", "self"))
    origin = Origin(str(entry.event.get("origin", "hunch")))
    return Provenance(origin, entry.branch_id if hasattr(entry, "branch_id")
                      else str(entry.event.get("branch", "?")), entry.seq, src,
                      float(entry.event.get("trust", 1.0)))


def journal_question(journal: Any, stamp: Any, h: Hypothesis) -> Any:
    """Записать вопрос в журнал. Отсюда его потом восстанавливает пересборка."""
    from ..core.journal import Actor, ActorLayer, Kind as EntryKind

    if h.stage is not Stage.DEFERRED:
        raise BeliefError("в журнал вопросов пишется только отложенное")
    return journal.append(
        EntryKind.HYPOTHESIS, stamp, Actor.AGENT, ActorLayer.PLANNER,
        event={"code": "asked", "claim": h.claim, "prior": h.prior,
               "because": h.deferred_reason, "needs": list(h.reopens_on),
               "origin": str(h.provenance.origin), "source": h.provenance.source,
               "trust": h.provenance.trust})


def journal_reopen(journal: Any, stamp: Any, rec: Reopened) -> Any:
    from ..core.journal import Actor, ActorLayer, Kind as EntryKind

    return journal.append(
        EntryKind.HYPOTHESIS, stamp, Actor.AGENT, ActorLayer.PLANNER,
        event={"code": "reopened", "claim": rec.claim, "by": rec.by, "test": rec.test})


def journal_resolve(journal: Any, stamp: Any, claim: str, outcome: bool) -> Any:
    from ..core.journal import Actor, ActorLayer, Kind as EntryKind

    return journal.append(
        EntryKind.HYPOTHESIS, stamp, Actor.AGENT, ActorLayer.PLANNER,
        event={"code": "resolved", "claim": claim, "outcome": bool(outcome)})


def journal_refused_offer(journal: Any, stamp: Any, claim: str, by: str) -> Any:
    """Ложное предложение тоже пишется: без него доля ложных считается по памяти."""
    from ..core.journal import Actor, ActorLayer, Kind as EntryKind

    return journal.append(
        EntryKind.HYPOTHESIS, stamp, Actor.AGENT, ActorLayer.PLANNER,
        event={"code": "offer_refused", "claim": claim, "by": by})


#: Коды событий записи `HYPOTHESIS`. Закрытый набор, как всё, что идёт в журнал.
CODES: Mapping[str, str] = {
    "asked": "вопрос заведён: теста нет, причина указана",
    "reopened": "недостающее появилось, тест построен",
    "resolved": "проверка выполнена, исход зафиксирован",
    "offer_refused": "нехватка сочлась закрытой, тест не построился",
}
