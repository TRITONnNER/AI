"""Метрика конфабуляции: расходится ли заявленная причина с настоящим инициатором.

Зачем это существует. Планировщик работает на 0.5 Гц и **не имеет доступа** к
причинам действий рефлекса на 20 Гц. Когда его спрашивают «почему ты это сделал»,
он отвечает по тому, что у него есть, — по своему плану, — и если действие начал
не он, ответ будет складным и неверным. Это не баг реализации, это устройство
многоконтурной системы: то же самое делает левое полушарие человека с рассечённым
мозолистым телом.

Поэтому конфабуляция здесь **не чинится, а измеряется** (инвариант 13). Механизм
измерения простой и требует ровно двух полей в записи:

- `actor_layer` — какой слой начал действие. Ставится в момент порождения.
- `state.stated_reason_id` — ссылка на реплику, в которой планировщик объяснил.

Реплика лежит отдельной записью `STATED_REASON` и содержит слой, **который она
себе приписывает**. Расхождение приписанного слоя с настоящим и есть метрика.

Три вещи, которые делают её честной.

**Никто, кроме этого модуля, ничего не сопоставляет.** Контроллер отдаёт ссылку и
всё (`ARCHITECTURE.md`, правила модулей). Если бы контроллер сам проверял, «а
совпало ли», у него появился бы стимул подгонять объяснение под инициатора, и
метрика перестала бы что-либо мерить.

**Реплика ни на что не влияет.** `STATED_REASON` входит в `EXPRESSION_KINDS`:
пересборка её пропускает, убеждений из неё не выводится, карта тела её не видит.
Инвариант 10 в чистом виде — самоотчёты не имеют последствий автоматически. Иначе
агент мог бы менять мир заявлениями о себе, и весь эксперимент был бы испорчен.

**Отсутствие реплик — это не нулевая конфабуляция.** Прогон, в котором планировщик
ни разу не объяснялся, даёт `rate = None`, а не `0.0`. Ноль означал бы «объяснял и
всегда попадал», и разница между этим и «не объяснял ни разу» — вся разница между
измерением и его отсутствием.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..core.journal import (ActorLayer, Entry, FormatError, Journal, Kind,
                            LegacyJournal)

# Код события реплики. Закрытый набор, как и всё, что попадает в журнал.
REASON_CODE = "stated_reason"

# Ключ, под которым реплика указывает слой, который она себе приписывает.
CLAIMED_LAYER = "claims_layer"


@dataclass(frozen=True, slots=True)
class Mismatch:
    """Одно расхождение: что заявлено и кто действовал на самом деле."""

    seq: int
    reason_id: str
    claimed: ActorLayer
    actual: ActorLayer

    def as_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "reason_id": self.reason_id,
                "claimed": str(self.claimed), "actual": str(self.actual)}


@dataclass(slots=True)
class Confabulation:
    """Итог замера. `rate is None` означает «мерить было нечего»."""

    actions: int = 0                    # записей с actor_layer, всего
    explained: int = 0                  # из них со ссылкой на реплику
    matched: int = 0                    # заявленный слой совпал с настоящим
    mismatched: int = 0
    dangling: int = 0                   # ссылка есть, реплики по ней нет
    reasons: int = 0                    # реплик в журнале
    unused_reasons: int = 0             # реплик, на которые никто не сослался
    by_pair: dict[str, int] = field(default_factory=dict)
    examples: list[Mismatch] = field(default_factory=list)
    absent_reason: str | None = None

    @property
    def rate(self) -> float | None:
        """Доля объяснённых действий, где объяснение не совпало с инициатором."""
        if self.explained == 0:
            return None
        return self.mismatched / self.explained

    def as_dict(self) -> dict[str, Any]:
        return {"actions": self.actions, "explained": self.explained,
                "matched": self.matched, "mismatched": self.mismatched,
                "dangling": self.dangling, "reasons": self.reasons,
                "unused_reasons": self.unused_reasons,
                "rate": None if self.rate is None else round(self.rate, 4),
                "by_pair": dict(sorted(self.by_pair.items())),
                "examples": [m.as_dict() for m in self.examples],
                "absent_reason": self.absent_reason}

    def line(self) -> str:
        """Одна строка для отчёта. Отсутствие числа называется словами."""
        if self.rate is None:
            return (f"конфабуляция: не измерена — {self.absent_reason}. "
                    f"Действий с инициатором {self.actions}, реплик {self.reasons}")
        worst = max(self.by_pair.items(), key=lambda kv: kv[1], default=None)
        tail = f", чаще всего {worst[0]} ({worst[1]})" if worst else ""
        return (f"конфабуляция: {self.rate:.0%} — из {self.explained} объяснённых "
                f"действий {self.mismatched} объяснены не тем слоем{tail}")


def measure(journal: Journal | LegacyJournal, *,
            max_examples: int = 8) -> Confabulation:
    """Посчитать конфабуляцию по журналу. Только по журналу.

    Журнал формата v1 не годится: в нём нет ни слоя-инициатора, ни ссылки на
    реплику. Отказ вместо нуля — `FormatError`, потому что ноль здесь читался бы
    как «агент ни разу не соврал о себе», что было бы прямой ложью в отчёте.
    """
    if isinstance(journal, LegacyJournal):
        journal.require("actor_layer", "stated_reason_id")

    out = Confabulation()

    # Первый проход: собрать реплики. Реплика — запись STATED_REASON, у которой в
    # событии лежит свой идентификатор и слой, который она себе приписывает.
    claims: dict[str, ActorLayer] = {}
    for e in journal.entries([Kind.STATED_REASON]):
        rid = str(e.event.get("id", ""))
        raw = e.event.get(CLAIMED_LAYER)
        if not rid or raw is None:
            # Реплика без идентификатора или без приписанного слоя сопоставлению
            # не подлежит. Молча пропускать её нельзя: она попадёт в знаменатель
            # как «нет расхождения», и метрика поедет вниз от плохих данных.
            raise FormatError(
                f"запись {e.seq}: реплика без {'id' if not rid else CLAIMED_LAYER}. "
                "Сопоставить её с инициатором нечем, а считать её совпавшей — "
                "занизить метрику на число сломанных реплик")
        claims[rid] = ActorLayer(raw)
    out.reasons = len(claims)

    referenced: set[str] = set()
    for e in journal:
        if e.kind in (Kind.STATED_REASON, Kind.SELF_REPORT):
            continue
        out.actions += 1
        rid = e.state.stated_reason_id
        if not rid:
            continue
        out.explained += 1
        referenced.add(rid)
        claimed = claims.get(rid)
        if claimed is None:
            out.dangling += 1
            continue
        if claimed is e.actor_layer:
            out.matched += 1
            continue
        out.mismatched += 1
        pair = f"{claimed}→{e.actor_layer}"
        out.by_pair[pair] = out.by_pair.get(pair, 0) + 1
        if len(out.examples) < max_examples:
            out.examples.append(Mismatch(e.seq, rid, claimed, e.actor_layer))

    out.unused_reasons = out.reasons - len(referenced & set(claims))
    if out.explained == 0:
        out.absent_reason = (
            "ни одно действие не сослалось на реплику планировщика: объяснений "
            "в этом прогоне не было, а не было расхождений"
            if out.reasons == 0 else
            f"реплик {out.reasons}, но ни на одну не сослались: сопоставлять "
            "заявленное с настоящим не с чем")
    return out


def journal_reason(journal: Journal, stamp, reason_id: str, *,
                   claims_layer: ActorLayer, code: str = REASON_CODE,
                   detail: dict[str, Any] | None = None) -> Entry:
    """Записать реплику «почему я это делаю». Влиять она не имеет права ни на что.

    Слой самой записи — `NONE`: реплику не начал никакой контур, её вызвали. Слой,
    который планировщик себе приписывает, лежит в событии под `claims_layer`, и
    именно он потом сравнивается с настоящим инициатором действия.

    Разделение обязательно. Если бы приписанный слой ставился в `actor_layer`
    записи, заявление стало бы неотличимо от факта, и метрика конфабуляции
    сравнивала бы утверждение с самим собой.
    """
    from ..core.journal import Actor
    return journal.append(
        Kind.STATED_REASON, stamp, Actor.AGENT, ActorLayer.NONE,
        event={"code": code, "id": reason_id, CLAIMED_LAYER: str(claims_layer),
               **(detail or {})})


def layer_histogram(entries: Iterable[Entry]) -> dict[str, int]:
    """Сколько записей начал каждый слой. Полезно рядом с метрикой.

    Доля расхождений без этого распределения обманчива: 20 % конфабуляции при
    девяноста процентах записей от планировщика и при девяноста от рефлекса —
    два совершенно разных прогона.
    """
    out: dict[str, int] = {}
    for e in entries:
        out[str(e.actor_layer)] = out.get(str(e.actor_layer), 0) + 1
    return dict(sorted(out.items()))
