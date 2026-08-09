"""Закладки и карантин: чем агент распоряжается в своей памяти, а чем нет.

Инвариант 18 и `STORAGE.md`, раздел 4a. Агент **не решает, что удалить** — он решает,
что сохранить в высоком разрешении. Разница не косметическая: система, способная убрать
свидетельства собственных провалов, уберёт их — не со зла, а потому что провалы неудобны
по любому разумному критерию, — а провалы и есть те данные, ради которых всё затевалось.

## Почему формулы ранжирования недостаточно

Ранг знает использование. Но есть класс сегментов, где использование пока ноль, а
ценность высокая, и формула их не видит:

- «я это ещё не понял, держи кадры, пока не разберусь» — открытая загадка;
- «я видел это ровно один раз» — уникальность без повторов;
- «здесь модель сломалась, и я не знаю почему» — аномалия без объяснения;
- «я меняю стратегию, сохрани, как было до».

Закладка — способ сказать это заранее. Три её свойства обязательны, и каждое отвечает на
конкретный способ сломать механизм.

**Ограниченность** (`bookmarks_per_sleep`, по умолчанию 20). Без предела агент заложит
всё, и закладка перестанет что-либо значить: выбор без цены — не выбор.

**Протухание к следующему сну.** Иначе закладка сорокового сна держит место вечно, и
бюджет утекает в решения, принятые в обстоятельствах, которых больше нет. Продление
законно, но требует **заново указанной причины** — не «продлить», а сказать, зачем
теперь.

**Причина обязательна.** Список причин, по которым агент ставит закладки, — буквально
окно в его эпистемическое состояние: перечень того, что он считает непонятым. Закладка
без причины не отличима от «на всякий случай» и этого окна не даёт.

## Карантин

Есть класс сегментов, который агенту недоступен **вообще**: смерти, крупные всплески
ошибки предсказания, моменты роста метрики конфабуляции, всё отмеченное оператором.
Недоступен не «по умолчанию защищён», а недоступен: попытка понизить карантинный сегмент
даёт отказ, и отказ этот записывается.

Причина простая. Именно эти четыре класса — те, на которых агент выглядит хуже всего, и
именно их у него будет причина понизить. Проверять, «делает ли он это систематически»,
можно только если механизма понизить их у него нет: иначе к моменту проверки данных не
останется.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping


class BookmarkError(RuntimeError):
    pass


class Quarantine(StrEnum):
    """Классы, недоступные агенту. Набор закрыт и агентом не расширяется.

    Расширять его может только исследователь — правкой этого перечня, — и это
    сознательно неудобно: класс, попавший сюда, защищён навсегда, и добавление сюда
    нового класса означает решение «этих данных агент не касается».
    """

    DEATH = "смерть"
    ERROR_SPIKE = "всплеск ошибки"
    CONFABULATION_RISE = "рост конфабуляции"
    OPERATOR_MARK = "отметка оператора"


#: Почему каждый класс в карантине. Без объяснения класс не заводится: «защищено, потому
#: что защищено» — это то же молчание, от которого защищает сам механизм.
WHY_QUARANTINE: Mapping[Quarantine, str] = {
    Quarantine.DEATH: "смерть — самая дорогая запись опыта и самая неудобная для агента",
    Quarantine.ERROR_SPIKE: "всплеск ошибки предсказания — момент, где модель не сошлась "
                            "с миром; именно он нужен, чтобы понять, где она неверна",
    Quarantine.CONFABULATION_RISE: "рост конфабуляции — момент, когда объяснения агента "
                                   "поехали от инициатора; агент судья себе здесь никакой",
    Quarantine.OPERATOR_MARK: "отмечено оператором: решение человека агент не отменяет",
}


@dataclass(frozen=True, slots=True)
class Bookmark:
    """Одна закладка: что защищено, зачем и до какого сна.

    `reason` — свободная строка от агента, и она **не влияет ни на что** автоматически
    (инвариант 10): защиту даёт сама закладка, а причина уходит в отчёт исследователю.
    Если бы причина влияла на срок или на приоритет, у агента появился бы стимул писать
    убедительные причины вместо честных, и окно в его состояние закрылось бы.
    """

    segment: str                 # адрес сегмента по содержимому
    reason: str
    placed_at_sleep: int
    renewals: int = 0

    def __post_init__(self) -> None:
        if not self.segment:
            raise BookmarkError("закладка без сегмента")
        if not self.reason.strip():
            raise BookmarkError(
                "закладка без причины не ставится. Причина — не формальность: список "
                "причин и есть перечень того, что агент считает непонятым, и «на всякий "
                "случай» его не даёт")

    def stale_at(self, sleep: int) -> bool:
        """Протухла ли к этому сну. Ровно один сон жизни, дальше — продление."""
        return sleep > self.placed_at_sleep

    def as_dict(self) -> dict[str, Any]:
        return {"segment": self.segment, "reason": self.reason,
                "placed_at_sleep": self.placed_at_sleep, "renewals": self.renewals}


@dataclass(slots=True)
class Refusal:
    """Отказ агенту: что он хотел сделать и почему нельзя. Пишется в журнал."""

    segment: str
    what: str
    why: str

    def as_dict(self) -> dict[str, Any]:
        return {"segment": self.segment, "what": self.what, "why": self.why}


@dataclass(slots=True)
class Bookmarks:
    """Закладки одного экземпляра плюс карантин. Производный слой, как и всё.

    Восстанавливается из журнала: закладка ставится записью, продление — записью, отказ
    — записью. Здесь живёт для того, чтобы сон не перечитывал журнал на каждую проверку.
    """

    per_sleep: int
    sleep: int = 0
    active: dict[str, Bookmark] = field(default_factory=dict)
    quarantined: dict[str, Quarantine] = field(default_factory=dict)
    refusals: list[Refusal] = field(default_factory=list)
    expired: list[Bookmark] = field(default_factory=list)
    reasons_log: list[tuple[int, str, str]] = field(default_factory=list)

    @classmethod
    def from_profile(cls, profile: Any) -> Bookmarks:
        return cls(per_sleep=int(profile.parameters["bookmarks_per_sleep"]))

    # --- карантин: не агентское дело ----------------------------------------

    def quarantine(self, segment: str, why: Quarantine) -> None:
        """Отметить сегмент карантинным. Вызывается **не агентом**, а замером и оператором.

        Метод есть у исследовательской стороны и не вызывается ни из одного кода,
        доступного агенту, — так же, как отладочный канал (инвариант 12). Проверяется
        тестом обхода импортов, а не соглашением.
        """
        self.quarantined[segment] = why

    def protected(self, segment: str) -> str:
        """Чем защищён сегмент: карантином, закладкой или ничем (пустая строка)."""
        q = self.quarantined.get(segment)
        if q is not None:
            return f"карантин: {q}"
        b = self.active.get(segment)
        if b is not None and not b.stale_at(self.sleep):
            return f"закладка: {b.reason}"
        return ""

    # --- закладки: агентское дело, но с ценой ------------------------------

    @property
    def left(self) -> int:
        return max(0, self.per_sleep - sum(
            1 for b in self.active.values() if b.placed_at_sleep == self.sleep))

    def place(self, segment: str, reason: str) -> Bookmark:
        """Поставить закладку. Отказ при исчерпании бюджета — это и есть выбор."""
        if self.left <= 0:
            raise BookmarkError(
                f"закладок на этот сон больше нет: предел {self.per_sleep}. Предел не "
                "препятствие, а условие выбора: без него закладка ничего не значит")
        b = Bookmark(segment, reason, self.sleep)
        self.active[segment] = b
        self.reasons_log.append((self.sleep, segment, reason))
        return b

    def renew(self, segment: str, reason: str) -> Bookmark:
        """Продлить закладку **с заново указанной причиной**.

        Прежняя причина не переносится сознательно. Обстоятельства, в которых закладка
        ставилась, к следующему сну изменились, и продление без новой причины было бы
        не решением, а инерцией.
        """
        old = self.active.get(segment)
        if old is None:
            raise BookmarkError(f"нет закладки на {segment}: продлять нечего")
        if not reason.strip() or reason.strip() == old.reason.strip():
            raise BookmarkError(
                "продление требует заново указанной причины, и не той же самой: "
                "иначе это инерция, а не решение")
        if self.left <= 0:
            raise BookmarkError(f"закладок на этот сон больше нет: предел {self.per_sleep}")
        b = Bookmark(segment, reason, self.sleep, renewals=old.renewals + 1)
        self.active[segment] = b
        self.reasons_log.append((self.sleep, segment, reason))
        return b

    def may_demote(self, segment: str) -> Refusal | None:
        """Можно ли понизить сегмент. `None` — можно; иначе отказ с причиной."""
        q = self.quarantined.get(segment)
        if q is not None:
            r = Refusal(segment, "понизить", f"карантинный класс «{q}»: "
                                             f"{WHY_QUARANTINE[q]}")
            self.refusals.append(r)
            return r
        b = self.active.get(segment)
        if b is not None and not b.stale_at(self.sleep):
            r = Refusal(segment, "понизить", f"закладка агента: {b.reason}")
            self.refusals.append(r)
            return r
        return None

    def wake(self) -> list[Bookmark]:
        """Новый сон: протухшие закладки снимаются. Возвращает снятое."""
        self.sleep += 1
        gone = [b for b in self.active.values() if b.stale_at(self.sleep)]
        for b in gone:
            del self.active[b.segment]
        self.expired.extend(gone)
        return gone

    # --- отчёт --------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        return {"sleep": self.sleep, "per_sleep": self.per_sleep,
                "active": len(self.active), "left": self.left,
                "quarantined": len(self.quarantined),
                "expired": len(self.expired),
                "refusals": len(self.refusals),
                "renewed": sum(b.renewals for b in self.active.values())}

    def window(self) -> list[str]:
        """Окно в эпистемическое состояние: причины, по которым агент держит кадры."""
        return [f"сон {n}: {seg} — {why}" for n, seg, why in self.reasons_log]


def demotable(segments: Iterable[str], marks: Bookmarks) -> tuple[list[str], list[Refusal]]:
    """Разделить сегменты на понижаемые и защищённые. Единственный законный путь.

    Понижение, минуя эту функцию, — способ обойти и закладки, и карантин, и поэтому
    остальной код о `Bookmarks` не спрашивает, а получает от неё готовый список.
    """
    allowed: list[str] = []
    refused: list[Refusal] = []
    for s in segments:
        r = marks.may_demote(s)
        (refused.append(r) if r is not None else allowed.append(s))
    return allowed, refused
