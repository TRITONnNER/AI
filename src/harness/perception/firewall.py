"""Файрвол восприятия.

Инвариант 6: «большая модель отвечает только на вопрос "что я вижу". Она никогда
не отвечает на "что делать". Предложения действий генерирует отдельный механизм
на основе накопленной модели и любопытства.»

Это самый хрупкий инвариант из двенадцати, потому что нарушается не кодом, а
одной фразой в ответе. «В левом углу враждебный моб» — это уже «что делать»: там
и метка враждебности, и подсказка приоритета. Поймать такое можно только читая
ответ, поэтому здесь:

1. **Вопрос фиксирован в коде.** Не шаблон, который можно подправить на ходу, а
   константа. Спросить что-то другое через этот модуль нельзя.
2. **Сырой запрос и сырой ответ пишутся в отладочный поток.** Без этого нарушение
   невидимо: в журнал уходит только очищенное восприятие, и по нему не понять,
   что модель на самом деле сказала.
3. **Ответ проверяется на императивы и оценки.** Нашлось — `FirewallViolation`,
   и восприятие не доходит до агента вообще. Не «предупреждение в логе»:
   пропустить один такой ответ значит испортить эксперимент незаметно.
4. **В журнал уходит только структура.** Символы, геометрия, счётчики. Никакого
   текста: он не проходит проверку `assert_no_plain_text`.

Что здесь нет и почему. Нет клиента большой модели. Описатель передаётся снаружи
как функция, и в вехе 0–1 это локальный описатель без всякой модели. Причина
прямая: подключить модель здесь нечем, а непроверенный клиент в этом месте
опаснее отсутствующего — он бы создал видимость работающего файрвола.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from ..core.clocks import Stamp
from ..core.profile import Profile
from ..core.symbols import assert_no_plain_text, is_symbol

# Единственный вопрос, который этот модуль умеет задавать. Константа, не шаблон.
QUESTION = "Что я вижу? Перечисли, что есть на изображении, и где это находится."

# Слова, наличие которых означает, что ответ съехал с «что я вижу» на «что делать»
# или на оценку. Список не про цензуру, а про жанр: описание не советует и не
# оценивает. Проверяется по корню слова, чтобы ловить любые формы.
IMPERATIVE_STEMS = (
    "нужно", "надо", "следует", "стоит ", "лучше ", "рекоменд", "советую", "совет ",
    "необходимо", "обязательно", "избега", "опасн", "угроз", "враждеб",
    "враг", "атаку", "беги", "бегите", "уходи", "нажми", "нажать", "используй",
    "выбери", "приоритет", "цель —", "цель:", "задача:", "план:", "шаг 1",
    "should", "must", "avoid", "danger", "hostile", "enemy", "attack", "press ",
    "recommend", "you can", "try to", "goal", "objective",
)

# Оценочные слова: они не приказывают, но расставляют важность, а важность агент
# обязан выводить сам из ошибки предсказания и драйвов.
VALUATION_STEMS = (
    "важн", "полезн", "бесполезн", "хорош", "плох", "лучш", "хуж", "главн",
    "ключев", "критичн", "ценн", "выгодн", "important", "useful", "valuable",
    "critical", "best", "worst", "key ",
)


class FirewallViolation(RuntimeError):
    """Ответ содержит «что делать» или оценку. Восприятие не пропущено."""

    def __init__(self, reason: str, found: list[str], raw: str) -> None:
        super().__init__(f"{reason}: {', '.join(found)}")
        self.reason = reason
        self.found = found
        self.raw = raw


@runtime_checkable
class Describer(Protocol):
    """Тот, кто отвечает на «что я вижу». Может быть моделью, может не быть."""

    name: str

    def describe(self, frame: Any, question: str) -> str: ...


@dataclass(frozen=True, slots=True)
class Sighting:
    """Одна увиденная вещь. Символ и грубое место — без экранных координат.

    Место — это девять зон (левый верх … правый низ), а не пиксели. Пиксельные
    координаты дали бы агенту систему отсчёта, которой у него быть не должно
    (инвариант 4). Зона выводима из самого кадра и потому не подсказка.
    """

    symbol: str
    zone: str
    size: str            # "мелкое" | "среднее" | "крупное" — тоже без чисел

    def as_dict(self) -> dict[str, str]:
        return {"symbol": self.symbol, "zone": self.zone, "size": self.size}


ZONES = ("верх-лево", "верх-центр", "верх-право",
         "центр-лево", "центр", "центр-право",
         "низ-лево", "низ-центр", "низ-право")
SIZES = ("мелкое", "среднее", "крупное")


@dataclass(frozen=True, slots=True)
class Percept:
    """То, что доходит до агента. Только символы, зоны и размеры."""

    stamp: Stamp
    sightings: tuple[Sighting, ...]
    describer: str

    def as_journal_payload(self) -> dict[str, Any]:
        return {"symbols": [s.symbol for s in self.sightings],
                "zones": [s.zone for s in self.sightings],
                "sizes": [s.size for s in self.sightings],
                "count": len(self.sightings)}


@dataclass(slots=True)
class Audit:
    """Что файрвол увидел и что сделал. Для пульта и для отладочного потока."""

    calls: int = 0
    violations: int = 0
    imperatives: int = 0
    valuations: int = 0
    dropped_symbols: int = 0
    last_raw: str | None = None
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "violations": self.violations,
                "imperatives": self.imperatives, "valuations": self.valuations,
                "dropped_symbols": self.dropped_symbols,
                "violation_rate": round(self.violations / self.calls, 4) if self.calls else 0.0,
                "reasons": self.reasons[-5:]}


def check_answer(raw: str) -> tuple[list[str], list[str]]:
    """Найти в ответе императивы и оценки. Возвращает (императивы, оценки)."""
    low = raw.casefold()
    imperatives = sorted({s.strip() for s in IMPERATIVE_STEMS if s in low})
    valuations = sorted({s.strip() for s in VALUATION_STEMS if s in low})
    return imperatives, valuations


_LINE = re.compile(
    r"^\s*(?P<symbol>SYM_[0-9A-F]{4})\s*\|\s*(?P<zone>[^|]+?)\s*\|\s*(?P<size>[^|]+?)\s*$")


def parse_answer(raw: str, *, max_symbols: int) -> tuple[list[Sighting], int]:
    """Разобрать ответ. Возвращает (увиденное, сколько строк отброшено).

    Формат строгий: `SYM_1A2B | зона | размер`. Всё, что не разобралось,
    отбрасывается и считается — молча проглатывать нельзя, иначе описатель,
    отвечающий свободным текстом, будет выглядеть как исправно работающий.
    """
    out: list[Sighting] = []
    dropped = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        m = _LINE.match(line)
        if m is None:
            dropped += 1
            continue
        zone, size = m.group("zone").strip(), m.group("size").strip()
        if zone not in ZONES or size not in SIZES or not is_symbol(m.group("symbol")):
            dropped += 1
            continue
        out.append(Sighting(m.group("symbol"), zone, size))
        if len(out) >= max_symbols:
            break
    return out, dropped


class PerceptionFirewall:
    """Единственная дорога от описателя к агенту."""

    def __init__(self, profile: Profile, describer: Describer, *,
                 debug_write: Callable[[Stamp, str, dict[str, Any]], None] | None = None) -> None:
        self.profile = profile
        self.describer = describer
        self.enabled = bool(profile.structural["firewall_enabled"])
        self.max_symbols = int(profile.parameters["firewall_max_symbols"])
        self.log_raw = bool(profile.parameters["firewall_log_raw"])
        self._debug_write = debug_write
        self.audit = Audit()

    def look(self, frame: Any, stamp: Stamp) -> Percept:
        """Спросить «что я вижу» и пропустить ответ через проверку."""
        raw = self.describer.describe(frame, QUESTION)
        self.audit.calls += 1
        self.audit.last_raw = raw

        # Сырое — в отладочный поток, и до всякой проверки: если проверка упадёт,
        # исследователь должен видеть, на чём именно.
        if self._debug_write is not None and self.log_raw:
            self._debug_write(stamp, "firewall_call",
                              {"question": QUESTION, "raw_answer": raw,
                               "describer": self.describer.name})

        if self.enabled:
            imperatives, valuations = check_answer(raw)
            if imperatives or valuations:
                self.audit.violations += 1
                self.audit.imperatives += len(imperatives)
                self.audit.valuations += len(valuations)
                reason = ("ответ содержит указания к действию" if imperatives
                          else "ответ содержит оценку важности")
                self.audit.reasons.append(reason)
                if self._debug_write is not None:
                    self._debug_write(stamp, "firewall_violation",
                                      {"reason": reason, "imperatives": imperatives,
                                       "valuations": valuations, "raw_answer": raw})
                raise FirewallViolation(reason, imperatives + valuations, raw)

        sightings, dropped = parse_answer(raw, max_symbols=self.max_symbols)
        self.audit.dropped_symbols += dropped
        percept = Percept(stamp, tuple(sightings), self.describer.name)

        # Последняя проверка: то, что уходит в журнал и агенту, не содержит текста.
        assert_no_plain_text({"symbols": [s.symbol for s in sightings]},
                             path="firewall")
        return percept


class LocalDescriber:
    """Описатель без всякой модели: символы приходят снаружи, зоны считаются.

    Существует не как заглушка, а как опорный уровень и как средство проверки
    самого файрвола: он гарантированно отвечает в правильном жанре, поэтому по
    нему видно, что файрвол пропускает верное и не придирается к нему.
    """

    name = "local"

    def __init__(self, found: Callable[[Any], list[tuple[str, str, str]]]) -> None:
        self._found = found

    def describe(self, frame: Any, question: str) -> str:
        if question != QUESTION:
            raise ValueError(
                "описателю задан не тот вопрос. Через файрвол проходит один "
                "вопрос — «что я вижу», и он константа")
        return "\n".join(f"{sym} | {zone} | {size}"
                         for sym, zone, size in self._found(frame))


def zone_of(row: float, col: float) -> str:
    """Грубая зона по относительному положению в кадре, без пиксельных координат."""
    r = 0 if row < 1 / 3 else (1 if row < 2 / 3 else 2)
    c = 0 if col < 1 / 3 else (1 if col < 2 / 3 else 2)
    return ZONES[r * 3 + c]


def size_of(fraction: float) -> str:
    """Грубый размер по доле кадра."""
    if fraction < 0.01:
        return SIZES[0]
    return SIZES[1] if fraction < 0.08 else SIZES[2]
