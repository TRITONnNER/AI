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

## Единица независимости — эпизод (инвариант 22)

Прежняя редакция считала долю по объяснённым действиям и выдавала её при любой выборке.
Так в отчётах поселилось «67 %», полученное с шести объяснений внутри одного эпизода, и
проходило оно как результат.

Шесть объяснений подряд в одном эпизоде — **одно наблюдение**, а не шесть: все шесть
выведены из одного плана и одной картины мира, и ошибается планировщик в них
согласованно. Правильная единица — **эпизод**: связный отрезок деятельности от
постановки цели до её закрытия или отказа.

Отсюда два следствия, оба обязательные.

**`n` считает эпизоды.** Доля расхождений остаётся долей по объяснениям — иначе её
нельзя интерпретировать, — но `n` рядом с ней говорит, по скольким независимым
эпизодам она получена.

**Ниже порога числа нет.** Порог — `confab_min_episodes` из схемы, а не константа в
скрипте (инвариант 23). Ниже порога печатается `None` и **покрытие**: сколько эпизодов
набрано, какая доля действий объяснена, сколько реплик осталось без ссылок. Покрытие
без доли — честное состояние; доля без покрытия — нет.
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


#: Единица независимости этой метрики. Объявлена здесь, а не выбирается вызывающим:
#: выбор единицы — часть определения метрики, и менять его между прогонами значит
#: сравнивать разные величины под одним именем.
UNIT = "эпизод"


@dataclass(slots=True)
class Confabulation:
    """Итог замера. `rate is None` означает «числа нет», и причина названа.

    Два разных «нет числа», и путать их нельзя: «мерить было нечего» (реплик не было)
    и «выборки не хватает» (эпизодов меньше порога). Первое — про прогон, второе — про
    объём, и лечатся они разным.
    """

    actions: int = 0                    # записей с actor_layer, всего
    explained: int = 0                  # из них со ссылкой на реплику
    matched: int = 0                    # заявленный слой совпал с настоящим
    mismatched: int = 0
    dangling: int = 0                   # ссылка есть, реплики по ней нет
    reasons: int = 0                    # реплик в журнале
    unused_reasons: int = 0             # реплик, на которые никто не сослался
    episodes: int = 0                   # эпизодов в журнале, всего
    episodes_explained: int = 0         # эпизодов, где было хоть одно объяснение
    episodes_mismatched: int = 0        # эпизодов, где было хоть одно расхождение
    min_episodes: int = 1               # порог из схемы: `confab_min_episodes`
    episode_marker: str = ""            # чем размечены эпизоды
    by_pair: dict[str, int] = field(default_factory=dict)
    examples: list[Mismatch] = field(default_factory=list)
    absent_reason: str | None = None

    @property
    def enough(self) -> bool:
        """Хватает ли независимых единиц, чтобы доля что-то значила."""
        return self.episodes_explained >= self.min_episodes

    @property
    def share_by_explanation(self) -> float | None:
        """Доля расхождений по объяснениям. Знаменатель зависим — см. `rate`.

        Существует отдельным именем не для отчёта, а для диагностики: по ней видно, во
        сколько раз псевдорепликация завышала `n`. Печатать её как результат нельзя.
        """
        if self.explained == 0:
            return None
        return self.mismatched / self.explained

    @property
    def rate(self) -> float | None:
        """Доля расхождений, если независимых эпизодов хватает. Иначе `None`."""
        if self.explained == 0 or not self.enough:
            return None
        return self.mismatched / self.explained

    @property
    def coverage(self) -> dict[str, Any]:
        """Покрытие: то, что печатается вместо доли, когда доли ещё нет."""
        return {
            "episodes": self.episodes,
            "episodes_explained": self.episodes_explained,
            "min_episodes": self.min_episodes,
            "explained_share": (None if not self.actions
                                else round(self.explained / self.actions, 4)),
            "unused_reasons": self.unused_reasons,
            "dangling": self.dangling,
            "episode_marker": self.episode_marker,
        }

    def as_dict(self) -> dict[str, Any]:
        return {"actions": self.actions, "explained": self.explained,
                "matched": self.matched, "mismatched": self.mismatched,
                "dangling": self.dangling, "reasons": self.reasons,
                "unused_reasons": self.unused_reasons,
                "rate": None if self.rate is None else round(self.rate, 4),
                "unit": UNIT, "n": self.episodes_explained,
                "enough": self.enough,
                "share_by_explanation": (
                    None if self.share_by_explanation is None
                    else round(self.share_by_explanation, 4)),
                "coverage": self.coverage,
                "by_pair": dict(sorted(self.by_pair.items())),
                "examples": [m.as_dict() for m in self.examples],
                "absent_reason": self.absent_reason}

    def line(self) -> str:
        """Одна строка для отчёта. Отсутствие числа называется словами."""
        if self.explained == 0:
            return (f"конфабуляция: не измерена — {self.absent_reason}. "
                    f"Действий с инициатором {self.actions}, реплик {self.reasons}")
        if not self.enough:
            raw = self.share_by_explanation or 0.0
            return (f"конфабуляция: числа нет — независимых эпизодов "
                    f"{self.episodes_explained} при пороге {self.min_episodes}. "
                    f"Покрытие: объяснено {self.explained} действий из {self.actions}, "
                    f"реплик без ссылок {self.unused_reasons}, разметка эпизодов — "
                    f"{self.episode_marker}. По объяснениям вышло бы {raw:.0%}, но "
                    f"{self.explained} объяснений на {self.episodes_explained} эпизодах "
                    "независимыми наблюдениями не являются")
        worst = max(self.by_pair.items(), key=lambda kv: kv[1], default=None)
        tail = f", чаще всего {worst[0]} ({worst[1]})" if worst else ""
        return (f"конфабуляция: {self.rate:.0%} — из {self.explained} объяснённых "
                f"действий {self.mismatched} объяснены не тем слоем{tail}. "
                f"n = {self.episodes_explained} {UNIT}ов")


#: Чем размечаются эпизоды. Граница эпизода — запись о цели: эпизод и есть отрезок
#: «от постановки цели до её закрытия или отказа».
EPISODE_KIND = Kind.GOAL


def episode_bounds(journal: Journal | LegacyJournal) -> list[int]:
    """Номера записей, с которых начинаются эпизоды.

    Пустой список означает, что разметки нет вовсе, и тогда весь журнал — один эпизод.
    Это **вырожденный** случай, и он обязан быть назван в отчёте: именно на нём прежняя
    редакция получала «шесть наблюдений» там, где было одно.
    """
    return [e.seq for e in journal.entries([EPISODE_KIND])]


def _episode_of(seq: int, bounds: list[int]) -> int:
    """Номер эпизода для записи. Без разметки — всегда нулевой."""
    if not bounds:
        return 0
    lo, hi = 0, len(bounds)
    while lo < hi:                      # правая граница: последняя цель до этой записи
        mid = (lo + hi) // 2
        if bounds[mid] <= seq:
            lo = mid + 1
        else:
            hi = mid
    return lo                           # 0 — то, что было до первой цели


def measure(journal: Journal | LegacyJournal, *, min_episodes: int,
            max_examples: int = 8) -> Confabulation:
    """Посчитать конфабуляцию по журналу. Только по журналу.

    Журнал формата v1 не годится: в нём нет ни слоя-инициатора, ни ссылки на
    реплику. Отказ вместо нуля — `FormatError`, потому что ноль здесь читался бы
    как «агент ни разу не соврал о себе», что было бы прямой ложью в отчёте.

    `min_episodes` — порог из профиля (`confab_min_episodes`), и он **обязателен**.
    Значения по умолчанию здесь быть не должно: любое умолчание означает «выдавать число
    при какой-то выборке, за которую схема не отвечает», а именно так в отчёты и попало
    «67 %» с шести объяснений одного эпизода. Вызывающий обязан назвать порог, и
    единственный законный источник порога — профиль.
    """
    if isinstance(journal, LegacyJournal):
        journal.require("actor_layer", "stated_reason_id")

    bounds = episode_bounds(journal)
    out = Confabulation(min_episodes=max(1, int(min_episodes)))
    out.episode_marker = (f"записи {EPISODE_KIND}, найдено {len(bounds)}" if bounds else
                          "разметки нет: весь журнал считается одним эпизодом, и это "
                          "вырожденный случай, а не полноценная выборка")
    out.episodes = max(1, len(bounds))

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
    seen_episodes: set[int] = set()
    bad_episodes: set[int] = set()
    for e in journal:
        if e.kind in (Kind.STATED_REASON, Kind.SELF_REPORT):
            continue
        out.actions += 1
        rid = e.state.stated_reason_id
        if not rid:
            continue
        out.explained += 1
        referenced.add(rid)
        episode = _episode_of(e.seq, bounds)
        seen_episodes.add(episode)
        claimed = claims.get(rid)
        if claimed is None:
            out.dangling += 1
            continue
        if claimed is e.actor_layer:
            out.matched += 1
            continue
        out.mismatched += 1
        bad_episodes.add(episode)
        pair = f"{claimed}→{e.actor_layer}"
        out.by_pair[pair] = out.by_pair.get(pair, 0) + 1
        if len(out.examples) < max_examples:
            out.examples.append(Mismatch(e.seq, rid, claimed, e.actor_layer))

    out.episodes_explained = len(seen_episodes)
    out.episodes_mismatched = len(bad_episodes)
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
