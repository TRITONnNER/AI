"""Тождество: одно ли это, или два. Слияние и расщепление задним числом.

Схема карточки — не главная трудность. Главная трудность в том, **та же это сущность или
другая**, и ошибиться здесь можно в обе стороны:

- **расщепление** — одно записано как двадцать. Агент видит верстак с двух сторон и
  заводит две карточки; заходит в ту же комнату при другом освещении и заводит третью.
  Признак: карточки с единственной встречей размножаются, а обобщения не возникает;
- **слипание** — двадцать записано как одно. Отпечаток слишком грубый, и все двери мира
  становятся одной дверью, у которой половина проходов ведёт наружу, половина внутрь.
  Признак: у карточки много разных отпечатков, а `mu` держится посередине при большом `n`.

Ошибки не симметричны по цене. Расщепление теряет обобщение — неприятно, но опыт цел'
целиком лежит по частям. Слипание **портит статистику**: μ склеенной карточки не описывает
ни одну из двух сущностей, и планировщик строит на ней планы, которые проваливаются в
половине случаев по причине, которой в модели нет вовсе.

## Тождество провизорно

Карточка не «есть» сущность, а «пока считается» сущностью. У тождества есть оценка
(`Entity.identity_confidence`), и она падает при каждом пересмотре: карточка, однажды
слитая или расщеплённая, менее надёжна как единица опыта, и это должно быть видно тому,
кто на неё опирается.

Пересмотр происходит **во сне**, когда накопилось свидетельство, а не в момент встречи.
Причина простая: в момент встречи свидетельства ещё нет — есть одно наблюдение, и решать
по нему тождество значит решать монеткой.

## Расщепление задним числом

Карточка, оказавшаяся двумя, делится, а ссылки на неё разводятся **по эпизодам**, а не
наугад. Для этого карточка держит встречи поштучно (`Sighting`): каждая знает свой номер
записи и свой исход, и при делении уходит той половине, к которой относится по исходу.
Связи от других карточек тоже разводятся по эпизоду — связь, наблюдённая на записи 40,
уходит той половине, которой принадлежит запись 40.

Наугад развести нельзя, и это не вопрос аккуратности. Ссылка, отданная не той половине,
создаёт третью ошибку поверх исправляемой: была одна склеенная карточка, стало две, и
обе с чужими связями.

## Функция отпечатка заменяема

Отпечаток строится восприятием, чьё качество на живом экране ещё не измерено. Поэтому
`Fingerprinter` — параметр, а не константа, и механика тождества от него не зависит:
если качество отпечатка поедет, менять придётся одну функцию, а не слияние с
расщеплением. Проверяется это тем, что одна и та же механика прогоняется на **двух**
источниках отпечатков, и числа приводятся по каждому отдельно.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Callable, Iterable, Protocol, Sequence

from .beliefs import (Belief, BeliefError, BeliefStore, Entity, EntityKind, Origin,
                      Provenance, Relation, Sighting, entity_id)


class Fingerprinter(Protocol):
    """Чем опознаётся сущность. Заменяемая часть: `name` попадает в каждую встречу.

    Необязательный `near(a, b)` — ответ источника на вопрос «это, возможно, один и тот же
    предмет, опознанный по-разному». Вопрос принадлежит **источнику**, а не механике
    тождества: близость отпечатков есть свойство функции отпечатка, и знать о ней может
    только она. Огрубляющий отпечаток на этот вопрос отвечает, точный хеш — нет, и это
    настоящая разница между источниками, а не недоделка одного из них.
    """

    name: str

    def __call__(self, payload: Any) -> str: ...

    def near(self, a: str, b: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class Quantized:
    """Отпечаток по огрублению: значения режутся на корзины заданной ширины.

    Ширина корзины и есть то, чем регулируется цена ошибки в обе стороны, и она **не
    захардкожена**: грубее — больше слипания, тоньше — больше расщепления. Прогон с
    другой шириной несравним с предыдущим, поэтому ширина живёт в профиле
    (`fingerprint_bucket`) и попадает в `profile_hash` (инвариант 23).
    """

    bucket: float
    name: str = "квантование"
    # Таблица «отпечаток → корзины». Заполняется при вычислении: иначе `near` не имеет
    # входных данных, ведь механика тождества исходных значений не видит.
    _seen: dict[str, tuple[int, ...]] = field(default_factory=dict, compare=False)

    def __call__(self, payload: Any) -> str:
        vals = payload if isinstance(payload, (list, tuple)) else [payload]
        cells = tuple(int(float(v) / self.bucket) for v in vals)
        blob = ",".join(str(c) for c in cells)
        got = hashlib.blake2b(blob.encode("utf-8"), digest_size=3).hexdigest().upper()
        self._seen[got] = cells
        return got

    def near(self, a: str, b: str) -> bool:
        """Отпечатки соседних корзин считаются близкими.

        Соседство считается по самим отпечаткам, а не по исходным значениям: значения
        механике тождества недоступны — она видит только то, что записано во встрече.
        Поэтому источник держит таблицу соседства сам (`_seen`), заполняя её при
        вычислении отпечатков: это единственное место, где известны и корзина, и её
        номер.
        """
        ca, cb = self._seen.get(a), self._seen.get(b)
        if ca is None or cb is None:
            return False
        return all(abs(x - y) <= 1 for x, y in zip(ca, cb)) and len(ca) == len(cb)

    @classmethod
    def from_profile(cls, profile: Any) -> "Quantized":
        # Настройка структурная: ширина корзины меняет форму опыта.
        return cls(float(profile.structural["fingerprint_bucket"]))


#: Что считает тождество, а что — свойство. Набор закрытый и объявленный.
#:
#: Из спецификации, A2.4: инвариантная часть — форма, положение, поведение; изменчивая —
#: цвет, яркость, размер на экране. Деление не вкусовое, а проверяемое: перекрашенный
#: предмет обязан остаться той же карточкой, а перекраска — стать событием.
INVARIANT_PARTS: tuple[str, ...] = ("shape", "position", "behaviour")
VARIABLE_PARTS: tuple[str, ...] = ("colour", "brightness", "size")


@dataclass(frozen=True, slots=True)
class PropertyChange:
    """Свойство сменилось при том же тождестве. Это событие, а не новая карточка.

    Ради этого A2.4 и делался: полоска здоровья, мигающий индикатор, смена дня и ночи
    — всё это один предмет с меняющимся свойством. Прежде каждое такое изменение
    заводило новую карточку, и предмет расщеплялся тем сильнее, чем живее он был.
    """

    fingerprint: str
    part: str
    before: Any
    after: Any
    seq: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"fingerprint": self.fingerprint, "part": self.part,
                "before": self.before, "after": self.after, "seq": self.seq}


@dataclass(frozen=True, slots=True)
class Split:
    """Отпечаток из двух частей: инвариантная считает тождество, изменчивая — свойства.

    Полезная нагрузка здесь — отображение с ключами из `INVARIANT_PARTS` и
    `VARIABLE_PARTS`, а не голое число: разделить части можно, только если источник
    сказал, что чем является. Ключ вне набора — ошибка, а не молчаливый пропуск:
    незнакомая часть, тихо ушедшая в свойства, сделала бы тождество слепым к тому,
    чего мы не предусмотрели.

    Тождество считает вложенный отпечаток (`inner`) — квантование или точный хеш, —
    и это осознанно: разделение частей и огрубление значений решают разные задачи, и
    смешивать их в одном классе значило бы получить один рычаг вместо двух.
    """

    inner: Fingerprinter
    name: str = "разделённый"
    _properties: dict[str, dict[str, Any]] = field(default_factory=dict, compare=False)
    _changes: list[PropertyChange] = field(default_factory=list, compare=False)

    def __call__(self, payload: Any) -> str:
        return self.observe(payload)[0]

    def observe(self, payload: Any, seq: int | None = None
                ) -> tuple[str, list[PropertyChange]]:
        """Отпечаток и список сменившихся свойств при том же тождестве."""
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"разделённому отпечатку нужна нагрузка с объявленными частями, "
                f"пришло {type(payload).__name__}. Без этого неизвестно, что здесь "
                "форма, а что цвет")
        unknown = sorted(set(payload) - set(INVARIANT_PARTS) - set(VARIABLE_PARTS))
        if unknown:
            raise ValueError(
                f"части вне набора: {unknown}. Набор закрыт: незнакомая часть, тихо "
                f"ушедшая в свойства, сделала бы тождество слепым. Известны "
                f"{sorted(INVARIANT_PARTS + VARIABLE_PARTS)}")
        core = [payload[k] for k in INVARIANT_PARTS if k in payload]
        if not core:
            raise ValueError(
                "в нагрузке нет ни одной инвариантной части: тождество считать нечем. "
                f"Инвариантные — {list(INVARIANT_PARTS)}")

        got = self.inner(core)
        now = {k: payload[k] for k in VARIABLE_PARTS if k in payload}
        was = self._properties.get(got)
        changes: list[PropertyChange] = []
        if was is not None:
            for part, value in now.items():
                if part in was and was[part] != value:
                    changes.append(PropertyChange(got, part, was[part], value, seq))
        self._properties[got] = {**(was or {}), **now}
        self._changes.extend(changes)
        return got, changes

    def near(self, a: str, b: str) -> bool:
        """Близость — свойство вложенного отпечатка: он один знает про корзины."""
        return self.inner.near(a, b)

    def properties(self, fingerprint: str) -> dict[str, Any]:
        """Последние известные свойства карточки. Пусто — свойств не приходило."""
        return dict(self._properties.get(fingerprint, {}))

    @property
    def changes(self) -> list[PropertyChange]:
        return list(self._changes)

    def state(self) -> dict[str, Any]:
        by_part: dict[str, int] = {}
        for c in self._changes:
            by_part[c.part] = by_part.get(c.part, 0) + 1
        return {"cards": len(self._properties), "changes": len(self._changes),
                "changes_by_part": dict(sorted(by_part.items())),
                "inner": self.inner.name}


@dataclass(frozen=True, slots=True)
class Exact:
    """Отпечаток по точному содержимому. Второй источник для сверки механики.

    Нужен именно как **другой** источник: если механика тождества где-то опирается на
    свойства огрубления, на точном отпечатке это вылезет числом — слияний станет ноль, а
    расщеплений столько же.
    """

    name: str = "точный"

    def __call__(self, payload: Any) -> str:
        blob = repr(payload)
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=3).hexdigest().upper()

    def near(self, a: str, b: str) -> bool:
        """Точный хеш о близости не знает ничего, и врать об этом не будет.

        Это не заглушка: два разных хеша действительно не несут сведений о том, близки ли
        их прообразы. Отсюда измеримое следствие — на точном отпечатке расщепление
        одного предмета на двадцать карточек механикой тождества не лечится, и лечить
        его надо в функции отпечатка.
        """
        return False


@dataclass(frozen=True, slots=True)
class Guess:
    """Догадка о тождестве: две карточки, оценка и на чём она стоит.

    `confidence` — не «похожесть», а оценка того, что это одна сущность. Разница
    существенна: две двери похожи и при этом различны, поэтому в оценку входит и
    согласие исходов, и совпадение отпечатков, и число наблюдений, на которых это видно.
    """

    a: str
    b: str
    confidence: float
    shared_keys: tuple[str, ...]
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "confidence": round(self.confidence, 4),
                "shared_keys": list(self.shared_keys), "evidence": self.evidence}


@dataclass(frozen=True, slots=True)
class SplitPlan:
    """План расщепления: по какому ключу и как расходятся встречи по эпизодам."""

    entity: str
    key: str
    yes_seqs: tuple[int, ...]        # эпизоды, где исход был положительным
    no_seqs: tuple[int, ...]
    confidence: float
    evidence: str

    def as_dict(self) -> dict[str, Any]:
        return {"entity": self.entity, "key": self.key,
                "yes": list(self.yes_seqs), "no": list(self.no_seqs),
                "confidence": round(self.confidence, 4), "evidence": self.evidence}


# ---------------------------------------------------------------------------
# Догадки о слиянии
# ---------------------------------------------------------------------------


def propose_merges(store: BeliefStore, *, min_shared: int, min_confidence: float = 0.0,
                   agree_band: float, fp: Any = None) -> list[Guess]:
    """Какие карточки, возможно, одна сущность.

    Три входа в оценку, и каждый отвечает на свой способ ошибиться:

    - **общие ключи** — если у двух карточек нет общих действий, сравнивать нечего, и
      «похожи» означало бы только «обе почти пусты»;
    - **согласие исходов** по общим ключам: одно и то же действие даёт одно и то же;
    - **общий отпечаток** — прямое свидетельство, что опознавали их одинаково. Оно
      сильнее согласия исходов, потому что согласие бывает случайным у двух похожих.
    """
    out: list[Guess] = []
    by_kind: dict[str, list[Entity]] = {}
    for e in store.entities.values():
        by_kind.setdefault(e.kind, []).append(e)
    for group in by_kind.values():
        group.sort(key=lambda e: e.id)
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                g = _guess(a, b, min_shared=min_shared, agree_band=agree_band, fp=fp)
                # Догадки возвращаются **все**, а фильтрует порог вызывающий. Иначе
                # знаменатель доли ложных срабатываний состоит только из применённых, и
                # доля структурно равна нулю — то есть проверка по инварианту 31
                # рапортует «ложных нет» по построению, а не по измерению.
                if g is not None and g.confidence >= min_confidence:
                    out.append(g)
    out.sort(key=lambda g: (-g.confidence, g.a, g.b))
    return out


def _guess(a: Entity, b: Entity, *, min_shared: int, agree_band: float,
           fp: Any = None) -> Guess | None:
    """Догадка о тождестве двух карточек, либо `None`.

    **Общий отпечаток обязателен, и это исправление, а не строгость.** Первая редакция
    считала достаточным согласие исходов по общим ключам, и замер показал, что она
    склеивает мир: шесть верно различённых сущностей превращались в две — все, что
    отвечают на нажатие, в одну, все, что не отвечают, в другую. Ошибка тождества росла
    с нуля до четырёх, то есть пересмотр делал модель **хуже**, чем до него.

    Причина в подмене понятий: согласие аффордансов — признак **категории**, а не
    тождества. Две двери похожи и при этом различны; `MIND.md`, раздел 4, так и говорит —
    категория есть кластер карточек со схожими аффордансами, — и кластер не значит «одна
    сущность». Поэтому похожесть без общего отпечатка идёт в `propose_categories`, а
    здесь требуется прямое свидетельство: их опознавали одним и тем же отпечатком.
    """
    same_print = set(a.fingerprints) & set(b.fingerprints)
    near_print = ""
    if not same_print and fp is not None and hasattr(fp, "near"):
        # Близость отпечатков спрашивается у источника: сама механика значений не видит.
        for x in a.fingerprints:
            for y in b.fingerprints:
                if fp.near(x, y):
                    near_print = f"{x}~{y}"
                    break
            if near_print:
                break
    if not same_print and not near_print:
        return None
    ka = {**a.affordances, **a.dynamics}
    kb = {**b.affordances, **b.dynamics}
    shared = sorted(set(ka) & set(kb))
    # Общий отпечаток — свидетельство прямое; близкий — слабее, и оценка ниже на четверть.
    evidence = (f"общий отпечаток {sorted(same_print)[0]}" if same_print
                else f"близкие отпечатки {near_print}")
    base = 0.5 if same_print else 0.25
    if len(shared) < min_shared:
        return Guess(a.id, b.id, base, (), f"{evidence}, общих ключей нет")
    agree = sum(1 for k in shared if abs(ka[k].mu - kb[k].mu) <= agree_band)
    conf = min(1.0, base + 0.5 * (agree / len(shared)))
    return Guess(a.id, b.id, conf, tuple(shared),
                 f"{evidence}, общих ключей {len(shared)}, согласны {agree}")


@dataclass(frozen=True, slots=True)
class Category:
    """Кластер карточек со схожими аффордансами. **Не** тождество.

    `MIND.md`, раздел 4: категория — не новый тип данных, а кластер карточек, у которых от
    одинаковых действий происходит одинаковое. Новая встреченная вещь наследует
    аффордансы рода **как гипотезы**, и отсюда способность предполагать, что делать с
    невиданным.

    Отделено от слияния намеренно и дорогой ценой: пока похожесть считалась тождеством,
    шесть сущностей склеивались в две. Категория даёт то же обобщение, ничего не теряя:
    карточки остаются различными, а общее у них становится отдельным утверждением.
    """

    signature: tuple[str, ...]        # ключи, по которым карточки согласны
    members: tuple[str, ...]
    agree: float

    def as_dict(self) -> dict[str, Any]:
        return {"signature": list(self.signature), "members": list(self.members),
                "agree": round(self.agree, 4)}


def propose_categories(store: BeliefStore, *, agree_band: float,
                       min_members: int = 2) -> list[Category]:
    """Собрать категории: кто на что отвечает одинаково.

    Подпись кластера — набор ключей с округлённым исходом. Огрубление до «отвечает / не
    отвечает» намеренно: категория — про род, а не про точное значение, и требовать
    совпадения `mu` до третьего знака значило бы завести по категории на карточку.
    """
    buckets: dict[tuple[str, ...], list[str]] = {}
    for e in sorted(store.entities.values(), key=lambda x: x.id):
        keys = {**e.affordances, **e.dynamics}
        if not keys:
            continue
        sig = tuple(f"{k}={'да' if b.mu > 0.5 else 'нет'}"
                    for k, b in sorted(keys.items()))
        buckets.setdefault(sig, []).append(e.id)
    out = [Category(sig, tuple(ids), 1.0 - agree_band)
           for sig, ids in sorted(buckets.items()) if len(ids) >= min_members]
    out.sort(key=lambda c: (-len(c.members), c.signature))
    return out


def apply_merge(store: BeliefStore, guess: Guess) -> bool:
    """Слить `b` в `a`: встречи, убеждения и связи. Тождество после этого менее надёжно.

    Конфликтующее убеждение остаётся у того, у кого больше **собственного** опыта, а не
    больше `n`: пересказ не должен побеждать проверку числом повторов.
    """
    a = store.entities.get(guess.a)
    b = store.entities.get(guess.b)
    if a is None or b is None:
        return False
    for src, dst in ((b.affordances, a.affordances), (b.dynamics, a.dynamics)):
        for key, belief in src.items():
            cur = dst.get(key)
            if cur is None or belief.n_experience > cur.n_experience:
                dst[key] = belief
    for s in b.sightings:
        a.see(s)
    a.uses += b.uses
    a.recalls += b.recalls
    for other_id, kind in b.links.items():
        if other_id != a.id:
            a.links.setdefault(other_id, kind)
    for e in store.entities.values():
        if b.id in e.links:
            kind = e.links.pop(b.id)
            if e.id != a.id:
                e.links.setdefault(a.id, kind)
    a.identity_confidence = round(min(a.identity_confidence, guess.confidence), 6)
    a.identity_note = f"слияние {a.id}+{b.id}: {guess.evidence}"
    del store.entities[b.id]
    return True


# ---------------------------------------------------------------------------
# Расщепление задним числом
# ---------------------------------------------------------------------------


def propose_splits(store: BeliefStore, *, min_n: int, band: float,
                   min_prints: int = 2) -> list[SplitPlan]:
    """Какие карточки, возможно, две сущности, и как разойдутся эпизоды.

    Два независимых признака, и оба нужны:

    - **бимодальность при насыщении**: наблюдений много, `mu` держится посередине. Одно
      и то же действие даёт разное — но не случайно, а надвое;
    - **много отпечатков**: карточку опознавали разными отпечатками. Само по себе это
      законно (вид меняется от освещения), но вместе с бимодальностью это уже не шум.

    Разделять по одному признаку нельзя. Середина при малом `n` — «ещё не знаю», а много
    отпечатков без бимодальности — просто грубая функция отпечатка, и лечится она в
    функции, а не расщеплением карточек.
    """
    out: list[SplitPlan] = []
    for e in sorted(store.entities.values(), key=lambda x: x.id):
        prints = len(e.fingerprints)
        for key, b in sorted({**e.affordances, **e.dynamics}.items()):
            if b.n < min_n or abs(b.mu - 0.5) > band:
                continue
            yes = tuple(s.seq for s in e.sightings if s.key == key and s.outcome is True)
            no = tuple(s.seq for s in e.sightings if s.key == key and s.outcome is False)
            if not yes or not no:
                # Бимодальность видна в `mu`, а встреч по этому ключу поштучно нет:
                # делить нечем. Это не «нет расщепления», а нехватка данных о встречах,
                # и молчать о ней нельзя — иначе расщепление задним числом выглядит
                # работающим там, где ему не на чём работать.
                continue
            conf = min(1.0, 0.5 + 0.1 * prints + min(0.3, b.n / 100.0))
            out.append(SplitPlan(
                e.id, key, yes, no, conf,
                f"наблюдений {b.n}, mu {b.mu:.2f} в полосе {band:.2f}, "
                f"отпечатков {prints}"))
        if prints < min_prints:
            continue
    out.sort(key=lambda p: (-p.confidence, p.entity, p.key))
    return out


def apply_split(store: BeliefStore, plan: SplitPlan, *,
                branch: str) -> tuple[str, str]:
    """Расщепить карточку. Встречи и ссылки разводятся **по эпизодам**.

    Возвращает идентификаторы двух половин. Исходная карточка исчезает: оставить её
    рядом значило бы держать в хранилище утверждение, которое сама система только что
    признала неверным.

    Убеждения половин пересчитываются **из их встреч**, а не делятся пополам. Деление
    пополам сохранило бы `mu` посередине у обеих — то есть ровно ту ошибку, ради
    исправления которой расщепление и делается.
    """
    old = store.entities.get(plan.entity)
    if old is None:
        raise BeliefError(f"нет карточки {plan.entity}: расщеплять нечего")
    yes_set, no_set = set(plan.yes_seqs), set(plan.no_seqs)

    def half(tag: str, seqs: set[int]) -> Entity:
        mine = [s for s in old.sightings if s.seq in seqs]
        if not mine:
            raise BeliefError(f"половина {tag} без встреч: делить нечем")
        new_id = entity_id(f"{plan.entity}|{plan.key}|{tag}")
        e = Entity(new_id, old.kind, first_seq=min(s.seq for s in mine),
                   last_seq=max(s.seq for s in mine), encounters=0)
        for s in mine:
            e.see(s)
        # Убеждения — из своих встреч, и происхождение у них собственное: половина
        # знает только то, что наблюдалось в её эпизодах.
        for s in mine:
            if not s.key or s.outcome is None:
                continue
            prov = Provenance(Origin.EXPERIENCE, branch, s.seq)
            cur = e.affordances.get(s.key)
            e.affordances[s.key] = (cur.observe(s.outcome, prov) if cur else Belief(
                f"{new_id}|afford|{s.key}", 1.0 if s.outcome else 0.0, 0.5, 1, prov, 1,
                s.seq, checked_at=(s.seq,)))
        e.identity_confidence = round(min(old.identity_confidence, plan.confidence), 6)
        e.identity_note = (f"расщепление {plan.entity} по «{plan.key}» ({tag}): "
                           f"{plan.evidence}")
        return e

    a, b = half("да", yes_set), half("нет", no_set)
    store.entities[a.id] = a
    store.entities[b.id] = b

    # Ссылки разводятся по эпизоду. Связь от другой карточки известна не по эпизоду
    # напрямую — эпизод берётся по её собственной последней встрече, и это объявленное
    # приближение, а не догадка: у связи нет своего времени, а у карточки есть.
    for e in list(store.entities.values()):
        if plan.entity not in e.links or e.id in (a.id, b.id):
            continue
        kind = e.links.pop(plan.entity)
        when = e.last_seq
        near = a if _closer(when, plan.yes_seqs) <= _closer(when, plan.no_seqs) else b
        e.links[near.id] = kind
        near.links.setdefault(e.id, kind)
    for other_id, kind in old.links.items():
        if other_id in store.entities and other_id not in (a.id, b.id):
            # Связь исходной карточки достаётся обеим половинам: чему именно она
            # принадлежала, наблюдением не разрешить, и приписать её одной было бы
            # выводом. `found_with` — честная позиция для такого случая.
            a.links.setdefault(other_id, kind)
            b.links.setdefault(other_id, str(Relation.FOUND_WITH))
    del store.entities[plan.entity]
    return a.id, b.id


def _closer(seq: int, seqs: Sequence[int]) -> int:
    return min((abs(seq - s) for s in seqs), default=10 ** 9)


# ---------------------------------------------------------------------------
# Показатели
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IdentityReport:
    """Что сделал пересмотр тождества. Единица независимости объявлена при каждом числе."""

    source: str                       # имя функции отпечатка
    before: int = 0
    after: int = 0
    merges: int = 0
    splits: int = 0
    merge_guesses: int = 0
    split_plans: int = 0
    single_encounter_before: int = 0
    single_encounter_after: int = 0
    unsplittable: int = 0             # бимодальные, но без поштучных встреч
    merge_passes: int = 0             # проходов слияния: тождество транзитивно
    categories: int = 0               # кластеров схожих карточек: обобщение без слияния
    categorised: int = 0              # карточек, попавших хоть в одну категорию
    merged_pairs: list[tuple[str, str]] = field(default_factory=list)
    split_into: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def merge_share(self) -> float | None:
        """Доля карточек, ушедших в слияние. Единица независимости — карточка."""
        return None if not self.before else self.merges / self.before

    @property
    def split_share(self) -> float | None:
        return None if not self.before else self.splits / self.before

    @property
    def false_merge_share(self) -> float | None:
        """Доля догадок о слиянии, которые не были применены (инвариант 31)."""
        if not self.merge_guesses:
            return None
        return (self.merge_guesses - self.merges) / self.merge_guesses

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "before": self.before, "after": self.after,
            "merges": self.merges, "splits": self.splits,
            "merge_guesses": self.merge_guesses, "split_plans": self.split_plans,
            "merge_share": self.merge_share, "split_share": self.split_share,
            "false_merge_share": self.false_merge_share,
            "single_encounter_before": self.single_encounter_before,
            "single_encounter_after": self.single_encounter_after,
            "unsplittable": self.unsplittable,
            "merged_pairs": [list(p) for p in self.merged_pairs],
            "split_into": [list(t) for t in self.split_into],
            "merge_passes": self.merge_passes,
            "categories": self.categories, "categorised": self.categorised,
            "unit": "карточка",
            "unit_for_method": "источник отпечатка",
        }

    def line(self) -> str:
        ms = "—" if self.merge_share is None else f"{self.merge_share:.0%}"
        ss = "—" if self.split_share is None else f"{self.split_share:.0%}"
        fm = "—" if self.false_merge_share is None else f"{self.false_merge_share:.0%}"
        return (f"{self.source}: карточек {self.before} → {self.after}, "
                f"слияний {self.merges} ({ms}), расщеплений {self.splits} ({ss}), "
                f"ложных догадок о слиянии {fm}, "
                f"с единственной встречей {self.single_encounter_before} → "
                f"{self.single_encounter_after}, "
                f"категорий {self.categories} на {self.categorised} карточек")


def revise(store: BeliefStore, *, profile: Any, source: str,
           fp: Any = None, apply: bool = True) -> IdentityReport:
    """Пересмотр тождества: слияния и расщепления. Вызывается сном.

    Порядок обязателен: сначала расщепления, потом слияния. Обратный порядок сначала
    склеивает две сущности в одну, а потом честно находит у неё бимодальность и делит
    обратно — и это не безобидный холостой ход: делит она уже по общему множеству
    встреч, где эпизоды двух исходных карточек перемешаны, и половины получаются не те,
    что были.
    """
    p = profile.parameters
    rep = IdentityReport(source=source, before=len(store.entities))
    rep.single_encounter_before = sum(1 for e in store.entities.values()
                                      if e.encounters == 1)
    branch = store.branch

    plans = propose_splits(store, min_n=int(p["split_min_observations"]),
                           band=float(p["split_middle_band"]))
    rep.split_plans = len(plans)
    rep.unsplittable = _unsplittable(store, min_n=int(p["split_min_observations"]),
                                     band=float(p["split_middle_band"]))
    if apply:
        for plan in plans:
            if plan.entity in store.entities:
                a, b = apply_split(store, plan, branch=branch)
                rep.split_into.append((plan.entity, a, b))
                rep.splits += 1

    # Слияния идут проходами до исчерпания, а не одним проходом. Тождество транзитивно:
    # если A и B — одно, а B и C — одно, то A и C тоже одно, но при вычислении догадок
    # один раз это не видно — на момент вычисления у A и C нет ни общего, ни близкого
    # отпечатка, он появляется только после того, как A вобрала отпечатки B.
    #
    # Замер показал цену одного прохода прямо: карточка, раздробленная отпечатком на
    # двенадцать, склеивалась обратно только до девяти. Проходов немного и они сходятся:
    # каждый проход уменьшает число карточек хотя бы на одну, иначе цикл кончается.
    for _ in range(int(p["identity_max_passes"])):
        guesses = propose_merges(store, min_shared=int(p["identity_min_shared_keys"]),
                                 agree_band=float(p["identity_agree_band"]), fp=fp)
        rep.merge_guesses += len(guesses)
        bar = float(p["identity_min_confidence"])
        if not apply:
            break
        did = 0
        for g in [x for x in guesses if x.confidence >= bar]:
            if g.a in store.entities and g.b in store.entities:
                if apply_merge(store, g):
                    rep.merged_pairs.append((g.a, g.b))
                    rep.merges += 1
                    did += 1
        rep.merge_passes += 1
        if not did:
            break

    cats = propose_categories(store, agree_band=float(p["identity_agree_band"]))
    rep.categories = len(cats)
    rep.categorised = len({m for c in cats for m in c.members})

    rep.after = len(store.entities)
    rep.single_encounter_after = sum(1 for e in store.entities.values()
                                     if e.encounters == 1)
    return rep


def _unsplittable(store: BeliefStore, *, min_n: int, band: float) -> int:
    """Сколько бимодальных карточек нельзя разделить: встреч поштучно нет.

    Число печатается всегда, даже когда оно нуль. Без него расщепление задним числом
    выглядело бы работающим на записях, где встречи не сохранялись, — а там оно просто
    не срабатывает, и молчание об этом неотличимо от «делить было нечего».
    """
    n = 0
    for e in store.entities.values():
        for key, b in {**e.affordances, **e.dynamics}.items():
            if b.n < min_n or abs(b.mu - 0.5) > band:
                continue
            yes = any(s.key == key and s.outcome is True for s in e.sightings)
            no = any(s.key == key and s.outcome is False for s in e.sightings)
            if not (yes and no):
                n += 1
    return n
