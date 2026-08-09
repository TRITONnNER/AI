"""Карточки, убеждения, гипотезы, свидетельства, происхождение.

Инвариант 7: у каждого убеждения есть происхождение (`hunch` | `testimony` |
`experience`), `mu`, `sigma`, `n` и ссылка на эпизод. Утверждение без
происхождения в хранилище не попадает — это проверяется конструктором, а не
намерением.

Инвариант 1: всё здесь — производное. `rebuild_from_journal` собирает хранилище с
нуля и обязан дважды дать одно и то же. Поэтому ни одно поле не зависит от
настенного времени, порядка обхода словаря или случайности.

Разница между тремя происхождениями содержательная, а не формальная:

- `hunch` — догадка из одного наблюдения; может быть отброшена без потерь.
- `testimony` — сказано кем-то (человеком, другим экземпляром, вики). Опытом не
  становится **никогда**, сколько бы раз ни повторили: повторение свидетельства
  не есть проверка. Только собственная попытка переводит утверждение в опыт.
- `experience` — проверено своим действием в своём мире.

Поэтому у убеждения два счётчика: `n` (сколько всего подтверждений) и
`n_experience` (сколько из них собственных). Убеждение, у которого `n` большое, а
`n_experience` нулевое, — это пересказ, и выглядеть уверенным оно не должно.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Iterator, Mapping


class BeliefError(ValueError):
    pass


class Origin(StrEnum):
    HUNCH = "hunch"
    TESTIMONY = "testimony"
    EXPERIENCE = "experience"


@dataclass(frozen=True, slots=True)
class Provenance:
    """Откуда взялось утверждение. Без этого в хранилище не попасть."""

    origin: Origin
    branch: str                 # ветка журнала: опыт из другой ветки несравним
    seq: int                    # номер записи журнала — тот самый «эпизод»
    source: str = "self"        # кто сказал, если это свидетельство
    trust: float = 1.0          # доверие к источнику; у собственного опыта 1.0

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise BeliefError(f"ссылка на эпизод отрицательная: {self.seq}")
        if not self.branch:
            raise BeliefError("происхождение без ветки журнала: опыт станет несравнимым")
        if not 0.0 <= self.trust <= 1.0:
            raise BeliefError(f"доверие вне [0,1]: {self.trust}")
        if self.origin is Origin.EXPERIENCE and self.source != "self":
            raise BeliefError(
                f"опыт не может быть от {self.source!r}: чужое — это свидетельство")
        if self.origin is Origin.TESTIMONY and self.source == "self":
            raise BeliefError("свидетельство от самого себя — это опыт или догадка")

    def as_dict(self) -> dict[str, Any]:
        return {"origin": str(self.origin), "branch": self.branch, "seq": self.seq,
                "source": self.source, "trust": self.trust}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Provenance:
        return cls(Origin(d["origin"]), str(d["branch"]), int(d["seq"]),
                   str(d.get("source", "self")), float(d.get("trust", 1.0)))


@dataclass(frozen=True, slots=True)
class Belief:
    """Утверждение с оценкой и происхождением.

    `claim` — непрозрачный ключ утверждения, не человеческая фраза: сюда нельзя
    положить текст с экрана (инвариант 5). Формат — `предмет|отношение|объект`,
    где все три части либо символы, либо идентификаторы.
    """

    claim: str
    mu: float
    sigma: float
    n: int
    provenance: Provenance
    n_experience: int = 0
    last_seq: int = 0

    def __post_init__(self) -> None:
        if not self.claim:
            raise BeliefError("утверждение без формулировки")
        if self.provenance is None:
            raise BeliefError(
                "утверждение без происхождения в хранилище не попадает (инвариант 7)")
        if self.n < 1:
            raise BeliefError(f"n должно быть не меньше 1: {self.n}")
        if not 0.0 <= self.mu <= 1.0:
            raise BeliefError(f"mu вне [0,1]: {self.mu}")
        if self.sigma < 0:
            raise BeliefError(f"sigma отрицательная: {self.sigma}")
        if self.n_experience > self.n:
            raise BeliefError(f"собственных подтверждений {self.n_experience} больше общих {self.n}")
        if self.provenance.origin is Origin.EXPERIENCE and self.n_experience < 1:
            raise BeliefError("происхождение «опыт» без ни одного собственного подтверждения")

    @property
    def is_hearsay(self) -> bool:
        """Ни разу не проверено своими руками. Уверенным выглядеть не должно."""
        return self.n_experience == 0

    @property
    def confidence(self) -> float:
        """Насколько можно опираться. Пересказ дисконтируется доверием источника.

        Не «уверенность модели», а пригодность для действия: убеждение с
        большим n, но нулевым собственным опытом даёт меньше, чем одна своя
        проверка, потому что рассказать могли и неправду.
        """
        base = 1.0 / (1.0 + self.sigma)
        if self.is_hearsay:
            return base * self.provenance.trust * 0.5
        own = self.n_experience / max(1, self.n)
        return base * (0.5 + 0.5 * own)

    def observe(self, outcome: bool, prov: Provenance) -> Belief:
        """Ещё одно подтверждение или опровержение. Возвращает новое убеждение."""
        n = self.n + 1
        mu = (self.mu * self.n + (1.0 if outcome else 0.0)) / n
        sigma = (max(mu * (1.0 - mu), 1e-9) / n) ** 0.5 if n > 1 else 0.5
        n_exp = self.n_experience + (1 if prov.origin is Origin.EXPERIENCE else 0)
        # Происхождение повышается: догадка, подтверждённая своими руками,
        # становится опытом. Обратно не понижается никогда.
        origin = self.provenance
        if prov.origin is Origin.EXPERIENCE and origin.origin is not Origin.EXPERIENCE:
            origin = prov
        return Belief(self.claim, mu, sigma, n, origin, n_exp, max(self.last_seq, prov.seq))

    def as_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "mu": round(self.mu, 6), "sigma": round(self.sigma, 6),
                "n": self.n, "n_experience": self.n_experience, "last_seq": self.last_seq,
                "provenance": self.provenance.as_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Belief:
        return cls(str(d["claim"]), float(d["mu"]), float(d["sigma"]), int(d["n"]),
                   Provenance.from_dict(d["provenance"]), int(d.get("n_experience", 0)),
                   int(d.get("last_seq", 0)))


class State(StrEnum):
    """Три исхода конвейера убеждений, а не два (инвариант 16).

    `DEFERRED` — не «пока не проверили», а «проверить нечем»: теста не
    существует. Такая гипотеза не тратит бюджет, не удаляется и переоткрывается
    при появлении новой возможности — нового места, нового навыка, умения читать,
    встречи с новым типом сущностей.

    Без третьего исхода отказ по перерасходу выбросил бы все загадки агента, и он
    вечно забывал бы, чего не понял.
    """

    VERIFIED = "verified"
    REFUTED = "refuted"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """Ещё не убеждение. «Гипотеза знанием не является»: нужна проверка в мире.

    `test` — что надо сделать, чтобы проверить, в терминах действий агента.
    **`None` — законное состояние, и означает оно вопрос.**

    Прежде здесь стоял запрет: гипотеза без теста не создавалась. Запрет выглядел
    дисциплиной, а был структурным исключением всего интересного. По `MIND.md`,
    раздел 5, **вопрос — это в точности гипотеза, для которой агент не может
    построить тест**, и распознаётся он именно по отсутствию конструируемой
    проверки. Пока такое состояние было запрещено, кода не существовало ни для
    `deferred`, ни для детектора недостающей категории, ни для гипотез
    `absent_agent`, ни для археологии: всё это начинается с утверждения, которое
    агент сформулировал и проверить не смог.

    Характерная сигнатура вопроса — расходимость двух кривых по одной сущности:
    аффордансы насыщаются нормально (`sigma` сужается), а происхождение не
    двигается вообще, потому что свидетельство лежит вне достижимого опыта.

    Что осталось обязательным: `claim` и `provenance`. Утверждение без
    происхождения в хранилище не попадает (инвариант 7), и на вопросы это
    распространяется — «откуда у меня взялся этот вопрос» не менее важно, чем
    откуда взялся ответ.
    """

    claim: str
    test: str | None
    prior: float
    provenance: Provenance
    checked: bool = False
    # Чем кончилась проверка. `None` — проверка есть, но её ещё не проводили; это
    # не исход конвейера, а работа в процессе, и путать одно с другим нельзя.
    outcome: bool | None = None
    # Почему теста нет. Обязательно при `test is None`: «не смог построить проверку»
    # без причины неотличимо от «забыл её написать», и множество отложенных
    # превратилось бы в свалку вместо перечня непонятого.
    deferred_reason: str = ""
    # Что могло бы дать проверку: новое место, навык, умение читать. Пустое —
    # законно и означает «не знаю даже того, чего мне не хватает».
    reopens_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.claim:
            raise BeliefError("гипотеза без формулировки")
        if self.provenance is None:
            raise BeliefError(
                "гипотеза без происхождения не хранится (инвариант 7): вопрос тоже "
                "откуда-то взялся")
        if self.test is None and not self.deferred_reason:
            raise BeliefError(
                f"гипотеза {self.claim!r} без теста и без причины. Отсутствие теста "
                "законно — это вопрос, — но причина обязательна: без неё множество "
                "отложенного станет свалкой, а оно должно быть перечнем того, что "
                "агент считает непонятым")
        if self.test is not None and not self.test:
            raise BeliefError(
                f"гипотеза {self.claim!r} с пустой строкой вместо теста. Пустая "
                "строка — это не «теста нет», это забытое поле. Вопрос пишется "
                "как test=None с указанной причиной")
        if self.test is not None and self.deferred_reason:
            raise BeliefError(
                f"гипотеза {self.claim!r} с тестом и с причиной отложить одновременно: "
                "либо проверка есть, либо её нет")
        if not 0.0 <= self.prior <= 1.0:
            raise BeliefError(f"prior вне [0,1]: {self.prior}")

    @property
    def is_question(self) -> bool:
        """Формальный признак вопроса: теста не существует, а не «пока не сделан»."""
        return self.test is None

    @property
    def state(self) -> State | None:
        """Исход конвейера, или `None`, если исхода ещё нет.

        Три исхода — `verified`, `refuted`, `deferred`. Гипотеза с тестом, который
        ещё не проводили, ни в один из них не попадает: она в работе. Вернуть для
        неё `deferred` значило бы записать «проверить нечем» там, где просто ещё не
        дошли руки, — и множество отложенного, то есть перечень непонятого, тут же
        забилось бы обычной очередью дел.
        """
        if self.is_question:
            return State.DEFERRED
        if self.outcome is None:
            return None
        return State.VERIFIED if self.outcome else State.REFUTED

    def with_test(self, test: str) -> Hypothesis:
        """Вопрос стал гипотезой: появилась возможность его проверить.

        Именно так работает переоткрытие: множество отложенного не удаляется, и
        когда у агента появляется новое место или новый навык, вопрос выходит из
        него обратно в работу. Обратного перехода нет: раз проверка нашлась, она
        не пропадает.

        `reopens_on` сохраняется: то, что дало проверку, — часть истории вопроса,
        и по нему потом видно, какая именно возможность разрешила загадку.
        """
        if not self.is_question:
            raise BeliefError(
                f"у гипотезы {self.claim!r} тест уже есть: {self.test!r}")
        if not test:
            raise BeliefError("переоткрытие без теста ничего не меняет")
        return Hypothesis(self.claim, test, self.prior, self.provenance,
                          checked=False, outcome=None, deferred_reason="",
                          reopens_on=self.reopens_on)

    def resolved(self, outcome: bool) -> Hypothesis:
        """Зафиксировать исход проверки: `verified` или `refuted`."""
        if self.is_question:
            raise BeliefError(
                f"вопрос {self.claim!r} не имеет исхода: проверять нечем")
        return Hypothesis(self.claim, self.test, self.prior, self.provenance,
                          checked=True, outcome=bool(outcome),
                          reopens_on=self.reopens_on)

    def confirm(self, outcome: bool, prov: Provenance) -> Belief:
        """Проверили в мире — стало убеждением с происхождением «опыт»."""
        if self.is_question:
            raise BeliefError(
                f"вопрос {self.claim!r} нельзя подтвердить: теста нет, и подтверждать "
                f"нечем. Причина, по которой он отложен: {self.deferred_reason}. "
                "Сначала with_test(), когда появится возможность")
        if prov.origin is not Origin.EXPERIENCE:
            raise BeliefError(
                "гипотеза становится убеждением только после собственной проверки; "
                "свидетельство её не подтверждает")
        return Belief(self.claim, 1.0 if outcome else 0.0, 0.5, 1, prov, 1, prov.seq)

    def as_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "test": self.test, "prior": self.prior,
                "checked": self.checked, "outcome": self.outcome,
                "state": None if self.state is None else str(self.state),
                "is_question": self.is_question,
                "deferred_reason": self.deferred_reason,
                "reopens_on": list(self.reopens_on),
                "provenance": self.provenance.as_dict()}

    @classmethod
    def question(cls, claim: str, prior: float, provenance: Provenance, *,
                 because: str, reopens_on: tuple[str, ...] = ()) -> Hypothesis:
        """Завести вопрос: утверждение, для которого проверка не строится.

        Отдельный конструктор, потому что это не «гипотеза с пропущенным полем», а
        другое эпистемическое состояние, и в коде оно должно читаться как таковое.
        """
        return cls(claim, None, prior, provenance, deferred_reason=because,
                   reopens_on=tuple(reopens_on))


@dataclass(frozen=True, slots=True)
class Testimony:
    """Сказанное кем-то. Ждёт перепроверки и опытом не становится."""

    claim: str
    source: str
    trust: float
    branch: str
    seq: int
    rechecked: bool = False

    def to_belief(self) -> Belief:
        prov = Provenance(Origin.TESTIMONY, self.branch, self.seq, self.source, self.trust)
        return Belief(self.claim, self.trust, 0.5, 1, prov, 0, self.seq)

    def as_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "source": self.source, "trust": self.trust,
                "branch": self.branch, "seq": self.seq, "rechecked": self.rechecked}


def entity_id(fingerprint: str) -> str:
    """Непрозрачный идентификатор карточки из отпечатка."""
    h = hashlib.blake2b(fingerprint.encode("utf-8"), digest_size=2).hexdigest().upper()
    return f"ENT_{h}"


@dataclass(slots=True)
class Entity:
    """Карточка сущности: что это, что я с этим умею, что оно делает само."""

    id: str
    kind: str                                    # "symbol" | "output" | "place" | "sound"
    first_seq: int
    last_seq: int
    encounters: int = 1
    affordances: dict[str, Belief] = field(default_factory=dict)   # действие → результат
    dynamics: dict[str, Belief] = field(default_factory=dict)      # что делает само
    links: dict[str, str] = field(default_factory=dict)            # id → вид связи
    uses: int = 0

    def value(self) -> float:
        """Ценность карточки: по ней решается, что забыть во сне.

        Считается из встреч, обращений и того, есть ли собственный опыт. Не из
        «важности»: важность — это оценка, а её надо откуда-то взять, и взять её
        честно неоткуда.
        """
        own = sum(1 for b in (*self.affordances.values(), *self.dynamics.values())
                  if not b.is_hearsay)
        total = len(self.affordances) + len(self.dynamics)
        proven = own / total if total else 0.0
        return min(1.0, 0.4 * min(1.0, self.encounters / 20.0)
                   + 0.3 * min(1.0, self.uses / 10.0)
                   + 0.3 * proven)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "first_seq": self.first_seq,
                "last_seq": self.last_seq, "encounters": self.encounters,
                "uses": self.uses, "value": round(self.value(), 4),
                "affordances": {k: v.as_dict() for k, v in sorted(self.affordances.items())},
                "dynamics": {k: v.as_dict() for k, v in sorted(self.dynamics.items())},
                "links": dict(sorted(self.links.items()))}


class BeliefStore:
    """Граф карточек. Пересобираем из журнала, поэтому ничего не хранит сверх него."""

    def __init__(self, branch: str) -> None:
        self.branch = branch
        self.entities: dict[str, Entity] = {}
        self.hypotheses: dict[str, Hypothesis] = {}
        self.testimonies: list[Testimony] = []

    # --- наполнение ---------------------------------------------------------

    def touch(self, ent_id: str, kind: str, seq: int) -> Entity:
        e = self.entities.get(ent_id)
        if e is None:
            e = Entity(ent_id, kind, first_seq=seq, last_seq=seq)
            self.entities[ent_id] = e
        else:
            e.encounters += 1
            e.last_seq = max(e.last_seq, seq)
        return e

    def learn_affordance(self, ent_id: str, action_key: str, outcome: bool,
                         prov: Provenance, *, kind: str = "output") -> Belief:
        """«Когда я делаю это с этим, получается вот то»."""
        e = self.touch(ent_id, kind, prov.seq)
        e.uses += 1
        claim = f"{ent_id}|afford|{action_key}"
        old = e.affordances.get(action_key)
        b = old.observe(outcome, prov) if old else Belief(
            claim, 1.0 if outcome else 0.0, 0.5, 1, prov,
            1 if prov.origin is Origin.EXPERIENCE else 0, prov.seq)
        e.affordances[action_key] = b
        return b

    def learn_dynamics(self, ent_id: str, what: str, outcome: bool,
                       prov: Provenance, *, kind: str = "symbol") -> Belief:
        """«Оно делает это само, без меня»."""
        e = self.touch(ent_id, kind, prov.seq)
        claim = f"{ent_id}|dyn|{what}"
        old = e.dynamics.get(what)
        b = old.observe(outcome, prov) if old else Belief(
            claim, 1.0 if outcome else 0.0, 0.5, 1, prov,
            1 if prov.origin is Origin.EXPERIENCE else 0, prov.seq)
        e.dynamics[what] = b
        return b

    def link(self, a: str, b: str, kind: str) -> None:
        if a in self.entities:
            self.entities[a].links[b] = kind
        if b in self.entities:
            self.entities[b].links[a] = kind

    def add_hypothesis(self, h: Hypothesis) -> None:
        self.hypotheses[h.claim] = h

    def add_testimony(self, t: Testimony) -> None:
        self.testimonies.append(t)

    # --- выборки ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entities)

    def beliefs(self) -> Iterator[Belief]:
        for e in self.entities.values():
            yield from e.affordances.values()
            yield from e.dynamics.values()

    def by_kind(self, kind: str) -> list[Entity]:
        return [e for e in self.entities.values() if e.kind == kind]

    def hearsay(self) -> list[Belief]:
        """Всё, что известно только с чужих слов. Требует своей проверки."""
        return [b for b in self.beliefs() if b.is_hearsay]

    def unchecked_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses.values() if not h.checked]

    def stats(self) -> dict[str, Any]:
        bs = list(self.beliefs())
        return {
            "branch": self.branch,
            "entities": len(self.entities),
            "by_kind": {k: len(self.by_kind(k))
                        for k in sorted({e.kind for e in self.entities.values()})},
            "beliefs": len(bs),
            "hearsay": sum(1 for b in bs if b.is_hearsay),
            "experience": sum(1 for b in bs if not b.is_hearsay),
            "hypotheses": len(self.hypotheses),
            "unchecked_hypotheses": len(self.unchecked_hypotheses()),
            "testimonies": len(self.testimonies),
            "mean_confidence": round(
                sum(b.confidence for b in bs) / len(bs), 4) if bs else 0.0,
        }

    def fingerprint(self) -> str:
        """Отпечаток всего хранилища: две пересборки обязаны дать одинаковый.

        Считается по отсортированному представлению, поэтому не зависит ни от
        порядка обхода, ни от того, в каком порядке шли записи одного эпизода.
        """
        import json
        blob = json.dumps(
            [e.as_dict() for _, e in sorted(self.entities.items())]
            + [h.as_dict() for _, h in sorted(self.hypotheses.items())]
            + [t.as_dict() for t in sorted(self.testimonies, key=lambda x: (x.seq, x.claim))],
            ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()

    def prune(self, below_value: float, *, keep: int | None = None) -> list[str]:
        """Забыть малоценное. Возвращает забытое.

        Забывание — операция над производным слоем, журнал при этом не меняется:
        забытая карточка восстановится при следующей пересборке, если в журнале
        для неё есть основания. Именно поэтому забывать безопасно.
        """
        # Два ограничения, а не одно. Сначала забывается всё, что ниже порога
        # ценности. Если после этого карточек всё равно больше предела, забывается
        # ещё — начиная с самых малоценных. Раньше `keep` глушил порог целиком, и
        # при большом пределе сон не забывал ничего вообще.
        victims = sorted((e for e in self.entities.values() if e.value() < below_value),
                         key=lambda e: (e.value(), e.id))
        if keep is not None:
            doomed = {e.id for e in victims}
            survivors = sorted((e for e in self.entities.values() if e.id not in doomed),
                               key=lambda e: (e.value(), e.id))
            over = max(0, len(survivors) - keep)
            victims.extend(survivors[:over])
        for e in victims:
            del self.entities[e.id]
        for e in self.entities.values():
            for gone in [k for k in e.links if k not in self.entities]:
                del e.links[gone]
        return [e.id for e in victims]


def merge_testimony(store: BeliefStore, testimonies: Iterable[Testimony], *,
                    requires_recheck: bool = True) -> dict[str, Any]:
    """Влить свидетельства в хранилище.

    Свидетельство никогда не становится опытом: оно ложится убеждением с
    происхождением `testimony`, и `n_experience` остаётся нулём. При
    `requires_recheck` для каждого заводится гипотеза с явной проверкой — иначе
    пересказ незаметно станет знанием.
    """
    added, conflicts = 0, []
    for t in testimonies:
        store.add_testimony(t)
        belief = t.to_belief()
        # Утверждение имеет вид `ENT_xxxx|вид|что`: три части, а не две. Разбор
        # на две склеивал вид с содержанием, и собственный опыт по тому же
        # утверждению не находился — то есть чужое слово тихо ложилось поверх.
        parts = belief.claim.split("|")
        ent_id = parts[0] if parts else ""
        what = parts[2] if len(parts) >= 3 else (parts[1] if len(parts) == 2 else t.claim)
        e = store.touch(ent_id or entity_id(t.claim), "symbol", t.seq)
        existing = e.dynamics.get(what)
        if existing is not None and not existing.is_hearsay and abs(
                existing.mu - belief.mu) > 0.5:
            conflicts.append({"claim": t.claim, "mine": existing.mu,
                              "theirs": belief.mu, "source": t.source})
            continue                      # свой опыт чужим словом не переписывается
        e.dynamics[what or t.claim] = belief
        added += 1
        if requires_recheck:
            store.add_hypothesis(Hypothesis(
                t.claim, test=f"проверить самому: {what or t.claim}",
                prior=t.trust, provenance=belief.provenance))
    return {"added": added, "conflicts": conflicts,
            "pending_recheck": len(store.unchecked_hypotheses())}
