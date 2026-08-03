"""Экземпляры: параллельные жизни.

Из дизайна пульта: «параллельные жизни. У каждого свой мир и свой журнал: общего
опыта не существует, только свидетельства».

Это не оптимизация и не масштабирование. Это защита от загрязнения: если два
экземпляра делятся опытом напрямую, нельзя сказать, что каждый из них открыл сам,
а что узнал. Поэтому обмен идёт только свидетельствами, и свидетельство опытом не
становится никогда — сколько бы экземпляров его ни повторили. Повторение
свидетельства не есть проверка.

Отсюда три правила, каждое проверяемое:

1. **У каждого экземпляра свой журнал и свой сид мира.** Общего журнала нет.
2. **Из очереди слияния приходят `Testimony`, а не `Belief`.** Доверие берётся из
   профиля (`testimony_trust_instance`), и `n_experience` остаётся нулём.
3. **Конфликт со своим опытом решается в пользу опыта.** Чужое слово не
   переписывает то, что проверено своими руками; конфликт при этом не
   проглатывается, а показывается исследователю.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

from .core.clocks import Stamp
from .core.journal import Actor, Journal, Kind as EntryKind
from .core.profile import Profile, short
from .model.beliefs import BeliefStore, Testimony, merge_testimony


class State(StrEnum):
    LIVE = "live"
    ASLEEP = "asleep"
    STOPPED = "stopped"
    FREE = "free"


@dataclass(slots=True)
class Instance:
    """Одна жизнь: свой мир, свой журнал, своя память."""

    id: str
    seed: int
    profile: Profile
    state: State = State.FREE
    root: Path | None = None
    journal: Journal | None = None
    beliefs: BeliefStore | None = None
    age_ticks: int = 0
    found: int = 0                 # сколько нового открыл
    error_mean: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "seed": self.seed, "state": str(self.state),
                "age_ticks": self.age_ticks, "found": self.found,
                "error_mean": round(self.error_mean, 6),
                "profile_hash": short(self.profile.profile_hash),
                "structure_hash": short(self.profile.structure_hash),
                "cards": 0 if self.beliefs is None else len(self.beliefs)}


@dataclass(frozen=True, slots=True)
class MergeItem:
    """Строка очереди слияния: кто сказал, что сказал, чему это противоречит."""

    claim: str
    from_instance: str
    trust: float
    their_sigma: float
    my_mu: float | None = None      # None — у меня об этом ничего нет

    @property
    def conflicts(self) -> bool:
        return self.my_mu is not None and abs(self.my_mu - self.trust) > 0.5

    def as_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "from": self.from_instance, "trust": self.trust,
                "their_sigma": self.their_sigma, "my_mu": self.my_mu,
                "conflicts": self.conflicts}


class Colony:
    """Набор экземпляров и очередь свидетельств между ними."""

    def __init__(self, profile: Profile) -> None:
        self.profile = profile
        self.limit = int(profile.structural["instances"])
        self.trust = float(profile.parameters["testimony_trust_instance"])
        self.requires_recheck = bool(profile.structural["testimony_requires_recheck"])
        self.instances: dict[str, Instance] = {}
        self.queue: list[MergeItem] = []

    # --- состав -------------------------------------------------------------

    def spawn(self, seed: int, *, root: Path | None = None,
              profile: Profile | None = None) -> Instance:
        """Завести экземпляр. Сид мира свой: иначе открытие не отличить от памяти."""
        if len(self.instances) >= self.limit:
            raise ValueError(
                f"экземпляров уже {len(self.instances)} при пределе {self.limit}. "
                "Число экземпляров — структурный переключатель: менять его на ходу "
                "нельзя, он форкает журнал")
        used = {i.seed for i in self.instances.values()}
        if seed in used and bool(self.profile.structural["randomize_world"]):
            raise ValueError(
                f"сид {seed} уже занят. При включённой рандомизации миры экземпляров "
                "обязаны различаться, иначе нельзя отличить открытие от воспоминания")
        iid = f"INST_{len(self.instances):02d}"
        inst = Instance(iid, seed, profile or self.profile, State.LIVE, root)
        self.instances[iid] = inst
        return inst

    def get(self, iid: str) -> Instance:
        try:
            return self.instances[iid]
        except KeyError:
            raise KeyError(f"нет экземпляра {iid}") from None

    def by_state(self, state: State) -> list[Instance]:
        return [i for i in self.instances.values() if i.state is state]

    # --- обмен --------------------------------------------------------------

    def share(self, from_instance: str, claims: Iterable[str], *,
              their_sigma: float = 0.5) -> int:
        """Экземпляр рассказывает остальным. Это свидетельство, а не опыт."""
        inst = self.get(from_instance)
        added = 0
        for claim in claims:
            my = None
            if inst.beliefs is not None:
                for b in inst.beliefs.beliefs():
                    if b.claim == claim and not b.is_hearsay:
                        my = b.mu
                        break
            self.queue.append(MergeItem(claim, from_instance, self.trust, their_sigma, my))
            added += 1
        return added

    def deliver(self, to_instance: str, stamp: Stamp) -> dict[str, Any]:
        """Влить очередь в память экземпляра. Свидетельством, с перепроверкой.

        Записи `TESTIMONY` идут в журнал получателя: иначе после пересборки из
        журнала чужие слова исчезнут, и модель перестанет объясняться журналом.
        """
        inst = self.get(to_instance)
        if inst.beliefs is None:
            inst.beliefs = BeliefStore(inst.journal.meta.branch_id if inst.journal else inst.id)
        branch = inst.beliefs.branch

        mine = [item for item in self.queue if item.from_instance != to_instance]
        testimonies = []
        for i, item in enumerate(mine):
            seq = inst.journal.seq if inst.journal is not None else i
            if inst.journal is not None and inst.journal.mode == "a":
                inst.journal.append(EntryKind.TESTIMONY, stamp, Actor.NONE,
                                    event={"code": "from_instance", "claim": item.claim,
                                           "source": item.from_instance,
                                           "trust": item.trust,
                                           "their_sigma": item.their_sigma})
            testimonies.append(Testimony(item.claim, item.from_instance, item.trust,
                                         branch, seq))
        result = merge_testimony(inst.beliefs, testimonies,
                                 requires_recheck=self.requires_recheck)
        self.queue = [item for item in self.queue if item.from_instance == to_instance]
        return {"to": to_instance, "delivered": len(testimonies), **result}

    # --- сводка -------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "limit": self.limit,
            "live": len(self.by_state(State.LIVE)),
            "asleep": len(self.by_state(State.ASLEEP)),
            "stopped": len(self.by_state(State.STOPPED)),
            "free": max(0, self.limit - len(self.instances)),
            "queue": len(self.queue),
            "conflicts": sum(1 for i in self.queue if i.conflicts),
            "trust_instance": self.trust,
            "requires_recheck": self.requires_recheck,
            "instances": [i.as_dict() for i in self.instances.values()],
        }
