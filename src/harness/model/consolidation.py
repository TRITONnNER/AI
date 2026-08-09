"""Сон: пересборка, слияние, забывание.

Что делает сон и чего он не делает.

**Делает:** пересобирает убеждения из журнала, предлагает слить похожие карточки,
забывает малоценное, считает, сколько времени прошло без сверки с реальностью.

**Не делает:** не выдумывает. Ни одного утверждения, которого нет в журнале, во
сне не появляется. Это важно, потому что соблазн ровно обратный: во сне удобно
«достроить» модель — и агент выучит свою модель вместо мира.

Предупреждение о сверке с реальностью — из дизайна пульта, и оно не косметика:
«обучаясь во сне, агент выучивает свою модель, а не мир». Если несколько прогонов
подряд шли только по воображаемому, точность в мире не проверялась, и об этом
надо сказать громко.

Слияние — предложение, а не действие. Две карточки, похожие на 0.9, могут быть
одной сущностью, а могут быть двумя похожими. Решает либо порог, либо
исследователь, но в любом случае решение пишется в журнал: слияние меняет модель
и должно быть объяснимо потом.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.clocks import Stamp
from ..core.journal import Actor, ActorLayer, Journal, Kind as EntryKind
from ..core.profile import Profile
from .beliefs import BeliefStore, Entity
from .rebuild import Rebuilt, rebuild_from_journal


@dataclass(frozen=True, slots=True)
class MergeProposal:
    """Предложение слить две карточки. Со сходством и с тем, что расходится."""

    a: str
    b: str
    similarity: float
    conflicts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"a": self.a, "b": self.b, "similarity": round(self.similarity, 4),
                "conflicts": self.conflicts}


@dataclass(slots=True)
class SleepReport:
    """Что было во сне. Пишется в журнал одной записью вида SLEEP."""

    run: int
    duration_s: float = 0.0          # сколько сон длится по профилю
    entries_read: int = 0
    entities_before: int = 0
    entities_after: int = 0
    forgotten: list[str] = field(default_factory=list)
    merges_proposed: list[MergeProposal] = field(default_factory=list)
    merges_applied: list[tuple[str, str]] = field(default_factory=list)
    hearsay_pending: int = 0
    reality_check_overdue: bool = False
    minutes_since_reality_check: float = 0.0
    fingerprint_before: str = ""
    fingerprint_after: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"run": self.run, "duration_s": round(self.duration_s, 3),
                "entries_read": self.entries_read,
                "entities_before": self.entities_before,
                "entities_after": self.entities_after,
                "forgotten": len(self.forgotten),
                "merges_proposed": [m.as_dict() for m in self.merges_proposed],
                "merges_applied": [list(p) for p in self.merges_applied],
                "hearsay_pending": self.hearsay_pending,
                "reality_check_overdue": self.reality_check_overdue,
                "minutes_since_reality_check": round(self.minutes_since_reality_check, 2),
                "fingerprint_before": self.fingerprint_before,
                "fingerprint_after": self.fingerprint_after}


def card_similarity(a: Entity, b: Entity) -> tuple[float, list[str]]:
    """Сходство двух карточек и то, в чём они расходятся.

    Считается по пересечению того, что о них известно: какие действия с ними
    дают какой результат и что они делают сами. Не по идентификаторам и не по
    порядку появления — только по содержанию.
    """
    if a.kind != b.kind:
        return 0.0, ["разный вид"]
    keys_a = set(a.affordances) | set(a.dynamics)
    keys_b = set(b.affordances) | set(b.dynamics)
    if not keys_a or not keys_b:
        return 0.0, ["нечего сравнивать"]
    common = keys_a & keys_b
    union = keys_a | keys_b
    conflicts: list[str] = []
    agree = 0
    for k in common:
        ba = a.affordances.get(k) or a.dynamics.get(k)
        bb = b.affordances.get(k) or b.dynamics.get(k)
        if ba is None or bb is None:
            continue
        if abs(ba.mu - bb.mu) <= 0.25:
            agree += 1
        else:
            conflicts.append(f"{k}: {ba.mu:.2f} против {bb.mu:.2f}")
    return (agree / len(union) if union else 0.0), conflicts


class Consolidator:
    """Прогон сна. Каждый прогон — запись в журнале и отчёт для исследователя."""

    def __init__(self, profile: Profile, *, journal: Journal | None = None) -> None:
        p = profile.parameters
        self.profile = profile
        self.journal = journal
        self.forget_below = float(p["forget_below_value"])
        self.merge_similarity = float(p["merge_similarity"])
        self.belief_cap = int(p["belief_cap"])
        self.overdue_minutes = float(p["reality_check_max_minutes"])
        self.duration_s = float(p["sleep_duration_s"])
        self.trust_human = float(p["testimony_trust_human"])
        self.live_min_responses = int(p["body_live_min_responses"])
        self.silent_min_deliveries = int(p["babble_repeats"])
        self.runs = 0
        self.last_reality_check_run: int | None = None

    def note_reality_check(self) -> None:
        """Отметить, что была сверка с реальностью: агент действовал в мире."""
        self.last_reality_check_run = self.runs

    def propose_merges(self, store: BeliefStore) -> list[MergeProposal]:
        """Найти похожие карточки. Квадратично, но только внутри одного вида."""
        out: list[MergeProposal] = []
        by_kind: dict[str, list[Entity]] = {}
        for e in store.entities.values():
            by_kind.setdefault(e.kind, []).append(e)
        for group in by_kind.values():
            group.sort(key=lambda e: e.id)
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    sim, conflicts = card_similarity(a, b)
                    if sim >= self.merge_similarity:
                        out.append(MergeProposal(a.id, b.id, sim, conflicts))
        return out

    def apply_merge(self, store: BeliefStore, a_id: str, b_id: str) -> bool:
        """Слить b в a. Убеждения складываются, конфликтующие остаются у сильнейшего.

        «Сильнейший» — тот, у кого больше собственного опыта, а не больше n:
        пересказ не должен побеждать проверку числом повторов.
        """
        a = store.entities.get(a_id)
        b = store.entities.get(b_id)
        if a is None or b is None:
            return False
        for src, dst in ((b.affordances, a.affordances), (b.dynamics, a.dynamics)):
            for key, belief in src.items():
                cur = dst.get(key)
                if cur is None:
                    dst[key] = belief
                elif belief.n_experience > cur.n_experience:
                    dst[key] = belief
        a.encounters += b.encounters
        a.uses += b.uses
        a.first_seq = min(a.first_seq, b.first_seq)
        a.last_seq = max(a.last_seq, b.last_seq)
        for other_id, kind in b.links.items():
            if other_id != a.id:
                a.links.setdefault(other_id, kind)
        for e in store.entities.values():
            if b.id in e.links:
                e.links.pop(b.id)
                if e.id != a.id:
                    e.links.setdefault(a.id, "merged")
        del store.entities[b.id]
        return True

    def run(self, journal: Journal, stamp: Stamp, *, apply_merges: bool = False,
            imagined_only: bool = False) -> tuple[Rebuilt, SleepReport]:
        """Один прогон сна поверх журнала.

        `imagined_only` — этот прогон учился только на воображаемом. Тогда сверки
        с реальностью не было, и счётчик просрочки растёт.
        """
        self.runs += 1
        # Доверие к человеческому свидетельству — настройка прогона, а не константа
        # пересборки: от неё зависит, насколько чужое слово двигает убеждение, и
        # сравнивать прогоны с разным доверием как равные нельзя.
        rebuilt = rebuild_from_journal(
            journal, trust_human=self.trust_human,
            live_min_responses=self.live_min_responses,
            silent_min_deliveries=self.silent_min_deliveries)
        store = rebuilt.beliefs
        report = SleepReport(run=self.runs, duration_s=self.duration_s,
                             entries_read=rebuilt.entries,
                             entities_before=len(store),
                             fingerprint_before=store.fingerprint())

        report.merges_proposed = self.propose_merges(store)
        if apply_merges:
            for m in report.merges_proposed:
                if not m.conflicts and self.apply_merge(store, m.a, m.b):
                    report.merges_applied.append((m.a, m.b))

        report.forgotten = store.prune(self.forget_below, keep=self.belief_cap)
        report.entities_after = len(store)
        report.hearsay_pending = len(store.hearsay())
        report.fingerprint_after = store.fingerprint()

        if not imagined_only:
            self.note_reality_check()
        runs_since = (self.runs if self.last_reality_check_run is None
                      else self.runs - self.last_reality_check_run)
        minutes = runs_since * float(self.profile.parameters["sleep_every_minutes"])
        report.minutes_since_reality_check = minutes
        report.reality_check_overdue = minutes > self.overdue_minutes

        target = self.journal or journal
        if target.mode == "a":
            target.append(EntryKind.SLEEP, stamp, Actor.NONE, ActorLayer.NONE,
                          event={"code": "consolidation", **report.as_dict()})
        return rebuilt, report

    def warning(self, report: SleepReport) -> str | None:
        """Текст предупреждения для исследователя, если сверка просрочена."""
        if not report.reality_check_overdue:
            return None
        return (f"давно не было сверки с реальностью: {report.minutes_since_reality_check:.0f} мин "
                f"при пределе {self.overdue_minutes:.0f}. Обучаясь во сне, агент "
                "выучивает свою модель, а не мир")
