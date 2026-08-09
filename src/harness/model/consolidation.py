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
from typing import Any, Iterable

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


@dataclass(frozen=True, slots=True)
class SplitProposal:
    """Предложение расщепить карточку: под одним отпечатком две разные сущности.

    Признак — **бимодальность при насыщении**. Убеждение, у которого наблюдений много
    (`n` велик), а `mu` держится посередине и `sigma` не сужается, описывает не
    неопределённость, а две разные вещи, слитые в одну: половина проходов даёт один
    исход, половина другой, и среднее по ним не описывает ни одной.

    Отличать от честной неопределённости обязательно, и различает их `n`. При малом `n`
    середина — это «ещё не знаю», и расщеплять там нечего.
    """

    entity: str
    key: str
    mu: float
    n: int
    why: str

    def as_dict(self) -> dict[str, Any]:
        return {"entity": self.entity, "key": self.key, "mu": round(self.mu, 4),
                "n": self.n, "why": self.why}


#: Полный набор операций сна по `TASK-12`, часть 2. Перечислен списком, а не выводится
#: из отчёта: недостающая операция должна быть видна отсутствием строки в отчёте против
#: этого перечня, а не догадкой читателя о том, чего он не увидел.
#:
#: Модульная константа, а не поле `SleepReport`: внутри dataclass со slots аннотация
#: превращает её в поле экземпляра, и обращение к ней у класса даёт дескриптор вместо
#: кортежа. Первая редакция так и сделала, и тест это поймал.
SLEEP_OPERATIONS: tuple[str, ...] = (
    "слияние и расщепление", "пересчёт mu/sigma", "добыча скриптов",
    "ранжирование", "понижение",
)


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
    splits_proposed: list[SplitProposal] = field(default_factory=list)
    edges_recomputed: int = 0
    scripts_mined: int = 0
    ranked: list[tuple[str, float]] = field(default_factory=list)
    demoted: list[str] = field(default_factory=list)
    demote_refused: list[dict[str, Any]] = field(default_factory=list)
    bookmarks_expired: int = 0
    hearsay_pending: int = 0
    verified_testimony: int = 0
    deferred_open: int = 0
    drift: float | None = None
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
                "splits_proposed": [s.as_dict() for s in self.splits_proposed],
                "edges_recomputed": self.edges_recomputed,
                "scripts_mined": self.scripts_mined,
                "ranked": [[k, v] for k, v in self.ranked[:10]],
                "demoted": self.demoted,
                "demote_refused": self.demote_refused,
                "bookmarks_expired": self.bookmarks_expired,
                "hearsay_pending": self.hearsay_pending,
                "verified_testimony": self.verified_testimony,
                "deferred_open": self.deferred_open,
                "drift": self.drift,
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
        self.split_min_n = int(p["split_min_observations"])
        self.split_band = float(p["split_middle_band"])
        self.macro_max_length = int(profile.structural["macro_max_length"])
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

    def propose_splits(self, store: BeliefStore) -> list[SplitProposal]:
        """Найти карточки, под которыми, судя по наблюдениям, две разные сущности.

        Порог насыщения и полоса середины берутся из профиля, а не рисуются здесь
        (инвариант 23): от них зависит, что попадёт в предложения, и прогоны с разными
        значениями сравнивать как равные нельзя.
        """
        out: list[SplitProposal] = []
        for e in sorted(store.entities.values(), key=lambda x: x.id):
            for key, b in sorted((*e.affordances.items(), *e.dynamics.items())):
                if b.n < self.split_min_n:
                    continue
                if abs(b.mu - 0.5) > self.split_band:
                    continue
                out.append(SplitProposal(
                    e.id, key, b.mu, b.n,
                    f"наблюдений {b.n}, mu держится на {b.mu:.2f} при полосе "
                    f"{self.split_band:.2f}: это не неопределённость, а две разные "
                    "сущности под одним отпечатком"))
        return out

    def recompute_edges(self, graph: Any) -> int:
        """Пересчитать `mu`/`sigma` по рёбрам с новыми наблюдениями.

        Возвращает число пересчитанных рёбер. Ноль — законный результат и означает
        «новых наблюдений не было»; отличать его от «пересчёт не вызывался» нужно по
        тому, что операция перечислена в `SleepReport.OPERATIONS` и её число печатается
        всегда.
        """
        if graph is None:
            return 0
        fn = getattr(graph, "recompute", None)
        if callable(fn):
            return int(fn() or 0)
        # У графа нет пересчёта — это не «нечего делать», а отсутствие операции, и
        # молчать о нём нельзя: сон обязан сказать, что пересчёт не состоялся.
        raise AttributeError(
            "у графа мест нет recompute(): операция сна «пересчёт mu/sigma» не "
            "выполнена. Молчаливый ноль здесь неотличим от «новых наблюдений не было»")

    def mine_scripts(self, journal: Journal) -> int:
        """Добыть повторяющиеся последовательности. Возвращает число найденных."""
        from ..behaviour.skills import mine

        found = mine(journal, min_length=2, max_length=self.macro_max_length or 2)
        return len(found)

    def run(self, journal: Journal, stamp: Stamp, *, apply_merges: bool = False,
            imagined_only: bool = False, graph: Any | None = None,
            marks: Any | None = None, live: BeliefStore | None = None,
            segments: Iterable[str] = ()) -> tuple[Rebuilt, SleepReport]:
        """Один прогон сна поверх журнала.

        `imagined_only` — этот прогон учился только на воображаемом. Тогда сверки
        с реальностью не было, и счётчик просрочки растёт.

        `graph`, `marks`, `segments` — то, над чем работают операции, появившиеся в
        TASK-12: пересчёт рёбер, закладки с карантином и понижение. Все три
        необязательны, и **их отсутствие докладывается числом**, а не подразумевается:
        сон без графа честно печатает «рёбер пересчитано 0», а не делает вид, что
        пересчёт был.
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
        report.splits_proposed = self.propose_splits(store)
        if apply_merges:
            for m in report.merges_proposed:
                if not m.conflicts and self.apply_merge(store, m.a, m.b):
                    report.merges_applied.append((m.a, m.b))

        report.edges_recomputed = self.recompute_edges(graph) if graph is not None else 0
        report.scripts_mined = self.mine_scripts(journal)
        report.ranked = store.rank()

        # Понижение — последняя операция и единственная, которая что-то теряет. Идёт
        # после ранжирования, потому что решает по рангу, и **через** закладки, потому
        # что иначе обходит и их, и карантин.
        if marks is not None:
            report.bookmarks_expired = len(marks.wake())
            from .bookmarks import demotable

            by_rank = {k: v for k, v in report.ranked}
            order = sorted(segments, key=lambda s: by_rank.get(s, 0.0))
            allowed, refused = demotable(order, marks)
            report.demoted = allowed
            report.demote_refused = [r.as_dict() for r in refused]

        report.forgotten = store.prune(self.forget_below, keep=self.belief_cap)
        report.entities_after = len(store)
        report.hearsay_pending = len(store.hearsay())
        report.verified_testimony = sum(1 for b in store.beliefs()
                                        if b.verified_testimony)
        report.deferred_open = len(store.deferred())
        report.fingerprint_after = store.fingerprint()

        # Дрейф памяти (показатель 9) считается ровно там, где есть с чем сравнивать:
        # живое хранилище против только что пересобранного. Без живого — `None`, а не
        # ноль: ноль означал бы «расхождений нет», то есть утверждение о том, чего не
        # проверяли.
        if live is not None:
            from .rebuild import drift

            report.drift = float(drift(
                live, journal, trust_human=self.trust_human,
                live_min_responses=self.live_min_responses,
                silent_min_deliveries=self.silent_min_deliveries)["drift"])

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
