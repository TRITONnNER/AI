"""Пересборка производного состояния из журнала.

Инвариант 1: «всё остальное — убеждения, карточки, веса — из него выводимо и
может быть пересобрано с нуля». Здесь это выполняется буквально: на входе только
журнал, на выходе хранилище убеждений и карта тела. Ничего не читается из
предыдущего состояния, ничего не берётся из настенного времени, никакой
случайности.

Проверка правильности одна и она жёсткая: две пересборки одного журнала дают
одинаковый отпечаток. Если не дают — где-то протекло состояние, и это ошибка,
а не мелочь: значит журнал перестал объяснять поведение.

Что именно выводится в вехе 0–1:

- **Карта тела.** Какие выходы отвечают, какие молчат, какие необратимы. Всё из
  записей действий и того, что случилось с кадром после них.
- **Карточки символов.** Каждый символ, увиденный на экране, — сущность со
  своей динамикой (появляется, исчезает, меняется сам).
- **Свидетельства.** Записи `TESTIMONY` ложатся как чужие слова с доверием.
- **Вмешательства.** Записи `INTERVENTION` не влияют на убеждения, но считаются:
  прогон с вмешательствами нельзя сравнивать с чистым как равный.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.action import Reversibility
from ..core.journal import Journal, Kind as EntryKind
from .beliefs import BeliefStore, Origin, Provenance, Testimony, entity_id

# Ключи утверждений о выходе. Непрозрачные, как и всё, что доступно агенту.
AFFORD_RESPONDS = "responds"        # выход что-то делает
AFFORD_REVERSIBLE = "reversible"    # последствие удалось откатить


@dataclass(slots=True)
class OutputFacts:
    """Что известно про один выход. Выводится только из журнала."""

    output: str
    tries: int = 0
    delivered: int = 0
    masked: int = 0
    responded: int = 0                    # после попытки кадр изменился сверх фона
    durations_ms: list[int] = field(default_factory=list)
    reversibility: Reversibility = Reversibility()
    # Сколько раз искали способ откатить и не нашли. Это не то же, что «откат не
    # работает»: неудачная догадка говорит о догадке, а не о мире. Поэтому такие
    # попытки не портят оценку обратимости, а считаются отдельно.
    undo_searches: int = 0
    undo_method: str | None = None        # найденный способ откатить, если найден
    # Сколько проб подряд не дали никакого изменения. Выход мог отвечать раньше и
    # перестать: сломать можно только то, что ещё не сломано. Это не «он молчит»,
    # а «сейчас он ничего не даёт», и тратить на него пробы больше незачем.
    no_change_streak: int = 0
    first_seq: int = 0
    last_seq: int = 0
    # Порог повторяемости: со скольких ответов вывод «этот выход отвечает»
    # считается сделанным, и со скольких доставленных попыток — вывод «молчит».
    # Живут здесь, а не в глобальной константе, потому что карта тела пересобирается
    # из журнала и должна пересобираться одинаково при тех же порогах.
    live_min_responses: int = 2
    silent_min_deliveries: int = 3
    # Доля ложных срабатываний порога «кадр изменился» на холостом ходу. Измеряется
    # там же, где и фон, и тем же тестом: по паре кадров без действия. Ноль значит
    # «ложных срабатываний не замечено», а не «их не бывает».
    background_rate: float = 0.0
    response_sigmas: float = 2.0

    @property
    def excess_sigmas(self) -> float:
        """Насколько ответов больше, чем даёт сам порог на холостом ходу.

        Порог «кадр изменился сверх фона» не идеален: он иногда срабатывает и без
        действия — от анимации интерфейса, от того, что в видео идёт картинка. Доля
        таких срабатываний измерима (`background_rate`), значит измеримо и ожидаемое
        число ложных ответов: `delivered × background_rate`. Разброс биномиальный,
        поэтому превышение считается в сигмах этого разброса.

        Пол сигмы в единицу нужен для случая, когда ложных срабатываний не
        замечено вовсе: тогда разброс нулевой, и деление на него дало бы
        бесконечную уверенность от одного-единственного ответа.
        """
        n, p = self.delivered, self.background_rate
        if n == 0:
            return 0.0
        mu = n * p
        sd = max(1.0, (n * p * (1.0 - p)) ** 0.5)
        return (self.responded - mu) / sd

    @property
    def state(self) -> str:
        """Живой, молчащий, ещё не пробованный или пока неясный.

        Четыре состояния, и каждое отделено от остальных по делу:

        - `untried` — не пробовал. Не то же, что «не работает».
        - `live` — ответил повторяемо (`live_min_responses`) и заметно чаще, чем
          порог врёт сам по себе (`excess_sigmas`).
        - `silent` — попыток хватило, а превышения над ложными срабатываниями нет.
        - `unclear` — попыток было мало. Не «наверное молчит», а «не знаю».

        Четвёртое состояние и статистика появились не из аккуратности, а по замеру.
        Пока живым считался выход, ответивший хотя бы раз, кросс-доменный прогон
        давал по два-четыре ложно живых выхода в каждом домене: ровно один ответ на
        26–37 попыток, то есть одно совпадение. В домене, где картинка меняется сама
        (проигрыватель), таких совпадений больше всего — и это ожидаемо, потому что
        фон там выше. Один раз совпало — не то же, что «отвечает»: если это принять,
        карта тела наполняется выходами, которые ничего не делают, и дальше на них
        строятся навыки.

        Обратный перекос так же вреден: требование «ни одного ответа» для вывода
        «молчит» означает, что одно случайное совпадение навсегда оставляет выход
        неопределённым. На том же замере это давало пятнадцать вечно неясных
        выходов из двадцати четырёх. Поэтому порог «молчит» сравнивается не с
        нулём, а с тем, сколько ложных ответов ожидается.
        """
        if self.delivered == 0:
            return "untried"
        confident = self.excess_sigmas >= self.response_sigmas
        if self.responded >= self.live_min_responses and confident:
            return "live"
        if self.delivered >= self.silent_min_deliveries and not confident:
            return "silent"
        return "unclear"

    @property
    def response_rate(self) -> float:
        return self.responded / self.delivered if self.delivered else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"output": self.output, "state": self.state, "tries": self.tries,
                "delivered": self.delivered, "masked": self.masked,
                "responded": self.responded,
                "response_rate": round(self.response_rate, 4),
                "background_rate": round(self.background_rate, 4),
                "excess_sigmas": round(self.excess_sigmas, 3),
                "hold_ms": {"min": min(self.durations_ms) if self.durations_ms else None,
                            "max": max(self.durations_ms) if self.durations_ms else None},
                "reversibility": self.reversibility.as_dict(),
                "caution": round(self.reversibility.caution, 4),
                "undo_searches": self.undo_searches, "undo_method": self.undo_method,
                "no_change_streak": self.no_change_streak,
                "first_seq": self.first_seq, "last_seq": self.last_seq}


@dataclass(slots=True)
class BodyMap:
    """Карта тела: сколько выходов, какие отвечают, какие опасны."""

    outputs: dict[str, OutputFacts] = field(default_factory=dict)
    live_min_responses: int = 2
    silent_min_deliveries: int = 3
    background_rate: float = 0.0
    response_sigmas: float = 2.0

    def set_background_rate(self, rate: float) -> None:
        """Обновить измеренную долю ложных срабатываний — сразу у всех выходов.

        Величина одна на всё тело: врёт не выход, а порог. Держать её копию у
        каждого выхода нужно только для того, чтобы карточка выхода была
        самодостаточной при пересборке.
        """
        self.background_rate = float(rate)
        for f in self.outputs.values():
            f.background_rate = float(rate)

    def fact(self, output: str, seq: int) -> OutputFacts:
        f = self.outputs.get(output)
        if f is None:
            f = OutputFacts(output, first_seq=seq, last_seq=seq,
                            live_min_responses=self.live_min_responses,
                            silent_min_deliveries=self.silent_min_deliveries,
                            background_rate=self.background_rate,
                            response_sigmas=self.response_sigmas)
            self.outputs[output] = f
        else:
            f.last_seq = max(f.last_seq, seq)
        return f

    def by_state(self, state: str) -> list[str]:
        return sorted(o for o, f in self.outputs.items() if f.state == state)

    def dangerous(self, threshold: float) -> list[str]:
        """Выходы, которые я не умею откатывать. Не «запрещённые» — неоткатываемые.

        Сюда попадает и то, для чего способ отката ещё не найден (обратимость
        неизвестна, значит осторожность максимальна), и то, для чего найденный
        способ перестал работать. Различие видно в `undo_method`: он пустой в
        первом случае и заполнен во втором.
        """
        return sorted(o for o, f in self.outputs.items()
                      if f.state == "live" and f.reversibility.caution >= threshold)

    def undoable(self) -> dict[str, str]:
        """Для чего способ отката найден: выход → чем откатывается."""
        return {o: f.undo_method for o, f in sorted(self.outputs.items())
                if f.undo_method is not None}

    def stats(self) -> dict[str, Any]:
        return {"known_outputs": len(self.outputs),
                "live": len(self.by_state("live")),
                "silent": len(self.by_state("silent")),
                "unclear": len(self.by_state("unclear")),
                "untried": len(self.by_state("untried")),
                "unknown_reversibility": sum(
                    1 for f in self.outputs.values() if not f.reversibility.is_known)}

    def as_dict(self) -> dict[str, Any]:
        return {"outputs": {o: f.as_dict() for o, f in sorted(self.outputs.items())},
                **self.stats()}


@dataclass(slots=True)
class Rebuilt:
    beliefs: BeliefStore
    body: BodyMap
    interventions: int = 0
    frames: int = 0
    thoughts: int = 0
    self_reports: int = 0
    entries: int = 0
    questions_asked: int = 0
    questions_reopened: int = 0
    questions_resolved: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"beliefs": self.beliefs.stats(), "body": self.body.stats(),
                "interventions": self.interventions, "frames": self.frames,
                "thoughts": self.thoughts, "self_reports": self.self_reports,
                "entries": self.entries,
                "questions_asked": self.questions_asked,
                "questions_reopened": self.questions_reopened,
                "questions_resolved": self.questions_resolved,
                "fingerprint": self.beliefs.fingerprint()}


def rebuild_from_journal(journal: Journal, *, response_field: str = "responded",
                         trust_human: float = 0.8,
                         live_min_responses: int = 2,
                         silent_min_deliveries: int = 3) -> Rebuilt:
    """Собрать убеждения и карту тела из журнала. Только из журнала.

    `response_field` — имя поля в событии записи действия, куда пишущая сторона
    положила «изменился ли мир после этой попытки». Если поля нет, вывод по этой
    попытке не делается: догадываться о последствии по соседним записям значило бы
    придумывать данные.

    Пороги повторяемости передаются снаружи (из профиля), а не берутся глобально:
    пересборка обязана быть воспроизводимой, значит всё, что влияет на итог, должно
    быть в аргументах.
    """
    branch = journal.meta.branch_id
    store = BeliefStore(branch)
    body = BodyMap(live_min_responses=live_min_responses,
                   silent_min_deliveries=silent_min_deliveries)
    out = Rebuilt(store, body)

    for e in journal:
        out.entries += 1

        if e.kind is EntryKind.FRAME:
            out.frames += 1
            continue

        if e.kind is EntryKind.THOUGHT:
            # Воображаемое не обновляет убеждения о мире. Оно вообще не про мир:
            # это опыт думания, и путать его с опытом действия нельзя.
            out.thoughts += 1
            continue

        if e.kind is EntryKind.SELF_REPORT:
            # Самоотчёт — не наблюдение. Слова агента о себе не создают убеждений и
            # не меняют карту тела: сказать «я умею это откатывать» и тем самым
            # сделать выход откатываемым невозможно (инвариант 10). Считаем их,
            # чтобы было видно, что записи есть и что они именно пропущены.
            out.self_reports += 1
            continue

        if e.kind is EntryKind.ACTION and e.action is not None:
            _absorb_action(e, body, store, branch, response_field)
            continue

        if e.kind is EntryKind.PERCEPTION and e.perception:
            _absorb_perception(e, store, branch)
            continue

        if e.kind is EntryKind.TESTIMONY:
            claim = str(e.event.get("claim", ""))
            if claim:
                store.add_testimony(Testimony(
                    claim, source=str(e.event.get("source", "human")),
                    trust=float(e.event.get("trust", trust_human)),
                    branch=branch, seq=e.seq))
            continue

        if e.kind is EntryKind.HYPOTHESIS:
            # Жизнь вопроса восстанавливается из журнала, а не хранится сбоку. Иначе
            # перечень непонятого — единственное, что не выводимо из следа, и он
            # тихо расходится с ним (инвариант 1, показатель 9 «дрейф памяти»).
            _absorb_hypothesis(e, store, out)
            continue

        if e.kind is EntryKind.INTERVENTION:
            out.interventions += 1

    return out


def _absorb_hypothesis(e, store: BeliefStore, out: Rebuilt) -> None:
    """Одна запись о жизни гипотезы. Порядок переходов проверяет сам конвейер."""
    from .beliefs import Hypothesis, Origin, Provenance

    code = str(e.event.get("code", ""))
    claim = str(e.event.get("claim", ""))
    if not claim:
        return
    if code == "asked":
        prov = Provenance(Origin(str(e.event.get("origin", "hunch"))),
                          str(e.event.get("branch", store.branch)), e.seq,
                          str(e.event.get("source", "self")),
                          float(e.event.get("trust", 1.0)))
        store.add_hypothesis(Hypothesis.question(
            claim, float(e.event.get("prior", 0.5)), prov,
            because=str(e.event.get("because", "не сказано")),
            reopens_on=tuple(str(x) for x in e.event.get("needs", ()))))
        out.questions_asked += 1
    elif code == "reopened":
        h = store.hypotheses.get(claim)
        if h is not None:
            store.add_hypothesis(h.with_test(str(e.event.get("test", "?"))))
            out.questions_reopened += 1
    elif code == "resolved":
        h = store.hypotheses.get(claim)
        if h is not None:
            store.add_hypothesis(h.resolved(bool(e.event.get("outcome", False))))
            out.questions_resolved += 1


def _absorb_action(e, body: BodyMap, store: BeliefStore, branch: str,
                   response_field: str) -> None:
    act = e.action
    for output in act.outputs_touched():
        f = body.fact(output, e.seq)
        f.tries += 1
        if act.masked:
            f.masked += 1
            # Заглушённая попытка не говорит ничего о выходе — но говорит о том,
            # что попытка была. Ровно для этого она и лежит в журнале.
            continue
        if e.event.get("code") != "delivered":
            continue
        f.delivered += 1
        f.durations_ms.append(act.duration_ms)

        responded = e.event.get(response_field)
        if responded is None:
            continue
        f.no_change_streak = 0 if responded else f.no_change_streak + 1
        prov = Provenance(Origin.EXPERIENCE, branch, e.seq)
        store.learn_affordance(entity_id(output), AFFORD_RESPONDS, bool(responded),
                               prov, kind="output")
        if responded:
            f.responded += 1

        undone = e.event.get("undone")
        if undone is not None:
            f.reversibility = f.reversibility.observe(bool(undone))
            store.learn_affordance(entity_id(output), AFFORD_REVERSIBLE, bool(undone),
                                   prov, kind="output")


def _absorb_perception(e, store: BeliefStore, branch: str) -> None:
    symbols = e.perception.get("symbols") or []
    prov = Provenance(Origin.EXPERIENCE, branch, e.seq)
    seen = [s for s in symbols if isinstance(s, str)]
    for sym in seen:
        store.touch(entity_id(sym), "symbol", e.seq)
        store.learn_dynamics(entity_id(sym), "visible", True, prov, kind="symbol")
    # Символы, встреченные в одном кадре, связаны «виделись вместе». Это самая
    # слабая из возможных связей и единственная, которую можно вывести без
    # разметки: ничего про смысл она не утверждает.
    for i, a in enumerate(seen):
        for b in seen[i + 1:]:
            store.link(entity_id(a), entity_id(b), "co_seen")


def rebuild_twice_matches(journal: Journal) -> tuple[bool, str, str]:
    """Проверка воспроизводимости: два прогона обязаны дать один отпечаток."""
    a = rebuild_from_journal(journal).beliefs.fingerprint()
    b = rebuild_from_journal(journal).beliefs.fingerprint()
    return a == b, a, b


def drift(live: BeliefStore, journal: Journal, **kw: Any) -> dict[str, Any]:
    """Дрейф памяти (показатель 9): чем живое хранилище отличается от пересобранного.

    Расхождение и есть **ложное воспоминание, измеренное**. Живое хранилище правится по
    ходу дела: слияния, забывание, реконсолидация. Пересобранное из журнала не знает
    ничего, кроме следа. Всё, что есть в живом и чего нет в пересобранном, агент помнит
    без основания в опыте, а всё обратное — забыл.

    Возвращает доли, а не только «совпало/нет»: отпечаток отвечает да или нет, а
    величина расхождения — это то, что нужно смотреть на графике. Единица независимости
    здесь — **утверждение** хранилища, и `n` в отчёте равен размеру объединения.
    """
    fresh = rebuild_from_journal(journal, **kw).beliefs
    a = {b.claim: b for b in live.beliefs()}
    b = {x.claim: x for x in fresh.beliefs()}
    only_live = sorted(set(a) - set(b))
    only_fresh = sorted(set(b) - set(a))
    common = sorted(set(a) & set(b))
    moved = [c for c in common if abs(a[c].mu - b[c].mu) > 1e-9]
    union = len(set(a) | set(b))
    return {
        "same_fingerprint": live.fingerprint() == fresh.fingerprint(),
        "n_claims": union,
        "only_live": only_live,
        "only_rebuilt": only_fresh,
        "mu_moved": moved,
        # Доля утверждений, по которым живое и пересобранное расходятся хоть в чём-то.
        "drift": (0.0 if not union
                  else round((len(only_live) + len(only_fresh) + len(moved)) / union, 6)),
        "unit": "утверждение",
    }
