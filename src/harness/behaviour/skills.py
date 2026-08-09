"""Библиотека навыков: макросы, добытые из журнала.

Навык здесь — не написанная руками последовательность, а найденная: агент
замечает, что одна и та же цепочка действий раз за разом приводит к одному и тому
же исходу, и запоминает её как одно действие. Это контур на ~2 Гц из архитектуры:
готовые макросы между планировщиком и рефлексами.

Три условия, без которых навык был бы самообманом:

1. **Навык выводится из журнала, а не записывается отдельно.** Как убеждения:
   `mine` читает журнал и возвращает найденное. Значит навык нельзя «сохранить»
   в обход опыта, и он пересобирается с нуля.
2. **У навыка есть происхождение и статистика.** μ — доля успешных применений,
   σ, n. Навык с n = 1 — это догадка, и выглядеть надёжным он не должен.
3. **Навык проверяется применением.** `verify` прогоняет макрос в мире и
   сравнивает с ожиданием. Непроверенный навык остаётся кандидатом: применять
   найденное в журнале совпадение как умение — самый быстрый способ выучить
   случайность.

Обратимость навыка — не сумма обратимостей шагов, а отдельный вопрос. Цепочка из
обратимых шагов может быть необратима целиком (открыл дверь, вошёл, дверь
закрылась за спиной), поэтому обратимость навыка неизвестна, пока не проверена
откатом целиком.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..core.action import (Action, ActionError, Reversibility, action_key,
                          macro_key)
from ..core.clocks import Stamp
from ..core.journal import Actor, ActorLayer, Journal, Kind as EntryKind
from ..model.beliefs import Origin, Provenance


class SkillError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Step:
    """Шаг навыка: выход и удержание. Длительность — часть шага (инвариант 8)."""

    output: str
    duration_ms: int
    modifiers: tuple[str, ...] = ()

    def to_action(self, reversibility: Reversibility) -> Action:
        return Action.key(self.output, self.duration_ms, self.modifiers,
                          reversibility=reversibility)

    def key(self) -> str:
        """Ключ шага — общая для проекта форма ключа действия (`core.action`)."""
        return action_key(self.output, self.duration_ms, self.modifiers)

    def as_dict(self) -> dict[str, Any]:
        return {"output": self.output, "duration_ms": self.duration_ms,
                "modifiers": list(self.modifiers)}


@dataclass(slots=True)
class Skill:
    """Найденная последовательность с оценкой того, насколько она работает."""

    id: str
    steps: tuple[Step, ...]
    mu: float
    sigma: float
    n: int
    provenance: Provenance
    reversibility: Reversibility = Reversibility()
    verified: int = 0
    verified_ok: int = 0

    def __post_init__(self) -> None:
        if len(self.steps) < 2:
            raise SkillError(
                "навык короче двух шагов — это одиночное действие, а не макрос")
        if self.n < 1:
            raise SkillError("навык без ни одного наблюдения")

    @property
    def is_guess(self) -> bool:
        """Найдено в журнале, но ни разу не применено намеренно."""
        return self.verified == 0

    @property
    def confidence(self) -> float:
        """Насколько можно опираться. Догадка дисконтируется вдвое."""
        base = self.mu / (1.0 + self.sigma)
        return base * (0.5 if self.is_guess else 1.0)

    @property
    def duration_ms(self) -> int:
        return sum(s.duration_ms for s in self.steps)

    def signature(self) -> str:
        return "|".join(s.key() for s in self.steps)

    def observe_use(self, worked: bool) -> None:
        """Применили намеренно. Это уже не наблюдение, а проверка."""
        self.verified += 1
        self.verified_ok += int(worked)
        n = self.n + 1
        self.mu = (self.mu * self.n + (1.0 if worked else 0.0)) / n
        self.sigma = (max(self.mu * (1.0 - self.mu), 1e-9) / n) ** 0.5 if n > 1 else 0.5
        self.n = n

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "steps": [s.as_dict() for s in self.steps],
                "signature": self.signature(),
                "mu": round(self.mu, 4), "sigma": round(self.sigma, 4), "n": self.n,
                "duration_ms": self.duration_ms,
                "verified": self.verified, "verified_ok": self.verified_ok,
                "is_guess": self.is_guess,
                "confidence": round(self.confidence, 4),
                "reversibility": self.reversibility.as_dict(),
                "caution": round(self.reversibility.caution, 4),
                "provenance": self.provenance.as_dict()}


def _skill_id(signature: str) -> str:
    import hashlib
    h = hashlib.blake2b(signature.encode("utf-8"), digest_size=2).hexdigest().upper()
    return f"SKILL_{h}"


def mine(journal: Journal, *, min_length: int = 2, max_length: int = 4,
         min_repeats: int = 3, min_success: float = 0.7) -> list[Skill]:
    """Найти в журнале повторяющиеся успешные цепочки.

    Успехом считается то, что каждый шаг цепочки доехал до устройства и что-то
    изменил (`responded`). Цепочки, где хотя бы один шаг молчал, не рассматриваются:
    макрос из молчащих шагов ничего не делает, сколько бы раз он ни повторился.

    Это не обучение, а подсчёт. Обучение начнётся тогда, когда найденное будет
    проверено применением — см. `verify`.
    """
    if min_length < 2:
        raise SkillError("макрос короче двух шагов не бывает")
    if max_length < min_length:
        raise SkillError("max_length меньше min_length")

    # Собираем подряд идущие успешные действия. Разрыв — любая запись, которая
    # не является успешным действием: пауза, стоп, вмешательство, молчащий шаг.
    runs: list[list[tuple[Step, int]]] = [[]]
    branch = journal.meta.branch_id
    for e in journal:
        if e.kind is EntryKind.ACTION and e.action is not None:
            act = e.action
            good = (not act.masked and e.event.get("code") == "delivered"
                    and bool(e.event.get("responded")) and act.output is not None)
            if good:
                runs[-1].append((Step(act.output, act.duration_ms, act.modifiers), e.seq))
                continue
        if runs[-1]:
            runs.append([])

    counts: Counter[str] = Counter()
    first_seq: dict[str, int] = {}
    steps_of: dict[str, tuple[Step, ...]] = {}
    for run in runs:
        for length in range(min_length, max_length + 1):
            for i in range(0, len(run) - length + 1):
                window = run[i:i + length]
                steps = tuple(s for s, _ in window)
                sig = "|".join(s.key() for s in steps)
                counts[sig] += 1
                steps_of[sig] = steps
                first_seq.setdefault(sig, window[0][1])

    out: list[Skill] = []
    for sig, n in counts.items():
        if n < min_repeats:
            continue
        steps = steps_of[sig]
        # Более длинный макрос, встречавшийся столько же раз, интереснее
        # короткого: он объясняет больше одним действием. Короткий, целиком
        # входящий в длинный с тем же числом повторов, отбрасывается.
        redundant = any(
            sig != other and sig in other and counts[other] >= n
            for other in counts if counts[other] >= min_repeats)
        if redundant:
            continue
        mu = 1.0            # в журнал попали только успешные повторы
        sigma = (max(mu * (1.0 - mu), 1e-9) / n) ** 0.5 if n > 1 else 0.5
        if mu < min_success:
            continue
        out.append(Skill(_skill_id(sig), steps, mu, sigma, n,
                         Provenance(Origin.EXPERIENCE, branch, first_seq[sig])))
    out.sort(key=lambda s: (-s.n, -len(s.steps), s.id))
    return out


class Library:
    """Набор навыков. Пересобираем из журнала, поэтому ничего не хранит сверх него."""

    def __init__(self) -> None:
        self.skills: dict[str, Skill] = {}

    def add(self, skill: Skill) -> Skill:
        existing = self.skills.get(skill.id)
        if existing is None:
            self.skills[skill.id] = skill
            return skill
        # Тот же макрос найден снова: складываем наблюдения, а не заменяем.
        existing.n += skill.n
        existing.sigma = (max(existing.mu * (1 - existing.mu), 1e-9)
                          / existing.n) ** 0.5
        return existing

    def from_journal(self, journal: Journal, **kwargs: Any) -> list[Skill]:
        found = mine(journal, **kwargs)
        return [self.add(s) for s in found]

    def best_for(self, length: int | None = None) -> list[Skill]:
        """Навыки по надёжности. Догадки внизу — они и есть догадки."""
        pool = [s for s in self.skills.values()
                if length is None or len(s.steps) == length]
        return sorted(pool, key=lambda s: (-s.confidence, s.id))

    def guesses(self) -> list[Skill]:
        return [s for s in self.skills.values() if s.is_guess]

    def __len__(self) -> int:
        return len(self.skills)

    def stats(self) -> dict[str, Any]:
        skills = list(self.skills.values())
        return {
            "skills": len(skills),
            "guesses": sum(1 for s in skills if s.is_guess),
            "verified": sum(1 for s in skills if not s.is_guess),
            "mean_length": round(sum(len(s.steps) for s in skills) / len(skills), 2)
            if skills else 0.0,
            "mean_confidence": round(
                sum(s.confidence for s in skills) / len(skills), 4) if skills else 0.0,
            "unknown_reversibility": sum(
                1 for s in skills if not s.reversibility.is_known),
        }


def verify(skill: Skill, world: Any, *, expect: Callable[[Any, Any], bool],
           clocks: Any = None, journal: Journal | None = None,
           stamp_of: Callable[[], Stamp] | None = None) -> bool:
    """Применить навык в мире и проверить, вышло ли ожидаемое.

    `expect(before, after)` — объективная проверка по состоянию мира. Без неё
    «применили и вроде сработало» не отличается от «применили».

    Обратимость навыка здесь не выясняется: цепочка из обратимых шагов может быть
    необратима целиком, и это отдельная проба.
    """
    before = world.step(None, with_audio=False)
    for step in skill.steps:
        action = step.to_action(skill.reversibility)
        world.step(action, with_audio=False)
        if journal is not None and stamp_of is not None:
            journal.append(EntryKind.ACTION, stamp_of(), Actor.AGENT,
                ActorLayer.SKILL, action=action,
                event={"code": "delivered", "responded": True, "device": "skill",
                       "skill": skill.id})
        if clocks is not None:
            clocks.tick_self()
            clocks.set_world(clocks.t_world + 1)
    after = world.step(None, with_audio=False)
    worked = bool(expect(before, after))
    skill.observe_use(worked)
    return worked


def try_undo(skill: Skill, world: Any, undo_steps: Sequence[Step], *,
             same_as: Callable[[Any, Any], bool]) -> bool:
    """Попробовать откатить навык целиком. Обратимость навыка — свой вопрос.

    Возвращает, удалось ли. Результат идёт в оценку обратимости навыка, а не его
    шагов: шаги про себя уже всё знают.
    """
    before = world.step(None, with_audio=False)
    for step in skill.steps:
        world.step(step.to_action(skill.reversibility), with_audio=False)
    for step in undo_steps:
        world.step(step.to_action(Reversibility()), with_audio=False)
    after = world.step(None, with_audio=False)
    undone = bool(same_as(before, after))
    skill.reversibility = skill.reversibility.observe(undone)
    return undone


# ---------------------------------------------------------------------------
# Макро-переходы: цепочка как один шаг плана
# ---------------------------------------------------------------------------


class MacroRecorder:
    """Замечает, что цепочка действий отсюда приводит туда, и пишет это ребром.

    Зачем это нужно отдельно от одиночных рёбер. Планировщик считает надёжность
    цепочки произведением надёжностей звеньев: два шага по 0.9 дают 0.81, три — 0.73.
    Для *независимых* шагов это верно, но цепочка, которую агент прошёл целиком двадцать
    раз и двадцать раз попал куда надо, — это одно наблюдение с надёжностью 1.0, а не
    произведение догадок. Разница не косметическая: план из трёх звеньев по 0.9
    отбраковывается порогом `plan_min_step_p`, а тот же путь одним подтверждённым
    макро-шагом проходит.

    Второе, что даёт макро-ребро: промежуточные места могут быть склеены или
    неустойчивы, и тогда план через них рвётся, хотя цепочка как целое работает.
    Макро-ребро не зависит от того, как выглядит середина пути.

    Одиночные рёбра при этом остаются. Макро не заменяет их, а добавляет альтернативу,
    и планировщик выбирает по измеренному времени.
    """

    def __init__(self, graph: Any, *, max_length: int = 3,
                 seconds_per_seq: float = 1.0) -> None:
        if max_length < 2:
            raise SkillError(
                "макрос короче двух шагов — это одиночное действие, и ребро у него "
                "уже есть")
        self.graph = graph
        self.max_length = int(max_length)
        self.seconds_per_seq = float(seconds_per_seq)
        # Окно последних шагов: (место до шага, ключ действия, номер такта)
        self._window: list[tuple[str, str, int]] = []
        self.recorded = 0

    @classmethod
    def from_profile(cls, graph: Any, profile: Any, *,
                     seconds_per_seq: float = 1.0) -> "MacroRecorder | None":
        """Записыватель по профилю, либо `None`, если макросы выключены.

        Читателем `macro_max_length` до этого был только `cli.py`, который передавал
        число в конструктор. Печать и разовая проводка в одной команде читателем не
        считаются (инвариант 30): всякий другой вызывающий получал зашитую тройку из
        значения по умолчанию, то есть ручка в профиле обещала поведение, которым не
        управляла. Отсюда `from_profile` — единственное место, где длина макроса
        берётся, и оно на агентском пути, где механизм и живёт.

        Значение меньше двух означает «макросов нет»: макрос из одного шага — это
        одиночное действие, и ребро у него уже есть. Поэтому здесь `None`, а не
        отказ, — выключение законно, и `tools/` им пользуются.
        """
        length = int(profile.structural["macro_max_length"])
        if length < 2:
            return None
        return cls(graph, max_length=length, seconds_per_seq=seconds_per_seq)

    def note(self, src: str, key: str, dst: str, seq: int) -> int:
        """Заметить один шаг. Вернуть, сколько макро-рёбер записалось.

        Записываются все цепочки, кончающиеся этим шагом, длиной от двух до
        `max_length`. Цепочка, которая привела обратно в начало, не записывается: это
        петля, а петля из цепочки — не переход, а его отсутствие, и ставить её шагом
        плана бессмысленно.
        """
        self._window.append((src, key, seq))
        del self._window[:-self.max_length]
        written = 0
        for length in range(2, len(self._window) + 1):
            tail = self._window[-length:]
            start, start_seq = tail[0][0], tail[0][2]
            if start == dst:
                continue
            keys = [k for _src, k, _seq in tail]
            try:
                mode = macro_key(keys)
            except ActionError:
                continue                    # среди звеньев не ключ действия
            seconds = max(0.0, (seq - start_seq + 1) * self.seconds_per_seq)
            self.graph.note_macro(start, dst, mode, seconds)
            written += 1
        self.recorded += written
        return written

    def stats(self) -> dict[str, Any]:
        return {"recorded": self.recorded, "max_length": self.max_length,
                "window": len(self._window)}
