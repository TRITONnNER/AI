"""Узнавание места: три средства из робототехники. SPEC-FULL, A2.1–A2.3.

22 789 узлов за час без насыщения. На отпечатке стоят граф мест и карточки, а на них
всё остальное, поэтому чинить надо здесь, и чинить **функцию узнавания**, а не порог:
порог уже двигали, и он держался только на синтетике.

Три средства, и каждое измеримо отдельно — иначе неизвестен вклад:

| Средство | Что меняет | Класс ошибки, который лечит |
|---|---|---|
| A2.1 сопоставление последовательностей | место узнаётся по отрезку пути | одиночный вид неоднозначен |
| A2.2 адаптивный порог | порог выводится из фона домена | нарисованный рукой 0.82 |
| A2.3 гипотеза нового места | совпадение проверяется, а не принимается | всякое совпадение — замыкание |

## Почему порог нельзя оставить числом

`place_same_similarity: 0.82` нарисован рукой. Число, нарисованное рукой, не знает
ничего о среде, в которой работает: в домене с однотонным фоном любые два вида похожи
на 0.9, в пёстром — 0.4, и один порог там и там означает разное. Это та же болезнь, от
которой лечился детектор режима: признак сравнивался с нулём вместо фона домена
(`MEASUREMENT.md`, раздел 35).

Фон здесь — распределение похожести между видами, **заведомо разнесёнными по времени**.
Два вида, снятые с промежутком в сотню кадров, обычно принадлежат разным местам; их
похожесть и есть «сколько дают случайные два места в этой среде». Порог берётся
квантилью этого распределения: выше него похожесть уже не объясняется случайностью.

**Фона может не быть.** Пока пар меньше объявленного минимума, адаптивный порог
отвечает «не определено», и узнавание честно берёт число из схемы — не молча, а с
пометкой, откуда порог взялся (`Threshold.source`). Домен без фона отвечает «не
определено», а не правдоподобным числом: это прямое требование раздела 35.

## Гипотеза нового места порога не содержит

Она отвечает не «да или нет», а «насколько вероятно, что этот вид пришёл из места, где
я никогда не был». Считается тем же фоном: если лучшее совпадение не лучше того, что
дают случайные два места, то совпадение ничего не значит. Доля фона, дотягивающего до
лучшего совпадения, и есть эта вероятность — величина без единого порога внутри.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .places import similarity

#: Значения по умолчанию — те же, что в схеме. Здесь нужны для вызовов без профиля.
SEQUENCE_LENGTH = 5
SEQUENCE_WEIGHT = 0.5
BACKGROUND_MIN_GAP = 60
BACKGROUND_MIN_N = 40
THRESHOLD_QUANTILE = 0.99


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Квантиль по уже отсортированному. Без numpy: список короткий, вызовов много."""
    if not sorted_values:
        raise ValueError("квантиль пустого распределения не существует")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


@dataclass(frozen=True, slots=True)
class Threshold:
    """Порог тождества места и то, откуда он взялся.

    `source` обязателен. Порог, о котором неизвестно, выведен он из среды или взят из
    схемы, — это два разных утверждения под одним числом, и различать их задним числом
    будет нечем.
    """

    value: float
    source: str            # "фон домена" | "схема"
    n: int                 # на скольких парах построен фон; 0 — фона нет
    background_p95: float | None = None

    @property
    def adaptive(self) -> bool:
        return self.source == "фон домена"

    def as_dict(self) -> dict[str, Any]:
        return {"value": round(self.value, 4), "source": self.source, "n": self.n,
                "adaptive": self.adaptive,
                "background_p95": (None if self.background_p95 is None
                                   else round(self.background_p95, 4))}


class Background:
    """Фон домена: похожесть между видами, заведомо разнесёнными по времени. A2.2.

    Пара берётся так: свежий вид против вида, отстоящего не менее чем на `min_gap`
    кадров. Такие два вида **обычно** из разных мест — не всегда, и в этом вся
    честность конструкции: фон загрязнён теми случаями, когда агент вернулся туда же.
    Загрязнение работает в безопасную сторону — оно завышает фон, а значит и порог,
    то есть делает узнавание строже, а не легковернее.

    Единица независимости — **пара видов**. Пары внутри одного прохода зависимы, и
    `n` здесь не доказательство, а условие применимости: пока пар меньше минимума,
    фона нет.
    """

    __slots__ = ("min_gap", "min_n", "quantile", "tolerance", "_history", "_pairs")

    def __init__(self, *, min_gap: int = BACKGROUND_MIN_GAP,
                 min_n: int = BACKGROUND_MIN_N,
                 quantile: float = THRESHOLD_QUANTILE,
                 tolerance: int = 0, keep: int = 4096) -> None:
        if min_gap < 1:
            raise ValueError("промежуток фона меньше кадра: пары будут соседними")
        if not 0.0 < quantile <= 1.0:
            raise ValueError("квантиль вне (0, 1]")
        self.min_gap = int(min_gap)
        self.min_n = int(min_n)
        self.quantile = float(quantile)
        self.tolerance = int(tolerance)
        # Длина буфера — ровно `min_gap`: тогда самый старый в нём отстоит на
        # `min_gap` кадров, и пара берётся на объявленном промежутке. При `min_gap + 1`
        # пара бралась на промежутке `min_gap + 1`, то есть механизм молча работал не
        # с тем расстоянием, которое объявлено настройкой.
        self._history: deque[tuple[int, tuple[int, ...]]] = deque(maxlen=self.min_gap)
        self._pairs: deque[float] = deque(maxlen=int(keep))

    @classmethod
    def from_profile(cls, profile: Any) -> "Background":
        p = profile.parameters
        return cls(min_gap=int(p["place_background_min_gap"]),
                   min_n=int(p["place_background_min_n"]),
                   quantile=float(p["place_threshold_quantile"]),
                   tolerance=int(profile.structural["place_level_tolerance"]))

    def feed(self, seq: int, fp: tuple[int, ...]) -> float | None:
        """Дать вид. Возвращает похожесть новой пары фона, если она набралась."""
        got: float | None = None
        if self._history:
            old_seq, old_fp = self._history[0]
            if seq - old_seq >= self.min_gap and len(old_fp) == len(fp):
                got = similarity(fp, old_fp, tolerance=self.tolerance)
                self._pairs.append(got)
        self._history.append((int(seq), fp))
        return got

    @property
    def n(self) -> int:
        return len(self._pairs)

    def threshold(self, fallback: float) -> Threshold:
        """Порог тождества. `fallback` — число из схемы на случай, когда фона нет."""
        if self.n < self.min_n:
            return Threshold(float(fallback), "схема", self.n)
        ordered = sorted(self._pairs)
        value = _quantile(ordered, self.quantile)
        p95 = _quantile(ordered, 0.95)
        # Порог не опускается ниже схемы: фон говорит, какая похожесть объясняется
        # случайностью, и это нижняя граница строгости, а не замена ей. Иначе в
        # однотонном домене фон уехал бы вниз и порог разрешил бы склеить всё.
        return Threshold(max(float(fallback), value), "фон домена", self.n, p95)

    def probability_at_least(self, value: float) -> float | None:
        """Доля фона, дотягивающего до этой похожести. `None` — фона нет.

        Это и есть A2.3 без порога: если случайные два места в этой среде сходятся
        не хуже, чем лучшее совпадение, — совпадение не значит ничего.
        """
        if self.n < self.min_n:
            return None
        return sum(1 for x in self._pairs if x >= value) / self.n

    def state(self) -> dict[str, Any]:
        ordered = sorted(self._pairs)
        return {"n": self.n, "min_n": self.min_n, "min_gap": self.min_gap,
                "usable": self.n >= self.min_n,
                "median": round(_quantile(ordered, 0.5), 4) if ordered else None,
                "p95": round(_quantile(ordered, 0.95), 4) if ordered else None}


class Sequences:
    """Сопоставление последовательностей: место узнаётся по отрезку пути. A2.1.

    Одиночный кадр неоднозначен — два коридора выглядят одинаково, — а отрезок пути
    почти нет: чтобы совпал и он, надо прийти тем же путём. Здесь для каждого места
    хранятся отрезки, которыми в него приходили, и свежий отрезок сравнивается с ними.

    Отрезок берётся из логарифмического буфера каскада (A3.4), а не из своего окна:
    два буфера с разной глубиной разошлись бы молча, и «тот же путь» означало бы
    разное в разных местах кода.

    Счёт по отрезку **не заменяет** счёт по одиночному виду, а смешивается с ним долей
    `weight`. Замена была бы хуже: в начале прогона отрезков нет ни у одного места, и
    узнавание перестало бы работать вовсе.
    """

    __slots__ = ("length", "weight", "tolerance", "keep", "_by_place", "_recent")

    def __init__(self, *, length: int = SEQUENCE_LENGTH,
                 weight: float = SEQUENCE_WEIGHT, tolerance: int = 0,
                 keep: int = 8) -> None:
        if length < 2:
            raise ValueError("отрезок короче двух видов — это одиночный вид")
        if not 0.0 <= weight <= 1.0:
            raise ValueError("доля отрезка вне [0, 1]")
        self.length = int(length)
        self.weight = float(weight)
        self.tolerance = int(tolerance)
        self.keep = int(keep)
        self._by_place: dict[str, list[tuple[tuple[int, ...], ...]]] = {}
        self._recent: deque[tuple[int, ...]] = deque(maxlen=self.length)

    @classmethod
    def from_profile(cls, profile: Any) -> "Sequences":
        p = profile.parameters
        return cls(length=int(profile.structural["place_sequence_length"]),
                   weight=float(p["place_sequence_weight"]),
                   tolerance=int(profile.structural["place_level_tolerance"]))

    def observe(self, fp: tuple[int, ...]) -> None:
        """Запомнить вид как часть текущего отрезка пути."""
        self._recent.append(fp)

    def remember(self, place: str) -> bool:
        """Привязать текущий отрезок к месту. `False`, если отрезок ещё не набрался."""
        if len(self._recent) < self.length:
            return False
        seqs = self._by_place.setdefault(place, [])
        current = tuple(self._recent)
        if current in seqs:
            return False
        seqs.append(current)
        if len(seqs) > self.keep:
            del seqs[0]
        return True

    def score(self, place: str) -> float | None:
        """Насколько текущий отрезок похож на отрезки этого места. `None` — сравнить не с чем.

        `None`, а не ноль: «отрезков у места нет» и «отрезки не совпали» — разные
        утверждения, и второе из первого не следует.
        """
        seqs = self._by_place.get(place)
        if not seqs or len(self._recent) < self.length:
            return None
        current = tuple(self._recent)
        best = 0.0
        for seq in seqs:
            got = [similarity(a, b, tolerance=self.tolerance)
                   for a, b in zip(current, seq) if len(a) == len(b)]
            if got:
                best = max(best, sum(got) / len(got))
        return best

    def blend(self, place: str, single: float) -> tuple[float, bool]:
        """Смешать счёт по одиночному виду со счётом по отрезку.

        Возвращает счёт и то, участвовал ли отрезок: без второго числа нельзя
        отделить вклад средства от вклада порога, а спецификация требует именно
        повкладный замер.
        """
        seq_score = self.score(place)
        if seq_score is None:
            return single, False
        return single * (1.0 - self.weight) + seq_score * self.weight, True

    def state(self) -> dict[str, Any]:
        return {"places_with_sequences": len(self._by_place),
                "sequences": sum(len(v) for v in self._by_place.values()),
                "length": self.length, "weight": self.weight}


@dataclass(frozen=True, slots=True)
class Recognition:
    """Ответ узнавания: место, счёт, вероятность нового места и откуда взят порог."""

    place: str | None
    score: float
    threshold: Threshold
    p_new: float | None
    used_sequence: bool

    @property
    def same(self) -> bool:
        """Признано тем же местом. Гипотеза нового места решает раньше порога.

        Порядок именно такой: если совпадение не лучше случайного, то оно не значит
        ничего, и сравнивать его с порогом уже бессмысленно — какой бы порог ни был.
        """
        if self.place is None:
            return False
        if self.p_new is not None and self.p_new >= 0.5:
            return False
        return self.score >= self.threshold.value

    def as_dict(self) -> dict[str, Any]:
        return {"place": self.place, "score": round(self.score, 4),
                "same": self.same, "used_sequence": self.used_sequence,
                "p_new": None if self.p_new is None else round(self.p_new, 4),
                "threshold": self.threshold.as_dict()}


class Recognizer:
    """Три средства вместе. Каждое включается отдельно — иначе неизвестен вклад.

    Выключенное средство не «работает вхолостую», а не работает: `Recognizer` с тремя
    выключенными средствами обязан давать ровно то же, что давал граф мест до этой
    задачи. Это и есть контрольный прогон, с которым сравниваются остальные семь
    сочетаний.
    """

    def __init__(self, *, fallback: float, background: Background | None = None,
                 sequences: Sequences | None = None,
                 novelty: bool = True) -> None:
        self.fallback = float(fallback)
        self.background = background
        self.sequences = sequences
        self.novelty = bool(novelty)
        self.seen = 0
        self.rejected_by_novelty = 0

    @classmethod
    def from_profile(cls, profile: Any) -> "Recognizer":
        st = profile.structural
        return cls(fallback=float(profile.parameters["place_same_similarity"]),
                   background=(Background.from_profile(profile)
                               if st["place_adaptive_threshold"] else None),
                   sequences=(Sequences.from_profile(profile)
                              if st["place_sequence_matching"] else None),
                   novelty=bool(st["place_novelty_hypothesis"]))

    def threshold(self) -> Threshold:
        if self.background is None:
            return Threshold(self.fallback, "схема", 0)
        return self.background.threshold(self.fallback)

    def look(self, fp: tuple[int, ...], seq: int,
             candidates: Iterable[tuple[str, float]]) -> Recognition:
        """Узнать место. `candidates` — пары «место, похожесть одиночного вида».

        Похожесть считает граф мест: у него отпечатки всех мест и своя сетка. Здесь
        решается, что с этими числами делать.
        """
        self.seen += 1
        if self.background is not None:
            self.background.feed(seq, fp)
        if self.sequences is not None:
            self.sequences.observe(fp)

        best_place: str | None = None
        best_score = 0.0
        used_sequence = False
        for place, single in candidates:
            score, used = (self.sequences.blend(place, single)
                           if self.sequences is not None else (single, False))
            if score > best_score:
                best_place, best_score, used_sequence = place, score, used

        p_new: float | None = None
        if self.novelty and self.background is not None and best_place is not None:
            p_new = self.background.probability_at_least(best_score)

        got = Recognition(best_place, best_score, self.threshold(), p_new, used_sequence)
        if got.place is not None and not got.same and p_new is not None and p_new >= 0.5:
            self.rejected_by_novelty += 1
        return got

    def remember(self, place: str) -> None:
        """Место опознано или заведено — привязать к нему пройденный отрезок."""
        if self.sequences is not None:
            self.sequences.remember(place)

    def state(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "rejected_by_novelty": self.rejected_by_novelty,
            "threshold": self.threshold().as_dict(),
            "background": None if self.background is None else self.background.state(),
            "sequences": None if self.sequences is None else self.sequences.state(),
            "novelty": self.novelty,
        }
