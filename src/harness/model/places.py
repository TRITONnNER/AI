"""Граф мест.

Из архитектуры: «пространство — граф мест. Узел: отпечаток вида. Ребро:
`(mode, mu_seconds, sigma, n, label)`. Никакой метрики глобально; локально в
ближнем слое — относительная геометрия».

Поэтому здесь нет ни одной координаты. Узел не знает, где он находится, — он
знает только, как выглядит. Ребро не знает расстояния — оно знает, сколько
секунд ходьбы туда уходило и с каким разбросом. «Направление» тоже отсутствует:
его агент выведет сам из того, каким действием он проходил ребро.

Отпечаток вида: кадр сжимается до сетки, яркость приводится к среднему по кадру
и квантуется. Приведение к среднему нужно, чтобы место оставалось тем же местом
при выключенном свете: темнее — не значит другое место. Сравнение — по доле
совпавших ячеек, то есть по расстоянию Хэмминга, а не по пикселям: место
узнаётся с другого угла, а попиксельно оно тогда совсем другое.

Экранный слой из отпечатка исключается, если маска известна: интерфейс одинаков
всюду и, попав в отпечаток, склеил бы все места в одно.

## Приведение к среднему обязано быть отменяемым

Приведение к среднему — это допущение о мире: «яркость не важна». Допущение
полезное, но не всегда верное, и хардкодить его в восприятие нельзя ни в ту, ни в
другую сторону. Замер показал ровно это: на сиде 5 переключатель света давал одно и
то же место при разном состоянии мира, и подтверждённый переход с двенадцати
наблюдений врал при исполнении плана — из 38 попыток доходили 2. На сиде 3, где
переключатель света в разведку не попал, доходили все 40 попыток. Одна причина, два
исхода.

Единица здесь — **попытка**, и как доля эти числа недействительны: попытки на одном
ребре зависимы, потому что приход инкрементирует силу ребра, а отказ — нет (см.
`MEASUREMENT.md`, раздел 5). Что они показывают законно — это контраст между двумя
сидами, то есть наличие самой поломки, а не её величину.

Поэтому здесь не «убрать нормировку» и не «оставить как есть», а третье: место
**делится само**, когда его собственные предсказания расходятся. Отброшенные при
нормировке величины — средний уровень и контраст — запоминаются на ребре; если у
пары «место, действие» два исхода, и уровни, при которых наблюдался один исход, не
пересекаются с уровнями другого, значит место склеило два разных состояния мира, и
порог по уровню известен. Это та же единая валюта: ошибка предсказания обновляет
карту, а не только внимание.

Статистика склеенного узла при делении **выбрасывается**, а не делится пополам:
она была собрана про узел, которого нет. Разделить её значило бы придумать данные.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

# Значения по умолчанию — те же, что в схеме настроек (`place_grid`, `place_levels`).
# Держать их здесь нужно только для вызовов без профиля: в рабочем пути параметры
# приходят из `Profile`, потому что это настройки поведения, а не свойства кода.
GRID = 8                  # сторона сетки отпечатка
LEVELS = 4                # на сколько уровней квантуется ячейка
LEVELS_KEPT = 32          # сколько последних уровней вида помнит ребро


@dataclass(frozen=True, slots=True)
class View:
    """Вид: отпечаток плюс то, что нормировка из него убрала.

    `level` и `contrast` — средняя яркость и разброс по ячейкам до приведения.
    Они не участвуют в узнавании места: место обязано узнаваться при выключенном
    свете. Но они запоминаются, потому что «не участвует в узнавании» и «не имеет
    значения» — разные утверждения, и второе решается замером, а не заранее.
    """

    cells: tuple[int, ...]
    level: float
    contrast: float


def view(frame: np.ndarray, *, exclude: np.ndarray | None = None,
         grid: int = GRID, levels: int = LEVELS) -> View:
    """Снять вид: отпечаток и отброшенные нормировкой величины."""
    cells, level, contrast = _cells(frame, exclude=exclude, grid=grid, levels=levels)
    return View(cells, level, contrast)


def fingerprint(frame: np.ndarray, *, exclude: np.ndarray | None = None,
                grid: int = GRID, levels: int = LEVELS) -> tuple[int, ...]:
    """Отпечаток вида: кортеж уровней по сетке.

    `exclude` — маска пикселей, которые в отпечаток не входят (экранный слой).
    Ячейка, полностью попавшая в маску, получает уровень −1: «не знаю». Это
    отдельное значение, а не ноль, иначе «тут интерфейс» стало бы «тут темно».
    """
    return _cells(frame, exclude=exclude, grid=grid, levels=levels)[0]


def _cells(frame: np.ndarray, *, exclude: np.ndarray | None,
           grid: int, levels: int) -> tuple[tuple[int, ...], float, float]:
    gray = frame if frame.ndim == 2 else frame[:, :, :3].mean(axis=2)
    f = gray.astype(np.float64)
    if exclude is not None:
        if exclude.shape != f.shape:
            raise ValueError(f"маска {exclude.shape} не по кадру {f.shape}")
        keep = ~exclude
    else:
        keep = np.ones(f.shape, dtype=bool)

    h, w = f.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    cells: list[float] = []
    valid: list[bool] = []
    for i in range(grid):
        for j in range(grid):
            block = f[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            mask = keep[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if mask.sum() == 0:
                cells.append(0.0)
                valid.append(False)
            else:
                cells.append(float(block[mask].mean()))
                valid.append(True)

    arr = np.asarray(cells)
    ok = np.asarray(valid)
    if ok.any():
        mean = float(arr[ok].mean())
        spread = float(arr[ok].std()) or 1.0
        # Приведение к среднему и разбросу: место остаётся собой при смене
        # освещения, но перестаёт быть собой при смене вида.
        z = (arr - mean) / spread
    else:
        mean, spread = 0.0, 1.0
        z = arr
    q = np.clip(((z + 2.0) / 4.0 * levels).astype(int), 0, levels - 1)
    return (tuple(int(v) if ok[k] else -1 for k, v in enumerate(q)), mean, spread)


def similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Доля совпавших ячеек. Ячейки «не знаю» не участвуют ни за, ни против."""
    if len(a) != len(b):
        raise ValueError("отпечатки разной длины: сетки не совпадают")
    pairs = [(x, y) for x, y in zip(a, b) if x >= 0 and y >= 0]
    if not pairs:
        return 0.0
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def place_id(fp: tuple[int, ...], band: int = 0) -> str:
    """Идентификатор места. `band` — номер полосы после деления места.

    Полоса входит в хеш, а не приписывается к имени суффиксом: идентификатор
    обязан остаться непрозрачным (инвариант 4). `PLACE_1A2B#hi` рассказал бы
    агенту, что это «то же место, но ярче», — а он должен выяснить это сам, по
    тому, что оттуда ведут другие переходы.
    """
    payload = bytes((v + 1) & 0xFF for v in fp) + bytes((band & 0xFF,))
    h = hashlib.blake2b(payload, digest_size=2).hexdigest().upper()
    return f"PLACE_{h}"


@dataclass(slots=True)
class Place:
    """Узел: отпечаток вида и когда его видели. Никаких координат."""

    id: str
    fingerprint: tuple[int, ...]
    first_seq: int
    last_seq: int
    visits: int = 1
    # Несколько отпечатков на одно место: то же место с другого угла выглядит
    # иначе, и держать один «правильный» вид было бы неверно.
    variants: list[tuple[int, ...]] = field(default_factory=list)
    # Полоса и основа — только если место получилось делением. `base` нужен, чтобы
    # узнавание шло по общему отпечатку, а полоса выбиралась после.
    band: int = 0
    base: str | None = None

    def best_similarity(self, fp: tuple[int, ...]) -> float:
        return max(similarity(fp, self.fingerprint),
                   *(similarity(fp, v) for v in self.variants)) if self.variants \
            else similarity(fp, self.fingerprint)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "visits": self.visits, "first_seq": self.first_seq,
                "last_seq": self.last_seq, "variants": len(self.variants) + 1,
                "band": self.band, "base": self.base}


@dataclass(slots=True)
class Traversal:
    """Ребро: как проходили и сколько это заняло. `(mode, mu, sigma, n, label)`."""

    src: str
    dst: str
    mode: str                  # каким действием прошли; для агента непрозрачно
    mu_seconds: float = 0.0
    sigma: float = 0.0
    n: int = 0
    label: str = ""
    # Отброшенные нормировкой величины вида, из которого уходили. Хранится
    # ограниченный хвост: для проверки «разделяются ли исходы по уровню» нужны
    # границы, а не вся история, и расти без предела здесь нечему.
    src_levels: list[float] = field(default_factory=list)
    src_contrasts: list[float] = field(default_factory=list)
    # Сам отпечаток вида, из которого уходили. Нужен, когда исходы расходятся не по
    # яркости: тогда различать состояния приходится ячейкой сетки.
    src_cells: list[tuple[int, ...]] = field(default_factory=list)

    def note_source(self, level: float | None, contrast: float | None,
                    cells: tuple[int, ...] | None = None) -> None:
        if level is not None:
            self.src_levels.append(float(level))
            del self.src_levels[:-LEVELS_KEPT]
        if contrast is not None:
            self.src_contrasts.append(float(contrast))
            del self.src_contrasts[:-LEVELS_KEPT]
        if cells is not None:
            self.src_cells.append(tuple(cells))
            del self.src_cells[:-LEVELS_KEPT]

    def observe(self, seconds: float) -> None:
        self.n += 1
        if self.n == 1:
            self.mu_seconds, self.sigma = seconds, 0.0
            return
        prev = self.mu_seconds
        self.mu_seconds += (seconds - prev) / self.n
        # Онлайновая дисперсия по Уэлфорду: без хранения всех проходов.
        self.sigma = (((self.n - 2) * self.sigma ** 2
                       + (seconds - prev) * (seconds - self.mu_seconds))
                      / max(1, self.n - 1)) ** 0.5

    @property
    def confidence(self) -> float:
        """Толщина ребра в пульте: чем больше проходов и меньше разброс, тем выше."""
        if self.n == 0:
            return 0.0
        return min(1.0, self.n / 10.0) * (1.0 / (1.0 + self.sigma))

    def as_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "mode": self.mode,
                "mu_seconds": round(self.mu_seconds, 3), "sigma": round(self.sigma, 3),
                "n": self.n, "label": self.label,
                "confidence": round(self.confidence, 4)}


class PlaceGraph:
    """Граф мест. Строится по записи офлайн и живёт в среднем слое (2 Гц)."""

    def __init__(self, *, same_place_similarity: float = 0.82,
                 variant_similarity: float = 0.6,
                 refine_margin: float = 6.0, refine_min_n: int = 2,
                 refine_cell_min_n: int = 4, refine_max_tests: int = 3,
                 record_loops: bool = True,
                 grid: int = GRID, levels: int = LEVELS) -> None:
        if not 0.0 < variant_similarity < same_place_similarity <= 1.0:
            raise ValueError("порог варианта должен быть ниже порога того же места")
        self.grid = int(grid)
        self.levels = int(levels)
        self.same = same_place_similarity
        self.variant = variant_similarity
        self.places: dict[str, Place] = {}
        self.edges: dict[tuple[str, str, str], Traversal] = {}
        self.current: str | None = None
        self._since_seq: int = 0
        self._last_seq: int = 0
        self._lost = 0
        # Деления: основа → (признак, порог). Признак — «level» или «contrast», то
        # есть ровно то, что нормировка выбросила.
        self.splits: dict[str, list[tuple[str, float]]] = {}
        self._refinements = 0
        self._dropped_edges = 0
        self._level: float | None = None
        self._contrast: float | None = None
        self._cells: tuple[int, ...] | None = None
        self.refine_margin = float(refine_margin)
        self.refine_min_n = int(refine_min_n)
        self.refine_cell_min_n = int(refine_cell_min_n)
        self.refine_max_tests = int(refine_max_tests)
        self.record_loops = bool(record_loops)

    @classmethod
    def from_profile(cls, profile: Any) -> "PlaceGraph":
        """Граф с параметрами из профиля. Рабочий путь — этот, а не конструктор."""
        p = profile.parameters
        return cls(same_place_similarity=float(p["place_same_similarity"]),
                   variant_similarity=float(p["place_variant_similarity"]),
                   refine_margin=float(p["place_refine_margin"]),
                   refine_min_n=int(p["place_refine_min_n"]),
                   refine_cell_min_n=int(p["place_refine_cell_min_n"]),
                   refine_max_tests=int(p["place_refine_max_tests"]),
                   record_loops=bool(profile.structural["place_record_loops"]),
                   grid=int(profile.structural["place_grid"]),
                   levels=int(profile.structural["place_levels"]))

    def see(self, frame: Any, seq: int, *, seconds_per_seq: float = 1.0,
            mode: str = "unknown", exclude: Any = None) -> str:
        """Отпечаток снять и сразу учесть — с сеткой этого графа, а не глобальной.

        Нужно затем, чтобы сетка не разъезжалась: отпечаток, снятый с одной сеткой,
        и граф, построенный на другой, дают места, которых нет.
        """
        v = view(frame, exclude=exclude, grid=self.grid, levels=self.levels)
        return self.observe(v.cells, seq, seconds_per_seq=seconds_per_seq, mode=mode,
                            level=v.level, contrast=v.contrast)

    # --- узлы ---------------------------------------------------------------

    def recognize(self, fp: tuple[int, ...]) -> tuple[str | None, float]:
        """Найти самое похожее место. `None`, если ничего похожего нет.

        Возвращается основа, а не полоса: полосы получились делением и имеют тот же
        отпечаток, поэтому выбирать между ними по похожести бессмысленно — их
        различает не вид, а то, что нормировка выбросила.
        """
        best, score = None, 0.0
        for p in self.places.values():
            s = p.best_similarity(fp)
            if s > score:
                best, score = (p.base or p.id), s
        return best, score

    def _feature_value(self, feature: str, fp: tuple[int, ...],
                       level: float | None, contrast: float | None) -> float | None:
        if feature == "level":
            return level
        if feature == "contrast":
            return contrast
        k = int(feature.split(":")[1])
        return float(fp[k]) if k < len(fp) else None

    def _band_of(self, base: str, fp: tuple[int, ...], level: float | None,
                 contrast: float | None) -> int:
        """В какой полосе разделённого места мы находимся. Ноль — делений нет.

        Признаков у места может быть несколько, и полоса — это их комбинация,
        разряд на признак. Один порог на место был первой версией, и он упирался в
        измеримое: состояние мира, различимое двумя признаками сразу, делилось
        только по первому, а второе расхождение оставалось неразделённым навсегда.
        """
        tests = self.splits.get(base)
        if not tests:
            return 0
        band = 0
        for i, (feature, threshold) in enumerate(tests):
            value = self._feature_value(feature, fp, level, contrast)
            if value is None:
                # Величина неизвестна — разряд не ставится. Ставить ноль значило бы
                # утверждать «здесь темно», не измерив.
                continue
            if value >= threshold:
                band |= 1 << i
        return band

    def _in_band(self, base: str, band: int, fp: tuple[int, ...], seq: int) -> Place:
        """Место-полоса. Ключ — отпечаток **основы**, а не пришедший вид.

        Разница не косметическая. Если ключевать полосу пришедшим отпечатком, то
        каждый ракурс того же места — а их у основы до шестнадцати вариантов —
        завёл бы свою полосу. Замер на синтетике: два вида, отличающиеся одной
        ячейкой из 64 (похожесть 0.98, заведомо одно место), после деления по
        яркости расходились в две разные полосы, и расхождение предсказаний между
        ними уже не было видно — второе деление не срабатывало никогда.
        """
        if band == 0:
            return self.places[base]
        anchor = self.places[base]
        pid = place_id(anchor.fingerprint, band)
        place = self.places.get(pid)
        if place is None:
            place = Place(pid, anchor.fingerprint, first_seq=seq, last_seq=seq,
                          band=band, base=base)
            self.places[pid] = place
        elif fp != place.fingerprint and len(place.variants) < 16 \
                and similarity(fp, place.fingerprint) >= self.variant:
            place.variants.append(fp)
        return place

    def observe(self, fp: tuple[int, ...], seq: int, *, seconds_per_seq: float = 1.0,
                mode: str = "unknown", level: float | None = None,
                contrast: float | None = None) -> str:
        """Увидеть вид. Возвращает место, в котором мы теперь.

        Три исхода: то же место, известное другое место (тогда появляется или
        уточняется ребро), совсем новое место.
        """
        # Уровень вида, из которого уходим, — это уровень *предыдущего* наблюдения.
        # Ребро описывает переход, значит помнить надо исходную сторону, а не ту, в
        # которую пришли: делить придётся место, откуда действие повело по-разному.
        src_level, src_contrast, src_cells = self._level, self._contrast, self._cells
        best, score = self.recognize(fp)

        if best is not None and score >= self.same:
            base = self.places[best]
            base.visits += 1
            base.last_seq = max(base.last_seq, seq)
            if score < 1.0 and score >= self.variant and len(base.variants) < 16:
                base.variants.append(fp)
            place = self._in_band(best, self._band_of(best, fp, level, contrast),
                                  fp, seq)
            if place is not base:
                place.visits += 1
                place.last_seq = max(place.last_seq, seq)
        else:
            pid = place_id(fp)
            if pid in self.places:
                # Коллизия отпечатков: разные виды дали один идентификатор.
                # Не сливаем молча — добавляем как вариант и отмечаем.
                self.places[pid].variants.append(fp)
                place = self.places[pid]
            else:
                place = Place(pid, fp, first_seq=seq, last_seq=seq)
                self.places[pid] = place

        if self.current is not None:
            if self.current != place.id:
                # Уход: время в пути — от последней смены места до сейчас.
                seconds = max(0.0, (seq - self._since_seq) * seconds_per_seq)
                self._observe_edge(self.current, place.id, mode, seconds,
                                   src_level, src_contrast, src_cells)
                self._since_seq = seq
            else:
                # Остался здесь — и это тоже факт о мире, причём самый частый.
                #
                # Раньше он выбрасывался, и это выяснилось замером, а не чтением
                # кода: на сиде 7 разведка сделала 1500 шагов, открыла одно место и
                # нажала один и тот же выход 1500 раз. Причина оказалась ровно
                # здесь. Модель перехода собирается из рёбер графа; если «нажал и
                # остался» ребром не становится, то про действие, которое никуда не
                # ведёт, модель не знает ничего — не «знает, что оно бесполезно», а
                # не знает вообще. Дальше `choose_probe` видит место без известных
                # действий, идёт в ветку «ничего не известно» и берёт наименее
                # использованный выход по модели — а модель пуста, значит выход
                # всегда один и тот же. Разведка запирается на одном нажатии.
                #
                # Время у петли — своё: длительность этого шага, а не накопленное
                # с прихода. Накопленное здесь означало бы «сколько я тут сижу», а
                # ребро отвечает на другой вопрос — «сколько занимает этот переход».
                if self.record_loops:
                    seconds = max(0.0, (seq - self._last_seq) * seconds_per_seq)
                    self._observe_edge(self.current, place.id, mode, seconds,
                                       src_level, src_contrast, src_cells)

        self.current = place.id
        self._last_seq = seq
        self._level, self._contrast, self._cells = level, contrast, fp
        return place.id

    def _observe_edge(self, src: str, dst: str, mode: str, seconds: float,
                      src_level: float | None = None,
                      src_contrast: float | None = None,
                      src_cells: tuple[int, ...] | None = None) -> None:
        key = (src, dst, mode)
        edge = self.edges.get(key)
        if edge is None:
            edge = Traversal(src, dst, mode)
            self.edges[key] = edge
        edge.observe(seconds)
        edge.note_source(src_level, src_contrast, src_cells)

    # --- уточнение карты по расхождению предсказаний -------------------------

    def refine(self) -> list[dict[str, Any]]:
        """Разделить места, чьи собственные переходы расходятся. Вернуть отчёт.

        Признак склейки простой и проверяемый: у пары «место, действие» два исхода,
        каждый наблюдён не меньше `refine_min_n` раз, и уровни вида, при которых
        наблюдался один исход, не пересекаются с уровнями другого с зазором
        `refine_margin`. Тогда место склеило два состояния мира, и порог известен —
        середина между границами.

        Чего здесь намеренно нет:

        - **Деления по одному наблюдению.** Один расходящийся проход — это шум
          узнавания, а не открытие. Отсюда `refine_min_n`.
        - **Деления «на всякий случай».** Если уровни пересекаются, дело не в
          яркости, и выдумывать порог посередине нельзя: получилось бы место,
          разделённое ни по чему.
        - **Сохранения статистики склеенного узла.** Рёбра делимого места
          выбрасываются целиком. Они собраны про узел, которого больше нет;
          разделить их пополам значило бы придумать данные. Разведка соберёт заново
          — это дорого и это честно.
        - **Бесконечного дробления.** Признаков у места не больше
          `refine_max_tests`: каждый удваивает число возможных узлов на один вид, и
          без предела место превращается в таблицу по пикселям, а граф — в набор
          одноразовых записей, по которому нельзя планировать.
        """
        report: list[dict[str, Any]] = []
        by_pair: dict[tuple[str, str], list[Traversal]] = {}
        for edge in self.edges.values():
            by_pair.setdefault((edge.src, edge.mode), []).append(edge)

        touched: set[str] = set()
        for (src, mode), edges in sorted(by_pair.items()):
            if len(edges) < 2:
                continue
            place = self.places.get(src)
            base = (place.base or src) if place is not None else src
            if base in touched:
                # Рёбра этой семьи только что выброшены: судить по ним больше нельзя,
                # они описывают узел, которого нет. Следующее деление — в следующий
                # вызов, по новым наблюдениям.
                continue
            tests = self.splits.get(base, [])
            if len(tests) >= self.refine_max_tests:
                continue
            strong = [e for e in edges if e.n >= self.refine_min_n]
            if len(strong) < 2:
                continue
            strong.sort(key=lambda e: -e.n)
            found = self._separating_threshold(strong[0], strong[1], existing=tests)
            if found is None:
                continue
            feature, threshold, gap = found
            self.splits.setdefault(base, []).append((feature, threshold))
            self._refinements += 1
            dropped = self._drop_edges_of(base)
            touched.add(base)
            report.append({"place": src, "base": base, "mode": mode,
                           "feature": feature, "test": len(self.splits[base]),
                           "threshold": round(threshold, 2), "gap": round(gap, 2),
                           "outcomes": [e.dst for e in strong[:2]],
                           "edges_dropped": dropped})
        return report

    def _separating_threshold(self, a: Traversal, b: Traversal, *,
                              existing: Sequence[tuple[str, float]] = ()
                              ) -> tuple[str, float, float] | None:
        """Порог, разделяющий два исхода. `None` — разделить нечем.

        `existing` — признаки, по которым это место уже поделено. Повторить один из
        них нельзя: полоса по нему уже разделена, значит новый разряд не изменит
        ничего, а рёбра будут выброшены зря. Дважды делить по одному и тому же —
        самый дешёвый способ зациклить уточнение карты.

        Сначала пробуются величины, выброшенные нормировкой: уровень и контраст. Они
        дешёвые и осмысленные — «здесь было светлее». Если не разделяют, пробуется
        одна ячейка сетки: бывает, что состояния различаются не яркостью целиком, а
        одним углом кадра — исчезнувшей панелью, открытым окном.

        Порог по ячейке требует больше наблюдений (`refine_cell_min_n`), и это не
        осторожность ради осторожности. Ячеек шестьдесят четыре, уровней четыре; при
        двух наблюдениях в группе какая-нибудь ячейка разделит их почти всегда, и
        деление окажется по шуму. Замер прямо это показал: с порогом в два
        наблюдения деления пошли на сидах, где расхождений не было вовсе.
        """
        used = {f for f, _ in existing}
        for feature, xs, ys in (("level", a.src_levels, b.src_levels),
                                ("contrast", a.src_contrasts, b.src_contrasts)):
            if feature in used:
                continue
            if len(xs) < self.refine_min_n or len(ys) < self.refine_min_n:
                continue
            lo, hi = (xs, ys) if max(xs) < max(ys) else (ys, xs)
            gap = min(hi) - max(lo)
            if gap >= self.refine_margin:
                return feature, max(lo) + gap / 2.0, gap

        if (self.refine_cell_min_n and len(a.src_cells) >= self.refine_cell_min_n
                and len(b.src_cells) >= self.refine_cell_min_n):
            width = min(len(c) for c in (*a.src_cells, *b.src_cells))
            for k in range(width):
                if f"cell:{k}" in used:
                    continue
                xs = [c[k] for c in a.src_cells]
                ys = [c[k] for c in b.src_cells]
                if -1 in xs or -1 in ys:
                    continue          # «не знаю» не разделяет: там был интерфейс
                lo_c, hi_c = (xs, ys) if max(xs) < max(ys) else (ys, xs)
                gap = float(min(hi_c) - max(lo_c))
                if gap >= 1.0:
                    return f"cell:{k}", max(lo_c) + gap / 2.0, gap
        return None

    def _drop_edges_of(self, place: str) -> int:
        """Выбросить рёбра делимого места — вместе с ведущими в него."""
        family = {place} | {p.id for p in self.places.values() if p.base == place}
        keys = [k for k, e in self.edges.items()
                if e.src in family or e.dst in family]
        for k in keys:
            del self.edges[k]
        self._dropped_edges += len(keys)
        return len(keys)

    def note_macro(self, src: str, dst: str, mode: str, seconds: float) -> None:
        """Записать переход, пройденный цепочкой действий, отдельным ребром.

        Зачем отдельным, если каждое нажатие цепочки уже дало своё ребро. Потому что
        план из навыка — это один шаг, и статистика у него своя: «эта цепочка отсюда
        приводит туда за столько секунд». По одиночным рёбрам такого не выводится —
        промежуточные места цепочки могут быть склеены, могут быть неустойчивы, и
        произведение долей по звеньям систематически занижает надёжность цепочки,
        которая как целое работает.

        Одиночные рёбра при этом остаются: макро-ребро их не заменяет, а добавляет
        альтернативу. Планировщик выберет то, что дешевле по измеренному времени.
        """
        self._observe_edge(src, dst, mode, seconds, self._level, self._contrast,
                           self._cells)

    def lost(self) -> None:
        """Потеря ориентации: вид ни на что не похож.

        Это не ошибка и не повод что-то придумывать. Считается отдельно, потому
        что частая потеря ориентации означает, что порог узнавания подобран не
        так, — и это надо видеть, а не сглаживать.
        """
        self._lost += 1
        self.current = None

    # --- выборки ------------------------------------------------------------

    def neighbours(self, place: str) -> list[Traversal]:
        """Рёбра отсюда, включая петли: «нажал и остался» — тоже известный исход."""
        return [e for e in self.edges.values() if e.src == place]

    def leaving(self, place: str) -> list[Traversal]:
        """Только рёбра, которые куда-то ведут. Для маршрута петля бесполезна."""
        return [e for e in self.edges.values() if e.src == place and e.dst != place]

    def loops(self) -> list[Traversal]:
        return [e for e in self.edges.values() if e.src == e.dst]

    def __len__(self) -> int:
        return len(self.places)

    def edges_from_to(self, src: str, dst: str) -> list[Traversal]:
        return [e for k, e in self.edges.items() if k[0] == src and k[1] == dst]

    def route(self, src: str, dst: str) -> list[Traversal] | None:
        """Путь с наименьшим ожидаемым временем. Дейкстра по `mu_seconds`.

        Никакой геометрии: только времена проходов, которые агент сам измерил.
        Ребро, пройденное один раз, считается по своему `mu`, но его `confidence`
        низкая — выбор маршрута по ненадёжным рёбрам должен быть виден.
        """
        if src not in self.places or dst not in self.places:
            return None
        import heapq

        best: dict[str, float] = {src: 0.0}
        prev: dict[str, Traversal] = {}
        heap: list[tuple[float, str]] = [(0.0, src)]
        while heap:
            cost, node = heapq.heappop(heap)
            if node == dst:
                break
            if cost > best.get(node, float("inf")):
                continue
            for edge in self.leaving(node):
                nxt = cost + max(0.001, edge.mu_seconds)
                if nxt < best.get(edge.dst, float("inf")):
                    best[edge.dst] = nxt
                    prev[edge.dst] = edge
                    heapq.heappush(heap, (nxt, edge.dst))
        if dst not in prev and dst != src:
            return None
        path: list[Traversal] = []
        node = dst
        while node != src:
            edge = prev[node]
            path.append(edge)
            node = edge.src
        return list(reversed(path))

    def stats(self) -> dict[str, Any]:
        edges = list(self.edges.values())
        loops = self.loops()
        # Петли считаются отдельно: смешать их с переходами значило бы сказать, что
        # граф связнее, чем он есть. «Отсюда есть двадцать рёбер» и «отсюда есть
        # двадцать рёбер, девятнадцать из них никуда» — разные утверждения.
        return {"places": len(self.places), "edges": len(edges),
                "loops": len(loops), "leaving": len(edges) - len(loops),
                "lost": self._lost, "current": self.current,
                "refinements": self._refinements, "splits": len(self.splits),
                "tests": sum(len(v) for v in self.splits.values()),
                "bands": sum(1 for p in self.places.values() if p.band),
                "edges_dropped": self._dropped_edges,
                "mean_visits": round(
                    sum(p.visits for p in self.places.values()) / len(self.places), 2)
                if self.places else 0.0,
                "mean_edge_confidence": round(
                    sum(e.confidence for e in edges) / len(edges), 4) if edges else 0.0,
                "one_shot_edges": sum(1 for e in edges if e.n == 1)}

    def as_dict(self) -> dict[str, Any]:
        return {"places": [p.as_dict() for _, p in sorted(self.places.items())],
                "edges": [e.as_dict() for _, e in sorted(self.edges.items())],
                **self.stats()}

    def __iter__(self) -> Iterator[Place]:
        return iter(self.places.values())
