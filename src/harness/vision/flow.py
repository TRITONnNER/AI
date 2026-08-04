"""Плотный оптический поток и дальность по нему.

Из стека в `CLAUDE.md`: «оптический поток: OpenCV. Нужен для отделения экранного слоя
от мирового и для оценки глубины по параллаксу». Здесь вторая половина.

## Почему отдельным модулем и с необязательным импортом

OpenCV — единственная тяжёлая зависимость во всём проекте, и всё остальное работает
без неё. Поэтому импорт здесь ленивый, а отсутствие библиотеки — **громкий отказ**, а
не тихий откат на что-нибудь похуже. Тихий откат в этом месте страшнее отсутствия
дальности: замер показал бы числа, а по числам было бы не видно, что считал не тот
метод.

## История замеров, потому что она объясняет решения

Дальность по параллаксу была объявлена неработающей после четырёх измеренных попыток
на фазовой корреляции: верным находился только тот план, который совпал с найденным
сдвигом кадра, то есть ответ «всё сместилось как весь кадр». Настоящий поток был
назван нужным следующим шагом.

Оказалось, что дело было наполовину не в методе. На мире с текстурой из блоков одного
размера **никакой** иерархический метод не может найти большое смещение, и это
проверено прямым замером на сдвиге ровно 34 px:

| Текстура | Farneback | DIS |
|---|---|---|
| один масштаб, блоки 6 px | −7.3 | −2.4 |
| четыре октавы: 24, 12, 6, 3 px | −0.0 | **+34.0** |

Пирамида начинает с грубого уровня, чтобы поймать смещение целиком; блоки одного
размера при уменьшении усредняются в ровное поле, и ловить на грубом уровне нечего.
Поэтому мир получил текстуру из нескольких октав — как у настоящей сцены, где
структура есть на всех масштабах сразу, — и только после этого сравнение методов
стало сравнением методов, а не проверкой вырожденности мира.

Из двух вариантов выбран DIS: на том же замере Farneback не нашёл смещения и с
многооктавной текстурой.

## Итог: все три метода неотличимы друг от друга, и разброс больше разницы

Это третья формулировка итога в этом файле, и первые две были неверны — обе потому,
что делались по одному прогону. Сначала «дальность не находится», потом «дальний план
находится впервые, 0.55», потом «все на уровне случайного, 0.38». Ни одно из этих чисел
не было результатом: это были отдельные точки из широкого распределения.

Замер, который можно называть замером: 5 сидов × 5 длин прогона = **25 прогонов на
метод**, мир с глубиной, движение мыши ±14 px, истинный параллакс планов 2.4 / 1.0 /
0.15. «Баланс» — средняя доля верно отнесённых пикселей по трём планам; случайное
угадывание при трёх классах даёт 0.33.

| Метод | Среднее | Мин | Макс | Разброс (σ) | Секунд на 25 прогонов |
|---|---|---|---|---|---|
| фазовая корреляция по гипотезам | 0.49 | 0.25 | 0.83 | 0.16 | 13 |
| плотный поток DIS | 0.50 | 0.30 | 0.73 | 0.14 | 5 |
| поиск с опорой на своё усилие | 0.54 | 0.27 | 0.82 | 0.20 | 86 |

Что из этого следует:

1. **Все три выше случайного** (0.49–0.54 против 0.33). Дальность частично
   находится — это не ноль.
2. **Разница между методами меньше разброса.** Стандартная ошибка среднего при 25
   прогонах — около 0.03–0.04, значит 0.54 против 0.49 это чуть больше одной ошибки.
   Утверждать, что какой-то из методов лучше, нельзя.
3. **Одиночный прогон не значит ничего.** От 0.25 до 0.83 у одного и того же метода;
   отдельные прогоны падают ниже случайного угадывания.
4. **Цена разная на порядок.** Поток — 0.2 секунды на прогон, перебор с опорой на
   усилие — 3.4. При равном результате это единственное различие, которое можно
   утверждать.

Поэтому по умолчанию стоит `depth_method = flow`, и не потому, что он лучше, а потому
что при неотличимом результате он в 17 раз дешевле. И поэтому же результат остаётся
помеченным `trustworthy=False`: опираться на величину с таким разбросом нельзя, а
скрывать разброс — тем более.

Причина разброса измерена: планы перекрывают друг друга на мелком масштабе, и окно
вокруг пикселя часто захватывает два движения. Какое из двух победит, зависит от
случайностей текстуры в этом месте, поэтому по пикселям ответ и скачет. Нужна
совместная оценка «где какой слой» и «как он движется» — послойное разделение
движения. Ни один из трёх методов её не делает.

## Что здесь считается и чего не считается

Считается **отношение**: во сколько раз пиксель смещается сильнее, чем типичный
пиксель кадра. Не расстояние в метрах и не расстояние в пикселях до объекта —
отношение. Это осознанное ограничение из замысла: «мы отказываемся от координат».
Дальше по отношению пиксель попадает в ближний, средний или дальний план порогами из
профиля.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.profile import Profile


class FlowUnavailable(RuntimeError):
    """OpenCV нет. Это отказ, а не повод посчитать дальность как-нибудь иначе."""


_CV2: Any = None
_CV2_TRIED = False


def cv2_module() -> Any:
    """Вернуть cv2 или упасть с объяснением. Тихого отката здесь нет намеренно."""
    global _CV2, _CV2_TRIED
    if _CV2 is None and not _CV2_TRIED:
        _CV2_TRIED = True
        try:
            import cv2                     # noqa: PLC0415 — импорт по требованию
        except ImportError:
            _CV2 = None
        else:
            _CV2 = cv2
    if _CV2 is None:
        raise FlowUnavailable(
            "нет OpenCV: плотный оптический поток посчитать нечем. Поставьте "
            "opencv-python-headless (extras: harness[flow]). Молча считать дальность "
            "другим способом нельзя — по числам потом не отличить, чем считали")
    return _CV2


def available() -> bool:
    """Есть ли чем считать поток. Для `harness backends` и для честных пропусков."""
    try:
        cv2_module()
    except FlowUnavailable:
        return False
    return True


def dense_flow(prev: np.ndarray, cur: np.ndarray, *, preset: str = "medium"
               ) -> np.ndarray:
    """Плотный поток между двумя кадрами: (h, w, 2), пиксели за кадр.

    `preset` — из профиля (`flow_preset`), потому что это компромисс скорость/точность,
    то есть настройка поведения, а не свойство кода.
    """
    cv2 = cv2_module()
    presets = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
               "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
               "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
    if preset not in presets:
        raise ValueError(f"неизвестный режим потока {preset!r}; есть {sorted(presets)}")
    a = prev if prev.ndim == 2 else prev[:, :, :3].mean(axis=2).astype(np.uint8)
    b = cur if cur.ndim == 2 else cur[:, :, :3].mean(axis=2).astype(np.uint8)
    dis = cv2.DISOpticalFlow_create(presets[preset])
    return dis.calc(np.ascontiguousarray(a), np.ascontiguousarray(b), None)


@dataclass(slots=True)
class FlowDepthResult:
    """Дальность по потоку. Отношения, а не расстояния."""

    bands: np.ndarray                 # (h, w): UNKNOWN/NEAR/MID/FAR
    ratio: np.ndarray                 # медиана отношения по кадрам, nan — не знаю
    votes: np.ndarray                 # по скольким кадрам судили пиксель
    voting_frames: int = 0
    skipped_frames: int = 0
    typical: float = 0.0
    # Тот же признак, что у `DepthResult`, и по той же причине: по замеру ни один из
    # трёх методов не отличается от случайного угадывания больше, чем на 0.05.
    # Всякий, кто захочет опереться на дальность, обязан это увидеть, а не выяснить
    # из странного поведения агента через месяц.
    trustworthy: bool = False

    @property
    def decided_fraction(self) -> float:
        from .layers import UNKNOWN

        return float((self.bands != UNKNOWN).mean())

    def mask(self, band: int) -> np.ndarray:
        return self.bands == band

    def summary(self) -> dict[str, Any]:
        from .layers import BAND_NAMES

        return {"trustworthy": self.trustworthy,
                "voting_frames": self.voting_frames,
                "skipped_frames": self.skipped_frames,
                "typical_ratio": round(self.typical, 3),
                "decided_fraction": round(self.decided_fraction, 4),
                **{BAND_NAMES[b]: int((self.bands == b).sum())
                   for b in sorted(BAND_NAMES)}}


class DepthFromFlow:
    """Дальность по плотному потоку. Копит отношения, решает один раз.

    Тот же принцип, что у прежней попытки на фазовой корреляции: сначала
    свидетельства, потом вывод. Отличие в том, чем меряется смещение — не невязкой
    сопоставления по гипотезам, а прямой оценкой потока.

    Нормировка на **медиану по кадру**, а не на найденный сдвиг. Медиана — это
    движение того плана, который занимает большую часть кадра; какого именно, метод не
    знает и знать не должен. Важно только, что она одна и та же во всех кадрах, значит
    отношения между кадрами сравнимы. По этой же причине ответ — отношение, а не
    расстояние: назвать один из планов «средним» можно только по договорённости.

    Замер по 25 прогонам: баланс 0.50 в среднем, от 0.30 до 0.73, разброс 0.14.
    Неотличимо от двух других методов, но в 17 раз дешевле перебора — поэтому это
    метод по умолчанию. Полная таблица и выводы — в описании модуля.

    Медианы найденных отношений на одном прогоне: 1.08 / 1.00 / 0.96 при истинных
    2.40 / 1.00 / 0.15 — то есть поток склонен находить одно и то же движение везде,
    движение того плана, который покрывает больше пикселей.
    """

    def __init__(self, profile: Profile) -> None:
        p = profile.parameters
        self.min_shift = float(p["flow_min_global_shift"])
        self.near_above = float(p["depth_near_ratio"])
        self.far_below = float(p["depth_far_ratio"])
        self.min_votes = int(p["depth_min_votes"])
        self.preset = str(profile.structural["flow_preset"])
        if not self.far_below < self.near_above:
            raise ValueError("порог дальнего должен быть ниже порога ближнего: "
                             f"{self.far_below} и {self.near_above}")
        # Проверяем наличие OpenCV сразу, а не на первом кадре: отказ должен прийти
        # до того, как накопится час записи.
        cv2_module()
        self._prev: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        self._ratios: list[np.ndarray] = []
        self._voting = 0
        self._skipped = 0

    def feed(self, frame: np.ndarray) -> float | None:
        """Дать кадр. Вернуть медиану |потока| по кадру или `None`, если судить не по чему."""
        cur = frame if frame.ndim == 2 else frame[:, :, :3].mean(axis=2).astype(np.uint8)
        if self._shape is None:
            self._shape = (cur.shape[0], cur.shape[1])
        elif (cur.shape[0], cur.shape[1]) != self._shape:
            raise ValueError(f"форма кадра изменилась: {self._shape} → {cur.shape}")

        prev, self._prev = self._prev, cur
        if prev is None:
            return None

        flow = dense_flow(prev, cur, preset=self.preset)
        mag = np.hypot(flow[..., 0], flow[..., 1])
        typical = float(np.median(mag))
        if typical < self.min_shift:
            # Камера стоит — параллакса нет. Это отсутствие данных, а не «всё далеко».
            self._skipped += 1
            return None
        self._ratios.append(mag / typical)
        self._voting += 1
        return typical

    def result(self) -> FlowDepthResult:
        from .layers import FAR, MID, NEAR, UNKNOWN

        shape = self._shape or (1, 1)
        if not self._ratios:
            return FlowDepthResult(
                np.full(shape, UNKNOWN, dtype=np.int8),
                np.full(shape, np.nan), np.zeros(shape, dtype=np.int32),
                self._voting, self._skipped, 0.0)

        stack = np.stack(self._ratios)
        ratio = np.median(stack, axis=0)
        votes = np.full(shape, stack.shape[0], dtype=np.int32)
        bands = np.full(shape, UNKNOWN, dtype=np.int8)
        enough = votes >= self.min_votes
        bands[enough & (ratio >= self.near_above)] = NEAR
        bands[enough & (ratio > self.far_below) & (ratio < self.near_above)] = MID
        bands[enough & (ratio <= self.far_below)] = FAR
        return FlowDepthResult(bands, ratio, votes, self._voting, self._skipped,
                               1.0)


def _windowed_mean_abs(diff: np.ndarray, half: int) -> np.ndarray:
    """Локальное среднее |разницы| окном (2*half+1). Через интегральное изображение.

    Прямая свёртка здесь была бы в разы дороже, а считать это придётся для каждой
    гипотезы смещения на каждом кадре — то есть в самом внутреннем цикле.
    """
    cs = np.pad(np.cumsum(np.cumsum(diff, 0), 1), ((1, 0), (1, 0)))
    h, w = diff.shape
    y0 = np.clip(np.arange(h) - half, 0, h)
    y1 = np.clip(np.arange(h) + half + 1, 0, h)
    x0 = np.clip(np.arange(w) - half, 0, w)
    x1 = np.clip(np.arange(w) + half + 1, 0, w)
    total = (cs[np.ix_(y1, x1)] - cs[np.ix_(y0, x1)]
             - cs[np.ix_(y1, x0)] + cs[np.ix_(y0, x0)])
    count = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).astype(np.float32)
    return total / count


class DepthFromEffort:
    """Дальность через **своё усилие**: сколько сместился пиксель на единицу действия.

    Отличие от двух предыдущих методов не в поиске, а в том, относительно чего меряют.
    И то, и другое раньше нормировалось на движение, найденное в самой картинке: фазовой
    корреляцией или медианой потока. У этого есть встроенный изъян — найденное в картинке
    движение принадлежит тому плану, который занимает больше пикселей, и относительно
    него все остальные планы выглядят «примерно как весь кадр». Замер это и показывал:
    верным находился ровно тот план, что совпал со сдвигом.

    Здесь опора другая: агент знает, **что сам сделал** — на сколько дёрнул мышь, сколько
    держал клавишу. Это его собственное тело, а не знание о мире, и никакого инварианта
    не нарушает. Величина усилия нужна только с точностью до множителя: отношение
    «пикселей на единицу усилия» сравнимо между кадрами, даже если сколько это в
    сантиметрах — неизвестно навсегда.

    Замер по 25 прогонам: баланс 0.54 в среднем, от 0.27 до 0.82, разброс 0.20 —
    самое высокое среднее из трёх и самый большой разброс. Разница с потоком (0.50)
    меньше полутора стандартных ошибок, то есть утверждать превосходство нельзя. Цена
    при этом в 17 раз выше: перебор ±48 px на каждый кадр против одного вызова DIS.

    Поэтому метод не стоит по умолчанию. Он оставлен по другой причине: это
    единственный из трёх, который **не зависит от того, что найдено в самой картинке**,
    и потому единственный, который останется осмысленным, если движется не камера, а
    предметы вокруг. Проверить это негде — в мире с глубиной движется камера, — и пока
    это не проверено, преимущество остаётся рассуждением, а не замером.
    """

    def __init__(self, profile: Profile) -> None:
        p = profile.parameters
        self.half = max(1, int(p["flow_window"]) // 2)
        self.search = int(p["depth_search_px"])
        self.min_effort = float(p["depth_min_effort"])
        self.near_above = float(p["depth_near_ratio"])
        self.far_below = float(p["depth_far_ratio"])
        self.min_votes = int(p["depth_min_votes"])
        if not self.far_below < self.near_above:
            raise ValueError("порог дальнего должен быть ниже порога ближнего: "
                             f"{self.far_below} и {self.near_above}")
        self._prev: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        self._ratios: list[np.ndarray] = []
        self._voting = 0
        self._skipped = 0

    def feed(self, frame: np.ndarray, effort: float = 0.0) -> float | None:
        """Дать кадр и величину своего усилия по горизонтали.

        `effort` — то, что агент сделал сам: смещение мыши, длительность удержания,
        что угодно пропорциональное. Ноль означает «я ничего не делал», и тогда судить
        не по чему: параллакса без движения не бывает. Это отсутствие данных, а не
        «всё далеко».
        """
        cur = frame if frame.ndim == 2 else frame[:, :, :3].mean(axis=2).astype(np.uint8)
        if self._shape is None:
            self._shape = (cur.shape[0], cur.shape[1])
        elif (cur.shape[0], cur.shape[1]) != self._shape:
            raise ValueError(f"форма кадра изменилась: {self._shape} → {cur.shape}")

        prev, self._prev = self._prev, cur.astype(np.int16)
        if prev is None:
            return None
        if abs(effort) < self.min_effort:
            self._skipped += 1
            return None

        cur16 = cur.astype(np.int16)
        best_cost: np.ndarray | None = None
        best_d: np.ndarray | None = None
        for d in range(-self.search, self.search + 1):
            diff = np.abs(cur16 - np.roll(prev, d, axis=1)).astype(np.float32)
            cost = _windowed_mean_abs(diff, self.half)
            if best_cost is None:
                best_cost, best_d = cost, np.full(cost.shape, float(d))
            else:
                better = cost < best_cost
                best_cost = np.where(better, cost, best_cost)
                best_d = np.where(better, float(d), best_d)
        assert best_d is not None
        # Знак: содержимое едет против движения камеры, поэтому делим на −усилие.
        self._ratios.append(best_d / (-float(effort)))
        self._voting += 1
        return float(np.median(best_d))

    def result(self) -> FlowDepthResult:
        from .layers import FAR, MID, NEAR, UNKNOWN

        shape = self._shape or (1, 1)
        if not self._ratios:
            return FlowDepthResult(
                np.full(shape, UNKNOWN, dtype=np.int8),
                np.full(shape, np.nan), np.zeros(shape, dtype=np.int32),
                self._voting, self._skipped, 0.0)
        stack = np.stack(self._ratios)
        ratio = np.median(stack, axis=0)
        votes = np.full(shape, stack.shape[0], dtype=np.int32)
        bands = np.full(shape, UNKNOWN, dtype=np.int8)
        enough = votes >= self.min_votes
        bands[enough & (ratio >= self.near_above)] = NEAR
        bands[enough & (ratio > self.far_below) & (ratio < self.near_above)] = MID
        bands[enough & (ratio <= self.far_below)] = FAR
        return FlowDepthResult(bands, ratio, votes, self._voting, self._skipped, 1.0)


def depth_from_profile(profile: Profile) -> "DepthFromEffort | DepthFromFlow | Any":
    """Собрать оценщик дальности по профилю (`depth_method`).

    Рабочий путь — этот. Прямой вызов конструктора оставлен для замеров, где методы
    сравниваются между собой: там выбор делается явно, а не настройкой.
    """
    method = str(profile.structural["depth_method"])
    if method == "effort":
        return DepthFromEffort(profile)
    if method == "flow":
        return DepthFromFlow(profile)
    if method == "parallax":
        from .layers import DepthFromParallax

        return DepthFromParallax(profile)
    raise ValueError(f"неизвестный метод дальности {method!r}")
