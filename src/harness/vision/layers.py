"""Слои взгляда: система отсчёта × дальность, и тау вместо расстояния.

Из замысла: «нужно делать послойно — прямо перед лицом (условно интерфейс), передний
план, средний, задний». И там же: «мы отказываемся от координат, ведь человек их не
использует, а на глаз и оценка многих вещей происходит временем».

Здесь это доведено до кода, и слоёв получается **два измерения**, а не одно:

- **система отсчёта**: прибито к экрану или прибито к миру. Это `selfworld`, и это
  другой вопрос, чем дальность: интерфейс — не «самое близкое», он вообще в другой
  системе отсчёта.
- **дальность**: насколько сильно пиксель смещается по сравнению со всем кадром.
  Сильнее — ближе, слабее — дальше. Никакой метрики: только отношение.

## Дальность без единого расстояния — и почему она пока не работает

Признак — параллакс движения, самый надёжный из доступных одной камере: при движении
камеры ближнее уезжает дальше, чем дальнее. Реализовано так: у каждого пикселя
проверяется набор гипотез «сместился в λ раз сильнее, чем сдвиг из фазовой
корреляции», стоимости гипотез копятся по кадрам, и решение принимается один раз по
накопленному. Отказ законен: если лучшая гипотеза не лучше второй заметнее чем на
`depth_margin`, пиксель остаётся неопределённым.

**И это не работает.** Четыре измеренные попытки на мире с настоящей глубиной
(`corpus/depth.py`, три плана с параллаксом 2.4 / 1.0 / 0.15):

| Попытка | Ближний | Средний | Дальний |
|---|---|---|---|
| Отношения к сдвигу корреляции, решение покадрово | 6 % | 84 % | 6 % |
| Поиск абсолютного смещения | 15 % | 73 % | 14 % |
| То же с нормировкой внутри кадра | 31 % | 75 % | 36 % |
| Накопление стоимостей, решение один раз | 3 % | 69 % | 3 % |

Верным получается только средний план — тот, который совпадает с найденным сдвигом,
то есть ответ «всё сместилось как весь кадр». Это не дальность, это отсутствие
дальности.

Что именно упирается, по замерам:

1. **Корреляция ловит не то, что нужно.** Она находит сдвиг того плана, который
   занимает больше кадра. В замере: −2 px при истинных −29 у ближнего, −12 у
   среднего, −2 у дальнего. Если единицей оказался дальний план, ближний надо искать
   в пятнадцать раз быстрее, а на сдвиге в 2 px гипотезы λ=0.25 и λ=0.5 обе дают
   ноль пикселей — разрешения не остаётся.
2. **Абсолютное смещение несравнимо между кадрами.** Камера каждый кадр движется
   по-разному, и медиана смещений в пикселях не значит ничего.
3. **Окно сопоставления смешивает глубины.** Это правда даже на исправленном мире с
   крупными связными планами: на границах областей окно накрывает два плана, и такое
   окно не совпадает ни при каком одном смещении.
4. **Средняя абсолютная разность в окне — слишком слабый матчер.** По накопленной
   стоимости минимум оказывается на λ=1 почти везде: свидетельств не хватает, чтобы
   пересилить смещение к «как весь кадр».

Что нужно, чтобы заработало, — и это уже отдельная работа, а не доводка: оценка
потока от грубого к точному (пирамида масштабов), субпиксельная точность и
регуляризация по гладкости с явным разрывом на границах. То есть настоящий оптический
поток, а не проверка нескольких гипотез. Ровно об этом и предупреждал разбор идеи:
«пространственный grounding — главная техническая боль».

Код оставлен, а не удалён, по двум причинам. Он даёт числа, по которым видно, где
стена, — а без них следующий подход начнётся с тех же четырёх попыток. И он честно
отвечает «не знаю» там, где не знает: `DepthResult.trustworthy` равно `False`, пока
эта таблица не станет другой, и тест это проверяет.

## Тау: время до контакта вместо расстояния

Дэвид Ли: животные при прыжке и посадке используют не расстояние, а тау — величину
«размер, делённый на скорость роста размера». Олуша складывает крылья при
фиксированном тау, а не на фиксированной высоте.

Здесь то же самое и так же дёшево: если область растёт в кадре, `тау = размер /
(прирост размера за шаг)`, и это сразу время до контакта **в шагах**, без знания
размера объекта и без знания скорости. Величина ровно в тех единицах, в которых
замысел и предлагал мерить мир: «от этого до того ну где-то N секунд».

## Частоты обновления

Слои обновляются с разной частотой, и это не оптимизация ради оптимизации: «дальний
слой не надо обновлять двадцать раз в секунду — горы никуда не денутся». Частоты
берутся из профиля (`layer_screen_hz` … `layer_far_hz`), а `LayerClock` говорит, чему
пора обновиться на этом кадре. Экономия здесь измеряется, а не предполагается:
`LayerClock.savings()` показывает, какую долю работы удалось не делать.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..core.profile import Profile
from .selfworld import (SCREEN, UNDECIDED, Shift, _local_mean_abs_diff,
                        _local_variance, estimate_global_shift)

# Ярлыки дальности. Ноль — «не знаю», дальше по возрастанию дальности.
UNKNOWN = 0
NEAR = 1
MID = 2
FAR = 3

BAND_NAMES = {UNKNOWN: "не знаю", NEAR: "ближний", MID: "средний", FAR: "дальний"}


# ---------------------------------------------------------------------------
# Дальность из параллакса
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DepthResult:
    """Что получилось по дальности. Отношения, а не расстояния."""

    bands: np.ndarray                 # (h, w) из UNKNOWN/NEAR/MID/FAR
    ratio: np.ndarray                 # выбранное отношение к сдвигу кадра, nan — не знаю
    votes: np.ndarray                 # сколько кадров судили пиксель
    voting_frames: int = 0
    skipped_frames: int = 0
    typical: float = 0.0              # типичное отношение: относительно него и судят

    # Признак, что этому ответу нельзя верить. Не украшение: по замеру дальность
    # находится только для того плана, который совпал с найденным сдвигом, то есть
    # фактически не находится. Пока это так, поле остаётся `False`, и всякий, кто
    # захочет опереться на дальность, обязан это увидеть. Подробности и таблица
    # замеров — в описании модуля.
    trustworthy: bool = False

    @property
    def decided_fraction(self) -> float:
        return float((self.bands != UNKNOWN).mean())

    def mask(self, band: int) -> np.ndarray:
        return self.bands == band

    def summary(self) -> dict[str, Any]:
        return {"trustworthy": self.trustworthy,
                "voting_frames": self.voting_frames,
                "skipped_frames": self.skipped_frames,
                "typical_ratio": round(self.typical, 3),
                "decided_fraction": round(self.decided_fraction, 4),
                **{BAND_NAMES[b]: int((self.bands == b).sum())
                   for b in (NEAR, MID, FAR, UNKNOWN)}}


class DepthFromParallax:
    """Накопитель дальности. Копит **стоимости гипотез**, а не голоса за решения.

    Разница принципиальная, и она стоила замера. Сначала здесь решение принималось
    покадрово (какая гипотеза лучше на этом кадре), а по кадрам бралась медиана
    решений. На шумной картинке покадровый минимум почти случаен, и медиана случайных
    решений сходится к середине списка гипотез: у всех трёх планов отношение выходило
    около единицы, а верных ответов было 0–5 % у ближнего плана при истинном
    смещении 29 px.

    Поэтому копится сама стоимость: для каждой гипотезы — средняя по кадрам невязка
    сопоставления. Решение принимается один раз, по накопленному. Это тот же принцип,
    что и везде в проекте: сначала свидетельства, потом вывод.

    Гипотезы — отношения к сдвигу, который дала фазовая корреляция. Что именно она
    поймала, неважно: она ловит один и тот же план из кадра в кадр, поэтому отношения
    сравнимы между кадрами. Важно только, чтобы набор отношений был широким: если
    корреляция зацепилась за дальний план, ближний окажется в шестнадцать раз быстрее.
    """

    def __init__(self, profile: Profile) -> None:
        p = profile.parameters
        self.window = int(p["flow_window"])
        self.radius = self.window // 2
        self.min_shift = float(p["flow_min_global_shift"])
        self.ratios = tuple(sorted(
            float(x) for x in str(profile.structural["depth_ratios"]).split(",")
            if x.strip()))
        if len(self.ratios) < 2:
            raise ValueError("гипотез дальности должно быть хотя бы две, "
                             f"задано {self.ratios}")
        self.margin = float(p["depth_margin"])
        self.min_votes = int(p["depth_min_votes"])
        self.near_above = float(p["depth_near_ratio"])
        self.far_below = float(p["depth_far_ratio"])
        if not self.far_below < self.near_above:
            raise ValueError("порог дальнего должен быть ниже порога ближнего: "
                             f"{self.far_below} и {self.near_above}")
        self._prev: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        self._cost: np.ndarray | None = None       # (гипотез, h, w) сумма невязок
        self._seen: np.ndarray | None = None       # (гипотез, h, w) сколько раз считали
        self._informative: np.ndarray | None = None
        self._voting = 0
        self._skipped = 0

    def feed(self, frame: np.ndarray, shift: Shift | None = None) -> Shift | None:
        """Дать кадр. `shift` можно передать готовым, чтобы не считать FFT дважды."""
        cur = frame[:, :, :3].mean(axis=2).astype(np.uint8) if frame.ndim == 3 else frame
        if self._shape is None:
            self._shape = (cur.shape[0], cur.shape[1])
            k = len(self.ratios)
            self._cost = np.zeros((k, *self._shape), dtype=np.float64)
            self._seen = np.zeros((k, *self._shape), dtype=np.int32)
            self._informative = np.zeros(self._shape, dtype=np.int32)
        elif (cur.shape[0], cur.shape[1]) != self._shape:
            raise ValueError(f"форма кадра изменилась: {self._shape} → {cur.shape}")

        prev, self._prev = self._prev, cur
        if prev is None:
            return None
        if shift is None:
            shift = estimate_global_shift(prev, cur)
        if shift.magnitude < self.min_shift:
            # Камера стоит: параллакса нет, дальность неразличима. Это отсутствие
            # данных, а не «всё далеко».
            self._skipped += 1
            return shift

        assert self._cost is not None and self._seen is not None
        assert self._informative is not None
        var = _local_variance(cur, self.radius)
        informative = var > max(1.0, float(np.median(var)) * 0.15)
        self._informative += informative.astype(np.int32)

        for i, lam in enumerate(self.ratios):
            dy = int(round(shift.dy * lam))
            dx = int(round(shift.dx * lam))
            mad, ok = _local_mean_abs_diff(cur, prev, dy, dx, self.radius)
            usable = ok & informative & np.isfinite(mad)
            self._cost[i] += np.where(usable, np.nan_to_num(mad, nan=0.0), 0.0)
            self._seen[i] += usable.astype(np.int32)
        self._voting += 1
        return shift

    def result(self) -> DepthResult:
        if self._shape is None or self._cost is None:
            raise ValueError("не подано ни одного кадра")
        assert self._seen is not None and self._informative is not None
        h, w = self._shape
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_cost = np.where(self._seen > 0,
                                 self._cost / np.maximum(self._seen, 1), np.inf)
        # Гипотеза, которую не удалось посчитать ни разу, не участвует.
        mean_cost = np.where(self._seen >= self.min_votes, mean_cost, np.inf)
        order = np.argsort(mean_cost, axis=0)
        best = np.take_along_axis(mean_cost, order[:1], axis=0)[0]
        second = np.take_along_axis(mean_cost, order[1:2], axis=0)[0]
        chosen = np.asarray(self.ratios)[order[0]]

        with np.errstate(invalid="ignore"):
            decided = (np.isfinite(best) & np.isfinite(second)
                       & ((second - best) > self.margin)
                       & (self._informative >= self.min_votes))
        ratio = np.where(decided, chosen, np.nan)
        counts = self._informative.copy()

        bands = np.full((h, w), UNKNOWN, dtype=np.int8)
        # Порог — относительно типичного отношения по кадру, а не абсолютный: что
        # именно поймала корреляция, неизвестно, и привязываться к её единице нельзя.
        with np.errstate(invalid="ignore"):
            typical = float(np.nanmedian(ratio)) if decided.any() else 0.0
        if typical > 0:
            near_at = typical * self.near_above
            far_at = typical * self.far_below
            with np.errstate(invalid="ignore"):
                bands[decided & (ratio >= near_at)] = NEAR
                bands[decided & (ratio <= far_at)] = FAR
                bands[decided & (ratio > far_at) & (ratio < near_at)] = MID
        return DepthResult(bands, ratio, counts, self._voting, self._skipped,
                           typical=typical)


# ---------------------------------------------------------------------------
# Тау: время до контакта
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Looming:
    """Оценка времени до контакта по скорости расширения.

    `tau_steps` — во сколько шагов ожидается контакт. `None` означает «не растёт» или
    «данных мало», и это не то же, что «контакта не будет»: отсутствие оценки должно
    быть отличимо от оценки «далеко».
    """

    size: float
    growth: float                    # во сколько раз размер вырос за шаг
    tau_steps: float | None
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {"size": round(self.size, 3), "growth": round(self.growth, 4),
                "tau_steps": None if self.tau_steps is None else round(self.tau_steps, 2),
                "n": self.n}


class TimeToContact:
    """Тау по последовательности кадров. Ни размеров, ни скоростей — только рост.

    Размер измеряется как площадь области, которая ярче окружения и связна вокруг
    центра внимания. Никакой сегментации объектов: нужен только **размер чего-то
    растущего**, а не «что это».

    Порог контакта задаётся в долях кадра (`looming_contact_fraction`): «контакт» —
    это когда предмет занял столько-то экрана. Так и у человека: он не знает
    расстояния, он видит, что предмет заполнил поле зрения.
    """

    def __init__(self, profile: Profile) -> None:
        p = profile.parameters
        self.contact_fraction = float(p["looming_contact_fraction"])
        self.min_growth = float(p["looming_min_growth"])
        self.history = int(p["looming_history"])
        self.bright_above = float(p["looming_bright_above"])
        self._sizes: list[float] = []

    def measure_size(self, frame: np.ndarray,
                     centre: tuple[int, int] | None = None) -> float:
        """Размер яркого пятна, **связного с центром внимания**, по стороне.

        Не «все яркие пиксели кадра»: первая версия считала именно так, и размер
        выходил 120 px при истинных 6 — она мерила текстуру фона, а не предмет. Связная
        область от центра внимания — то, что человек и назвал бы «этот предмет».

        Центр приходит снаружи: куда смотреть, решает внимание, а не этот модуль. Без
        него берётся середина кадра.

        Сторона, а не площадь: тау определено через линейный размер, и корень из
        площади — это и есть переход к линейному размеру.

        Условие применимости, которое надо назвать прямо: предмет должен быть **ярче
        окружения**. Если он неотличим по яркости, размер измерить нечем, и тау
        неопределимо — не «далеко», а именно неопределимо.
        """
        gray = frame[:, :, :3].mean(axis=2) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        cy, cx = centre if centre is not None else (h // 2, w // 2)
        cy = int(np.clip(cy, 0, h - 1))
        cx = int(np.clip(cx, 0, w - 1))

        med = float(np.median(gray))
        top = float(gray.max())
        thr = med + self.bright_above * (top - med)
        bright = gray >= thr
        if not bright[cy, cx]:
            return 0.0

        # Заливка от центра. Реализована волнами по маске, а не рекурсией: numpy
        # делает шаг волны одним сдвигом, и на кадре 320×180 это доли миллисекунды.
        region = np.zeros_like(bright)
        region[cy, cx] = True
        for _ in range(max(h, w)):
            grown = region.copy()
            grown[1:, :] |= region[:-1, :]
            grown[:-1, :] |= region[1:, :]
            grown[:, 1:] |= region[:, :-1]
            grown[:, :-1] |= region[:, 1:]
            grown &= bright
            if grown.sum() == region.sum():
                break
            region = grown
        return float(np.sqrt(float(region.sum())))

    def feed(self, frame: np.ndarray,
             centre: tuple[int, int] | None = None) -> Looming:
        size = self.measure_size(frame, centre)
        self._sizes.append(size)
        if len(self._sizes) > self.history:
            self._sizes.pop(0)
        n = len(self._sizes)
        if n < 3 or self._sizes[0] <= 0.0:
            return Looming(size, 1.0, None, n)

        # Рост берётся по всей истории, а не по последней паре: одна пара шумит, и
        # тау от неё скачет так, что пользоваться им нельзя.
        span = n - 1
        growth = float((self._sizes[-1] / self._sizes[0]) ** (1.0 / span))
        if growth <= self.min_growth:
            return Looming(size, growth, None, n)

        contact = float(np.sqrt(self.contact_fraction * frame.shape[0] * frame.shape[1]))
        if size >= contact:
            return Looming(size, growth, 0.0, n)
        tau = float(np.log(contact / max(size, 1e-6)) / np.log(growth))
        return Looming(size, growth, tau, n)

    def reset(self) -> None:
        self._sizes.clear()


# ---------------------------------------------------------------------------
# Частоты слоёв и фовеация
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LayerClock:
    """Кому пора обновиться на этом кадре.

    Смысл не в аккуратности, а в стоимости: дальний слой не надо считать двадцать раз
    в секунду — «горы никуда не денутся». Экономия здесь измеряется: `savings()`
    показывает, какую долю обновлений удалось не делать.
    """

    fps: float
    hz: dict[str, float]
    _due: dict[str, float] = field(default_factory=dict)
    frames: int = 0
    updates: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_profile(cls, profile: Profile) -> "LayerClock":
        p = profile.parameters
        return cls(fps=float(p["capture_fps"]),
                   hz={"screen": float(p["layer_screen_hz"]),
                       "near": float(p["layer_near_hz"]),
                       "mid": float(p["layer_mid_hz"]),
                       "far": float(p["layer_far_hz"])})

    def tick(self) -> list[str]:
        """Продвинуть кадр и вернуть слои, которым пора обновиться."""
        self.frames += 1
        due: list[str] = []
        for name, hz in self.hz.items():
            every = max(1.0, self.fps / max(hz, 1e-9))
            acc = self._due.get(name, 0.0) + 1.0
            if acc >= every:
                self._due[name] = acc - every
                self.updates[name] = self.updates.get(name, 0) + 1
                due.append(name)
            else:
                self._due[name] = acc
        return due

    def savings(self) -> float:
        """Доля обновлений, которых удалось не делать по сравнению с «всё каждый кадр»."""
        if not self.frames or not self.hz:
            return 0.0
        did = sum(self.updates.values())
        full = self.frames * len(self.hz)
        return 1.0 - did / full

    def summary(self) -> dict[str, Any]:
        return {"frames": self.frames, "updates": dict(sorted(self.updates.items())),
                "hz": dict(sorted(self.hz.items())),
                "savings": round(self.savings(), 4)}


def foveate(frame: np.ndarray, boxes: Iterable[tuple[int, int, int, int]],
            *, periphery: int = 4) -> tuple[np.ndarray, float]:
    """Высокое разрешение в окнах внимания, огрублённая периферия.

    Ровно то, что делает глаз, и ровно та причина: «разница между 40 и 400
    миллисекундами на кадр». Возвращает кадр и долю сохранённых деталей — то есть
    то, чем экономия оплачена, чтобы её не пришлось оценивать на глаз.

    Периферия огрубляется усреднением по блокам, а не выбрасыванием пикселей:
    выбрасывание теряет мелкое, а мелкое в периферии как раз и служит поводом
    перевести туда внимание.
    """
    if periphery < 1:
        raise ValueError("огрубление периферии должно быть хотя бы 1")
    gray = frame
    h, w = gray.shape[:2]
    if periphery == 1:
        return gray.copy(), 1.0

    hh, ww = (h // periphery) * periphery, (w // periphery) * periphery
    cut = gray[:hh, :ww]
    if cut.ndim == 2:
        blocks = cut.reshape(hh // periphery, periphery, ww // periphery, periphery)
        coarse_small = blocks.mean(axis=(1, 3))
    else:
        planes = cut.shape[2]
        blocks = cut.reshape(hh // periphery, periphery, ww // periphery, periphery,
                             planes)
        coarse_small = blocks.mean(axis=(1, 3, 4))
    coarse = np.repeat(np.repeat(coarse_small, periphery, axis=0), periphery, axis=1)
    out = gray.copy()
    if out.ndim == 2:
        out[:hh, :ww] = coarse.round().astype(out.dtype)
    else:
        out[:hh, :ww] = coarse.round().astype(out.dtype)[:, :, None]

    kept = np.zeros((h, w), dtype=bool)
    for top, left, height, width in boxes:
        t = max(0, int(top))
        l = max(0, int(left))
        out[t:t + int(height), l:l + int(width)] = gray[t:t + int(height),
                                                       l:l + int(width)]
        kept[t:t + int(height), l:l + int(width)] = True
    return out, float(kept.mean())


def look(profile: Profile, frame: np.ndarray,
         centres: Iterable[tuple[int, int]] | None = None
         ) -> tuple[np.ndarray, list[tuple[int, int, int, int]], float]:
    """Взгляд целиком: окна внимания из профиля плюс огрубление периферии из профиля.

    Рабочий путь — этот, а не `foveate` напрямую: сколько окон и насколько грубить
    периферию — настройки прогона (`attention_windows`, `periphery_coarsening`), и
    решать это на месте вызова значило бы завести захардкоженную константу поведения.

    Возвращает (кадр, окна, доля сохранённых деталей). Последнее — то, чем оплачена
    экономия, чтобы её не пришлось оценивать на глаз.
    """
    boxes = attention_boxes(profile, frame.shape, centres)
    coarse = int(profile.parameters["periphery_coarsening"])
    out, kept = foveate(frame, boxes, periphery=coarse)
    return out, boxes, kept


def attention_boxes(profile: Profile, shape: tuple[int, int],
                    centres: Iterable[tuple[int, int]] | None = None
                    ) -> list[tuple[int, int, int, int]]:
    """Окна внимания: сколько и какого размера. Число — из профиля.

    Центры задаются снаружи (их выбирает внимание, а не этот модуль); без них окна
    ставятся по центру кадра и по местам наибольшей неопределённости — но выбор
    «куда смотреть» здесь не решается, здесь только нарезка.
    """
    n = int(profile.parameters["attention_windows"])
    h, w = shape[:2]
    side = max(8, int(min(h, w) * 0.35))
    if centres is None:
        centres = [(h // 2, w // 2)]
        if n > 1:
            centres = centres + [(h // 3, w // 3), (2 * h // 3, 2 * w // 3),
                                 (h // 3, 2 * w // 3), (2 * h // 3, w // 3)]
    boxes = []
    for cy, cx in list(centres)[:n]:
        top = int(np.clip(cy - side // 2, 0, max(0, h - side)))
        left = int(np.clip(cx - side // 2, 0, max(0, w - side)))
        boxes.append((top, left, side, side))
    return boxes
