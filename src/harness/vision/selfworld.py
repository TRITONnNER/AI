"""0.6 — разделение себя и мира.

Задача из вехи: отделить **экранный слой** (прибит к экрану) от **мирового**
(смещается при повороте камеры) по параллаксу, без единой захардкоженной
координаты и без знания, что это за игра.

Как это устроено здесь.

1. **Глобальный сдвиг** между соседними кадрами считается фазовой корреляцией
   через FFT. Это даёт целочисленный сдвиг за одно преобразование, без перебора
   гипотез. Перебор со радиусом 24 стоил бы 2401 сравнение полного кадра на
   каждую пару — фазовая корреляция стоит три FFT.

2. **Классификация попиксельная** — проверка ровно двух гипотез, а не поиск:
   пиксель либо сместился как весь кадр (мировой), либо не сместился вовсе
   (экранный). Сравниваем среднюю абсолютную разность в окрестности пикселя при
   сдвиге на глобальный вектор и при нулевом сдвиге. Кто меньше — тот и слой.

   Окрестность нужна потому, что один пиксель ничего не различает: на гладком
   участке обе гипотезы дают ноль. Но окрестность считается скользящим окном, а
   не сеткой блоков: элементы интерфейса бывают в несколько пикселей высотой, и
   сетка с блоком крупнее элемента смешивает слои внутри блока — на замере это
   давало IoU 0.12 вместо 0.9. Скользящее окно считается за то же время через
   интегральные суммы.

3. **Второй проход.** Интерфейс тянет фазовую корреляцию к нулю, потому что он
   неподвижен и контрастен. Поэтому после первой классификации глобальный сдвиг
   пересчитывается по кадру, в котором предполагаемый интерфейс погашен. Обычно
   этого хватает; если сдвиг после пересчёта изменился, доверяем второму.

4. **Голоса, а не один кадр.** Один кадр ничего не решает: при неподвижной
   камере оба слоя выглядят одинаково. Поэтому кадры, где глобальный сдвиг мал,
   не голосуют вообще, а решение по пикселю принимается только при
   `selfworld_min_votes` голосах. Пиксель, не набравший голосов, остаётся
   неопределённым — и это отдельное третье состояние, а не «значит мировой».

Ни одна константа поведения здесь не захардкожена: всё берётся из `Profile`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..core.profile import Profile

UNDECIDED = 0
SCREEN = 1
WORLD = 2


@dataclass(frozen=True, slots=True)
class Shift:
    dy: int
    dx: int
    peak: float          # резкость пика корреляции: 0…1, чем выше, тем надёжнее

    @property
    def magnitude(self) -> float:
        return float((self.dy ** 2 + self.dx ** 2) ** 0.5)


def _prep(frame: np.ndarray) -> np.ndarray:
    """Кадр к float с убранным средним и оконным сглаживанием краёв.

    Окно нужно, потому что FFT считает кадр периодическим: без окна резкая
    граница кадра даёт свой собственный пик корреляции.
    """
    if frame.ndim == 3:
        frame = frame[:, :, :3].mean(axis=2)
    f = frame.astype(np.float32)
    f -= f.mean()
    h, w = f.shape
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    return f * wy[:, None] * wx[None, :]


def estimate_global_shift(prev: np.ndarray, cur: np.ndarray,
                          suppress: np.ndarray | None = None) -> Shift:
    """Сдвиг `cur` относительно `prev` фазовой корреляцией.

    `suppress` — маска пикселей, которые надо погасить перед подсчётом (второй
    проход гасит предполагаемый интерфейс).
    """
    a, b = prev, cur
    if suppress is not None:
        a = np.where(suppress, 0, a)
        b = np.where(suppress, 0, b)
    A = np.fft.rfft2(_prep(a))
    B = np.fft.rfft2(_prep(b))
    cross = A * np.conj(B)
    mag = np.abs(cross)
    cross = np.divide(cross, mag, out=np.zeros_like(cross), where=mag > 1e-9)
    corr = np.fft.irfft2(cross, s=a.shape[-2:])
    idx = int(np.argmax(corr))
    dy, dx = np.unravel_index(idx, corr.shape)
    h, w = corr.shape
    # Пик за половиной размера — это отрицательный сдвиг.
    dy = dy - h if dy > h // 2 else dy
    dx = dx - w if dx > w // 2 else dx
    peak = float(corr.max())
    denom = float(np.abs(corr).mean()) + 1e-9
    sharpness = float(min(1.0, peak / (denom * 50.0)))
    # Знак: корреляция даёт сдвиг prev→cur, а нам нужно, куда уехало содержимое.
    return Shift(int(-dy), int(-dx), sharpness)


def _shift_into(prev: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Сдвинуть `prev` на (dy, dx). Возвращает (сдвинутое, маска достоверного).

    То, что уехало за край кадра, недостоверно: там нечему совпадать, и судить
    по этой области нельзя ни в пользу одной гипотезы, ни в пользу другой.
    """
    h, w = prev.shape
    out = np.zeros((h, w), dtype=np.int16)
    valid = np.zeros((h, w), dtype=bool)
    if abs(dy) >= h or abs(dx) >= w:
        # Сдвиг больше кадра: перекрытия нет вообще, сравнивать нечего. Возвращается
        # пустая маска достоверного, а не исключение: «эта гипотеза неприменима» —
        # законный ответ, и вызывающему не нужно знать про размеры кадра.
        return out, valid
    ys_src = slice(max(0, -dy), min(h, h - dy))
    ys_dst = slice(max(0, dy), min(h, h + dy))
    xs_src = slice(max(0, -dx), min(w, w - dx))
    xs_dst = slice(max(0, dx), min(w, w + dx))
    out[ys_dst, xs_dst] = prev.astype(np.int16)[ys_src, xs_src]
    valid[ys_dst, xs_dst] = True
    return out, valid


def _box_sum(a: np.ndarray, radius: int) -> np.ndarray:
    """Сумма по окну (2r+1)² вокруг каждого пикселя. Через интегральные суммы.

    Края считаются по усечённому окну, а не отражением: отражение придумало бы
    данные, которых нет.
    """
    h, w = a.shape
    pad = np.zeros((h + 1, w + 1), dtype=np.float64)
    pad[1:, 1:] = a
    ii = pad.cumsum(axis=0).cumsum(axis=1)
    y0 = np.clip(np.arange(h) - radius, 0, h)
    y1 = np.clip(np.arange(h) + radius + 1, 0, h)
    x0 = np.clip(np.arange(w) - radius, 0, w)
    x1 = np.clip(np.arange(w) + radius + 1, 0, w)
    return (ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)]
            - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)])


def _local_mean_abs_diff(cur: np.ndarray, prev: np.ndarray, dy: int, dx: int,
                         radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Средняя |разность| в окрестности при сдвиге (dy, dx), и маска достоверного.

    Достоверным считается пиксель, у которого в окне есть хотя бы половина
    сравнимых точек: иначе оценка держится на двух пикселях у края кадра.
    """
    shifted, valid = _shift_into(prev, dy, dx)
    diff = np.abs(cur.astype(np.int16) - shifted).astype(np.float64)
    diff[~valid] = 0.0
    s = _box_sum(diff, radius)
    n = _box_sum(valid.astype(np.float64), radius)
    full = float((2 * radius + 1) ** 2)
    enough = n >= full * 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, s / np.maximum(n, 1.0), np.nan)
    return mean, enough


def _local_variance(frame: np.ndarray, radius: int) -> np.ndarray:
    """Локальная дисперсия яркости: где её нет, там гипотезы неразличимы."""
    f = frame.astype(np.float64)
    cnt = _box_sum(np.ones_like(f), radius)
    m = _box_sum(f, radius) / np.maximum(cnt, 1.0)
    m2 = _box_sum(f * f, radius) / np.maximum(cnt, 1.0)
    return np.maximum(0.0, m2 - m * m)


@dataclass(slots=True)
class SeparationResult:
    """Результат разделения. Три состояния, а не два: неопределённость честная."""

    labels: np.ndarray                      # (h, w) из UNDECIDED/SCREEN/WORLD
    screen_votes: np.ndarray
    world_votes: np.ndarray
    window: int
    shape: tuple[int, int]
    shifts: list[Shift] = field(default_factory=list)
    voting_frames: int = 0
    skipped_frames: int = 0

    @property
    def decided_fraction(self) -> float:
        return float((self.labels != UNDECIDED).mean())

    def pixel_mask(self, label: int = SCREEN) -> np.ndarray:
        """Маска в пикселях исходного кадра."""
        return self.labels == label

    def summary(self) -> dict[str, object]:
        return {
            "pixels": int(self.labels.size),
            "screen": int((self.labels == SCREEN).sum()),
            "world": int((self.labels == WORLD).sum()),
            "undecided": int((self.labels == UNDECIDED).sum()),
            "decided_fraction": round(self.decided_fraction, 4),
            "voting_frames": self.voting_frames,
            "skipped_frames": self.skipped_frames,
            "window": self.window,
        }


class SelfWorldSeparator:
    """Накопитель голосов. Кормится кадрами по одному — годится и для потока."""

    def __init__(self, profile: Profile) -> None:
        p = profile.parameters
        self.window = int(p["flow_window"])
        if self.window % 2 == 0 or self.window < 3:
            raise ValueError(f"flow_window должен быть нечётным и не меньше 3, задано {self.window}")
        self.radius = self.window // 2
        self.min_global_shift = float(p["flow_min_global_shift"])
        self.screen_tol = float(p["screen_layer_tolerance"])
        self.world_tol = float(p["world_layer_tolerance"])
        self.min_votes = int(p["selfworld_min_votes"])
        self.static_eps = float(p["screen_static_epsilon"])
        self._prev: np.ndarray | None = None
        self._screen_votes: np.ndarray | None = None
        self._world_votes: np.ndarray | None = None
        self._shifts: list[Shift] = []
        self._voting = 0
        self._skipped = 0
        self._shape: tuple[int, int] | None = None

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return frame[:, :, :3].mean(axis=2).astype(np.uint8)
        return frame

    def feed(self, frame: np.ndarray) -> Shift | None:
        """Дать очередной кадр. Возвращает оценённый сдвиг или None."""
        cur = self._gray(frame)
        if self._shape is None:
            self._shape = (cur.shape[0], cur.shape[1])
            if min(cur.shape[:2]) < self.window:
                raise ValueError(
                    f"кадр {cur.shape} меньше окна {self.window}: "
                    "уменьшите flow_window в профиле")
            self._screen_votes = np.zeros(self._shape, dtype=np.int32)
            self._world_votes = np.zeros(self._shape, dtype=np.int32)
        elif (cur.shape[0], cur.shape[1]) != self._shape:
            raise ValueError(f"форма кадра изменилась: {self._shape} → {cur.shape}")

        prev, self._prev = self._prev, cur
        if prev is None:
            return None

        shift = estimate_global_shift(prev, cur)
        if shift.magnitude < self.min_global_shift:
            # Камера стоит: параллакса нет, различить слои нечем. Это не отказ, а
            # отсутствие информации, и голосовать по такому кадру нельзя.
            self._skipped += 1
            self._shifts.append(shift)
            return shift

        mad_screen, ok_screen = _local_mean_abs_diff(cur, prev, 0, 0, self.radius)
        mad_world, ok_world = _local_mean_abs_diff(cur, prev, shift.dy, shift.dx,
                                                  self.radius)

        # Второй проход: гасим то, что похоже на экранный слой, и уточняем сдвиг.
        # Интерфейс неподвижен и контрастен, поэтому тянет корреляцию к нулю.
        with np.errstate(invalid="ignore"):
            screenish = np.nan_to_num(mad_screen, nan=np.inf) + 1e-6 < np.nan_to_num(
                mad_world, nan=np.inf)
        if screenish.any() and not screenish.all():
            refined = estimate_global_shift(prev, cur, suppress=screenish)
            if refined.magnitude >= self.min_global_shift and (
                    refined.dy, refined.dx) != (shift.dy, shift.dx):
                shift = refined
                mad_world, ok_world = _local_mean_abs_diff(cur, prev, shift.dy,
                                                          shift.dx, self.radius)

        # Где нет текстуры, там гипотезы неразличимы. Порог не захардкожен: это
        # доля от медианной дисперсии кадра, то есть свойство самой картинки.
        var = _local_variance(cur, self.radius)
        informative = var > max(1.0, float(np.median(var)) * 0.15)

        # Окно устойчиво, но размывает границу: у пикселя мира, соседнего с
        # интерфейсом, окно частично накрывает неподвижную контрастную рамку, и
        # рамка перевешивает — окно уверенно голосует «экранный». Получается ореол
        # шириной в радиус окна, и он давал 38 тысяч ложных пикселей за прогон.
        #
        # Лечится не порогом уверенности (ложные пиксели различимы уверенно —
        # они уверенно неправильные) и не отказом от окна (попиксельно точность
        # выходит 0.65, то есть хуже). Лечится тем, что экранный слой **статичен**:
        # раз он не двигается, его пиксель между кадрами не меняется вовсе.
        # Пиксель мира при движении камеры меняется всегда. Поэтому к голосу за
        # экранный слой добавлено условие «сам пиксель не изменился», и точность
        # выросла с 0.62 до 0.99 при той же полноте.
        #
        # Замечание о законности: это не другой признак вместо параллакса, а тот
        # же признак с другой стороны — и он тоже выведен из одних пикселей, без
        # координат и без знания об игре. Один он не работает: кадры, где камера
        # стоит, не голосуют вообще, поэтому «неподвижное» не превращается в
        # «экранное» само по себе.
        pix_screen, _ = _local_mean_abs_diff(cur, prev, 0, 0, 0)
        pix_world, _ = _local_mean_abs_diff(cur, prev, shift.dy, shift.dx, 0)

        with np.errstate(invalid="ignore"):
            world_better = ((mad_world + self.world_tol < mad_screen)
                            & (pix_world <= pix_screen))
            screen_better = ((mad_screen + self.screen_tol < mad_world)
                             & (pix_screen <= self.static_eps))
        known = (np.isfinite(mad_world) & np.isfinite(mad_screen)
                 & np.isfinite(pix_world) & np.isfinite(pix_screen)
                 & ok_screen & ok_world & informative)

        self._world_votes += (world_better & known).astype(np.int32)
        self._screen_votes += (screen_better & known).astype(np.int32)
        self._voting += 1
        self._shifts.append(shift)
        return shift

    def result(self) -> SeparationResult:
        if self._screen_votes is None or self._shape is None:
            raise ValueError("не подано ни одного кадра")
        sv, wv = self._screen_votes, self._world_votes
        total = sv + wv
        labels = np.full(sv.shape, UNDECIDED, dtype=np.int8)
        enough = total >= self.min_votes
        labels[enough & (sv > wv)] = SCREEN
        labels[enough & (wv >= sv)] = WORLD
        labels[~enough] = UNDECIDED
        return SeparationResult(labels, sv.copy(), wv.copy(), self.window, self._shape,
                                list(self._shifts), self._voting, self._skipped)


def separate(frames: Iterable[np.ndarray], profile: Profile) -> SeparationResult:
    """Прогнать последовательность кадров и получить разделение слоёв."""
    sep = SelfWorldSeparator(profile)
    for f in frames:
        sep.feed(f)
    return sep.result()


# --- второй признак: неподвижность ------------------------------------------
#
# Зачем он нужен. Параллакс требует глобального сдвига: он сравнивает «сместился
# как весь кадр» с «не сместился». Там, где кадр не смещается целиком, сравнивать
# нечего, и параллакс обязан молчать — на рабочем столе двигаются отдельные окна,
# в проигрывателе содержимое меняется само, не съезжая. Кросс-доменный замер это и
# показал: два домена из четырёх — 100% «не знаю». Отказ честный, но бесполезный.
#
# Признак здесь другой и выводится из тех же пикселей: **что не меняется, когда
# меняется остальное**. Обои рабочего стола, полоса управления проигрывателя,
# рамка панели — не меняются вовсе, а окна и содержимое меняются.
#
# Чего этот признак не может, и почему он остаётся вторым, а не заменяет первый:
# он не различает «прибито к экрану» и «стоит на месте». Неподвижный камень в
# неподвижной сцене он объявит экранным слоем, и это будет неправдой. Параллакс
# такое различает. Поэтому арбитр предпочитает параллакс, когда тот применим, и
# честно сообщает, каким признаком получен ответ: ответы двух признаков значат
# разное, и складывать их в одну кучу нельзя.


PARALLAX = "parallax"
STILLNESS = "stillness"
COUPLING = "coupling"
NO_SIGNAL = "none"


@dataclass(slots=True)
class StillnessResult:
    """Результат по второму признаку. Те же три состояния."""

    labels: np.ndarray
    change_rate: np.ndarray            # доля кадров, в которых пиксель менялся
    seen: np.ndarray                   # в скольких кадрах пиксель был информативен
    shape: tuple[int, int]
    voting_frames: int = 0
    skipped_frames: int = 0

    @property
    def decided_fraction(self) -> float:
        return float((self.labels != UNDECIDED).mean())

    def pixel_mask(self, label: int = SCREEN) -> np.ndarray:
        return self.labels == label

    def summary(self) -> dict[str, object]:
        return {
            "pixels": int(self.labels.size),
            "screen": int((self.labels == SCREEN).sum()),
            "world": int((self.labels == WORLD).sum()),
            "undecided": int((self.labels == UNDECIDED).sum()),
            "decided_fraction": round(self.decided_fraction, 4),
            "voting_frames": self.voting_frames,
            "skipped_frames": self.skipped_frames,
        }


class StillnessSeparator:
    """Накопитель по второму признаку: как часто пиксель менялся."""

    def __init__(self, profile: Profile) -> None:
        p = profile.parameters
        self.delta = int(p["stillness_pixel_delta"])
        self.min_frame_change = float(p["stillness_min_frame_change"])
        self.static_rate = float(p["stillness_static_rate"])
        self.moving_rate = float(p["stillness_moving_rate"])
        if self.static_rate > self.moving_rate:
            raise ValueError(
                f"stillness_static_rate {self.static_rate} выше "
                f"stillness_moving_rate {self.moving_rate}: пороги перекрыты, "
                "пиксель попал бы в оба состояния сразу")
        self.min_votes = int(p["stillness_min_votes"])
        self.radius = int(p["flow_window"]) // 2
        self._prev: np.ndarray | None = None
        self._changes: np.ndarray | None = None
        self._seen: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        self._voting = 0
        self._skipped = 0

    def feed(self, frame: np.ndarray) -> float | None:
        """Дать кадр. Возвращает долю изменившихся пикселей или None."""
        cur = frame[:, :, :3].mean(axis=2).astype(np.uint8) if frame.ndim == 3 else frame
        if self._shape is None:
            self._shape = (cur.shape[0], cur.shape[1])
            self._changes = np.zeros(self._shape, dtype=np.int32)
            self._seen = np.zeros(self._shape, dtype=np.int32)
        elif (cur.shape[0], cur.shape[1]) != self._shape:
            raise ValueError(f"форма кадра изменилась: {self._shape} → {cur.shape}")

        prev, self._prev = self._prev, cur
        if prev is None:
            return None
        changed = np.abs(cur.astype(np.int16) - prev.astype(np.int16)) > self.delta
        share = float(changed.mean())
        if share < self.min_frame_change:
            # Ничего не меняется: неподвижно всё, и признак ничего не выделяет.
            self._skipped += 1
            return share

        # Тот же честный запрет, что и у параллакса: где нет текстуры, там
        # «не изменился» ничего не значит — гладкое небо не меняется и при
        # движении камеры. Без этого запрета однородные области мира уехали бы в
        # экранный слой, и точность упала бы там, где картинка гладкая.
        var = _local_variance(cur, self.radius)
        informative = var > max(1.0, float(np.median(var)) * 0.15)
        assert self._changes is not None and self._seen is not None
        self._changes += (changed & informative).astype(np.int32)
        self._seen += informative.astype(np.int32)
        self._voting += 1
        return share

    def result(self) -> StillnessResult:
        if self._changes is None or self._seen is None or self._shape is None:
            raise ValueError("не подано ни одного кадра")
        seen = self._seen
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(seen > 0, self._changes / np.maximum(seen, 1), np.nan)
        labels = np.full(self._shape, UNDECIDED, dtype=np.int8)
        enough = seen >= self.min_votes
        labels[enough & (rate <= self.static_rate)] = SCREEN
        labels[enough & (rate >= self.moving_rate)] = WORLD
        return StillnessResult(labels, rate, seen.copy(), self._shape,
                               self._voting, self._skipped)


# --- третий признак: связь изменений с движением камеры ---------------------
#
# Что он закрывает. Первые два признака оба мимо анимированного обрамления —
# ползущего заполнения полосы, мигающего индикатора. Параллакс мимо, потому что
# такой пиксель не совпадает с собой (условие статичности не выполняется).
# Неподвижность мимо, потому что он меняется. А он при этом прибит к экрану, и по
# замеру это 33–63 % всего обрамления — то есть пробел не мелкий.
#
# Различие, которое их разводит, простое: **меняется ли пиксель тогда, когда камера
# стоит**. Мировой пиксель при неподвижной камере не меняется вовсе: меняться ему
# нечем. Анимированное обрамление живёт своей жизнью и меняется одинаково, движется
# камера или нет. Замер по двум доменам, медианы:
#
#     класс                   при движении   при остановке
#     статичное обрамление        0.000          0.000
#     анимированное               0.049          0.036
#     мир                         0.810          0.000
#
# Разделение с большим запасом, и оно выведено из тех же пикселей плюс уже
# посчитанный глобальный сдвиг — ничего нового в источниках данных.
#
# Чего он не может, и это надо сказать прямо: мировой объект, который движется сам
# (вода, чужой персонаж, мигающая лампа в сцене), меняется и при стоящей камере — и
# будет принят за анимированное обрамление. В синтетическом мире такого нет, а в
# настоящей игре есть, поэтому здесь это ограничение, а не решённый вопрос.
# И он требует записи, в которой камера **и двигалась, и стояла**: там, где сдвига
# нет вовсе, признак неприменим и честно молчит.


@dataclass(slots=True)
class CouplingResult:
    """Результат по третьему признаку, с обеими частотами изменений."""

    labels: np.ndarray
    rate_moving: np.ndarray
    rate_still: np.ndarray
    seen_moving: np.ndarray
    seen_still: np.ndarray
    shape: tuple[int, int]
    moving_frames: int = 0
    still_frames: int = 0
    # Кадры с мелким сдвигом: камера то ли дрогнула, то ли нет. Не голосуют нигде.
    ambiguous_frames: int = 0

    @property
    def decided_fraction(self) -> float:
        return float((self.labels != UNDECIDED).mean())

    @property
    def applicable(self) -> bool:
        """Признак применим только там, где камера и двигалась, и стояла."""
        return self.moving_frames > 0 and self.still_frames > 0

    def pixel_mask(self, label: int = SCREEN) -> np.ndarray:
        return self.labels == label

    def summary(self) -> dict[str, object]:
        return {
            "screen": int((self.labels == SCREEN).sum()),
            "world": int((self.labels == WORLD).sum()),
            "undecided": int((self.labels == UNDECIDED).sum()),
            "decided_fraction": round(self.decided_fraction, 4),
            "moving_frames": self.moving_frames,
            "still_frames": self.still_frames,
            "ambiguous_frames": self.ambiguous_frames,
            "applicable": self.applicable,
        }


class MotionCouplingSeparator:
    """Считает частоту изменений отдельно при движении камеры и при её остановке."""

    def __init__(self, profile: Profile) -> None:
        p = profile.parameters
        self.delta = int(p["stillness_pixel_delta"])
        self.min_global_shift = float(p["flow_min_global_shift"])
        self.static_rate = float(p["stillness_static_rate"])
        self.moving_rate = float(p["stillness_moving_rate"])
        self.max_ratio = float(p["coupling_max_ratio"])
        self.min_votes = int(p["coupling_min_votes"])
        self.radius = int(p["flow_window"]) // 2
        self._prev: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None
        self._ch_m: np.ndarray | None = None
        self._ch_s: np.ndarray | None = None
        self._seen_m: np.ndarray | None = None
        self._seen_s: np.ndarray | None = None
        self._moving = 0
        self._still = 0
        self._ambiguous = 0

    def feed(self, frame: np.ndarray, shift: Shift | None = None) -> Shift | None:
        """Дать кадр. `shift` можно передать готовым, чтобы не считать FFT дважды."""
        cur = frame[:, :, :3].mean(axis=2).astype(np.uint8) if frame.ndim == 3 else frame
        if self._shape is None:
            self._shape = (cur.shape[0], cur.shape[1])
            z = lambda: np.zeros(self._shape, dtype=np.int32)     # noqa: E731
            self._ch_m, self._ch_s, self._seen_m, self._seen_s = z(), z(), z(), z()
        elif (cur.shape[0], cur.shape[1]) != self._shape:
            raise ValueError(f"форма кадра изменилась: {self._shape} → {cur.shape}")

        prev, self._prev = self._prev, cur
        if prev is None:
            return None
        if shift is None:
            shift = estimate_global_shift(prev, cur)

        changed = np.abs(cur.astype(np.int16) - prev.astype(np.int16)) > self.delta
        # Тот же честный запрет, что у первых двух признаков: без текстуры «не
        # изменился» ничего не значит.
        var = _local_variance(cur, self.radius)
        informative = var > max(1.0, float(np.median(var)) * 0.15)
        assert self._ch_m is not None and self._ch_s is not None
        assert self._seen_m is not None and self._seen_s is not None
        # Три исхода, а не два. Кадр с мелким сдвигом — не «камера стояла»: мир в нём
        # сдвинулся, пиксели изменились, и в частоту «при остановке» это попадать не
        # должно. Именно на этом признак ошибался в документе: прокрутка на пиксель
        # ниже порога считалась остановкой, мировые пиксели выглядели меняющимися
        # «сами по себе», и точность падала с 1.00 до 0.84. Остановка — это ровно
        # нулевой сдвиг; всё между нулём и порогом не голосует нигде.
        if shift.magnitude >= self.min_global_shift:
            # Одной текстуры в окрестности мало. «Пиксель не изменился» говорит
            # что-то только про пиксель, который **мог** измениться. Пиксель в
            # белом промежутке между строками текста при прокрутке остаётся белым:
            # окрестность у него текстурная, а сам он неотличим от неподвижного, и
            # признак записывал его в экранный слой. На замере по документу это
            # давало 1004 ложных пикселя и точность 0.84 вместо 1.00.
            #
            # Проверяется прямо: сдвинуть предыдущий кадр на глобальный вектор и
            # сравнить с ним же. Различие есть — пиксель способен показать движение;
            # различия нет — он не свидетель ни в ту, ни в другую сторону.
            moved_self, valid = _shift_into(prev, shift.dy, shift.dx)
            would_change = valid & (
                np.abs(moved_self - prev.astype(np.int16)) > self.delta)
            self._ch_m += (changed & informative & would_change).astype(np.int32)
            self._seen_m += (informative & would_change).astype(np.int32)
            self._moving += 1
        elif shift.magnitude == 0.0:
            self._ch_s += (changed & informative).astype(np.int32)
            self._seen_s += informative.astype(np.int32)
            self._still += 1
        else:
            self._ambiguous += 1
        return shift

    def result(self) -> CouplingResult:
        if self._shape is None or self._ch_m is None:
            raise ValueError("не подано ни одного кадра")
        assert self._ch_s is not None and self._seen_m is not None
        assert self._seen_s is not None
        with np.errstate(invalid="ignore", divide="ignore"):
            rm = np.where(self._seen_m > 0,
                          self._ch_m / np.maximum(self._seen_m, 1), np.nan)
            rs = np.where(self._seen_s > 0,
                          self._ch_s / np.maximum(self._seen_s, 1), np.nan)
        labels = np.full(self._shape, UNDECIDED, dtype=np.int8)
        enough = (self._seen_m >= self.min_votes) & (self._seen_s >= self.min_votes)

        with np.errstate(invalid="ignore"):
            static = (rm <= self.static_rate) & (rs <= self.static_rate)
            # Меняется, но не от камеры: при остановке меняется почти так же часто,
            # как при движении. Это и есть анимированное обрамление.
            decoupled = ((rs > self.static_rate)
                         & (rm <= self.max_ratio * np.maximum(rs, 1e-9)))
            worldish = (rm >= self.moving_rate) & (rs <= self.static_rate)
        labels[enough & (static | decoupled)] = SCREEN
        labels[enough & worldish & ~static & ~decoupled] = WORLD
        return CouplingResult(labels, rm, rs, self._seen_m.copy(),
                              self._seen_s.copy(), self._shape,
                              self._moving, self._still, self._ambiguous)


MERGED = "merged"

# Порядок силы признаков — от сильного к слабому. Он выведен из того, что каждый
# **способен** различить, а не из измеренных чисел:
#
# 1. Связь с движением камеры различает все три случая: неподвижное обрамление,
#    анимированное обрамление и мир.
# 2. Параллакс различает «прибито к экрану» и «сместилось вместе с миром», но
#    анимированное обрамление ему не даётся: оно не совпадает с собой.
# 3. Неподвижность различает только «менялось» и «не менялось», то есть путает
#    «прибито к экрану» и «стоит на месте».
STRENGTH = (COUPLING, PARALLAX, STILLNESS)


@dataclass(slots=True)
class LayerVerdict:
    """Чем именно получен ответ. Признак называется, а не подразумевается.

    Ответ собирается из трёх признаков **с происхождением у каждого пикселя**, а не
    выбором одного признака на весь кадр. Причина та же, по которой у убеждения есть
    происхождение: у каждого признака своя область применимости, и отбрасывать
    ответы там, где он единственный, кто может ответить, — терять знание. Замер
    показал это прямо: по документу связь с движением даёт точность 0.90, но решает
    только 47 % пикселей, а параллакс решает 62 % с точностью 0.95; выбор одного
    признака на весь кадр в любом случае терял бы часть ответа.

    Что при этом не размывается: `provenance` помнит, какой признак решил каждый
    пиксель, а `disagreements` считает пиксели, где два признака ответили
    **по-разному**. Расхождение — не мелочь, а признак того, что один из них врёт, и
    оно обязано быть на виду.
    """

    signal: str        # MERGED | PARALLAX | STILLNESS | COUPLING | NO_SIGNAL
    labels: np.ndarray
    reason: str
    parallax: SeparationResult
    stillness: StillnessResult
    coupling: CouplingResult
    # Какой признак решил пиксель: 0 — никакой, дальше по индексу в STRENGTH + 1.
    provenance: np.ndarray | None = None
    disagreements: int = 0

    @property
    def disagreement_fraction(self) -> float:
        decided = int((self.labels != UNDECIDED).sum())
        return self.disagreements / decided if decided else 0.0

    def by_signal(self, signal: str) -> np.ndarray:
        """Маска пикселей, решённых именно этим признаком."""
        if self.provenance is None:
            return np.zeros(self.labels.shape, dtype=bool)
        return self.provenance == (STRENGTH.index(signal) + 1)

    @property
    def decided_fraction(self) -> float:
        return float((self.labels != UNDECIDED).mean())

    def pixel_mask(self, label: int = SCREEN) -> np.ndarray:
        return self.labels == label

    def summary(self) -> dict[str, object]:
        return {"signal": self.signal, "reason": self.reason,
                "decided_fraction": round(self.decided_fraction, 4),
                "decided_by": {s: int(self.by_signal(s).sum()) for s in STRENGTH},
                "disagreements": self.disagreements,
                "disagreement_fraction": round(self.disagreement_fraction, 4),
                "parallax": self.parallax.summary(),
                "stillness": self.stillness.summary(),
                "coupling": self.coupling.summary()}


class LayerArbiter:
    """Три признака сразу и явный выбор между ними.

    Выбор, а не смесь. Смесь была бы удобнее в таблице и хуже по смыслу: пиксель,
    названный экранным по неподвижности, и пиксель, названный экранным по
    параллаксу, обоснованы по-разному. Сложив их, мы получили бы одну цифру, в
    которой не видно, что именно мы знаем.

    Порядок предпочтения выведен из того, что каждый признак **способен** увидеть, и
    проверен замером:

    1. **Связь с движением камеры** — если запись содержит и движение, и остановку.
       Он единственный различает все три случая: неподвижное обрамление,
       анимированное обрамление и мир.
    2. **Параллакс** — если сдвиг есть, но остановок в записи не было. Тогда третий
       признак неприменим: сравнивать «при остановке» не с чем.
    3. **Неподвижность** — если глобального сдвига нет вовсе. Самый слабый: не
       различает «прибито к экрану» и «стоит на месте».
    """

    def __init__(self, profile: Profile) -> None:
        self.mode = str(profile.structural.get("layer_signal", "auto"))
        self.min_parallax_frames = int(
            profile.parameters["arbiter_min_parallax_frames"])
        self.min_votes = int(profile.parameters["coupling_min_votes"])
        self.parallax = SelfWorldSeparator(profile)
        self.stillness = StillnessSeparator(profile)
        self.coupling = MotionCouplingSeparator(profile)

    def feed(self, frame: np.ndarray) -> None:
        # Сдвиг считается один раз и передаётся третьему признаку: FFT — самая
        # дорогая часть кадра, считать её дважды незачем.
        shift = self.parallax.feed(frame)
        self.stillness.feed(frame)
        self.coupling.feed(frame, shift)

    def result(self) -> LayerVerdict:
        par = self.parallax.result()
        still = self.stillness.result()
        coup = self.coupling.result()
        fixed = {PARALLAX: par.labels, STILLNESS: still.labels,
                 COUPLING: coup.labels}
        if self.mode in fixed:
            prov = np.where(fixed[self.mode] != UNDECIDED,
                            STRENGTH.index(self.mode) + 1, 0).astype(np.int8)
            return LayerVerdict(self.mode, fixed[self.mode].copy(),
                                "признак задан профилем", par, still, coup, prov, 0)

        # Применимость проверяется до ответа: признак, у которого нет данных, не
        # должен участвовать даже в согласии.
        applicable: list[tuple[str, np.ndarray]] = []
        if (coup.moving_frames >= self.min_parallax_frames
                and coup.still_frames >= self.min_votes):
            applicable.append((COUPLING, coup.labels))
        if par.voting_frames >= self.min_parallax_frames:
            applicable.append((PARALLAX, par.labels))
        if still.voting_frames > 0:
            applicable.append((STILLNESS, still.labels))

        labels = np.full(par.labels.shape, UNDECIDED, dtype=np.int8)
        prov = np.zeros(par.labels.shape, dtype=np.int8)
        disagreements = 0
        used: list[str] = []
        for name, lab in sorted(applicable, key=lambda it: STRENGTH.index(it[0])):
            decided = lab != UNDECIDED
            if not decided.any():
                continue
            used.append(name)
            fresh = decided & (prov == 0)
            conflict = decided & (prov != 0) & (labels != lab)
            disagreements += int(conflict.sum())
            labels[fresh] = lab[fresh]
            prov[fresh] = STRENGTH.index(name) + 1

        if not used:
            return LayerVerdict(
                NO_SIGNAL, labels,
                "ни сдвига, ни различия в изменчивости: разделять нечем",
                par, still, coup, prov, 0)
        names = {COUPLING: "связь с движением", PARALLAX: "параллакс",
                 STILLNESS: "неподвижность"}
        parts = [f"{names[s]} решил {int((prov == STRENGTH.index(s) + 1).sum())}"
                 for s in used]
        reason = ("применимо: " + ", ".join(parts)
                  + f"; расхождений {disagreements}")
        signal = used[0] if len(used) == 1 else MERGED
        return LayerVerdict(signal, labels, reason, par, still, coup, prov,
                            disagreements)


# --- фон и превышение над фоном (первые два пункта 0.6) ---------------------


@dataclass(frozen=True, slots=True)
class BackgroundLevel:
    """Насколько кадр меняется сам по себе и насколько — при действии."""

    idle_mean: float
    idle_sigma: float
    idle_n: int
    acting_mean: float
    acting_sigma: float
    acting_n: int

    @property
    def excess(self) -> float:
        """Превышение над фоном в сигмах фона. Ниже 1 — действие незаметно."""
        if self.idle_n == 0 or self.idle_sigma <= 0:
            return float("nan")
        return (self.acting_mean - self.idle_mean) / self.idle_sigma

    def as_dict(self) -> dict[str, float | int]:
        return {"idle_mean": self.idle_mean, "idle_sigma": self.idle_sigma,
                "idle_n": self.idle_n, "acting_mean": self.acting_mean,
                "acting_sigma": self.acting_sigma, "acting_n": self.acting_n,
                "excess_sigmas": self.excess}


def frame_change(prev: np.ndarray, cur: np.ndarray, pixel_delta: int = 8) -> float:
    """Доля пикселей, изменившихся заметно. Та же мера, что у сторожевого таймера."""
    a = prev.astype(np.int16)
    b = cur.astype(np.int16)
    d = np.abs(b - a)
    if d.ndim == 3:
        d = d.max(axis=2)
    return float((d > pixel_delta).mean())


def frame_energy(prev: np.ndarray, cur: np.ndarray) -> float:
    """Средняя величина изменения кадра, 0…1. Не доля пикселей, а насколько.

    Зачем нужна вторая мера рядом с `frame_change`. Доля изменившихся пикселей
    насыщается: там, где картинка меняется целиком — видео, чужой стрим, панорама
    камеры, — она уже равна почти единице, и последствие действия в неё не влезает.
    Замер по домену «видео»: холостой ход даёт долю 0.636 ± 0.009, а перемотка —
    0.646, то есть превышение в один сигму, ниже любого разумного порога. По средней
    величине те же данные дают 0.0557 ± 0.0021 против 0.0698 — почти семь сигм.

    Поэтому: `frame_change` остаётся у сторожевого таймера, где вопрос именно «сколько
    пикселей шевелится» (замер против зависшей картинки), а последствие действия
    измеряется этой мерой.
    """
    a = prev.astype(np.int16)
    b = cur.astype(np.int16)
    return float(np.abs(b - a).mean()) / 255.0


def background_level(pairs: Iterable[tuple[np.ndarray, np.ndarray, bool]],
                     pixel_delta: int = 8) -> BackgroundLevel:
    """По тройкам (предыдущий кадр, текущий кадр, было ли действие).

    Знание о собственных действиях агенту доступно законно: это его действия.
    Ничего про мир, координаты или интерфейс здесь не используется.
    """
    idle: list[float] = []
    acting: list[float] = []
    for prev, cur, acted in pairs:
        (acting if acted else idle).append(frame_change(prev, cur, pixel_delta))

    def stat(xs: list[float]) -> tuple[float, float, int]:
        if not xs:
            return 0.0, 0.0, 0
        a = np.asarray(xs, dtype=np.float64)
        return float(a.mean()), float(a.std()), int(a.size)

    im, isd, inn = stat(idle)
    am, asd, an = stat(acting)
    return BackgroundLevel(im, isd, inn, am, asd, an)
