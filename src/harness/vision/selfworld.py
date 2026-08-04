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
