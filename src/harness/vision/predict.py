"""Ошибка предсказания — единая валюта.

Из архитектуры: «единая валюта — ошибка предсказания. Она обновляет карту,
направляет внимание и решает, что запоминать». Значит она должна быть одна, а не
три разных числа в трёх модулях, и должна считаться из того, что агенту доступно:
из кадров и своих действий.

Предсказатель здесь нарочно простой — постоянная скорость: если мир ехал влево,
он и дальше поедет влево с той же скоростью. Это не «пока сойдёт», а осмысленная
опорная точка: любая будущая обученная модель обязана быть лучше постоянной
скорости, иначе она не выучила ничего. И считается она за одно FFT, то есть
годится для контура на 20 Гц.

Что считается ошибкой: средняя по кадру абсолютная разница между предсказанным и
случившимся, приведённая к доле от полного диапазона яркости. Не среднеквадратичная:
квадрат отдаёт всю ошибку нескольким ярким пикселям, а нам важнее, что ошиблись
широко, чем что ошиблись сильно в одной точке.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.profile import Profile
from .selfworld import Shift, estimate_global_shift


@runtime_checkable
class Predictor(Protocol):
    name: str

    def predict(self) -> np.ndarray | None: ...
    def observe(self, frame: np.ndarray) -> None: ...


class CopyPredictor:
    """«Дальше будет то же». Опорный уровень, ниже которого падать некуда."""

    name = "copy"

    def __init__(self) -> None:
        self._last: np.ndarray | None = None

    def predict(self) -> np.ndarray | None:
        return None if self._last is None else self._last

    def observe(self, frame: np.ndarray) -> None:
        self._last = frame


class ShiftPredictor:
    """Постоянная скорость: сдвинуть последний кадр на тот же вектор, что и раньше.

    Экранный слой при этом предсказывается неверно — он ведь не двигается, — и
    это правильно: ошибка предсказания на интерфейсе и есть тот признак, из
    которого потом вырастает разделение слоёв. Единая валюта не должна знать про
    слои заранее.
    """

    name = "shift"

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None
        self._last: np.ndarray | None = None
        self._shift: Shift | None = None

    def predict(self) -> np.ndarray | None:
        if self._last is None:
            return None
        if self._shift is None:
            return self._last
        return _shift_frame(self._last, self._shift.dy, self._shift.dx)

    def observe(self, frame: np.ndarray) -> None:
        if self._last is not None:
            self._shift = estimate_global_shift(self._last, frame)
        self._prev, self._last = self._last, frame

    @property
    def velocity(self) -> Shift | None:
        return self._shift


def _shift_frame(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Сдвинуть кадр, оставив на освободившемся краю то, что там было.

    Дополнять нулями было бы хуже: чёрная полоса даст огромную ошибку на краю и
    заглушит настоящую ошибку в середине. Повтор края — не выдумка данных, а
    честное «про эту область я не знаю ничего нового».
    """
    out = np.empty_like(frame)
    h, w = frame.shape[:2]
    ys_src = slice(max(0, -dy), min(h, h - dy))
    ys_dst = slice(max(0, dy), min(h, h + dy))
    xs_src = slice(max(0, -dx), min(w, w - dx))
    xs_dst = slice(max(0, dx), min(w, w + dx))
    out[...] = frame
    out[ys_dst, xs_dst] = frame[ys_src, xs_src]
    return out


def make_predictor(profile: Profile) -> Predictor:
    kind = str(profile.structural["predictor"])
    if kind == "copy":
        return CopyPredictor()
    if kind == "shift":
        return ShiftPredictor()
    raise ValueError(f"нет предсказателя {kind!r}")


@dataclass(frozen=True, slots=True)
class ErrorSample:
    """Ошибка на одном кадре и её место в истории."""

    value: float            # доля от полного диапазона яркости, 0…1
    smoothed: float         # сглаженная по прошлому
    sigma: float            # разброс сглаженной
    is_spike: bool          # превышение больше `spike_sigmas` сигм
    n: int                  # сколько кадров уже видели

    @property
    def excess_sigmas(self) -> float:
        if self.sigma <= 0:
            return 0.0
        return (self.value - self.smoothed) / self.sigma

    def as_dict(self) -> dict[str, float | bool | int]:
        return {"value": round(self.value, 6), "smoothed": round(self.smoothed, 6),
                "sigma": round(self.sigma, 6), "is_spike": self.is_spike,
                "excess_sigmas": round(self.excess_sigmas, 3), "n": self.n}


class PredictionError:
    """Считает ошибку, сглаживает её и отмечает всплески.

    Всплеск — это событие, а не просто большое число: он значит «мир повёл себя
    не так, как я думал» и годится как повод запомнить эпизод, обновить карту и
    перенаправить внимание. Порог во сигмах, а не в абсолютной величине: у
    спокойной сцены и у боя разный нормальный уровень.
    """

    def __init__(self, profile: Profile, *, predictor: Predictor | None = None) -> None:
        p = profile.parameters
        self.tau = float(p["prediction_error_tau"])
        self.spike_sigmas = float(p["spike_sigmas"])
        self.min_sigma = float(p["spike_min_sigma"])
        self.novelty_threshold = float(p["novelty_threshold"])
        self.predictor = predictor or make_predictor(profile)
        self._mean = 0.0
        self._var = 0.0
        self._n = 0
        self._recent: deque[float] = deque(maxlen=512)
        self.spikes: list[int] = []

    def feed(self, frame: np.ndarray) -> ErrorSample | None:
        """Дать кадр. `None`, пока предсказывать нечем — первого кадра мало."""
        gray = frame if frame.ndim == 2 else frame[:, :, :3].mean(axis=2).astype(np.uint8)
        predicted = self.predictor.predict()
        self.predictor.observe(gray)
        if predicted is None or predicted.shape != gray.shape:
            return None

        value = float(np.abs(gray.astype(np.int16)
                             - predicted.astype(np.int16)).mean()) / 255.0
        self._n += 1
        self._recent.append(value)

        # Сглаживание по экспоненте: постоянная задана в кадрах, как и все прочие
        # времена агента — в его собственных циклах, а не в секундах стены.
        k = 1.0 / max(1.0, self.tau)
        if self._n == 1:
            self._mean, self._var = value, 0.0
            return ErrorSample(value, value, 0.0, False, self._n)

        # Всплеск определяется до обновления фона, по фону *до* этого кадра.
        # Иначе получается замкнутый круг: всплески раздувают дисперсию, дисперсия
        # поднимает порог, порог перестаёт ловить всплески — и детектор молчит
        # именно тогда, когда должен срабатывать.
        sigma = self._var ** 0.5
        # Пол разброса: без него побитово одинаковые кадры дают sigma == 0, и
        # всплеск не обнаружим ни при каком пороге. А цифровой захват как раз
        # даёт одинаковые кадры, когда на экране ничего не меняется.
        effective = max(sigma, self.min_sigma)
        is_spike = (self._n > int(self.tau)
                    and (value - self._mean) > self.spike_sigmas * effective)

        # Всплеск почти не двигает фон: фон обязан описывать обычный уровень, а
        # не подстраиваться под удивление. Полностью игнорировать тоже нельзя —
        # тогда переход в новый режим (вошли в бой) никогда не станет нормой.
        weight = k * 0.1 if is_spike else k
        delta = value - self._mean
        self._mean += weight * delta
        self._var = (1 - weight) * (self._var + weight * delta * delta)

        if is_spike:
            self.spikes.append(self._n)
        return ErrorSample(value, self._mean, sigma, is_spike, self._n)

    # --- то, что из ошибки следует -----------------------------------------

    def is_novel(self, sample: ErrorSample) -> bool:
        """Достаточно ли необычно, чтобы считать вид новым (порог из профиля)."""
        return sample.value >= self.novelty_threshold

    def worth_remembering(self, sample: ErrorSample) -> bool:
        """Стоит ли запоминать эпизод. Единая валюта решает и это.

        Запоминается либо всплеск (мир удивил), либо необычно спокойный момент
        при высоком фоне — он тоже информативен: значит что-то стабилизировалось.
        """
        if sample.is_spike:
            return True
        if sample.sigma <= 0:
            return False
        return sample.excess_sigmas < -self.spike_sigmas

    def summary(self) -> dict[str, float | int]:
        recent = list(self._recent)
        return {"n": self._n, "mean": round(self._mean, 6),
                "sigma": round(self._var ** 0.5, 6),
                "spikes": len(self.spikes),
                "last": round(recent[-1], 6) if recent else 0.0,
                "predictor": self.predictor.name}
