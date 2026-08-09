"""Источник кадров и звука: общий интерфейс и честные отказы.

Из правил работы: «никаких молчаливых заглушек. Функция, возвращающая
правдоподобное значение вместо реального, — худшее, что может здесь произойти:
она сломает эксперимент незаметно».

Поэтому backend, которого на этой машине нет, не возвращает чёрные кадры и не
подставляет синтетику. Он бросает `BackendUnavailable` с текстом, из которого
понятно, что именно поставить или включить. Синтетика существует, но берётся
только тогда, когда её попросили по имени.

Настенное время (`monotonic_ns`) на кадре есть, но в журнал как часы оно не
попадает: оно нужно ровно для диагностики пропусков и рассинхрона звука
(критерий 0.1 — не больше 50 мс). Часов в журнале три, и все три не настенные.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


class BackendUnavailable(RuntimeError):
    """Backend на этой машине не работает. Текст обязан говорить, что сделать."""


@dataclass(frozen=True, slots=True)
class Frame:
    image: np.ndarray          # (h, w) uint8 для gray8 или (h, w, 3) для rgb8
    t_world: int               # тик мира; в вехе 0 — номер кадра
    monotonic_ns: int          # только для диагностики, не часы

    def __post_init__(self) -> None:
        if self.image.dtype != np.uint8:
            raise ValueError(f"кадр должен быть uint8, пришёл {self.image.dtype}")
        if self.image.ndim not in (2, 3):
            raise ValueError(f"кадр должен быть (h,w) или (h,w,c), пришло {self.image.shape}")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.image.shape)


@dataclass(frozen=True, slots=True)
class AudioBlock:
    samples: np.ndarray        # (n, channels) int16
    rate: int
    monotonic_ns: int

    def __post_init__(self) -> None:
        if self.samples.dtype != np.int16:
            raise ValueError(f"звук должен быть int16, пришёл {self.samples.dtype}")
        if self.samples.ndim != 2:
            raise ValueError(f"звук должен быть (n, channels), пришло {self.samples.shape}")

    @property
    def duration_ms(self) -> float:
        return 1000.0 * self.samples.shape[0] / self.rate


@runtime_checkable
class CaptureSource(Protocol):
    """Источник кадров. Реализации: синтетика, воспроизведение записи, экран."""

    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self) -> Frame | None: ...
    def frames(self) -> Iterator[Frame]: ...


@runtime_checkable
class AudioSource(Protocol):
    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self) -> AudioBlock | None: ...


def to_gray(image: np.ndarray) -> np.ndarray:
    """В серый по яркости. Коэффициенты BT.601 — те же, что у всех остальных."""
    if image.ndim == 2:
        return image
    rgb = image[:, :, :3].astype(np.float32)
    y = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return np.clip(y, 0, 255).astype(np.uint8)


def describe_backends() -> dict[str, dict[str, object]]:
    """Что доступно на этой машине. Для CLI и для отчёта в журнал сессии."""
    import importlib.util
    import os

    def has(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    from ..machine import Session, detect

    system = platform.system()
    machine = detect()
    # Реестр отвечает по тому же знанию, что и сам backend (`harness.machine`).
    # Раньше здесь была своя копия проверки, и она уже расходилась с backend'ом:
    # реестр считал Wayland годным, backend на нём отдавал бы чёрный кадр.
    display = machine.session is not Session.NONE
    return {
        "synthetic": {"available": True,
                      "why": "не требует ничего; помечает записи как синтетические"},
        "replay": {"available": True, "why": "читает записанную сессию с диска"},
        "screen_mss": {
            "available": machine.capture_backend == "screen_mss" and has("mss"),
            "why": f"нужны графическая сессия (не Wayland) и пакет mss "
                   f"(система {machine.os_name}, сессия {machine.session}, "
                   f"mss {'есть' if has('mss') else 'нет'})",
        },
        "screen_dxcam": {
            "available": system == "Windows" and has("dxcam"),
            "why": "нужны Windows + пакет dxcam (Desktop Duplication)",
        },
        "audio_loopback": {
            "available": has("sounddevice"),
            "why": "нужен пакет sounddevice с петлевым устройством "
                   "(WASAPI loopback на Windows, monitor-источник в PulseAudio)",
        },
        "inject_uinput": {
            "available": system == "Linux" and has("evdev") and os.path.exists("/dev/uinput"),
            "why": "нужны Linux + пакет evdev + доступ на запись в /dev/uinput",
        },
        "inject_sendinput": {
            "available": system == "Windows",
            "why": "SendInput со скан-кодами; виртуальные коды многие игры игнорируют",
        },
        "optical_flow": {
            "available": has("cv2"),
            "why": "нужен OpenCV (extras: harness[flow]) для плотного потока DIS; "
                   "без него дальность по потоку отказывается считаться, а не "
                   "считается чем-то другим молча",
        },
    }
