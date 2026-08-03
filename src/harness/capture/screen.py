"""Захват экрана: реальные backend'ы.

Проверить их в этом контейнере невозможно — здесь нет ни дисплея, ни звукового
устройства. Поэтому код написан так, чтобы отсутствие условий обнаруживалось при
`start()` и заканчивалось внятным отказом, а не чёрными кадрами. То, что
непроверено на живой машине, помечено в
`docs/ARCHITECTURE-HARNESS.md`, раздел «Что не проверено».
"""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .base import AudioBlock, BackendUnavailable, Frame, to_gray


class ScreenCapture:
    """Захват области экрана через mss (X11) — Linux.

    В CLAUDE.md для Windows предложен dxcam (Desktop Duplication), для macOS —
    ScreenCaptureKit. Ниже отдельные классы; каждый честно сообщает, что он
    не реализован, вместо того чтобы делать вид, что работает.
    """

    name = "screen_mss"

    def __init__(self, region: tuple[int, int, int, int] | None = None, *,
                 gray: bool = True) -> None:
        self.region = region      # (left, top, width, height); None — весь экран
        self.gray = gray
        self._sct = None
        self._monitor = None
        self._t_world = 0

    def start(self) -> None:
        try:
            import mss  # noqa: PLC0415 — импорт по требованию: это платформенная зависимость
        except ImportError as e:
            raise BackendUnavailable(
                "нет пакета mss. Поставьте: pip install 'harness[linux]' "
                "или pip install mss"
            ) from e
        import os
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            raise BackendUnavailable(
                "нет ни DISPLAY, ни WAYLAND_DISPLAY: захватывать нечего. "
                "Запускайте на машине с графической сессией"
            )
        self._sct = mss.mss()
        if self.region is None:
            self._monitor = self._sct.monitors[1]
        else:
            left, top, width, height = self.region
            self._monitor = {"left": left, "top": top, "width": width, "height": height}

    def stop(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def read(self) -> Frame | None:
        if self._sct is None:
            raise BackendUnavailable("захват не запущен: сначала start()")
        raw = self._sct.grab(self._monitor)
        # mss отдаёт BGRA; порядок каналов важен, поэтому переворачиваем явно
        arr = np.asarray(raw, dtype=np.uint8)[:, :, :3][:, :, ::-1]
        image = to_gray(arr) if self.gray else np.ascontiguousarray(arr)
        frame = Frame(image, self._t_world, time.monotonic_ns())
        self._t_world += 1
        return frame

    def frames(self) -> Iterator[Frame]:
        while True:
            f = self.read()
            if f is None:
                return
            yield f


class WindowsScreenCapture:
    """Desktop Duplication через dxcam. Не реализован и об этом сообщает."""

    name = "screen_dxcam"

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def start(self) -> None:
        raise BackendUnavailable(
            "захват через dxcam не реализован. Нужен Windows и работа на живой "
            "машине: без неё нельзя проверить ни частоту кадров, ни то, что "
            "Desktop Duplication не отдаёт устаревшие кадры. "
            "Писать этот backend вслепую — значит получить незаметно битый корпус"
        )

    def stop(self) -> None:
        pass

    def read(self) -> Frame | None:
        self.start()
        return None

    def frames(self) -> Iterator[Frame]:
        self.start()
        return iter(())


class MacScreenCapture(WindowsScreenCapture):
    """ScreenCaptureKit. Не реализован."""

    name = "screen_sck"

    def start(self) -> None:
        raise BackendUnavailable(
            "захват через ScreenCaptureKit не реализован: нужен macOS и живая машина"
        )


class LoopbackAudio:
    """Петлевой захват звука. Стерео обязательно: из разницы каналов пеленг."""

    name = "audio_loopback"

    def __init__(self, rate: int = 48000, channels: int = 2, block_ms: float = 20.0,
                 device: object | None = None) -> None:
        if channels != 2:
            raise ValueError(
                f"channels={channels}: моно ломает пеленг и потому не принимается "
                "(структурный переключатель audio_channels)"
            )
        self.rate = rate
        self.channels = channels
        self.block = max(1, int(rate * block_ms / 1000.0))
        self.device = device
        self._stream = None

    def start(self) -> None:
        try:
            import sounddevice as sd  # noqa: PLC0415
        except ImportError as e:
            raise BackendUnavailable(
                "нет пакета sounddevice. Поставьте: pip install sounddevice"
            ) from e
        try:
            self._stream = sd.InputStream(samplerate=self.rate, channels=self.channels,
                                          dtype="int16", blocksize=self.block,
                                          device=self.device)
            self._stream.start()
        except Exception as e:  # sounddevice бросает свои типы; наружу отдаём один
            raise BackendUnavailable(
                f"не открылся петлевой вход ({e}). Нужен monitor-источник PulseAudio "
                "или WASAPI loopback; обычный микрофон не подходит — он не пишет то, "
                "что слышит агент"
            ) from e

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self) -> AudioBlock | None:
        if self._stream is None:
            raise BackendUnavailable("звук не запущен: сначала start()")
        data, overflowed = self._stream.read(self.block)
        block = AudioBlock(np.asarray(data, dtype=np.int16), self.rate, time.monotonic_ns())
        if overflowed:
            # Не глотаем: пропуск звука — это разрыв синхронизации, и он обязан
            # попасть в журнал записью CAPTURE_GAP.
            raise AudioOverflow(block)
        return block


class AudioOverflow(RuntimeError):
    """Звуковой буфер переполнился: часть звука потеряна.

    Несёт блок, который всё-таки прочитан, чтобы вызывающий записал и разрыв,
    и то, что уцелело.
    """

    def __init__(self, block: AudioBlock) -> None:
        super().__init__("переполнение звукового буфера: часть блока потеряна")
        self.block = block
