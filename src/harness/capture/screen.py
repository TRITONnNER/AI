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

from .base import (UNCHANGED, AudioBlock, BackendUnavailable, Frame, Unchanged,
                   to_gray)


class ScreenCapture:
    """Захват области экрана через mss. Работает на X11, Windows и macOS.

    В CLAUDE.md для Windows предложен dxcam (Desktop Duplication), для macOS —
    ScreenCaptureKit. Они быстрее и остаются отдельными классами, каждый из которых
    честно сообщает, что не реализован. Но mss кроссплатформенна, и раньше её здесь
    запирал на Linux не механизм, а проверка переменных окружения: `DISPLAY` на
    Windows не выставлен никогда. Из-за одной строки проверки оператор с Windows или
    macOS не мог записать ничего, хотя рабочий путь был на месте.

    **Wayland отказывается, а не пробуется.** Под Wayland mss либо видит только окна
    XWayland, либо отдаёт чёрный кадр, и второе выглядит как успешная запись. Разница
    между «отказался» и «записал мусор» — это разница между потерянной минутой и
    потерянным днём разбирательств, почему живые числа не похожи ни на что.
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
        from ..machine import Session, detect, install_capture, switch_to_x11

        machine = detect()
        if machine.session is Session.NONE:
            raise BackendUnavailable(
                "графической сессии нет: захватывать нечего "
                f"({machine.session_source}). Запускайте на машине с экраном; "
                "по ssh без проброса X это не работает")
        if machine.session is Session.WAYLAND:
            raise BackendUnavailable(
                "сеанс Wayland: захвата под него в проекте нет. Нужен PipeWire с "
                "портальным разрешением, и этот backend не написан. mss под Wayland "
                "отдала бы чёрный кадр или только окна XWayland — то есть запись, "
                "неотличимую по формату от настоящей и мусорную по содержанию. "
                f"Что делать: {switch_to_x11()}")
        try:
            import mss  # noqa: PLC0415 — импорт по требованию: это платформенная зависимость
        except ImportError as e:
            raise BackendUnavailable(
                f"нет пакета mss. Поставьте: {install_capture(machine)}"
            ) from e
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
    """Desktop Duplication через dxcam — предпочтительный путь на Windows.

    Раньше здесь стоял честный отказ: backend нельзя было написать вслепую, не
    проверив ни частоту, ни то, что Desktop Duplication не отдаёт устаревшие кадры.
    Проверка состоялась на машине оператора — `dxcam.create()` и `grab()` отдали
    полный кадр `(1080, 1920, 3)` с первой попытки, — и отказ заменён работой.

    ## Почему он предпочтительнее mss

    mss на Windows идёт через GDI BitBlt: высокая нагрузка на процессор, реальная
    частота ниже заявленной и — главное — **полноэкранные DirectX-приложения не
    захватываются**. Игра отдаст рабочий стол или чёрный кадр вместо себя, причём
    молча: кадры будут, содержимое будет не то. Замер на 1080p через mss дал
    19.9 кадр/с при заявленных 30 — то есть критерий М0 (устойчивые 30+) через mss
    недостижим, и это измерено, а не предположено.

    ## `grab()` вернул `None` — это не сбой

    Неизменённый кадр Desktop Duplication не выдаёт вовсе. Здесь это превращается
    в `UNCHANGED`, а не в `None`: см. `capture.base.Unchanged`. Разница между ними —
    разница между «запись неподвижности прошла идеально» и «потеряно 1700 кадров».
    """

    name = "screen_dxcam"

    def __init__(self, region: tuple[int, int, int, int] | None = None, *,
                 gray: bool = True, device: int = 0, output: int | None = None) -> None:
        self.region = region
        self.gray = gray
        self.device = device
        self.output = output
        self._camera = None
        self._t_world = 0
        self.unchanged = 0          # сколько раз экран не менялся: это число нужно

    def start(self) -> None:
        try:
            import dxcam  # noqa: PLC0415 — платформенная зависимость
        except ImportError as e:
            raise BackendUnavailable(
                "нет пакета dxcam. Поставьте: py -m pip install dxcam"
            ) from e
        try:
            self._camera = dxcam.create(device_idx=self.device,
                                        output_idx=self.output,
                                        output_color="RGB")
        except Exception as e:      # dxcam бросает свои типы
            raise BackendUnavailable(
                f"dxcam не инициализировался ({e}). Обычные причины: нет адаптера "
                "DirectX 11, устаревший драйвер видеокарты, запуск в сеансе без "
                "рабочего стола (служба, ssh). Запасной путь — mss, он медленнее и "
                "не видит полноэкранные игры") from e
        if self._camera is None:
            raise BackendUnavailable(
                "dxcam.create() вернул None: устройство или выход не найдены. "
                f"Пробовали устройство {self.device}, выход {self.output}")

    def stop(self) -> None:
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                # Освобождение не удалось — не повод падать на выходе: запись уже
                # сделана, и терять её из-за уборки нельзя.
                pass
            self._camera = None

    def read(self) -> "Frame | Unchanged | None":
        if self._camera is None:
            raise BackendUnavailable("захват не запущен: сначала start()")
        raw = self._camera.grab(region=self.region)
        if raw is None:
            # Экран не менялся. Данные, а не пропуск: время всё равно идёт.
            self.unchanged += 1
            self._t_world += 1
            return UNCHANGED
        arr = np.asarray(raw, dtype=np.uint8)
        image = to_gray(arr) if self.gray else np.ascontiguousarray(arr)
        frame = Frame(image, self._t_world, time.monotonic_ns())
        self._t_world += 1
        return frame

    def frames(self) -> Iterator[Frame]:
        while True:
            f = self.read()
            if f is None:
                return
            if f is UNCHANGED:
                continue
            yield f                                        # type: ignore[misc]


class MacScreenCapture:
    """ScreenCaptureKit. Не реализован, и об этом сообщает.

    Наследоваться от dxcam-версии больше нельзя: та стала рабочей, и наследник тихо
    получил бы работающий `read()` при неработающем `start()`.
    """

    name = "screen_sck"

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def start(self) -> None:
        raise BackendUnavailable(
            "захват через ScreenCaptureKit не реализован: нужен macOS и живая "
            "машина, чтобы проверить и частоту, и то, что разрешение на запись "
            "экрана действительно выдано. На macOS работает mss — медленнее, но "
            "проверяемо")

    def stop(self) -> None:
        pass

    def read(self) -> "Frame | Unchanged | None":
        self.start()
        return None

    def frames(self) -> Iterator[Frame]:
        self.start()
        return iter(())


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

    @property
    def pending(self) -> int:
        """Сколько отсчётов уже лежит в буфере и ждёт чтения.

        Нужно затем, чтобы читать звук **в темпе кадров, не отставая**: `read()`
        блокирует до набора блока, и цикл, читающий по одному блоку на кадр, отстаёт
        от звука на разнице частот — 50 блоков в секунду против тридцати кадров, — и
        буфер переполняется. С этим числом вызывающий выгребает всё накопленное.
        """
        if self._stream is None:
            return 0
        try:
            return int(self._stream.read_available)
        except Exception:
            return 0

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
