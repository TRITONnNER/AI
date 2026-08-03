"""Устройство ввода через uinput (Linux).

Скан-коды приходят снаружи резолвером — модуль не знает ни одного имени клавиши
и не импортирует `harness.debug` (инварианты 4 и 12).

Почему uinput, а не `pynput`: в CLAUDE.md прямо сказано, что многие игры
игнорируют виртуальные коды клавиш. uinput создаёт устройство на уровне ядра, и
для игры оно неотличимо от настоящей клавиатуры — события приходят со скан-кодами
через тот же путь, что от физического устройства.

Проверить это в контейнере нельзя: нет `/dev/uinput` и нет игры. Отказ поэтому
громкий и с указанием, что сделать. Что именно осталось непроверенным — в
`docs/ARCHITECTURE-HARNESS.md`, «Что не проверено».
"""

from __future__ import annotations

from typing import Callable, Iterable

from .base import InjectionUnavailable

EV_KEY = 0x01
EV_REL = 0x02
EV_SYN = 0x00
REL_X = 0x00
REL_Y = 0x01


class UinputDevice:
    """Клавиатура и мышь на уровне ядра.

    `resolve` — выход → скан-код (см. `harness.debug.keymap.Keymap.resolver`).
    `buttons` — какие выходы являются кнопками мыши: у них другой тип события.
    """

    name = "uinput"

    def __init__(self, resolve: Callable[[str], int], *,
                 buttons: Iterable[str] = (), device_name: str = "harness-input") -> None:
        self._resolve = resolve
        self._buttons = frozenset(buttons)
        self._device_name = device_name
        self._ui = None
        self._down: set[str] = set()

    def open(self) -> None:
        import os
        if not os.path.exists("/dev/uinput"):
            raise InjectionUnavailable(
                "нет /dev/uinput. Загрузите модуль: sudo modprobe uinput, "
                "и дайте доступ на запись (группа input или udev-правило). "
                "Без этого инъекция невозможна — и подменять её ничем нельзя"
            )
        if not os.access("/dev/uinput", os.W_OK):
            raise InjectionUnavailable(
                "/dev/uinput есть, но недоступен на запись. Добавьте пользователя в "
                "группу input или создайте udev-правило"
            )
        try:
            import evdev  # noqa: PLC0415
            from evdev import UInput, ecodes  # noqa: PLC0415
        except ImportError as e:
            raise InjectionUnavailable(
                "нет пакета evdev. Поставьте: pip install 'harness[linux]'"
            ) from e
        del evdev
        caps = {
            ecodes.EV_KEY: sorted(self._resolve_all()),
            ecodes.EV_REL: [ecodes.REL_X, ecodes.REL_Y],
        }
        self._ui = UInput(caps, name=self._device_name)

    def _resolve_all(self) -> set[int]:
        """Все коды, которые устройство обязано объявить в своих возможностях.

        Резолвер не перечисляется — он односторонний. Поэтому объявляем полный
        диапазон клавиш и три кнопки мыши: объявить лишнее безвредно, а не
        объявить нужное — значит потерять события молча.
        """
        return set(range(1, 128)) | {0x110, 0x111, 0x112}

    # --- DeviceSink ---------------------------------------------------------

    def key_down(self, output: str) -> None:
        self._emit(output, 1)
        self._down.add(output)

    def key_up(self, output: str) -> None:
        self._emit(output, 0)
        self._down.discard(output)

    def _emit(self, output: str, value: int) -> None:
        if self._ui is None:
            raise InjectionUnavailable("устройство не открыто: сначала open()")
        code = self._resolve(output)
        self._ui.write(EV_KEY, code, value)
        self._ui.write(EV_SYN, 0, 0)
        self._ui.syn()

    def mouse_move(self, dx: int, dy: int) -> None:
        if self._ui is None:
            raise InjectionUnavailable("устройство не открыто: сначала open()")
        if dx:
            self._ui.write(EV_REL, REL_X, int(dx))
        if dy:
            self._ui.write(EV_REL, REL_Y, int(dy))
        self._ui.syn()

    def close(self) -> None:
        # Отпустить всё зажатое обязательно: иначе после падения останется
        # клавиша, зажатая на уровне ядра, и игра будет идти вперёд сама.
        for output in list(self._down):
            try:
                self.key_up(output)
            except Exception:
                pass
        if self._ui is not None:
            self._ui.close()
            self._ui = None


class WindowsSendInput:
    """SendInput со скан-кодами. Не реализован."""

    name = "sendinput"

    def __init__(self, *_: object, **__: object) -> None:
        pass

    def _fail(self) -> None:
        raise InjectionUnavailable(
            "инъекция через SendInput не реализована. Нужен Windows и живая игра: "
            "проверять надо именно то, что событие доходит со скан-кодом "
            "(KEYEVENTF_SCANCODE), а виртуальные коды игры игнорируют. "
            "Непроверенный backend инъекции опаснее отсутствующего"
        )

    def key_down(self, output: str) -> None:
        self._fail()

    def key_up(self, output: str) -> None:
        self._fail()

    def mouse_move(self, dx: int, dy: int) -> None:
        self._fail()

    def close(self) -> None:
        pass
