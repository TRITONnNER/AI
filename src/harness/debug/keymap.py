"""Соответствие выход ↔ скан-код ↔ настоящее имя клавиши.

Это отладочные данные: настоящие имена клавиш агенту недоступны (инвариант 4),
поэтому таблица живёт здесь, а не в `harness.inject`.

Важное следствие для архитектуры: устройственный backend **не импортирует** этот
модуль. Ему передают функцию-резолвер, собранную здесь исследователем при
настройке:

    from harness.debug.keymap import Keymap
    from harness.inject.uinput_device import UinputDevice
    km = Keymap.linux_evdev()
    device = UinputDevice(resolve=km.resolver())

Так `harness.inject` остаётся без пути к `harness.debug`, и тест на изоляцию
(инвариант 12) проходит не по договорённости, а по графу импортов.

Скан-коды здесь — линуксовые из `linux/input-event-codes.h`. Для Windows нужны
коды набора 1 (`MapVirtualKey`/`SendInput` со флагом `SCANCODE`); отдельная
таблица появится вместе с backend'ом, а выдумывать её вслепую нельзя — в
CLAUDE.md прямо сказано, что многие игры игнорируют виртуальные коды, и ошибка
здесь выглядит как «агент ничего не умеет».
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from ..core.action import output_id

# Подмножество, которого хватает для игры и браузера. Имена — для исследователя,
# агенту они не показываются ни в каком виде.
LINUX_EVDEV: dict[str, int] = {
    "ESC": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "9": 10,
    "0": 11, "MINUS": 12, "EQUAL": 13, "BACKSPACE": 14, "TAB": 15,
    "Q": 16, "W": 17, "E": 18, "R": 19, "T": 20, "Y": 21, "U": 22, "I": 23, "O": 24,
    "P": 25, "LEFTBRACE": 26, "RIGHTBRACE": 27, "ENTER": 28, "LEFTCTRL": 29,
    "A": 30, "S": 31, "D": 32, "F": 33, "G": 34, "H": 35, "J": 36, "K": 37, "L": 38,
    "SEMICOLON": 39, "APOSTROPHE": 40, "GRAVE": 41, "LEFTSHIFT": 42, "BACKSLASH": 43,
    "Z": 44, "X": 45, "C": 46, "V": 47, "B": 48, "N": 49, "M": 50, "COMMA": 51,
    "DOT": 52, "SLASH": 53, "RIGHTSHIFT": 54, "KPASTERISK": 55, "LEFTALT": 56,
    "SPACE": 57, "CAPSLOCK": 58,
    "F1": 59, "F2": 60, "F3": 61, "F4": 62, "F5": 63, "F6": 64, "F7": 65, "F8": 66,
    "F9": 67, "F10": 68, "F11": 87, "F12": 88,
    "RIGHTCTRL": 97, "RIGHTALT": 100,
    "UP": 103, "LEFT": 105, "RIGHT": 106, "DOWN": 108,
}

# Кнопки мыши: коды BTN_* из того же заголовка.
LINUX_BUTTONS: dict[str, int] = {"LEFT": 0x110, "RIGHT": 0x111, "MIDDLE": 0x112}

MODIFIERS = frozenset({"LEFTSHIFT", "RIGHTSHIFT", "LEFTCTRL", "RIGHTCTRL",
                       "LEFTALT", "RIGHTALT"})


@dataclass(frozen=True, slots=True)
class Keymap:
    """Таблица в обе стороны. Наружу отдаёт только резолвер выход → скан-код."""

    keys: Mapping[str, int]
    buttons: Mapping[str, int]

    @classmethod
    def linux_evdev(cls) -> Keymap:
        return cls(dict(LINUX_EVDEV), dict(LINUX_BUTTONS))

    def output_for_key(self, name: str) -> str:
        """Настоящее имя → непрозрачный выход. Для настройки маски исследователем."""
        if name not in self.keys:
            raise KeyError(f"нет такой клавиши в таблице: {name!r}")
        prefix = "MOD" if name in MODIFIERS else "OUT"
        return output_id(prefix, self.keys[name])

    def output_for_button(self, name: str) -> str:
        if name not in self.buttons:
            raise KeyError(f"нет такой кнопки мыши: {name!r}")
        return output_id("BTN", self.buttons[name])

    def outputs_for_keys(self, names: Iterable[str]) -> list[str]:
        return [self.output_for_key(n) for n in names]

    def name_for_output(self, output: str) -> str | None:
        """Расшифровка для колонки «Что это на самом деле» в пульте."""
        for name in self.keys:
            if self.output_for_key(name) == output:
                return name
        for name in self.buttons:
            if self.output_for_button(name) == output:
                return f"MOUSE_{name}"
        return None

    def table(self) -> list[dict[str, object]]:
        """Вся таблица для отладочного потока: выход, скан-код, настоящее имя."""
        rows: list[dict[str, object]] = []
        for name, code in sorted(self.keys.items(), key=lambda kv: kv[1]):
            rows.append({"output": self.output_for_key(name), "scancode": code,
                         "name": name, "kind": "key",
                         "modifier": name in MODIFIERS})
        for name, code in sorted(self.buttons.items(), key=lambda kv: kv[1]):
            rows.append({"output": self.output_for_button(name), "scancode": code,
                         "name": f"MOUSE_{name}", "kind": "button", "modifier": False})
        return rows

    def total_outputs(self) -> int:
        """Сколько всего выходов у тела. В пульте это «из 512 выходов проверено»."""
        return len(self.keys) + len(self.buttons)

    def resolver(self) -> Callable[[str], int]:
        """Функция, которую отдают устройственному backend'у.

        Backend получает только её: перечислить таблицу, узнать имя клавиши или
        пройти в обратную сторону через неё нельзя.
        """
        table = {self.output_for_key(n): c for n, c in self.keys.items()}
        table |= {self.output_for_button(n): c for n, c in self.buttons.items()}

        def resolve(output: str) -> int:
            try:
                return table[output]
            except KeyError:
                raise KeyError(
                    f"выход {output!r} не отображается ни на один скан-код. "
                    "Либо он не из этой раскладки, либо действие собрано мимо тела"
                ) from None

        return resolve

    def button_outputs(self) -> frozenset[str]:
        """Какие выходы — кнопки мыши. Нужно backend'у: коды кнопок и клавиш
        уходят разными типами событий."""
        return frozenset(self.output_for_button(n) for n in self.buttons)
