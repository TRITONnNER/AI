"""Что агент получает. Больше ничего он не получает ни по какому каналу.

Инвариант 4: агент не получает координат, названий клавиш, меток предметов,
разметки интерфейса, счёта, флагов враждебности и нерасшифрованного текста.
Инвариант 5: весь текст с экрана уже заменён символами.

Поэтому `Percept` содержит ровно четыре вещи: пиксели, два канала звука, три
часов и список найденных на экране символов. Ни одно поле не несёт смысла:
символ — непрозрачный, пиксели — сырые, часы — счётчики.

Разметки интерфейса здесь нет намеренно, хотя харнесс её вычисляет (0.6). Знание
«вот это интерфейс» агент обязан добыть сам из тех же пикселей; результат 0.6 —
инструмент исследователя для проверки, а не подсказка. Если однажды его решат
подавать агенту, это будет структурный переключатель с форком журнала, а не
тихое добавление поля.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.clocks import Stamp
from ..core.symbols import assert_no_plain_text, is_symbol


class PerceptError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Percept:
    """Один цикл восприятия."""

    stamp: Stamp
    pixels: np.ndarray                       # (h, w) или (h, w, 3), uint8
    audio: np.ndarray | None = None          # (n, 2) int16; стерео обязательно
    symbols: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.stamp, Stamp):
            raise PerceptError("восприятие без трёх часов не бывает")
        if self.pixels.dtype != np.uint8:
            raise PerceptError(f"пиксели должны быть uint8, пришло {self.pixels.dtype}")
        if self.pixels.ndim not in (2, 3):
            raise PerceptError(f"пиксели должны быть (h,w) или (h,w,c), пришло {self.pixels.shape}")
        if self.audio is not None:
            if self.audio.ndim != 2 or self.audio.shape[1] != 2:
                raise PerceptError(
                    f"звук должен быть (n, 2): пеленг считается из разницы каналов, "
                    f"пришло {self.audio.shape}")
            if self.audio.dtype != np.int16:
                raise PerceptError(f"звук должен быть int16, пришло {self.audio.dtype}")
        for s in self.symbols:
            if not is_symbol(s):
                raise PerceptError(
                    f"{s!r} не символ. Надписи заменяются до передачи агенту "
                    "(инвариант 5), читаемый текст сюда не доходит")
        object.__setattr__(self, "symbols", tuple(self.symbols))

    def assert_opaque(self) -> None:
        """Проверка границы: в восприятии нет ничего читаемого.

        Вызывается на границе восприятия и в тестах инварианта 5. Пиксели
        пропускаются: сырой кадр — законный канал, его смысл агент выводит сам.
        """
        assert_no_plain_text({"symbols": list(self.symbols)}, path="percept")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.pixels.shape)
