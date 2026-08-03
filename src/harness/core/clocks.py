"""Три часов.

Инвариант 2: каждая запись журнала штампуется хешем профиля и тремя часами.
Здесь только сами часы; штамповку обеспечивает журнал, который без штампа
записать нельзя физически.

- `t_self`   — циклы восприятия агента. Монотонно растёт, шаг ровно 1 за цикл.
              Это единственные часы, которые принадлежат агенту.
- `t_world`  — тики мира. В вехе 0 это счётчик кадров захвата (так и написано
              в MILESTONE-0-HARNESS.md, 0.1), позже — тики самого мира.
- `t_content` — время внутри просматриваемого содержимого: позиция в видео,
              в записи, в чужом стриме. Законно равно None, когда содержимого
              нет. None — это значение, а не ошибка.

Настенного времени здесь нет намеренно. Оно не принадлежит ни одному из трёх
контуров и в журнал не попадает: сравнивать записи по нему бессмысленно, потому
что скорость мира и скорость агента меняются независимо.
"""

from __future__ import annotations

from dataclasses import dataclass


class ClockError(RuntimeError):
    """Часы попытались пойти назад или прыгнуть."""


@dataclass(frozen=True, slots=True)
class Stamp:
    """Отметка трёх часов. Неизменяема: запись журнала переписать нельзя."""

    t_self: int
    t_world: int
    t_content: float | None = None

    def __post_init__(self) -> None:
        if self.t_self < 0:
            raise ClockError(f"t_self отрицательный: {self.t_self}")
        if self.t_world < 0:
            raise ClockError(f"t_world отрицательный: {self.t_world}")
        if self.t_content is not None and self.t_content < 0:
            raise ClockError(f"t_content отрицательный: {self.t_content}")

    def as_dict(self) -> dict[str, int | float | None]:
        return {"t_self": self.t_self, "t_world": self.t_world, "t_content": self.t_content}

    @classmethod
    def from_dict(cls, d: dict) -> Stamp:
        missing = {"t_self", "t_world", "t_content"} - set(d)
        if missing:
            raise ClockError(f"в отметке нет часов: {sorted(missing)}")
        return cls(int(d["t_self"]), int(d["t_world"]), _opt_float(d["t_content"]))

    def __str__(self) -> str:
        c = "—" if self.t_content is None else f"{self.t_content:.3f}"
        return f"self={self.t_self} world={self.t_world} content={c}"


def _opt_float(v: object) -> float | None:
    return None if v is None else float(v)


class Clocks:
    """Держит три счётчика и выдаёт отметки.

    Ни один счётчик нельзя уменьшить: журнал дозаписывается, а значит время в
    нём монотонно. Попытка отмотать — ошибка, а не тихая коррекция.
    """

    __slots__ = ("_t_self", "_t_world", "_t_content")

    def __init__(self, t_self: int = 0, t_world: int = 0, t_content: float | None = None) -> None:
        self._t_self = int(t_self)
        self._t_world = int(t_world)
        self._t_content = _opt_float(t_content)
        Stamp(self._t_self, self._t_world, self._t_content)  # проверка на отрицательные

    @property
    def t_self(self) -> int:
        return self._t_self

    @property
    def t_world(self) -> int:
        return self._t_world

    @property
    def t_content(self) -> float | None:
        return self._t_content

    def tick_self(self) -> int:
        """Один цикл восприятия агента."""
        self._t_self += 1
        return self._t_self

    def set_world(self, t_world: int) -> int:
        """Тики мира приходят извне (кадры, тики сервера) и только вперёд."""
        t_world = int(t_world)
        if t_world < self._t_world:
            raise ClockError(f"t_world назад: {self._t_world} → {t_world}")
        self._t_world = t_world
        return self._t_world

    def set_content(self, t_content: float | None) -> float | None:
        """Время содержимого. Может прыгать назад — это перемотка видео, а не
        ход времени, — но обязано быть неотрицательным. None означает, что
        содержимого сейчас нет."""
        self._t_content = _opt_float(t_content)
        if self._t_content is not None and self._t_content < 0:
            raise ClockError(f"t_content отрицательный: {self._t_content}")
        return self._t_content

    def stamp(self) -> Stamp:
        return Stamp(self._t_self, self._t_world, self._t_content)
