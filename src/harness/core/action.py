"""Действие.

Инвариант 8: действие — это `(key, duration_ms, modifiers)`, а не дискретное
событие. Здесь это заложено в тип: конструктор без длительности не существует,
поэтому «нажать W» невыразимо, а «W на 340 мс» выразимо.

Инвариант 9: у каждого действия есть оценка обратимости. Осторожность
определяется как «я не умею это откатить», а не списком запретов. Поэтому
`Reversibility` — обязательное поле, а незнание представлено явно
(`n == 0`) и трактуется как максимальная опасность, а не как «наверное можно».

Инвариант 4: агент не получает названий клавиш. Поэтому действие адресует
**выход** непрозрачным идентификатором (`OUT_2C`), а соответствие выход →
скан-код живёт только в драйвере ввода вместе с отладочным потоком. В этом
модуле слова «W», «Space», «Escape» не встречаются и встречаться не должны.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence

OUTPUT_RE = re.compile(r"^(OUT|MOD|BTN)_[0-9A-F]{2,4}$")

# Ключ действия: выход вместе с удержанием и модификаторами. Одна форма на весь
# проект, потому что «одно и то же нажатие» — это нажатие той же длительности:
# действие невыразимо без длительности (инвариант 8). Ребро графа мест, шаг навыка и
# переход модели пользуются этой формой, иначе они говорили бы о разном.
ACTION_KEY_RE = re.compile(
    r"^(?:(?P<mods>(?:OUT|MOD|BTN)_[0-9A-F]{2,4}(?:\+(?:OUT|MOD|BTN)_[0-9A-F]{2,4})*)>)?"
    r"(?P<out>(?:OUT|MOD|BTN)_[0-9A-F]{2,4})@(?P<ms>\d+)$")


def action_key(output: str, duration_ms: int, modifiers: tuple[str, ...] = ()) -> str:
    """Ключ действия: `OUT_xx@200` или `MOD_yy>OUT_xx@200`."""
    mods = "+".join(modifiers)
    return f"{mods}>{output}@{int(duration_ms)}" if mods else f"{output}@{int(duration_ms)}"


def parse_action_key(key: str) -> tuple[str, int, tuple[str, ...]] | None:
    """Разобрать ключ действия. `None`, если это не ключ действия.

    `None`, а не догадка о длительности: подставить длительность «по умолчанию»
    значило бы предсказывать последствие другого действия. Нажатие на 40 мс и на 200
    мс — разные действия, и в замере это стоило целого плана: модель обещала место,
    наблюдённое при 200 мс, а планировщик жал 40 мс и попадал не туда.
    """
    m = ACTION_KEY_RE.match(key)
    if not m:
        return None
    mods = tuple(m.group("mods").split("+")) if m.group("mods") else ()
    return m.group("out"), int(m.group("ms")), mods


MACRO_SEP = "|"


def macro_key(keys: Sequence[str]) -> str:
    """Ключ макроса: ключи действий через `|`, в порядке нажатия.

    Форма та же, что у `Skill.signature()`, и это не совпадение: навык в журнале и
    шаг плана из навыка обязаны иметь один ключ, иначе статистика по ним не сойдётся.

    Отдельная функция, а не «просто join», нужна ради проверки: макрос из одного шага
    — это одиночное действие, и записывать его макросом значит завести второй ключ для
    того же самого. Тогда наблюдения разъедутся по двум ключам, и оба будут выглядеть
    менее подтверждёнными, чем есть.
    """
    if len(keys) < 2:
        raise ActionError(
            f"макрос из {len(keys)} шагов: это одиночное действие, и ключ у него "
            "уже есть. Два ключа на одно действие разведут его статистику надвое")
    for k in keys:
        if parse_action_key(k) is None:
            raise ActionError(f"в макросе не ключ действия: {k!r}")
    return MACRO_SEP.join(keys)


def parse_macro_key(key: str) -> tuple[str, ...] | None:
    """Разобрать ключ макроса. `None`, если это не макрос."""
    if MACRO_SEP not in key:
        return None
    parts = key.split(MACRO_SEP)
    if len(parts) < 2 or any(parse_action_key(p) is None for p in parts):
        return None
    return tuple(parts)


def parse_any_key(key: str) -> tuple[str, ...] | None:
    """Ключи действий в этом ключе: один для одиночного, несколько для макроса.

    `None` — не ключ вовсе. Нужна там, где всё равно, одиночное действие или макрос:
    модели перехода, например, важно только, что переход можно повторить.
    """
    macro = parse_macro_key(key)
    if macro is not None:
        return macro
    return (key,) if parse_action_key(key) is not None else None


class ActionError(ValueError):
    """Действие собрано так, что его нельзя ни выполнить, ни записать."""


class Kind(StrEnum):
    KEY = "key"                  # удержание выхода клавиатуры
    BUTTON = "button"            # удержание кнопки мыши
    MOUSE_MOVE = "mouse_move"    # относительное перемещение
    NOTHING = "nothing"          # осознанное бездействие: тоже действие и тоже пишется


def output_id(prefix: str, scancode: int) -> str:
    """Непрозрачный идентификатор выхода из скан-кода.

    Соль не нужна: идентификатор обязан быть одинаковым между запусками, иначе
    карта тела агента рассыпается при перезапуске. Скрывать надо не сам код, а
    его *значение* — название клавиши и её функцию, чего здесь нет.
    """
    h = hashlib.blake2b(str(int(scancode)).encode(), digest_size=2).hexdigest().upper()
    return f"{prefix}_{h}"


@dataclass(frozen=True, slots=True)
class Reversibility:
    """Оценка обратимости: μ, σ, n — как у любого убеждения.

    `mu` — доля случаев, когда откат удался. `n == 0` значит «не пробовал»,
    и тогда `mu` не определено: спрашивать надо `is_known`, а не сравнивать
    `mu` с порогом.
    """

    mu: float = 0.0
    sigma: float = 1.0
    n: int = 0

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ActionError(f"n отрицательное: {self.n}")
        if not 0.0 <= self.mu <= 1.0:
            raise ActionError(f"mu вне [0,1]: {self.mu}")
        if self.sigma < 0.0:
            raise ActionError(f"sigma отрицательная: {self.sigma}")
        if self.n == 0 and (self.mu != 0.0 or self.sigma != 1.0):
            raise ActionError(
                "n == 0 означает «не пробовал»: mu должно быть 0.0, sigma 1.0. "
                "Иначе незнание выглядит как знание"
            )

    @property
    def is_known(self) -> bool:
        return self.n > 0

    @property
    def caution(self) -> float:
        """Осторожность в [0,1]. Незнание — максимум.

        Не список запретов: значение выводится только из того, умеем ли мы
        откатывать. Ни одно имя клавиши и ни одна «опасная функция» на это
        значение не влияет.
        """
        if not self.is_known:
            return 1.0
        return max(0.0, min(1.0, (1.0 - self.mu) + self.sigma * (1.0 - self.mu)))

    def observe(self, undone: bool) -> Reversibility:
        """Ещё одна попытка откатить. Возвращает новую оценку, старую не портит."""
        n = self.n + 1
        mu = ((self.mu * self.n) + (1.0 if undone else 0.0)) / n
        # σ биномиальной доли; при n == 1 разброс максимальный, а не нулевой
        sigma = (max(mu * (1.0 - mu), 1e-9) / n) ** 0.5 if n > 1 else 0.5
        return Reversibility(mu, sigma, n)

    def as_dict(self) -> dict[str, float | int]:
        return {"mu": self.mu, "sigma": self.sigma, "n": self.n}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Reversibility:
        return cls(float(d.get("mu", 0.0)), float(d.get("sigma", 1.0)), int(d.get("n", 0)))


UNKNOWN = Reversibility()


@dataclass(frozen=True, slots=True)
class Action:
    """Удержание выхода в течение времени, возможно с зажатыми модификаторами."""

    kind: Kind
    duration_ms: int
    output: str | None = None
    modifiers: tuple[str, ...] = ()
    dx: int = 0
    dy: int = 0
    reversibility: Reversibility = UNKNOWN
    # Заглушена ли инъекция маской ввода. Действие с masked=True всё равно
    # попадает в журнал — см. docs/ARCHITECTURE-HARNESS.md, «Маска ввода».
    masked: bool = False
    mask_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.duration_ms, int):
            raise ActionError("duration_ms должно быть целым числом миллисекунд")
        if self.duration_ms < 0:
            raise ActionError(f"duration_ms отрицательное: {self.duration_ms}")
        if self.kind in (Kind.KEY, Kind.BUTTON):
            if not self.output:
                raise ActionError(f"{self.kind}: не указан выход")
            if not OUTPUT_RE.match(self.output):
                raise ActionError(
                    f"выход {self.output!r} не похож на непрозрачный идентификатор "
                    "(OUT_xx / BTN_xx / MOD_xx). Имена клавиш в действие не попадают"
                )
            if self.duration_ms == 0:
                raise ActionError(
                    "удержание 0 мс — это дискретное событие, которого в модели нет "
                    "(инвариант 8). Даже самое короткое нажатие имеет длительность"
                )
            if self.dx or self.dy:
                raise ActionError(f"{self.kind}: dx/dy не имеют смысла")
        elif self.kind is Kind.MOUSE_MOVE:
            if self.output is not None:
                raise ActionError("mouse_move не адресует выход")
            if self.dx == 0 and self.dy == 0:
                raise ActionError("mouse_move без смещения — это NOTHING, а не движение")
        elif self.kind is Kind.NOTHING:
            if self.output is not None or self.dx or self.dy:
                raise ActionError("NOTHING не несёт ни выхода, ни смещения")
        for m in self.modifiers:
            if not OUTPUT_RE.match(m):
                raise ActionError(f"модификатор {m!r} должен быть непрозрачным идентификатором")
        if len(set(self.modifiers)) != len(self.modifiers):
            raise ActionError(f"модификатор повторяется: {self.modifiers}")
        if self.masked and not self.mask_reason:
            raise ActionError("masked=True требует mask_reason: иначе журнал не объяснит пропуск")
        if not self.masked and self.mask_reason:
            raise ActionError("mask_reason без masked=True")
        object.__setattr__(self, "modifiers", tuple(self.modifiers))

    # --- конструкторы -------------------------------------------------------

    @classmethod
    def key(cls, output: str, duration_ms: int, modifiers: tuple[str, ...] = (),
            reversibility: Reversibility = UNKNOWN) -> Action:
        return cls(Kind.KEY, duration_ms, output=output, modifiers=tuple(modifiers),
                   reversibility=reversibility)

    @classmethod
    def button(cls, output: str, duration_ms: int, modifiers: tuple[str, ...] = (),
               reversibility: Reversibility = UNKNOWN) -> Action:
        return cls(Kind.BUTTON, duration_ms, output=output, modifiers=tuple(modifiers),
                   reversibility=reversibility)

    @classmethod
    def mouse(cls, dx: int, dy: int, duration_ms: int,
              reversibility: Reversibility = UNKNOWN) -> Action:
        return cls(Kind.MOUSE_MOVE, duration_ms, dx=int(dx), dy=int(dy),
                   reversibility=reversibility)

    @classmethod
    def nothing(cls, duration_ms: int) -> Action:
        # Бездействие обратимо по определению: ничего не произошло.
        return cls(Kind.NOTHING, duration_ms, reversibility=Reversibility(1.0, 0.0, 1))

    # --- производные --------------------------------------------------------

    def masked_as(self, reason: str) -> Action:
        """Та же попытка, но заглушённая маской. Пишется в журнал наравне."""
        return Action(self.kind, self.duration_ms, self.output, self.modifiers,
                      self.dx, self.dy, self.reversibility, masked=True, mask_reason=reason)

    def outputs_touched(self) -> tuple[str, ...]:
        """Все выходы, которые действие задействует: модификаторы и сам выход.

        Порядок значим: модификаторы зажимаются первыми и отпускаются последними.
        """
        return (*self.modifiers, *((self.output,) if self.output else ()))

    @property
    def caution(self) -> float:
        return self.reversibility.caution

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": str(self.kind),
            "duration_ms": self.duration_ms,
            "output": self.output,
            "modifiers": list(self.modifiers),
            "reversibility": self.reversibility.as_dict(),
        }
        if self.kind is Kind.MOUSE_MOVE:
            d["dx"], d["dy"] = self.dx, self.dy
        if self.masked:
            d["masked"] = True
            d["mask_reason"] = self.mask_reason
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Action:
        return cls(
            Kind(d["kind"]),
            int(d["duration_ms"]),
            output=d.get("output"),
            modifiers=tuple(d.get("modifiers", ())),
            dx=int(d.get("dx", 0)),
            dy=int(d.get("dy", 0)),
            reversibility=Reversibility.from_dict(d.get("reversibility", {})),
            masked=bool(d.get("masked", False)),
            mask_reason=d.get("mask_reason"),
        )

    def __str__(self) -> str:
        mods = ("+".join(self.modifiers) + "+") if self.modifiers else ""
        if self.kind is Kind.MOUSE_MOVE:
            body = f"мышь {self.dx:+d},{self.dy:+d}"
        elif self.kind is Kind.NOTHING:
            body = "ничего"
        else:
            body = f"{mods}{self.output}"
        tail = f" [заглушено: {self.mask_reason}]" if self.masked else ""
        return f"{body} {self.duration_ms} мс{tail}"
