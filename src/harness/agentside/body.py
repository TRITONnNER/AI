"""Тело: единственный способ, которым агент может что-то сделать.

Что агент знает о своём теле и что нет.

**Знает:** сколько у него выходов и как каждый адресовать (`OUT_2C`). Это
законно: человек, впервые севший за ПК, видит, что на клавиатуре примерно сотня
клавиш, и может ткнуть в любую. Незнание — не в количестве, а в смысле.

**Не знает:** что каждый выход делает, как он называется, какие выходы
«системные», «опасные» или «бесполезные». Ничего этого в `Body` нет — ни поля,
ни метода, ни списка.

`Body` не импортирует ни `harness.inject`, ни `harness.debug`. Устройство
передаётся снаружи как функция: так агентская сторона не получает пути ни к
таблице скан-кодов, ни к потоку истины (инварианты 4 и 12).

Инвариант 9 виден в подписи: `press` требует длительность и возвращает исход, а
осторожность считается из обратимости — из `Action.caution`, то есть из «умею ли
я это откатить», а не из списка запретов.
"""

from __future__ import annotations

from typing import Callable, Iterable, Protocol, runtime_checkable

from ..core.action import Action, Reversibility


@runtime_checkable
class Submit(Protocol):
    """Куда уходит действие. Реализуется `Injector` на стороне харнесса."""

    def __call__(self, action: Action) -> object: ...


class Body:
    def __init__(self, outputs: Iterable[str], submit: Submit,
                 *, buttons: Iterable[str] = (),
                 reversibility: Callable[[Action], Reversibility] | None = None) -> None:
        self._outputs = tuple(dict.fromkeys(outputs))   # порядок стабилен, дублей нет
        self._buttons = frozenset(buttons)
        self._submit = submit
        self._reversibility = reversibility

    @property
    def outputs(self) -> tuple[str, ...]:
        """Все адресуемые выходы. Без смысла: только идентификаторы."""
        return self._outputs

    def __len__(self) -> int:
        return len(self._outputs)

    def is_button(self, output: str) -> bool:
        """Кнопка мыши или клавиша. Это различие агент может заметить сам —
        кнопок мало и они ведут себя иначе, — поэтому скрывать его смысла нет."""
        return output in self._buttons

    def _estimate(self, action: Action) -> Action:
        if self._reversibility is None:
            return action
        from dataclasses import replace
        return replace(action, reversibility=self._reversibility(action))

    def press(self, output: str, duration_ms: int,
              modifiers: tuple[str, ...] = ()) -> object:
        """Удержать выход. Дискретного «нажать» нет (инвариант 8)."""
        if output not in self._outputs:
            raise ValueError(f"выход {output!r} не принадлежит этому телу")
        for m in modifiers:
            if m not in self._outputs:
                raise ValueError(f"модификатор {m!r} не принадлежит этому телу")
        make = Action.button if self.is_button(output) else Action.key
        return self._submit(self._estimate(make(output, duration_ms, modifiers)))

    def move(self, dx: int, dy: int, duration_ms: int) -> object:
        return self._submit(self._estimate(Action.mouse(dx, dy, duration_ms)))

    def wait(self, duration_ms: int) -> object:
        """Ничего не делать — тоже действие, и оно тоже пишется в журнал."""
        return self._submit(Action.nothing(duration_ms))
