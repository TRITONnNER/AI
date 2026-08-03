"""Маска ввода: что агенту разрешено нажимать.

Две вещи, которые в дизайне пульта сделаны иначе и здесь исправлены (см.
`docs/DESIGN-REVIEW-CONSOLE.md`):

1. **Маска ничего не скрывает от журнала.** Заглушённая попытка пишется в
   журнал с `masked=True` и причиной — этим занимается `Injector`. Здесь только
   решение «заглушить или нет».
2. **Маска говорит на языке выходов, а не клавиш.** Ключи здесь — непрозрачные
   `OUT_2C`, а не `Q`, `F5`, `Alt`. Исследователь задаёт маску в настоящих
   именах через `harness.debug.keymap`, который переводит имена в выходы; сама
   маска настоящих имён не хранит и не может их выдать (инвариант 4).

Область действия (`scope`) взята из пульта: `window`, `profile`, `app`,
`device`. Смысл — насколько широко правило распространяется. Маска
`device`-уровня действует всегда; `window` — только для конкретного окна.
"""

from __future__ import annotations

from typing import Iterable

from ..core.action import OUTPUT_RE

SCOPES = ("device", "app", "profile", "window")


class MaskError(ValueError):
    pass


class InputMask:
    """Наборы заблокированных выходов по областям действия."""

    def __init__(self, blocked: dict[str, Iterable[str]] | None = None) -> None:
        self._blocked: dict[str, set[str]] = {s: set() for s in SCOPES}
        for scope, outputs in (blocked or {}).items():
            for out in outputs:
                self.block(out, scope=scope)

    def _check(self, output: str, scope: str) -> None:
        if scope not in SCOPES:
            raise MaskError(f"нет такой области действия: {scope!r}; есть {SCOPES}")
        if not OUTPUT_RE.match(output):
            raise MaskError(
                f"{output!r} не выход. Маска задаётся непрозрачными идентификаторами "
                "(OUT_xx / BTN_xx / MOD_xx). Настоящие имена клавиш живут только в "
                "отладочном потоке — переводите их через harness.debug.keymap"
            )

    def block(self, output: str, *, scope: str = "window") -> None:
        self._check(output, scope)
        self._blocked[scope].add(output)

    def unblock(self, output: str, *, scope: str = "window") -> None:
        self._check(output, scope)
        self._blocked[scope].discard(output)

    def blocked_outputs(self, outputs: Iterable[str], *, scope: str = "window") -> set[str]:
        """Какие из перечисленных выходов заглушены для этой области.

        Область действия вложена: `device` шире `app`, `app` шире `profile`,
        `profile` шире `window`. Заблокированное шире — заблокировано и уже.
        """
        if scope not in SCOPES:
            raise MaskError(f"нет такой области действия: {scope!r}; есть {SCOPES}")
        wide = SCOPES[: SCOPES.index(scope) + 1]
        active: set[str] = set()
        for s in wide:
            active |= self._blocked[s]
        return {o for o in outputs if o in active}

    def count(self, scope: str | None = None) -> int:
        if scope is None:
            return len(set().union(*self._blocked.values())) if self._blocked else 0
        return len(self._blocked[scope])

    def as_dict(self) -> dict[str, list[str]]:
        return {s: sorted(v) for s, v in self._blocked.items() if v}

    def __repr__(self) -> str:
        return f"InputMask({self.as_dict()})"
