"""Отладочный поток исследователя.

Инвариант 12: истинные координаты, имена предметов, состояние мира пишутся в
отдельный поток для исследователя. Ни одна функция, доступная агенту, не должна
иметь к нему доступа. Проверяется тестом.

Технически изоляция здесь держится на трёх вещах, и ни одна из них не является
обещанием в комментарии:

1. **Отдельный пакет.** Всё, что читает или пишет истину, лежит в
   `harness.debug`. Тест `tests/test_invariant_12_isolation.py` строит граф
   импортов и падает, если что-нибудь из `harness.agentside` дотягивается до
   этого пакета — хоть напрямую, хоть через десять посредников.
2. **Отдельный каталог.** Поток пишется не в журнал, а в свой подкаталог. Путь
   не выводится из пути журнала: его передают снаружи, и агентская сторона его
   не знает, потому что не получает.
3. **Отсутствие обратной функции у символов.** `harness.core.symbols` умеет
   только вперёд. Таблица расшифровки лежит здесь и нигде больше.

Что здесь законно писать: истинные координаты, настоящие имена предметов и
клавиш, состояние мира, счёт, флаги враждебности, расшифровку символов, истину
синтетического корпуса. То есть ровно то, что инвариант 4 запрещает показывать
агенту.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Literal

from ..core.clocks import Stamp

TRUTH = "truth.jsonl"
SYMBOLS = "symbols.jsonl"


class DebugChannelError(RuntimeError):
    pass


class DebugChannel:
    """Дозаписываемый поток истины. Живёт рядом с журналом, но не внутри него."""

    @classmethod
    def for_profile(cls, root: str | Path, profile: Any,
                    mode: Literal["a", "r"] = "a") -> "DebugChannel":
        """Канал по профилю. `debug_channel_enabled=False` — канала нет.

        Выключенный канал ничего не пишет и в этом честен: истины про прогон потом
        не будет, и сверить результат будет не с чем. Это законный режим — например,
        когда запись отдаётся кому-то, кому истину видеть нельзя, — но он должен быть
        объявлен ручкой, а не получаться случайно.
        """
        return cls(root, mode,
                   enabled=bool(profile.structural["debug_channel_enabled"]))

    def __init__(self, root: str | Path, mode: Literal["a", "r"] = "r", *,
                 enabled: bool = True) -> None:
        self.root = Path(root)
        self.mode = mode
        self.enabled = bool(enabled)
        if not self.enabled:
            self._fh = None
            self._sym_fh = None
            return
        if mode == "a":
            self.root.mkdir(parents=True, exist_ok=True)
            self._fh = (self.root / TRUTH).open("a", encoding="utf-8")
            self._sym_fh = (self.root / SYMBOLS).open("a", encoding="utf-8")
        else:
            if not self.root.is_dir():
                raise DebugChannelError(f"нет отладочного потока: {self.root}")
            self._fh = None
            self._sym_fh = None

    @staticmethod
    def beside(journal_root: str | Path, mode: Literal["a", "r"] = "r") -> DebugChannel:
        """Стандартное размещение: `<сессия>/debug/` рядом с `<сессия>/journal/`.

        Это удобство для исследователя, а не путь, доступный агенту: функция
        живёт в `harness.debug` и агентской стороне недоступна по построению.
        """
        journal_root = Path(journal_root)
        return DebugChannel(journal_root.parent / "debug", mode=mode)

    # --- запись -------------------------------------------------------------

    def write(self, stamp: Stamp, code: str, truth: dict[str, Any]) -> None:
        """Записать истину. `code` — что за факт, `truth` — сам факт как есть.

        Никакой фильтрации содержимого здесь нет и не должно быть: это тот
        единственный канал, где читаемый текст и настоящие координаты законны.
        """
        if self.mode != "a":
            raise DebugChannelError("поток открыт только на чтение")
        if not code:
            raise DebugChannelError("у записи истины должен быть код факта")
        if not self.enabled:
            return
        line = {"stamp": stamp.as_dict(), "code": code, "truth": truth}
        self._fh.write(json.dumps(line, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")) + "\n")
        self._fh.flush()

    def write_symbol(self, symbol: str, text: str, *, salt_id: str) -> None:
        """Строка таблицы расшифровки: символ → настоящая надпись."""
        if self.mode != "a":
            raise DebugChannelError("поток открыт только на чтение")
        if not self.enabled:
            return
        self._sym_fh.write(json.dumps(
            {"symbol": symbol, "text": text, "salt_id": salt_id},
            ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        self._sym_fh.flush()

    # --- чтение -------------------------------------------------------------

    def read(self, code: str | None = None) -> Iterator[dict[str, Any]]:
        path = self.root / TRUTH
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if code is None or rec.get("code") == code:
                    yield rec

    def symbol_table(self) -> dict[str, str]:
        """Таблица расшифровки. Коллизии видны: один символ на разные надписи."""
        path = self.root / SYMBOLS
        table: dict[str, str] = {}
        collisions: dict[str, set[str]] = {}
        if not path.exists():
            return table
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sym, text = rec["symbol"], rec["text"]
                if sym in table and table[sym] != text:
                    collisions.setdefault(sym, {table[sym]}).add(text)
                table[sym] = text
        if collisions:
            # Не молчим: коллизия символов означает, что агент видит две разные
            # надписи как одну, и это меняет смысл его опыта.
            raise DebugChannelError(
                "коллизии символов: "
                + "; ".join(f"{s} → {sorted(v)}" for s, v in sorted(collisions.items()))
                + ". Увеличьте SYMBOL_DIGITS и перезапишите корпус")
        return table

    def decrypt(self, symbol: str) -> str | None:
        return self.symbol_table().get(symbol)

    def close(self) -> None:
        for fh in (self._fh, self._sym_fh):
            if fh:
                fh.close()
        self._fh = self._sym_fh = None
        self.mode = "r"

    def __enter__(self) -> DebugChannel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
