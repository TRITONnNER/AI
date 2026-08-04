"""Хеширование текста в непрозрачный символ.

Инвариант 5: весь текст с экрана хешируется в непрозрачный символ (`SYM_7A3F`)
до того, как попадёт в планировщик. Таблица расшифровки хранится отдельно и
доступна только исследователю.

Здесь — только необратимая половина. Обратной функции в этом модуле нет и быть
не может: таблица живёт в `harness.debug.symbol_table`, куда агентской стороне
нет пути (это проверяется тестом на изоляцию).

Почему соль вообще нужна. Без неё символ — это чистый хеш строки, одинаковый во
всех прогонах и на всех машинах; кто угодно, у кого есть словарь игры, строит
таблицу расшифровки за минуту. Соль привязана к профилю
(`structural.symbol_salt_id`), потому что смена соли меняет разом все символы —
то есть меняет форму опыта и обязана форкать журнал.

Нормализация нарочно грубая: `«Здоровье: 20»` и `«здоровье: 3»` должны давать
*разные* символы (число — часть надписи и меняется по делу), а `«Health »` и
`«Health»` — одинаковые (разница от шума OCR). Поэтому режем только пробелы по
краям и внутренние повторы, регистр не трогаем.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

SYMBOL_RE = re.compile(r"^SYM_[0-9A-F]{4}$")
_WS = re.compile(r"\s+")

# Сколько шестнадцатеричных цифр в символе. 4 → 65 536 значений. Для одного
# интерфейса игры этого хватает с запасом; коллизии считаются и логируются
# исследователем, а не игнорируются.
SYMBOL_DIGITS = 4


class SymbolError(ValueError):
    pass


def normalize(text: str) -> str:
    """Приводит надпись к виду, устойчивому к шуму распознавания."""
    if not isinstance(text, str):
        raise SymbolError(f"ожидалась строка, пришло {type(text).__name__}")
    t = unicodedata.normalize("NFKC", text)
    t = _WS.sub(" ", t).strip()
    return t


class Symbolizer:
    """Односторонняя замена надписи на символ.

    Экземпляр не хранит ни одной исходной строки: `symbolize` ничего не
    запоминает. Если понадобится расшифровка — её пишет граница восприятия в
    отладочный поток, а не этот объект.
    """

    __slots__ = ("_salt_id", "_key", "_enabled")

    @classmethod
    def from_profile(cls, profile: Any, secret: bytes | None = None) -> "Symbolizer":
        """Хешировать ли текст — решает профиль, а не вызывающий.

        `text_symbolized=False` — законный режим ablation: агент получает читаемый
        язык до того, как его заслужил, и это надо иметь возможность замерить. Ручка
        структурная, значит журнал форкается, и смешать такой опыт с обычным нельзя
        (инвариант 11). Но молча игнорировать ручку нельзя тем более: объявленная и
        никем не читаемая настройка — это ложь о возможностях.
        """
        return cls(str(profile.structural["symbol_salt_id"]), secret,
                   enabled=bool(profile.structural["text_symbolized"]))

    def __init__(self, salt_id: str, secret: bytes | None = None, *,
                 enabled: bool = True) -> None:
        if not salt_id:
            raise SymbolError("salt_id обязателен: он часть структурного профиля")
        self._salt_id = salt_id
        self._enabled = bool(enabled)
        # Секрет по умолчанию выводится из salt_id. Этого достаточно, чтобы
        # символы не совпадали между профилями, но недостаточно против того, у
        # кого есть исходники и словарь. Настоящий секрет задаётся явно, и тогда
        # он не должен попадать ни в журнал, ни в профиль.
        self._key = secret if secret is not None else b"harness-symbol-" + salt_id.encode()

    @property
    def salt_id(self) -> str:
        return self._salt_id

    @property
    def enabled(self) -> bool:
        """Хеширует ли этот символизатор вообще. Видно в отчётах и в тестах."""
        return self._enabled

    def symbolize(self, text: str) -> str:
        norm = normalize(text)
        if not norm:
            raise SymbolError("пустая надпись символом не становится")
        if not self._enabled:
            # Режим ablation: возвращается сама надпись. Ни одна проверка на
            # непрозрачность её не пропустит, и это правильно — прогон с читаемым
            # текстом обязан быть отличим от обычного на всех уровнях.
            return norm
        h = hashlib.blake2b(norm.encode("utf-8"), key=self._key, digest_size=8)
        return "SYM_" + h.hexdigest()[:SYMBOL_DIGITS].upper()

    def symbolize_all(self, texts: list[str]) -> list[str]:
        return [self.symbolize(t) for t in texts]


def is_symbol(value: object) -> bool:
    return isinstance(value, str) and bool(SYMBOL_RE.match(value))


def assert_no_plain_text(payload: object, *, path: str = "") -> None:
    """Проверка границы: в структуре, уходящей агенту, нет читаемого текста.

    Строка допустима, только если это символ (`SYM_1A2B`) или непрозрачный
    идентификатор выхода/сущности (`OUT_2C`, `ПРЕДМЕТ_1C90` в терминах пульта —
    здесь `ENT_1C90`). Всё остальное — утечка надписи, то есть нарушение
    инварианта 5, и падать надо громко.

    Ключи словарей не проверяются: это имена полей схемы, а не содержимое
    экрана. Схема агенту известна по построению — он видит структуру своего
    восприятия, но не смысл надписей в ней.
    """
    if isinstance(payload, str):
        if is_symbol(payload) or re.match(r"^(OUT|MOD|BTN|ENT|PLACE|SND)_[0-9A-F]{2,4}$", payload):
            return
        raise SymbolError(
            f"читаемый текст на пути к агенту{' в ' + path if path else ''}: {payload!r}. "
            "Надпись обязана быть заменена символом до передачи (инвариант 5)"
        )
    if isinstance(payload, dict):
        for k, v in payload.items():
            assert_no_plain_text(v, path=f"{path}.{k}" if path else str(k))
        return
    if isinstance(payload, (list, tuple, set, frozenset)):
        for i, v in enumerate(payload):
            assert_no_plain_text(v, path=f"{path}[{i}]")
        return
    if isinstance(payload, (bool, int, float)) or payload is None:
        return
    if hasattr(payload, "shape") and hasattr(payload, "dtype"):
        return  # массив пикселей: это и есть законный канал восприятия
    raise SymbolError(f"неизвестный тип на пути к агенту{' в ' + path if path else ''}: "
                      f"{type(payload).__name__}")
