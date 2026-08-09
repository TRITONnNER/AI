"""Граница восприятия для текста: надписи становятся символами и только символами.

`TASK-06-CLOSE-M3.md`, часть 3. Единственная часть М3, которой не было вовсе, и она
блокирует М4: карточки заземлённых символов не построить, пока текст не превращается
в символы.

## Главное правило и почему оно устроено так

**Хеширование происходит до того, как строка покинет этот модуль.** Не «после
распознавания», не «перед передачей планировщику», а внутри одного вызова: строка
рождается в чтении и умирает в символизаторе, ни разу не оказавшись в возвращаемом
значении.

Это не осторожность, а единственный способ, которым инвариант 5 может быть выполнен
проверяемо. Если бы `read()` возвращал строки, а хеширование делал вызывающий, то
корректность зависела бы от того, что все вызывающие помнят про хеширование — а они
не помнят, и первый же новый вызывающий получит читаемый текст. Поэтому:

- `TextBoundary.read()` возвращает `list[SymbolObservation]` — там нет поля, куда
  строка могла бы попасть;
- сырые строки видит только `_reader` и только внутри `read()`;
- расшифровка уходит в **отладочный канал**, а он агентскому коду недостижим
  (инвариант 12, проверяется обходом графа импортов).

## Что тут не решается

**Качество распознавания.** Читатель подключается снаружи; в этом окружении нет ни
дисплея, ни установленного OCR, и `TesseractReader` честно отказывается, а не
возвращает пустые строки. Пустая строка вместо отказа была бы худшим исходом: она
неотличима от «текста нет», и на записи с текстом дала бы «ни одна строка не
доехала» при полностью сломанном чтении.

**Что считать областью текста.** Области приходят снаружи — от разметки исследователя
или от детектора. Внутри агентского пути координаты областей не появляются: в
наблюдение попадают только символ, непрозрачная метка источника и доверие. Имя
источника (`chat`, `ui`) остаётся по эту сторону границы вместе с координатами — оно
такая же разметка, как они.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from ..core.symbols import (Namespace, SymbolError, Symbolizer, is_source_tag,
                            is_symbol, normalize)


class TextError(RuntimeError):
    pass


class ReaderUnavailable(TextError):
    """Читателя нет. Громкий отказ вместо пустого списка строк.

    Пустой список неотличим от «текста на экране нет», и на записи с текстом он дал
    бы «ни одна строка не доехала до планировщика» при полностью сломанном чтении —
    то есть зелёный отчёт над сломанным механизмом.
    """


@dataclass(frozen=True, slots=True)
class TextArea:
    """Где искать надпись и чем считать источник.

    Координаты — вход, а не выход: они приходят от исследователя или от детектора и
    в наблюдение не попадают. Агент не получает разметки интерфейса ни одним каналом
    (инвариант 4), в том числе и через «вот тут была надпись».
    """

    top: int
    left: int
    height: int
    width: int
    namespace: Namespace = Namespace.WORLD

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.top:self.top + self.height,
                     self.left:self.left + self.width]

    def as_debug_dict(self) -> dict[str, Any]:
        """Только для отладочного канала: там координаты законны."""
        return {"top": self.top, "left": self.left, "height": self.height,
                "width": self.width, "namespace": str(self.namespace)}


@dataclass(frozen=True, slots=True)
class SymbolObservation:
    """Что уходит дальше границы. Строки здесь нет и быть не может.

    Полей ровно четыре, и ни одно не может содержать читаемого: символ непрозрачен,
    источник — непрозрачная метка, доверие — число, `at_frame` — номер кадра.
    Проверка `assert_no_plain_text` на этом наблюдении проходит по построению.

    **Источник — метка, а не имя.** Первая версия несла здесь `Namespace`, и
    проверка на читаемый текст поймала это сразу: в записи оказывалось слово `chat`
    рядом с символом. Само хеширование при этом было верным — пространство имён
    подмешано в ключ и в символе не видно, — но соседнее поле выдавало ровно то, что
    символ скрывал: классификацию источников, то есть разметку (инвариант 4). Метка
    `SRC_1A2B` разделяет потоки так же надёжно и не сообщает, который из них чат.
    """

    symbol: str
    source: str
    trust: float
    at_frame: int

    def __post_init__(self) -> None:
        if not is_symbol(self.symbol):
            raise TextError(
                f"{self.symbol!r} не символ. За границу восприятия уходит только "
                "непрозрачный SYM_XXXX; читаемая строка здесь означала бы, что "
                "хеширование не случилось (инвариант 5)")
        if not is_source_tag(self.source):
            raise TextError(
                f"{self.source!r} не метка источника. Наружу уходит непрозрачный "
                "SRC_XXXX: имя источника было бы готовой классификацией каналов, "
                "то есть разметкой (инвариант 4)")

    def as_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "source": self.source,
                "trust": round(self.trust, 3), "at_frame": self.at_frame}


class TextReader(Protocol):
    """Чем читать надписи. Возвращает строки, и они не покидают `TextBoundary`."""

    name: str

    def read(self, image: np.ndarray) -> str: ...


class TesseractReader:
    """OCR через tesseract. В этом окружении отсутствует и отказывается громко."""

    name = "tesseract"

    def __init__(self, lang: str = "rus+eng") -> None:
        self.lang = lang
        self._api: Any = None

    def _load(self) -> Any:
        if self._api is None:
            try:
                import pytesseract  # noqa: PLC0415 — платформенная зависимость
            except ImportError as e:
                raise ReaderUnavailable(
                    "нет пакета pytesseract. Поставьте tesseract и pytesseract, "
                    "либо подключите другой читатель. Возвращать пустые строки "
                    "вместо отказа нельзя: «текста нет» и «читать нечем» — разные "
                    "утверждения, и первое молча выдало бы зелёный отчёт") from e
            self._api = pytesseract
        return self._api

    def read(self, image: np.ndarray) -> str:
        api = self._load()
        return str(api.image_to_string(image, lang=self.lang))


class ScriptedReader:
    """Читатель для замеров: отдаёт заранее известные строки по номеру области.

    Нужен затем, что проверять надо **не качество OCR, а непроницаемость границы**.
    Для этого текст должен быть известен точно, иначе непонятно, дошла строка или
    просто не распозналась. Здесь он известен по построению.

    Это не заглушка в запрещённом смысле: он не изображает работу распознавания, он
    её заменяет объявленным входом, и его имя видно в каждом отчёте.
    """

    name = "scripted"

    def __init__(self, lines: Sequence[str]) -> None:
        self._lines = list(lines)
        self.calls = 0

    def read(self, image: np.ndarray) -> str:
        line = self._lines[self.calls % len(self._lines)]
        self.calls += 1
        return line


@dataclass(slots=True)
class BoundaryStats:
    """Что произошло на границе.

    Две сводки, а не одна, и разделены они не из осторожности, а по замеру: первая
    версия отдавала один словарь с именем читателя внутри, и проверка журнала на нём
    отказала — `'scripted'` читаемое слово. Имя аппаратуры законно в отчёте
    исследователя (по нему видно, чем читали) и незаконно в перцепте, поэтому
    `as_dict` для отчёта, `as_percept` для агентского пути.
    """

    reader: str
    areas: int = 0
    symbolized: int = 0
    empty: int = 0                 # область была, текста в ней не нашлось
    by_source: dict[str, int] = field(default_factory=dict)
    grounded: int = 0              # сколько различных символов попало в таблицу
    collisions: int = 0            # разных надписей, схлопнувшихся в один символ

    def as_percept(self) -> dict[str, Any]:
        """Только числа. Разбивка по непрозрачным меткам, а не по именам источников."""
        return {"areas": self.areas, "symbolized": self.symbolized,
                "empty": self.empty,
                "by_source": dict(sorted(self.by_source.items())),
                "grounded": self.grounded}

    def as_dict(self) -> dict[str, Any]:
        """Отчёт исследователю: то же плюс имя читателя и число коллизий."""
        return {"reader": self.reader, **self.as_percept(),
                "collisions": self.collisions}


class TextBoundary:
    """Единственное место, где надпись превращается в символ.

    Расшифровка пишется в отладочный канал, если он передан. Канал необязателен: без
    него символы всё равно получаются, просто исследователь потом не сможет сказать,
    чему они соответствуют. Обратное — символы без хеширования — невозможно.
    """

    def __init__(self, symbolizer: Symbolizer, reader: TextReader, *,
                 debug: Any = None) -> None:
        self.symbolizer = symbolizer
        self.reader = reader
        self._debug = debug
        self.stats = BoundaryStats(reader=getattr(reader, "name", "?"))
        # Таблица заземления стартует **пустой** и в пределах линии пополняется.
        # Хеш-функция при этом стабильна между линиями (иначе линии несравнимы), а
        # заземление — нет: что символ означает, каждая линия узнаёт сама.
        #
        # Здесь именно словарь символ → надпись, а не множество символов. Множество
        # было первой версией, и замер на корпусе в две тысячи надписей показал, чем
        # оно плохо: при коллизии второй текст с тем же символом считался «уже
        # заземлённым» и в отладочную таблицу не попадал. Коллизия — это когда агент
        # видит две разные надписи как одну, то есть порча его опыта; она обязана
        # доехать до исследователя, а не быть отфильтрованной по дороге.
        self._grounded: dict[str, str] = {}
        self._sources_logged: set[Namespace] = set()

    def read(self, frame: np.ndarray, areas: Iterable[TextArea], *,
             at_frame: int) -> list[SymbolObservation]:
        """Прочитать области и вернуть **только символы**.

        Сырая строка существует внутри тела этой функции и нигде больше. Она не
        возвращается, не складывается в поле объекта и не попадает в статистику —
        туда идут числа.
        """
        out: list[SymbolObservation] = []
        for area in areas:
            self.stats.areas += 1
            raw = self.reader.read(area.crop(frame))
            try:
                symbol = self.symbolizer.symbolize(raw, area.namespace)
            except SymbolError:
                # Пустая надпись символом не становится — это законный исход, а не
                # ошибка: в области могло не оказаться текста.
                self.stats.empty += 1
                continue
            if not is_symbol(symbol):
                # Режим ablation `text_symbolized=False`: символизатор вернул саму
                # надпись. Наружу она не уходит даже тогда — иначе переключатель,
                # заведённый для замера, стал бы дырой в инварианте 5. Прогон с
                # читаемым текстом отличим по профилю, а не по утечке.
                raise TextError(
                    "символизатор выключен (`text_symbolized=False`): в этом режиме "
                    "граница восприятия текст не пропускает. Замер с читаемым "
                    "языком ставится на отдельной линии, а не утечкой через границу")
            self.stats.symbolized += 1
            tag = self.symbolizer.source_tag(area.namespace)
            self.stats.by_source[tag] = self.stats.by_source.get(tag, 0) + 1
            if (area.namespace not in self._sources_logged
                    and self._debug is not None):
                # Расшифровка метки источника — туда же, где расшифровка надписей.
                self._sources_logged.add(area.namespace)
                self._debug.write_symbol(tag, str(area.namespace),
                                         salt_id=self.symbolizer.salt_id,
                                         kind="source")
            known = self._grounded.get(symbol)
            if known is None or known != normalize(raw):
                if known is not None:
                    # Разные надписи, один символ. Считается и пишется, а не глотается.
                    self.stats.collisions += 1
                self._grounded[symbol] = normalize(raw)
                self.stats.grounded = len(self._grounded)
                if self._debug is not None:
                    # Расшифровка — в отладочный поток. Сюда строка попадает
                    # законно: этот файл исследовательский и агентскому коду
                    # недостижим (инвариант 12).
                    self._debug.write_symbol(symbol, raw,
                                             salt_id=self.symbolizer.salt_id)
            out.append(SymbolObservation(
                symbol=symbol, source=tag,
                trust=self.symbolizer.trust(area.namespace), at_frame=at_frame))
        return out

    @property
    def grounded(self) -> int:
        """Сколько различных символов уже встречено. Таблица начиналась пустой."""
        return len(self._grounded)
