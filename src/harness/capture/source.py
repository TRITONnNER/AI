"""Что попадает в кадр: экран, окно или прямоугольник. TASK-21, часть 1.

## Почему это переключатель структуры, а не удобство

До этой задачи захват был один — «весь экран», — и выбор источника выглядел бы удобством:
поменьше пикселей, побыстрее запись. Он им не является.

**Захват окна выдаёт агенту границы окна, известные операционной системе.** Край кадра
совпадает с краем окна; обрамление приложения оказывается прибитым к рамке кадра
неподвижно и навсегда; рабочий стол за окном исчезает из поля зрения вовсе. Это половина
задачи разделения экранного и мирового слоя, отданная агенту готовой — причём отданная
незаметно, потому что в кадре ничего лишнего не видно, из него **вырезано** лишнее.

Отсюда следствие, из-за которого переключатель структурный: **числа несравнимы**. IoU
разделения себя и мира на записи окна отвечает на вопрос «умеет ли метод отделять
интерфейс приложения от его содержимого», а на записи всего экрана — на вопрос «умеет ли
он отделять неподвижное обрамление от движущегося мира, не зная, где кончается окно». Это
разные вопросы, и усреднять ответы нельзя ни в какую сторону. Поэтому `capture_source`
объявлен `structural=True` и меняет `structure_hash`, то есть форкает журнал.

## Что здесь параметр, а что условие записи

Вид источника — параметр профиля (`capture_source`), и он в хеше. **Сам прямоугольник —
нет.** Координаты окна зависят от машины, монитора, масштабирования и того, куда оператор
подвинул окно мышью; они не описывают устройство эксперимента, а описывают обстоятельства
одной записи. Поэтому рамка живёт в `session.json` и в журнале рядом с записью, а не в
`profile_hash`: иначе каждое перетаскивание окна форкало бы журнал.

## Чего здесь нет

Ни одного человеческого имени окна в том, что уходит агенту. Заголовок окна — это текст с
экрана (инварианты 4 и 5): он нужен исследователю, чтобы указать окно, и превращается в
непрозрачный идентификатор до того, как попадёт куда-либо ещё.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import Any, Protocol

from .base import GONE, UNCHANGED, BackendUnavailable, Frame, Gone, Unchanged

#: Виды источников. Совпадают со значениями `capture_source` в схеме — список здесь и
#: список там сверяются тестом: два перечисления одного набора расходятся молча.
KINDS: tuple[str, ...] = ("display", "window", "region")


class SourceError(ValueError):
    """Источник задан неверно: нет окна, пустая рамка, вид не тот."""


@dataclass(frozen=True, slots=True)
class Region:
    """Прямоугольник в экранных координатах. **Только для исследователя.**

    Агенту эти числа недоступны: он видит размер из самого кадра, а положение ему знать
    незачем (инвариант 4). Отсюда и `for_agent` ниже, отдающий одну площадь.
    """

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise SourceError(f"пустая рамка: {self.width}×{self.height}")

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)

    def as_dict(self) -> dict[str, Any]:
        return {"left": self.left, "top": self.top,
                "width": self.width, "height": self.height, "pixels": self.pixels}

    def for_agent(self) -> dict[str, Any]:
        """Что об источнике знает агент: сколько пикселей, и всё."""
        return {"pixels": self.pixels}


@dataclass(frozen=True, slots=True)
class WindowTarget:
    """Окно, которое просили снимать: чем его нашли и где оно сейчас."""

    handle: int
    title: str                    # человеческое имя; агенту не уходит
    region: Region

    @property
    def id(self) -> str:
        """Непрозрачный идентификатор окна — то, что видит агент вместо заголовка."""
        from ..devices import source_id

        return source_id(f"window:{self.title}")


class WindowSystem(Protocol):
    """Что нужно от системы, чтобы снимать окно. Ровно две операции.

    Протокол узкий нарочно: в тестах его подменяет поддельная система, и чем меньше
    операций, тем меньше расхождений между подделкой и настоящей. Реальный оконный
    менеджер умеет ещё сто вещей, и ни одна из них здесь не нужна.
    """

    name: str

    def find(self, title_part: str) -> WindowTarget: ...
    def rect(self, handle: int) -> Region | None: ...


class NoWindowSystem:
    """Системы, где спросить рамку окна нечем. Отказывается **громко**.

    Не заглушка: вернуть весь экран вместо запрошенного окна значило бы записать не то,
    что просили, и не сказать об этом. Запись выглядела бы удачной, а `capture_source` в
    профиле врал бы про содержимое кадра — то есть испортил бы сравнение молча.
    """

    name = "нет"

    def __init__(self, why: str) -> None:
        self.why = why

    def find(self, title_part: str) -> WindowTarget:
        raise BackendUnavailable(self.why)

    def rect(self, handle: int) -> Region | None:
        raise BackendUnavailable(self.why)


class Win32Windows:
    """Рамка окна через `user32` (ctypes). Windows.

    Через ctypes, а не через `pywin32`: ставить пакет ради двух вызовов — лишнее звено на
    машине оператора, а `user32.dll` есть в любой Windows.

    **Про масштабирование.** `GetWindowRect` отдаёт физические пиксели, если процесс
    объявлен DPI-aware, и логические, если нет; при масштабе 125 % это разница в четверть
    кадра. Заявление делается явным вызовом `SetProcessDpiAwarenessContext`, и результат
    вызова записывается в `dpi_aware` — если объявиться не удалось, это видно, а не
    подразумевается. Не проверено на живой машине: в контейнере нет ни Windows, ни окон.
    """

    name = "user32"

    def __init__(self) -> None:
        import ctypes

        self._ctypes = ctypes
        self._user32 = ctypes.windll.user32          # type: ignore[attr-defined]
        # -4 — DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2. Значение объявлено числом,
        # потому что в ctypes константы Windows не приходят ниоткуда сами.
        self.dpi_aware = False
        try:
            self.dpi_aware = bool(self._user32.SetProcessDpiAwarenessContext(-4))
        except Exception:
            # Старая Windows или уже объявленный контекст. Это не повод отказываться от
            # захвата — но рамка может оказаться в логических пикселях, и знать об этом
            # надо. Записывается в отчёт как есть.
            self.dpi_aware = False

    def _rect(self, handle: int) -> Region | None:
        import ctypes

        class RECT(ctypes.Structure):
            _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                        ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

        r = RECT()
        ok = self._user32.GetWindowRect(ctypes.c_void_p(handle), ctypes.byref(r))
        if not ok:
            return None
        w, h = int(r.right - r.left), int(r.bottom - r.top)
        if w <= 0 or h <= 0:
            # Свёрнутое окно даёт вырожденную или отрицательную рамку. Для агента это
            # исчезновение источника, а не ошибка (см. `WindowFollowing`).
            return None
        return Region(int(r.left), int(r.top), w, h)

    def find(self, title_part: str) -> WindowTarget:
        import ctypes

        found: list[tuple[int, str]] = []
        needle = title_part.casefold()

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def each(hwnd, _param):  # noqa: ANN001 — подпись задана Windows
            if not self._user32.IsWindowVisible(hwnd):
                return True
            n = self._user32.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            self._user32.GetWindowTextW(hwnd, buf, n + 1)
            if needle in buf.value.casefold():
                found.append((int(hwnd), buf.value))
            return True

        self._user32.EnumWindows(each, 0)
        if not found:
            raise SourceError(
                f"окна с «{title_part}» в заголовке не нашлось среди видимых. "
                "Проверьте, что окно открыто и не свёрнуто")
        if len(found) > 1:
            # Двусмысленность не разрешается «первым попавшимся»: записали бы не то окно,
            # и узнали бы об этом после записи, если вообще узнали.
            names = ", ".join(f"«{t}»" for _, t in found[:5])
            raise SourceError(
                f"под «{title_part}» подходит {len(found)} окон: {names}. "
                "Уточните часть заголовка так, чтобы подходило одно")
        handle, title = found[0]
        region = self._rect(handle)
        if region is None:
            raise SourceError(f"окно «{title}» найдено, но рамка пустая: оно свёрнуто?")
        return WindowTarget(handle, title, region)

    def rect(self, handle: int) -> Region | None:
        return self._rect(handle)


def system_windows() -> WindowSystem:
    """Чем спрашивать рамку окна на этой машине.

    На Linux и macOS — нечем, и это объявлено отказом с текстом «что делать», а не
    подменой на весь экран. X11 умеет отвечать через `python-xlib` или `xdotool`, macOS —
    через ScreenCaptureKit; ни то, ни другое не написано, и писать вслепую механизм, чей
    единственный смысл — точность рамки, значило бы получить неотличимо неверную рамку.
    """
    system = platform.system()
    if system == "Windows":
        return Win32Windows()
    if system == "Linux":
        return NoWindowSystem(
            "рамку окна на X11 спросить нечем: путь через python-xlib/xdotool не "
            "написан. Записывайте экран целиком (--source display) либо задайте "
            "рамку руками (--source region --region ЛЕВО ВЕРХ ШИРИНА ВЫСОТА)")
    if system == "Darwin":
        return NoWindowSystem(
            "рамку окна на macOS спросить нечем: ScreenCaptureKit не написан. "
            "Записывайте экран целиком (--source display) либо задайте рамку руками "
            "(--source region)")
    return NoWindowSystem(f"система {system} неизвестна: рамку окна спросить нечем")


Resolved = tuple  # (Region | None, WindowTarget | None, str) — рамка, окно, происхождение


def resolve(kind: str, *, region: Region | None = None, window: str = "",
            windows: WindowSystem | None = None) -> Resolved:
    """Что снимать: рамка, окно и **словами** — откуда рамка взялась.

    Третье значение возвращается не для красоты вывода. Рамка, попавшая в запись без
    происхождения, через месяц неотличима от рамки, взятой наугад: «960×540» не говорит,
    было это окно браузера, половина монитора или ошибка масштабирования.
    """
    if kind not in KINDS:
        raise SourceError(f"нет такого источника: {kind!r}; есть {list(KINDS)}")
    if kind == "display":
        if region is not None or window:
            raise SourceError(
                "источник «display» — весь экран, рамка и окно к нему не задаются. "
                "Для части экрана есть «region», для окна — «window»")
        return None, None, "весь экран монитора целиком"
    if kind == "region":
        if region is None:
            raise SourceError(
                "источник «region» требует рамку: --region ЛЕВО ВЕРХ ШИРИНА ВЫСОТА")
        if window:
            raise SourceError("«region» и окно вместе не задаются: рамка уже задана")
        return region, None, f"рамка задана исследователем: {region.as_tuple()}"
    if not window:
        raise SourceError(
            "источник «window» требует часть заголовка окна: --window ЧАСТЬ. "
            "Заголовок нужен, чтобы указать окно, и в кадр агента он не попадает")
    if region is not None:
        raise SourceError("«window» и рамка вместе не задаются: рамку даёт система")
    ws = windows or system_windows()
    target = ws.find(window)
    return target.region, target, (
        f"рамка получена от системы ({ws.name}) по заголовку окна: "
        f"{target.region.as_tuple()}")


# ---------------------------------------------------------------------------
# Источник захвата против области действия ввода. TASK-21, часть 2
# ---------------------------------------------------------------------------
#
# Профиль по умолчанию говорит `capture_source: display` и `input_scope: window`: снимаем
# весь экран, действуем в одном окне. Это **решение**, а не недосмотр, и вот его основания.
#
# 1. Кадр обязан содержать то, что видит человек. Сузить захват до окна значит отнять у
#    агента контекст, который у человека есть: часы, уведомления, второе окно рядом. Хуже
#    того, сузить захват до окна — значит выдать ему границы окна готовыми (см. описание
#    модуля), то есть решить за него половину задачи разделения слоёв.
# 2. Ввод обязан быть уже кадра. Инъекция в области `device` означает, что агент может
#    нажать на что угодно на машине — включая окно исследователя, диалог подтверждения и
#    кнопку выключения. Пол осторожности (`CONTACT.md`) стоит не на выученной оценке
#    обратимости, а на категориях, и «щёлкнуть мимо» тут не бывает безобидным.
#
# Из этих двух вместе выходит третье, и оно ценно само по себе: часть кадра **меняется и
# не поддаётся влиянию**. У проекта для этого уже есть слово — «погода»
# (`devices.Device.is_weather`): её надо предсказывать, но не получится изменить. Это
# законный обучающий сигнал, а не дефект постановки.
#
# Обратное сочетание — снимать окно, а вводить шире — называется «невидимая рука»
# (`devices.Device.is_blind_hand`) и является дефектом: действие с последствием за краем
# поля зрения нельзя ни предсказать, ни откатить, и агент не сможет даже заметить, что
# натворил. Такое сочетание объявляется вслух.

#: Насколько область шире: чем меньше индекс, тем шире. Порядок тот же, что в
#: `inject.mask.SCOPES`, и сверяется с ним тестом — два порядка одного набора расходятся
#: молча, и тогда «уже» и «шире» меняются местами.
SCOPE_WIDTH: dict[str, int] = {"device": 0, "app": 1, "profile": 2, "window": 3}


def pairing(capture_source: str, input_scope: str) -> dict[str, Any]:
    """Что означает это сочетание источника захвата и области ввода.

    Возвращает `kind` (одно из четырёх), текст и признак `declared`: объявлено ли
    сочетание как осмысленное. Не проверка настроек на допустимость — сочетания все
    допустимы, — а **объяснение того, какой опыт получится**, напечатанное до записи.
    """
    if capture_source not in KINDS:
        raise SourceError(f"нет такого источника: {capture_source!r}")
    if input_scope not in SCOPE_WIDTH:
        raise SourceError(f"нет такой области ввода: {input_scope!r}")
    narrow = SCOPE_WIDTH[input_scope] >= SCOPE_WIDTH["window"]
    if capture_source == "display" and narrow:
        return {"kind": "погода", "declared": True,
                "text": "снимается весь экран, ввод ограничен окном. Часть кадра меняется "
                        "и не поддаётся влиянию — это «погода»: её надо предсказывать, "
                        "изменить нельзя. Сочетание объявлено осмысленным: кадр обязан "
                        "содержать то, что видит человек, а ввод обязан быть уже кадра"}
    if capture_source in ("window", "region") and not narrow:
        return {"kind": "невидимая рука", "declared": False,
                "text": f"снимается только "
                        f"{'окно' if capture_source == 'window' else 'рамка'}, "
                        f"а ввод идёт в области «{input_scope}», которая шире кадра. Это "
                        "«невидимая рука»: последствие действия окажется за краем поля "
                        "зрения, и агент не заметит, что натворил. Сочетание дефектное — "
                        "сузьте input_scope до window или снимайте экран целиком"}
    if capture_source in ("window", "region") and narrow:
        return {"kind": "совпадает", "declared": True,
                "text": "кадр и область ввода совпадают: агент видит ровно то, на что "
                        "влияет. Погоды в кадре нет вовсе — и это отдельное условие "
                        "опыта, а не просто «чище»: учиться отличать своё от чужого "
                        "будет негде"}
    return {"kind": "шире кадра нет", "declared": True,
            "text": f"снимается весь экран, ввод в области «{input_scope}» — тоже во весь "
                    "экран. Агент влияет на всё, что видит; погоды нет, но и защиты от "
                    "нажатия мимо тоже. Для записей с человеком за рулём это нормально, "
                    "для самостоятельных прогонов — нет"}


class WindowFollowing:
    """Источник, следящий за окном. Исчезновение окна — **событие мира**, а не сбой.

    TASK-21, часть 6. Окно можно закрыть, свернуть, перетащить на другой монитор. Для
    механизма захвата это три разных состояния API; для агента — одно происшествие: то,
    на что он смотрел, пропало. Смешивать его со сбоем захвата нельзя в обе стороны:

    - объявить исчезновение сбоем — значит записать разрыв там, где мир просто изменился,
      и обучать агента на разрыве, которого не было;
    - объявить сбой исчезновением — значит спрятать поломку механизма под видом события,
      и искать её потом в поведении агента.

    Поэтому здесь три разных ответа: `Frame` — кадр, `GONE` — окно пропало (наблюдение),
    исключение — механизм сломался (разрыв). Четвёртый, `UNCHANGED`, приходит от нижнего
    источника как есть: неизменённое окно — это неизменённый кадр, и ничего больше.

    **Переезд окна — не исчезновение.** Рамка перепрашивается каждый оборот, и если окно
    подвинули, кадр идёт из нового места. Это тоже решение, а не побочный эффект: «снимаю
    окно» означает «снимаю окно», а не «снимаю прямоугольник, где окно однажды было».
    """

    def __init__(self, inner: Any, target: WindowTarget, windows: WindowSystem, *,
                 name: str = "") -> None:
        self.inner = inner
        self.target = target
        self.windows = windows
        self.name = name or f"{getattr(inner, 'name', 'screen')}+window"
        self.gone = False
        self.moves = 0             # сколько раз окно переехало: это число нужно
        self.region = target.region

    def start(self) -> None:
        self.inner.start()

    def stop(self) -> None:
        self.inner.stop()

    def read(self) -> "Frame | Unchanged | Gone | None":
        if self.gone:
            # Один раз исчезло — второй раз об этом не сообщаем: событие мира случается
            # однажды, а не каждый оборот. Повтор превратил бы одно происшествие в поток.
            return None
        now = self.windows.rect(self.target.handle)
        if now is None:
            self.gone = True
            return GONE
        if now != self.region:
            self.region = now
            self.moves += 1
            # Рамка нижнего источника меняется на ходу: у обоих backend'ов она читается
            # при каждом `grab`, поэтому достаточно присвоить.
            self.inner.region = now.as_tuple()
        return self.inner.read()

    def frames(self):
        while True:
            f = self.read()
            if f is None or f is GONE:
                return
            if f is UNCHANGED:
                continue
            yield f

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "window": self.target.id,
                "region": self.region.as_dict(), "moves": self.moves,
                "gone": self.gone}
