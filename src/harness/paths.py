"""Пути в форме той системы, где их будут набирать. `TASK-11`, часть 3; `TASK-17`, пункт 5.

Тильда не раскрылась дважды, и второй раз попала в наблюдаемое из отчёта: оператор набрал

    harness record ~\\harness-live\\proba

и получил каталог с именем `~` **внутри проекта**, а следом печатную команду

    harness ingest ~\\harness-live\\proba --corpus C:\\Users\\chiha\\ai\\harness\\~\\harness-live\\corpus

Причина не в опечатке. `cmd.exe` и `powershell` тильду не раскрывают вовсе — это соглашение
`sh`. Значит раскрывать её обязана программа: `Path("~")` — это буквальное имя каталога, и
кто его не раскроет, тот его и создаст.

Отсюда два правила, и они разные:

1. **на входе** — `expanduser`, а буквальный `~` в середине пути (`ai/harness/~/...`) —
   громкий отказ: он означает, что раскрытия не случилось ни у оболочки, ни у нас, и
   молча писать в такой каталог нельзя;
2. **на выходе** — печатать в форме целевой системы: `C:\\Users\\chiha\\harness-live\\proba`,
   а не `~/harness-live/proba`. Строка для копирования обязана работать при вставке, а
   `~/` в `cmd.exe` не работает.
"""

from __future__ import annotations

from pathlib import Path, PurePath, PureWindowsPath

#: Каталог живых записей по умолчанию. Один на весь проект: три места, печатающие
#: «куда писать», разошлись бы молча.
LIVE_DIR_NAME = "harness-live"


class PathError(ValueError):
    """Путь, по которому нельзя писать, и сказано почему."""


def live_root(home: Path | None = None) -> Path:
    """Каталог живых записей: `~/harness-live`, уже раскрытый."""
    return (home or Path.home()) / LIVE_DIR_NAME


def live_path(name: str, home: Path | None = None) -> Path:
    return live_root(home) / name


def show(path: Path | PurePath | str, *, windows: bool | None = None) -> str:
    """Путь в форме целевой системы. Для печати оператору.

    `windows` задаётся только в проверках: в работе форма берётся у той системы, где идёт
    прогон, потому что печатается она тому, кто на ней и набирает.
    """
    if windows is None:
        return str(path)
    if not windows:
        return str(path).replace("\\", "/")
    # `PureWindowsPath` сам ставит разделители целевой системы; `as_posix` обратного не
    # делает, поэтому обратная сторона проверки — именно этот класс, а не замена строк.
    return str(PureWindowsPath(str(path).replace("/", "\\")))


def resolve_input(raw: str | Path, *, home: Path | None = None) -> Path:
    """Путь, введённый человеком, — в путь, по которому можно писать.

    Раскрывает `~` и `~user`. Отказывает, если `~` стоит **не** в начале: такой путь
    означает, что тильду не раскрыл никто, и записывать в каталог с именем `~` значит
    прятать запись там, где её никто не найдёт.
    """
    p = Path(raw)
    parts = p.parts
    inner = [i for i, part in enumerate(parts)
             if i > 0 and (part == "~" or part.startswith("~"))]
    if inner:
        bad = parts[inner[0]]
        good = Path(*parts[:inner[0]]) if inner[0] else Path()
        raise PathError(
            f"в пути {show(p)} компонент {bad!r} стоит не в начале, то есть тильду никто "
            "не раскрыл: `cmd.exe` и `powershell` этого не делают.\n"
            f"Домашний каталог здесь — {show((home or Path.home()))}. "
            f"Наберите путь целиком:\n"
            f"  {show(live_path('проба', home))}\n"
            f"или относительный от {show(good) or '.'} без тильды")
    try:
        expanded = p.expanduser()
    except RuntimeError as e:                  # домашнего каталога не видно
        raise PathError(f"не удалось раскрыть {show(p)}: {e}") from None
    if home is not None and parts and parts[0] in ("~",):
        expanded = home / Path(*parts[1:])
    return expanded
