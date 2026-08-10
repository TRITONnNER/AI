"""Плагин pytest: подменить окружение на объявленное и прогнать набор целиком.

Зачем это существует. Первый прогон на машине оператора уронил три теста, которые
проверяли не поведение, а окружение: они были зелёными только потому, что исполнялись в
контейнере без дисплея. Из этого следует вопрос, на который ответа не было: **сколько
ещё таких тестов?** 636 зелёных рядом с тремя такими падениями означают, что доля
ложно-зелёных неизвестна (инвариант 31).

Ответ можно только измерить, и способ один: прогнать тот же набор в **другом**
объявленном окружении и посмотреть, у каких тестов изменился исход. Тест, чей исход
зависит от машины, но который об этом не объявил, — дефект того же класса, что и три
найденных.

Как пользоваться:

    HARNESS_FAKE_ENV=windows PYTHONPATH=src pytest -p tools.envplug -q

Окружения объявлены в `ENVIRONMENTS`. Подменяется `machine.detect` — единственное место,
где проект отвечает на вопрос «что это за машина», — и наличие пакетов захвата.
Переменные окружения не подменяются: на Windows их нет вовсе, и подмена их ничего не
значила бы (ровно эта ошибка и была в трёх тестах).

**Фикстуры тестов сильнее плагина.** Плагин подменяет окружение на старте сессии, фикстура
`as_session` — внутри теста, то есть позже. Поэтому тест, объявивший своё окружение,
плагином не задет, и это правильно: он и не должен зависеть от машины.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import Any

#: Объявленные окружения: сессия, система, какие пакеты считать установленными.
#:
#: `container` — то, в чём идёт разработка: Linux без графической сессии, пакетов захвата
#: нет. `windows` — машина оператора: Windows 10, дисплей есть, dxcam и mss установлены.
ENVIRONMENTS: dict[str, dict[str, Any]] = {
    "container": {"session": "none", "system": "Linux", "packages": ()},
    "x11": {"session": "x11", "system": "Linux", "packages": ("mss",)},
    "wayland": {"session": "wayland", "system": "Linux", "packages": ("mss",)},
    "windows": {"session": "windows", "system": "Windows",
                "packages": ("dxcam", "mss")},
    "macos": {"session": "macos", "system": "Darwin", "packages": ("mss",)},
}


def _fake_module(name: str) -> types.ModuleType:
    """Пустышка вместо пакета захвата.

    Она **не работает** — и не должна: проверяется не захват, а то, зависит ли исход
    теста от того, установлен пакет или нет. Пустышка, которая делала бы вид, что
    захватывает, была бы той самой молчаливой заглушкой, которую проект запрещает.
    """
    mod = types.ModuleType(name)

    def unavailable(*a: object, **kw: object) -> None:
        raise RuntimeError(
            f"{name}: это пустышка плагина окружения, захвата в ней нет. "
            "Она отвечает только на вопрос «установлен ли пакет»")

    for attr in ("create", "mss", "MSS", "grab"):
        setattr(mod, attr, unavailable)
    mod.__harness_fake__ = True
    return mod


def pytest_configure(config: Any) -> None:
    name = os.environ.get("HARNESS_FAKE_ENV", "").strip()
    if not name:
        return
    if name not in ENVIRONMENTS:
        raise SystemExit(
            f"неизвестное окружение {name!r}. Объявлены: {sorted(ENVIRONMENTS)}")
    spec = ENVIRONMENTS[name]

    sys.path.insert(0, str(os.path.join(os.getcwd(), "src")))
    from harness import machine as machine_mod
    from harness.machine import Machine, Session

    fake = Machine(system=str(spec["system"]), session=Session(spec["session"]),
                   session_source=f"подменено плагином: окружение {name}",
                   python=sys.version_info[:3], release="плагин")

    def detect() -> Machine:
        return fake

    machine_mod.detect = detect
    for mod_name in ("harness.capture.select", "harness.doctor", "harness.selftest"):
        try:
            mod = __import__(mod_name, fromlist=["detect"])
        except Exception:
            continue
        if hasattr(mod, "detect"):
            mod.detect = detect

    wanted = set(spec["packages"])
    for pkg in wanted:
        if pkg not in sys.modules:
            sys.modules[pkg] = _fake_module(pkg)
    real_find = importlib.util.find_spec

    def find_spec(pkg: str, package: str | None = None):
        head = pkg.split(".")[0]
        if head in wanted:
            return importlib.util.spec_from_loader(head, loader=None)
        if head in ("dxcam", "mss", "sounddevice") and head not in wanted:
            return None
        return real_find(pkg, package)

    importlib.util.find_spec = find_spec
    config.stash["harness_fake_env"] = name


def pytest_report_header(config: Any) -> str | None:
    name = os.environ.get("HARNESS_FAKE_ENV", "").strip()
    if not name:
        return None
    spec = ENVIRONMENTS[name]
    return (f"окружение подменено: {name} — сессия {spec['session']}, "
            f"система {spec['system']}, пакеты {spec['packages'] or 'нет'}")
