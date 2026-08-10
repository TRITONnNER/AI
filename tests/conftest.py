"""Общая обвязка тестов.

Пакет лежит в `src/`, поэтому путь добавляется здесь: тесты должны запускаться
через `pytest` без установки пакета, иначе «проверить на записанной сессии»
превращается в ритуал.

Синтетический корпус генерируется один раз на весь прогон: он детерминирован по
сиду, а генерация — единственная заметно небыстрая операция в этих тестах.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Синтетическая сессия с известной истиной. Только для чтения."""
    from harness.corpus.synthetic import generate_session

    root = tmp_path_factory.mktemp("corpus") / "synthetic-0"
    generate_session(root, seed=7)
    return root


@pytest.fixture()
def session(corpus: Path):
    from harness.session import Session

    s = Session.open(corpus)
    yield s
    s.close()


@pytest.fixture()
def profile():
    from harness.core.profile import MILESTONE_0

    return MILESTONE_0


# ---------------------------------------------------------------------------
# Объявление окружения. Инвариант: тест не предполагает контейнер
# ---------------------------------------------------------------------------
#
# Первый прогон на машине оператора (Windows 10, дисплей есть) уронил три теста, и ни
# один из них не был про код: они сравнивали поведение с наличием `DISPLAY`, ждали
# отказа от установленного пакета и проверяли, что записи не вышло. То есть проверяли
# **окружение**, а не поведение, и зелёными были только потому, что исполнялись в
# контейнере без дисплея.
#
# Два утверждения, которые надо разделить навсегда:
#
# - «без графической сессии код отказывает громко» — про поведение. Проверяется
#   **подменой** окружения и обязано проходить на любой машине;
# - «на этой машине графической сессии нет» — свойство машины. Предметом теста быть не
#   может вообще: сегодня контейнер, завтра ноутбук оператора.
#
# Удаление `DISPLAY`/`WAYLAND_DISPLAY` подменой окружения **не является**: на Windows
# этих переменных нет вовсе, и графическая сессия определяется через `platform.system()`.
# Тест, удаляющий переменные, на Windows не меняет ничего и продолжает проверять машину.
# Поэтому подмена делается на уровне `machine.detect` — единственного места, где вся
# система отвечает на вопрос «что это за машина».

import os


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "environment(name): тест зависит от окружения и объявляет какого. "
        "Объявление обязательно: без него тест проверяет машину, а не поведение")


@pytest.fixture()
def as_session(monkeypatch: pytest.MonkeyPatch):
    """Объявить, какая графическая сессия у машины. Работает на любой платформе.

    Подменяется `harness.machine.detect`, а не переменные окружения: переменные — это
    способ узнать сессию **на Linux**, и подмена их на Windows не делает ничего.

    Возвращает функцию: `as_session(Session.NONE)` — и весь код проекта видит машину без
    экрана независимо от того, на чём идёт прогон.
    """
    from harness import machine as machine_mod
    from harness.machine import Machine, Session

    def force(session: "Session", *, system: str | None = None) -> Machine:
        sys_name = system or {Session.WINDOWS: "Windows", Session.MACOS: "Darwin"}.get(
            session, "Linux")
        fake = Machine(system=sys_name, session=session,
                       session_source=f"подменено тестом: {session}",
                       python=(3, 11, 0), release="тест")
        monkeypatch.setattr(machine_mod, "detect", lambda: fake)
        # Модули, импортировавшие `detect` по имени, держат свою ссылку — их тоже надо
        # подменить, иначе подмена окажется частичной и тест снова начнёт зависеть от
        # того, кто как импортировал.
        for mod_name in ("harness.capture.select", "harness.doctor",
                         "harness.selftest"):
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, "detect"):
                monkeypatch.setattr(mod, "detect", lambda: fake, raising=False)
        return fake

    return force


@pytest.fixture()
def without_package(monkeypatch: pytest.MonkeyPatch):
    """Объявить, что пакета нет, — не полагаясь на то, что его нет на самом деле.

    `dxcam` установлен на машине оператора и работает (TASK-08). Тест «недоступный
    backend отказывает громко» обязан проверять **отказ**, а не отсутствие пакета,
    поэтому отсутствие объявляется здесь: импорт падает, `find_spec` возвращает `None`.
    """
    import builtins
    import importlib.util

    real_import = builtins.__import__
    real_find = importlib.util.find_spec

    def force(*names: str) -> None:
        gone = set(names)

        def fake_import(name, *a, **kw):
            if name.split(".")[0] in gone:
                raise ImportError(f"нет пакета {name} (объявлено тестом)")
            return real_import(name, *a, **kw)

        def fake_find(name, package=None):
            if name.split(".")[0] in gone:
                return None
            return real_find(name, package)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setattr(importlib.util, "find_spec", fake_find)
        for name in gone:
            monkeypatch.delitem(sys.modules, name, raising=False)

    return force


@pytest.fixture()
def with_package(monkeypatch: pytest.MonkeyPatch):
    """Объявить, что пакет есть, подсунув модуль. Обратная сторона `without_package`.

    Нужна для проверки путей, доступных только там, где пакет установлен: иначе такой
    путь проверяется только на машине оператора, то есть не проверяется.
    """
    import importlib.util
    import types

    real_find = importlib.util.find_spec

    def force(name: str, **attrs: object) -> types.ModuleType:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        monkeypatch.setitem(sys.modules, name, mod)
        spec = importlib.util.spec_from_loader(name, loader=None)
        monkeypatch.setattr(
            importlib.util, "find_spec",
            lambda n, package=None: spec if n.split(".")[0] == name
            else real_find(n, package))
        return mod

    return force


@pytest.fixture()
def without_key(monkeypatch: pytest.MonkeyPatch):
    """Объявить, что ключа к внешнему сервису нет, — не полагаясь на то, что его нет.

    Второй вид зависимости от окружения, помимо машины: переменная с ключом. Тест
    «без ключа описатель отказывается громко» на машине, где ключ **есть**, проверял бы
    ровно обратный путь и был бы зелёным по другой причине. Это тот же класс дефекта,
    что и пять падений TASK-15, только ось другая: не свойство машины, а свойство
    настроек оператора.

    Подмена через `monkeypatch.delenv`, а не через `os.environ.pop` с восстановлением в
    `finally`: восстановление руками отваливается при падении внутри и при вложенных
    подменах, и тогда следующий тест получает уже испорченное окружение.
    """
    def force(*names: str) -> tuple[str, ...]:
        for name in names:
            monkeypatch.delenv(name, raising=False)
        return names

    return force


@pytest.fixture(scope="session")
def machine_now():
    """Настоящая машина, на которой идёт прогон. Только для того, чтобы пропускаться."""
    from harness.machine import detect

    return detect()


@pytest.fixture()
def requires_display(machine_now):
    """Пропустить тест, если графической сессии нет. Объявление, а не предположение.

    Тест, которому нужен настоящий экран, пропускается там, где его нет, и **не**
    выдаёт зелёное: зелёное означало бы, что проверка прошла, а она не выполнялась.
    """
    from harness.machine import Session

    if machine_now.session is Session.NONE:
        pytest.skip(f"нужна графическая сессия, её нет: {machine_now.session_source}")
    return machine_now
