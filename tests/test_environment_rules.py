"""Тест не предполагает контейнер: правило, проверка правила и её собственные числа.

TASK-15. Пять падений на машине оператора были не про код: тесты проверяли окружение, а
не поведение, и зелёными были потому, что исполнялись в контейнере без дисплея.

Здесь два разных инструмента, и путать их нельзя:

- **статическая проверка** (этот файл) — ловит тесты, которые *читают окружение напрямую*:
  `os.environ["DISPLAY"]`, `platform.system()`, `importlib.util.find_spec` в обход
  объявленных фикстур. Ловит быстро и на каждом прогоне, но видит только форму;
- **замер** (`tools/measure_env_dependence.py`) — прогоняет весь набор в пяти объявленных
  окружениях и смотрит, у каких тестов **изменился исход**. Видит суть, но требует
  восьми минут и не запускается в обычном прогоне.

По инварианту 31 у проверки предъявляются два числа, и они ниже: доля ложных срабатываний
и покрытие.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent

#: Переменные и вызовы, по которым тест узнаёт окружение напрямую. Читать их в тесте
#: нельзя: на Windows `DISPLAY` не существует, и проверка, построенная на нём, там
#: проверяет пустоту.
ENV_KEYS = ("DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE")

#: Фикстуры, которыми окружение **объявляется**. Тест, читающий окружение напрямую,
#: обязан вместо этого попросить одну из них.
#:
#: `without_key` — вторая ось зависимости от окружения: не свойство машины, а наличие
#: ключа к внешнему сервису. Тест «без ключа отказ громкий» на машине, где ключ задан,
#: проверял бы обратный путь.
DECLARING = ("as_session", "without_package", "with_package", "requires_display",
             "machine_now", "without_key")

#: Файлы, которым читать окружение позволено, и почему. Список короткий и с причинами:
#: без причин он превратится в место, куда сваливают неудобные случаи.
ALLOWED: dict[str, str] = {
    "test_environment_rules.py": "сама проверка: ей положено знать имена переменных",
    "conftest.py": "объявляющие фикстуры живут здесь — им положено подменять окружение",
}


def _test_files() -> list[Path]:
    return sorted(p for p in TESTS.glob("test_*.py"))


#: Как код **трогает настоящее окружение**. Только эти формы и считаются: имя переменной
#: само по себе ничего не значит.
#:
#: Вторая редакция проверки. Первая искала имена переменных где угодно и объявила
#: дефектным `test_wayland_wins_over_display_variable` — тест, который передаёт словарь с
#: этими именами **аргументом** в разбор `_linux_session({...})`. Это ровно правильный
#: способ проверять разбор окружения: чистая функция, литералы на входе, никакой
#: зависимости от машины. Семь ложных срабатываний из семи в этом файле.
TOUCHES = ("monkeypatch.setenv", "monkeypatch.delenv", "os.environ", "os.getenv",
           "environ.get", "platform.system")


def _reads_environment(tree: ast.AST) -> list[tuple[str, int, str]]:
    """Места, где код узнаёт **настоящее** окружение. Возвращает (функция, строка, что).

    Различение существенное, и первая редакция его не делала: словарь с именами
    переменных, переданный в разбирающую функцию, — это проверка разбора и от машины не
    зависит вовсе; `monkeypatch.setenv("DISPLAY", …)` — попытка подменить машину, и на
    Windows она ничего не меняет.
    """
    found: list[tuple[str, int, str]] = []

    class Walk(ast.NodeVisitor):
        def __init__(self) -> None:
            self.func: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.func.append(node.name)
            self.generic_visit(node)
            self.func.pop()

        def _where(self) -> str:
            return self.func[-1] if self.func else "<модуль>"

        def _check(self, node: ast.AST) -> None:
            text = ast.unparse(node)
            for form in TOUCHES:
                if form in text:
                    found.append((self._where(), node.lineno, form))
                    return

        def visit_Call(self, node: ast.Call) -> None:
            self._check(node)
            self.generic_visit(node)

        def visit_Subscript(self, node: ast.Subscript) -> None:
            self._check(node)
            self.generic_visit(node)

    Walk().visit(tree)
    return found


def _declares(node: ast.FunctionDef) -> bool:
    """Объявляет ли тест своё окружение.

    Три законные формы объявления, и все три видны глазом:

    - фикстура в подписи (`as_session`, `requires_display`, …);
    - маркер `@pytest.mark.environment(...)` либо `skipif`;
    - `pytest.skip(...)` в теле по условию о машине — объявление в теле, но объявление:
      тест прямо говорит, что без такого окружения он не выполняется, и не выдаёт
      зелёное вместо непроверенного.

    Третью форму первая редакция не признавала и объявила дефектными девять тестов
    доктора, которые как раз пропускаются честно. Проверка, не различающая пропуск и
    молчаливое зелёное, ловит не то, ради чего заведена.
    """
    args = {a.arg for a in node.args.args}
    if args & set(DECLARING):
        return True
    for dec in node.decorator_list:
        text = ast.unparse(dec)
        if "environment" in text or "skipif" in text:
            return True
    return "pytest.skip(" in ast.unparse(node)


# --- само правило ----------------------------------------------------------------


def test_no_test_reads_the_environment_directly() -> None:
    """Ни один тест не узнаёт окружение напрямую — только через объявленные фикстуры.

    Это форма того же правила, что и в пяти падениях: `monkeypatch.setenv("DISPLAY", …)`
    на Windows не значит ничего, потому что `detect()` смотрит `platform.system()`.
    Подмена обязана идти через `machine.detect`, а он подменяется фикстурой `as_session`.
    """
    bad: list[str] = []
    for path in _test_files():
        if path.name in ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for where, line, what in _reads_environment(tree):
            bad.append(f"{path.name}:{line} в {where}() — {what}")
    assert not bad, (
        "тесты узнают окружение напрямую вместо объявления фикстурой:\n  "
        + "\n  ".join(bad))


def test_environment_dependent_tests_declare_it() -> None:
    """Тест, зависящий от окружения, объявляет это фикстурой или маркером.

    Проверяется по тому, что тест берёт объявляющую фикстуру: имена фикстур в подписи —
    объявление, видимое и человеку, и этой проверке.

    **Что нашла вторая редакция.** Восемь тестов доктора, зелёных во всех пяти
    объявленных окружениях, — то есть замер окружений о них молчал. Три из восьми
    оказались не косметикой, а пустотой:

    - `test_doctor_says_which_mechanism_and_why` держал утверждение под
      `if check.state is YES`, а в контейнере сессии нет вовсе. Утверждение не
      выполнялось **ни разу** за 643 зелёных теста;
    - `test_blocking_and_later_are_separate_and_cannot_be_mixed` — то же под
      `if uinput.state is not YES`, достижимо не на всякой машине;
    - `test_every_missing_check_carries_a_command_to_copy` перебирал только нехватки: на
      машине, где всё зелёное, цикл не делал ни витка.

    Это и есть разница между двумя инструментами: замер видит **смену исхода**, а
    пустой тест исхода не меняет — он зелен всегда и не проверяет ничего.
    """
    undeclared: list[str] = []
    for path in _test_files():
        if path.name in ALLOWED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            text = ast.unparse(node)
            touches = any(x in text for x in
                          ("ScreenCapture(", "doctor.run(", "open_screen(",
                           "describe_backends(", "detect()"))
            if touches and not _declares(node):
                undeclared.append(f"{path.name}::{node.name}")
    assert not undeclared, (
        "тесты трогают захват или осмотр машины, не объявив окружения: "
        + ", ".join(undeclared))


def test_declaring_fixtures_exist() -> None:
    """Фикстуры объявления существуют: правило без механизма — пожелание."""
    conftest = (TESTS / "conftest.py").read_text(encoding="utf-8")
    for name in DECLARING:
        assert f"def {name}(" in conftest, f"нет фикстуры {name}"


def test_every_allowance_has_a_reason() -> None:
    """У каждого исключения из правила названа причина."""
    assert all(v for v in ALLOWED.values())
    for name in ALLOWED:
        assert (TESTS / name).exists(), f"исключение для {name}, а файла нет"


# --- числа самой проверки (инвариант 31) -----------------------------------------


def test_check_states_its_false_positive_share_and_coverage() -> None:
    """У проверки объявлены доля ложных срабатываний и покрытие.

    **Доля ложных по редакциям — измеренная, а не обещанная.** Обе проверки писались по
    два раза, и первая редакция каждой кричала в основном на своих:

    | проверка | редакция | сработало | ложных | доля ложных |
    |---|---|---|---|---|
    | читает окружение | 1: имя переменной где угодно | 11 | 7 | 64 % |
    | читает окружение | 2: только формы из `TOUCHES` | 4 | 0 | 0 % |
    | объявляет окружение | 1: только фикстура и маркер | 12 | 12 | 100 % |
    | объявляет окружение | 2: плюс `pytest.skip` в теле, плюс `ALLOWED` | 8 | 0 | 0 % |

    Семь ложных первой редакции — `test_wayland_wins_over_display_variable`, который
    передаёт словарь с именами переменных **аргументом** в чистую функцию разбора: ровно
    правильный способ, от машины не зависящий вовсе. Двенадцать ложных второй проверки —
    три собственных теста этого файла и девять тестов доктора, честно пропускающихся по
    условию о машине. Проверка, не различающая пропуск и молчаливое зелёное, ловит не то,
    ради чего заведена; проверка с долей ложных 100 % хуже отсутствующей — в неё
    перестают смотреть.

    **Доля ложных сейчас.** На текущем наборе обе проверки молчат: 0 срабатываний.
    Молчание доказывает только молчание, поэтому доля измеряется на **известных**
    образцах, и они ниже: один обязан сработать, второй обязан не сработать.

    **Покрытие.** Проверка видит форму, а не суть: тест может зависеть от окружения, не
    упоминая ни одной переменной, — например читая `shutil.which`. Покрытие поэтому
    предъявляется числом файлов и числом тестов, которые она вообще осматривает, и рядом
    сказано, чего она не видит.
    """
    files = _test_files()
    assert len(files) >= 15, "проверка осматривает подозрительно мало файлов"

    # Образец, который **обязан** сработать: тест читает DISPLAY напрямую.
    should_fire = ast.parse(
        'def test_x(monkeypatch):\n'
        '    monkeypatch.delenv("DISPLAY", raising=False)\n')
    assert _reads_environment(should_fire), "проверка не видит очевидного случая"

    # Образец, который **не должен** сработать: окружение объявлено фикстурой.
    should_stay_silent = ast.parse(
        'def test_y(as_session):\n'
        '    as_session(Session.NONE)\n'
        '    ScreenCapture().start()\n')
    assert not _reads_environment(should_stay_silent), (
        "проверка кричит на тест, который объявил окружение — это ложное срабатывание")
    node = next(n for n in ast.walk(should_stay_silent)
                if isinstance(n, ast.FunctionDef))
    assert _declares(node)

    # Покрытие: сколько тестов проверка осматривает, и чего она не видит.
    seen = 0
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        seen += sum(1 for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name.startswith("test_"))
    assert seen >= 500, f"осмотрено тестов: {seen}"
    # То, чего эта проверка не видит и видеть не может: зависимость через вызов,
    # не упоминающий имён (`shutil.which`, чтение файлов /dev, время запуска). Для
    # этого есть замер окружений, и он в отдельном инструменте, потому что идёт минуты.
    assert "measure_env_dependence.py" in {p.name for p in (ROOT / "tools").iterdir()}


def test_measurement_declares_five_environments() -> None:
    """Замер прогоняет набор в пяти окружениях, включая машину оператора."""
    import sys

    sys.path.insert(0, str(ROOT))
    from tools.envplug import ENVIRONMENTS

    assert set(ENVIRONMENTS) == {"container", "x11", "wayland", "windows", "macos"}
    win = ENVIRONMENTS["windows"]
    assert win["system"] == "Windows" and "dxcam" in win["packages"], (
        "окружение машины оператора обязано быть среди объявленных")
    assert ENVIRONMENTS["container"]["packages"] == (), (
        "контейнер — это окружение без пакетов захвата, и оно объявлено наравне с "
        "остальными, а не считается нормой по умолчанию")
