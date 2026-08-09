"""Граница восприятия для текста. Инвариант 5, `TASK-06`, часть 3.

Проверяется не качество распознавания, а **непроницаемость границы**: ни одна
настоящая строка не должна её пересечь. Для этого текст известен точно — иначе
непонятно, дошла строка или просто не распозналась.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harness.core.symbols import (Namespace, SOURCE_TRUST, Symbolizer,
                                  is_source_tag, is_symbol)
from harness.perception.text import (ReaderUnavailable, ScriptedReader, TextArea,
                                     TextBoundary, TextError, TesseractReader,
                                     SymbolObservation)

# Настоящие строки, которые не должны появиться ни в одном возвращаемом значении.
LINES = ["Осталось 3 патрона", "Инвентарь полон", "Press E to open",
         "Игрок2: беги оттуда", "нажми чтобы продолжить"]


def _frame() -> np.ndarray:
    return np.full((180, 320), 128, dtype=np.uint8)


def _areas() -> list[TextArea]:
    return [TextArea(4, 4, 20, 120, Namespace.WORLD),
            TextArea(30, 4, 20, 120, Namespace.UI),
            TextArea(60, 4, 20, 120, Namespace.CHAT),
            TextArea(90, 4, 20, 120, Namespace.WEB),
            TextArea(120, 4, 20, 120, Namespace.WORLD)]


def _boundary(debug=None) -> TextBoundary:
    return TextBoundary(Symbolizer("линия-1"), ScriptedReader(LINES), debug=debug)


def test_no_real_string_crosses_the_boundary() -> None:
    """Главный критерий: ни одна настоящая строка не доходит до планировщика."""
    b = _boundary()
    obs = b.read(_frame(), _areas(), at_frame=7)
    assert len(obs) == 5

    # Всё, что уходит дальше, сериализуется и проверяется целиком, а не по полям:
    # новое поле, добавленное однажды, тоже попадёт под проверку.
    blob = json.dumps([o.as_dict() for o in obs], ensure_ascii=False)
    for line in LINES:
        assert line not in blob, f"надпись {line!r} пересекла границу восприятия"
        for word in line.split():
            if len(word) > 3:
                assert word not in blob, f"слово {word!r} утекло за границу"
    assert all(is_symbol(o.symbol) for o in obs)


def test_payload_passes_the_plain_text_check() -> None:
    """То же самое той проверкой, которой пользуется журнал (инвариант 5)."""
    from harness.core.symbols import assert_no_plain_text

    obs = _boundary().read(_frame(), _areas(), at_frame=1)
    assert_no_plain_text({"symbols": [o.as_dict() for o in obs]},
                         path="perception")


def test_observation_refuses_a_readable_string() -> None:
    """Наблюдение с читаемой строкой не собирается вообще."""
    tag = Symbolizer("линия-1").source_tag(Namespace.WORLD)
    with pytest.raises(TextError, match="не символ"):
        SymbolObservation("Осталось 3 патрона", tag, 1.0, 0)


def test_observation_refuses_a_readable_source() -> None:
    """Имя источника — такая же разметка, как координаты (инвариант 4)."""
    with pytest.raises(TextError, match="не метка источника"):
        SymbolObservation("SYM_1A2B", "chat", 0.4, 0)


def test_source_name_never_crosses_the_boundary() -> None:
    """Ни в наблюдении, ни в сводке границы нет слова `chat`, `ui`, `world`, `web`."""
    b = _boundary()
    obs = b.read(_frame(), _areas(), at_frame=1)
    blob = json.dumps([o.as_dict() for o in obs] + [b.stats.as_dict()],
                      ensure_ascii=False)
    for ns in Namespace:
        assert str(ns) not in blob, (
            f"имя источника {str(ns)!r} пересекло границу: это готовая "
            "классификация каналов, то есть разметка")
    assert all(is_source_tag(o.source) for o in obs)


def test_sources_are_separated_by_the_tag() -> None:
    """Метка непрозрачна, но разделяет: четыре источника — четыре метки."""
    z = Symbolizer("линия-1")
    tags = {z.source_tag(ns) for ns in Namespace}
    assert len(tags) == len(list(Namespace)), (
        "метки источников склеились: два канала с равным доверием — мир и "
        "интерфейс — стали бы неразличимы")
    # Стабильность в пределах соли и различие между солями — как у символов.
    assert z.source_tag(Namespace.CHAT) == Symbolizer("линия-1").source_tag(Namespace.CHAT)
    assert z.source_tag(Namespace.CHAT) != Symbolizer("линия-2").source_tag(Namespace.CHAT)


def test_same_text_in_two_namespaces_gives_two_symbols() -> None:
    """Экран и чат — разные пространства: склеивать их нельзя.

    «Осталось 3 патрона» своими глазами и то же в чате от другого игрока — разные
    утверждения с разным доверием.
    """
    z = Symbolizer("линия-1")
    text = "Осталось 3 патрона"
    seen = {z.symbolize(text, ns) for ns in Namespace}
    assert len(seen) == len(list(Namespace)), (
        "пространства имён не разделяют символы: доверие к чужим словам "
        "оказалось бы равно доверию к своим глазам")


def test_trust_comes_from_the_source() -> None:
    z = Symbolizer("линия-1")
    obs = _boundary().read(_frame(), _areas(), at_frame=1)
    by_tag = {o.source: o.trust for o in obs}
    assert by_tag[z.source_tag(Namespace.WORLD)] == SOURCE_TRUST[Namespace.WORLD] == 1.0
    assert by_tag[z.source_tag(Namespace.CHAT)] < by_tag[z.source_tag(Namespace.WORLD)], (
        "чужая речь обязана иметь доверие ниже, чем свои глаза")
    assert by_tag[z.source_tag(Namespace.WEB)] < by_tag[z.source_tag(Namespace.CHAT)]


def test_namespace_is_not_visible_in_the_symbol() -> None:
    """Иначе агент получил бы читаемую классификацию источников — то есть разметку."""
    z = Symbolizer("линия-1")
    for ns in Namespace:
        sym = z.symbolize("что-нибудь", ns)
        assert str(ns) not in sym.lower()
        assert is_symbol(sym), f"{sym} не соответствует непрозрачному виду"


def test_decryption_goes_to_the_debug_channel_only(tmp_path: Path) -> None:
    """По отладочной таблице видно, какие символы чему соответствуют."""
    from harness.debug.channel import DebugChannel

    ch = DebugChannel(tmp_path / "debug", mode="a")
    b = _boundary(debug=ch)
    obs = b.read(_frame(), _areas(), at_frame=3)
    ch.close()

    table = (tmp_path / "debug" / "symbols.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(x) for x in table.splitlines() if x.strip()]
    texts = [r for r in rows if r["kind"] == "text"]
    assert len(texts) == 5, "таблица заземления обязана содержать все пять надписей"
    # В таблице настоящий текст есть — она исследовательская.
    assert any(r["text"] == "Осталось 3 патрона" for r in texts)
    # И символы в ней те же, что уехали агенту.
    assert {r["symbol"] for r in texts} == {o.symbol for o in obs}
    # Метки источников расшифровываются там же: без этого исследователь видел бы
    # четыре непрозрачных канала и не знал, который из них чат.
    sources = {r["symbol"]: r["text"] for r in rows if r["kind"] == "source"}
    assert sources == {b.symbolizer.source_tag(ns): str(ns)
                       for ns in (Namespace.WORLD, Namespace.UI,
                                  Namespace.CHAT, Namespace.WEB)}


def test_grounding_table_starts_empty_and_fills_within_a_lineage() -> None:
    """Хеш стабилен между линиями, заземление — нет: каждая линия узнаёт сама."""
    b = _boundary()
    assert b.grounded == 0, "таблица заземления обязана стартовать пустой"
    b.read(_frame(), _areas(), at_frame=1)
    assert b.grounded == 5
    # Повторное чтение тех же надписей новых символов не добавляет.
    b.read(_frame(), _areas(), at_frame=2)
    assert b.grounded == 5


def test_hash_is_stable_across_lineages_but_grounding_is_not() -> None:
    """Одна и та же строка даёт один символ в разных линиях при том же salt_id."""
    a, b = Symbolizer("линия-1"), Symbolizer("линия-1")
    assert a.symbolize("Инвентарь полон") == b.symbolize("Инвентарь полон")
    other = Symbolizer("линия-2")
    assert other.symbolize("Инвентарь полон") != a.symbolize("Инвентарь полон"), (
        "разные соли обязаны давать разные символы, иначе профили не различимы")


def test_ablation_mode_does_not_leak_through_the_boundary() -> None:
    """`text_symbolized=False` — отдельная линия, а не дыра в инварианте 5."""
    from harness.core.profile import from_schema

    profile = from_schema("ablation", capture_width=64, capture_height=64,
                          text_symbolized=False)
    z = Symbolizer.from_profile(profile)
    assert not z.enabled
    b = TextBoundary(z, ScriptedReader(LINES))
    with pytest.raises(TextError, match="граница восприятия текст не пропускает"):
        b.read(_frame(), _areas()[:1], at_frame=1)


def test_missing_reader_refuses_loudly() -> None:
    """Пустая строка вместо отказа дала бы зелёный отчёт над сломанным чтением."""
    reader = TesseractReader()
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        with pytest.raises(ReaderUnavailable, match="разные утверждения"):
            reader.read(_frame())
    else:
        pytest.skip("pytesseract установлен: отказ проверить нечем")


def test_empty_area_is_not_an_error_and_is_counted() -> None:
    b = TextBoundary(Symbolizer("линия-1"), ScriptedReader(["", "  ", "видно"]))
    obs = b.read(_frame(), _areas()[:3], at_frame=1)
    assert len(obs) == 1
    assert b.stats.empty == 2 and b.stats.symbolized == 1


def test_stats_carry_numbers_not_strings() -> None:
    b = _boundary()
    b.read(_frame(), _areas(), at_frame=1)
    blob = json.dumps(b.stats.as_dict(), ensure_ascii=False)
    for line in LINES:
        assert line not in blob
    z = Symbolizer("линия-1")
    assert b.stats.by_source == {z.source_tag(Namespace.CHAT): 1,
                                 z.source_tag(Namespace.UI): 1,
                                 z.source_tag(Namespace.WEB): 1,
                                 z.source_tag(Namespace.WORLD): 2}


def test_area_coordinates_never_reach_the_observation() -> None:
    """Координаты — вход, а не выход: агент не получает разметки (инвариант 4)."""
    obs = _boundary().read(_frame(), _areas(), at_frame=1)
    fields = set()
    for o in obs:
        fields |= set(o.as_dict())
    assert not (fields & {"top", "left", "height", "width"})


def test_only_this_module_sees_raw_strings() -> None:
    """Статически: `reader.read` вызывается ровно в одном месте кода."""
    import ast

    root = Path(__file__).resolve().parent.parent / "src"
    callers: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "read"
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "reader"):
                callers.append(f"{path.name}:{node.lineno}")
    assert callers == ["text.py:" + str(next(
        n.lineno for n in ast.walk(ast.parse(
            (root / "harness/perception/text.py").read_text(encoding="utf-8")))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "read" and isinstance(n.func.value, ast.Attribute)
        and n.func.value.attr == "reader"))], (
        f"чтение сырых строк вызывается не только на границе: {callers}")


def test_symbol_width_is_a_profile_setting_not_a_constant() -> None:
    """Ширина символа меняет опыт, значит живёт в схеме (инварианты 23 и 11)."""
    from harness.core.profile import from_schema

    profile = from_schema("узкий", capture_width=64, capture_height=64,
                          symbol_digits=4)
    narrow = Symbolizer.from_profile(profile)
    assert narrow.digits == 4
    assert len(narrow.symbolize("Здоровье")) == len("SYM_") + 4

    wide = Symbolizer.from_profile(from_schema("широкий", capture_width=64,
                                               capture_height=64))
    assert wide.digits == 8, "по умолчанию 8: на 4 замер дал 31 коллизию из 2058"
    assert is_symbol(wide.symbolize("Здоровье"))
    # Ширина структурная: смена меняет разом все символы, поэтому она обязана
    # попадать в хеш профиля, иначе два несравнимых прогона выглядят одинаковыми.
    assert profile.profile_hash != from_schema(
        "узкий", capture_width=64, capture_height=64).profile_hash


def test_expected_collisions_says_what_to_expect() -> None:
    """Ноль коллизий на маленьком корпусе — свойство корпуса, а не ширины."""
    narrow, wide = Symbolizer("л", digits=4), Symbolizer("л", digits=8)
    assert narrow.expected_collisions(100) < 0.1, (
        "на сотне надписей и четырёх цифрах коллизий ожидаемо нет — значит их "
        "отсутствие ничего не доказывает")
    assert 25 < narrow.expected_collisions(2058) < 40, (
        "формула должна давать примерно те 31, что дал замер")
    assert wide.expected_collisions(2058) < 0.01


def test_collision_reaches_the_debug_table(tmp_path: Path) -> None:
    """Две надписи, слипшиеся в один символ, обязаны дойти до исследователя.

    Первая версия помнила заземление множеством символов и вторую надпись с тем же
    символом считала уже заземлённой — то есть отфильтровывала ровно тот случай,
    для обнаружения которого таблица и нужна.
    """
    from harness.debug.channel import DebugChannel, DebugChannelError

    # Ширина 4 берётся нарочно: на ней коллизию можно построить, и это единственный
    # способ проверить, что механизм доклада работает.
    z = Symbolizer("линия-коллизий", digits=4)
    a = "Здоровье"
    target = z.symbolize(a)
    # Суффикс — счётчик, а не дописывание к той же строке: растущая строка делает
    # нормализацию квадратичной, и поиск коллизии, который должен занимать секунду,
    # не заканчивается вовсе.
    b = next(cand for i in range(1 << 20)
             if z.symbolize(cand := f"Голод {i}") == target)
    # Оба чтения — из одного источника: пространство имён подмешано в ключ, и в
    # разных источниках та же пара символов уже не совпадёт.
    same_source = [TextArea(0, 0, 20, 120, Namespace.WORLD),
                   TextArea(30, 0, 20, 120, Namespace.WORLD)]
    ch = DebugChannel(tmp_path / "debug", mode="a")
    boundary = TextBoundary(z, ScriptedReader([a, b]), debug=ch)
    obs = boundary.read(_frame(), same_source, at_frame=1)
    ch.close()

    assert obs[0].symbol == obs[1].symbol
    assert boundary.stats.collisions == 1, "коллизия обязана быть посчитана"
    with pytest.raises(DebugChannelError, match="коллизии символов"):
        DebugChannel(tmp_path / "debug", mode="r").symbol_table()


def test_stats_for_the_percept_carry_no_apparatus_name() -> None:
    """Имя читателя законно в отчёте и незаконно в перцепте."""
    from harness.core.symbols import assert_no_plain_text

    b = _boundary()
    b.read(_frame(), _areas(), at_frame=1)
    assert_no_plain_text(b.stats.as_percept(), path="perception.text")
    assert b.stats.as_dict()["reader"] == "scripted", (
        "в отчёте исследователя чем читали видно, иначе непонятно, что за числа")
