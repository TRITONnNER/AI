"""Панель исследователя: проверки готовности, которые видны из кода. TASK-13.

Что здесь проверяется и чего здесь **не** проверяется.

Здесь — то, что читается статически: сколько величин объявлено на первом экране, у каждой
ли есть происхождение, ведёт ли подпись к настоящему идентификатору, в одном ли месте
задан источник данных, нет ли в оформлении градиентов и теней.

Не здесь — то, что видно только в отрисованном документе: сколько величин **видно** в
покое, какую долю площади занимает главный объект, виден ли СТОП на каждом экране,
переключаются ли экраны с клавиатуры. Это проверяет `panel/check.mjs` в настоящем
браузере, и подменять браузерную проверку разбором разметки нельзя: разметка может быть
верной, а отрисовка — нет, и ровно так «шесть величин» превращаются в стену.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "panel" / "index.html"
FIXTURE = ROOT / "panel" / "fixture.json"

#: Сколько величин допускается на первом экране в покое. Требование задачи, и число
#: здесь, а не в разметке: разметка меняется, требование нет.
LIVE_LIMIT = 6


@pytest.fixture(scope="module")
def html() -> str:
    return PANEL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# --- первый экран: шесть величин и ни одной сверх --------------------------------


def test_live_screen_declares_six_values(fixture: dict) -> None:
    """Ровно шесть величин объявлено, и все шесть — те, что перечислены в задаче."""
    values = fixture["live"]["values"]
    assert len(values) == LIVE_LIMIT
    labels = [v["label"] for v in values]
    assert labels == ["точность предсказания", "бюджет активной цели", "настроение",
                      "ведущий драйв", "конфабуляция", "задержка планировщика"]


def test_rail_is_rendered_only_from_the_fixture(html: str) -> None:
    """Величины на первом экране берутся из фикстуры, а не дописываются в разметке.

    Проверяется числом: шаблон величины в файле ровно один. Второй шаблон означал бы
    величину, которой в фикстуре нет, — и предел в шесть штук перестал бы что-либо
    ограничивать, потому что считать стало бы нечего.
    """
    assert html.count('className = "value"') == 1
    assert 'class="value"' not in html, "величины не пишутся в разметке руками"


def test_every_value_carries_its_origin(fixture: dict) -> None:
    """У каждой величины есть источник, и отсутствие числа объяснено словами."""
    for v in fixture["live"]["values"] + fixture["state"]["body"]:
        assert v["source"], f"{v['label']}: нет источника"
        assert v["origin"], f"{v['label']}: нет происхождения"
        if v["value"] is None:
            assert v["note"], (f"{v['label']}: пустое место без причины. Непонятно, не "
                               "измерили или измерили и вышло нуль")


def test_absent_values_are_not_zeros(fixture: dict) -> None:
    """Величины без замера приходят `null`, а не нулём.

    Ноль здесь читался бы как «измерили и получилось нуль» — и панель врала бы ровно там,
    где должна показывать незнание. В этом прогоне таких величин две, и обе названы.
    """
    absent = [v for v in fixture["live"]["values"] if v["value"] is None]
    assert [v["label"] for v in absent] == ["конфабуляция", "задержка планировщика"]
    assert all(v["origin"] == "нет источника" for v in absent)


# --- подписи не выдуманы --------------------------------------------------------


def _resolves(source: str) -> bool:
    """Ведёт ли источник к настоящему месту в проекте.

    Три законных вида: путь модуля (`model.vitals.prediction_error_mean`), путь файла
    (`docs/measurements/`) и ссылка на документ (`INTERFACE.md, каналы общения`).
    """
    head = source.split(",")[0].strip()
    if head.endswith("/") or head.endswith(".md"):
        return (ROOT / head).exists()
    parts = head.split(".")
    for cut in range(len(parts), 0, -1):
        mod = ROOT / "src" / "harness" / Path(*parts[:cut])
        if mod.with_suffix(".py").exists() or (mod / "__init__.py").exists():
            return True
    return False


def test_no_invented_labels(fixture: dict) -> None:
    """Каждая подпись ведёт к настоящему идентификатору проекта.

    «Не выдуманное слово» проверяется механически: у подписи есть источник, и источник
    существует. Красивое слово без источника — ровно то, чем панель обманывает: читатель
    решает, что величина есть, потому что у неё есть название.
    """
    sources: list[tuple[str, str]] = []
    for v in fixture["live"]["values"] + fixture["state"]["body"]:
        sources.append((v["label"], v["source"]))
    for key in ("drives_source",):
        sources.append((key, fixture["state"][key]))
    sources.append(("стек целей", fixture["goals"]["stack_source"]))
    sources.append(("вопросы", fixture["goals"]["questions"]["source"]))
    sources.append(("речь", fixture["goals"]["speech"]["source"]))
    sources.append(("карточки", fixture["memory"]["cards_source"]))
    sources.append(("конвейер", fixture["memory"]["pipeline_source"]))
    sources.append(("слои", fixture["journal"]["by_layer_source"]))
    sources.append(("тело", fixture["body"]["source"]))
    sources.append(("опыты", fixture["experiments"]["source"]))
    sources.append(("профиль", fixture["profile"]["source"]))
    for o in fixture["live"]["overlays"]:
        sources.append((o["label"], o["source"]))

    bad = [(label, src) for label, src in sources if not _resolves(src)]
    assert not bad, "подписи без настоящего источника: " + "; ".join(
        f"{label} → {src}" for label, src in bad)


def test_overlay_labels_are_named_mechanisms(fixture: dict) -> None:
    """Накладки названы механизмами проекта и по умолчанию выключены."""
    overlays = fixture["live"]["overlays"]
    assert len(overlays) == 5
    assert all(o["on"] is False for o in overlays), "по умолчанию всё выключено"
    assert len({o["key"] for o in overlays}) == 5, "по клавише на накладку"


# --- источник данных в одном месте ----------------------------------------------


def test_source_is_declared_once(html: str) -> None:
    """Подмена фикстуры на журнал — правка в одном месте."""
    decls = re.findall(r'const SOURCE = "(.+?)"', html)
    assert decls == ["fixture.json"]
    # Объявление ровно одно, а обращений к нему сколько угодно: правка в одном месте
    # означает единственность **объявления**, а не единственность упоминания.
    assert html.count("const SOURCE") == 1
    assert len(re.findall(r"fetch\(", html)) == 1, "второй запрос — второй путь к данным"
    assert "fetch(SOURCE)" in html


def test_fixture_says_how_to_swap(fixture: dict) -> None:
    about = fixture["about"]
    assert "make_panel_fixture" in about["how_to_swap"]
    assert set(about["origins"]) == {"живая запись оператора", "синтетический прогон",
                                     "нет источника"}


# --- визуальные правила ---------------------------------------------------------


def test_no_gradients_glow_or_shadows(html: str) -> None:
    """Ни градиентов, ни теней, ни свечения: за панелью сидят часами."""
    for bad in ("linear-gradient", "radial-gradient", "box-shadow", "text-shadow",
                "blur("):
        assert bad not in html, f"в оформлении есть {bad}"


def test_background_is_dark_but_not_black(html: str) -> None:
    """Фон тёмно-нейтральный, не чёрный."""
    m = re.search(r"--bg:#([0-9a-fA-F]{6})", html)
    assert m, "фон не объявлен переменной"
    r, g, b = (int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4))
    assert 0 < max(r, g, b) < 60, f"фон #{m.group(1)} слишком тёмный или слишком светлый"
    assert max(r, g, b) - min(r, g, b) <= 12, "фон должен быть нейтральным"


def test_numbers_are_monospaced_with_tabular_figures(html: str) -> None:
    """Цифры меняются каждый кадр и не должны прыгать по ширине."""
    assert "font-variant-numeric:tabular-nums" in html
    assert html.count("font-variant-numeric:tabular-nums") >= 4


def test_exactly_one_saturated_colour(html: str) -> None:
    """Насыщенный цвет один и означает только тревогу."""
    assert "--alarm:#c0392b" in html
    body = html.split("</style>")[0]
    # Тревожный цвет применяется к индикатору, к рамке величины за границей и к СТОПу —
    # и больше нигде: иначе «светится всё» и не выделено ничего.
    assert body.count("var(--alarm)") <= 6


# --- семь экранов, у каждого свой вопрос ----------------------------------------


def test_seven_screens_each_with_a_question(html: str) -> None:
    """Один экран — один вопрос, и вопрос записан рядом с экраном."""
    screens = re.findall(r'\["(s-[a-z]+)",\s*"([^"]+)",\s*"([^"]+)"\]', html)
    assert len(screens) == 7
    ids = [s[0] for s in screens]
    assert ids[0] == "s-live", "первый экран — «Живое», он же по умолчанию"
    assert len(set(ids)) == 7
    for _id, name, question in screens:
        assert question and not question.endswith("."), f"{name}: вопрос не записан"
    assert 'class="screen sel" id="s-live"' in html, "по умолчанию открыт первый"


def test_stop_lives_outside_every_screen(html: str) -> None:
    """СТОП стоит в полосе состояния, а не внутри экрана: спрятать его нельзя."""
    bar = html[html.index('<div id="bar">'):html.index('<div id="tabs">')]
    assert 'id="stop"' in bar
    body = html[html.index("<main>"):html.index("</main>")]
    assert 'id="stop"' not in body


def test_storage_is_hours_not_percent(fixture: dict, html: str) -> None:
    """«Осталось N часов записи», а не проценты: проценты ни о чём не говорят."""
    hours = fixture["bar"]["hours_left"]
    assert hours["unit"] == "ч"
    assert hours["value"] is None or hours["value"] > 0
    assert "часов записи" in html
    assert "%" not in html.split("</style>")[1].split("hours_basis")[0][-400:]


# --- экран «Опыты» на настоящих данных проекта -----------------------------------


def test_experiments_come_from_real_measurements(fixture: dict) -> None:
    """Экран 7 показывает замеры из `docs/measurements/`, а не пересчитывает их."""
    e = fixture["experiments"]
    assert e["source"] == "docs/measurements/"
    on_disk = {p.stem for p in (ROOT / "docs" / "measurements").glob("*.json")}
    assert set(e["files"]) == on_disk, "панель читает все файлы замеров, а не часть"
    assert len(e["runs"]) >= 8


def test_every_experiment_row_has_unit_and_n(fixture: dict) -> None:
    """Инвариант 22 на панели: у каждой строки объявлена единица и `n`.

    И `n` не бывает нулём при непустом значении: нуль в этой колонке утверждает, что
    наблюдений не было. Первая редакция сборщика искала в файле замера ключи, которых
    там нет, и печатала `n = 0` рядом с настоящим числом столкновений.
    """
    from harness.model.units import UNITS

    for r in fixture["experiments"]["runs"]:
        assert r["unit"], f"{r['name']}: единица независимости не объявлена"
        assert r["n"] > 0, f"{r['name']}: n = {r['n']} при значении «{r['value']}»"
        assert r["was"], f"{r['name']}: не сказано, как было"
        base = r["unit"].split()[-1]
        assert base in UNITS or r["unit"] in UNITS, (
            f"{r['name']}: единица «{r['unit']}» не объявлена в model.units.UNITS")


def test_profile_blocks_are_separated(fixture: dict) -> None:
    """Блок А и блок Б разделены, и касание Б предупреждает про форк журнала."""
    rows = fixture["profile"]["rows"]
    blocks = {r["block"] for r in rows}
    assert blocks == {"А", "Б"}
    assert "форк" in fixture["profile"]["warning"] or \
           "форкается" in fixture["profile"]["warning"]
    from harness.core.settings import SCHEMA

    assert len(rows) == len(SCHEMA), "конфигуратор показывает всю схему"
    structural = sum(1 for r in rows if r["block"] == "Б")
    assert structural == sum(1 for s in SCHEMA if s.structural)
