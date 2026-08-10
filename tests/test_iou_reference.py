"""Опорное число синтетики: одно, с единицей усреднения. `TASK-11`, часть 1.

В проекте жили **два** числа, и оба назывались «IoU на синтетике»: 0.97 и 0.667. Ожидание
по живому — «около 0.6» с границами 0.4 и 0.85 — было откалибровано против 0.97. Против
0.667 то же ожидание означает «ничего не изменится»; хуже того, нижняя граница 0.4 попадает
**внутрь** разброса синтетики: два домена из пяти дают меньше.

Здесь проверяется четыре разных вещи, и смешивать их нельзя:

1. опорное число одно, и его единица усреднения объявлена (инвариант 22);
2. выбор статистики обоснован **числом**, а не вкусом;
3. обе границы ожидания выведены из опорного числа, а не нарисованы рукой;
4. «IoU на синтетике» **нельзя опубликовать** без единицы усреднения — и это проверка с
   предъявленной долей ложных срабатываний и покрытием (инвариант 31).
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import numpy as np
import pytest

from harness.corpus.live import (DOMAIN_FOR_KIND, KINDS, LiveScore, OUTCOMES,
                                 SYNTHETIC_IOU, SYNTHETIC_STATISTIC, SYNTHETIC_UNIT,
                                 compare_to_synthetic, method_comparable, refute,
                                 synthetic_reference, synthetic_text, trivial_iou)

ROOT = Path(__file__).resolve().parent.parent


def _score(kind: str, iou: float, trivial: float) -> LiveScore:
    s = LiveScore(path=Path("/x") / kind, kind=kind, frames=10)
    s.iou, s.trivial = iou, trivial
    s.matched_domain = DOMAIN_FOR_KIND.get(kind, "")
    s.matched_iou = SYNTHETIC_IOU.get(s.matched_domain)
    return s


# --- 1. одно опорное число, и единица при нём --------------------------------


def test_the_reference_is_one_number_with_its_unit() -> None:
    """Опорное число одно и приходит вместе с единицей усреднения, `n` и разбросом.

    Функция, отдающая одинокий `float`, — это и есть тот способ, которым «IoU на синтетике»
    разошлось в две разные величины: получатель числа не знал, что он получил.
    """
    r = synthetic_reference()
    assert r["value"] == pytest.approx(0.667, abs=0.001)
    assert r["unit"] == "домен" == SYNTHETIC_UNIT
    assert r["statistic"] == "медиана" == SYNTHETIC_STATISTIC
    assert r["n"] == len(SYNTHETIC_IOU) == 5
    # Разброс обязателен: 0.667 при разбросе 0.196–0.815 и 0.667 при разбросе 0.66–0.67 —
    # разные утверждения, а выглядят одинаково.
    assert r["min"] == pytest.approx(0.196) and r["max"] == pytest.approx(0.815)
    assert r["sd"] > 0.2


def test_the_unit_is_the_domain_because_the_spread_says_so() -> None:
    """Единица — домен, и это установлено замером, а не выбрано.

    Разброс между доменами вчетверо больше разброса внутри домена: смена сида наблюдения не
    добавляет, смена домена добавляет всё. Инвариант 22: `n` считает единицы, а не события,
    а событий было 30.
    """
    between = statistics.stdev(SYNTHETIC_IOU.values())
    # Разбросы внутри доменов из `MEASUREMENT.md` §17: 0.019, 0.024, 0.004, 0.000, 0.062.
    within_worst = 0.062
    assert between > 4 * within_worst, (
        f"между доменами {between:.3f}, внутри домена не больше {within_worst:.3f} — "
        "единица «домен» обоснована именно этим соотношением")
    assert synthetic_reference()["n"] == 5, "n считает домены, а не 30 прогонов"


def test_the_median_is_chosen_by_a_number_not_by_taste() -> None:
    """Медиана, а не среднее, — потому что на пяти единицах среднее втрое чувствительнее.

    `depth` (0.196) — домен, где допущение метода «у мира одно движение» ломается нарочно.
    Выбросить его из набора значило бы улучшить число, ничего не улучшив в методе, поэтому
    устойчивость статистики к одному домену — не эстетика, а требование.
    """
    xs = sorted(SYNTHETIC_IOU.values())
    without = sorted(v for k, v in SYNTHETIC_IOU.items() if k != "depth")
    move_median = abs(statistics.median(without) - statistics.median(xs))
    move_mean = abs(statistics.mean(without) - statistics.mean(xs))
    assert move_median < move_mean / 2, (
        f"медиана сдвигается на {move_median:.3f}, среднее на {move_mean:.3f}")
    assert synthetic_reference()["statistic"] == "медиана"


def test_the_old_0_97_is_marked_as_not_a_reference() -> None:
    """0.97 объявлено **не** опорным: у него одна независимая единица.

    Оно остаётся в проекте как число второго пути (одиночный разделитель), и это законно —
    незаконно калибровать против него ожидание.
    """
    # Проверяется утверждение, а не его написание: первая редакция теста искала подстроку
    # «0.97» и упала, когда то же число стали печатать как 0.9696 — тест ловил формат.
    from harness.corpus.live import PARALLAX_REFERENCE

    said = method_comparable("parallax")
    assert PARALLAX_REFERENCE["is_reference"] is False
    assert PARALLAX_REFERENCE["n"] == 1 and PARALLAX_REFERENCE["unit"] == "генератор"
    assert f"{PARALLAX_REFERENCE['value']:.4f}" in said
    assert f"n={PARALLAX_REFERENCE['n']}" in said
    assert "не** опорное" in said or "не опорное" in said
    assert synthetic_text() in method_comparable("arbiter")


# --- 2. границы выведены, а не нарисованы ------------------------------------


def test_the_floor_is_the_trivial_answer_not_a_hand_drawn_04() -> None:
    """Пол — тривиальный ответ «весь кадр — экранный слой», посчитанный по разметке.

    Прежние 0.4 попадали внутрь разброса синтетики: `video` даёт 0.417, `depth` — 0.196.
    То есть «метод не работает» объявлялось бы на значении, которое сама синтетика выдаёт
    как норму.
    """
    # 0.4 проходит **между** двумя доменами: `depth` даёт 0.196, `video` — 0.417. То есть
    # граница «метод не работает» стояла внутри разброса синтетики, и один домен из пяти
    # уже её не проходит. (Первая редакция этой проверки утверждала «два домена из пяти» —
    # арифметическая ошибка, и поймала её сама проверка.)
    assert SYNTHETIC_IOU["depth"] < 0.4 < SYNTHETIC_IOU["video"], (
        "проверка потеряла смысл: 0.4 больше не внутри разброса синтетики")

    truth = np.zeros((100, 100), dtype=bool)
    truth[:30] = True
    assert trivial_iou(truth) == pytest.approx(0.30), (
        "тривиальный ответ «весь кадр — экран» даёт |истина| / |кадр|")

    # Пол зависит от записи, а не от проекта: у разметки в треть кадра и в две трети он
    # разный, и одно число на все записи было бы неверно для обеих.
    truth2 = np.zeros((100, 100), dtype=bool)
    truth2[:66] = True
    assert trivial_iou(truth2) > trivial_iou(truth)


def test_the_ceiling_is_the_best_synthetic_domain_not_085() -> None:
    """Потолок — лучший синтетический домен (0.815), а не 0.85.

    0.85 стояло **выше** любого синтетического домена и потому не отличало «подозрительно
    хорошо» от «как лучшая синтетика». Живое обязано быть хуже по четырём названным
    причинам, значит превышение лучшей синтетики и есть повод искать загрязнение.
    """
    ceiling = max(SYNTHETIC_IOU.values())
    assert ceiling == pytest.approx(0.815)
    assert 0.85 > ceiling, "проверка потеряла смысл: 0.85 уже не выше синтетики"

    good = refute(_score("stillness", iou=0.80, trivial=0.30))
    assert good["outcome"] == "не опровергнуто"
    too_good = refute(_score("stillness", iou=0.83, trivial=0.30))
    assert too_good["outcome"] == "опровергнуто: подозрительно хорошо"
    assert "загрязнени" in too_good["why"]


def test_the_margin_over_the_floor_lives_in_the_schema() -> None:
    """Запас над полом — настройка схемы, а не число в прозе.

    Инвариант 23: число, нарисованное рукой в аналитическом тексте, не существует нигде в
    записи системы и не сравнимо между прогонами. Прежние 0.4 и 0.85 были именно такими.
    """
    from harness.core.settings import SCHEMA, RunPath, applies_of

    setting = [s for s in SCHEMA if s.key == "live_iou_margin_over_trivial"]
    assert setting, "запас не объявлен в схеме"
    assert applies_of(setting[0]) == (RunPath.REPORT,)
    assert setting[0].unit == "доля"

    from harness.core.profile import MILESTONE_0

    margin = float(MILESTONE_0.parameters["live_iou_margin_over_trivial"])
    just_above = _score("stillness", iou=0.30 + margin, trivial=0.30)
    just_below = _score("stillness", iou=0.30 + margin / 2, trivial=0.30)
    assert just_above.beats_trivial is True
    assert just_below.beats_trivial is False


def test_the_expectation_is_refutable_in_both_directions() -> None:
    """Три исхода, и каждый достижим. Ожидание, которое нельзя опровергнуть, — не ожидание.

    Прежнее «около 0.6» против опорного 0.667 означало «ничего не изменится»: разница 0.067
    при разбросе синтетики по доменам 0.196–0.815 — четверть одного sd.
    """
    got = {refute(s)["outcome"] for s in (
        _score("stillness", iou=0.31, trivial=0.30),      # пол не перебит
        _score("stillness", iou=0.60, trivial=0.30),      # внутри
        _score("stillness", iou=0.90, trivial=0.30),      # выше лучшей синтетики
    )}
    assert got == {"опровергнуто: метод не работает", "не опровергнуто",
                   "опровергнуто: подозрительно хорошо"}
    assert set(OUTCOMES) >= got
    # Четвёртый исход — «сравнивать не с чем», и он не про качество: без разметки истины
    # нет вовсе.
    blank = LiveScore(path=Path("/x/s"), kind="stillness", frames=3)
    blank.absent_reason = "разметки нет"
    assert refute(blank)["outcome"] == "сравнивать не с чем"


def test_the_comparison_is_domain_matched() -> None:
    """Живая запись сравнивается с похожим доменом, а не с медианой по всем пяти.

    Запись браузера против `depth` — мира из трёх планов — это не сравнение, а разница
    постановок.
    """
    for kind in KINDS:
        assert kind in DOMAIN_FOR_KIND, f"для вида {kind} не назван домен"
        assert DOMAIN_FOR_KIND[kind] in SYNTHETIC_IOU

    got = compare_to_synthetic([_score("scroll", iou=0.55, trivial=0.20),
                                _score("video", iou=0.30, trivial=0.20)])
    kinds = {m["kind"]: m for m in got["matched"]}
    assert kinds["scroll"]["domain"] == "document"
    assert kinds["video"]["domain"] == "video"
    # Падение считается против **своего** домена: 0.30 против 0.417, а не против 0.667.
    assert kinds["video"]["drop"] == pytest.approx(0.30 - 0.417, abs=0.001)


def test_the_verdict_band_comes_from_the_synthetic_spread() -> None:
    """«В пределах разброса» — полуразброс синтетики по доменам, а не 0.02 из головы."""
    band = (max(SYNTHETIC_IOU.values()) - min(SYNTHETIC_IOU.values())) / 2
    got = compare_to_synthetic([_score("scroll", iou=0.60, trivial=0.20)])
    assert f"{band:.3f}" in got["verdict"], got["verdict"]
    assert "опорного" in got["verdict"] and SYNTHETIC_UNIT in got["verdict"]


def test_comparison_carries_the_reference_text_everywhere() -> None:
    """Сводка несёт опорное число **с единицей**, в том числе когда сравнивать нечего."""
    empty = compare_to_synthetic([])
    assert empty["reference_text"] == synthetic_text()
    assert "единица усреднения" in empty["reference_text"]
    full = compare_to_synthetic([_score("scroll", iou=0.5, trivial=0.2)])
    assert full["reference_text"] == synthetic_text()
    assert full["reference"]["unit"] == "домен"


# --- 3. публиковать без единицы усреднения нельзя ----------------------------

#: Где ищется опубликованное число. Документы и код, которые читает человек: панель,
#: командная строка, проза замеров.
PUBLISHED: tuple[str, ...] = (
    "MEASUREMENT.md", "ARCHITECTURE.md", "ROADMAP.md", "PANEL.md", "SETUP.md",
    "CLAUDE.md", "MILESTONE-0-HARNESS.md", "docs/STATUS.md",
    "src/harness/corpus/live.py", "src/harness/cli.py", "src/harness/benchmark.py",
    "panel/index.html",
)

#: Слова, любое из которых рядом с числом означает, что единица усреднения названа.
#: Список короткий нарочно: «объявлено где-то выше» не считается — читатель видит строку,
#: а не абзац.
UNIT_WORDS: tuple[str, ...] = (
    "домен", "единица", "медиана", "сессия", "генератор", "прогон", "n=", "n =",
    # «сид 1: IoU 0.9703» — это одно наблюдение, а не среднее; назвав сид, строка назвала
    # и то, чем это число является. Первая редакция проверки объявила дефектной таблицу
    # сырых наблюдений — четыре строки из одиннадцати.
    "сид",
)

#: Строки **о самом запрете**. Они содержат и слово IoU, и число, и не публикуют его, а
#: объясняют, почему так публиковать нельзя. Первая редакция проверки нашла собственный
#: докстринг и объявила его нарушением — ровно как проверка адреса нашла в TASK-19 своё же
#: объяснение про `0.0.0.0`.
ABOUT_THE_RULE: tuple[str, ...] = (
    "без единицы усреднения", "запрещено", "не опорное",
)

#: Строки, где число 0.667 или 0.97 стоит **без** единицы усреднения законно, и почему.
#: Список с причинами: без причин он превращается в место, куда сваливают неудобное.
ALLOWED: dict[str, str] = {
    "0.667 (медиана)": "таблица §17: столбец рядом называет и единицу, и n",
}

#: Слова, после которых число в строке — **не IoU**, а что-то другое про него: доля от
#: чистого значения, множитель, разброс доли. У такого числа единицы усреднения IoU быть не
#: может, потому что это не IoU.
#:
#: Класс найден при перекалибровке ожидания (TASK-11, часть 1): строка «курсор ... даёт долю
#: 1.000 (0.981…1.063), то есть на IoU не влияет измеримо» была объявлена нарушением. Она
#: не публикует IoU вовсе — она сообщает, что IoU не изменился. Без этого списка проверка
#: требовала бы приписывать единицу усреднения к множителю.
NOT_AN_IOU: tuple[str, ...] = (
    "долю", "доля", "доли", "раза", "множител",
    # «Курсор занимает 0.09 % кадра ... и на IoU не влияет» — это площадь, а не IoU.
    # Найдено при TASK-21: правило поймало строку решения о курсоре, где число означает
    # долю площади кадра. Формулировка узкая нарочно — «% кадра», а не «пиксел»: второе
    # закрыло бы и настоящие публикации вида «IoU 0.667 на N пикселях разметки».
    "% кадра",
)


def _published_lines() -> list[tuple[str, int, str]]:
    out = []
    for name in PUBLISHED:
        path = ROOT / name
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            out.append((name, i, line))
    return out


def test_iou_is_never_published_without_its_unit() -> None:
    """«IoU на синтетике» не публикуется без единицы усреднения.

    Запрет механический, а не устный: именно из-за отсутствия единицы одно и то же число
    означало два разных утверждения, и калибровка ожидания ушла не туда.
    """
    bad = []
    for name, i, line in _published_lines():
        if not re.search(r"\bIoU\b", line):
            continue
        # Число в строке есть? Ищем доли вида 0.xxx — только они бывают IoU.
        if not re.search(r"0\.\d{2,4}", line):
            continue
        if any(k in line for k in ALLOWED):
            continue
        if any(w in line for w in ABOUT_THE_RULE):
            continue
        if any(w in line for w in NOT_AN_IOU):
            continue
        if any(w in line for w in UNIT_WORDS):
            continue
        bad.append(f"{name}:{i}: {line.strip()[:90]}")
    assert not bad, ("IoU опубликован без единицы усреднения:\n  " + "\n  ".join(bad))


def test_this_check_states_its_false_positive_share_and_coverage() -> None:
    """У проверки предъявлены доля ложных срабатываний и покрытие (инвариант 31).

    **Доля ложных, измеренная.** Первая редакция нашла 11 строк. Разобранные по одной:

    | Что нашла | Строк | Ложное? |
    |---|---|---|
    | таблица сырых наблюдений «сид 1: IoU 0.9703» | 4 | да: сид назван, среднего нет |
    | собственный докстринг про запрет | 1 | да: строка о запрете, а не публикация |
    | проза, критикующая 0.97, с переносом строки | 1 | да: единица названа строкой ниже |
    | **старая граница «IoU выше 0.85»** | 1 | нет |
    | **«IoU 0.97 на синтетике» в SETUP.md и STATUS.md** | 2 | нет |
    | **«IoU 0.73–0.77» в benchmark.py** | 1 | нет |
    | **опорное число в `--plan`** | 1 | нет |

    Итого 6 ложных из 11 — **55 %**, и все шесть от одной причины: правило смотрит строку, а
    прозе свойственно переноситься. Сузилось двумя списками (`UNIT_WORDS` плюс `сид`,
    `ABOUT_THE_RULE`), после чего ложных 0, а пять настоящих находок исправлены.

    **Вторая редакция, вторая доля.** После перекалибровки ожидания проверка нашла ещё
    3 строки, из которых 1 ложная — **33 %**: строка «курсор ... даёт долю 1.000 ... на IoU
    не влияет измеримо» публикует не IoU, а множитель, и единицы усреднения IoU у него быть
    не может. Сузилось списком `NOT_AN_IOU`; две настоящие находки исправлены (граница §18 и
    строка `docs/STATUS.md`). Доля ложных считается **при каждой правке правила**, а не один
    раз: 0 % однажды не остаётся нулём после следующего абзаца.

    **Третья редакция, и снова того же класса.** TASK-21 добавил в `ARCHITECTURE.md` решение
    о курсоре: «занимает 0.09 % кадра ... и на IoU не влияет измеримо». Проверка нашла
    1 строку, ложную — **100 % этой находки**: число означает долю площади кадра. Сузилось
    добавлением «% кадра». Накопительно по трём редакциям: 8 ложных из 15 находок (53 %), и
    все восемь — одной причины: правило смотрит строку, а рядом с IoU в тексте живут числа,
    которые IoU не являются.

    **Покрытие.** Проверка смотрит перечисленные файлы, которые читает человек, и **не**
    видит: числа, собранные из переменных в момент печати (там единица приходит из
    `synthetic_text`), картинки и чужие заметки. Второе прикрыто иначе: единственный
    источник числа отдаёт его вместе с единицей, и одинокого `float` из него не достать.
    """
    lines = _published_lines()
    assert len(lines) > 2000, f"осмотрено строк: {len(lines)}"
    files = {n for n, _, _ in lines}
    assert len(files) >= 8, f"осмотрено файлов: {len(files)}"

    # Образец, который **обязан** сработать: число без единицы.
    sample_bad = "На синтетике IoU 0.667, на живом посмотрим."
    assert re.search(r"\bIoU\b", sample_bad) and re.search(r"0\.\d{2,4}", sample_bad)
    assert not any(w in sample_bad for w in UNIT_WORDS), (
        "образец с числом без единицы проходит проверку — она не ловит своего случая")

    # Образец, который **не должен** сработать: единица названа в той же строке.
    sample_ok = "IoU 0.667 (медиана по 5 доменам, единица усреднения — домен)"
    assert any(w in sample_ok for w in UNIT_WORDS)

    # И то, ради чего исключения существуют: каждое названо с причиной.
    assert all(ALLOWED.values())

    # Образец из второй редакции: множитель, а не IoU. Проверка обязана его пропустить, и
    # обязана пропускать **по этой причине**, а не потому, что в строке случайно есть
    # «домен».
    sample_ratio = "курсор даёт долю 1.000 (0.981…1.063), то есть на IoU не влияет"
    assert any(w in sample_ratio for w in NOT_AN_IOU)
    assert not any(w in sample_ratio for w in UNIT_WORDS), (
        "образец проходит по единице, а не по причине — исключение проверяется не тем")


# --- 5. ожидание выведено из замера, а не выбрано ----------------------------


def test_the_expectation_is_derived_from_a_measurement() -> None:
    """Ожидание = опорное число × измеренная доля, и обе части названы.

    Прежнее «около 0.6» не было выведено ниоткуда: оно стояло числом в прозе. Проверяется
    не значение, а его **происхождение**: если ожидание перестанет быть произведением
    опорного числа на измеренную долю, тест упадёт, даже если само число совпадёт.
    """
    from harness.corpus.live import (LIVE_EXPECTATION, expectation_text,
                                     synthetic_reference)

    e = LIVE_EXPECTATION
    ref = synthetic_reference()["value"]
    assert e["unit"] == "домен" and e["n"] == 5
    assert abs(e["value"] - ref * e["ratio"]) < 0.01, (
        f"ожидание {e['value']} не равно {ref} × {e['ratio']} = {ref * e['ratio']:.3f}")
    assert e["tool"].endswith("measure_live_penalty.py"), "у ожидания нет инструмента"
    # Разброс доли обязан быть при ней: медиана 0.762 при разбросе 0.461…1.666 и та же
    # медиана при разбросе 0.75…0.77 — разные утверждения, а выглядят одинаково.
    lo, hi = e["ratio_range"]
    assert lo < e["ratio"] < hi
    said = expectation_text()
    assert f"{e['value']:.2f}" in said and f"{ref:.3f}" in said
    assert "домен" in said and "n=5" in said


def test_the_measurement_refuted_the_direction_it_was_run_to_confirm() -> None:
    """«Живое обязано быть хуже» — опровергнуто, и это записано, а не сглажено.

    Замер порчей ставился, чтобы получить долю падения. На двух доменах из пяти те же четыре
    порчи IoU **подняли**. Обоснование потолка обязано это учитывать: «обязано быть хуже» и
    «повод искать загрязнение» — разные утверждения, и первое замером опровергнуто.
    """
    from harness.corpus.live import LIVE_EXPECTATION, refute

    assert LIVE_EXPECTATION["improved_domains"] == 2
    assert LIVE_EXPECTATION["ratio_range"][1] > 1.0, (
        "разброс доли не содержит роста — тогда опровержения не было")

    score = _scored(kind="scroll", iou=0.90, trivial=0.10)
    said = refute(score)["why"]
    # Ищется **утвердительная** форма, а не слово: снятую формулировку сообщение цитирует
    # нарочно, и запрет на слово «обязано» запретил бы говорить, что именно снято. Первая
    # редакция теста искала слово и падала на цитате.
    assert "хотя живое обязано" not in said, (
        "потолок всё ещё обоснован обязанностью, которую замер опроверг: " + said)
    assert "снята" in said, "снятая формулировка нигде не названа — читатель не узнает"
    assert "загрязнен" in said


def test_iou_can_be_vacuous_by_a_property_of_the_recording() -> None:
    """Пол выше потолка — исход «вакуумно», а не «подозрительно хорошо» (инвариант 27).

    На записи, где обрамление занимает почти весь кадр, тривиальный ответ набирает больше
    лучшего синтетического домена. IoU там не различает ничего, и объявлять по нему исход
    нельзя ни в одну сторону. Это первая запись оператора — неподвижность рабочего стола.
    """
    from harness.corpus.live import OUTCOMES, SYNTHETIC_IOU, refute

    vacuous = "вакуумно: тривиальный ответ выше потолка"
    assert vacuous in OUTCOMES

    ceiling = max(SYNTHETIC_IOU.values())
    # Обрамление на 90 % кадра: тривиальный ответ 0.90 выше потолка 0.815.
    got = refute(_scored(kind="stillness", iou=0.93, trivial=0.90))
    assert got["outcome"] == vacuous, got
    # И то, что отличает вакуумность от «подозрительно хорошо»: при том же IoU, но малой
    # доле обрамления, исход — именно подозрение.
    other = refute(_scored(kind="stillness", iou=0.93, trivial=0.10))
    assert other["outcome"] == "опровергнуто: подозрительно хорошо", other
    assert got["ceiling"] == round(ceiling, 4)


def _scored(*, kind: str, iou: float, trivial: float):
    """Готовый `LiveScore` с проставленными полями. Собирается здесь, а не в каждом тесте."""
    from pathlib import Path

    from harness.corpus.live import DOMAIN_FOR_KIND, SYNTHETIC_IOU, LiveScore

    s = LiveScore(path=Path("x"), kind=kind, frames=10, method="arbiter",
                  decided=0.5, screen_share=0.5)
    s.iou = iou
    s.trivial = trivial
    s.matched_domain = DOMAIN_FOR_KIND.get(kind, "")
    s.matched_iou = SYNTHETIC_IOU.get(s.matched_domain)
    return s


def test_the_plan_prints_the_reference_with_its_unit() -> None:
    """`--plan` печатает опорное число **с единицей усреднения**.

    Именно оттуда 0.667 читал оператор, и именно там оно стояло одиноким числом.
    """
    from harness.corpus.live import plan_text

    text = plan_text("minimal")
    assert "0.667" in text
    line = [ln for ln in text.splitlines() if "0.667" in ln]
    assert line, "опорное число исчезло из плана"
    for ln in line:
        assert any(w in ln for w in UNIT_WORDS), f"в плане число без единицы: {ln}"
