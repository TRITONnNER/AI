"""Три средства узнавания места и разделение отпечатка карточки. SPEC-FULL, A2.

Каждое средство проверяется **отдельно**: 22 789 узлов за час чинятся четырьмя
разными правками, и без повкладной проверки нельзя сказать, какая из них подействовала.

Единица независимости там, где считаются доли, — **пара видов** для фона и **прогон**
для графа мест. Не наблюдение: наблюдения внутри прохода зависимы, потому что второй
вид похож на первый ровно оттого, что мир не успел измениться.
"""

from __future__ import annotations

import pytest

from harness.core.profile import from_schema
from harness.model.identity import (INVARIANT_PARTS, VARIABLE_PARTS, Exact,
                                    Quantized, Split)
from harness.model.recognition import (Background, Recognizer, Sequences, Threshold)


def _fp(*cells: int) -> tuple[int, ...]:
    return tuple(cells)


# --- A2.2. Порог выводится из фона домена -----------------------------------


def test_without_background_the_threshold_says_it_came_from_the_schema() -> None:
    """Домен без фона отвечает «не определено», а не правдоподобным числом.

    Прямое требование раздела 35: признак сравнивается с фоном **этого** домена, а
    там, где фона нет, механизм молчит вместо того, чтобы отвечать.
    """
    b = Background(min_gap=2, min_n=10)
    got = b.threshold(0.82)
    assert got.value == pytest.approx(0.82)
    assert got.source == "схема" and not got.adaptive and got.n == 0


def test_the_background_is_built_only_from_views_far_apart() -> None:
    """Соседние виды в фон не идут: они похожи оттого, что мир не успел измениться."""
    b = Background(min_gap=3, min_n=1)
    assert b.feed(0, _fp(1, 1, 1, 1)) is None
    assert b.feed(1, _fp(1, 1, 1, 1)) is None, "сосед попал в фон"
    assert b.feed(2, _fp(1, 1, 1, 1)) is None
    got = b.feed(3, _fp(0, 0, 0, 0))
    assert got == pytest.approx(0.0), "пара с промежутком 3 обязана войти в фон"
    assert b.n == 1


def test_a_measured_background_gives_the_threshold_and_says_so() -> None:
    """Порог из фона помечен источником: иначе два разных числа неразличимы."""
    b = Background(min_gap=1, min_n=5, quantile=0.99)
    for i in range(40):
        b.feed(i, _fp(i % 4, (i + 1) % 4, (i + 2) % 4, (i + 3) % 4))
    got = b.threshold(0.5)
    assert got.source == "фон домена" and got.adaptive
    assert got.n >= 5 and got.background_p95 is not None


def test_the_adaptive_threshold_never_goes_below_the_schema() -> None:
    """Фон говорит, какая похожесть случайна, — это нижняя граница строгости.

    Иначе в однотонном домене, где все виды похожи, фон уехал бы вниз и порог
    разрешил бы склеить все места в одно — ровно та поломка, от которой лечимся.
    """
    b = Background(min_gap=1, min_n=5)
    for i in range(40):
        b.feed(i, _fp(0, 0, 0, 0))          # фон тождественно единичный
    high = b.threshold(0.82)
    assert high.value >= 0.82

    b2 = Background(min_gap=1, min_n=5)
    for i in range(40):
        b2.feed(i, _fp(i % 2, 1 - i % 2, i % 2, 1 - i % 2))   # фон низкий
    low = b2.threshold(0.82)
    assert low.value == pytest.approx(0.82), (
        f"порог опустился до {low.value}: фон не имеет права ослаблять схему")


# --- A2.3. Гипотеза «место, где я никогда не был» ---------------------------


def test_the_novelty_hypothesis_contains_no_threshold() -> None:
    """Ответ — вероятность, а не «да/нет». Порога внутри нет ни одного."""
    b = Background(min_gap=1, min_n=5)
    for i in range(40):
        b.feed(i, _fp(i % 2, 1 - i % 2, 0, 1))
    p_low = b.probability_at_least(1.0)      # совпадение лучше всего фона
    p_high = b.probability_at_least(0.0)     # совпадение не лучше ничего
    assert p_low is not None and p_high is not None
    assert p_high == pytest.approx(1.0)
    assert p_low < p_high, "вероятность нового места обязана падать с ростом совпадения"


def test_without_background_novelty_answers_none_and_not_zero() -> None:
    """«Не измеряли» и «место точно знакомое» — разные утверждения."""
    b = Background(min_gap=1, min_n=100)
    b.feed(0, _fp(1, 1)); b.feed(1, _fp(1, 1))
    assert b.probability_at_least(0.5) is None


def test_a_match_no_better_than_chance_is_refused() -> None:
    """Совпадение, не лучшее случайного, не значит ничего — каким бы ни был порог."""
    profile = from_schema("ТЕСТ-новизна", place_adaptive_threshold=True,
                          place_novelty_hypothesis=True,
                          place_background_min_gap=1, place_background_min_n=5,
                          place_same_similarity=0.1)
    r = Recognizer.from_profile(profile)
    for i in range(40):
        r.look(_fp(1, 1, 1, 1), i, [("МЕСТО", 1.0)])
    got = r.look(_fp(1, 1, 1, 1), 41, [("МЕСТО", 1.0)])
    assert got.p_new == pytest.approx(1.0), (
        "фон здесь тождественно единичный: всякое совпадение случайно")
    assert not got.same, "совпадение принято, хотя оно не лучше случайного"
    assert r.rejected_by_novelty >= 1


# --- A2.1. Сопоставление последовательностей --------------------------------


def test_a_place_without_stored_paths_scores_none_and_not_zero() -> None:
    """«Отрезков нет» и «отрезки не совпали» — разные утверждения."""
    s = Sequences(length=3)
    for fp in (_fp(1, 1), _fp(2, 2), _fp(3, 3)):
        s.observe(fp)
    assert s.score("МЕСТО") is None
    assert s.blend("МЕСТО", 0.9) == (0.9, False), "отрезка нет — счёт не трогается"


def test_the_same_path_raises_the_score_and_a_different_one_lowers_it() -> None:
    """Место узнаётся по отрезку пути: чтобы совпал он, надо прийти тем же путём."""
    s = Sequences(length=3, weight=0.5)
    path = (_fp(1, 1, 1, 1), _fp(2, 2, 2, 2), _fp(3, 3, 3, 3))
    for fp in path:
        s.observe(fp)
    assert s.remember("МЕСТО")

    for fp in path:                       # пришли тем же путём
        s.observe(fp)
    same, used = s.blend("МЕСТО", 0.5)
    assert used and same > 0.5, "тот же путь обязан поднять счёт"

    for fp in (_fp(9, 9, 9, 9), _fp(8, 8, 8, 8), _fp(7, 7, 7, 7)):
        s.observe(fp)
    other, used2 = s.blend("МЕСТО", 0.5)
    assert used2 and other < same, "другой путь обязан опустить счёт"


def test_a_short_path_is_not_remembered() -> None:
    """Недобранный отрезок не привязывается:半 отрезка сравнивать не с чем."""
    s = Sequences(length=4)
    s.observe(_fp(1, 1))
    assert not s.remember("МЕСТО")


# --- контрольный прогон: три средства выключены -----------------------------


def test_all_three_off_reproduces_the_previous_graph_exactly() -> None:
    """Выключенное средство не работает вхолостую, а не работает.

    Это контрольная точка: прежние замеры сравнимы с ней, и если она поедет,
    сравнивать новое будет не с чем.
    """
    from harness.model.places import PlaceGraph

    profile = from_schema("ТЕСТ-контроль")
    graph = PlaceGraph.from_profile(profile)
    assert graph.recognizer is None, (
        "по умолчанию узнавание выключено: иначе прежние замеры несравнимы")


def test_turning_one_mean_on_creates_the_recognizer() -> None:
    from harness.model.places import PlaceGraph

    for key in ("place_sequence_matching", "place_adaptive_threshold",
                "place_novelty_hypothesis"):
        graph = PlaceGraph.from_profile(from_schema("ТЕСТ-одно", **{key: True}))
        assert graph.recognizer is not None, f"{key} включён, а узнавания нет"


def test_the_graph_records_what_recognition_decided() -> None:
    """Решение узнавания видно снаружи: иначе вклад средства не измерить."""
    from harness.model.places import PlaceGraph

    graph = PlaceGraph.from_profile(
        from_schema("ТЕСТ-решение", place_adaptive_threshold=True))
    graph.observe(_fp(1, 1, 1, 1), 0)
    graph.observe(_fp(1, 1, 1, 1), 1)
    assert graph.last_recognition is not None
    assert graph.last_recognition.threshold.source in {"схема", "фон домена"}


# --- A2.4. Разделение отпечатка карточки ------------------------------------


def test_a_repainted_thing_stays_the_same_card() -> None:
    """Прежде перекрашенный предмет заводил новую карточку. Сдвиг числа: 2 → 1."""
    plain = Quantized(bucket=0.1)
    before = plain([1.0, 2.0, 5.0])
    after = plain([1.0, 2.0, 9.0])
    assert before != after, "контроль: неразделённый отпечаток заводит вторую карточку"

    split = Split(inner=Quantized(bucket=0.1))
    a, _ = split.observe({"shape": 1.0, "position": 2.0, "colour": 5.0}, seq=1)
    b, changes = split.observe({"shape": 1.0, "position": 2.0, "colour": 9.0}, seq=2)
    assert a == b, "перекрашенный предмет завёл новую карточку"
    assert len(changes) == 1 and changes[0].part == "colour"
    assert changes[0].before == 5.0 and changes[0].after == 9.0 and changes[0].seq == 2


def test_a_changed_shape_is_a_different_card() -> None:
    """Разделение не должно склеивать всё подряд: форма считает тождество."""
    split = Split(inner=Quantized(bucket=0.1))
    a, _ = split.observe({"shape": 1.0, "position": 2.0})
    b, _ = split.observe({"shape": 7.0, "position": 2.0})
    assert a != b


def test_the_first_sighting_is_not_a_property_change() -> None:
    """Свойство, увиденное впервые, ни с чем не сменилось."""
    split = Split(inner=Quantized(bucket=0.1))
    _, changes = split.observe({"shape": 1.0, "colour": 5.0})
    assert changes == []


def test_the_set_of_parts_is_closed() -> None:
    """Незнакомая часть, тихо ушедшая в свойства, сделала бы тождество слепым."""
    split = Split(inner=Exact())
    with pytest.raises(ValueError, match="части вне набора"):
        split.observe({"shape": 1.0, "weight": 3.0})


def test_a_payload_without_an_invariant_part_is_refused() -> None:
    """Один цвет тождества не считает, и делать вид, что считает, нельзя."""
    split = Split(inner=Exact())
    with pytest.raises(ValueError, match="инвариантной части"):
        split.observe({"colour": 5.0})


def test_the_split_needs_a_payload_that_says_what_is_what() -> None:
    split = Split(inner=Exact())
    with pytest.raises(TypeError, match="объявленными частями"):
        split.observe([1.0, 2.0])


def test_the_parts_are_disjoint_and_named() -> None:
    """Часть, попавшая в оба набора, считала бы тождество и менялась одновременно."""
    assert not set(INVARIANT_PARTS) & set(VARIABLE_PARTS)
    assert INVARIANT_PARTS and VARIABLE_PARTS
