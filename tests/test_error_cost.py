"""Цена ошибки в порядке проб. TASK-33, направление A.

Числа с единицами живут в `tools/measure_reversibility.py` и
`docs/measurements/reversibility.json`. Здесь — утверждения об устройстве, и каждое из
них соответствует одной уже совершённой ошибке:

1. Цена ошибки и обратимость — **разные величины**. Неудачная догадка об откате
   поднимает цену и не трогает обратимость: первая про риск, вторая про мир.
2. Незнание **не объявляется дорогим заранее**: фон тела начинается с нуля. Иначе
   первый же порог отправил бы в хвост всё тело, и разведка встала бы (инвариант 17).
3. Порог 1.0 обязан **снимать** гейт. Сравнение строгое, потому что цена незнания —
   ровно единица, и при нестрогом сравнении контроль «порог снят» не снимал ничего.
4. В ветви «ни разу не пробован» цена ничего решить не может: у всех кандидатов она
   одна и та же. Это объявлено, а не спрятано, — и именно поэтому закреплённый тест
   TASK-32 C проходил и до правки, и после.
5. Повтор, который **решает** (состояние выхода ещё не выведено), делается при любой
   цене. Снимается только тот, который ничего не изменит в выводе.
6. У каждой ветви выбора пробы есть счётчик, и имя ветви совпадает с началом
   `Probe.why` — того самого поля, которое уходит в журнал (инвариант 26).

Единица независимости — **утверждение об устройстве механизма**; для последнего теста
— **прогон** (мир, сид). Окружения не требует: интерактивный мир синтетический.
"""

from __future__ import annotations

from typing import Any

from harness.behaviour.babbling import (BRANCHES, THIN, UNTRIED, Babbler,
                                        run_babbling)
from harness.core.action import Reversibility, too_risky
from harness.core.clocks import Clocks
from harness.core.profile import from_schema
from harness.corpus.world import InteractiveWorld

OUTS = tuple(f"OUT_{i:02X}" for i in range(2, 12))


def _prof(**kw: Any):
    base: dict[str, Any] = dict(capture_width=64, capture_height=48)
    base.update(kw)
    return from_schema("ЦЕНА", **base)


def _live(b: Babbler, output: str, *, delivered: int = 3,
          undo_searches: int = 0, known: bool | None = None) -> Any:
    """Живой выход с заданной историей. Состояние выводится, а не назначается."""
    f = b.body.fact(output, 1)
    f.tries = f.delivered = delivered
    f.responded = delivered
    f.undo_searches = undo_searches
    if known is not None:
        f.reversibility = Reversibility().observe(known)
    return f


def _body(b: Babbler, **special: Any) -> None:
    """Всё тело разобрано: обратимость известна у всех, кроме перечисленных."""
    for o in OUTS:
        if o not in special:
            _live(b, o, known=True)
    for o, kw in special.items():
        _live(b, o, **kw)


# --- 1. цена и обратимость — разные величины ----------------------------------


def test_a_failed_guess_raises_the_price_and_leaves_reversibility_alone() -> None:
    b = Babbler(_prof(), OUTS, rng_seed=1)
    _body(b, **{OUTS[0]: dict(undo_searches=0)})
    assert b.error_cost(OUTS[0]) == 0.0, (
        "выход без единой попытки отката не может быть дорогим: свидетельств нет")

    f = b.body.outputs[OUTS[0]]
    f.undo_searches = 1
    assert b.error_cost(OUTS[0]) == 1.0, (
        "последствие видели, вернуть не смогли — это и есть «не умею откатить» "
        "(инвариант 9), а не полдороги к нему")
    assert f.reversibility.n == 0, (
        "неудачная догадка говорит о догадке, а не о мире: обратимость обязана "
        "остаться неизвестной")


def test_a_found_undo_cancels_every_earlier_failure() -> None:
    """Свидетельство «не умею» обязано отзываться. Три ошибки подряд были об этом.

    Каждая держала цену на единице после того, как способ отката находился, и каждая
    давала около 98 % ложных срабатываний на проверке «предсказывает ли цена неудачу
    следующей попытки» (`tools/measure_reversibility.py`).
    """
    b = Babbler(_prof(), OUTS, rng_seed=1)
    # Фон тела высокий: почти всё в этом теле откатить не удалось.
    _body(b, **{o: dict(undo_searches=3) for o in OUTS})
    assert b.base_cost() > 0.72, "тело должно быть «дорогим», иначе тест ничего не ловит"

    hero = OUTS[0]
    f = b.body.outputs[hero]
    f.undo_searches = 3                     # три догадки не подошли
    b._undo_exhausted.add(hero)             # и догадки кончились
    assert b.error_cost(hero) == 1.0

    # Способ нашёлся и работает.
    f.reversibility = Reversibility().observe(True).observe(True)
    f.undo_method = OUTS[5]
    assert b.error_cost(hero) == 0.0, (
        "способ найден и проверен — ни прошлые неудачные догадки, ни флаг «догадки "
        "кончились», ни высокий фон тела не имеют права держать цену на единице")
    assert not b.costly(hero, 0.72)


def test_the_price_of_the_unknown_starts_at_zero() -> None:
    """Инвариант 17: заранее опасным не помечено ничего, в том числе незнание."""
    b = Babbler(_prof(), OUTS, rng_seed=1)
    assert b.base_cost() == 0.0
    assert b.error_cost(OUTS[3]) == 0.0
    assert not b.costly(OUTS[3], 0.72)

    _body(b, **{OUTS[0]: dict(undo_searches=len(OUTS))})
    assert b.base_cost() > 0.0, "фон тела обязан вырасти после первого необратимого"
    assert b.base_cost() < 1.0, "один необратимый выход не делает дорогим всё тело"


# --- 2. порог 1.0 снимает гейт ------------------------------------------------


def test_a_threshold_of_one_really_removes_the_gate() -> None:
    """Иначе контроль «порог снят» измеряет сам себя (вырождение 13.9)."""
    assert too_risky(1.0, 0.72) is True
    assert too_risky(1.0, 1.0) is False, (
        "цена незнания — ровно единица, и при нестрогом сравнении порог 1.0 "
        "оставлял бы гейт включённым для всего непробованного")
    assert too_risky(0.72, 0.72) is False, "равенство порогу — не «выше порога»"

    b = Babbler(_prof(), OUTS, rng_seed=1)
    _body(b, **{OUTS[0]: dict(undo_searches=len(OUTS))})
    assert b.costly(OUTS[0], 0.72) is True
    assert b.costly(OUTS[0], 1.0) is False


# --- 3. где цена решает, а где нет -------------------------------------------


def test_the_price_moves_the_order_where_it_carries_information() -> None:
    """Ветвь «обратимость неизвестна»: два кандидата, и цена меняет выбор.

    Полнота знания говорит взять `hot` (проб меньше), цена — взять `cool`. При
    работающем пороге обязано выиграть второе, при снятом — первое. Именно этого
    расхождения не было до правки: ветвь порога не спрашивала вовсе.
    """
    hot, cool = OUTS[0], OUTS[1]
    order: dict[float, str] = {}
    for threshold in (0.72, 1.0):
        b = Babbler(_prof(), OUTS, rng_seed=1)
        _body(b, **{hot: dict(delivered=3, undo_searches=len(OUTS)),
                    cool: dict(delivered=4, undo_searches=0)})
        probe = b.next_probe(caution_threshold=threshold)
        assert probe is not None
        order[threshold] = probe.output

    assert order[0.72] == cool, "при работающем пороге дорогое обязано уйти в хвост"
    assert order[1.0] == hot, (
        "при снятом пороге обязана остаться прежняя расстановка — по полноте знания. "
        "Если и здесь выбран не тот, порядок решает что-то третье")


def test_the_untried_branch_cannot_read_the_price_and_that_is_declared() -> None:
    """Почему закреплённый тест TASK-32 C проходил и после починки.

    Он сравнивал первые четыре пробы, а они все приходят из ветви «ни разу не
    пробован», где цена у всех кандидатов одинакова — фон тела. Различать нечем, и
    первое нажатие необратимого выхода поэтому неизбежно: заранее про выход не известно
    ничего. Доказательством правки служат счётчики, а не эта ветвь.
    """
    first: dict[bool, list[str]] = {}
    whys: list[str] = []
    for undone in (False, True):
        b = Babbler(_prof(), OUTS, rng_seed=1)
        f = b.body.fact(OUTS[0], 1)
        f.reversibility = Reversibility().observe(undone)
        probes = [b.next_probe() for _ in range(4)]
        assert all(p is not None for p in probes)
        first[undone] = [p.output for p in probes if p]
        whys += [p.why for p in probes if p]

    assert first[False] == first[True]
    assert all(w == UNTRIED for w in whys), (
        "тест обязан идти именно по ветви «ни разу не пробован»: иначе он проверяет "
        "не то, о чём говорит")


def test_a_useless_repeat_is_dropped_and_a_deciding_one_is_not() -> None:
    """Цена снимает только тот повтор, который ничего не решает."""
    thin = OUTS[0]

    # Состояние ещё не выведено: повтор решает, и цена его не отменяет.
    b = Babbler(_prof(), OUTS, rng_seed=1)
    _body(b, **{thin: dict(delivered=2, undo_searches=len(OUTS))})
    b.body.outputs[thin].responded = 1          # один ответ: ещё не «отвечает»
    assert b.body.outputs[thin].state == "unclear"
    probe = b.next_probe()
    assert probe is not None and probe.output == thin and probe.why.startswith(THIN)
    assert b.repeats_dropped_by_cost == 0

    # Состояние выведено: повтор вывода не изменит, а цену заплатит.
    c = Babbler(_prof(), OUTS, rng_seed=1)
    _body(c, **{thin: dict(delivered=2, undo_searches=len(OUTS))})
    assert c.body.outputs[thin].state == "live"
    probe = c.next_probe()
    assert probe is not None
    assert not (probe.output == thin and probe.why.startswith(THIN)), (
        "повтор при выведенном состоянии и максимальной цене — чистый расход")
    assert c.repeats_dropped_by_cost == 1, "снятый повтор обязан быть посчитан"


# --- 4. счётчики ветвей -------------------------------------------------------


def test_every_probe_names_a_declared_branch() -> None:
    """Инвариант 26: ветвь без счётчика невидима, а счётчик по необъявленному списку
    врёт умолчанием. Поэтому проверяется и то, что имена совпадают с `Probe.why`."""
    prof = _prof()
    world = InteractiveWorld(prof, seed=3, n_outputs=12)
    b = Babbler(prof, world.outputs, rng_seed=3)
    whys: list[str] = []
    inner = b.next_probe

    def watched(**kw: Any) -> Any:
        p = inner(**kw)
        if p is not None:
            whys.append(p.why)
        return p

    b.next_probe = watched                      # type: ignore[method-assign]
    progress = run_babbling(world, b, steps=120, clocks=Clocks())

    assert whys, "лепет не сделал ни одной пробы"
    for why in whys:
        assert any(why.startswith(name) for name in BRANCHES), why
    assert sum(progress["branch_hits"].values()) == progress["probes_done"]
    assert set(progress["branch_hits"]) == set(BRANCHES)


def test_the_counters_of_irreversible_presses_diverge_with_and_without_the_gate() -> None:
    """Замена закреплённого дефекта TASK-32 C: расхождение обязано появиться.

    Замер на пяти сидах по 900 шагов (`tools/measure_reversibility.py`) даёт 4/62,
    4/95, 6/55, 6/67, 10/41 нажатий по-настоящему необратимого выхода при работающем
    пороге и при снятом. Здесь тот же опыт в уменьшенном виде — одного сида и 250
    шагов достаточно, чтобы расхождение было, а прогон оставался быстрым.

    Единица независимости — прогон. Утверждение слабее замера нарочно: тест
    закрепляет **наличие** расхождения, а его величина живёт в замере и меняется от
    правки к правке.
    """
    def presses(gate: bool) -> int:
        prof = _prof(**({} if gate else {"irreversibility_threshold": 1.0}))
        world = InteractiveWorld(prof, seed=1, n_outputs=12)
        truly = set(world.truth().get("irreversible_outputs", ()))
        assert truly, "в этом мире нет необратимых выходов — сравнивать нечего"
        pressed: list[str] = []
        inner = world.step

        def logging_step(action: Any = None, **kw: Any) -> Any:
            if action is not None:
                pressed.extend(action.outputs_touched())
            return inner(action, **kw)

        world.step = logging_step               # type: ignore[method-assign]
        b = Babbler(prof, world.outputs, rng_seed=1)
        run_babbling(world, b, steps=250, clocks=Clocks())
        return sum(1 for o in pressed if o in truly)

    with_gate, without = presses(True), presses(False)
    assert with_gate < without, (
        f"нажатий необратимого при пороге {with_gate}, при снятом {without}. "
        "Совпадение означает, что цена ошибки в разведке снова не читается")
