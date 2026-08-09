"""Единица независимости метрики. Инвариант 22, `TASK-03`, часть 0.

Правило проверяется на том, из-за чего оно появилось: доля, посчитанная по
зависимым событиям, выглядит убедительно и не отличима по числу от доли,
посчитанной по независимым единицам. Значит объявление обязано быть в типе.
"""
from __future__ import annotations

import pytest

from harness.model.units import (NOT_INDEPENDENT, UNITS, Aggregate, Independence,
                                 UnitError, count, mean, pseudoreplication,
                                 reduce_to_units, share)


def test_metric_without_a_unit_is_not_created() -> None:
    with pytest.raises(UnitError, match="без объявленной единицы"):
        Independence("", 10)


def test_pixel_and_frame_are_never_units() -> None:
    """Самая частая форма псевдорепликации в измерениях по изображениям."""
    for bad in ("пиксель", "кадр"):
        with pytest.raises(UnitError, match="не бывает единицей"):
            share(bad, 2_000_000)
        assert bad in NOT_INDEPENDENT


def test_attempt_is_not_a_unit_because_that_is_where_it_broke() -> None:
    """«Попытка» запрещена по имени: именно на ней сломался замер планов."""
    with pytest.raises(UnitError, match="попытки на одном и том же объекте"):
        share("попытка", 1913)
    # А маршрут — законная единица, и это тот же замер, сделанный правильно.
    assert share("маршрут", 94).n == 94


def test_unknown_unit_is_refused_with_the_closed_list() -> None:
    with pytest.raises(UnitError, match="Набор закрыт"):
        share("что-нибудь", 5)
    assert "прогон" in UNITS and "домен" in UNITS


def test_share_without_n_is_refused() -> None:
    """Доля без знаменателя в единицах — это и есть псевдорепликация."""
    with pytest.raises(UnitError, match="без числа единиц"):
        Independence("прогон", None, Aggregate.SHARE)
    # У счёта `n` необязателен: он ничего не утверждает за пределами выборки.
    assert count("прогон").is_claim is False


def test_count_does_not_claim_independence() -> None:
    c = count("запись", 412)
    assert not c.is_claim
    assert "счёт" in c.text()


def test_zero_units_means_no_number_at_all() -> None:
    empty = share("цель", 0)
    assert empty.empty and "считать нечего" in empty.text()


def test_thin_sample_is_marked_but_not_refused() -> None:
    """Две единицы — не ошибка, но по такой доле выводов делать нельзя."""
    assert share("домен", 2).thin
    assert not share("домен", 5).thin
    assert not count("домен", 2).thin, "счёт тонким не бывает"


def test_reduce_to_units_gives_the_pseudoreplication_factor() -> None:
    """Ровно та операция, которой не хватало: события против единиц."""
    attempts = [{"route": "A"}] * 60 + [{"route": "B"}] * 30 + [{"route": "C"}]
    events, units = reduce_to_units(attempts, key=lambda a: a["route"])
    assert (events, units) == (91, 3)
    assert pseudoreplication(events, units) == pytest.approx(91 / 3)


def test_every_vital_declares_a_unit(tmp_path) -> None:
    """Ни одного показателя без единицы во всей сводке."""
    import numpy as np

    from harness.core.profile import MILESTONE_0
    from harness.model import vitals
    from harness.session import Recorder, Session

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        for i in range(4):
            rec.record_frame(np.full((180, 320), i * 11, dtype=np.uint8))

    with Session.open(tmp_path / "s") as s:
        v = vitals.from_journal(s.journal, profile=s.profile)

    assert v.vitals, "сводка пуста — проверять нечего"
    for one in v.vitals:
        assert one.independence is not None, f"{one.code} без единицы"
    # И у каждой доли единица названа вслух, а не унаследована по умолчанию.
    for one in v.claims():
        assert one.independence is not None and one.independence.unit in UNITS


def test_vital_refuses_a_number_over_zero_units(tmp_path) -> None:
    """Доля по нулю единиц — не ноль, а отсутствие данных."""
    from harness.model.vitals import Vital

    with pytest.raises(ValueError, match="единиц ноль, а число выведено"):
        Vital("выдумка", 0.5, "доля", "журнал", independence=share("прогон", 0))


def test_vital_without_independence_is_refused() -> None:
    from harness.model.vitals import Vital

    with pytest.raises(ValueError, match="без объявленной единицы независимости"):
        Vital("без-единицы", 1, "штук", "журнал")


def test_report_shows_the_unit_next_to_every_share(tmp_path) -> None:
    """Правило, которого не видно человеку, не работает."""
    import numpy as np

    from harness.core.profile import MILESTONE_0
    from harness.model import vitals
    from harness.session import Recorder, Session

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        for i in range(3):
            rec.record_frame(np.full((180, 320), i * 3, dtype=np.uint8))
    with Session.open(tmp_path / "s") as s:
        text = vitals.from_journal(s.journal, profile=s.profile).render_text()

    assert "единиц «запись»" in text
    # У счёта единицы в строке нет: она тривиальна и была бы шумом.
    assert "счёт в единицах" not in text
