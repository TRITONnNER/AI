"""Критерий «запуск состоялся». SPEC-FULL, A4.

Главное, что здесь проверяется, — что критерий **умеет не засчитать**. Прозаический
критерий проверяется чтением, а чтение уже дважды объявляло сошедшимся то, что не
сходилось: «узлы вышли на полку» при неверной оси и «отпечаток различает шум» при
неверном способе кормления.

Поэтому у пункта четыре исхода, а не два, и два из четырёх не засчитываются:
«не проверен» (данных нет) и «выполнен вакуумно» (сошлось потому, что механизм не
работал). Второй добавлен по замеру каскада, где пункт 5 формально выполнен во всех
60 прогонах при недостижимой ступени на четырёх доменах из пяти.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.launch import (FAILED, MET, POINTS, UNCHECKED, VACUOUS, Point,
                            Verdict, evaluate)


def _measurements(tmp_path: Path, **files: dict) -> Path:
    root = tmp_path / "measurements"
    root.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (root / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False),
                                           encoding="utf-8")
    return root


def _session(tmp_path: Path, *, synthetic: bool = False, fps: float = 30.0,
             entries: list[dict] | None = None) -> Path:
    root = tmp_path / "сессия"
    (root / "journal" / "branches" / "000").mkdir(parents=True, exist_ok=True)
    (root / "session.json").write_text(json.dumps({
        "synthetic": synthetic, "profile": {"parameters": {"capture_fps": fps}},
    }, ensure_ascii=False), encoding="utf-8")
    lines = "\n".join(json.dumps(e, ensure_ascii=False) for e in (entries or []))
    (root / "journal" / "branches" / "000" / "entries.jsonl").write_text(
        lines, encoding="utf-8")
    return root


# --- исходы, которые не засчитываются ---------------------------------------


def test_a_vacuous_point_does_not_count() -> None:
    """Весь смысл четвёртого исхода: «сошлось, потому что не работало» — не «сошлось»."""
    p = Point(5, "любой", VACUOUS, value=0.0)
    assert not p.counts, "вакуумный пункт засчитан — критерий объявит запуск состоявшимся"
    assert not Point(5, "любой", UNCHECKED).counts
    assert not Point(5, "любой", FAILED).counts
    assert Point(5, "любой", MET).counts


def test_all_seven_are_required() -> None:
    """«Шесть из семи» не бывает: спецификация говорит «все семь обязательны»."""
    six = tuple(Point(n, name, MET) for n, name in POINTS[:6])
    assert not Verdict(six).launched, "критерий засчитал неполный набор"
    seven = tuple(Point(n, name, MET) for n, name in POINTS)
    assert Verdict(seven).launched and Verdict(seven).met == 7


def test_one_vacuous_point_blocks_the_launch() -> None:
    points = [Point(n, name, MET) for n, name in POINTS]
    points[4] = Point(5, POINTS[4][1], VACUOUS)
    v = Verdict(tuple(points))
    assert not v.launched and v.met == 6
    assert v.as_dict()["blocks_b"] is True


# --- пункты без данных называют, чего не хватает -----------------------------


def test_missing_data_is_unchecked_and_says_what_to_run(tmp_path: Path) -> None:
    """Правдоподобное значение вместо отсутствующего — худшее, что тут может быть."""
    v = evaluate(tmp_path / "нет-такой", measurements=_measurements(tmp_path))
    assert not v.launched and v.met == 0
    by_number = {p.number: p for p in v.points}
    for n in (1, 2, 4, 6, 7):
        assert by_number[n].verdict == UNCHECKED, f"пункт {n}: {by_number[n].verdict}"
        assert by_number[n].fix, f"пункт {n} не сказал, чем добыть данные"


def test_a_synthetic_session_does_not_satisfy_the_hour(tmp_path: Path) -> None:
    """Пункт 1 требует настоящего экрана, и синтетика его не заменяет."""
    session = _session(tmp_path, synthetic=True,
                       entries=[{"kind": "frame", "stamp": {"t_world": i}}
                                for i in range(10_000)])
    v = evaluate(session, measurements=_measurements(tmp_path))
    first = next(p for p in v.points if p.number == 1)
    assert first.verdict == UNCHECKED
    assert "синтетическая" in first.why


# --- пункты, которые считаются ------------------------------------------------


def test_the_hour_needs_both_length_and_no_intervention(tmp_path: Path) -> None:
    frames = [{"kind": "frame", "stamp": {"t_world": i}} for i in range(108_000)]
    long_enough = _session(tmp_path, entries=frames)
    v = evaluate(long_enough, measurements=_measurements(tmp_path))
    assert next(p for p in v.points if p.number == 1).verdict == MET

    with_human = _session(tmp_path / "второй",
                          entries=frames + [{"kind": "intervention", "stamp": {}}])
    v2 = evaluate(with_human, measurements=_measurements(tmp_path))
    got = next(p for p in v2.points if p.number == 1)
    assert got.verdict == FAILED and "прерыв" in got.why


def test_a_paused_world_is_caught_by_turns_against_ticks(tmp_path: Path) -> None:
    """Мир ушёл вперёд без оборота — значит где-то была пауза (инвариант 3)."""
    jumped = [{"kind": "frame", "stamp": {"t_world": i}} for i in range(50)]
    jumped += [{"kind": "frame", "stamp": {"t_world": i}} for i in range(500, 550)]
    v = evaluate(_session(tmp_path, entries=jumped),
                 measurements=_measurements(tmp_path))
    got = next(p for p in v.points if p.number == 2)
    assert got.verdict == FAILED and "пауза" in got.why


def test_sleep_that_merges_nothing_is_vacuous_not_met(tmp_path: Path) -> None:
    """Число не выросло, но и падать было нечему: пункт сошёлся сам собой."""
    quiet = [{"kind": "sleep", "stamp": {},
              "event": {"entities_before": 5, "entities_after": 5,
                        "merges_applied": []}}]
    v = evaluate(_session(tmp_path, entries=quiet),
                 measurements=_measurements(tmp_path))
    got = next(p for p in v.points if p.number == 4)
    assert got.verdict == VACUOUS and not got.counts

    working = [{"kind": "sleep", "stamp": {},
                "event": {"entities_before": 7, "entities_after": 5,
                          "merges_applied": [["a", "b"], ["c", "d"]]}}]
    v2 = evaluate(_session(tmp_path / "второй", entries=working),
                  measurements=_measurements(tmp_path))
    assert next(p for p in v2.points if p.number == 4).verdict == MET


def test_a_grown_entity_count_after_sleep_fails(tmp_path: Path) -> None:
    grew = [{"kind": "sleep", "stamp": {},
             "event": {"entities_before": 5, "entities_after": 9,
                       "merges_applied": [["a", "b"]]}}]
    v = evaluate(_session(tmp_path, entries=grew),
                 measurements=_measurements(tmp_path))
    assert next(p for p in v.points if p.number == 4).verdict == FAILED


def test_the_cascade_point_reads_the_vacuity_the_measurement_reported(
        tmp_path: Path) -> None:
    """Пункт 5 обязан читать не только долю, но и то, была ли она вакуумной."""
    vacuous = _measurements(tmp_path, cascade={
        "outcome": {"runs_over_launch_ceiling": 0, "model_share_median_all": 0.0017,
                    "launch_ceiling_met_but_vacuous_on": ["game", "video"]}})
    got = next(p for p in evaluate(tmp_path / "нет", measurements=vacuous).points
               if p.number == 5)
    assert got.verdict == VACUOUS and "недостижима" in got.why

    honest = _measurements(tmp_path / "чистый", cascade={
        "outcome": {"runs_over_launch_ceiling": 0, "model_share_median_all": 0.03,
                    "launch_ceiling_met_but_vacuous_on": []}})
    got2 = next(p for p in evaluate(tmp_path / "нет", measurements=honest).points
                if p.number == 5)
    assert got2.verdict == MET


def test_saturation_is_read_from_the_measurement(tmp_path: Path) -> None:
    """Пункт 3 берёт хвост кривой, и берёт худшее сочетание, а не среднее."""
    bad = _measurements(tmp_path, recognition={
        "rows": [{"new_per_observation_tail": 0.0001},
                 {"new_per_observation_tail": 0.22}]})
    got = next(p for p in evaluate(tmp_path / "нет", measurements=bad).points
               if p.number == 3)
    assert got.verdict == FAILED and got.value == pytest.approx(0.22)

    good = _measurements(tmp_path / "хороший", recognition={
        "rows": [{"new_per_observation_tail": 0.0001},
                 {"new_per_observation_tail": 0.002}]})
    got2 = next(p for p in evaluate(tmp_path / "нет", measurements=good).points
                if p.number == 3)
    assert got2.verdict == MET


def test_a_budget_breach_fails_the_spend_point(tmp_path: Path) -> None:
    over = [{"kind": "resource", "stamp": {}, "event": {"code": "spend_cap"}}]
    v = evaluate(_session(tmp_path, entries=over),
                 measurements=_measurements(tmp_path))
    assert next(p for p in v.points if p.number == 6).verdict == FAILED


def test_a_goal_must_be_passed_and_not_merely_set(tmp_path: Path) -> None:
    only_set = [{"kind": "goal", "stamp": {}, "event": {"code": "set"}},
                {"kind": "goal", "stamp": {}, "event": {"code": "abandoned"}}]
    v = evaluate(_session(tmp_path, entries=only_set),
                 measurements=_measurements(tmp_path))
    got = next(p for p in v.points if p.number == 7)
    assert got.verdict == FAILED and got.value == 0

    passed = only_set + [{"kind": "goal", "stamp": {}, "event": {"code": "passed"}}]
    v2 = evaluate(_session(tmp_path / "второй", entries=passed),
                  measurements=_measurements(tmp_path))
    assert next(p for p in v2.points if p.number == 7).verdict == MET


# --- пороги в схеме (инвариант 23) -------------------------------------------


def test_the_thresholds_live_in_the_schema() -> None:
    """Число в `launch.py` не попало бы в `profile_hash`, и прогоны стали бы несравнимы."""
    from harness.core.settings import SCHEMA

    keys = {s.key for s in SCHEMA}
    for key in ("launch_hour_seconds", "launch_max_deep_share",
                "launch_max_new_per_observation"):
        assert key in keys, f"порог критерия {key} не объявлен в схеме"
