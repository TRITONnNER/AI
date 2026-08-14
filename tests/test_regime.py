"""Режим среды: четыре признака и гейт механизмов. TASK-32, направление A.

Числа с единицами живут в `tools/measure_regime.py` и `docs/measurements/regime.json`.
Здесь — утверждения об устройстве, и каждое из них уже один раз оказалось неверным:

1. Признак определяется **наблюдением**, а не настройкой: иначе это подсказка (инвариант 4).
2. «Не определено» — не то же, что «нет»: механизм при первом неизвестен, при втором
   неприменим, и это разные решения.
3. Неприменимый механизм **не выдаёт числа** — он называет отсутствующий признак.
4. Эго-движение сравнивается с бездействием, а не с нулём: в среде, которая идёт сама,
   сдвиг находится и без всякого действия.
5. «Мир идёт без меня» — величина относительная: у всех доменов обрамление анимировано, и
   по абсолютной мерке признак находился везде.
6. Возврат вида проверяется отпечатком, а не средней разностью пикселей: мигающий
   индикатор не даёт разности упасть до нуля никогда.

Единица независимости — **утверждение об устройстве механизма**. Окружения не требует.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.core.profile import from_schema
from harness.perception.regime import (EGO_MOTION, FEATURES, JUDGE, MECHANISMS,
                                       REVERSIBLE, WORLD_ALONE, Evidence, Regime,
                                       RegimeError, applicable, detect, report)


def _prof(**kw):
    base = dict(capture_width=64, capture_height=48)
    base.update(kw)
    return from_schema("РЕЖИМ", **base)


def _regime(**feats) -> Regime:
    r = Regime()
    for f, v in feats.items():
        name = {"ego": EGO_MOTION, "alone": WORLD_ALONE,
                "rev": REVERSIBLE, "judge": JUDGE}[f]
        r.add(Evidence(name, v, 3, "проба", "поставлено тестом"))
    return r


# --- «не определено» против «нет» --------------------------------------------


def test_undetermined_is_not_the_same_as_absent() -> None:
    """Механизм при «не определено» неизвестен, при «нет» неприменим. Разные решения."""
    unknown = applicable("цель с внешним судьёй", _regime(judge=None))
    assert unknown.applicable is None and "неизвестно" in unknown.line

    absent = applicable("цель с внешним судьёй", _regime(judge=False))
    assert absent.applicable is False and "неприменимо" in absent.line
    assert absent.missing == (JUDGE,)


def test_a_feature_never_asked_is_undetermined_not_false() -> None:
    """Признак, который не спрашивали, не бывает «нет»: спрашивать было нечем."""
    empty = Regime()
    assert empty.has(EGO_MOTION) is None
    assert applicable("параллакс", empty).applicable is None


def test_the_feature_set_is_closed() -> None:
    """Пятый признак — изменение схемы записи, а не строка на месте вызова."""
    assert len(FEATURES) == 4
    with pytest.raises(RegimeError, match="нет такого признака"):
        Regime().add(Evidence("пятый", True, 1, "проба", ""))
    with pytest.raises(RegimeError, match="нет такого механизма"):
        applicable("телепатия", _regime(ego=True))


# --- гейт: неприменимый механизм не выдаёт числа -----------------------------


def test_inapplicable_names_the_missing_feature() -> None:
    """«Неприменимо» без названного признака — то же молчание, только вежливее."""
    v = applicable("параллакс", _regime(ego=False, alone=False))
    assert v.applicable is False
    assert EGO_MOTION in v.missing
    assert EGO_MOTION in v.line


def test_parallax_needs_a_quiet_world_and_not_only_a_moving_self() -> None:
    """Второе условие параллакса выяснилось замером, а не рассуждением.

    Мало того, чтобы мои действия давали сдвиг: надо ещё, чтобы мир не ехал сам между
    кадрами, — иначе сдвиг вызван не мной, а признак приписывает его моему движению.
    Именно так проигрыватель видео и дал IoU 0.196.
    """
    assert applicable("параллакс", _regime(ego=True, alone=False)).applicable is True
    noisy = applicable("параллакс", _regime(ego=True, alone=True))
    assert noisy.applicable is False
    assert any(m.startswith("не ") for m in noisy.missing), (
        "отрицательное требование обязано быть видно в списке недостающего")


def test_every_mechanism_says_why_it_needs_its_features() -> None:
    """Таблица без обоснования — список запретов, а его в проекте быть не должно."""
    for name, spec in MECHANISMS.items():
        assert spec["needs"], name
        assert len(spec["why"]) > 60, f"{name}: обоснование слишком короткое"


def test_report_splits_three_outcomes_not_two() -> None:
    """Применимо, неприменимо, не определено — три исхода, как у конвейера убеждений."""
    got = report(_regime(ego=True, alone=False, rev=False, judge=None))
    assert "параллакс" in got["applicable"]
    assert "оценка обратимости" in got["inapplicable"]
    assert "цель с внешним судьёй" in got["undetermined"]
    assert set(got["applicable"]) & set(got["inapplicable"]) == set()


# --- определение наблюдением -------------------------------------------------


def test_detect_takes_nothing_but_frames_and_outputs() -> None:
    """Ни маски, ни состояния, ни имён: единственный доступ — подать действие и смотреть.

    Проверяется подписью и поведением: детектору даётся замкнутая среда, которая на
    любое действие отвечает одним и тем же кадром. Правильный ответ — «ничего нет».
    """
    frame = np.full((48, 64), 100.0)
    calls: list[object] = []

    def step(action):
        calls.append(action)
        return frame

    r = detect(step=step, outputs=("OUT_01", "OUT_02"), profile=_prof())
    assert calls, "детектор обязан действовать, а не догадываться"
    assert r.has(EGO_MOTION) is False
    assert r.has(WORLD_ALONE) is False
    assert r.has(REVERSIBLE) is None, (
        "в замкнутой среде ни одно действие не изменило вид, значит возвращать было "
        "нечего. Это «проверить было нечем», а не «обратимых действий нет» — до TASK-33 B "
        "здесь стояло False, и различие терялось")
    assert r.evidence[REVERSIBLE].n == 0, "пар не было — и это видно по n"
    assert r.has(JUDGE) is None, "канал оценки не выдумывается по пикселям"


def test_a_world_that_moves_by_itself_is_not_ego_motion() -> None:
    """Сдвиг при бездействии не считается эго-движением, даже если он большой.

    Первая редакция сравнивала сдвиг с нулём и объявила эго-движение на проигрывателе
    видео, где камеры нет вовсе: содержимое едет само, фазовая корреляция находит сдвиг.
    """
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (48, 64)).astype(np.float64)
    state = {"t": 0}

    def step(action):
        # Мир едет сам на 8 пикселей за кадр, что бы я ни делал.
        state["t"] += 8
        return np.roll(base, state["t"], axis=1)

    r = detect(step=step, outputs=("OUT_01", "OUT_02"), profile=_prof())
    assert r.has(WORLD_ALONE) is True, "мир едет сам — это обязано находиться"
    assert r.has(EGO_MOTION) is False, (
        "сдвиг есть, но он не от моих действий: эго-движение объявлять нельзя")
    assert applicable("параллакс", r).applicable is False


def test_ego_motion_is_found_when_the_action_shifts_the_frame() -> None:
    """И обратное: действие, двигающее кадр, обязано находиться."""
    rng = np.random.default_rng(2)
    base = rng.integers(0, 255, (48, 64)).astype(np.float64)
    state = {"x": 0}

    def step(action):
        if action is not None:
            state["x"] += 12          # двигаю себя, и только когда действую
        return np.roll(base, state["x"], axis=1)

    r = detect(step=step, outputs=("OUT_01",), profile=_prof())
    assert r.has(EGO_MOTION) is True
    assert r.has(WORLD_ALONE) is False, "без действий мир стоит — признака быть не должно"
    assert applicable("параллакс", r).applicable is True


def test_thresholds_live_in_the_schema(profile=None) -> None:
    """Инвариант 23: порог, живущий в коде детектора, не сравним между прогонами."""
    prof = _prof()
    for key in ("regime_alone_margin", "regime_alone_ratio", "regime_ego_over_idle"):
        assert key in prof.parameters, f"{key} обязан жить в схеме"
    # И порог действительно читается — но **только там, где ему есть что ограничивать**.
    # Множитель сравнивает сдвиг от действия со сдвигом при бездействии; в неподвижном
    # мире второй равен нулю, и любой множитель на ноль даёт ноль. Это свойство величины,
    # а не недоделка: там, где мир стоит, любой согласованный сдвиг и есть эго-движение,
    # и ограничивать его нечем, кроме абсолютного пола `flow_min_global_shift`.
    rng = np.random.default_rng(3)
    base = rng.integers(0, 255, (48, 64)).astype(np.float64)
    state = {"x": 0}

    def moving_world(action):
        state["x"] += 4               # мир едет сам на 4 px
        if action is not None:
            state["x"] += 6           # и ещё на 6, если я действую
        return np.roll(base, state["x"], axis=1)

    assert detect(step=moving_world, outputs=("OUT_01",),
                  profile=_prof()).has(EGO_MOTION) is True
    strict = _prof(regime_ego_over_idle=100.0)
    assert detect(step=moving_world, outputs=("OUT_01",),
                  profile=strict).has(EGO_MOTION) is False, (
        "со строгим множителем эго-движение обязано перестать находиться там, где мир "
        "и сам едет — иначе множитель не читается")

    still = {"x": 0}

    def still_world(action):
        if action is not None:
            still["x"] += 12
        return np.roll(base, still["x"], axis=1)

    assert detect(step=still_world, outputs=("OUT_01",),
                  profile=strict).has(EGO_MOTION) is True, (
        "в неподвижном мире множитель ограничивать нечего: сдвиг при бездействии ноль")


# --- TASK-32, C → TASK-33, A: дефект исправлен, тест переехал -------------------
#
# Здесь стоял закреплённый дефект `test_caution_does_not_yet_reach_the_exploration_path`:
# порядок проб не спрашивал цену ошибки, и счётчики нажатий необратимого выхода совпадали
# при работающем пороге и при снятом. Дефект исправлен в TASK-33 A, и проверки переехали в
# `tests/test_error_cost.py` — там и расхождение счётчиков, и обе ошибки метки «дорого».
#
# Отдельно стоит запомнить, **чем закреплённый тест оказался плох**: он сравнивал первые
# четыре пробы, а они все приходят из ветви «ни разу не пробован», где цена у всех
# кандидатов одинакова по построению. Поэтому он прошёл бы и до правки, и после — то есть
# упасть при починке, как обещал, не мог. Доказательством стали счётчики, а не он.


# --- TASK-33, B: опорный уровень домена ---------------------------------------


def _noisy_world(seed: int = 5, *, own_motion: int = 0):
    """Мир, который едет сам на `own_motion` пикселей за шаг, и на 6 — от действия."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (48, 64)).astype(np.float64)
    state = {"x": 0}

    def step(action):
        state["x"] += own_motion
        if action is not None:
            state["x"] += 6
        return np.roll(base, state["x"], axis=1)

    return step


def test_without_a_stillness_record_every_feature_says_undetermined() -> None:
    """Главное требование B: нет фона — нет ответа. Порог не подбирается.

    Признак, оставшийся без опорного уровня, обязан отвечать «не определено», а не
    сравнивать с нулём: сравнение с нулём уже трижды дало неверный признак — эго-движение
    в проигрывателе видео, «мир идёт сам» во всех пяти доменах и «вид не вернулся» нигде.
    """
    from harness.perception.regime import NO_BASELINE

    r = detect(step=_noisy_world(own_motion=4), outputs=("OUT_01",), profile=_prof(),
               baseline=NO_BASELINE)
    for f in (EGO_MOTION, WORLD_ALONE, REVERSIBLE):
        assert r.has(f) is None, f"{f} ответил без опорного уровня"
        assert "записи неподвижности нет" in r.evidence[f].detail
        assert r.evidence[f].reference == "нет"
    # А признак, которому фон не нужен, ведёт себя как раньше: канал оценки — не измерение
    # кадра, и опорный уровень ему не требуется.
    assert r.evidence[JUDGE].reference != "нет"
    assert not r.baseline.present


def test_a_short_stillness_record_is_also_undetermined() -> None:
    """Мало кадров — тоже «не определено»: одна пара даёт уверенность, которой нет."""
    from harness.perception.regime import measure_baseline

    prof = _prof(regime_baseline_min_frames=8)
    step = _noisy_world(own_motion=3)
    short = measure_baseline(step=step, frames=2, tolerance=1,
                             print_of=lambda fr: _fingerprint(fr, prof))
    assert short.present and not short.enough(8)
    r = detect(step=step, outputs=("OUT_01",), profile=prof, baseline=short)
    assert r.has(EGO_MOTION) is None
    assert f"нужно {8}" in r.evidence[EGO_MOTION].detail


def _fingerprint(frame, prof):
    from harness.model.places import fingerprint
    return fingerprint(frame, grid=int(prof.structural["place_grid"]),
                       levels=int(prof.structural["place_levels"]),
                       blur_px=int(prof.structural["place_blur_px"]))


def test_the_baseline_measures_the_gap_the_check_actually_uses() -> None:
    """Фон снимается через тот же промежуток, что и проверка обратимости, а не через один.

    Между видом до действия и видом после двух действий проходит два шага мира. Мерка
    через один шаг строже настоящего фона ровно на один шаг дрейфа, и в едущем домене
    «вид вернулся» не находилось бы никогда.
    """
    from harness.perception.regime import measure_baseline

    prof = _prof()
    step = _noisy_world(own_motion=5)
    bl = measure_baseline(step=step, frames=10, tolerance=1,
                          print_of=lambda fr: _fingerprint(fr, prof), gap=2)
    assert bl.gap == 2
    assert bl.sim_gap_med <= bl.sim_med, (
        "за два шага домен уезжает дальше, чем за один: фон через больший промежуток не "
        "может быть строже")

    still = measure_baseline(step=_noisy_world(own_motion=0), frames=10, tolerance=1,
                             print_of=lambda fr: _fingerprint(fr, prof), gap=2)
    assert still.sim_gap_med == 1.0 and still.shift_max == 0.0, (
        "в неподвижном домене опорный уровень обязан быть строгим, а не «примерно»")


def test_the_reference_of_the_view_check_is_declared_and_measured() -> None:
    """Инвариант 23: отсчёт для «вид вернулся» живёт в схеме и меняет ответ.

    Три отсчёта замерены по пяти доменам, и победил критерий тождества вида
    (`MEASUREMENT.md`, 35). Здесь проверяется не победитель, а то, что выбор действительно
    читается: настройка, не меняющая ни одного ответа, была бы мёртвой.
    """
    prof = _prof()
    assert prof.structural["regime_view_reference"] == "criterion"
    step = _noisy_world(own_motion=4)
    bars = set()
    for choice in ("criterion", "background", "looser"):
        r = detect(step=_noisy_world(own_motion=4), outputs=("OUT_01", "OUT_02"),
                   profile=_prof(regime_view_reference=choice))
        bars.add(r.evidence[REVERSIBLE].reference.split("не ниже ")[1][:5])
    assert len(bars) > 1, "выбор отсчёта не меняет планку — значит настройка мёртвая"
    del step


def test_the_reference_is_written_down_for_every_feature() -> None:
    """У каждого признака сказано, с чем сравнивали. Иначе «сравнили с фоном» — слова."""
    r = detect(step=_noisy_world(own_motion=2), outputs=("OUT_01",), profile=_prof())
    for f in FEATURES:
        assert r.evidence[f].reference, f"{f} не говорит, с чем сравнивали"
    assert "бездействие" in r.evidence[EGO_MOTION].reference
    assert r.baseline.frames > 0
    assert r.baseline.as_dict()["unit"] == "пара кадров без действия"
