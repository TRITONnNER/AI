"""Тесты послойного взгляда: дальность, тау, частоты слоёв, фовеация.

Из замысла: «нужно делать послойно — интерфейс, передний план, средний, задний», «мы
отказываемся от координат», «оценка многих вещей происходит временем».

Тут проверяются четыре вещи, и одна из них — **отрицательный результат**. Дальность по
параллаксу не работает, это измерено, и тест на это есть: иначе через месяц кто-нибудь
(в том числе я) начнёт опираться на дальность как на рабочую величину. Тест написан
так, чтобы он **упал**, когда дальность начнёт работать, — тогда придётся обновить и
таблицу замеров, и признак `trustworthy`.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.core.action import Action
from harness.core.profile import from_schema
from harness.corpus import depth as dep
from harness.vision.layers import (FAR, MID, NEAR, UNKNOWN, DepthFromParallax,
                                   LayerClock, TimeToContact, attention_boxes,
                                   foveate, look)


def _profile(**kw: object):
    base: dict[str, object] = dict(capture_width=320, capture_height=180)
    base.update(kw)
    return from_schema("ТЕСТ-слои", **base)


# --- мир с глубиной ---------------------------------------------------------


def test_depth_world_planes_move_at_different_rates() -> None:
    """Глубина в мире должна быть, иначе её проверка проверяет фантазии."""
    from harness.vision.selfworld import estimate_global_shift

    p = _profile()
    w = dep.DepthWorld(p, seed=3)
    w.looming_on = False
    prev = w.step(None, with_audio=False).frame
    truth = w.depth_truth()
    w.step(Action.mouse(12, 0, duration_ms=33), with_audio=False)

    # Планы действительно разной «скорости»: это свойство мира, а не метода.
    rates = sorted(pl.parallax for pl in w.planes)
    assert rates[-1] / rates[0] > 10, "планы почти одинаковой скорости — нет глубины"
    for t in (dep.TRUTH_NEAR, dep.TRUTH_MID, dep.TRUTH_FAR):
        assert (truth == t).sum() > 500, f"плана {t} почти нет в кадре"
    assert estimate_global_shift(prev, w.render()).magnitude > 0


def test_depth_world_planes_are_large_and_connected() -> None:
    """Разрежённая сыпь смешивает глубины в каждом окне — на ней метод бессилен.

    Это была настоящая ошибка мира: пятна по 3–9 px давали 0–29 % верных, потому что
    окно сопоставления всегда накрывало два плана сразу.
    """
    p = _profile()
    w = dep.DepthWorld(p, seed=3)
    for plane in w.planes:
        solid = plane.texture > 0
        # Доля пикселей, у которых все четыре соседа тоже «свои»: у сыпи она мала.
        inner = (solid[1:-1, 1:-1] & solid[:-2, 1:-1] & solid[2:, 1:-1]
                 & solid[1:-1, :-2] & solid[1:-1, 2:])
        share = inner.sum() / max(1, solid[1:-1, 1:-1].sum())
        assert share > 0.8, f"план {plane.name}: области рваные ({share:.0%})"


# --- дальность: измеренный отрицательный результат ---------------------------


def test_depth_from_parallax_does_not_work_yet_and_says_so() -> None:
    """Дальность по параллаксу не находится. Это измерено, и это не скрыто.

    Верным получается только средний план — тот, что совпал с найденным сдвигом, то
    есть ответ «всё сместилось как весь кадр». Тест закрепляет числа: когда метод
    заработает, тест упадёт, и это правильно — придётся обновить таблицу замеров в
    описании модуля и снять `trustworthy=False`.
    """
    p = _profile(flow_window=11)
    w = dep.DepthWorld(p, seed=3)
    w.looming_on = False
    d = DepthFromParallax(p)
    rng = np.random.default_rng(1)
    for _ in range(60):
        dx = int(rng.integers(-14, 15))
        act = Action.mouse(dx, 0, duration_ms=33) if dx else None
        d.feed(w.step(act, with_audio=False).frame)

    r = d.result()
    truth = w.depth_truth()
    acc = {}
    for t, band, name in ((dep.TRUTH_NEAR, NEAR, "ближний"),
                          (dep.TRUTH_MID, MID, "средний"),
                          (dep.TRUTH_FAR, FAR, "дальний")):
        m = (truth == t) & (r.bands != UNKNOWN)
        acc[name] = 0.0 if not m.any() else float((r.bands[m] == band).mean())

    assert not r.trustworthy, "дальность объявлена надёжной — покажите замер"
    assert not r.summary()["trustworthy"]
    assert acc["средний"] > 0.4, (
        "даже средний план перестал находиться: сломалось что-то ещё")
    assert acc["ближний"] < 0.5 and acc["дальний"] < 0.5, (
        f"дальность заработала: {acc}. Это хорошая новость — обновите таблицу "
        "замеров в описании harness.vision.layers и снимите trustworthy=False")


def test_depth_refuses_when_the_camera_stands_still() -> None:
    """Нет движения — нет параллакса. Это отсутствие данных, а не «всё далеко»."""
    p = _profile()
    w = dep.DepthWorld(p, seed=3)
    w.looming_on = False
    d = DepthFromParallax(p)
    for _ in range(20):
        d.feed(w.step(None, with_audio=False).frame)
    r = d.result()
    assert r.voting_frames == 0 and r.skipped_frames >= 18
    assert (r.bands == UNKNOWN).all()


def test_depth_thresholds_must_not_overlap() -> None:
    with pytest.raises(ValueError, match="ниже порога ближнего"):
        DepthFromParallax(_profile(depth_near_ratio=1.0, depth_far_ratio=1.0))


# --- тау: время до контакта --------------------------------------------------


def test_tau_predicts_time_to_contact_without_any_distance() -> None:
    """Тау из скорости расширения. Ни расстояния, ни размера предмета, ни скорости.

    Так делают животные: олуша складывает крылья при фиксированном тау, а не на
    фиксированной высоте.
    """
    p = _profile()
    w = dep.DepthWorld(p, seed=3)
    tau = TimeToContact(p)
    pairs: list[tuple[float, float]] = []
    for _ in range(70):
        looming = tau.feed(w.step(None, with_audio=False).frame)
        truth = w.frames_to_contact()
        if looming.tau_steps is not None and truth:
            pairs.append((looming.tau_steps, truth))

    assert len(pairs) > 25, "тау не посчиталось почти нигде"
    rel = np.array([abs(a - b) / b for a, b in pairs])
    assert float(np.median(rel)) < 0.15, (
        f"медианная относительная ошибка тау {np.median(rel):.0%} — слишком много")


def test_tau_measures_the_object_not_the_whole_frame() -> None:
    """Размер — связная область от центра внимания, а не все яркие пиксели кадра.

    Первая версия считала все и давала 120 px при истинных 6: она мерила фон.
    """
    p = _profile()
    w = dep.DepthWorld(p, seed=3)
    tau = TimeToContact(p)
    frame = w.step(None, with_audio=False).frame
    size = tau.measure_size(frame)
    assert 0.5 * w.looming_size < size < 2.0 * w.looming_size, (
        f"размер {size:.1f} вместо примерно {w.looming_size:.1f}")


def test_tau_is_undefined_when_nothing_approaches() -> None:
    """Не растёт — тау нет. И это не то же, что «контакт далеко»."""
    p = _profile()
    w = dep.DepthWorld(p, seed=3)
    w.looming_on = False
    tau = TimeToContact(p)
    last = None
    for _ in range(20):
        last = tau.feed(w.step(None, with_audio=False).frame)
    assert last is not None and last.tau_steps is None


def test_tau_and_world_agree_on_what_contact_means() -> None:
    """Контакт — «предмет занял такую-то долю кадра», и так у обеих сторон.

    При разных определениях сверять тау с истиной бессмысленно: одна сторона считает
    до одного момента, другая до другого. На замере расхождение определений давало
    96 % относительной ошибки почти целиком из ничего.
    """
    for fraction in (0.1, 0.3):
        p = _profile(looming_contact_fraction=fraction)
        w = dep.DepthWorld(p, seed=3)
        # Размер по стороне, при котором мир объявляет контакт, должен совпадать с
        # тем, который агент считает контактом.
        agent_side = float(np.sqrt(fraction * w.width * w.height))
        world_side = float(np.sqrt(np.pi * (w.looming_contact_px / 2) ** 2))
        assert abs(agent_side - world_side) / agent_side < 0.02


# --- частоты слоёв и фовеация -----------------------------------------------


def test_layer_clock_updates_far_layer_rarely() -> None:
    """«Горы никуда не денутся»: дальний слой не надо считать каждый кадр.

    Экономия измеряется, а не предполагается.
    """
    p = _profile()
    clock = LayerClock.from_profile(p)
    for _ in range(300):
        clock.tick()
    updates = clock.updates
    assert updates["screen"] > updates["near"] > updates["mid"] > updates["far"]
    assert updates["far"] <= 3, "дальний слой обновлялся слишком часто"
    assert clock.savings() > 0.4, f"экономия всего {clock.savings():.0%}"
    assert clock.summary()["frames"] == 300


def test_layer_clock_honours_profile_frequencies() -> None:
    p = _profile(layer_screen_hz=10.0, layer_near_hz=10.0, layer_mid_hz=10.0,
                 layer_far_hz=10.0, capture_fps=10.0)
    clock = LayerClock.from_profile(p)
    for _ in range(50):
        due = clock.tick()
        assert set(due) == {"screen", "near", "mid", "far"}
    assert clock.savings() == 0.0, "при равных частотах экономить нечего"


def test_foveation_keeps_detail_only_where_attention_is() -> None:
    """Высокое разрешение в окне внимания, огрублённая периферия — как у глаза."""
    p = _profile(attention_windows=1)
    w = dep.DepthWorld(p, seed=3)
    frame = w.step(None, with_audio=False).frame
    boxes = attention_boxes(p, frame.shape)
    out, kept = foveate(frame, boxes, periphery=4)

    assert out.shape == frame.shape
    assert 0.02 < kept < 0.3, f"окно внимания занимает {kept:.0%} кадра"
    top, left, side, _ = boxes[0]
    inside = out[top:top + side, left:left + side]
    assert np.array_equal(inside, frame[top:top + side, left:left + side]), \
        "в окне внимания детали потеряны"
    # А в периферии — потеряны, иначе экономии нет.
    edge_out = out[:8, :8]
    edge_in = frame[:8, :8]
    assert not np.array_equal(edge_out, edge_in), "периферия не огрублена"


def test_look_takes_both_settings_from_the_profile() -> None:
    """Рабочий путь взгляда: и число окон, и огрубление — из профиля."""
    p = _profile(attention_windows=2, periphery_coarsening=8)
    w = dep.DepthWorld(p, seed=3)
    frame = w.step(None, with_audio=False).frame
    out, boxes, kept = look(p, frame)
    assert len(boxes) == 2
    assert out.shape == frame.shape
    assert 0.05 < kept < 0.4
    # Огрубление действительно применено: периферия отличается от исходника.
    assert not np.array_equal(out[:8, :8], frame[:8, :8])


def test_attention_window_count_comes_from_the_profile() -> None:
    frame_shape = (180, 320)
    assert len(attention_boxes(_profile(attention_windows=1), frame_shape)) == 1
    assert len(attention_boxes(_profile(attention_windows=3), frame_shape)) == 3


def test_foveation_of_one_means_no_coarsening() -> None:
    p = _profile()
    frame = np.arange(180 * 320, dtype=np.uint8).reshape(180, 320)
    out, kept = foveate(frame, attention_boxes(p, frame.shape), periphery=1)
    assert np.array_equal(out, frame) and kept == 1.0
    with pytest.raises(ValueError):
        foveate(frame, [], periphery=0)


# --- домен глубины проходит общий замер --------------------------------------


def test_depth_world_has_the_same_interface_as_other_domains() -> None:
    """Иначе кросс-доменный замер по нему не пройдёт."""
    from harness.corpus.domains import Domain

    p = _profile()
    w = dep.DepthWorld(p, seed=1)
    assert isinstance(w, Domain)
    obs = w.step(None, with_audio=False)
    assert obs.frame.shape == (180, 320)
    assert w.screen_mask().shape == (180, 320)
    assert w.animated_mask().shape == (180, 320)
    assert w.truth()["domain"] == "depth"
    assert len(w.outputs) >= 12
    assert w.last_action_changed is None
