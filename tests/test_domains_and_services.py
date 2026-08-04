"""Тесты доменной независимости и бесплатных сервисов описания.

Три части, и все три про одно требование из `CLAUDE.md`: «архитектура обязана быть
доменно-независимой: если решение работает только в Minecraft, оно неправильное».

1. **Домены.** Четыре разных мира с одним интерфейсом. Проверяется не то, что они
   красивые, а то, что они действительно разные: если бы все они двигали содержимое
   одинаково, замер по ним ничего бы не проверял.
2. **Второй признак разделения слоёв и арбитр.** Что признак работает там, где
   параллакса нет, что он не выдаёт себя за параллакс, и что арбитр называет
   применённый признак, а не умалчивает о нём.
3. **Сервисы описания.** Что без ключа модуль отказывается громко, что квота
   считается, что кэш держится на побитовом тождестве кадра, и что формат ответа
   навязан промптом — иначе файрвол восприятия нечем проверять.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from harness.core.action import Action
from harness.core.clocks import Clocks
from harness.core.profile import from_schema

# --- домены -----------------------------------------------------------------


def _profile(**kw: object):
    base = dict(capture_width=320, capture_height=180)
    base.update(kw)
    return from_schema("ТЕСТ-домены", **base)


def test_all_domains_share_one_interface() -> None:
    """Один интерфейс на все домены. Иначе кросс-доменный замер невозможен."""
    from harness.corpus.domains import DOMAINS, Domain, make_domain

    p = _profile()
    for name in DOMAINS:
        d = make_domain(name, p, seed=1)
        assert isinstance(d, Domain), name
        obs = d.step(None, with_audio=False)
        assert obs.frame.shape == (180, 320)
        assert d.screen_mask().shape == (180, 320)
        assert d.animated_mask().shape == (180, 320)
        assert d.truth()["domain"] == name
        assert len(d.outputs) >= 12


def test_domains_move_content_differently() -> None:
    """Домены обязаны различаться механикой движения, иначе они один домен.

    Проверяется по глобальному сдвигу: в игре он двумерный, в документе — только по
    вертикали, на рабочем столе его нет вовсе.
    """
    from harness.corpus.domains import make_domain
    from harness.vision.selfworld import estimate_global_shift

    p = _profile()
    seen: dict[str, set[tuple[int, int]]] = {}
    for name in ("game", "document", "desktop"):
        d = make_domain(name, p, seed=2)
        prev = d.step(None, with_audio=False).frame
        shifts = set()
        for _ in range(12):
            cur = d.step(Action.mouse(30, 20, duration_ms=33), with_audio=False).frame
            s = estimate_global_shift(prev, cur)
            if s.magnitude >= 1.5:
                shifts.add((int(np.sign(s.dy)), int(np.sign(s.dx))))
            prev = cur
        seen[name] = shifts

    assert any(dx != 0 for _, dx in seen["game"]), "в игре нет сдвига по горизонтали"
    assert seen["document"], "в документе нет сдвига вовсе"
    assert all(dx == 0 for _, dx in seen["document"]), \
        "документ прокручивается по горизонтали, а должен только по вертикали"
    assert not seen["desktop"], "на рабочем столе появился глобальный сдвиг"


def test_video_content_moves_without_actions() -> None:
    """В видео третьи часы идут сами. На этом проверяется, что агент не приписывает
    себе чужое движение."""
    from harness.corpus.domains import make_domain

    d = make_domain("video", _profile(), seed=3)
    a = d.step(None, with_audio=False).frame
    t0 = d.t_content
    b = d.step(None, with_audio=False).frame
    assert d.t_content is not None and t0 is not None and d.t_content > t0
    assert not np.array_equal(a, b), "содержимое не идёт само"


def test_content_never_leaves_a_static_stripe() -> None:
    """Содержимое обязано покрывать кадр целиком.

    Полоса по краю, оставшаяся от растяжения, не двигается — и разделение слоёв
    честно объявит её экранным слоем. Результат замера от этого вырастет, а причина
    будет в генераторе. Такую ошибку по числам не видно, поэтому она проверяется.
    """
    from harness.corpus.domains import _upscale_to

    with pytest.raises(ValueError, match="меньше кадра"):
        _upscale_to(np.zeros((4, 4), dtype=np.uint8), 2, 180, 320)


def test_desktop_truth_counts_wallpaper_as_screen_layer() -> None:
    """Обои прибиты к экрану так же, как панель задач.

    Прежняя истина («экранный слой — только панель») наказывала метод за правильный
    ответ: неподвижные обои считались миром.
    """
    from harness.corpus.domains import make_domain

    d = make_domain("desktop", _profile(), seed=4)
    d.step(None, with_audio=False)
    mask = d.screen_mask()
    assert mask.mean() > 0.5, "обои не попали в экранный слой"


def test_desktop_window_becomes_content_only_after_it_moves() -> None:
    """Окно, ни разу не двинувшееся, в этой записи прибито к экрану.

    Слой определён системой отсчёта, а система отсчёта видна только по движению.
    """
    from harness.corpus.domains import make_domain

    d = make_domain("desktop", _profile(), seed=4)
    d.step(None, with_audio=False)
    before = d.screen_mask().sum()
    for _ in range(6):
        d.step(Action.mouse(9, 7, duration_ms=33), with_audio=False)
    after = d.screen_mask().sum()
    assert after < before, "двинувшееся окно осталось в экранном слое"


def test_last_action_changed_is_not_the_same_as_world_changed() -> None:
    """«Подействовало моё действие» и «мир изменился» — разные величины.

    В проигрывателе картинка идёт каждый кадр: по общему изменению живыми выглядели
    бы все выходы разом.
    """
    from harness.corpus.domains import make_domain

    d = make_domain("video", _profile(), seed=5)
    silent = [o for o, e in d._effects.items() if e is None]
    assert silent, "в этом сиде нет молчащих выходов, тест бессмысленен"
    obs = d.step(Action.key(silent[0], 100), with_audio=False)
    assert obs.changed, "содержимое видео не идёт — тест проверяет не то"
    assert d.last_action_changed is False


# --- второй признак и арбитр ------------------------------------------------


def test_stillness_finds_chrome_where_parallax_has_nothing_to_go_on() -> None:
    """Там, где нет глобального сдвига, отвечает второй признак."""
    from harness.corpus.domains import make_domain
    from harness.vision.selfworld import SCREEN, SelfWorldSeparator, StillnessSeparator

    p = _profile()
    d = make_domain("video", p, seed=6)
    par, still = SelfWorldSeparator(p), StillnessSeparator(p)
    for _ in range(60):
        f = d.step(None, with_audio=False).frame
        par.feed(f)
        still.feed(f)

    truth = d.screen_mask() & ~d.animated_mask()
    assert par.result().voting_frames <= 1, "в видео нашёлся глобальный сдвиг"
    found = still.result().pixel_mask(SCREEN)
    assert (found & truth).sum() / truth.sum() > 0.9
    assert (found & ~d.screen_mask()).sum() == 0, "содержимое попало в экранный слой"


def test_stillness_refuses_when_nothing_changes() -> None:
    """Если не меняется ничего, «неподвижное» ничего не выделяет."""
    from harness.vision.selfworld import UNDECIDED, StillnessSeparator

    p = _profile()
    still = StillnessSeparator(p)
    frame = np.zeros((180, 320), dtype=np.uint8)
    for _ in range(20):
        still.feed(frame)
    r = still.result()
    assert r.voting_frames == 0 and r.skipped_frames == 19
    assert (r.labels == UNDECIDED).all()


def test_stillness_ignores_pixels_without_texture() -> None:
    """На однородном участке «не изменился» ничего не значит.

    Гладкое небо не меняется и при движении камеры. Без этого запрета однородные
    области мира уехали бы в экранный слой.
    """
    from harness.vision.selfworld import SCREEN, StillnessSeparator

    p = _profile()
    still = StillnessSeparator(p)
    rng = np.random.default_rng(0)
    for i in range(30):
        f = np.zeros((180, 320), dtype=np.uint8)
        f[:, 160:] = rng.integers(0, 255, (180, 160), dtype=np.uint8)   # шумит
        still.feed(f)
    labels = still.result().labels
    assert (labels[:, :150] == SCREEN).sum() == 0, \
        "однородная половина кадра объявлена экранным слоем"


def test_arbiter_names_the_signal_it_used() -> None:
    """Арбитр обязан сказать, какой признак решил какие пиксели.

    Ответы трёх признаков обоснованы по-разному, и складывать их в одну величину
    без происхождения нельзя — по той же причине, по которой у убеждения есть
    происхождение.
    """
    from harness.corpus.domains import make_domain
    from harness.vision.selfworld import (COUPLING, MERGED, STILLNESS, STRENGTH,
                                          LayerArbiter, SCREEN)

    p = _profile()
    got = {}
    for name in ("game", "video"):
        d = make_domain(name, p, seed=7)
        arb = LayerArbiter(p)
        for i in range(60):
            act = Action.mouse(40, 20, duration_ms=33) if i % 2 else None
            arb.feed(d.step(act, with_audio=False).frame)
        v = arb.result()
        got[name] = v.signal
        assert v.reason, "признак выбран, а причина не названа"
        assert v.summary()["signal"] == v.signal
        # Происхождение согласовано с ответом: решённый пиксель имеет признак,
        # нерешённый — не имеет.
        assert v.provenance is not None
        from harness.vision.selfworld import UNDECIDED as U
        assert ((v.labels != U) == (v.provenance != 0)).all()
        decided = sum(int(v.by_signal(sig).sum()) for sig in STRENGTH)
        assert decided == int((v.labels != U).sum())

    # В игре камера то движется, то стоит — работает третий признак, и он же
    # объединяется с остальными.
    assert got["game"] in (COUPLING, MERGED)
    # В видео глобального сдвига нет вовсе: остаётся только неподвижность.
    assert got["video"] == STILLNESS


def test_third_signal_finds_animated_chrome_that_the_other_two_miss() -> None:
    """Ползущее заполнение полосы не находят ни параллакс, ни неподвижность.

    Оно не смещается вместе с миром и при этом не совпадает с собой. Третий признак
    берёт его тем, что оно меняется **и при стоящей камере**.
    """
    from harness.corpus.domains import make_domain
    from harness.vision.selfworld import COUPLING, LayerArbiter, SCREEN

    p = _profile()
    d = make_domain("game", p, seed=3)
    arb = LayerArbiter(p)
    rng = np.random.default_rng(1)
    for i in range(140):
        act = None
        if i % 3 != 0:
            dx, dy = int(rng.integers(-40, 41)), int(rng.integers(-26, 27))
            if dx or dy:
                act = Action.mouse(dx, dy, duration_ms=33)
        arb.feed(d.step(act, with_audio=False).frame)
    v = arb.result()
    anim = d.animated_mask()
    assert anim.any(), "в этом сиде нет анимированного обрамления"

    def recall(labels: np.ndarray) -> float:
        return float(((labels == SCREEN) & anim).sum() / anim.sum())

    # Объединение обязано быть лучше любого признака в одиночку. Именно объединение,
    # а не третий признак сам по себе: рамка анимированной панели неподвижна, её
    # берёт параллакс, а ползущее заполнение внутри — третий признак.
    assert recall(v.labels) > recall(v.parallax.labels)
    assert recall(v.labels) > recall(v.stillness.labels)
    assert recall(v.labels) > recall(v.coupling.labels)

    # И третий признак действительно приносит своё: пиксели анимированного
    # обрамления, которых нет ни у одного из первых двух.
    own = (v.by_signal(COUPLING) & anim & (v.labels == SCREEN)
           & (v.parallax.labels != SCREEN) & (v.stillness.labels != SCREEN))
    assert own.sum() > 0, "третий признак не добавил ни одного пикселя"

    # Ложных срабатываний по миру он при этом не добавляет.
    world = ~d.screen_mask()
    assert ((v.coupling.labels == SCREEN) & world).sum() == 0


def test_third_signal_needs_both_motion_and_stillness_in_the_record() -> None:
    """Связывать изменения с движением можно только там, где было и то и другое."""
    from harness.corpus.domains import make_domain
    from harness.vision.selfworld import MotionCouplingSeparator

    p = _profile()
    d = make_domain("game", p, seed=3)
    sep = MotionCouplingSeparator(p)
    for _ in range(40):                      # камера движется каждый кадр
        sep.feed(d.step(Action.mouse(40, 20, duration_ms=33), with_audio=False).frame)
    r = sep.result()
    assert r.still_frames == 0 and not r.applicable
    assert r.decided_fraction == 0.0, "признак решил что-то без остановок в записи"


def test_third_signal_ignores_pixels_that_could_not_show_motion() -> None:
    """«Не изменился» говорит что-то только про пиксель, который мог измениться.

    Пиксель в белом промежутке между строками текста при прокрутке остаётся белым.
    Без этой проверки признак записывал такие пиксели в экранный слой: по документу
    1004 ложных пикселя и точность 0.84 вместо 0.94.
    """
    from harness.corpus.domains import make_domain
    from harness.vision.selfworld import MotionCouplingSeparator, SCREEN

    p = _profile()
    d = make_domain("document", p, seed=3)
    sep = MotionCouplingSeparator(p)
    rng = np.random.default_rng(2)
    for i in range(140):
        act = None
        if i % 3 != 0:
            dy = int(rng.integers(-26, 27))
            if dy:
                act = Action.mouse(0, dy, duration_ms=33)
        sep.feed(d.step(act, with_audio=False).frame)
    r = sep.result()
    page = ~d.screen_mask()
    false_screen = int(((r.labels == SCREEN) & page).sum())
    assert false_screen < 400, f"страница попала в экранный слой: {false_screen} px"


def test_disagreement_between_signals_is_counted_not_averaged() -> None:
    """Расхождение признаков значит, что один врёт. Его надо видеть."""
    from harness.corpus.domains import make_domain
    from harness.vision.selfworld import LayerArbiter

    p = _profile()
    d = make_domain("game", p, seed=3)
    arb = LayerArbiter(p)
    rng = np.random.default_rng(3)
    for i in range(120):
        act = Action.mouse(int(rng.integers(-40, 41)), 0, duration_ms=33) \
            if i % 3 else None
        arb.feed(d.step(act, with_audio=False).frame)
    v = arb.result()
    assert v.disagreements >= 0
    assert 0.0 <= v.disagreement_fraction < 0.1, (
        f"признаки расходятся на {v.disagreement_fraction:.1%} — один из них врёт")
    assert v.summary()["disagreements"] == v.disagreements


def test_arbiter_respects_the_signal_fixed_in_the_profile() -> None:
    """`layer_signal` — структурный переключатель, а не подсказка."""
    from harness.corpus.domains import make_domain
    from harness.vision.selfworld import STILLNESS, LayerArbiter

    p = _profile(layer_signal="stillness")
    assert "layer_signal" in p.structural, "выбор признака должен форкать журнал"
    d = make_domain("game", p, seed=7)
    arb = LayerArbiter(p)
    for i in range(40):
        arb.feed(d.step(Action.mouse(40, 20, duration_ms=33), with_audio=False).frame)
    assert arb.result().signal == STILLNESS


def test_stillness_thresholds_cannot_overlap() -> None:
    """Пороги «неподвижен» и «движется» не должны пересекаться: иначе пиксель
    попал бы в оба состояния сразу."""
    from harness.vision.selfworld import StillnessSeparator

    with pytest.raises(ValueError, match="пороги перекрыты"):
        StillnessSeparator(_profile(stillness_static_rate=0.5,
                                   stillness_moving_rate=0.2))


def test_frame_energy_does_not_saturate_where_frame_change_does() -> None:
    """Доля изменившихся пикселей насыщается там, где картинка меняется целиком.

    Это причина, по которой у последствия действия своя мера. Проверяется на том же
    домене, на котором это и обнаружилось: в проигрывателе содержимое идёт само, доля
    изменившихся пикселей уже около 0.64 на холостом ходу, и перемотка добавляет к
    ней меньше, чем разброс самого фона.
    """
    from harness.corpus.domains import make_domain
    from harness.vision.selfworld import frame_change, frame_energy

    d = make_domain("video", _profile(), seed=9)
    live = [o for o, e in d._effects.items() if e is not None]
    idle_change, idle_energy = [], []
    prev = d.step(None, with_audio=False).frame
    for _ in range(20):
        cur = d.step(None, with_audio=False).frame
        idle_change.append(frame_change(prev, cur))
        idle_energy.append(frame_energy(prev, cur))
        prev = cur

    def sigmas(xs: list[float], value: float) -> float:
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / len(xs)
        return abs(value - mean) / max(var ** 0.5, 1e-9)

    # Сравнивать надо по самому слабому последствию, а не по самому яркому: находятся
    # ли все живые выходы, решает именно оно. По самому яркому обе меры хороши.
    worst_change = worst_energy = float("inf")
    for out in live:
        before = d.step(None, with_audio=False).frame
        after = d.step(Action.key(out, 100), with_audio=False).frame
        worst_change = min(worst_change, sigmas(idle_change, frame_change(before, after)))
        worst_energy = min(worst_energy, sigmas(idle_energy, frame_energy(before, after)))

    assert sum(idle_change) / len(idle_change) > 0.5, \
        "доля изменившихся пикселей не насыщена — тест проверяет не то"
    assert worst_energy > worst_change, (
        f"мера по величине не лучше меры по доле на слабом последствии: "
        f"{worst_energy:.1f}σ против {worst_change:.1f}σ")


# --- карта тела: четыре состояния -------------------------------------------


def test_one_coincidence_is_not_a_discovery() -> None:
    """Выход, ответивший один раз из сорока, не считается отвечающим."""
    from harness.model.rebuild import OutputFacts

    f = OutputFacts("OUT_0001", delivered=40, responded=1)
    assert f.state == "silent"
    f2 = OutputFacts("OUT_0002", delivered=40, responded=20)
    assert f2.state == "live"


def test_silence_tolerates_the_thresholds_own_false_alarms() -> None:
    """«Молчит» сравнивается не с нулём, а с тем, сколько порог врёт сам.

    Требование «ни одного ответа» означало бы, что одно случайное совпадение
    навсегда оставляет выход неопределённым.
    """
    from harness.model.rebuild import OutputFacts

    f = OutputFacts("OUT_0003", delivered=40, responded=2, background_rate=0.05)
    assert f.state == "silent", f.excess_sigmas
    g = OutputFacts("OUT_0004", delivered=40, responded=12, background_rate=0.05)
    assert g.state == "live"


def test_untried_and_unclear_are_different_states() -> None:
    """«Не пробовал» и «пробовал мало» — разные вещи, и оба не «молчит»."""
    from harness.model.rebuild import OutputFacts

    assert OutputFacts("OUT_0005").state == "untried"
    assert OutputFacts("OUT_0006", delivered=1, responded=1,
                       silent_min_deliveries=3).state == "unclear"


def test_background_rate_is_measured_not_assumed() -> None:
    """Доля ложных срабатываний порога измеряется на холостом ходу."""
    from harness.behaviour.babbling import Babbler, run_babbling
    from harness.corpus.domains import make_domain

    p = _profile(babble_repeats=2)
    d = make_domain("video", p, seed=8)
    b = Babbler(p, d.outputs, rng_seed=8)
    rep = run_babbling(d, b, steps=200, clocks=Clocks())
    assert rep["background_rate"] > 0.0, \
        "в видео порог обязан иногда срабатывать сам: картинка идёт"
    assert round(b.body.background_rate, 4) == rep["background_rate"]


# --- кросс-доменный замер ---------------------------------------------------


def test_cross_domain_benchmark_passes_everywhere() -> None:
    """Главный тест доменной независимости: одни модули по четырём мирам."""
    from harness.benchmark import run_all

    rep = run_all(seed=3, frames=120, babble_steps=600)
    assert len(rep.results) == 5
    assert not rep.as_dict()["unexpected_failures"], (
        "домен провалился без объявленной причины: "
        f"{rep.as_dict()['unexpected_failures']}")
    for r in rep.results:
        if rep.passed(r):
            continue
        # Провал допустим только объявленный, и причина обязана быть внятной.
        why = rep.expected_failure(r)
        assert len(why) > 80, f"{r.domain}: причина провала слишком короткая"
    assert "домен" in rep.table()


def test_benchmark_reports_wrong_answers_not_just_missing_ones() -> None:
    """Уверенно наоборот — единственная категория, которая всё портит."""
    from harness.benchmark import DomainResult, _body_ok

    base = dict(domain="x", frames=1, signal="parallax", signal_reason="",
                iou=0.9, recall=0.9, precision=0.9, recall_static=0.9,
                recall_animated=0.5, undecided=0.0, parallax_iou=0.9,
                parallax_decided=0.5, parallax_frames=9, stillness_iou=None,
                stillness_decided=0.0, stillness_frames=0,
                coupling_iou=None, coupling_decided=0.0, coupling_moving_frames=0,
                coupling_still_frames=0, disagreements=0,
                disagreement_fraction=0.0, decided_by={}, error_mean=0.0,
                error_spikes=0, shift_vs_copy=1.0, places=1, edges=0,
                mask_unstable=False, live_found=1, live_true=1, silent_found=1,
                silent_true=1, ambiguous=0, faint=0, background_rate=0.0,
                t_content_moves=False)
    ok, why = _body_ok(DomainResult(wrong_body=0, **base))
    assert ok and "полностью" in why
    bad, why2 = _body_ok(DomainResult(wrong_body=1, **base))
    assert not bad and "наоборот" in why2


def test_benchmark_does_not_score_where_truth_changed() -> None:
    """Пиксель, у которого истина за прогон менялась, из подсчёта выпадает.

    Иначе метод наказан за правильный ответ: пиксель, бывший фоном большую часть
    записи, честно получает голос «неподвижный», а истина последнего кадра называет
    его окном.
    """
    from harness.benchmark import bench_domain

    r = bench_domain("desktop", seed=3, frames=100, babble_steps=200)
    assert r.mask_unstable, "на рабочем столе окна не двигались — тест не про то"
    assert r.precision is not None and r.precision > 0.9


# --- бесплатные сервисы описания --------------------------------------------


def test_no_key_means_loud_refusal_not_a_stub() -> None:
    """Молчаливая заглушка здесь — худшее, что может быть: эксперимент испортится
    незаметно. Без ключа описатель обязан отказаться и назвать переменную."""
    from harness.capture.base import BackendUnavailable
    from harness.perception import describers as mod

    d = mod.make("groq")
    monkey = os.environ.pop("GROQ_API_KEY", None)
    try:
        ok, why = d.probe()
        assert not ok and "GROQ_API_KEY" in why
        with pytest.raises(BackendUnavailable, match="GROQ_API_KEY"):
            d.require()
    finally:
        if monkey is not None:
            os.environ["GROQ_API_KEY"] = monkey


def test_available_returns_empty_when_nothing_is_configured() -> None:
    """Пустая цепочка — честный ответ, а не повод подсунуть локальную выдумку."""
    from harness.capture.base import BackendUnavailable
    from harness.perception import describers as mod

    saved = {k: os.environ.pop(k, None) for k in ("GROQ_API_KEY", "MISTRAL_API_KEY")}
    try:
        chain = mod.available(names=["groq", "mistral"])
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    assert chain.describers == []
    with pytest.raises(BackendUnavailable, match="ни одного описателя"):
        chain.describe(np.zeros((8, 8), dtype=np.uint8), "что?")


def test_rate_limiter_counts_minute_and_day_separately() -> None:
    """Бесплатные тарифы ограничивают и в минуту, и в сутки. Считать надо оба."""
    from harness.perception.describers import RateLimiter

    clock = [1000.0]
    lim = RateLimiter(rpm=2, rpd=3, now=lambda: clock[0])
    assert lim.allow() and lim.allow()
    assert not lim.allow(), "третий запрос в ту же минуту прошёл"
    clock[0] += 61
    assert lim.allow()
    clock[0] += 61
    assert not lim.allow(), "суточный предел не сработал"
    assert lim.refused == 2


def test_answer_cache_is_keyed_on_bitwise_identity() -> None:
    """Приблизительный кэш выдал бы ответ про другой кадр, и поймать это почти
    невозможно."""
    from harness.perception.describers import AnswerCache
    from harness.perception.imagecodec import frame_fingerprint

    a = np.zeros((8, 8), dtype=np.uint8)
    b = a.copy()
    b[0, 0] = 1
    assert frame_fingerprint(a) == frame_fingerprint(a.copy())
    assert frame_fingerprint(a) != frame_fingerprint(b)

    cache = AnswerCache(size=2)
    cache.put(frame_fingerprint(a), "ответ-а")
    assert cache.get(frame_fingerprint(a.copy())) == "ответ-а"
    assert cache.get(frame_fingerprint(b)) is None
    assert cache.state()["hits"] == 1


def test_ask_gate_saves_quota_on_repeated_frames() -> None:
    """Квота — главный дефицит бесплатного тарифа, и экономия начинается здесь."""
    from harness.perception.describers import AskGate

    gate = AskGate(min_novelty=0.02, min_gap_cycles=4)
    assert gate.should_ask(None, 1), "на первом кадре спросить надо"
    assert not gate.should_ask(0.001, 20), "мир не удивил, а вопрос ушёл"
    assert gate.should_ask(0.5, 20), "мир удивил, а вопрос не ушёл"
    assert not gate.should_ask(0.9, 21), "минимальный промежуток не соблюдён"
    assert gate.state()["asked"] == 2 and gate.state()["skipped_quiet"] == 1


def test_downscale_keeps_small_marks_readable() -> None:
    """Уменьшение усреднением, а не выборкой: выборка теряет мелкие надписи."""
    from harness.perception.imagecodec import downscale

    img = np.zeros((64, 64), dtype=np.uint8)
    img[30:32, :] = 255                        # тонкая полоса «надписи»
    small = downscale(img, 16)
    assert small.shape == (16, 16)
    assert small.max() > 40, "полоса исчезла при уменьшении"


def test_png_is_lossless_and_readable() -> None:
    """PNG, а не JPEG: на побитовой одинаковости держится и разделение слоёв, и кэш."""
    import zlib

    from harness.perception.imagecodec import png_bytes

    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, (12, 20), dtype=np.uint8)
    blob = png_bytes(img)
    assert blob.startswith(b"\x89PNG\r\n\x1a\n")
    idat = blob.split(b"IDAT")[1][:-12]
    raw = zlib.decompress(idat)
    rows = [raw[i * 21:(i + 1) * 21] for i in range(12)]
    assert all(r[0] == 0 for r in rows), "фильтр не нулевой"
    back = np.array([list(r[1:]) for r in rows], dtype=np.uint8)
    assert np.array_equal(back, img)


def test_prompt_forces_the_answer_format_the_firewall_can_check() -> None:
    """Файрвол проверяет форму ответа. Значит форма должна быть навязана промптом,
    а не оставлена на усмотрение модели."""
    from harness.perception import describers as mod
    from harness.perception.firewall import QUESTION

    from harness.perception.firewall import SIZES, ZONES

    assert "зона" in mod.FORMAT_RULES
    assert all(z in mod.FORMAT_RULES for z in ZONES)
    assert all(sz in mod.FORMAT_RULES for sz in SIZES)
    assert "не предлагай действий" in mod.FORMAT_RULES

    d = mod.make("groq")
    full = d.prompt(QUESTION)
    assert full.startswith(QUESTION) and mod.FORMAT_RULES in full
    with pytest.raises(ValueError, match="не тот вопрос"):
        d.prompt("что мне делать?")


def test_local_first_ordering_is_not_an_accident() -> None:
    """Локальные сервисы идут первыми: у них нет ни квоты, ни зависимости от сети."""
    from harness.perception.describers import PROVIDERS

    local = [i for i, p in enumerate(PROVIDERS) if p.key_env is None]
    remote = [i for i, p in enumerate(PROVIDERS) if p.key_env is not None]
    assert local and remote
    assert max(local) < min(remote), "сетевой сервис оказался раньше локального"
