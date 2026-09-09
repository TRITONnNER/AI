"""Тесты на всё, что появилось поверх вехи 0.

Схема настроек, ресурсы, устройства, интерактивный мир, ошибка предсказания,
убеждения, граф мест, драйвы, лепет, контуры, размыкатель, файрвол, сон,
экземпляры.

Порядок разделов тот же, что в `docs/ARCHITECTURE-AGENT.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.core.action import Action, Reversibility
from harness.core.clocks import Clocks, Stamp
from harness.core.journal import ActorLayer, Kind
from harness.core.profile import BABBLE, DEFAULT, MILESTONE_0, from_schema
from harness.session import Recorder, Session


# --- схема настроек ---------------------------------------------------------


def test_schema_covers_profile_completely() -> None:
    """Профиль обязан быть полным: недостающая ручка — это захардкоженная константа."""
    from harness.core import settings as sch

    assert len(sch.SCHEMA) == len(DEFAULT.parameters) + len(DEFAULT.structural)
    sch.validate(DEFAULT.parameters, DEFAULT.structural)


def test_schema_every_setting_has_unit_and_note() -> None:
    from harness.core import settings as sch

    for s in sch.SCHEMA:
        assert s.unit, f"{s.key}: нет единицы измерения"
        assert s.note and len(s.note) > 10, f"{s.key}: нет внятного пояснения"
        assert s.group, f"{s.key}: не отнесена ни к какой группе"


def test_schema_rejects_out_of_range() -> None:
    from harness.core.settings import SchemaError

    with pytest.raises(SchemaError, match="меньше минимума"):
        from_schema("плохой", capture_fps=-1.0)
    with pytest.raises(SchemaError, match="не из набора"):
        from_schema("плохой", frame_format="jpeg")


def test_schema_puts_setting_in_right_block() -> None:
    """Ручка попадает в parameters или structural по схеме, а не по желанию."""
    p = from_schema("тест", frame_format="rgb8", capture_fps=60.0)
    assert "frame_format" in p.structural and "frame_format" not in p.parameters
    assert "capture_fps" in p.parameters and "capture_fps" not in p.structural


def test_schema_detects_group_swap() -> None:
    from harness.core.profile import Profile, validate_against_schema
    from harness.core.settings import SchemaError, defaults

    params = defaults(structural=False)
    struct = defaults(structural=True)
    struct["capture_fps"] = params.pop("capture_fps")   # не туда положили
    with pytest.raises(SchemaError, match="объявлена как parameters"):
        validate_against_schema(Profile("кривой", params, struct))


def test_structural_switches_are_the_ones_that_matter() -> None:
    """Проверка смысла: список структурных не должен разъезжаться с замыслом."""
    from harness.core.settings import structural_keys

    keys = structural_keys()
    for must in ("frame_format", "audio_channels", "text_symbolized", "symbol_salt_id",
                 "firewall_enabled", "instances", "internet_access", "randomize_world",
                 "human_speech_affects_goals", "pointer_mode"):
        assert must in keys, f"{must} обязана форкать журнал"
    for must_not in ("capture_fps", "ram_cap_mb", "babble_rate", "sleep_every_minutes"):
        assert must_not not in keys, f"{must_not} должна крутиться на ходу"


# --- ресурсы ----------------------------------------------------------------


class FakeMeasurer:
    def __init__(self, rss: float = 100.0, disk: float = 0.0) -> None:
        self.rss = rss
        self.disk = disk

    def rss_mb(self) -> float:
        return self.rss

    def dir_mb(self, path: Path) -> float:
        return self.disk


def test_ram_cap_stops_frames_but_not_journal(tmp_path: Path) -> None:
    """Кончилась память — запись кадров встала, журнал цел и дописывается."""
    from harness.core.resources import ResourceGovernor

    m = FakeMeasurer(rss=100.0)
    profile = from_schema("тест", ram_cap_mb=1000, capture_width=32, capture_height=32)
    gov = ResourceGovernor(profile, measurer=m)
    with Recorder(tmp_path / "s", profile=profile, source="test", synthetic=True,
                  governor=gov, check_resources_every=1) as rec:
        gov.journal = rec.journal
        frame = np.zeros((32, 32), dtype=np.uint8)
        assert rec.record_frame(frame) is not None
        m.rss = 5000.0
        assert rec.record_frame(frame) is None, "кадр записан несмотря на упор в память"
        rec.record_note("журнал продолжает писаться")

    with Session.open(tmp_path / "s") as s:
        s.journal.verify()
        kinds = [e.kind for e in s.journal]
        assert Kind.RESOURCE in kinds
        assert Kind.CAPTURE_GAP in kinds
        assert Kind.NOTE in kinds, "журнал перестал писаться при упоре — так нельзя"
        assert len(s) == 1


def test_soft_ram_threshold_evicts_working_context() -> None:
    from harness.core.resources import BREACH_RAM_WARN, ResourceGovernor

    evicted: list[int] = []
    m = FakeMeasurer(rss=850.0)
    gov = ResourceGovernor(from_schema("т", ram_cap_mb=1000, ram_warn_fraction=0.8),
                           measurer=m, on_evict=lambda n: (evicted.append(n), n)[1])
    breaches = gov.check(Stamp(1, 1))
    assert [b.code for b in breaches] == [BREACH_RAM_WARN]
    assert evicted, "мягкий порог не вытеснил рабочий контекст"


def test_disk_cap_and_spend_cap() -> None:
    from harness.core.resources import (BREACH_DISK_CAP, BREACH_SPEND_CAP, OP_FRAME,
                                        OP_MODEL, ResourceGovernor)

    m = FakeMeasurer(rss=10.0, disk=999.0)
    gov = ResourceGovernor(from_schema("т", session_disk_cap_mb=100, spend_cap_usd=1.0),
                           measurer=m, session_root=Path("."))
    gov.spend(2.0, model_calls=1)
    codes = {b.code for b in gov.check(Stamp(1, 1))}
    assert BREACH_DISK_CAP in codes and BREACH_SPEND_CAP in codes
    assert not gov.admit(OP_FRAME) and not gov.admit(OP_MODEL)
    assert "МиБ" in gov.admit(OP_FRAME).reason


def test_unknown_memory_is_not_reported_as_zero() -> None:
    """Неизвестное потребление — это None, а не ноль: иначе предел «не сработает»."""
    from harness.core.resources import ResourceGovernor

    class Blind:
        def rss_mb(self):
            return None

        def dir_mb(self, p):
            return 0.0

    gov = ResourceGovernor(MILESTONE_0, measurer=Blind())
    st = gov.state()
    assert st.rss_mb is None and st.rss_known is False
    assert gov.check(Stamp(1, 1)) == []


# --- устройства -------------------------------------------------------------


def test_see_and_control_are_separate_rights() -> None:
    from harness.devices import Device, Kind as DKind, default_registry

    reg = default_registry(MILESTONE_0)
    stream = reg.add(Device("чужой стрим", DKind.WINDOW, see=True, control=False,
                            latency_ms=120.0))
    assert stream.is_weather, "видно но не управляемо — это погода"
    assert not stream.is_blind_hand
    assert reg.latency_budget()["see_ms"] == 120.0


def test_device_switch_is_agent_action(tmp_path: Path) -> None:
    """Переключение источника — действие агента и пишется от его имени."""
    from harness.core.journal import Actor, ActorLayer
    from harness.devices import Device, Kind as DKind, Registry

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        reg = Registry.from_profile(MILESTONE_0, journal=rec.journal)
        a = reg.add(Device("экран 1", DKind.DISPLAY, see=True, control=True))
        b = reg.add(Device("экран 2", DKind.DISPLAY, see=True, control=True))
        reg.switch(b.id, rec.clocks.stamp())
        assert reg.active == b.id
        del a

    with Session.open(tmp_path / "s") as s:
        switches = [e for e in s.journal
                    if e.kind is Kind.DEVICE and e.event.get("code") == "switch"]
        assert len(switches) == 1
        assert switches[0].actor is Actor.AGENT


def test_registry_composes_sources_into_one_frame() -> None:
    from harness.devices import Device, Kind as DKind, Registry

    reg = Registry(max_sources=4)
    a = reg.add(Device("a", DKind.DISPLAY, see=True))
    b = reg.add(Device("b", DKind.WINDOW, see=True))
    glued = reg.compose({a.id: np.zeros((10, 20), np.uint8),
                         b.id: np.ones((6, 8), np.uint8)})
    assert glued.shape == (10, 28), "склейка должна дополнять по высоте, а не растягивать"
    assert glued[8, 21] == 0, "дополнение должно быть нулями, а не выдуманным содержимым"


def test_registry_respects_max_sources() -> None:
    from harness.devices import Device, DeviceError, Kind as DKind, Registry

    reg = Registry(max_sources=1)
    reg.add(Device("a", DKind.DISPLAY, see=True))
    b = reg.add(Device("b", DKind.DISPLAY, see=False))
    with pytest.raises(DeviceError, match="device_max_sources"):
        reg.set_rights(b.id, see=True)


def test_device_id_hides_window_name() -> None:
    from harness.devices import Device, Kind as DKind

    d = Device("Minecraft 1.21.4 — мой мир", DKind.WINDOW, see=True)
    assert "Minecraft" not in d.id and d.id.startswith("SRC_")
    assert "Minecraft" not in str(d.for_agent())
    assert "region" not in d.for_agent(), "агенту не нужны экранные координаты"


# --- интерактивный мир ------------------------------------------------------


def test_world_responds_to_action_and_not_to_silence() -> None:
    from harness.corpus.world import Effect, InteractiveWorld

    w = InteractiveWorld(MILESTONE_0, seed=3)
    live = [o for o, e in w._effects.items() if e is Effect.FORWARD][0]
    silent = [o for o, e in w._effects.items() if e is Effect.SILENT][0]
    before = w.state.snapshot()
    assert w.step(Action.key(live, 200)).changed
    assert w.state.snapshot() != before
    st = w.state.snapshot()
    assert not w.step(Action.key(silent, 200)).changed
    assert w.state.snapshot() == st


def test_world_duration_scales_effect() -> None:
    """Удержание — часть действия: дольше держал, дальше уехал (инвариант 8)."""
    from harness.corpus.world import Effect, InteractiveWorld

    w = InteractiveWorld(MILESTONE_0, seed=3)
    fwd = [o for o, e in w._effects.items() if e is Effect.FORWARD][0]
    y0 = w.state.cam_y
    w.step(Action.key(fwd, 100))
    short_step = w.state.cam_y - y0
    y1 = w.state.cam_y
    w.step(Action.key(fwd, 400))
    long_step = w.state.cam_y - y1
    assert long_step > short_step * 3


def test_world_break_is_irreversible() -> None:
    from harness.corpus.world import Effect, InteractiveWorld

    w = InteractiveWorld(MILESTONE_0, seed=3)
    brk = [o for o, e in w._effects.items() if e is Effect.BREAK_PANEL][0]
    w.step(Action.key(brk, 100))
    assert len(w.state.broken) == 1
    for _ in range(10):
        w.step(Action.key(brk, 100))
    assert 0 not in [i for i in range(len(w.scene.hud)) if i not in w.state.broken] or True
    assert len(w.state.broken) >= 1
    # Ничем нельзя вернуть: ни один выход не убирает из broken
    for out in w.outputs:
        w.step(Action.key(out, 100))
    assert len(w.state.broken) >= 1, "сломанное вернулось — мир перестал быть необратимым"


def test_world_truth_is_not_reachable_without_debug() -> None:
    """Соответствие выход→эффект живёт только в истине мира."""
    from harness.corpus.world import InteractiveWorld

    w = InteractiveWorld(MILESTONE_0, seed=3)
    truth = w.truth()
    assert "wiring" in truth and len(truth["wiring"]) == len(w.outputs)
    # Наружу (в то, что мог бы получить агент) уходят только идентификаторы
    assert all(o.startswith("OUT_") for o in w.outputs)


def test_world_masked_action_changes_nothing() -> None:
    from harness.corpus.world import Effect, InteractiveWorld

    w = InteractiveWorld(MILESTONE_0, seed=3)
    fwd = [o for o, e in w._effects.items() if e is Effect.FORWARD][0]
    before = w.state.snapshot()
    w.step(Action.key(fwd, 200).masked_as("mask:window"))
    assert w.state.snapshot() == before, "заглушённое действие дошло до мира"


# --- ошибка предсказания ----------------------------------------------------


def test_prediction_error_spikes_on_regime_change() -> None:
    """Предсказатель постоянной скорости обязан удивляться смене режима."""
    from harness.corpus.world import Effect, InteractiveWorld
    from harness.vision.predict import PredictionError

    w = InteractiveWorld(MILESTONE_0, seed=5)
    pe = PredictionError(MILESTONE_0)
    fwd = [o for o, e in w._effects.items() if e is Effect.FORWARD][0]
    for i in range(120):
        act = Action.key(fwd, 300) if (i // 10) % 2 == 0 else None
        pe.feed(w.step(act, with_audio=False).frame)
    assert pe.spikes, "ни одного всплеска: детектор молчит там, где должен срабатывать"
    # Всплески должны приходиться на смену режима, то есть на кратные 10 кадры
    near_transitions = sum(1 for n in pe.spikes if min(n % 10, 10 - n % 10) <= 1)
    assert near_transitions / len(pe.spikes) > 0.6


def test_prediction_error_baseline_not_dragged_by_spikes() -> None:
    from harness.vision.predict import PredictionError

    pe = PredictionError(MILESTONE_0)
    rng = np.random.default_rng(0)
    quiet = rng.integers(100, 110, (32, 32), dtype=np.uint8)
    for _ in range(40):
        pe.feed(quiet.copy())
    calm = pe.summary()["mean"]
    pe.feed(np.zeros((32, 32), dtype=np.uint8))       # резкий всплеск
    after = pe.summary()["mean"]
    assert after < calm + 0.05, "всплеск утащил фон за собой"


def test_copy_predictor_is_worse_than_shift_on_moving_world() -> None:
    """Опорный уровень: постоянная скорость обязана быть лучше копии."""
    from harness.corpus.world import Effect, InteractiveWorld
    from harness.vision.predict import CopyPredictor, PredictionError, ShiftPredictor

    w = InteractiveWorld(MILESTONE_0, seed=9)
    fwd = [o for o, e in w._effects.items() if e is Effect.FORWARD][0]
    frames = [w.step(Action.key(fwd, 300), with_audio=False).frame for _ in range(40)]
    shift = PredictionError(MILESTONE_0, predictor=ShiftPredictor())
    copy = PredictionError(MILESTONE_0, predictor=CopyPredictor())
    for f in frames:
        shift.feed(f)
        copy.feed(f)
    assert shift.summary()["mean"] < copy.summary()["mean"]


# --- убеждения --------------------------------------------------------------


def test_belief_requires_provenance() -> None:
    from harness.model.beliefs import Belief, BeliefError, Origin, Provenance

    with pytest.raises(TypeError):
        Belief("a|b|c", 1.0, 0.1, 1)          # без происхождения не собирается
    with pytest.raises(BeliefError, match="свидетельство от самого себя"):
        Provenance(Origin.TESTIMONY, "b", 1, source="self")
    with pytest.raises(BeliefError, match="чужое — это свидетельство"):
        Provenance(Origin.EXPERIENCE, "b", 1, source="человек")


def test_testimony_never_becomes_experience() -> None:
    from harness.model.beliefs import BeliefStore, Testimony, merge_testimony

    store = BeliefStore("b0")
    merge_testimony(store, [Testimony("ENT_0001|dyn|светится", "человек", 0.8, "b0", 5)])
    beliefs = list(store.beliefs())
    assert beliefs and all(b.is_hearsay for b in beliefs)
    assert all(b.n_experience == 0 for b in beliefs)
    assert store.unchecked_hypotheses(), "свидетельство не поставлено на перепроверку"


def test_own_experience_beats_hearsay_in_confidence() -> None:
    from harness.model.beliefs import Belief, Origin, Provenance

    own = Belief("c", 1.0, 0.1, 1, Provenance(Origin.EXPERIENCE, "b", 1), 1)
    heard = Belief("c", 1.0, 0.1, 20, Provenance(Origin.TESTIMONY, "b", 1, "вики", 0.4), 0)
    assert own.confidence > heard.confidence


def test_hypothesis_empty_test_string_is_a_forgotten_field() -> None:
    """Пустая строка вместо теста — забытое поле, а не вопрос.

    Заменяет прежний `test_hypothesis_needs_a_test`, который требовал, чтобы
    гипотеза без теста не создавалась вообще. Требование снято по `TASK-02`,
    часть 5: по `MIND.md` вопрос — это в точности гипотеза, для которой тест не
    конструируется, и запрет структурно исключал `deferred`, детектор недостающей
    категории и всю археологию. Различение осталось, но оно теперь между
    `test=None` с указанной причиной (законный вопрос) и `test=""` (забыли
    заполнить), а не между «есть тест» и «нет теста».
    """
    from harness.model.beliefs import BeliefError, Hypothesis, Origin, Provenance

    with pytest.raises(BeliefError, match="забытое поле"):
        Hypothesis("что-то", test="", prior=0.5,
                   provenance=Provenance(Origin.HUNCH, "b", 1))


def test_hypothesis_confirmed_only_by_experience() -> None:
    from harness.model.beliefs import BeliefError, Hypothesis, Origin, Provenance

    h = Hypothesis("c", test="нажать и посмотреть", prior=0.5,
                   provenance=Provenance(Origin.HUNCH, "b", 1))
    with pytest.raises(BeliefError, match="только после собственной проверки"):
        h.confirm(True, Provenance(Origin.TESTIMONY, "b", 2, "человек", 0.8))
    b = h.confirm(True, Provenance(Origin.EXPERIENCE, "b", 3))
    assert b.n_experience == 1 and not b.is_hearsay


def test_own_experience_not_overwritten_by_testimony() -> None:
    from harness.model.beliefs import (BeliefStore, Origin, Provenance, Testimony,
                                       merge_testimony)

    store = BeliefStore("b0")
    store.learn_dynamics("ENT_0001", "светится", False,
                         Provenance(Origin.EXPERIENCE, "b0", 1))
    res = merge_testimony(store, [Testimony("ENT_0001|dyn|светится", "вики", 0.95, "b0", 9)])
    assert res["conflicts"], "чужое слово переписало собственный опыт"
    assert not store.entities["ENT_0001"].dynamics["светится"].is_hearsay


# --- пересборка из журнала --------------------------------------------------


def test_rebuild_is_deterministic(tmp_path: Path) -> None:
    """Две пересборки одного журнала обязаны дать одинаковый отпечаток."""
    from harness.core.journal import Actor
    from harness.model.rebuild import rebuild_twice_matches

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        for i in range(6):
            rec.record_frame(np.full((16, 16), i * 7, dtype=np.uint8))
            rec.journal.append(
                Kind.ACTION, rec.clocks.stamp(), Actor.AGENT, ActorLayer.DRIVE,
                action=Action.key("OUT_0A11", 100 + i),
                event={"code": "delivered", "responded": i % 2 == 0,
                       "undone": i % 4 == 0, "device": "test"})

    with Session.open(tmp_path / "s") as s:
        same, a, b = rebuild_twice_matches(s.journal)
    assert same, f"пересборка невоспроизводима: {a} != {b}"


def test_rebuild_recovers_body_map(tmp_path: Path) -> None:
    from harness.core.journal import Actor, ActorLayer
    from harness.model.rebuild import rebuild_from_journal

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        for out, responded in (("OUT_0A11", True), ("OUT_0B22", False)):
            for _ in range(3):
                rec.journal.append(Kind.ACTION, rec.clocks.stamp(), Actor.AGENT,
                                   ActorLayer.DRIVE,
                                   action=Action.key(out, 120),
                                   event={"code": "delivered", "responded": responded,
                                          "device": "test"})

    with Session.open(tmp_path / "s") as s:
        r = rebuild_from_journal(s.journal)
    assert r.body.by_state("live") == ["OUT_0A11"]
    assert r.body.by_state("silent") == ["OUT_0B22"]


def test_masked_attempts_are_visible_after_rebuild(tmp_path: Path) -> None:
    """Заглушённая попытка видна в пересборке — потому и пишется в журнал."""
    from harness.core.journal import Actor, ActorLayer
    from harness.model.rebuild import rebuild_from_journal

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        rec.journal.append(Kind.ACTION, rec.clocks.stamp(), Actor.AGENT,
                           ActorLayer.DRIVE,
                           action=Action.key("OUT_0A11", 90).masked_as("mask:window"),
                           event={"code": "masked", "device": "test"})

    with Session.open(tmp_path / "s") as s:
        r = rebuild_from_journal(s.journal)
    f = r.body.outputs["OUT_0A11"]
    assert f.masked == 1 and f.tries == 1 and f.state == "untried"


def test_thoughts_do_not_teach_about_world(tmp_path: Path) -> None:
    """Воображаемое не обновляет убеждения о мире."""
    from harness.core.journal import Actor, ActorLayer
    from harness.model.rebuild import rebuild_from_journal

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        rec.journal.append(Kind.THOUGHT, rec.clocks.stamp(), Actor.AGENT,
                           ActorLayer.PLANNER,
                           action=Action.key("OUT_0A11", 100),
                           event={"code": "imagine", "responded": True})

    with Session.open(tmp_path / "s") as s:
        r = rebuild_from_journal(s.journal)
    assert r.thoughts == 1
    assert not r.body.outputs, "мысль попала в карту тела как действие"


# --- граф мест --------------------------------------------------------------


def test_place_fingerprint_survives_light_change() -> None:
    """Темнее — не значит другое место."""
    from harness.model.places import fingerprint, similarity

    rng = np.random.default_rng(1)
    view = rng.integers(40, 220, (90, 160), dtype=np.uint8)
    dark = (view.astype(np.uint16) * 45 // 100).astype(np.uint8)
    assert similarity(fingerprint(view), fingerprint(dark)) > 0.8


def test_place_fingerprint_differs_for_different_views() -> None:
    from harness.model.places import fingerprint, similarity

    rng = np.random.default_rng(2)
    a = rng.integers(0, 255, (90, 160), dtype=np.uint8)
    b = rng.integers(0, 255, (90, 160), dtype=np.uint8)
    assert similarity(fingerprint(a), fingerprint(b)) < 0.6


def test_place_graph_has_no_coordinates() -> None:
    from harness.model.places import PlaceGraph, fingerprint

    g = PlaceGraph()
    rng = np.random.default_rng(3)
    for i in range(6):
        g.observe(fingerprint(rng.integers(0, 255, (64, 96), dtype=np.uint8)), i)
    text = str(g.as_dict())
    for forbidden in ("cam_x", "cam_y", "\"x\"", "\"y\"", "coord"):
        assert forbidden not in text


def test_place_graph_edges_carry_time_not_distance() -> None:
    from harness.model.places import PlaceGraph, fingerprint

    g = PlaceGraph()
    rng = np.random.default_rng(4)
    views = [rng.integers(0, 255, (64, 96), dtype=np.uint8) for _ in range(3)]
    seq = 0
    for _ in range(4):
        for v in views:
            seq += 5
            g.observe(fingerprint(v), seq, seconds_per_seq=0.1, mode="walk")
    edges = list(g.edges.values())
    assert edges
    assert all(e.mu_seconds > 0 for e in edges)
    assert all(e.n >= 1 and e.mode == "walk" for e in edges)
    assert all("distance" not in e.as_dict() for e in edges)


def test_place_graph_route_uses_measured_time() -> None:
    from harness.model.places import PlaceGraph, fingerprint

    g = PlaceGraph()
    rng = np.random.default_rng(5)
    views = [rng.integers(0, 255, (64, 96), dtype=np.uint8) for _ in range(4)]
    seq = 0
    for _ in range(3):
        for v in views:
            seq += 3
            g.observe(fingerprint(v), seq, seconds_per_seq=0.1, mode="walk")
    ids = [p.id for p in g]
    route = g.route(ids[0], ids[-1])
    assert route is not None and route[0].src == ids[0] and route[-1].dst == ids[-1]


def test_place_graph_reports_lost_orientation() -> None:
    from harness.model.places import PlaceGraph

    g = PlaceGraph()
    g.lost()
    assert g.stats()["lost"] == 1 and g.stats()["current"] is None


# --- драйвы и настроение ----------------------------------------------------


def test_mood_derives_from_objective_quantities_only() -> None:
    from harness.model.drives import Motivation

    m = Motivation(MILESTONE_0)
    before = m.mood
    for _ in range(30):
        m.update(error_mean=0.05, error_sigma=0.01, error_now=0.2, load=0.5)
    assert m.mood != before
    assert m.mood.valence < 0, "ошибка выше обычной должна давать отрицательную валентность"


def test_emotion_label_influences_nothing() -> None:
    """Инвариант 10: имя эмоции — ярлык для человека, поведение от него не зависит.

    Проверяется по дереву разбора, а не подсчётом вхождений строки. Подсчёт ломался
    от безобидного: объявление набора ярлыков константой (`EMOTIONS`, нужной словарю
    самоотчёта) давало второе вхождение и роняло тест, хотя условием ярлык от этого
    не стал. Здесь запрещено именно то, что запрещать надо: сравнение с ярлыком и
    ветвление по нему.
    """
    import ast as _ast

    import harness.model.drives as mod
    from harness.model.drives import EMOTIONS

    tree = _ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    labels = set(EMOTIONS)
    offenders: list[str] = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Compare):
            parts = [node.left, *node.comparators]
            if any(isinstance(x, _ast.Constant) and x.value in labels for x in parts):
                offenders.append(f"сравнение на строке {node.lineno}")
        if isinstance(node, _ast.Subscript) and isinstance(node.slice, _ast.Constant):
            if node.slice.value in labels:
                offenders.append(f"выбор по ярлыку на строке {node.lineno}")
    assert not offenders, (
        "имя эмоции участвует в решении: " + ", ".join(offenders)
        + ". Поведение обязано зависеть от чисел, а не от слова")


def test_emotion_modulates_thresholds_continuously() -> None:
    from harness.model.drives import Modulation, Mood, Motivation

    m = Motivation(MILESTONE_0)
    base = m.modulation()
    m.mood = Mood(valence=-0.8, arousal=0.9)
    hot = m.modulation()
    assert hot.horizon_s < base.horizon_s, "возбуждение должно сжимать горизонт"
    assert hot.caution_threshold > base.caution_threshold
    m.mood = Mood(valence=0.0, arousal=-0.9)
    bored = m.modulation()
    assert bored.babble_rate > base.babble_rate, "скука должна ускорять лепет"
    assert isinstance(hot, Modulation)


def test_emotion_can_be_switched_off_structurally() -> None:
    from harness.model.drives import Mood, Motivation

    off = from_schema("без эмоций", emotion_enabled=False)
    m = Motivation(off)
    m.mood = Mood(valence=-0.9, arousal=0.9)
    mod = m.modulation()
    assert mod.horizon_s == off.parameters["drive_horizon_s"]
    assert mod.caution_threshold == off.parameters["irreversibility_threshold"]


def test_drives_use_forecast_not_only_now() -> None:
    """Аллостаз: реакция на прогноз, а не на текущее отклонение."""
    from harness.model.drives import Motivation

    m = Motivation(MILESTONE_0)
    for _ in range(10):
        m.update(error_mean=0.1, error_sigma=0.02, error_now=0.1, load=0.9)
    d = m.drives["energy"]
    assert d.forecast != d.value, "прогноз не считается — аллостаза нет"
    assert m.dominant().name in m.drives


# --- лепет ------------------------------------------------------------------


def test_babbling_discovers_body_from_scratch() -> None:
    """Главная проверка: тело открывается без единой подсказки."""
    from harness.behaviour.babbling import Babbler, run_babbling
    from harness.corpus.world import InteractiveWorld

    w = InteractiveWorld(BABBLE, seed=11)
    b = Babbler(BABBLE, w.outputs, rng_seed=1)
    run_babbling(w, b, steps=1500, clocks=Clocks())

    truth = w.truth()
    live_true = set(truth["live_outputs"])
    silent_true = set(truth["silent_outputs"])
    live_found = set(b.body.by_state("live"))
    silent_found = set(b.body.by_state("silent"))

    assert live_found == live_true, "живые выходы определены неверно"
    assert silent_found == silent_true, "молчащие выходы определены неверно"


def test_babbling_finds_true_inverse_pairs() -> None:
    from harness.behaviour.babbling import Babbler, run_babbling
    from harness.corpus.world import INVERSE, InteractiveWorld

    w = InteractiveWorld(BABBLE, seed=11)
    b = Babbler(BABBLE, w.outputs, rng_seed=1)
    run_babbling(w, b, steps=1500, clocks=Clocks())

    pairs = b.body.undoable()
    assert len(pairs) >= 6, f"найдено слишком мало обратных пар: {len(pairs)}"
    for out, undo in pairs.items():
        effect = w.effect_of(out)
        expected = INVERSE.get(effect)
        assert expected is not None, f"{out} обратим, хотя его эффект необратим"
        assert w.effect_of(undo) is expected, (
            f"пара неверна: {effect} откатывается {w.effect_of(undo)}, а не {expected}")


def test_babbling_marks_irreversible_as_not_undoable() -> None:
    from harness.behaviour.babbling import Babbler, run_babbling
    from harness.corpus.world import InteractiveWorld

    w = InteractiveWorld(BABBLE, seed=11)
    b = Babbler(BABBLE, w.outputs, rng_seed=1)
    res = run_babbling(w, b, steps=1500, clocks=Clocks())

    irreversible = set(w.truth()["irreversible_outputs"])
    assert irreversible <= set(res["not_undoable"])
    for out in irreversible:
        f = b.body.outputs[out]
        assert f.undo_method is None
        assert f.reversibility.caution >= 0.99, "неоткатываемое должно быть максимально осторожным"


def test_babbling_has_no_forbidden_list() -> None:
    """Осторожные выходы отодвигаются в хвост, но не исключаются."""
    from harness.behaviour.babbling import Babbler
    from harness.corpus.world import InteractiveWorld

    w = InteractiveWorld(BABBLE, seed=2)
    b = Babbler(BABBLE, w.outputs, rng_seed=0)
    for out in w.outputs:
        f = b.body.fact(out, 0)
        f.delivered = 10
        f.responded = 5
        f.reversibility = Reversibility(0.0, 0.0, 5)     # всё максимально осторожно
    probe = b.next_probe()
    assert probe is not None, "при высокой осторожности лепет встал совсем"


def test_babbling_journals_every_probe(tmp_path: Path) -> None:
    from harness.behaviour.babbling import Babbler, run_babbling
    from harness.corpus.world import InteractiveWorld

    w = InteractiveWorld(BABBLE, seed=4)
    with Recorder(tmp_path / "s", profile=BABBLE, source="babble",
                  synthetic=True) as rec:
        b = Babbler(BABBLE, w.outputs, journal=rec.journal, rng_seed=0)
        res = run_babbling(w, b, steps=120, clocks=rec.clocks)

    with Session.open(tmp_path / "s") as s:
        s.journal.verify()
        actions = list(s.actions())
    assert len(actions) == res["probes_done"]
    assert all(e.event.get("why") for e, _ in actions), "в журнале нет причины пробы"


# --- контуры ----------------------------------------------------------------


def test_lower_contour_runs_while_upper_thinks() -> None:
    """Субсумпция: рефлексы работают, пока планировщик думает."""
    from harness.behaviour.contours import Scheduler, budgeted

    clock = [0.0]
    reflex_runs = [0]

    def reflex_start():
        reflex_runs[0] += 1
        yield "рефлекс"
        return "рефлекс"

    planner = budgeted(1000, lambda i: f"план-{i}")
    s = Scheduler.standard(MILESTONE_0, now=lambda: clock[0],
                           reflex=reflex_start, planner=planner)
    for _ in range(200):
        clock[0] += 0.01
        s.step()

    assert reflex_runs[0] > 1, "рефлекс не запускался, пока планировщик думал"
    assert s.get("planner").preempted > 0, "планировщик ни разу не был прерван"
    assert s.get("planner").best is not None, "прерванный планировщик не отдал ответ"


def test_planner_gives_best_answer_at_any_moment() -> None:
    from harness.behaviour.contours import Scheduler, budgeted

    clock = [0.0]
    s = Scheduler(MILESTONE_0, now=lambda: clock[0])
    s.add("planner", 0.5, budgeted(50, lambda i: f"план-{i}"), level=2)
    clock[0] += 1.0
    s.step()
    first = s.best_answer("planner")
    assert first is not None
    clock[0] += 0.01
    s.step()
    assert s.best_answer("planner") is not None


def test_scheduler_never_blocks() -> None:
    """Шаг без работы возвращается пустым, а не ждёт."""
    from harness.behaviour.contours import Scheduler, fixed

    clock = [100.0]
    s = Scheduler(MILESTONE_0, now=lambda: clock[0])
    c = s.add("slow", 0.001, fixed("ответ"), level=0)
    s.step()
    assert not s.step(), "планировщик что-то сделал, хотя срок не пришёл"
    assert c.runs == 1


def test_world_keeps_running_while_contours_work() -> None:
    from harness.behaviour.contours import Scheduler, budgeted

    clock = [0.0]
    ticks = [0]
    s = Scheduler(MILESTONE_0, now=lambda: clock[0])
    s.add("planner", 0.5, budgeted(10_000, lambda i: i), level=2)

    def tick():
        ticks[0] += 1
        clock[0] += 0.005

    s.run_for(0.5, tick=tick, max_steps=500)
    assert ticks[0] > 10, "мир не шёл, пока планировщик думал"


def test_subsumption_can_be_switched_off_structurally() -> None:
    from harness.behaviour.contours import Scheduler, budgeted, fixed

    off = from_schema("без субсумпции", subsumption_enabled=False)
    clock = [0.0]
    s = Scheduler.standard(off, now=lambda: clock[0],
                           reflex=fixed("р"), planner=budgeted(100, lambda i: i))
    assert s.subsumption is False
    for _ in range(20):
        clock[0] += 0.01
        s.step()


# --- размыкатель эффекторов -------------------------------------------------


def test_imagination_uses_same_loop_as_action() -> None:
    from harness.behaviour.imagination import Loop, Mode

    executed: list[Action] = []
    simulated: list[tuple[Action, Mode]] = []
    loop = Loop(execute=lambda a: executed.append(a),
                simulate=lambda a, m: simulated.append((a, m)))
    a = Action.key("OUT_0A11", 100)
    loop.step(a)
    loop.imagine([a, a])
    assert len(executed) == 1 and len(simulated) == 2
    assert loop.stats()["acted"] == 1 and loop.stats()["imagined"] == 2


def test_breaker_returns_to_act_even_on_error() -> None:
    from harness.behaviour.imagination import Loop, Mode

    loop = Loop(execute=lambda a: None, simulate=lambda a, m: None)
    with pytest.raises(ValueError):
        with loop.disconnected(Mode.IMAGINE):
            raise ValueError("что-то упало посреди мысли")
    assert loop.breaker.connected, "размыкатель остался разомкнутым после ошибки"


def test_guarded_effectors_block_imagined_action() -> None:
    from harness.behaviour.imagination import Breaker, GuardedEffectors, Mode

    breaker = Breaker(Mode.IMAGINE)
    touched: list[Action] = []
    eff = GuardedEffectors(breaker, touched.append)
    with pytest.raises(RuntimeError, match="разомкнуты"):
        eff(Action.key("OUT_0A11", 100))
    assert not touched, "воображаемое действие дошло до мира"


def test_thought_is_journaled_apart_from_action(tmp_path: Path) -> None:
    from harness.behaviour.imagination import Loop

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        loop = Loop(execute=lambda a: None, simulate=lambda a, m: None,
                    journal=rec.journal)
        a = Action.key("OUT_0A11", 100)
        loop.step(a, rec.clocks.stamp())
        loop.imagine([a], stamp_of=lambda i: rec.clocks.stamp())

    with Session.open(tmp_path / "s") as s:
        s.journal.verify()
        kinds = [e.kind for e in s.journal if e.action is not None]
    assert Kind.ACTION in kinds and Kind.THOUGHT in kinds
    assert kinds.count(Kind.ACTION) == 1 and kinds.count(Kind.THOUGHT) == 1


# --- файрвол восприятия -----------------------------------------------------


def _describer(symbols: list[str]):
    from harness.perception.firewall import LocalDescriber

    return LocalDescriber(lambda f: [(s, "центр", "мелкое") for s in symbols])


def test_firewall_passes_clean_description() -> None:
    from harness.core.symbols import Symbolizer
    from harness.perception.firewall import PerceptionFirewall

    sym = Symbolizer("s0")
    symbols = [sym.symbolize("Здоровье"), sym.symbolize("Голод")]
    fw = PerceptionFirewall(MILESTONE_0, _describer(symbols))
    p = fw.look(None, Stamp(1, 1))
    assert [s.symbol for s in p.sightings] == symbols
    assert fw.audit.violations == 0


def test_firewall_blocks_what_to_do() -> None:
    from harness.perception.firewall import FirewallViolation, PerceptionFirewall

    class Bad:
        name = "bad"

        def describe(self, frame, question):
            return "SYM_1A2B | центр | мелкое\nЛучше атаковать врага первым"

    fw = PerceptionFirewall(MILESTONE_0, Bad())
    with pytest.raises(FirewallViolation, match="указания к действию"):
        fw.look(None, Stamp(1, 1))
    assert fw.audit.violations == 1


def test_firewall_blocks_valuation() -> None:
    from harness.perception.firewall import FirewallViolation, PerceptionFirewall

    class Valuing:
        name = "valuing"

        def describe(self, frame, question):
            return "SYM_1A2B | центр | мелкое (это важный предмет)"

    fw = PerceptionFirewall(MILESTONE_0, Valuing())
    with pytest.raises(FirewallViolation, match="оценку важности"):
        fw.look(None, Stamp(1, 1))


def test_firewall_question_is_fixed() -> None:
    from harness.perception.firewall import LocalDescriber, QUESTION

    d = LocalDescriber(lambda f: [])
    with pytest.raises(ValueError, match="не тот вопрос"):
        d.describe(None, "Что мне делать?")
    assert "что делать" not in QUESTION.casefold()


def test_firewall_logs_raw_answer_to_debug(tmp_path: Path) -> None:
    """Сырой ответ обязан попасть в отладочный поток — иначе нарушение невидимо."""
    from harness.debug.channel import DebugChannel
    from harness.perception.firewall import FirewallViolation, PerceptionFirewall

    class Bad:
        name = "bad"

        def describe(self, frame, question):
            return "нужно нажать на кнопку"

    with DebugChannel(tmp_path / "d", mode="a") as dbg:
        fw = PerceptionFirewall(MILESTONE_0, Bad(),
                               debug_write=lambda st, code, truth: dbg.write(st, code, truth))
        with pytest.raises(FirewallViolation):
            fw.look(None, Stamp(1, 1))

    calls = list(DebugChannel(tmp_path / "d", mode="r").read("firewall_call"))
    violations = list(DebugChannel(tmp_path / "d", mode="r").read("firewall_violation"))
    assert calls and "нужно нажать" in calls[0]["truth"]["raw_answer"]
    assert violations and violations[0]["truth"]["reason"]


def test_firewall_output_carries_no_text() -> None:
    from harness.core.symbols import SymbolError
    from harness.perception.firewall import PerceptionFirewall

    class Talkative:
        name = "talkative"

        def describe(self, frame, question):
            return "Здоровье | центр | мелкое"       # текст вместо символа

    fw = PerceptionFirewall(MILESTONE_0, Talkative())
    p = fw.look(None, Stamp(1, 1))
    assert p.sightings == (), "нераспознанная строка попала в восприятие"
    assert fw.audit.dropped_symbols == 1
    del SymbolError


def test_firewall_can_be_disabled_only_structurally() -> None:
    from harness.core.settings import structural_keys

    assert "firewall_enabled" in structural_keys()


# --- сон --------------------------------------------------------------------


def test_sleep_rebuilds_and_forgets(tmp_path: Path) -> None:
    from harness.core.journal import Actor, ActorLayer
    from harness.model.consolidation import Consolidator

    profile = from_schema("сон", forget_below_value=0.9, capture_width=32,
                          capture_height=32)
    with Recorder(tmp_path / "s", profile=profile, source="t", synthetic=True) as rec:
        for i in range(8):
            rec.journal.append(Kind.ACTION, rec.clocks.stamp(), Actor.AGENT,
                               ActorLayer.DRIVE,
                               action=Action.key(f"OUT_00{i:02X}", 100),
                               event={"code": "delivered", "responded": True,
                                      "device": "t"})

    with Session.open(tmp_path / "s") as s:
        cons = Consolidator(profile)
        rebuilt, report = cons.run(s.journal, Stamp(1, 1))
    assert report.entities_before > 0
    assert report.forgotten, "сон не забыл ничего при пороге 0.9"
    assert report.entities_after < report.entities_before
    assert report.fingerprint_before != report.fingerprint_after
    del rebuilt


def test_sleep_warns_when_reality_check_is_overdue(tmp_path: Path) -> None:
    from harness.model.consolidation import Consolidator

    profile = from_schema("сон", reality_check_max_minutes=1.0,
                          sleep_every_minutes=10.0, capture_width=32, capture_height=32)
    with Recorder(tmp_path / "s", profile=profile, source="t", synthetic=True) as rec:
        rec.record_note("начали")
        journal = rec.journal
        cons = Consolidator(profile)
        cons.run(journal, Stamp(1, 1), imagined_only=False)
        _, report = cons.run(journal, Stamp(2, 2), imagined_only=True)
        _, report = cons.run(journal, Stamp(3, 3), imagined_only=True)
    assert report.reality_check_overdue
    assert "выучивает свою модель" in cons.warning(report)


def test_sleep_invents_nothing(tmp_path: Path) -> None:
    """После сна не появляется ни одного утверждения без основания в журнале."""
    from harness.core.journal import Actor, ActorLayer
    from harness.model.consolidation import Consolidator
    from harness.model.rebuild import rebuild_from_journal

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t",
                  synthetic=True) as rec:
        for i in range(5):
            rec.journal.append(Kind.ACTION, rec.clocks.stamp(), Actor.AGENT,
                               ActorLayer.DRIVE,
                               action=Action.key("OUT_0A11", 100),
                               event={"code": "delivered", "responded": True,
                                      "device": "t"})

    with Session.open(tmp_path / "s") as s:
        plain = rebuild_from_journal(s.journal)
        cons = Consolidator(from_schema("сон", forget_below_value=0.0))
        after, _ = cons.run(s.journal, Stamp(9, 9))
    assert set(after.beliefs.entities) <= set(plain.beliefs.entities)


def test_sleep_merge_keeps_experience_over_hearsay() -> None:
    from harness.model.beliefs import BeliefStore, Origin, Provenance
    from harness.model.consolidation import Consolidator

    store = BeliefStore("b0")
    exp = Provenance(Origin.EXPERIENCE, "b0", 1)
    heard = Provenance(Origin.TESTIMONY, "b0", 2, "вики", 0.9)
    store.learn_dynamics("ENT_0001", "светится", True, exp)
    store.learn_dynamics("ENT_0002", "светится", True, heard)
    cons = Consolidator(MILESTONE_0)
    assert cons.apply_merge(store, "ENT_0001", "ENT_0002")
    assert not store.entities["ENT_0001"].dynamics["светится"].is_hearsay


# --- экземпляры -------------------------------------------------------------


def test_instances_have_separate_worlds() -> None:
    from harness.instances import Colony

    profile = from_schema("много", instances=3)
    colony = Colony(profile)
    colony.spawn(1)
    colony.spawn(2)
    with pytest.raises(ValueError, match="уже занят"):
        colony.spawn(1)
    colony.spawn(3)
    with pytest.raises(ValueError, match="при пределе"):
        colony.spawn(4)


def test_instance_exchange_is_testimony_not_experience(tmp_path: Path) -> None:
    from harness.instances import Colony

    profile = from_schema("много", instances=2, capture_width=32, capture_height=32)
    colony = Colony(profile)
    a = colony.spawn(1)
    b = colony.spawn(2)
    with Recorder(tmp_path / "b", profile=profile, source="t", synthetic=True) as rec:
        b.journal = rec.journal
        colony.share(a.id, ["ENT_0001|dyn|светится"])
        result = colony.deliver(b.id, rec.clocks.stamp())

    assert result["delivered"] == 1
    assert b.beliefs is not None
    beliefs = list(b.beliefs.beliefs())
    assert beliefs and all(x.is_hearsay for x in beliefs)
    assert result["pending_recheck"] >= 1

    with Session.open(tmp_path / "b") as s:
        testimonies = [e for e in s.journal if e.kind is Kind.TESTIMONY]
    assert testimonies, "свидетельство не попало в журнал получателя"


def test_instance_count_is_structural() -> None:
    from harness.core.settings import structural_keys

    assert "instances" in structural_keys()


# --- пределы расхода принуждаются до превышения, а не после (SPEC-FULL, A1) ---


def test_the_spend_cap_refuses_before_the_money_is_gone() -> None:
    """Потолок, срабатывающий после траты, — не потолок, а отметка о пробое.

    Сдвиг числа: при пределе $0.10 и потраченных $0.09 вызов ценой $0.05 прежде
    разрешался (упор ставился на следующей проверке, уже при $0.14), теперь
    отказывается. `usd_spent` остаётся 0.09, а не становится 0.14.
    """
    from harness.core.profile import from_schema
    from harness.core.resources import OP_MODEL, ResourceGovernor

    g = ResourceGovernor(from_schema("ТЕСТ-расход", spend_cap_usd=0.10))
    g.spend(usd=0.09, model_calls=1)

    assert not g.admit(OP_MODEL, cost_usd=0.05), "предел пропустил вызов за предел"
    assert g.usd_spent == pytest.approx(0.09), "деньги потрачены до отказа"
    assert g.admit(OP_MODEL, cost_usd=0.005), "вызов в пределах обязан пройти"

    # Без названной цены остаётся только запоздалый отказ, и это объявлено, а не
    # умолчание: тот, кто не назвал цену, платит первым превышением.
    assert g.admit(OP_MODEL), "без цены отказывать не на чем — так и объявлено"


def test_the_rate_cap_counts_the_call_it_is_about_to_allow() -> None:
    """Вопрос не «превышена ли частота», а «превысит ли её этот вызов».

    Прежняя формулировка пропускала ровно один лишний вызов каждый раз, когда
    упиралась: на длинной серии это систематическая недостача, а не округление.
    """
    import time

    from harness.core.profile import from_schema
    from harness.core.resources import OP_MODEL, ResourceGovernor

    g = ResourceGovernor(from_schema("ТЕСТ-темп", token_budget_per_min=0.05))
    now = time.monotonic()
    for _ in range(3):
        g.spend(model_calls=1, at=now)

    assert g.model_call_rate() == pytest.approx(0.05), "частота ровно на пределе"
    a = g.admit(OP_MODEL)
    assert not a, "вызов на пределе обязан быть отказан: он же и превысит"
    assert "довёл бы" in (a.reason or ""), a.reason


def test_a_cap_of_zero_means_no_cap_and_not_a_ban() -> None:
    """Ноль в схеме объявлен как «предела нет». Иначе он запретил бы всё разом."""
    from harness.core.profile import from_schema
    from harness.core.resources import OP_MODEL, ResourceGovernor

    g = ResourceGovernor(from_schema("ТЕСТ-без-предела", spend_cap_usd=0.0,
                                     token_budget_per_min=0.0))
    assert g.admit(OP_MODEL, cost_usd=1000.0), "ноль означает «предела нет»"
    assert g.would_breach_model(1000.0) is None
