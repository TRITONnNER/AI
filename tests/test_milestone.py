"""Критерии готовности вехи 0 — по тесту на каждый пункт.

Где критерий требует живой машины (десять минут захвата без пропусков, ввод,
дошедший до настоящей игры), проверяется то, что проверяемо офлайн: что харнесс
**замечает** пропуск, рассинхрон и зависание. Сам захват и сама инъекция
остаются непроверенными, и это записано в `docs/ARCHITECTURE-HARNESS.md`,
раздел «Что не проверено», а не спрятано в зелёном тесте.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from harness.core.action import Action
from harness.core.journal import Actor, ActorLayer, Kind
from harness.core.profile import MILESTONE_0
from harness.session import Recorder, Session


# --- 0.1. Захват -----------------------------------------------------------


def test_0_1_frames_carry_three_clocks(session: Session) -> None:
    for c, _ in session:
        assert c.stamp.t_self > 0
        assert c.stamp.t_world >= 0
        assert c.stamp.t_content is None      # в вехе 0 содержимого нет


def test_0_1_audio_is_stereo_and_synced(session: Session) -> None:
    block = session.audio_block(0)
    assert block is not None, "в корпусе нет звука"
    assert block.ndim == 2 and block.shape[1] == 2, "звук обязан быть стерео"
    report = session.verify()
    assert report["worst_audio_offset_ms"] <= report["audio_sync_tolerance_ms"]


def test_0_1_channels_differ_so_bearing_is_recoverable(session: Session) -> None:
    """Пеленг считается из разницы каналов, значит разница обязана быть."""
    diffs = []
    for i in range(0, len(session), 7):
        b = session.audio_block(i)
        if b is None:
            continue
        left = b[:, 0].astype(np.float64)
        right = b[:, 1].astype(np.float64)
        diffs.append(abs(np.abs(left).mean() - np.abs(right).mean()))
    assert diffs and max(diffs) > 1.0, "каналы неразличимы: пеленг не восстановить"


def test_0_1_desync_is_detected(tmp_path: Path) -> None:
    """Рассинхрон больше допуска обязан попасть в отчёт, а не пройти молча."""
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="test",
                  synthetic=True) as rec:
        audio = np.zeros((160, 2), dtype=np.int16)
        rec.record_frame(np.zeros((16, 16), dtype=np.uint8), audio=audio,
                         audio_offset_ms=5.0)
        rec.record_frame(np.zeros((16, 16), dtype=np.uint8), audio=audio,
                         audio_offset_ms=180.0)      # заведомо больше допуска 50 мс
    with Session.open(tmp_path / "s") as s:
        report = s.verify()
    assert not report["ok"]
    assert any("рассинхрон" in p for p in report["problems"])


def test_0_1_frame_gap_is_detected(tmp_path: Path) -> None:
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="test",
                  synthetic=True) as rec:
        rec.record_frame(np.zeros((16, 16), dtype=np.uint8), t_world=0)
        rec.record_frame(np.zeros((16, 16), dtype=np.uint8), t_world=1)
        rec.record_frame(np.zeros((16, 16), dtype=np.uint8), t_world=9)   # пропуск
    with Session.open(tmp_path / "s") as s:
        report = s.verify()
    assert not report["ok"]
    assert any("тиках мира" in p for p in report["problems"])


# --- 0.2. Инъекция ввода ---------------------------------------------------


def test_0_2_api_shape() -> None:
    """`press(key, duration_ms, modifiers)` и `move_mouse(dx, dy)`."""
    from harness.agentside.body import Body

    sent: list[Action] = []
    body = Body(["OUT_0A11", "MOD_1B2C"], submit=sent.append)
    body.press("OUT_0A11", 340, modifiers=("MOD_1B2C",))
    body.move(12, -4, 16)
    body.wait(50)
    assert [a.kind.value for a in sent] == ["key", "mouse_move", "nothing"]
    assert sent[0].duration_ms == 340 and sent[0].modifiers == ("MOD_1B2C",)
    assert (sent[1].dx, sent[1].dy) == (12, -4)


def test_0_2_stop_interrupts_within_one_frame() -> None:
    """СТОП обрывает удержание не позже одного кадра.

    Считаем не по стенным часам, а по числу срезов сна: удержание 1000 мс с
    срезом 5 мс, стоп после третьего среза — значит спали не больше четырёх раз.
    Один кадр при 30 к/с — 33 мс, то есть шесть срезов.
    """
    from harness.inject.base import InjectionSink, NullDevice
    from harness.inject.stop import StopSwitch

    stop = StopSwitch()
    dev = NullDevice()
    calls: list[float] = []

    def sleep(dt: float) -> None:
        calls.append(dt)
        if len(calls) == 3:
            stop.engage("тест")

    sink = InjectionSink(dev, slice_ms=5.0, sleep=sleep)
    completed, _ = sink.hold(["OUT_0A11"], 1000, abort=lambda: stop.is_engaged)

    assert not completed
    assert len(calls) <= 4, f"после стопа поспали ещё {len(calls) - 3} раз"
    frame_ms = 1000.0 / MILESTONE_0.parameters["capture_fps"]
    assert sum(calls) * 1000.0 <= frame_ms, "обрыв занял больше кадра"
    assert dev.calls[-1] == ("up", "OUT_0A11"), "выход не отпущен после стопа"


def test_0_2_held_output_is_released_even_on_failure() -> None:
    """Если выше по стеку всё рухнуло, зажатая клавиша всё равно отпускается."""
    from harness.inject.base import InjectionSink, NullDevice

    dev = NullDevice()

    def boom(_: float) -> None:
        raise KeyboardInterrupt("как будто прервали процесс")

    sink = InjectionSink(dev, slice_ms=5.0, sleep=boom)
    with pytest.raises(KeyboardInterrupt):
        sink.hold(["OUT_0A11"], 100, abort=lambda: False)
    assert ("up", "OUT_0A11") in dev.calls


def test_0_2_watchdog_trips_on_still_screen(tmp_path: Path) -> None:
    from harness.inject.stop import StopSwitch
    from harness.inject.watchdog import Watchdog

    clock = [0.0]
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="test",
                  synthetic=True) as rec:
        stop = StopSwitch(rec.journal)
        wd = Watchdog(still_seconds=1.0, threshold=0.002, stop=stop,
                      journal=rec.journal, now=lambda: clock[0])
        still = np.zeros((32, 32), dtype=np.uint8)
        trip = None
        for _ in range(30):
            clock[0] += 0.1
            rec.record_frame(still)
            trip = wd.feed(still, rec.clocks.stamp()) or trip
        assert trip is not None and trip.code == "screen_still"
        assert stop.is_engaged and stop.reason.startswith("watchdog:")

    with Session.open(tmp_path / "s") as s:
        kinds = [e.kind for e in s.journal]
        assert Kind.WATCHDOG in kinds and Kind.STOP in kinds


def test_0_2_watchdog_does_not_trip_on_moving_screen(tmp_path: Path) -> None:
    from harness.inject.watchdog import Watchdog

    clock = [0.0]
    wd = Watchdog(still_seconds=1.0, threshold=0.002, now=lambda: clock[0])
    rng = np.random.default_rng(0)
    for _ in range(30):
        clock[0] += 0.1
        assert wd.feed(rng.integers(0, 255, (32, 32), dtype=np.uint8),
                       __import__("harness.core.clocks", fromlist=["Stamp"]).Stamp(1, 1)) is None
    assert wd.tripped is None


def test_0_2_watchdog_trips_when_process_gone(tmp_path: Path) -> None:
    from harness.core.clocks import Stamp
    from harness.inject.watchdog import Watchdog

    alive = [True]
    wd = Watchdog(still_seconds=99.0, threshold=0.002,
                  process_alive=lambda: alive[0])
    frame = np.zeros((8, 8), dtype=np.uint8)
    assert wd.feed(frame, Stamp(1, 1)) is None
    alive[0] = False
    trip = wd.feed(frame, Stamp(2, 2))
    assert trip is not None and trip.code == "process_gone"


def test_0_2_masked_attempt_is_journaled(tmp_path: Path) -> None:
    """Главное исправление против дизайна пульта: маска не прячет попытку.

    Иначе агент завёл бы убеждение «выход молчит», а причины в журнале не было
    бы, и пересборка убеждений дала бы другую карту тела.
    """
    from harness.inject.base import Injector, InjectionSink, NullDevice
    from harness.inject.mask import InputMask

    mask = InputMask()
    mask.block("OUT_0A11", scope="window")
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="test",
                  synthetic=True) as rec:
        inj = Injector(InjectionSink(NullDevice(), sleep=lambda _: None), rec.journal,
                       mask=mask)
        out = inj.submit(Action.key("OUT_0A11", 120), rec.clocks.stamp(),
                         ActorLayer.REFLEX)
        assert out.masked and not out.delivered

    with Session.open(tmp_path / "s") as s:
        masked = [a for _, a in s.actions() if a.masked]
        assert len(masked) == 1
        assert masked[0].mask_reason and "mask:window" in masked[0].mask_reason
        assert masked[0].duration_ms == 120, "длительность попытки тоже сохранена"
        assert s.journal.stats()["masked_actions"] == 1


def test_0_2_mask_scopes_are_nested() -> None:
    from harness.inject.mask import InputMask

    m = InputMask()
    m.block("OUT_0A11", scope="device")
    assert m.blocked_outputs(["OUT_0A11"], scope="window") == {"OUT_0A11"}
    m2 = InputMask()
    m2.block("OUT_0A11", scope="window")
    assert m2.blocked_outputs(["OUT_0A11"], scope="device") == set()


# --- 0.3. Журнал -----------------------------------------------------------


def test_0_3_frames_are_separate_storage(corpus: Path) -> None:
    """Кадры отдельным хранилищем, метаданные индексом. Не файл на кадр."""
    assert (corpus / "frames" / "index.jsonl").exists()
    shards = list((corpus / "frames").glob("shard-*.bin"))
    assert shards, "нет шардов кадров"
    with Session.open(corpus) as s:
        n = len(s)
    assert len(shards) < n, "кадров столько же, сколько файлов: это файл на кадр"


def test_0_3_random_access_is_fast(session: Session) -> None:
    """«Произвольный момент открывается за доли секунды»."""
    idx = len(session) - 1
    t0 = time.perf_counter()
    session.image(idx)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    session.image(len(session) // 3)
    second = time.perf_counter() - t0
    assert first < 0.2, f"последний кадр открывался {first * 1000:.0f} мс"
    assert second < 0.2, f"произвольный кадр открывался {second * 1000:.0f} мс"


def test_0_3_delta_encoding_is_exact(tmp_path: Path) -> None:
    """Разностное кодирование обратимо точно: ни одного изменённого пикселя."""
    from harness.core.blobstore import FrameStore

    rng = np.random.default_rng(3)
    base = rng.integers(0, 255, (48, 64), dtype=np.uint8)
    frames = [base]
    for i in range(1, 40):
        f = frames[-1].copy()
        f[i % 48, :] = (f[i % 48, :].astype(np.int16) + 37) % 256   # с переполнением
        frames.append(f)

    with FrameStore(tmp_path / "fs", mode="a", keyframe_interval=10) as fs:
        for f in frames:
            fs.append_frame(f)
    with FrameStore(tmp_path / "fs", mode="r", keyframe_interval=10) as fs:
        stats = fs.encoding_stats()
        # обратный порядок: проверяем, что сборка не зависит от кэша
        for i in reversed(range(len(frames))):
            assert np.array_equal(fs.read(i), frames[i]), f"кадр {i} восстановлен неверно"
    assert stats["deltas"] > stats["keyframes"]


def test_0_3_still_frames_are_cheap(tmp_path: Path) -> None:
    """Неподвижный экран почти не занимает места — за этим и нужна разность."""
    from harness.core.blobstore import FrameStore

    rng = np.random.default_rng(5)
    frame = rng.integers(0, 255, (180, 320), dtype=np.uint8)
    with FrameStore(tmp_path / "fs", mode="a", keyframe_interval=60) as fs:
        refs = [fs.append_frame(frame) for _ in range(30)]
    key, deltas = refs[0], refs[1:]
    assert key.nbytes > 10_000, "опорный кадр подозрительно мал"
    assert max(r.nbytes for r in deltas) < key.nbytes / 100, (
        "разность неподвижного кадра должна сжиматься почти в ноль")


def test_0_3_actor_is_recorded(session: Session) -> None:
    actors = {e.actor for e in session.journal}
    assert actors <= {Actor.AGENT, Actor.HUMAN, Actor.NONE}
    assert Actor.HUMAN in actors, "в корпусе играет человек, это должно быть видно"


# --- 0.4. Воспроизведение --------------------------------------------------


def test_0_4_frame_by_frame_without_game(session: Session) -> None:
    seen = 0
    prev_world = -1
    for c, img in session:
        assert img.shape[0] > 0 and img.shape[1] > 0
        assert c.stamp.t_world > prev_world
        prev_world = c.stamp.t_world
        seen += 1
    assert seen == len(session)


def test_0_4_seek_and_step(session: Session) -> None:
    mid = len(session) // 2
    c = session.seek(mid)
    assert c.index == mid
    assert session.step(1).index == mid + 1
    assert session.step(-1).index == mid
    assert session.step(-10 ** 6).index == 0          # упирается, а не падает
    assert session.step(10 ** 6).index == len(session) - 1


def test_0_4_seek_by_world_tick(session: Session) -> None:
    target = session.cursor(len(session) // 3).stamp.t_world
    c = session.seek_world(target)
    assert c.stamp.t_world == target


def test_0_4_map_over_whole_session(session: Session) -> None:
    """«Прогнать произвольную функцию по всей записи и получить результат на
    каждом кадре» — критерий готовности 0.4."""
    means = session.map(lambda img, cur: float(img.mean()))
    assert len(means) == len(session)
    assert all(0.0 <= m <= 255.0 for m in means)

    with_audio = session.map(lambda img, aud, cur: (img.shape, None if aud is None else aud.shape),
                             with_audio=True)
    assert len(with_audio) == len(session)
    assert with_audio[0][1] is not None


def test_0_4_replay_does_not_modify_session(corpus: Path) -> None:
    """Воспроизведение не может испортить запись: всё открыто только на чтение."""
    before = {p: p.stat().st_mtime_ns for p in corpus.rglob("*") if p.is_file()}
    with Session.open(corpus) as s:
        s.map(lambda img, cur: img.mean())
        s.verify()
    after = {p: p.stat().st_mtime_ns for p in corpus.rglob("*") if p.is_file()}
    assert before == after


# --- 0.5. Профиль ----------------------------------------------------------


def test_0_5_filter_journal_by_hash(tmp_path: Path) -> None:
    """«По хешу можно отфильтровать журнал и получить только совместимые записи»."""
    rec = Recorder(tmp_path / "s", profile=MILESTONE_0, source="test", synthetic=True)
    rec.record_frame(np.zeros((8, 8), dtype=np.uint8))
    rec.record_frame(np.ones((8, 8), dtype=np.uint8))
    first_structure = rec.profile.structure_hash
    rec.fork(MILESTONE_0.with_structural(frame_format="rgb8"), reason="сменили формат")
    rec.record_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    rec.close()

    from harness.core.journal import Journal, branch_chain

    chain = branch_chain(tmp_path / "s" / "journal")
    assert len(chain) == 2
    old = Journal(tmp_path / "s" / "journal" / "branches" / chain[0].branch_id)
    new = Journal(tmp_path / "s" / "journal" / "branches" / chain[1].branch_id)
    assert all(e.structure_hash == first_structure for e in old.compatible())
    assert all(e.structure_hash != first_structure for e in new)
    old.close()
    new.close()


def test_0_5_hash_is_stable_across_processes() -> None:
    """Хеш обязан совпадать между запусками, иначе фильтр по нему бессмыслен."""
    import subprocess
    import sys

    code = ("import sys; sys.path.insert(0, 'src');"
            "from harness.core.profile import MILESTONE_0 as p;"
            "print(p.profile_hash); print(p.structure_hash)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parent.parent, check=True)
    a, b = out.stdout.split()
    assert a == MILESTONE_0.profile_hash
    assert b == MILESTONE_0.structure_hash


def test_0_5_parameter_change_keeps_structure_hash() -> None:
    tuned = MILESTONE_0.with_parameters(capture_fps=15.0)
    assert tuned.structure_hash == MILESTONE_0.structure_hash
    assert tuned.profile_hash != MILESTONE_0.profile_hash
    assert MILESTONE_0.is_clean_ablation(tuned)


def test_0_5_profile_roundtrip(tmp_path: Path) -> None:
    from harness.core.profile import Profile

    path = tmp_path / "p.json"
    MILESTONE_0.save(path)
    back = Profile.load(path)
    assert back.profile_hash == MILESTONE_0.profile_hash
    assert back.parameters == MILESTONE_0.parameters


# --- 0.6. Разделение себя и мира -------------------------------------------


def test_0_6_finds_interface_without_coordinates(corpus: Path) -> None:
    """Критерий: «стабильно выделяет область интерфейса без единой захардкоженной
    координаты и без знания, что это за игра».

    Замеренные значения на шести сидах: IoU 0.969…0.976, полнота 0.979…0.981,
    точность 0.989…0.995. Порог поставлен ниже замеров, но не настолько, чтобы
    проходить при поломке метода.

    Раньше точность была 0.62: окно захватывало контрастную рамку интерфейса и
    давало ореол вокруг него. Разбор — в ARCHITECTURE-AGENT.md.
    """
    from harness.corpus.synthetic import load_hud_mask
    from harness.vision.selfworld import SCREEN, SelfWorldSeparator

    with Session.open(corpus) as s:
        sep = SelfWorldSeparator(s.profile)
        for _, img in s:
            sep.feed(img)
        res = sep.result()

    truth = load_hud_mask(corpus)
    found = res.pixel_mask(SCREEN)
    inter = int((found & truth).sum())
    union = int((found | truth).sum())
    recall = inter / int(truth.sum())
    precision = inter / max(1, int(found.sum()))

    assert recall >= 0.95, f"пропущено слишком много интерфейса: полнота {recall:.3f}"
    assert precision >= 0.95, f"слишком много ложного интерфейса: точность {precision:.3f}"
    assert inter / union >= 0.92, f"IoU {inter / union:.3f}"


def test_0_6_static_screen_condition_is_what_gives_precision(corpus: Path) -> None:
    """Точность держится на том, что экранный слой между кадрами не меняется.

    Если ослабить это условие — как пришлось бы для источника, сжатого с потерями, —
    точность падает. Проверяем, что падает именно она, а не что-то другое: иначе
    настройка `screen_static_epsilon` окажется ручкой без смысла.
    """
    from harness.core.profile import from_schema
    from harness.corpus.synthetic import load_hud_mask
    from harness.vision.selfworld import SCREEN, SelfWorldSeparator

    truth = load_hud_mask(corpus)
    scores = {}
    for eps in (0.5, 64.0):
        with Session.open(corpus) as s:
            profile = from_schema(
                "тест", capture_width=int(s.profile.parameters["capture_width"]),
                capture_height=int(s.profile.parameters["capture_height"]),
                screen_static_epsilon=eps)
            sep = SelfWorldSeparator(profile)
            for _, img in s:
                sep.feed(img)
            found = sep.result().pixel_mask(SCREEN)
        tp = int((found & truth).sum())
        scores[eps] = tp / max(1, int(found.sum()))

    assert scores[0.5] > scores[64.0] + 0.2, (
        f"ослабление условия статичности не ухудшило точность: {scores}. "
        "Значит точность держится на чём-то другом, и настройка вводит в заблуждение")


def test_0_6_still_camera_gives_no_votes(profile) -> None:
    """При неподвижной камере слои неразличимы, и это честное «не знаю»."""
    from harness.vision.selfworld import UNDECIDED, SelfWorldSeparator

    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (64, 96), dtype=np.uint8)
    sep = SelfWorldSeparator(profile)
    for _ in range(10):
        sep.feed(frame.copy())
    res = sep.result()
    assert res.voting_frames == 0
    assert res.skipped_frames == 9
    assert (res.labels == UNDECIDED).all()


def test_0_6_global_shift_is_recovered(profile) -> None:
    """Сдвиг камеры оценивается верно — на этом стоит всё остальное."""
    from harness.vision.selfworld import estimate_global_shift

    rng = np.random.default_rng(2)
    big = rng.integers(0, 255, (200, 260), dtype=np.uint8)
    for dy, dx in ((0, 5), (3, 0), (-4, 7), (6, -6)):
        a = big[20:120, 20:140]
        b = big[20 - dy:120 - dy, 20 - dx:140 - dx]
        got = estimate_global_shift(a, b)
        assert (got.dy, got.dx) == (dy, dx), f"ожидали {(dy, dx)}, получили {(got.dy, got.dx)}"


def test_0_6_background_excess_is_measurable(corpus: Path) -> None:
    """Первые два пункта 0.6: фон без действий и превышение над ним при действии."""
    from harness.vision.selfworld import background_level

    with Session.open(corpus) as s:
        acted = {e.stamp.t_world for e, _ in s.actions()}
        pairs = []
        prev = None
        for c, img in s:
            if prev is not None:
                pairs.append((prev, img, c.stamp.t_world in acted))
            prev = img
        bg = background_level(pairs)

    assert bg.idle_n > 0 and bg.acting_n > 0
    assert bg.acting_mean > bg.idle_mean
    assert bg.excess > 3.0, f"действие не отличается от фона: {bg.excess:.1f} σ"


def test_0_6_uses_no_truth_from_debug_channel() -> None:
    """Модуль 0.6 не имеет пути к отладочному потоку: он не может подсмотреть."""
    import harness.vision.selfworld as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "debug" not in src.replace("# ", ""), "в 0.6 упомянут отладочный поток"
    assert "hud" not in src.lower(), "в 0.6 есть знание про интерфейс"
