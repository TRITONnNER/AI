"""Пакет для оператора: doctor, selftest, план записи, приём чужих записей.

`TASK-07`. Проверяется не «команда запускается», а то, ради чего она есть: человек,
впервые получивший репозиторий, доходит до записи без чтения кода и без угадывания.
Поэтому тесты спрашивают у вывода то же, что спросил бы оператор: что чинить первым,
куда именно нажать, чем отличается пустой захват от настоящего.

Главные из них — про **чёрный кадр**. Захват без разрешения на macOS возвращает кадр
правильной формы, заполненный нулями; захват под Wayland через XWayland — то же самое.
Проверка «вернулся ли кадр» отвечает на это «да», и потому здесь её нет ни в одном
месте: смотрится содержимое.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from harness import doctor, selftest
from harness.capture.base import BackendUnavailable, Frame
from harness.machine import Machine, Session, detect


# ---------------------------------------------------------------------------
# Определение машины: X11 и Wayland — разные механизмы, и путать их нельзя
# ---------------------------------------------------------------------------


def test_wayland_wins_over_display_variable() -> None:
    """Под Wayland `DISPLAY` тоже выставлен (XWayland), и решать по нему нельзя.

    Решение по `DISPLAY` дало бы «x11» на Wayland-машине, а это самый дорогой из
    возможных ответов: захват не откажется, а вернёт чёрный кадр или окна XWayland.
    """
    from harness.machine import _linux_session

    session, source = _linux_session({"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"})
    assert session is Session.WAYLAND, source

    session, _ = _linux_session({"XDG_SESSION_TYPE": "wayland", "DISPLAY": ":0"})
    assert session is Session.WAYLAND

    session, _ = _linux_session({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"})
    assert session is Session.X11

    session, source = _linux_session({})
    assert session is Session.NONE and "DISPLAY" in source


def test_no_capture_backend_for_wayland() -> None:
    """Под Wayland механизма нет, и это объявлено, а не выясняется чёрным кадром."""
    from harness.capture.select import CANDIDATES
    from harness.machine import HAS_CAPTURE

    assert HAS_CAPTURE[Session.WAYLAND] is False
    assert CANDIDATES[Session.WAYLAND] == ()
    assert [b.name for b in CANDIDATES[Session.X11]] == ["screen_mss"]
    assert [b.name for b in CANDIDATES[Session.MACOS]] == ["screen_mss"]


def test_windows_prefers_desktop_duplication_over_mss() -> None:
    """На Windows dxcam первым, mss запасным. Порядок измерен, а не выбран.

    Через mss на 1080p замер дал 19.9 кадр/с при заявленных 30, и полноэкранные
    DirectX-приложения он не видит вовсе. Прежняя таблица «одна система — один
    backend» брала mss при установленном и работающем dxcam.
    """
    from harness.capture.select import CANDIDATES

    order = [b.name for b in CANDIDATES[Session.WINDOWS]]
    assert order == ["screen_dxcam", "screen_mss"], order
    mss = CANDIDATES[Session.WINDOWS][1]
    assert mss.caveat, "у запасного пути обязана быть названа его слабость"
    assert "не" in mss.caveat and "захватыва" in mss.caveat


def test_screen_capture_refuses_wayland_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness.capture.screen import ScreenCapture

    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    with pytest.raises(BackendUnavailable, match="Wayland"):
        ScreenCapture().start()


def test_backend_registry_agrees_with_the_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Реестр и сам backend отвечают по одному знанию, а не по двум копиям.

    Копий было две, и они уже расходились: реестр считал Wayland годным.
    """
    from harness.capture.base import describe_backends

    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert describe_backends()["screen_mss"]["available"] is False


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def _blind(state: doctor.State, detail: str, size: Any = None, black: bool = False):
    return lambda _m: (state, detail, size, black)


def test_doctor_on_a_machine_without_a_display_says_no_and_why() -> None:
    """Критерий готовности 2. Здесь дисплея нет, и это тот самый случай."""
    rep = doctor.run()
    if rep.machine.session is not Session.NONE:
        pytest.skip("на этой машине есть графическая сессия")
    assert not rep.can_record
    assert "ЗАПИСЫВАТЬ НЕЛЬЗЯ" in rep.verdict()
    assert "экраном" in rep.verdict(), "не сказано, что именно чинить первым"


def test_every_missing_check_carries_a_command_to_copy() -> None:
    """Правило вывода: не «установите ffmpeg», а строка, которую можно скопировать."""
    rep = doctor.run()
    for c in rep.checks:
        if c.state is not doctor.State.YES:
            assert c.fix, f"{c.name}: нехватка без команды"
            assert len(c.fix) > 10, f"{c.name}: подсказка {c.fix!r} слишком коротка"


def test_a_check_cannot_be_created_without_a_fix() -> None:
    """Механически, а не по памяти: пункт без команды не собирается."""
    with pytest.raises(ValueError, match="не сказала, что делать"):
        doctor.Check("что-то", doctor.State.NO, "нет")


def test_blocking_and_later_are_separate_and_cannot_be_mixed() -> None:
    """Отсутствие /dev/uinput записи не мешает — иначе оператор чинит лишнее."""
    rep = doctor.run()
    uinput = next(c for c in rep.checks if "инъекц" in c.name)
    assert not uinput.blocks, "права на инъекцию ввода записи не мешают"
    if uinput.state is not doctor.State.YES:
        assert uinput.later, "сказано «нет», но не сказано, к чему понадобится"
    with pytest.raises(ValueError, match="разные вещи"):
        doctor.Check("х", doctor.State.NO, "н", blocks=True, later="М5", fix="ввести")

    ffmpeg = next(c for c in rep.checks if c.name == "ffmpeg")
    assert not ffmpeg.blocks, "ffmpeg нужен третьему уровню журнала, не записи"


def test_black_frame_is_a_failure_not_a_success() -> None:
    """Кадр правильной формы, заполненный нулями, — отказ.

    Ровно так ведёт себя macOS без разрешения «Запись экрана». Проверка «вернулся ли
    кадр» отвечает «да», и потому её здесь нет.
    """
    m = detect()
    probe = _blind(doctor.State.NO, "кадр 1920×1080 целиком чёрный",
                   (1920, 1080), True)
    check = doctor.check_display(m, probe(m))
    assert check.state is doctor.State.NO and check.blocks

    perm = doctor.check_permissions(
        Machine("Darwin", Session.MACOS, "тест", (3, 11, 0), "23"), probe(m))
    assert perm.state is doctor.State.NO
    assert "Запись экрана" in perm.fix, (
        "на macOS чёрный кадр обязан вести к разрешению, а не к установке пакетов")


def test_probe_rejects_a_uniform_frame() -> None:
    """Однородный кадр — тоже пустышка: разброс яркости нулевой.

    Полностью серый кадр не чёрный по максимуму, но экраном не является. Без этой
    проверки backend, отдающий залитый буфер, прошёл бы как рабочий.
    """
    class Uniform:
        name = "тест-однородный"

        def start(self) -> None: ...
        def stop(self) -> None: ...

        def read(self) -> Frame:
            return Frame(np.full((64, 64), 128, dtype=np.uint8), 0, 0)

    state, detail, _size, black = _probe_with(Uniform())
    assert state is doctor.State.NO and black
    assert "однород" in detail


def _probe_with(source: Any) -> tuple[Any, str, Any, bool]:
    """Прогнать проверку содержимого через подменённый источник.

    Подменяется **выбор механизма**, а не класс захвата: с TASK-08 `probe_screen`
    берёт источник у `capture.select.open_screen`, чтобы доктор проверял тот же
    механизм, которым потом пойдёт запись.
    """
    import harness.capture.select as select_mod

    real = select_mod.open_screen
    try:
        select_mod.open_screen = lambda m, **kw: select_mod.Choice(   # type: ignore[assignment]
            chosen=select_mod.MSS, considered=(select_mod.MSS,), source=source)
        return doctor.probe_screen(detect(), pause_s=0.0)
    finally:
        select_mod.open_screen = real                     # type: ignore[assignment]


def test_free_space_is_reported_in_hours_not_gigabytes(tmp_path: Path) -> None:
    """Гигабайты оператору ничего не говорят: чтобы понять, хватит ли, нужен код."""
    check = doctor.check_disk(detect(), path=tmp_path, fps=30.0, size=(1920, 1080))
    assert "ч " in check.detail and "кадр/с" in check.detail


def test_doctor_json_is_machine_readable() -> None:
    payload = json.dumps(doctor.run().as_dict(), ensure_ascii=False)
    back = json.loads(payload)
    assert "verdict" in back and "checks" in back and "machine" in back


# ---------------------------------------------------------------------------
# selftest: настоящий захват против пустышки
# ---------------------------------------------------------------------------


class FakeSource:
    """Источник кадров для проверки самой проверки."""

    name = "тест-источник"

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._i = 0
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def read(self) -> Frame | None:
        if self._i >= len(self._frames):
            return None
        img = self._frames[self._i]
        self._i += 1
        return Frame(img, self._i, 0)


class FakeAudio:
    name = "тест-звук"

    def __init__(self, samples: np.ndarray) -> None:
        self._samples = samples

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def read(self) -> Any:
        from harness.capture.base import AudioBlock

        return AudioBlock(self._samples, 48000, 0)


def _live(n: int = 40, rng: int = 0) -> list[np.ndarray]:
    """Кадры, похожие на настоящий экран: шум и различия между соседними."""
    r = np.random.default_rng(rng)
    return [r.integers(0, 256, (48, 64), dtype=np.uint8) for _ in range(n)]


def _stereo(n: int = 480) -> np.ndarray:
    r = np.random.default_rng(1)
    left = r.integers(-2000, 2000, n, dtype=np.int16)
    right = r.integers(-2000, 2000, n, dtype=np.int16)
    return np.stack([left, right], axis=1)


def test_selftest_passes_on_a_real_looking_capture(tmp_path: Path) -> None:
    res = selftest.run(tmp_path / "s", seconds=0.5,
                       source=FakeSource(_live()), audio_source=FakeAudio(_stereo()))
    assert res.ok, res.render_text()
    names = [s.name for s in res.steps]
    for must in ("кадры приходят", "кадры не чёрные", "кадры не одинаковые",
                 "каналы различаются", "actor_layer = human",
                 "воспроизведение работает"):
        assert must in names, f"шаг «{must}» не выполнялся"


def test_selftest_catches_black_frames(tmp_path: Path) -> None:
    black = [np.zeros((48, 64), dtype=np.uint8) for _ in range(40)]
    res = selftest.run(tmp_path / "s", seconds=0.5, source=FakeSource(black),
                       audio_source=FakeAudio(_stereo()))
    assert not res.ok
    failed = res.failed[0]
    assert failed.name == "кадры не чёрные"
    assert "Запись экрана" in failed.where or "разрешен" in failed.where


def test_selftest_catches_frozen_frames(tmp_path: Path) -> None:
    """Один и тот же буфер: формально кадры есть, фактически записан один."""
    one = np.random.default_rng(3).integers(0, 256, (48, 64), dtype=np.uint8)
    res = selftest.run(tmp_path / "s", seconds=0.5,
                       source=FakeSource([one.copy() for _ in range(40)]),
                       audio_source=FakeAudio(_stereo()))
    assert not res.ok
    assert res.failed[0].name == "кадры не одинаковые"
    assert "повторённый буфер" in res.failed[0].where
    assert "утверждает" in res.failed[0].where, (
        "отказ обязан отличать «источник соврал» от «экран не менялся»")


def test_selftest_catches_mono_dressed_as_stereo(tmp_path: Path) -> None:
    """Два одинаковых канала — это моно, и пеленг по ним не считается вовсе."""
    r = np.random.default_rng(2).integers(-2000, 2000, 480, dtype=np.int16)
    mono = np.stack([r, r], axis=1)
    res = selftest.run(tmp_path / "s", seconds=0.5, source=FakeSource(_live()),
                       audio_source=FakeAudio(mono))
    assert not res.ok
    assert res.failed[0].name == "каналы различаются"
    assert "микрофон" in res.failed[0].where


def test_selftest_says_where_not_just_that(tmp_path: Path) -> None:
    """У каждого провала есть место, а не только факт."""
    black = [np.zeros((48, 64), dtype=np.uint8) for _ in range(40)]
    res = selftest.run(tmp_path / "s", seconds=0.5, source=FakeSource(black),
                       audio_source=FakeAudio(_stereo()))
    text = res.render_text()
    assert "Чинить первым:" in text and "Где именно:" in text
    assert "Прошло" in text, "сводка обязана сказать, сколько прошло и сколько нет"
    with pytest.raises(ValueError, match="не сказал, где именно"):
        selftest.Step("шаг", False, "не вышло")


def test_selftest_marks_the_operator_as_the_actor(tmp_path: Path) -> None:
    """В живой записи действует человек, и в журнале это обязано быть видно."""
    from harness.session import Session as ReadSession

    root = tmp_path / "s"
    res = selftest.run(root, seconds=0.5, source=FakeSource(_live()),
                       audio_source=FakeAudio(_stereo()))
    assert res.ok
    with ReadSession(root) as sess:
        frames = list(sess.journal.frames())
        assert frames
        assert {str(x.actor_layer) for x in frames} == {"human"}


def test_selftest_without_audio_still_checks_frames(tmp_path: Path) -> None:
    """Звук записи не блокирует: без него проверка кадров и журнала остаётся."""
    res = selftest.run(tmp_path / "s", seconds=0.5, source=FakeSource(_live()),
                       with_audio=False)
    assert res.ok
    assert not any("канал" in s.name for s in res.steps)


# ---------------------------------------------------------------------------
# План записи: минимальный набор без игры
# ---------------------------------------------------------------------------


def test_minimal_set_needs_no_game() -> None:
    from harness.corpus.live import KINDS, SETS

    minimal = SETS["minimal"]
    assert "ничего ставить не надо" in minimal["needs"]
    # Проверяется указание оператору («как»), а не объяснение («зачем»): в
    # объяснении игра упоминается законно — там сказано, чем эта запись её заменяет.
    for name in minimal["kinds"]:
        how = KINDS[name]["how"].lower()
        for word in ("игр", "minecraft", "камеру мышью", "инвентар", "возрожд"):
            assert word not in how, (
                f"вид {name}: в указании «как» встречается «{word}», значит запись "
                "всё-таки требует игры")
    assert "play" not in minimal["kinds"] and "death" not in minimal["kinds"]


def test_plan_defaults_to_minimal_and_says_it_is_enough() -> None:
    from harness.corpus.live import plan_text

    text = plan_text()
    assert "минимальный" in text
    assert "Игра для этого не нужна" in text
    assert "первое сравнение синтетики с живым" in text
    # Порядок по умолчанию решает, начнёт ли оператор вообще.
    assert "Minecraft" not in text


def test_plan_commands_carry_numbers_not_placeholders() -> None:
    """В команде стоит число секунд, а не «N»: умножать в голове — место ошибки."""
    from harness.corpus.live import plan_text

    for line in plan_text().splitlines():
        if "harness record ~" in line:
            assert "--seconds " in line
            tail = line.split("--seconds ")[1].split()[0]
            assert tail.isdigit() and int(tail) >= 60, line


def test_full_set_still_exists() -> None:
    from harness.corpus.live import SETS, plan_text

    assert "play" in SETS["full"]["kinds"]
    assert "Minecraft" in plan_text("full")


def test_unknown_set_refuses() -> None:
    from harness.corpus.live import LiveError, plan_text

    with pytest.raises(LiveError, match="нет набора"):
        plan_text("никакого")


# ---------------------------------------------------------------------------
# Приём чужих записей: четыре условия части 6
# ---------------------------------------------------------------------------


def _foreign_session(root: Path, *, width: int = 96, height: int = 64,
                     fps: float = 24.0, wall_offset_s: float = -37 * 3600.0,
                     lineage: str = "чужая-линия-оператора") -> Path:
    """Запись, сделанную на другой машине, приходится изготовить: её негде взять.

    Отличается от контейнерной по всем четырём осям части 6 сразу: другое
    разрешение и частота, `wall_clock` из другого часового пояса и **идущий назад**,
    чужой `lineage_id`, и лежит она не там, где записывалась.
    """
    from harness.core.journal import Actor, ActorLayer
    from harness.core.profile import from_schema
    from harness.session import Recorder

    r = np.random.default_rng(7)
    profile = from_schema("ЧУЖАЯ-МАШИНА", capture_width=width,
                          capture_height=height, capture_fps=fps)
    made = root / "как-записали"
    with Recorder(made, profile=profile, source="screen_mss", synthetic=False,
                  note="запись оператора", lineage_id=lineage) as rec:
        for i in range(12):
            # Настенное время скачет назад и в чужой зоне: часы оператора никто не
            # синхронизировал, и это законно — wall_clock не часы, а привязка.
            rec.record_frame(r.integers(0, 256, (height, width), dtype=np.uint8),
                             t_world=i, actor=Actor.HUMAN,
                             actor_layer=ActorLayer.HUMAN,
                             wall_clock=1_700_000_000.0 + wall_offset_s - i * 0.5)
    # «Пути не совпадают с контейнерными»: каталог переносят, как флешку.
    moved = root / "как-доехало" / "flash" / "stillness"
    moved.parent.mkdir(parents=True, exist_ok=True)
    made.rename(moved)
    return moved


def test_ingest_accepts_a_recording_from_another_machine(tmp_path: Path) -> None:
    """Четыре условия части 6 разом: путь, часы, разрешение, линия."""
    from harness.corpus.live import ingest

    foreign = _foreign_session(tmp_path)
    got = ingest(foreign, tmp_path / "corpus", kind="stillness")
    assert got.frames == 12
    assert got.kind == "stillness"
    assert not got.synthetic
    assert got.actor_layers.get("human", 0) >= 12, (
        "записи оператора обязаны нести слой human: без него живой корпус "
        "неотличим по атрибуции от синтетического")


def test_ingest_does_not_care_about_wall_clock_going_backwards(tmp_path: Path) -> None:
    """Часы оператора не синхронизированы, и это не повод отказать.

    `wall_clock` — привязка для исследователя, а не четвёртые часы, и монотонность с
    него не спрашивается. Спрашивалась бы — ни одна запись с чужой машины не
    прошла бы, и причина выглядела бы как «запись испорчена».
    """
    from harness.corpus.live import ingest
    from harness.session import Session as ReadSession

    foreign = _foreign_session(tmp_path, wall_offset_s=+11 * 3600.0)
    got = ingest(foreign, tmp_path / "corpus", kind="stillness")
    with ReadSession(got.path) as s:
        walls = [e.wall_clock for e in s.journal.frames() if e.wall_clock]
        assert walls == sorted(walls, reverse=True), (
            "в этой записи настенное время нарочно идёт назад — иначе проверять "
            "нечего")
        # А три настоящих часа при этом монотонны, и это проверяет сам журнал.
        selves = [e.stamp.t_self for e in s.journal.frames()]
        assert selves == sorted(selves)


def test_ingest_accepts_another_resolution_and_frame_rate(tmp_path: Path) -> None:
    """Разрешение и частота у оператора свои, и подгонять их нельзя.

    Пересчёт кадров к синтетическому размеру был бы худшим из решений: он сделал бы
    живую запись похожей на синтетику ровно в том, чем она от неё отличается.
    """
    from harness.corpus.live import ingest
    from harness.session import Session as ReadSession

    foreign = _foreign_session(tmp_path, width=176, height=112, fps=17.0)
    got = ingest(foreign, tmp_path / "corpus", kind="stillness")
    with ReadSession(got.path) as s:
        image = s.image(0)
        assert image.shape == (112, 176), (
            f"кадр приехал размером {image.shape}: ingest что-то пересчитал")
        meta = json.loads((got.path / "session.json").read_text(encoding="utf-8"))
        assert float(meta["profile"]["parameters"]["capture_fps"]) == 17.0


def test_ingest_keeps_the_foreign_lineage(tmp_path: Path) -> None:
    """Машина оператора не имеет отношения к линии контейнера.

    Подмена линии на свою означала бы, что чужой опыт записан как собственный, а
    это ровно то, что запрещает манифест непереносимого.
    """
    from harness.corpus.live import ingest

    foreign = _foreign_session(tmp_path, lineage="линия-с-ноутбука-оператора")
    got = ingest(foreign, tmp_path / "corpus", kind="stillness")
    branch = sorted((got.path / "journal" / "branches").glob("*"))[-1]
    meta = json.loads((branch / "branch.json").read_text(encoding="utf-8"))
    assert meta["lineage_id"] == "линия-с-ноутбука-оператора"


def test_live_and_synthetic_are_not_mixed_in_measurements(tmp_path: Path) -> None:
    """Смешать — значит потерять то сравнение, ради которого всё делается."""
    from harness.corpus.live import ingest
    from harness.session import Session as ReadSession

    foreign = _foreign_session(tmp_path)
    got = ingest(foreign, tmp_path / "corpus", kind="stillness")
    # Отдельный источник виден в самой записи, а не выводится из имени каталога.
    with ReadSession(got.path) as s:
        assert s.meta.synthetic is False
        assert s.meta.source == "screen_mss"
    assert got.path.parent.name == "corpus", "живой корпус лежит своим каталогом"
    assert got.path.name.startswith("stillness-"), (
        "вид записи виден в имени: по нему замер выбирает, что с чем сравнивать")


def test_ingest_refuses_a_synthetic_recording_dressed_as_live(tmp_path: Path) -> None:
    """Иначе первое живое число окажется синтетическим, и об этом никто не узнает."""
    from harness.core.profile import MILESTONE_0
    from harness.corpus.live import LiveError, ingest
    from harness.session import Recorder

    fake = tmp_path / "fake"
    r = np.random.default_rng(1)
    with Recorder(fake, profile=MILESTONE_0, source="synthetic",
                  synthetic=True) as rec:
        rec.record_frame(r.integers(0, 256, (64, 96), dtype=np.uint8), t_world=0)
    with pytest.raises(LiveError, match="синтетической"):
        ingest(fake, tmp_path / "corpus", kind="stillness")


# ---------------------------------------------------------------------------
# SETUP.md: обещания документа проверяются, а не принимаются на слово
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def test_setup_names_every_command_it_promises() -> None:
    """Каждая команда из SETUP.md существует в CLI.

    Документ, обещающий несуществующую команду, хуже отсутствующего: оператор
    вводит её, получает ошибку и идёт читать код — то есть делает ровно то, чего
    этот файл должен избавить.
    """
    import re

    text = (ROOT / "SETUP.md").read_text(encoding="utf-8")
    from harness.cli import build_parser

    known = set()
    for action in build_parser()._actions:
        if getattr(action, "choices", None) and isinstance(action.choices, dict):
            known |= set(action.choices)
    assert known, "не удалось получить список подкоманд"

    used = set(re.findall(r"harness ([a-z-]+)", text))
    unknown = used - known - {"live"}          # «harness-live» — путь, не команда
    assert not unknown, f"SETUP.md обещает команды, которых нет: {sorted(unknown)}"


def test_setup_installs_extras_that_exist() -> None:
    """Секции extras, на которые ссылается SETUP.md, объявлены в pyproject."""
    import re

    setup = (ROOT / "SETUP.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    declared = set(re.findall(r"^(\w+) = \[", pyproject, re.MULTILINE))
    for extras in set(re.findall(r"pip install -e ['\"]?\.\[([a-z,]+)\]", setup)):
        for name in extras.split(","):
            assert name in declared, (
                f"SETUP.md предлагает extras [{name}], которых нет в pyproject")


def test_setup_names_the_branch_because_main_is_empty() -> None:
    """Клон без имени ветки даёт пустой каталог — это надо сказать громко."""
    text = (ROOT / "SETUP.md").read_text(encoding="utf-8")
    assert "claude/save-claude-design-draft-ul62eo" in text
    assert "git clone -b" in text, "команда клонирования без -b даст пустой каталог"
    assert "только README" in text or "только файл README" in text


def test_setup_has_a_section_per_os_and_splits_wayland() -> None:
    text = (ROOT / "SETUP.md").read_text(encoding="utf-8")
    for must in ("### Linux", "### macOS", "### Windows"):
        assert must in text, f"нет раздела {must}"
    assert "Wayland" in text and "X11" in text
    for symptom in ("Чёрный кадр на macOS", "Wayland", "Моно вместо стерео",
                    "ffmpeg не в PATH"):
        assert symptom in text, f"в «что пойдёт не так» нет пункта про {symptom}"


def test_ingest_does_not_call_an_empty_index_a_sound(tmp_path: Path) -> None:
    """Наличие файла — не наличие звука.

    `AudioStore` создаёт индекс при открытии сессии, поэтому пустой файл есть
    всегда. Проверка на существование отвечала «звук есть» на записи без единого
    блока, и по такой записи потом искали бы пеленг, которого в ней нет.
    """
    from harness.corpus.live import ingest

    foreign = _foreign_session(tmp_path)
    assert (foreign / "audio" / "index.jsonl").exists(), (
        "пустой индекс обязан существовать — иначе проверять нечего")
    assert (foreign / "audio" / "index.jsonl").stat().st_size == 0
    got = ingest(foreign, tmp_path / "corpus", kind="stillness")
    assert not got.has_audio
    assert any("звука нет" in w for w in got.warnings)


def test_a_recording_carries_a_real_time_anchor(tmp_path: Path) -> None:
    """Настенное время ставит сам журнал, и живой записи оно необходимо.

    Три журнальных часа считают циклы, а не дату: без якоря нельзя сказать, когда
    запись сделана. На чужой машине это единственная привязка к реальному времени.
    """
    import time

    from harness.core.profile import MILESTONE_0
    from harness.session import Recorder, Session as ReadSession

    r = np.random.default_rng(4)
    before = time.time()
    root = tmp_path / "live"
    frame = r.integers(0, 256, (64, 96), dtype=np.uint8)
    # Профиль строится по кадру: живая запись с профилем, заявляющим другой размер,
    # врёт о себе — на этом сломался разбор расхода места (TASK-09, часть 2).
    with Recorder(root, profile=MILESTONE_0.for_frame(frame), source="screen_mss",
                  synthetic=False) as rec:
        rec.record_frame(frame, t_world=0)
    with ReadSession(root) as s:
        walls = [e.wall_clock for e in s.journal.frames()]
        assert walls and walls[0] >= before - 1


def test_a_forged_clock_survives_the_trip(tmp_path: Path) -> None:
    """Чужие часы записываются как есть и не подменяются местными при приёме.

    Подмена сделала бы запись с ноутбука оператора выглядящей как сделанная здесь,
    и восстановить настоящий момент было бы уже нечем.
    """
    from harness.corpus.live import ingest
    from harness.session import Session as ReadSession

    foreign = _foreign_session(tmp_path, wall_offset_s=-37 * 3600.0)
    got = ingest(foreign, tmp_path / "corpus", kind="stillness")
    with ReadSession(got.path) as s:
        walls = [e.wall_clock for e in s.journal.frames()]
        assert walls, "часов в записи нет вовсе"
        assert all(w < 1_700_000_000.0 for w in walls), (
            f"часы подменены местными: {walls[:2]}")


def test_setup_gives_windows_commands_for_cmd_not_only_powershell() -> None:
    """Оператор пришёл в `cmd.exe`, а файл давал `ls` и только PowerShell.

    Это случилось на самом деле, на первой же строке: `ls` в командной строке
    отвечает «не является внутренней или внешней командой», а `Activate.ps1` там не
    запускается. Дальше человек остановился. Поэтому проверяется механически: для
    Windows названы обе оболочки и их **разные** файлы активации.
    """
    text = (ROOT / "SETUP.md").read_text(encoding="utf-8")
    assert "cmd.exe" in text
    assert "activate.bat" in text and "Activate.ps1" in text
    assert "dir" in text, "список каталога в cmd.exe — dir, и это надо сказать"
    assert "не является внутренней или внешней командой" in text, (
        "симптом, на котором оператор встал, обязан быть в разделе «что пойдёт "
        "не так» — дословно, чтобы находился поиском по тексту ошибки")


def test_setup_warns_against_system_directories() -> None:
    """Клон в System32 удаётся из окна администратора, а окружение там не живёт."""
    text = (ROOT / "SETUP.md").read_text(encoding="utf-8")
    assert "System32" in text
    assert "move C:\\Windows\\System32\\harness" in text, (
        "мало сказать «не туда»: нужна команда, которая переносит уже склонированное")


def test_setup_puts_the_code_in_the_home_directory() -> None:
    """Каталог в корне `C:\\`, созданный администратором, закрыт на запись.

    Это случилось: клон прошёл, `dir` показал все файлы, и `py -m venv` упал с
    «Отказано в доступе». Читать можно, писать нельзя — и выглядит это как поломка
    Python, а не как права каталога. Поэтому файл обязан вести в домашнюю папку и
    обязан сказать, что переносить бесполезно: права переедут вместе с каталогом.
    """
    text = (ROOT / "SETUP.md").read_text(encoding="utf-8")
    assert "%USERPROFILE%" in text
    assert "не в корень диска" in text
    assert "WinError 5" in text, (
        "симптом обязан быть дословно: его ищут поиском по тексту ошибки")
    assert "harness.egg-info" in text, (
        "второе сообщение о той же причине сбивает с толку и должно быть названо")
    assert "не переносите" in text.lower(), (
        "перенос выглядит очевидным решением и не работает")


# ---------------------------------------------------------------------------
# Ловушка `None`: «экран не менялся» — это данные, а не пропуск (TASK-08)
# ---------------------------------------------------------------------------


class StaticScreen:
    """Источник, ведущий себя как Desktop Duplication на неподвижном экране.

    Первый `read()` отдаёт кадр, остальные — `UNCHANGED`. Именно так работает
    `dxcam.grab()`: неизменённый кадр не выдаётся вовсе.
    """

    name = "тест-dxcam-статика"

    def __init__(self, turns: int = 40) -> None:
        self._turns = turns
        self._i = 0

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def read(self) -> Any:
        from harness.capture.base import UNCHANGED

        self._i += 1
        if self._i > self._turns:
            return None
        if self._i == 1:
            return Frame(np.random.default_rng(2).integers(
                0, 256, (48, 64), dtype=np.uint8), 0, 0)
        return UNCHANGED


def test_unchanged_is_not_none_and_is_falsy() -> None:
    """`None` — «кадра не пришло», `UNCHANGED` — «кадр тот же». Разные ответы."""
    from harness.capture.base import UNCHANGED, Unchanged

    assert UNCHANGED is not None
    assert isinstance(UNCHANGED, Unchanged)
    assert not UNCHANGED, (
        "ложное значение нарочно: `if frame:` без явной проверки обрабатывает его "
        "как «кадра нет» — это безопасная сторона ошибки")


def test_stillness_recording_does_not_look_like_lost_frames(tmp_path: Path) -> None:
    """Главная ловушка задачи, целиком.

    Запись «неподвижность, 60 с» статична по замыслу. Наивный счётчик отрапортовал бы
    «потеряно 1700 кадров из 1800» на записи, прошедшей идеально. Проверяется, что
    отметки «без изменений» попали в журнал как данные и что ни одного разрыва не
    записано.
    """
    from harness.session import Session as ReadSession

    res = selftest.run(tmp_path / "s", seconds=0.5, with_audio=False,
                       expect_change=False, source=StaticScreen(turns=20))
    assert res.ok, res.render_text()

    with ReadSession(tmp_path / "s") as s:
        frames = list(s.journal.frames())
        codes = [(e.event or {}).get("code") for e in frames]
        assert codes.count("unchanged") >= 10, (
            f"отметок «без изменений» {codes.count('unchanged')}: статика должна "
            "давать их, а не разрывы")
        assert codes.count("frame") == 1, "изменившийся кадр здесь ровно один"
        gaps = [e for e in s.journal if str(e.kind) == "capture_gap"]
        assert not gaps, (
            f"записано {len(gaps)} разрывов на записи, где ничего не потеряно: "
            "именно это и есть та ошибка, из-за которой опорный замер сломался бы "
            "первым")


def test_unchanged_reuses_the_previous_frame_reference(tmp_path: Path) -> None:
    """Ссылка на прежний сегмент; содержимое не дублируется."""
    from harness.core.journal import Actor, ActorLayer
    from harness.core.profile import MILESTONE_0
    from harness.session import Recorder, Session as ReadSession

    root = tmp_path / "s"
    frame = np.random.default_rng(3).integers(0, 256, (48, 64), dtype=np.uint8)
    with Recorder(root, profile=MILESTONE_0.for_frame(frame), source="screen_dxcam",
                  synthetic=False) as rec:
        rec.record_frame(frame, actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN)
        for _ in range(4):
            rec.record_unchanged(actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN)

    with ReadSession(root) as s:
        frames = list(s.journal.frames())
        refs = {(e.frame or {}).get("digest") if isinstance(e.frame, dict)
                else getattr(e.frame, "digest", None) for e in frames}
        assert len(refs) == 1, (
            f"ссылок {len(refs)}: у отметки «без изменений» она обязана быть та же, "
            "что у предыдущего кадра")
        # И часы при этом идут: момент времени у каждой отметки свой.
        selves = [e.stamp.t_self for e in frames]
        assert selves == sorted(selves) and len(set(selves)) == len(selves)


def test_unchanged_before_the_first_frame_is_a_defect(tmp_path: Path) -> None:
    """До первого кадра ссылаться не на что, и это поломка, а не статика.

    Первый кадр Desktop Duplication отдаёт всегда, даже на неподвижном экране.
    Поэтому «без изменений» на первом обороте означает неисправный механизм, и
    молчаливо превращать это в отметку нельзя.
    """
    from harness.core.profile import MILESTONE_0
    from harness.session import Recorder, SessionError

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="screen_dxcam",
                  synthetic=False) as rec:
        with pytest.raises(SessionError, match="до первого кадра"):
            rec.record_unchanged()


def test_frequency_is_two_numbers(tmp_path: Path) -> None:
    """Обороты цикла и изменившиеся кадры — разные величины.

    На статичном экране второе законно равно нулю, и отказом это не считается.
    """
    res = selftest.run(tmp_path / "s", seconds=0.5, with_audio=False,
                       expect_change=False, source=StaticScreen(turns=30))
    names = [s.name for s in res.steps]
    assert "частота оборотов не ниже заявленной" in names
    assert "изменившиеся кадры считаются отдельно" in names
    changed = next(s for s in res.steps
                   if s.name == "изменившиеся кадры считаются отдельно")
    assert changed.ok, "малое число изменившихся кадров на статике — не отказ"


def test_identity_check_depends_on_the_kind_of_scene(tmp_path: Path) -> None:
    """На записи неподвижности кадры обязаны быть одинаковыми."""
    # Один кадр, размноженный копиями: если поправить только первый, он станет
    # отличаться от остальных, и «все одинаковы» окажется неверным по построению.
    base = np.full((32, 32), 90, dtype=np.uint8)
    base[0, 0] = 255                         # чтобы кадр не был однородным
    same = [base.copy() for _ in range(12)]

    static = selftest.run(tmp_path / "static", seconds=0.3, with_audio=False,
                          expect_change=False, source=FakeSource(list(same)))
    step = next(s for s in static.steps if "одинаков" in s.name)
    assert step.ok, "на статичной сцене совпадение кадров — правильный ответ"

    moving = selftest.run(tmp_path / "moving", seconds=0.3, with_audio=False,
                          expect_change=True, source=FakeSource(list(same)))
    step = next(s for s in moving.steps if "одинаков" in s.name)
    assert not step.ok, "там, где ждали движения, совпадение кадров — отказ"


def test_setup_has_the_three_symptoms_from_the_first_live_run() -> None:
    """`TASK-08`, часть 4: три симптома, на которых оператор стоял на самом деле."""
    text = (ROOT / "SETUP.md").read_text(encoding="utf-8")
    assert "WinError 5" in text
    assert "Permission denied" in text and "python.exe" in text, (
        "повторное `py -m venv` при активном окружении падает на python.exe, и это "
        "не поломка: окружение уже есть")
    assert "не является внутренней или внешней командой" in text
    # И для последнего названы **обе** причины: их две, и различаются они по (.venv).
    assert "Установка не прошла" in text and "не активировано" in text


def test_doctor_says_which_mechanism_and_why() -> None:
    """Строка «есть» без причины выбора — дефект вывода, а не мелочь оформления."""
    check = doctor.check_session(detect())
    if check.state is doctor.State.YES:
        assert ":" in check.detail, "не сказано, почему выбран именно этот механизм"
    text = (ROOT / "SETUP.md").read_text(encoding="utf-8")
    assert "screen_dxcam" in text and "screen_mss" in text, (
        "оператор должен знать имена механизмов: доктор печатает именно их")


def test_storage_estimate_names_both_ends() -> None:
    """Одного числа расхода не существует: разброс тысячекратный по содержимому."""
    check = doctor.check_disk(detect(), path=Path("."), fps=30.0, size=(1920, 1080))
    assert "замера не было" in check.detail, (
        "без замера оценка обязана называться оценкой, а не выдаваться за факт")
    assert "несжимаемом" in check.detail and "экранном" in check.detail, (
        "оба края обязаны быть названы: разница между 0.37 и 209 ГиБ/ч — это "
        "разница между неделей и двадцатью минутами")
    # Константа — живое число, и вилка обязана накрывать действительность: 3.6 КиБ
    # были замером на нарисованном экране и не попадали в неё в четырнадцать раз.
    assert 30 * 1024 <= doctor.BYTES_PER_FRAME_1080P <= 80 * 1024, (
        f"{doctor.BYTES_PER_FRAME_1080P / 1024:.1f} КиБ на кадр: живой замер дал "
        "49.3, синтетика, похожая на рабочий стол — 65.8. Оценка вне этого промежутка "
        "снова не накроет обычный экран")


def test_doctor_prefers_the_measurement_over_the_estimate(tmp_path: Path) -> None:
    """`TASK-09`, часть 2, пункт 2: после selftest доктор берёт измеренное.

    Расчёт по константе разошёлся с действительностью в 25 раз в TASK-08 и в 14 в
    TASK-09. Замер на этой машине точнее любой константы по определению.
    """
    from harness.selftest import Measured

    measured = Measured(bytes_per_frame=50483.0, frame_w=1920, frame_h=1080,
                        fps=30.0, compression=41.1, mechanism="screen_dxcam",
                        entries=300, at_unix=1_700_000_000.0)
    check = doctor.check_disk(detect(), path=tmp_path, fps=30.0,
                              size=(1920, 1080), measured=measured)
    assert "по замеру" in check.detail
    assert "41×" in check.detail, "степень сжатия обязана быть видна"
    assert "screen_dxcam" in check.detail, "чем мерили — часть числа"
    assert "замера не было" not in check.detail


def test_a_measurement_survives_a_restart(tmp_path: Path) -> None:
    """Замер хранится с отметкой времени, содержимого и механизма."""
    from harness.selftest import Measured

    one = Measured(1000.0, 640, 480, 30.0, 12.0, "screen_mss", 90, 1_700_000_000.0)
    one.save(tmp_path)
    back = Measured.load(tmp_path)
    assert back is not None
    assert back.as_dict() == one.as_dict()
    # Испорченный файл — это отсутствие замера, а не замер.
    (tmp_path / "storage-measured.json").write_text("{не json", encoding="utf-8")
    assert Measured.load(tmp_path) is None
    assert Measured.load(tmp_path / "нет-такого") is None


def test_selftest_runs_everything_that_does_not_depend_on_the_failure(tmp_path: Path) -> None:
    """`TASK-08`, часть 3a, дефект 1: останов на первом отказе стоил захода.

    Различие кадров, звук, журнал, воспроизведение и расход места от частоты захвата
    не зависят. Здесь частота нарочно провалена — источник отдаёт мало кадров, — и
    все независимые проверки обязаны быть выполнены.
    """
    class Slow:
        name = "тест-медленный"

        def __init__(self) -> None:
            self._i = 0

        def start(self) -> None: ...
        def stop(self) -> None: ...

        def read(self) -> Any:
            import time as _t

            self._i += 1
            _t.sleep(0.05)              # 20 оборотов в секунду вместо тридцати
            return Frame(np.random.default_rng(self._i).integers(
                0, 256, (32, 48), dtype=np.uint8), self._i, 0)

    res = selftest.run(tmp_path / "s", seconds=0.6, with_audio=False, source=Slow())
    names = [s.name for s in res.steps]
    rate = next(s for s in res.steps if "частота оборотов" in s.name)
    assert not rate.ok, "частота должна была провалиться — иначе тест ничего не ловит"
    for must in ("кадры не чёрные", "кадры не одинаковые",
                 "записи журнала создаются", "actor_layer = human",
                 "воспроизведение работает", "расход места измерен"):
        assert must in names, (
            f"после провала частоты не выполнено «{must}», хотя оно от неё не зависит")
    assert res.summary().startswith("Прошло ")


def test_rate_failure_blames_the_mechanism_before_the_operator(tmp_path: Path) -> None:
    """`TASK-08`, часть 3a, дефект 2: совет не должен опережать причину в коде.

    Оператору напечатали «закройте тяжёлые окна или уменьшите capture_fps», тогда как
    дело было в выбранном механизме: он получил бы те же двадцать кадров.
    """
    import harness.capture.select as select_mod

    class Slow:
        name = "screen_mss"

        def __init__(self) -> None:
            self._i = 0

        def start(self) -> None: ...
        def stop(self) -> None: ...

        def read(self) -> Any:
            import time as _t

            self._i += 1
            _t.sleep(0.05)
            return Frame(np.random.default_rng(self._i).integers(
                0, 256, (32, 48), dtype=np.uint8), self._i, 0)

    real = select_mod.open_screen
    try:
        select_mod.open_screen = lambda m, **kw: select_mod.Choice(  # type: ignore[assignment]
            chosen=select_mod.MSS, considered=(select_mod.DXCAM, select_mod.MSS),
            rejected=[("screen_dxcam", "нет пакета dxcam")], source=Slow())
        res = selftest.run(tmp_path / "s", seconds=0.8, with_audio=False)
    finally:
        select_mod.open_screen = real                                # type: ignore[assignment]

    rate = next(s for s in res.steps if "частота оборотов" in s.name)
    assert not rate.ok
    where = rate.where
    assert "механизм" in where, "не названа возможная причина в коде"
    assert where.index("механизм") < where.index("окна"), (
        "совет оператору стоит раньше причины в коде: именно так оператор и потратил "
        "заход впустую")
    assert "harness doctor" in where


# ---------------------------------------------------------------------------
# TASK-09: находки первого живого selftest
# ---------------------------------------------------------------------------


class Mic:
    """Петлевой вход, у которого блоки накапливаются во времени.

    Именно так ведёт себя настоящий: блоков в секунду больше, чем кадров, и читать
    надо всё накопленное, а не по одному на кадр.
    """

    name = "тест-звук"

    def __init__(self, rate: int = 48000, block_ms: float = 20.0,
                 mono: bool = False) -> None:
        self.rate = rate
        self.block = int(rate * block_ms / 1000)
        self.mono = mono
        self._t0: float | None = None
        self._given = 0

    def start(self) -> None:
        import time as _t

        self._t0 = _t.monotonic()

    def stop(self) -> None: ...

    @property
    def pending(self) -> int:
        import time as _t

        if self._t0 is None:
            return 0
        return max(0, int((_t.monotonic() - self._t0) * self.rate) - self._given)

    def read(self) -> Any:
        import time as _t

        from harness.capture.base import AudioBlock

        r = np.random.default_rng(self._given)
        n = self.block
        self._given += n
        left = r.integers(-999, 999, n, dtype=np.int16)
        right = left if self.mono else r.integers(-999, 999, n, dtype=np.int16)
        return AudioBlock(np.stack([left, right], axis=1), self.rate,
                          _t.monotonic_ns())


class Screen:
    name = "тест-экран"

    def __init__(self, pause: float = 0.01) -> None:
        self._i = 0
        self._pause = pause

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def read(self) -> Any:
        import time as _t

        self._i += 1
        _t.sleep(self._pause)
        return Frame(np.random.default_rng(self._i).integers(
            0, 256, (48, 64), dtype=np.uint8), self._i, _t.monotonic_ns())


def test_audio_is_measured_by_duration_not_by_fact(tmp_path: Path) -> None:
    """`TASK-09`, часть 1: звук пишется всю сессию, и проверка мерит длительность.

    Прежняя проверка читала **один** блок и печатала «звук пишется: 960 отсчётов,
    2 канала» на сессии длиной 6.7 с, где блоков должно быть около 335. Проверка
    подтверждала факт и молчала о непрерывности — то есть проверяла не то.
    """
    res = selftest.run(tmp_path / "s", seconds=1.0, source=Screen(),
                       audio_source=Mic())
    names = [s.name for s in res.steps]
    assert "покрытие звука" in names, "непрерывность не проверяется"
    wrote = next(s for s in res.steps if s.name == "звук пишется")
    assert "блоков" in wrote.detail and "с при" in wrote.detail, (
        f"в строке про звук нет ни числа блоков, ни секунд: {wrote.detail!r}")
    assert "1 блоков" not in wrote.detail, "снова прочитан один блок"
    cover = next(s for s in res.steps if s.name == "покрытие звука")
    assert cover.ok and "%" in cover.detail


def test_audio_sync_tolerance_is_actually_checked(tmp_path: Path) -> None:
    """`audio_sync_tolerance_ms` был в профиле и не проверялся ни одной строкой."""
    res = selftest.run(tmp_path / "s", seconds=0.6, source=Screen(),
                       audio_source=Mic())
    step = next(s for s in res.steps if "сошёлся с кадрами" in s.name)
    assert "мс при допуске" in step.detail
    from harness.core.profile import MILESTONE_0

    assert f"{MILESTONE_0.parameters['audio_sync_tolerance_ms']:g}" in step.detail, (
        "допуск обязан быть взят из профиля, а не написан в коде")


def test_mono_folded_by_the_driver_is_caught(tmp_path: Path) -> None:
    """Стерео микшер Realtek умеет складывать каналы в моно на уровне драйвера."""
    res = selftest.run(tmp_path / "s", seconds=0.5, source=Screen(),
                       audio_source=Mic(mono=True))
    step = next(s for s in res.steps if s.name == "каналы различаются")
    assert not step.ok
    assert "Realtek" in step.where, "самая частая причина обязана быть названа"


def test_compression_ratio_is_printed_so_the_number_can_be_checked(tmp_path: Path) -> None:
    """`TASK-09`, часть 2, пункт 4: «49.3 КиБ на запись» нечем было поверить.

    Первый разбор поделил это на размер кадра **из профиля** — 320×180 — и получил
    «сжатие не работает». Делить надо было на 1920×1080, и сжатие работает в сорок
    один раз. Степень сжатия и настоящий размер кадра печатаются рядом с расходом.
    """
    res = selftest.run(tmp_path / "s", seconds=0.4, with_audio=False,
                       source=Screen())
    step = next(s for s in res.steps if s.name == "расход места измерен")
    assert "при кадре 64×48" in step.detail, (
        f"настоящий размер кадра не назван: {step.detail!r}")
    assert "сырой" in step.detail and "сжатие" in step.detail


def test_a_live_profile_cannot_lie_about_the_frame_size(tmp_path: Path) -> None:
    """Профиль живой записи обязан совпадать с кадром — иначе запись врёт о себе."""
    from harness.core.profile import MILESTONE_0
    from harness.session import Recorder, SessionError

    big = np.random.default_rng(1).integers(0, 256, (1080, 1920), dtype=np.uint8)
    with Recorder(tmp_path / "lie", profile=MILESTONE_0, source="screen_dxcam",
                  synthetic=False) as rec:
        with pytest.raises(SessionError, match="профиль записи заявляет"):
            rec.record_frame(big)

    # А с профилем по кадру — записывается.
    with Recorder(tmp_path / "true", profile=MILESTONE_0.for_frame(big),
                  source="screen_dxcam", synthetic=False) as rec:
        assert rec.record_frame(big) is not None

    # У синтетики размер задаёт профиль, и миры его читают: там проверка не мешает.
    with Recorder(tmp_path / "synth", profile=MILESTONE_0, source="synthetic",
                  synthetic=True) as rec:
        assert rec.record_frame(big) is not None


def test_empty_lineage_is_impossible_by_construction() -> None:
    """`TASK-09`, часть 3: пустая строка проходит любую проверку на присутствие."""
    from harness.core.journal import BranchMeta, JournalError

    def meta(lineage: str | None) -> BranchMeta:
        return BranchMeta("000-x", "h", {}, None, None, None, "тест",
                          lineage_id=lineage)

    with pytest.raises(JournalError, match="пустой строкой"):
        meta("")
    with pytest.raises(JournalError, match="пустой строкой"):
        meta("   ")
    # `None` законен и означает «линии нет, это пробный прогон».
    assert meta(None).lineage_id is None
    assert meta("LIN_1").lineage_id == "LIN_1"


def test_a_recording_writes_null_not_an_empty_string(tmp_path: Path) -> None:
    from harness.core.profile import MILESTONE_0
    from harness.session import Recorder

    frame = np.zeros((8, 8), dtype=np.uint8)
    with Recorder(tmp_path / "s", profile=MILESTONE_0.for_frame(frame),
                  source="тест", synthetic=False) as rec:
        rec.record_frame(frame)
    branch = sorted((tmp_path / "s" / "journal" / "branches").glob("*"))[-1]
    raw = json.loads((branch / "branch.json").read_text(encoding="utf-8"))
    assert raw["lineage_id"] is None, (
        f"в записи {raw['lineage_id']!r}: пустая строка тихо сольёт разные линии")
    # И ни одного другого пустого поля.
    empties = [k for k, v in raw.items() if v == ""]
    assert not empties, f"пустые строки в branch.json: {empties}"


def test_old_records_with_an_empty_lineage_still_open(tmp_path: Path) -> None:
    """Записи с пустой линией уже существуют, и отказ их открыть терял бы данные."""
    from harness.core.journal import BranchMeta

    meta = BranchMeta.from_dict({"branch_id": "000-x", "structure_hash": "h",
                                 "profile": {}, "lineage_id": ""})
    assert meta.lineage_id is None, "пустая строка читается как «не объявлена»"


def test_synthetic_world_settings_are_marked_not_removed() -> None:
    """`TASK-09`, часть 4: решение — оставить в профиле и подписать применимость."""
    from harness.core.profile import MILESTONE_0
    from harness.core.settings import SYNTHETIC_ONLY, applies_to_live

    assert "world_bounded" in SYNTHETIC_ONLY
    assert not applies_to_live("world_bounded")
    assert applies_to_live("capture_fps")

    a = MILESTONE_0
    b = a.with_structural(world_bounded=True).with_parameters(capture_fps=25.0)
    keys = {d["key"]: d["live"] for d in a.diff(b)}
    assert keys == {"world_bounded": False, "capture_fps": True}
    assert [d["key"] for d in a.live_diff(b)] == ["capture_fps"], (
        "различие, которое не могло повлиять на живую запись, обязано быть отделимо")
    # Но из профиля не выброшено: запись остаётся самоописывающей.
    assert "world_bounded" in MILESTONE_0.structural
