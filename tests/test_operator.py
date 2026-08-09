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
    from harness.machine import CAPTURE_FOR

    assert CAPTURE_FOR[Session.WAYLAND] is None
    assert CAPTURE_FOR[Session.X11] == "screen_mss"
    # Windows и macOS тоже умеют mss: раньше их запирала не машинерия, а проверка
    # переменной DISPLAY, которой на этих системах не бывает вовсе.
    assert CAPTURE_FOR[Session.WINDOWS] == "screen_mss"
    assert CAPTURE_FOR[Session.MACOS] == "screen_mss"


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
    """Прогнать проверку содержимого через подменённый источник."""
    import harness.capture.screen as screen_mod

    real = screen_mod.ScreenCapture
    try:
        screen_mod.ScreenCapture = lambda **_: source     # type: ignore[assignment]
        return doctor.probe_screen(detect(), pause_s=0.0)
    finally:
        screen_mod.ScreenCapture = real                   # type: ignore[assignment]


def test_free_space_is_reported_in_hours_not_gigabytes(tmp_path: Path) -> None:
    """Гигабайты оператору ничего не говорят: чтобы понять, хватит ли, нужен код."""
    check = doctor.check_disk(detect(), path=tmp_path, fps=30.0, size=(1920, 1080))
    assert "ч записи" in check.detail
    assert "кадр/с" in check.detail


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
    assert "один и тот же буфер" in res.failed[0].where


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
    assert "Сломалось:" in text and "Где именно:" in text
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
    with Recorder(root, profile=MILESTONE_0, source="screen_mss",
                  synthetic=False) as rec:
        rec.record_frame(r.integers(0, 256, (64, 96), dtype=np.uint8), t_world=0)
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
