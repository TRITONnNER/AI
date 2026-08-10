"""Ход записи и прерывание записи. `TASK-16`.

Почему это тесты, а не «проверено глазами на машине оператора»: у проекта нет живого
корпуса **именно** потому, что запись молчала, и оператор дважды прервал её, приняв за
зависшую. Значит и ход записи, и поведение при `Ctrl+C` — часть механизма, и обязаны
проверяться там, где дисплея нет: на поддельном источнике кадров.

Три утверждения проверяются раздельно, и смешивать их нельзя:

1. **строка хода идёт и обновляется** — не чаще, чем велит профиль;
2. **`Ctrl+C` оставляет годную запись** с отметкой «прервано на N из M» — в любой момент,
   включая момент внутри сжатия кадра, где оператора и застало прерывание;
3. **повторный запуск в тот же каталог** объясняет, что там лежит, и даёт команду — а не
   отказывает по непустому каталогу.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.capture.base import UNCHANGED, Frame
from harness.capture.record_loop import Turns, run_turns
from harness.core.journal import Actor, ActorLayer, Kind
from harness.core.profile import MILESTONE_0
from harness.progress import (CHANNELS, FILE_KINDS, Progress, Sample, channel_for,
                              render_final, render_line)
from harness.session import INTERRUPTED, Recorder, Session, describe_existing, next_free_path


# --- поддельный экран --------------------------------------------------------


class FakeScreen:
    """Источник кадров без дисплея. Отдаёт кадры, статику и прерывание по расписанию.

    Не «пустышка, делающая вид, что захватывает»: она не отвечает на вопрос «работает ли
    захват» вовсе. Она отвечает на другой вопрос — что делает **цикл записи**, когда
    экран отвечает так или иначе, — и это единственный способ проверить его без дисплея.
    """

    name = "тест-экран"

    def __init__(self, *, w: int = 64, h: int = 48, still_every: int = 0,
                 interrupt_at: int | None = None, ends_at: int | None = None,
                 delay_s: float = 0.0) -> None:
        self.w, self.h = w, h
        self.still_every = still_every
        self.interrupt_at = interrupt_at
        self.ends_at = ends_at
        # Задержка на оборот нужна ровно одному тесту: чтобы прогон целой команды занял
        # секунды и **настоящий** ограничитель частоты успел сработать больше одного раза.
        # Без неё 300 оборотов проходят за миллисекунды, и «строка обновляется» проверялось
        # бы на единственной строке.
        self.delay_s = float(delay_s)
        self.reads = 0

    def first(self) -> Frame:
        return Frame(np.full((self.h, self.w), 7, dtype=np.uint8), 0, 0)

    def read(self):
        self.reads += 1
        if self.delay_s:
            import time

            time.sleep(self.delay_s)
        if self.interrupt_at is not None and self.reads >= self.interrupt_at:
            raise KeyboardInterrupt
        if self.ends_at is not None and self.reads >= self.ends_at:
            return None
        if self.still_every and self.reads % self.still_every == 0:
            return UNCHANGED
        img = np.full((self.h, self.w), (self.reads * 13) % 251, dtype=np.uint8)
        return Frame(img, self.reads, 0)

    def stop(self) -> None: ...


class FakeClock:
    """Часы, которыми управляет тест. Настоящие сделали бы проверку частоты гадательной."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _profile(w: int = 64, h: int = 48):
    from harness.core.profile import from_schema

    return from_schema("ТЕСТ-запись", capture_width=w, capture_height=h)


def _record(path: Path, *, screen: FakeScreen, frames: int, prog: Progress | None = None):
    p = _profile(screen.w, screen.h)
    with Recorder(path, profile=p, source=screen.name, synthetic=False,
                  lineage_id="lin-test") as rec:
        return run_turns(rec, source=screen, frames=frames, first=screen.first(),
                         actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN,
                         progress=prog)


# --- 1. строка хода ----------------------------------------------------------


def test_progress_line_carries_all_five_numbers() -> None:
    """В строке хода есть всё, за чем оператор к ней пришёл, и в одну строку.

    Пять величин из задания: прошло из общего и доля, кадры и статика, расход в МиБ и в
    ГиБ/ч, оценка остатка. Проверяется наличие каждой, а не длина строки.
    """
    s = Sample(elapsed_s=12.0, total_s=60.0, written=300, unchanged=60,
               total_turns=1800, disk_bytes=15 << 20)
    line = render_line(s)
    assert "0:12 из 1:00" in line
    assert "20%" in line, line
    assert "кадров 300" in line and "без изменений 60" in line
    assert "МиБ" in line and "ГиБ/ч" in line
    assert "осталось ~0:48" in line, line
    assert "\n" not in line, "ход — одна строка: она обновляется на месте"


def test_share_counts_turns_not_seconds() -> None:
    """Доля — по оборотам, а не по времени.

    Машина, отдающая 20 кадров вместо тридцати, по времени показала бы «100 %» на
    середине записи. Обороты — то, чем задана длина.
    """
    s = Sample(elapsed_s=60.0, total_s=60.0, written=900, unchanged=0,
               total_turns=1800, disk_bytes=1 << 20)
    assert s.share == pytest.approx(0.5)


def test_estimate_uses_achieved_rate_not_declared_fps() -> None:
    """Остаток считается по достигнутому темпу.

    По заявленной частоте оценка врала бы в полтора раза ровно там, где оператор решает,
    ждать ли дальше: 900 оборотов за 60 с — это 15 Гц, а не 30.
    """
    s = Sample(elapsed_s=60.0, total_s=60.0, written=900, unchanged=0,
               total_turns=1800, disk_bytes=1 << 20)
    assert s.left_s == pytest.approx(60.0)


def test_rate_is_none_until_there_is_something_to_measure() -> None:
    """Пока байтов нет, расход не выдумывается.

    Ноль вместо отказа выглядел бы как измеренное «0 ГиБ/ч» — то есть как утверждение о
    записи, которого никто не делал.
    """
    s = Sample(elapsed_s=0.0, total_s=60.0, written=0, unchanged=0,
               total_turns=1800, disk_bytes=0)
    assert s.gib_per_hour is None and s.left_s is None
    assert "расход пока не измерен" in render_line(s)
    assert "остаток неизвестен" in render_line(s)


def test_progress_updates_no_more_often_than_the_profile_says(tmp_path: Path) -> None:
    """Обновление не чаще раза в секунду, и подавленные обороты посчитаны.

    Частота — настройка профиля, а не константа: она влияет и на удобство, и на вклад
    терминала в кадр, поэтому обязана попадать в `profile_hash`.
    """
    clock = FakeClock()
    prog = Progress(total_turns=60, fps=30.0, path=tmp_path / "s", channel="none",
                    min_interval_s=1.0, now=clock)
    printed = 0
    for turn in range(60):
        clock.t = turn / 30.0                      # 30 оборотов в секунду
        if prog.update(written=turn, unchanged=0, disk_bytes=1000) is not None:
            printed += 1
    assert printed == 2, f"за две секунды строка обновилась {printed} раз"
    assert prog.suppressed == 58
    assert prog.stats()["min_interval_s"] == 1.0


def test_progress_says_something_immediately(tmp_path: Path) -> None:
    """Первая строка есть сразу: молчание в начале — это ровно то, что было принято за
    зависание."""
    prog = Progress(total_turns=1800, fps=30.0, path=tmp_path / "s", channel="line",
                    why="проверка")
    hello = prog.open()
    assert hello and "1:00" in hello


def test_disk_is_counted_only_when_the_line_is_printed(tmp_path: Path) -> None:
    """Байты на диске считаются раз в секунду, а не тридцать раз.

    Обход каталога на частоте оборотов стоил бы дороже самой записи, и запись бы от
    показа хода замедлилась — то есть показ испортил бы то, что показывает.
    """
    clock = FakeClock()
    prog = Progress(total_turns=60, fps=30.0, path=tmp_path / "s", channel="none",
                    min_interval_s=1.0, now=clock)
    calls = []
    for turn in range(60):
        clock.t = turn / 30.0
        prog.update(written=turn, unchanged=0,
                    disk_bytes=lambda: calls.append(1) or 123)
    assert len(calls) == 2, f"каталог обошли {len(calls)} раз вместо двух"


def test_stillness_writes_progress_to_a_file_and_says_why(tmp_path: Path) -> None:
    """На записи неподвижности ход идёт в файл, и причина названа.

    Изменение экрана здесь — измеряемая величина: при 30 кадр/с обновление раз в секунду
    сделало бы изменившимся до 3.3 % оборотов, то есть испортило бы именно опорный
    уровень, а не оформление.
    """
    channel, why = channel_for("stillness", default="line")
    assert channel == "file" and "измен" in why
    prog = Progress(total_turns=60, fps=30.0, path=tmp_path / "s", channel=channel,
                    why=why, min_interval_s=0.0)
    hello = prog.open()
    assert "не на экран" in hello and str(prog.log_path) in hello
    prog.update(written=1, unchanged=2, disk_bytes=4096)
    prog.finish(written=1, unchanged=2, disk_bytes=4096)
    text = prog.log_path.read_text(encoding="utf-8")
    assert "кадров 1" in text and "записано за" in text
    assert prog.log_path.parent == tmp_path, "файл хода лежит рядом с сессией, не внутри"


def test_progress_file_never_lands_inside_the_session(tmp_path: Path) -> None:
    """Файл хода не внутри каталога записи.

    Внутри он сделал бы каталог непустым до начала записи, и повторная попытка упёрлась
    бы в тот самый отказ, из-за которого задача и появилась.
    """
    root = tmp_path / "session"
    prog = Progress(total_turns=10, fps=30.0, path=root, channel="file", why="тест")
    prog.open()
    prog.close()
    assert not root.exists() or not any(root.iterdir())


def test_channel_set_is_closed() -> None:
    """Неизвестный канал — отказ, а не тихий откат к печати на экран."""
    assert set(CHANNELS) == {"line", "file", "none"}
    with pytest.raises(ValueError, match="неизвестный канал"):
        channel_for(None, default="куда-нибудь")
    assert "stillness" in FILE_KINDS


def test_progress_is_written_into_the_journal(tmp_path: Path) -> None:
    """Каким был ход — часть записи.

    Иначе вопрос «почему на этой записи изменившихся кадров 3 %» останется без ответа, а
    ответом может быть мигавшая в терминале строка.
    """
    screen = FakeScreen()
    prog = Progress(total_turns=6, fps=30.0, path=tmp_path / "s", channel="none")
    _record(tmp_path / "s", screen=screen, frames=6, prog=prog)
    with Session.open(tmp_path / "s") as s:
        marks = [e.event for e in s.journal
                 if e.kind is Kind.INTERVENTION and e.event.get("code") == "progress"]
    assert len(marks) == 1 and marks[0]["channel"] == "none"


# --- 2. прерывание ----------------------------------------------------------


def test_interrupt_leaves_a_readable_session_with_a_mark(tmp_path: Path) -> None:
    """`Ctrl+C` на середине: запись закрыта, читается и помечена прерванной.

    До TASK-16 отсюда летела трассировка из середины `zlib.compress`, а отметки не было
    вовсе — прерванная запись была неотличима от полной.
    """
    root = tmp_path / "s"
    screen = FakeScreen(interrupt_at=5)
    clock = FakeClock()
    prog = Progress(total_turns=300, fps=30.0, path=root, channel="none", now=clock)
    clock.t = 4.0
    turns = _record(root, screen=screen, frames=300, prog=prog)

    # Пять, а не четыре: первый оборот берёт кадр, снятый до открытия записи, поэтому
    # пятое обращение к экрану приходится на шестой оборот.
    assert turns.interrupted and turns.written == 5
    with Session.open(root) as s:
        mark = s.interrupted
        assert mark is not None, "отметки о прерывании нет"
        assert mark["frames_written"] == 5
        assert mark["total_s"] == pytest.approx(10.0)
        assert "прервано на 4 с из 10 с" in mark["note"]
        assert s.verify()["ok"], "прерванная запись обязана быть целой"
        assert len(s) == 5, "кадры до прерывания читаются"


def test_interrupt_inside_frame_compression_is_survivable(tmp_path: Path) -> None:
    """Прерывание **внутри сжатия кадра** — тот самый случай оператора.

    Оба раза `KeyboardInterrupt` пришёл внутри `zlib.compress`, то есть в момент
    нормальной работы. Проверять только «прерывание между кадрами» значило бы проверять
    не тот случай.
    """
    import zlib

    root = tmp_path / "s"
    screen = FakeScreen()
    real = zlib.compress
    state = {"n": 0}

    def compress(data, level=-1, **kw):
        state["n"] += 1
        if state["n"] == 4:
            raise KeyboardInterrupt
        return real(data, level, **kw)

    prog = Progress(total_turns=300, fps=30.0, path=root, channel="none")
    zlib.compress = compress                    # type: ignore[assignment]
    try:
        turns = _record(root, screen=screen, frames=300, prog=prog)
    finally:
        zlib.compress = real                    # type: ignore[assignment]

    assert turns.interrupted
    with Session.open(root) as s:
        assert s.interrupted is not None
        assert s.verify()["ok"], "запись, прерванная в сжатии, обязана быть целой"


def test_interrupted_is_not_a_verify_problem(tmp_path: Path) -> None:
    """Прерванная запись цела. `verify` говорит об этом, а не бракует её.

    Иначе оператор второй раз получает приговор своей записи: он остановил запись сам, и
    частичный корпус — корпус.
    """
    root = tmp_path / "s"
    _record(root, screen=FakeScreen(interrupt_at=3), frames=300,
            prog=Progress(total_turns=300, fps=30.0, path=root, channel="none"))
    with Session.open(root) as s:
        report = s.verify()
    assert report["ok"] is True
    assert report["interrupted"] is not None


def test_static_marks_are_not_counted_as_stored_frames(tmp_path: Path) -> None:
    """Запись со статикой цела, и `verify` это подтверждает.

    Нашёл замер, а не чтение: 17 из 19 прерванных прогонов не проходили `verify` в
    **обоих** плечах, то есть дефект не имел отношения к правке. Причина — отметка «без
    изменений» есть запись журнала вида FRAME, ссылающаяся на **уже лежащий** блок:
    нового блока она не добавляет, и считать её кадром хранилища нельзя.

    Первой такой записью была бы «неподвижность» — опорная запись минимального набора,
    где отметок больше, чем кадров. То есть оператор получил бы приговор своей записи
    ровно на самой важной из них.
    """
    root = tmp_path / "s"
    screen = FakeScreen(still_every=3)
    turns = _record(root, screen=screen, frames=30,
                    prog=Progress(total_turns=30, fps=30.0, path=root, channel="none"))
    assert turns.unchanged >= 5, "в этом прогоне мало статики — тест проверяет не то"
    with Session.open(root) as s:
        rep = s.verify()
        assert rep["unchanged_marks"] == turns.unchanged
        assert rep["frames_new"] == turns.written == rep["frames_on_disk"]
        assert rep["ok"], rep["problems"]


def test_a_finished_recording_has_no_interrupted_mark(tmp_path: Path) -> None:
    """Отметка не ставится там, где её быть не должно: иначе она ничего не различает."""
    root = tmp_path / "s"
    turns = _record(root, screen=FakeScreen(), frames=8,
                    prog=Progress(total_turns=8, fps=30.0, path=root, channel="none"))
    assert not turns.interrupted
    with Session.open(root) as s:
        assert s.interrupted is None


def test_the_mark_is_an_intervention_not_a_gap(tmp_path: Path) -> None:
    """Прерывание — действие человека, а не потеря кадров.

    `CAPTURE_GAP` означает «кадры были, но не записались». Здесь ничего не потеряно, и
    смешение этих видов записи сделало бы нажатие Ctrl+C поломкой захвата в статистике.
    """
    root = tmp_path / "s"
    _record(root, screen=FakeScreen(interrupt_at=3), frames=50,
            prog=Progress(total_turns=50, fps=30.0, path=root, channel="none"))
    with Session.open(root) as s:
        kinds = {e.kind for e in s.journal if e.event.get("code") == INTERRUPTED}
        assert kinds == {Kind.INTERVENTION}
        actors = {e.actor for e in s.journal if e.event.get("code") == INTERRUPTED}
        assert actors == {Actor.HUMAN}, "прерывание записал не человек"


def test_unchanged_turns_are_counted_separately_after_interrupt(tmp_path: Path) -> None:
    """Статика и кадры считаются раздельно и в прерванной записи тоже.

    Одно число здесь означало бы либо потерянную статику, либо мнимые потери — и то и
    другое ломает опорный замер неподвижности.
    """
    root = tmp_path / "s"
    screen = FakeScreen(still_every=2, interrupt_at=9)
    turns = _record(root, screen=screen, frames=300,
                    prog=Progress(total_turns=300, fps=30.0, path=root, channel="none"))
    assert turns.interrupted
    assert turns.written and turns.unchanged, (turns.written, turns.unchanged)
    with Session.open(root) as s:
        assert s.interrupted["unchanged"] == turns.unchanged


def test_source_ending_is_not_an_interrupt(tmp_path: Path) -> None:
    """Источник кончился — это разрыв, а не прерывание оператором. Разные записи."""
    root = tmp_path / "s"
    turns = _record(root, screen=FakeScreen(ends_at=5), frames=300,
                    prog=Progress(total_turns=300, fps=30.0, path=root, channel="none"))
    assert turns.source_ended and not turns.interrupted
    with Session.open(root) as s:
        assert s.interrupted is None
        codes = {e.event.get("code") for e in s.journal if e.kind is Kind.CAPTURE_GAP}
        assert "source_ended" in codes


# --- 3. повторный запуск в занятый каталог ----------------------------------


def test_second_run_into_the_same_directory_explains_and_gives_a_command(
        tmp_path: Path) -> None:
    """Отказ называет, что лежит в каталоге, и даёт строку, которую можно скопировать.

    Оператор напоролся именно здесь: прервав запись, он получил на повторную попытку
    «каталог не пуст» — то есть отказ, из которого не следует ни причина, ни что делать.
    """
    root = tmp_path / "s"
    _record(root, screen=FakeScreen(interrupt_at=3), frames=300,
            prog=Progress(total_turns=300, fps=30.0, path=root, channel="none"))

    text = describe_existing(root)
    assert "прервана" in text and "из" in text, text
    assert "harness ingest" in text, "не сказано, что делать с уже записанным"
    assert "harness record" in text and "-2" in text, "нет готовой команды для следующей"
    assert "записей" in text
    # Длина в команде — та, которую оператор просил, а не круглое число из головы: запись
    # заводили на 300 оборотов при 30 кадр/с, то есть на 10 с.
    assert "--seconds 10" in text, text


def test_the_message_distinguishes_three_different_situations(tmp_path: Path) -> None:
    """Годная запись, прерванная запись и посторонние файлы — три разных ответа.

    Прежний отказ говорил одно и то же на все три, и оператор с целой записью читал его
    как приговор.
    """
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "заметки.txt").write_text("не сессия", encoding="utf-8")
    assert "это не сессия" in describe_existing(junk)

    whole = tmp_path / "whole"
    _record(whole, screen=FakeScreen(), frames=6,
            prog=Progress(total_turns=6, fps=30.0, path=whole, channel="none"))
    assert "дописана до конца" in describe_existing(whole)

    stopped = tmp_path / "stopped"
    _record(stopped, screen=FakeScreen(interrupt_at=3), frames=300,
            prog=Progress(total_turns=300, fps=30.0, path=stopped, channel="none"))
    assert "прервана" in describe_existing(stopped)


def test_free_path_is_actually_free(tmp_path: Path) -> None:
    """Предложенный путь свободен. Предложить занятый значило бы послать по кругу."""
    root = tmp_path / "s"
    root.mkdir()
    (root / "x").write_text("занято", encoding="utf-8")
    first = next_free_path(root)
    assert first == tmp_path / "s-2"
    first.mkdir()
    (first / "x").write_text("занято", encoding="utf-8")
    assert next_free_path(root) == tmp_path / "s-3"


def test_recorder_refuses_a_used_directory_with_the_explanation(tmp_path: Path) -> None:
    """Отказ самого `Recorder` несёт то же объяснение, а не короткое «не пуст».

    Проверяется на `Recorder`, а не только на выводе команды: путь записи один, и
    объяснение обязано быть там, где отказ, иначе оно потеряется у второго вызывающего.
    """
    from harness.session import SessionError

    root = tmp_path / "s"
    _record(root, screen=FakeScreen(), frames=4,
            prog=Progress(total_turns=4, fps=30.0, path=root, channel="none"))
    with pytest.raises(SessionError) as exc:
        Recorder(root, profile=_profile(), source="тест", synthetic=False)
    assert "harness record" in str(exc.value) and "записей" in str(exc.value)


# --- вся команда целиком, без дисплея ---------------------------------------


def _run_record(argv: list[str], screen: FakeScreen, monkeypatch) -> int:
    """Прогнать `harness record` целиком, подменив **выбор механизма**.

    Подменяется `capture.select.open_screen` — то есть та же точка, через которую идёт и
    доктор. Ниже подменять нечего: дисплея нет, и без этой подмены проверить саму команду
    невозможно, а проверять надо именно её: цикл, ход и прерывание в отдельности уже
    проверены выше, а собрано это в `cmd_record`.
    """
    import harness.capture.select as select_mod
    from harness.cli import main

    monkeypatch.setattr(
        select_mod, "open_screen",
        lambda m, **kw: select_mod.Choice(chosen=select_mod.MSS,
                                          considered=(select_mod.MSS,), source=screen))
    return main(argv)


def test_the_command_prints_progress_while_it_records(tmp_path: Path, monkeypatch,
                                                      capsys) -> None:
    """Проверка из задания: записать 10 секунд и увидеть, что строка хода обновляется.

    Обновление «на месте» — это `\\r` без перевода строки: за минуту записи иначе набегает
    шестьдесят строк, и то, что оператор искал, тонет в том, что он уже видел.
    """
    root = tmp_path / "rec"
    # Оборот с задержкой: иначе десять секунд записи проходят за миллисекунды, и
    # настоящий ограничитель частоты (раз в секунду, из профиля) сработает единожды —
    # то есть «обновляется» проверялось бы на одной строке.
    screen = FakeScreen(still_every=7, delay_s=0.02)
    code = _run_record(["record", str(root), "--frames", "120", "--no-audio",
                        "--actor", "human"], screen, monkeypatch)
    out = capsys.readouterr()
    assert code == 0, out.err

    # Ход идёт в **стандартный вывод**, тем же потоком, что и всё остальное, что оператор
    # читает. Поток при этом не терминал (его перехватывает pytest), поэтому строки идут
    # с переводом, а не через возврат каретки: в перенаправленном потоке `\r` не виден до
    # конца записи, и ровно на это оператор и напоролся.
    running = [x for x in out.out.splitlines()
               if "из" in x and "кадров" in x and "записано" not in x]
    assert len(running) >= 2, f"строк хода за прогон: {len(running)} — {out.out}"
    # Строки обязаны **различаться**: одинаковая строка дважды означала бы, что счётчики
    # в неё не попадают, и оператор смотрел бы на замерший вывод — то же самое молчание.
    assert running[0] != running[-1], running
    assert "осталось ~" in out.out or "остаток неизвестен" in out.out
    assert "записано за" in out.out and str(root) in out.out
    # Итоговая строка **одна**: путь завершения проходится один раз (пункт 4 задачи).
    assert out.out.count("записано за") == 1, out.out
    assert "Дальше: harness ingest" in out.out
    with Session.open(root) as s:
        assert s.verify()["ok"] and s.interrupted is None


def test_the_command_survives_ctrl_c_and_says_what_is_left(tmp_path: Path, monkeypatch,
                                                           capsys) -> None:
    """Проверка из задания: прервать на середине, сессия закрыта и помечена.

    Код возврата 0, а не 2: прерывание — решение оператора, а не поломка команды, и
    частичная запись годна. Сценарий `harness record && harness ingest` обязан пройти
    дальше, а не встать на том, что человек остановил запись сам.
    """
    root = tmp_path / "rec"
    screen = FakeScreen(interrupt_at=40)
    code = _run_record(["record", str(root), "--seconds", "10", "--no-audio"],
                       screen, monkeypatch)
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "прервано на" in out.out, out.out
    assert out.out.count("прервано на") == 1, "итог напечатан дважды"
    assert "годна к приёму" in out.out

    with Session.open(root) as s:
        assert s.interrupted is not None
        assert s.verify()["ok"]


def test_the_command_refuses_a_used_directory_and_says_what_to_do(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """Повторный запуск в тот же каталог: внятное сообщение и готовая команда.

    Отказ приходит **до** открытия захвата: ждать выбора механизма и первого кадра, чтобы
    услышать про каталог, было бы вторым ожиданием на ровном месте.
    """
    root = tmp_path / "rec"
    _run_record(["record", str(root), "--seconds", "10", "--no-audio"],
                FakeScreen(interrupt_at=40), monkeypatch)
    capsys.readouterr()

    second = FakeScreen()
    code = _run_record(["record", str(root), "--seconds", "10", "--no-audio"],
                       second, monkeypatch)
    err = capsys.readouterr().err
    assert code == 2
    assert "прервана" in err and "harness record" in err and "-2" in err
    assert second.reads == 0, "захват открывали, хотя отказ был известен заранее"


def test_stillness_recording_keeps_the_terminal_out_of_the_frame(
        tmp_path: Path, monkeypatch, capsys) -> None:
    """`--kind stillness`: ход уходит в файл, на экране ни одной строки хода.

    Это и есть та правка, которая не про удобство: на этой записи мерится отсутствие
    изменений экрана, и собственная мигающая строка испортила бы именно её.
    """
    root = tmp_path / "rec"
    code = _run_record(["record", str(root), "--seconds", "10", "--no-audio",
                        "--kind", "stillness"], FakeScreen(still_every=2), monkeypatch)
    out = capsys.readouterr()
    assert code == 0
    assert "\r" not in out.out + out.err, "ход печатался на экран на записи неподвижности"
    log = root.parent / f"{root.name}-progress.log"
    assert log.exists() and "кадров" in log.read_text(encoding="utf-8")
    assert str(log) in out.out, "оператору не сказали, куда смотреть"


# --- итог -------------------------------------------------------------------


def test_final_line_names_numbers_and_the_path(tmp_path: Path) -> None:
    """Итоговая строка: числа и путь. Прерванная говорит, на чём прервана."""
    s = Sample(elapsed_s=4.0, total_s=60.0, written=100, unchanged=20,
               total_turns=1800, disk_bytes=2 << 20)
    done = render_final(s, path=tmp_path / "s", interrupted=False, audio_note="со звуком")
    assert "записано за 0:04" in done and str(tmp_path / "s") in done
    assert "100 изменившихся" in done and "20 отметок" in done and "со звуком" in done

    stopped = render_final(s, path=tmp_path / "s", interrupted=True)
    assert "прервано на 0:04 из 1:00" in stopped


def test_progress_settings_live_in_the_schema() -> None:
    """Частота и канал — настройки профиля, а не константы в коде.

    Иначе они не попадут в `profile_hash` (инвариант 23), и два прогона с разной частотой
    обновления окажутся неразличимы, хотя вклад терминала в кадр у них разный.
    """
    p = MILESTONE_0.parameters
    assert p["progress_min_interval_s"] == 1.0
    assert p["progress_channel"] in CHANNELS
    from harness.core.settings import SCHEMA, GROUP_APPLIES, RunPath

    group = {s.key: s.group for s in SCHEMA}
    assert group["progress_min_interval_s"] == "Ход записи"
    assert GROUP_APPLIES["Ход записи"] == (RunPath.LIVE,)


def test_turns_counts_are_one_object(tmp_path: Path) -> None:
    """Числа цикла возвращаются, а не печатаются внутри.

    Печать внутри цикла сделала бы его непроверяемым: пришлось бы перехватывать поток
    вместо того, чтобы сравнить числа.
    """
    t = Turns(written=3, unchanged=2)
    assert t.turns == 5


# --- TASK-17: срок, разбивка, поток вывода, пути -----------------------------


def test_seconds_is_a_deadline_not_a_frame_count(tmp_path: Path) -> None:
    """`--seconds` — срок. Запись кончается по времени, а не по числу кадров.

    До TASK-17 десять секунд превращались в 300 кадров, и на машине, отдающей 6.5 кадр/с,
    запись шла 46 секунд — молча. Оператор просил десять секунд и получал минуту.
    """
    root = tmp_path / "s"
    # Медленный источник: 300 кадров он отдал бы за 6 с, но срок — 0.3 с.
    screen = FakeScreen(delay_s=0.02)
    p = _profile(screen.w, screen.h)
    with Recorder(root, profile=p, source=screen.name, synthetic=False) as rec:
        turns = run_turns(rec, source=screen, frames=300, first=screen.first(),
                          actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN,
                          seconds=0.3, expected_fps=30.0)
    assert turns.stop_reason == "deadline"
    assert turns.turns < 300, "запись досидела до числа кадров вместо срока"
    assert 0.3 <= turns.elapsed_s < 2.0, turns.elapsed_s


def test_frames_still_means_frames(tmp_path: Path) -> None:
    """`--frames` не изменился: столько кадров, сколько названо.

    Два флага — два разных смысла, и это решение, а не недосмотр: за числом кадров
    приходят, когда нужен ровный объём для сравнения прогонов.
    """
    root = tmp_path / "s"
    screen = FakeScreen()
    p = _profile(screen.w, screen.h)
    with Recorder(root, profile=p, source=screen.name, synthetic=False) as rec:
        turns = run_turns(rec, source=screen, frames=25, first=screen.first(),
                          actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN,
                          expected_fps=30.0)
    assert turns.stop_reason == "frames" and turns.turns == 25


def test_shortfall_against_the_target_rate_is_a_printed_number(tmp_path: Path) -> None:
    """Недобор частоты виден числом, а не растянутой записью.

    `capture_fps` — цель цикла, а не свойство записи. Тот, кто посчитает длительность как
    «кадров делить на capture_fps», получит 10 с там, где прошло 46, — поэтому ожидаемое и
    достигнутое стоят рядом.
    """
    root = tmp_path / "s"
    # 60 мс на оборот — это 16 кадр/с при цели 30, то есть недобор вдвое. Первая редакция
    # теста брала 20 мс и получала 50 кадр/с, то есть **перебор**: проверка утверждала не
    # то, что называла, и поймала это сама.
    screen = FakeScreen(delay_s=0.06)
    p = _profile(screen.w, screen.h)
    with Recorder(root, profile=p, source=screen.name, synthetic=False) as rec:
        turns = run_turns(rec, source=screen, frames=1000, first=screen.first(),
                          actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN,
                          seconds=0.5, expected_fps=30.0)
    st = turns.stages()
    assert st["expected_turns"] > turns.turns, st
    assert st["capture_ms"] >= 40.0, ("ожидание кадра не измерено: "
                                      f"{st['capture_ms']:.1f} мс")


def test_time_breakdown_separates_capture_encode_and_write(tmp_path: Path) -> None:
    """Разбивка раздельная: ожидание кадра, разность, сжатие, файл.

    «Узкое место где-то в записи на диск» — не диагноз. Стадии считаются там, где
    происходят, и на машине оператора тоже: в контейнере другой диск и другой процессор.
    """
    root = tmp_path / "s"
    screen = FakeScreen(w=320, h=240)
    p = _profile(320, 240)
    with Recorder(root, profile=p, source=screen.name, synthetic=False) as rec:
        turns = run_turns(rec, source=screen, frames=40, first=screen.first(),
                          actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN,
                          expected_fps=30.0)
        cost = rec.cost()
    fr = cost["frames"]
    assert fr["calls"] == turns.written
    assert fr["keyframes"] >= 1 and fr["keyframes"] < fr["calls"], fr
    assert fr["compress_ms_per_call"] > 0 and fr["write_ms_per_call"] >= 0
    assert fr["ratio"] and fr["ratio"] > 1, "сжатие не сжало — проверять нечего"
    assert turns.stages()["record_ms"] > 0


def test_the_breakdown_lands_in_the_journal(tmp_path: Path, monkeypatch,
                                            capsys) -> None:
    """Разбивка лежит в записи, а не только в консоли, которую закрыли."""
    root = tmp_path / "rec"
    _run_record(["record", str(root), "--frames", "20", "--no-audio"],
                FakeScreen(w=320, h=240), monkeypatch)
    out = capsys.readouterr().out
    assert "на что ушло время" in out and "сжатие" in out
    with Session.open(root) as s:
        costs = [e.event for e in s.journal
                 if e.kind is Kind.INTERVENTION and e.event.get("code") == "cost"]
    assert len(costs) == 1 and costs[0]["frames"]["calls"] == 20


def test_progress_prints_lines_when_the_stream_is_not_a_terminal(tmp_path: Path) -> None:
    """В перенаправленном потоке ход идёт строками, а не возвратом каретки.

    Пункт 3 задачи: строка хода не появилась вовсе. Возврат каретки в неперенаправленном
    терминале обновляет строку, а в файле остаётся байтом и не виден до конца записи.
    """
    import io

    class Piped(io.StringIO):
        def isatty(self) -> bool:
            return False

    clock = FakeClock()
    out = Piped()
    prog = Progress(total_turns=60, fps=30.0, path=tmp_path / "s", channel="line",
                    min_interval_s=1.0, out=out, now=clock)
    for turn in range(60):
        clock.t = turn / 30.0
        prog.update(written=turn, unchanged=0, disk_bytes=1000)
    text = out.getvalue()
    assert "\r" not in text, "в перенаправленный поток ушёл возврат каретки"
    assert len([x for x in text.splitlines() if x.strip()]) == 2, text


def test_progress_fits_the_terminal_width(tmp_path: Path, monkeypatch) -> None:
    """Строка не длиннее окна: перенос уводит возврат каретки не туда.

    Дополнение до 96 столбцов в окне шириной 80 переносило строку, и следующее обновление
    писалось поверх переноса — виден мусор либо пустота, то есть отчёт оператора.
    """
    import io
    import shutil

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(shutil, "get_terminal_size", lambda default=(80, 24): __import__(
        "os").terminal_size((60, 24)))
    out = Tty()
    prog = Progress(total_turns=1800, fps=30.0, path=tmp_path / "очень-длинный-путь-записи",
                    channel="line", min_interval_s=0.0, out=out)
    prog.update(written=1234, unchanged=567, disk_bytes=987 << 20)
    body = out.getvalue().lstrip("\r")
    assert len(body) == 59, f"строка длиной {len(body)} в окне 60"
    assert "…" in body, "строка обрезана без знака обрезки"


def test_the_final_line_is_printed_once(tmp_path: Path) -> None:
    """`finish` возвращает итог и не печатает его сам.

    Пункт 4 задачи: путь завершения проходился дважды — строку писал и `finish`, и
    `cmd_record`. Одна работа — одно место.
    """
    import io

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    out = Tty()
    prog = Progress(total_turns=10, fps=30.0, path=tmp_path / "s", channel="line",
                    min_interval_s=0.0, out=out)
    line = prog.finish(written=5, unchanged=1, disk_bytes=1024)
    assert "записано за" in line
    assert "записано за" not in out.getvalue(), "итог напечатан и каналом, и вызывающим"


def test_tilde_is_expanded_by_us_because_cmd_exe_does_not(tmp_path: Path) -> None:
    """`~` раскрывается программой: `cmd.exe` и `powershell` этого не делают.

    Оператор набрал `~\\harness-live\\proba` и получил каталог с именем `~` внутри
    проекта. Тильда — соглашение `sh`, и на Windows раскрыть её больше некому.
    """
    from harness.paths import PathError, live_path, resolve_input, show

    home = tmp_path / "дом"
    got = resolve_input("~/harness-live/proba", home=home)
    assert got == home / "harness-live" / "proba", got
    assert live_path("proba", home) == home / "harness-live" / "proba"

    # Тильда в середине пути — не раскрыл никто, и писать туда молча нельзя.
    with pytest.raises(PathError) as exc:
        resolve_input("ai/harness/~/harness-live/proba", home=home)
    assert "не раскрыл" in str(exc.value) and show(home) in str(exc.value)


def test_printed_paths_use_the_shape_of_the_target_system() -> None:
    """Пути печатаются в форме той системы, где их будут набирать.

    `~/harness-live/stillness` в `cmd.exe` не работает дважды: и тильдой, и наклонными.
    Строка для копирования обязана работать при вставке.
    """
    from harness.paths import show

    assert show("C:/Users/chiha/harness-live", windows=True) == \
        "C:\\Users\\chiha\\harness-live"
    assert show("/home/o/harness-live", windows=False) == "/home/o/harness-live"


def test_the_plan_prints_real_paths_not_tildes() -> None:
    """В плане записи — настоящие пути этой машины, а не `~`.

    Дважды напоролись на одно: план печатал `~/harness-live/...`, оператор копировал, и
    тильда доезжала до нас именем каталога.
    """
    from harness.corpus.live import plan_text
    from harness.paths import live_root, show

    text = plan_text("minimal")
    assert "~" not in text, "в плане осталась тильда"
    assert show(live_root()) in text


def test_stillness_plan_says_to_get_out_of_the_frame() -> None:
    """План записи неподвижности говорит убрать себя из кадра.

    Замер оператора дал 300 изменившихся кадров и **ни одной** отметки «без изменений»:
    опорный уровень мерил мигающий курсор и открытое окно записи, а не фон экрана.
    """
    from harness.corpus.live import KINDS, plan_text

    how = KINDS["stillness"]["how"]
    assert "курсор" in how and ("сверн" in how or "убрать" in how), how
    assert "курсор" in plan_text("minimal")


def test_every_path_argument_expands_the_tilde() -> None:
    """Раскрытие стоит типом аргумента, а не проверкой в каждой команде.

    Тильда доехала до `ingest` **вторым** заходом именно потому, что раскрытие было делом
    каждой команды по отдельности. Проверяется механически: ни один аргумент пути не
    объявлен как `type=Path`.
    """
    import ast

    src = (Path(__file__).resolve().parent.parent / "src" / "harness" / "cli.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    plain = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        text = ast.unparse(node)
        if "add_argument" in text and "type=Path" in text:
            plain.append((node.lineno, text[:70]))
    assert not plain, f"аргументы пути без раскрытия тильды: {plain}"

    # И сама подмена работает: неизвестная команда с путём в тильде даёт отказ argparse,
    # а не создаёт каталог с именем `~`.
    from harness.cli import _path_arg
    import argparse

    assert _path_arg("~").is_absolute()
    with pytest.raises(argparse.ArgumentTypeError, match="не раскрыл"):
        _path_arg("a/~/b")
