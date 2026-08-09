"""Инвариант 30: у каждого параметра есть читатель, и он покрывает объявленные пути.

Проверяется не только сам детектор, но и **его собственная слепота**. Первая редакция
детектора объявила однобокими 140 настроек из 155, вторая — 27, и почти все были её
ошибкой модели путей, а не дефектами кода. Детектор, кричащий ложно, хуже
отсутствующего: настоящие находки тонут, и смотреть в него перестают. Поэтому здесь есть
и тест на то, что найденное число дефектов равно нулю, и тесты на то, что ноль этот не
получен занижением требований.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.core.profile import from_schema
from harness.core.settings import (GROUP_APPLIES, SCHEMA, SYNTHETIC_ONLY, RunPath,
                                   applies_of, applies_to_live)
from harness.params import (Diagnosis, Finding, Path_, Site, audit, counting_profile,
                            counts, defect_count, read_sites, render_table,
                            runtime_probe)
from harness.session import RECORD_CLAIMS, Recorder, SessionError


# --- сам детектор ---------------------------------------------------------------


def test_no_defective_parameters() -> None:
    """Ноль мёртвых, однобоких и «только отчёт». Это число не должно расти молча."""
    findings = audit()
    bad = defect_count(findings)
    assert bad == 0, "\n" + render_table(findings, only_defects=True)


def test_every_group_declares_applicability() -> None:
    """Незаявленная группа — недосмотр, и он ловится здесь, а не молчаливым «агент»."""
    missing = sorted({s.group for s in SCHEMA} - set(GROUP_APPLIES))
    assert not missing, f"группы без объявленной применимости: {missing}"


def test_report_reader_is_not_a_reader() -> None:
    """Чтение внутри печати читателем не считается: печать не меняет поведения."""
    site = Site("model/drives.py", "as_dict", (Path_.AGENT,), 10)
    assert site.report_only
    f = Finding("что-нибудь", Diagnosis.REPORT_ONLY, (site,), (Path_.AGENT,))
    assert f.readers == ()


def test_one_sided_is_heavier_than_dead() -> None:
    """Порядок тяжести: однобокий выше мёртвого, иначе сортировка отчёта врёт."""
    reader = Site("capture/screen.py", "start", (Path_.LIVE,), 10)
    one_sided = Finding("к1", Diagnosis.ONE_SIDED, (reader,),
                        (Path_.LIVE, Path_.SYNTHETIC), (Path_.SYNTHETIC,))
    dead = Finding("к2", Diagnosis.DEAD, (), (Path_.AGENT,), (Path_.AGENT,))
    table = render_table([one_sided, dead])
    assert table.index("к1") < table.index("к2")
    assert "нет читателя на: синтетика" in table


def test_counts_cover_every_diagnosis() -> None:
    got = counts(audit())
    assert set(got) == {str(d) for d in Diagnosis}
    assert got[str(Diagnosis.OK)] > 100


# --- два способа, и у каждого своя слепота --------------------------------------


def test_static_pass_finds_reads_by_function() -> None:
    """Статика знает не только «где», но и «в какой функции» — иначе печать не отсечь."""
    sites = read_sites()
    where = {(s.module, s.function) for s in sites["frame_format"]}
    assert ("session.py", "_check_declared") in where


def test_frame_format_is_read_on_both_paths() -> None:
    """Формат кадра читается и на живом пути, и на синтетическом. Был только живой."""
    f = {x.key: x for x in audit()}["frame_format"]
    covered = {p for s in f.readers for p in s.paths}
    assert Path_.LIVE in covered and Path_.SYNTHETIC in covered
    assert f.diagnosis is Diagnosis.OK


def test_runtime_counter_sees_computed_key() -> None:
    """Счётчик видит чтение по вычисленному ключу, которого статике не видно вовсе.

    Это и есть смысл второго способа: `p[name]` в цикле по именам статический обход
    пропускает целиком, и параметр, читаемый только так, был бы объявлен мёртвым.
    """
    profile, seen = counting_profile(from_schema("вычисленный ключ"))
    name = "capture_" + "fps"
    _ = profile.parameters[name]
    assert seen["capture_fps"] == 1


def test_runtime_probe_reads_something(tmp_path) -> None:
    """Прогон со счётчиками читает настройки, и число прочитанных ключей известно."""
    seen = runtime_probe(steps=12, root=tmp_path / "sess")
    assert sum(seen.values()) > 0
    # Прогон синтетический и агентский: живого пути в нём нет. Настройки, читаемые
    # только там (обратный сторож читает их в `inject/watchdog.py`), он прочитать не
    # мог — если прочёл, значит куда-то подсунута заглушка, и счётчик показывает
    # чтения, которых на настоящей машине не случилось.
    assert seen.get("watchdog_still_seconds", 0) == 0
    # А запись сессии в прогоне есть, и общие для обоих путей настройки хранения она
    # читает: это и отличает «пути нет» от «чтения нет вовсе».
    assert seen.get("capture_drop_tolerance", 0) > 0


def test_method_is_stated_on_every_conclusion() -> None:
    """Требование части 2: у каждого вывода указан способ, которым он получен."""
    only_static = Finding("к", Diagnosis.OK, (), (Path_.AGENT,))
    assert only_static.method == "статика"
    reader = Site("model/places.py", "from_profile", (Path_.AGENT,), 3)
    both = Finding("к", Diagnosis.OK, (reader,), (Path_.AGENT,), runtime_reads=5)
    assert both.method == "статика и прогон"
    blind = Finding("к", Diagnosis.DEAD, (), (Path_.AGENT,), (Path_.AGENT,),
                    runtime_reads=3)
    assert blind.method == "расхождение способов"
    assert "диагноз «мёртвый» здесь неверен" in blind.disagreement
    unvisited = Finding("к", Diagnosis.OK, (reader,), (Path_.AGENT,), runtime_reads=0)
    assert "не дефект" in unvisited.disagreement


# --- применимость объявляется, а не угадывается ---------------------------------


def test_watchdog_is_declared_live_only() -> None:
    """Обратный сторож живёт там, где ввод вводится: у синтетики инъекции нет."""
    by_key = {s.key: s for s in SCHEMA}
    for key in ("watchdog_still_seconds", "watchdog_still_threshold"):
        assert applies_of(by_key[key]) == (RunPath.LIVE,)
        assert applies_to_live(key)


def test_synthetic_only_is_derived_not_listed() -> None:
    """Список из семи имён заменён выводом из схемы: два места правды расходились."""
    assert "world_bounded" in SYNTHETIC_ONLY
    assert "randomize_world" in SYNTHETIC_ONLY
    assert not applies_to_live("world_bounded")
    assert applies_to_live("capture_fps")
    for key in SYNTHETIC_ONLY:
        assert applies_of({s.key: s for s in SCHEMA}[key]) == (RunPath.SYNTHETIC,)


# --- часть 4: профиль обязан соответствовать записи -----------------------------


def test_record_claims_are_declared() -> None:
    """Список полей, описывающих свойство записи, объявлен, а не рассыпан по коду."""
    keys = {k for k, _where in RECORD_CLAIMS}
    assert keys == {"capture_width", "capture_height", "frame_format",
                    "audio_channels", "audio_rate"}
    # `capture_fps` намеренно не в списке: это цель для цикла, а не свойство записи.
    assert "capture_fps" not in keys


def test_synthetic_record_refuses_lying_frame_format(tmp_path) -> None:
    """Профиль заявил rgb8, мир нарисовал серый кадр — отказ, а не тихая запись.

    Сдвиг числа: до правки такая сессия записывала кадры (1 запись, 0 отказов) и
    оставляла в `session.json` заявление о трёхканальном кадре, которого в ней нет.
    """
    from harness.session import Session

    profile = from_schema("ложный формат", frame_format="rgb8")
    frame = np.zeros((16, 24), dtype=np.uint8)
    root = tmp_path / "s"
    with Recorder(root, profile=profile, source="synthetic", synthetic=True) as rec:
        with pytest.raises(SessionError, match="frame_format=rgb8"):
            rec.record_frame(frame, t_world=0)
    with Session(root) as s:
        assert len(list(s.journal.frames())) == 0


def test_synthetic_record_accepts_declared_rgb(tmp_path) -> None:
    """Тот же профиль с настоящим трёхканальным кадром записывается без возражений."""
    profile = from_schema("честный формат", frame_format="rgb8")
    frame = np.zeros((16, 24, 3), dtype=np.uint8)
    with Recorder(tmp_path / "s", profile=profile, source="synthetic",
                  synthetic=True) as rec:
        assert rec.record_frame(frame, t_world=0) is not None


def test_gray_profile_refuses_colour_frame(tmp_path) -> None:
    """И наоборот: серый профиль не принимает цветной кадр."""
    frame = np.zeros((16, 24, 3), dtype=np.uint8)
    profile = from_schema("серый профиль")          # frame_format=gray8 по умолчанию
    with Recorder(tmp_path / "s", profile=profile, source="synthetic",
                  synthetic=True) as rec:
        with pytest.raises(SessionError, match="frame_format=gray8"):
            rec.record_frame(frame, t_world=0)


class _FakeStream:
    """Поток, который открылся, но отдаёт свою частоту, а не запрошенную."""

    def __init__(self, *, samplerate: float, **_kw: object) -> None:
        self.samplerate = samplerate
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        pass


def _fake_sounddevice(monkeypatch, rate: float) -> None:
    """Подсунуть модуль `sounddevice`, который открывает поток на частоте `rate`."""
    import sys
    import types

    mod = types.ModuleType("sounddevice")
    mod.InputStream = lambda **kw: _FakeStream(**{**kw, "samplerate": rate})
    monkeypatch.setitem(sys.modules, "sounddevice", mod)


def test_audio_rate_mismatch_refuses(monkeypatch) -> None:
    """Устройство отдаёт другую частоту — отказ. Иначе весь звук уезжает молча.

    Драйвер вправе подставить свою частоту: попросили 48 кГц, получили 44.1, и `read()`
    об этом не скажет. Пеленг считается по разнице каналов в отсчётах, длительность
    блока — делением на заявленную частоту, и всё уезжает на 8.8 % без единого признака.
    """
    from harness.capture.screen import BackendUnavailable, LoopbackAudio

    _fake_sounddevice(monkeypatch, 44100.0)
    audio = LoopbackAudio(rate=48000)
    with pytest.raises(BackendUnavailable, match="44100 Гц"):
        audio.start()
    # Поток закрыт: отказ не оставляет открытого устройства.
    assert audio._stream is None


def test_audio_rate_match_starts(monkeypatch) -> None:
    """Совпала — поток остаётся открытым, и отказа нет."""
    from harness.capture.screen import LoopbackAudio

    _fake_sounddevice(monkeypatch, 48000.0)
    audio = LoopbackAudio(rate=48000)
    audio.start()
    assert audio.actual_rate == 48000
    audio.stop()


# --- часть 3: макросы читают профиль, а не значение по умолчанию ----------------


def test_macro_recorder_reads_profile() -> None:
    """Длина макроса берётся из профиля. Раньше её передавал только `cli.py`.

    Сдвиг числа: `MacroRecorder(graph)` без профиля даёт `max_length == 3` при любом
    значении в профиле; `from_profile` на том же профиле даёт 4.
    """
    from harness.behaviour.skills import MacroRecorder
    from harness.model.places import PlaceGraph

    profile = from_schema("макросы", macro_max_length=4)
    graph = PlaceGraph.from_profile(profile)
    assert MacroRecorder(graph).max_length == 3
    rec = MacroRecorder.from_profile(graph, profile)
    assert rec is not None and rec.max_length == 4


def test_macro_recorder_off_when_profile_says_so() -> None:
    """Ноль и единица означают «макросов нет»: это законное выключение, не отказ."""
    from harness.behaviour.skills import MacroRecorder
    from harness.model.places import PlaceGraph

    profile = from_schema("без макросов", macro_max_length=0)
    graph = PlaceGraph.from_profile(profile)
    assert MacroRecorder.from_profile(graph, profile) is None


# --- показатель -----------------------------------------------------------------


def test_defect_count_is_a_permanent_vital() -> None:
    """Число дефектных параметров идёт в постоянные показатели с единицей «настройка»."""
    from harness.model.vitals import _dead_params

    bad, total = _dead_params()
    assert total > 100
    assert bad == 0


def test_verify_reports_observed_frame_rate(tmp_path) -> None:
    """Достигнутая частота читается по записи, а не берётся из профиля.

    Сдвиг числа: до правки `verify()` не содержал наблюдённой частоты вовсе, и
    единственным источником оставался профиль — то есть заявление, а не наблюдение.
    Отказа здесь нет намеренно: `capture_fps` — цель для цикла, и порог стоит в
    `selftest`, где оператор ещё может что-то поменять.
    """
    import time

    from harness.session import Session

    profile = from_schema("частота", capture_fps=30.0)
    root = tmp_path / "s"
    with Recorder(root, profile=profile, source="synthetic", synthetic=True) as rec:
        for i in range(5):
            rec.record_frame(np.full((16, 24), i * 9, dtype=np.uint8), t_world=i)
            time.sleep(0.01)
    with Session(root) as s:
        report = s.verify()
    assert report["capture_fps_declared"] == 30.0
    observed = report["capture_fps_observed"]
    assert observed is not None and observed > 0
    # Расхождение с заявленной частотой проблемой не считается.
    assert report["ok"]
