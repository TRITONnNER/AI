"""Тесты проводки: объявленное должно работать, а не только числиться.

Ревизия всего проекта нашла 22 настройки, которые были объявлены в схеме и не
читались ни одной строкой кода. Это ровно та тихая ложь, которую запрещает
`CLAUDE.md`: ручка обещает поведение, поворачивается, ничего не меняет — и понять
это можно только чтением всего кода.

Здесь два вида тестов. Первый — сторож: ни одна настройка не может остаться
непрочитанной молча, для этого есть явная отметка «ещё не реализовано» с причиной.
Остальные проверяют, что проведённое действительно действует.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest

from harness.core.action import Action
from harness.core.clocks import Stamp
from harness.core.journal import Kind
from harness.core.profile import from_schema
from harness.core.settings import SCHEMA

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "harness"


def _profile(**kw: object):
    base: dict[str, object] = dict(capture_width=64, capture_height=48)
    base.update(kw)
    return from_schema("ТЕСТ-проводка", **base)


# --- сторож -----------------------------------------------------------------


def test_every_setting_is_either_used_or_planned() -> None:
    """Настройка либо читается кодом, либо честно помечена нереализованной.

    Третьего быть не может. Объявленная и никем не читаемая ручка — это обещание
    поведения, которого нет; такое находится только чтением всего кода, то есть
    практически никогда.
    """
    text = "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py")
                     if p.name != "settings.py")
    silent = [s.key for s in SCHEMA
              if not s.planned and not re.search(rf'["\']{re.escape(s.key)}["\']', text)]
    assert not silent, (
        "объявлены и нигде не читаются: " + ", ".join(silent)
        + ". Либо провести, либо пометить planned= с причиной")


def test_planned_settings_say_what_is_missing() -> None:
    """«Ещё не реализовано» без причины — отговорка, а не отметка."""
    for s in SCHEMA:
        if s.planned:
            assert len(s.planned) > 20, f"{s.key}: причина слишком короткая"


def test_status_report_is_generated_from_code() -> None:
    """Отчёт о состоянии обязан считаться по коду, а не быть текстом.

    Иначе он расходится с проектом незаметно — тем же способом, что и список
    настроек.
    """
    import importlib.util

    tools = pathlib.Path(__file__).resolve().parent.parent / "tools"
    import sys

    spec = importlib.util.spec_from_file_location("ps", tools / "project_status.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ps"] = mod          # dataclass внутри модуля требует его в sys.modules
    spec.loader.exec_module(mod)

    data = mod.collect(with_tests=False)
    assert data["invariants_covered"] == 12, "не все инварианты покрыты тестами"
    assert not data["settings"]["declared_but_unread"]
    assert data["tests_total"] > 250
    # У каждой части либо есть замер, либо явно сказано, чего не хватает.
    for part in data["parts"]:
        assert part["measured"] or part["absent"], f"{part['name']}: ни замера, ни причины"
        assert not part["missing_modules"], f"{part['name']}: модуля нет"
    # У каждого пункта «чего нет» есть причина.
    for missing in data["missing"]:
        assert missing["why"], missing["what"]
    assert "Состояние проекта" in mod.render_markdown(data)


# --- проведённое действует ---------------------------------------------------


def test_dropped_frames_are_noticed_without_being_told() -> None:
    """Пропуск кадров замечается записью, а не остаётся на совести вызывающего.

    Иначе запись выглядит непрерывной там, где источник отвалился, и всё, что
    считается по соседним кадрам, молча считается по разным моментам времени.
    """
    from harness.session import Recorder, Session

    profile = _profile(capture_drop_tolerance=2)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "s"
        with Recorder(root, profile=profile, source="t", synthetic=True) as rec:
            frame = np.zeros((48, 64), dtype=np.uint8)
            for t in (0, 1, 2, 10):        # прыжок на семь кадров
                rec.record_frame(frame, t_world=t)
        with Session.open(root) as s:
            gaps = [e.event for e in s.journal if e.kind is Kind.CAPTURE_GAP]
    assert len(gaps) == 1
    assert gaps[0]["code"] == "frames_dropped" and gaps[0]["missed"] == 7


def test_drop_tolerance_actually_tolerates() -> None:
    from harness.session import Recorder, Session
    import tempfile

    profile = _profile(capture_drop_tolerance=5)
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "s"
        with Recorder(root, profile=profile, source="t", synthetic=True) as rec:
            frame = np.zeros((48, 64), dtype=np.uint8)
            for t in (0, 4):              # пропущено три — в пределах нормы
                rec.record_frame(frame, t_world=t)
        with Session.open(root) as s:
            gaps = [e for e in s.journal if e.kind is Kind.CAPTURE_GAP]
    assert not gaps


def test_model_call_rate_is_capped_by_profile() -> None:
    """`token_budget_per_min` — предел темпа обращений, а не украшение отчёта."""
    from harness.core.resources import BREACH_MODEL_RATE, ResourceGovernor

    clock = [1000.0]
    gov = ResourceGovernor(_profile(token_budget_per_min=0.5),
                           now=lambda: clock[0])
    for _ in range(40):
        gov.spend(model_calls=1)
    codes = [b.as_dict()["code"] for b in gov.check(Stamp(0, 0))]
    assert BREACH_MODEL_RATE in codes
    assert gov.model_call_rate() > 0.5
    clock[0] += 61
    assert gov.model_call_rate() == 0.0, "окно не минутное"


def test_watchdog_takes_its_thresholds_from_the_profile() -> None:
    from harness.inject.watchdog import Watchdog

    profile = _profile(watchdog_still_seconds=3.5, watchdog_still_threshold=0.05)
    w = Watchdog.from_profile(profile)
    assert w.still_seconds == 3.5 and w.threshold == 0.05


def test_text_symbolized_switch_actually_switches() -> None:
    """Выключение хеширования текста — законный ablation, но он должен работать.

    И он обязан быть заметен на всех уровнях: символизатор возвращает надпись,
    хеш структуры другой, а журнал перестаёт требовать непрозрачности — потому что
    прогон объявлен другим.
    """
    from harness.core.symbols import Symbolizer, is_symbol

    on, off = _profile(), _profile(text_symbolized=False)
    a, b = Symbolizer.from_profile(on), Symbolizer.from_profile(off)
    assert a.enabled and not b.enabled
    assert is_symbol(a.symbolize("Здоровье"))
    assert b.symbolize("Здоровье") == "Здоровье"
    assert on.structure_hash != off.structure_hash, "ручка обязана форкать журнал"


def test_journal_still_refuses_plain_text_by_default(tmp_path) -> None:
    from harness.core.symbols import SymbolError
    from harness.session import Recorder

    with Recorder(tmp_path / "on", profile=_profile(), source="t",
                  synthetic=True) as rec:
        with pytest.raises(SymbolError):
            rec.record_perception({"objects": [{"label": "Здоровье"}]})


def test_journal_allows_plain_text_only_in_the_declared_ablation(tmp_path) -> None:
    from harness.session import Recorder

    profile = _profile(text_symbolized=False)
    with Recorder(tmp_path / "off", profile=profile, source="t",
                  synthetic=True) as rec:
        entry = rec.record_perception({"objects": [{"label": "здоровье"}]})
    assert entry.perception["objects"][0]["label"] == "здоровье"
    assert entry.structure_hash == profile.structure_hash


def test_debug_channel_can_be_switched_off_and_then_there_is_no_truth(tmp_path) -> None:
    """Выключенный отладочный канал не пишет истину — и сверять потом не с чем.

    Это законный режим (запись можно отдать тому, кому истину видеть нельзя), но он
    обязан быть объявлен ручкой, а не получаться случайно.
    """
    from harness.corpus.synthetic import generate_session, load_hud_mask
    from harness.debug.channel import DebugChannelError

    off = _profile(debug_channel_enabled=False)
    root = generate_session(tmp_path / "off", profile=off, seed=1)
    assert not (root / "debug").exists() or not any((root / "debug").iterdir())
    with pytest.raises(DebugChannelError):
        load_hud_mask(root)

    on = _profile(debug_channel_enabled=True)
    root2 = generate_session(tmp_path / "on", profile=on, seed=1)
    assert load_hud_mask(root2).any()


def test_no_internet_means_no_network_service() -> None:
    """`internet_access=False` — прогон без сети, и он обязан быть без сети.

    Локальный узел спрашивать можно: он не сеть. Сетевой сервис — нельзя, и отказ
    громкий, с названием ручки.
    """
    from harness.capture.base import BackendUnavailable
    from harness.perception import describers as mod

    offline = _profile(internet_access=False, model_provider="groq")
    d = mod.from_profile(None, offline)
    ok, why = d.probe()
    assert not ok and "internet_access" in why
    with pytest.raises(BackendUnavailable, match="интернет"):
        d.require()

    local = mod.from_profile("ollama", offline)
    ok2, why2 = local.probe()
    assert "internet_access" not in why2, "локальный узел — не сеть"


def test_model_provider_and_fallback_come_from_the_profile() -> None:
    from harness.perception import describers as mod

    p = _profile(model_provider="gemini", model_fallback=False,
                 internet_access=True)
    assert mod.from_profile(None, p).provider.name == "gemini"
    gate = mod.gate_from_profile(_profile(model_min_novelty=0.42,
                                         model_min_gap_cycles=7))
    assert gate.min_novelty == 0.42 and gate.min_gap == 7


def test_human_speech_cannot_set_goals_unless_declared() -> None:
    """Инвариант 10: реакция только на поведение и объективные величины.

    Ручка `human_speech_affects_goals` — заявленный другой эксперимент, и она
    структурная: смешивать такой прогон с чистым нельзя.
    """
    from harness.behaviour.goals import Candidate, GoalError, GoalStack
    from harness.model.drives import Motivation

    stack = GoalStack(_profile())
    said = Candidate(kind="reach_place", target="PLACE_0001", drive="human",
                     gain=1.0, test=lambda: False, test_text="человек попросил")
    with pytest.raises(GoalError, match="human_speech_affects_goals"):
        stack.push(said, Motivation(_profile()), seq=0, branch="b0")

    allowed = GoalStack(_profile(human_speech_affects_goals=True))
    goal = allowed.push(said, Motivation(_profile()), seq=0, branch="b0")
    assert goal.drive == "human"
    assert (_profile().structure_hash
            != _profile(human_speech_affects_goals=True).structure_hash)


def test_sleep_uses_profile_duration_and_human_trust() -> None:
    from harness.model.consolidation import Consolidator

    c = Consolidator(_profile(sleep_duration_s=12.5, testimony_trust_human=0.3))
    assert c.duration_s == 12.5 and c.trust_human == 0.3
