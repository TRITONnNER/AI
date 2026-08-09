"""Формат журнала v2: слой-инициатор, срез состояния, уровни, линии, вопросы.

Восемь критериев готовности из `TASK-02-JOURNAL.md`, часть 6, по тесту на каждый.
Порядок тестов в файле — порядок критериев в задаче, чтобы сверять было можно
глазами, а не поиском.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from harness.core.clocks import Stamp
from harness.core.journal import (EMPTY_STATE, FORMAT, FORMAT_V1, Actor, ActorLayer,
                                  BranchMeta, FormatError, Journal, JournalError,
                                  Kind, LegacyEntry, LegacyJournal, StateSnapshot,
                                  open_branch)
from harness.core.levels import (Ceiling, Level, LevelError, ReserveError,
                                 SegmentStore, TraceReserve, fork_cost)
from harness.core.profile import from_schema
from harness.session import Recorder, Session

PROFILE = from_schema("тест v2", capture_width=32, capture_height=32)


# --- критерий 1: ни одна запись не создаётся без actor_layer ----------------


def test_criterion_1_actor_layer_has_no_default_in_the_signature() -> None:
    """Проверяется типом, а не проверкой в теле: у аргумента нет значения по умолчанию.

    Именно подписью, потому что проверку в теле функции можно обойти или снять
    одной строкой, а отсутствующий обязательный аргумент — нельзя.
    """
    p = inspect.signature(Journal.append).parameters["actor_layer"]
    assert p.default is inspect.Parameter.empty
    assert p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_criterion_1_entry_dataclass_also_requires_it() -> None:
    """И у самой записи: собрать `Entry` без слоя нельзя даже вручную."""
    from harness.core.journal import Entry

    with pytest.raises(TypeError):
        Entry(seq=0, kind=Kind.NOTE, stamp=Stamp(0, 0), actor=Actor.NONE,  # type: ignore[call-arg]
              profile_hash="x", structure_hash="y")


def test_criterion_1_every_append_call_site_passes_a_layer() -> None:
    """Ни одного вызова `append` без слоя во всём `src/`.

    Статически, а не по факту прохождения тестов: тест ловит только те строки,
    которые исполнились, а путь обхода не должен существовать в коде вообще.
    """
    import ast

    root = Path(__file__).resolve().parent.parent / "src"
    bad: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not isinstance(fn, ast.Attribute) or fn.attr != "append":
                continue
            # Вызов журнала узнаётся по первому аргументу вида `Kind.X`/`EntryKind.X`.
            if not node.args or not isinstance(node.args[0], ast.Attribute):
                continue
            owner = node.args[0].value
            if not (isinstance(owner, ast.Name)
                    and owner.id in {"Kind", "EntryKind"}):
                continue
            if len(node.args) < 4:
                bad.append(f"{path.name}:{node.lineno}: аргументов {len(node.args)}")
    assert not bad, "вызовы append без слоя-инициатора: " + "; ".join(bad)


# --- критерий 2: попытка записать без слоя падает --------------------------


def test_criterion_2_append_without_layer_raises(tmp_path: Path) -> None:
    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        with pytest.raises(TypeError, match="actor_layer"):
            rec.journal.append(Kind.NOTE, Stamp(1, 1), Actor.HUMAN)  # type: ignore[call-arg]


def test_criterion_2_layer_must_be_the_enum_not_a_string(tmp_path: Path) -> None:
    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        with pytest.raises(JournalError, match="набор слоёв закрыт"):
            rec.journal.append(Kind.NOTE, Stamp(1, 1), Actor.HUMAN, "human")  # type: ignore[arg-type]


def test_criterion_2_agent_layers_cannot_be_claimed_by_a_human(tmp_path: Path) -> None:
    """Чужой поступок не попадает в биографию агента как его собственный."""
    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        with pytest.raises(JournalError, match="подделанной атрибуции"):
            rec.journal.append(Kind.GOAL, Stamp(1, 1), Actor.HUMAN, ActorLayer.PLANNER)
        with pytest.raises(JournalError, match="человек действует слоем human"):
            rec.journal.append(Kind.NOTE, Stamp(1, 1), Actor.HUMAN, ActorLayer.NONE)
        with pytest.raises(JournalError, match="человек действует слоем human"):
            rec.journal.append(Kind.GOAL, Stamp(1, 1), Actor.AGENT, ActorLayer.HUMAN)


def test_layer_is_written_and_read_back(tmp_path: Path) -> None:
    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        rec.journal.append(Kind.GOAL, Stamp(1, 1), Actor.AGENT, ActorLayer.DRIVE,
                           event={"code": "set"})
    with Session.open(tmp_path / "s") as s:
        layers = [e.actor_layer for e in s.journal]
        assert ActorLayer.DRIVE in layers
        assert all(isinstance(x, ActorLayer) for x in layers)
        s.journal.verify()


# --- критерий 3: метрика конфабуляции даёт число ----------------------------


def test_criterion_3_confabulation_gives_a_number(tmp_path: Path) -> None:
    """Планировщик объяснил три действия, два из них начал не он."""
    from harness.model.confabulation import journal_reason, measure

    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        j = rec.journal
        journal_reason(j, Stamp(1, 1), "R1", claims_layer=ActorLayer.PLANNER)
        journal_reason(j, Stamp(2, 2), "R2", claims_layer=ActorLayer.PLANNER)
        journal_reason(j, Stamp(3, 3), "R3", claims_layer=ActorLayer.PLANNER)
        # Первое действие планировщик и правда начал.
        j.append(Kind.PLAN, Stamp(4, 4), Actor.AGENT, ActorLayer.PLANNER,
                 state=StateSnapshot(stated_reason_id="R1"), event={"code": "s"})
        # Два следующих начал рефлекс, а объяснение приписано планировщику.
        j.append(Kind.PLAN, Stamp(5, 5), Actor.AGENT, ActorLayer.REFLEX,
                 state=StateSnapshot(stated_reason_id="R2"), event={"code": "s"})
        j.append(Kind.PLAN, Stamp(6, 6), Actor.AGENT, ActorLayer.REFLEX,
                 state=StateSnapshot(stated_reason_id="R3"), event={"code": "s"})

    with Session.open(tmp_path / "s") as s:
        c = measure(s.journal)
    assert c.explained == 3 and c.matched == 1 and c.mismatched == 2
    assert c.rate == pytest.approx(2 / 3)
    assert c.by_pair == {"planner→reflex": 2}
    assert "конфабуляция: 67%" in c.line()


def test_confabulation_without_explanations_is_not_zero(tmp_path: Path) -> None:
    """Ноль означал бы «объяснял и попадал». Не объяснял — это `None`."""
    from harness.model.confabulation import measure

    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        rec.journal.append(Kind.GOAL, Stamp(1, 1), Actor.AGENT, ActorLayer.DRIVE,
                           event={"code": "set"})
    with Session.open(tmp_path / "s") as s:
        c = measure(s.journal)
    assert c.rate is None
    assert c.absent_reason and "объяснений в этом прогоне не было" in c.absent_reason
    assert "не измерена" in c.line()


def test_stated_reason_does_not_reach_rebuild(tmp_path: Path) -> None:
    """Реплика — выражение, а не показание: пересборка её не видит (инвариант 10)."""
    from harness.model.confabulation import journal_reason
    from harness.model.rebuild import rebuild_from_journal

    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        for i in range(4):
            rec.record_frame(np.full((32, 32), i * 9, dtype=np.uint8))
        before = rebuild_from_journal(rec.journal).beliefs.fingerprint()
        journal_reason(rec.journal, rec.clocks.stamp(), "R1",
                       claims_layer=ActorLayer.PLANNER,
                       detail={"note": "я делаю это ради цели"})
        after = rebuild_from_journal(rec.journal).beliefs.fingerprint()
    assert before == after


def test_claimed_layer_is_not_the_records_own_layer(tmp_path: Path) -> None:
    """Заявление лежит в событии, а не в `actor_layer` записи.

    Иначе метрика сравнивала бы утверждение с самим собой и всегда давала нуль.
    """
    from harness.model.confabulation import CLAIMED_LAYER, journal_reason

    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        e = journal_reason(rec.journal, Stamp(1, 1), "R1",
                           claims_layer=ActorLayer.PLANNER)
    assert e.actor_layer is ActorLayer.NONE
    assert e.event[CLAIMED_LAYER] == "planner"


# --- критерий 4: ветка занимает место по расхождению ------------------------


def test_criterion_4_fork_costs_only_the_divergence(tmp_path: Path) -> None:
    """Общий префикс делится физически: платишь за расхождение, не за длину."""
    store = SegmentStore(tmp_path / "seg")
    prefix = [bytes([i]) * 4096 for i in range(20)]
    first = fork_cost(store, Level.KEYFRAMES, prefix)
    assert first.shared_segments == 0 and first.new_segments == 20

    # Ветка от середины: первые десять сегментов те же, дальше расхождение.
    on_disk = store.bytes_on_disk()
    branch = prefix[:10] + [bytes([200 + i]) * 4096 for i in range(3)]
    second = fork_cost(store, Level.KEYFRAMES, branch)
    assert second.shared_segments == 10
    assert second.new_segments == 3
    # Занятое место выросло на размер расхождения, а не на размер журнала.
    assert store.bytes_on_disk() - on_disk == 3 * 4096
    assert second.new_bytes < first.new_bytes / 3

    # Самая резкая форма того же: ветка длиной во весь журнал, расхождение в один
    # сегмент, — рост на один сегмент. Замер на кадрах реального размера
    # (`tools/`, черновик) даёт то же: 200 сегментов, 199 общих, диск +56 КиБ.
    on_disk = store.bytes_on_disk()
    tail = prefix[:19] + [b"\xff" * 4096]
    third = fork_cost(store, Level.KEYFRAMES, tail)
    assert third.shared_segments == 19 and third.new_segments == 1
    assert store.bytes_on_disk() - on_disk == 4096


def test_identical_content_is_stored_once(tmp_path: Path) -> None:
    store = SegmentStore(tmp_path / "seg")
    a = store.put(Level.FEATURES, b"one and the same")
    b = store.put(Level.FEATURES, b"one and the same")
    assert a.segment_id == b.segment_id
    assert store.count_at(Level.FEATURES) == 1


def test_journal_reference_survives_demotion(tmp_path: Path) -> None:
    """Ссылка в журнале неизменяема, поэтому понижение не имеет права её ломать.

    Это то, из-за чего личность сегмента отделена от адреса содержимого. При
    понижении содержимое другое, значит и адрес другой; если бы ссылка была
    адресом, понижение обязано было бы переписать запись журнала — а журнал
    дозаписывается и не правится (инвариант 1). Ссылка молча перестала бы
    разрешаться.
    """
    store = SegmentStore(tmp_path / "seg")
    placement = store.put(Level.STREAM, b"z" * 10_000)
    ref = placement.ref()                      # это уходит в запись журнала
    assert store.get(ref) == b"z" * 10_000

    store.demote(ref, b"z" * 800)
    # Та же самая ссылка, не тронутая, по-прежнему разрешается — но уже в другое.
    assert store.get(ref) == b"z" * 800
    assert store.where(ref).level is Level.KEYFRAMES
    assert store.where(ref).born_level is Level.STREAM
    assert store.where(ref).demoted


def test_demotion_is_a_ladder_not_a_cliff(tmp_path: Path) -> None:
    store = SegmentStore(tmp_path / "seg")
    ref = store.put(Level.STREAM, b"z" * 10_000).ref()
    assert store.demote(ref, b"z" * 800).level is Level.KEYFRAMES
    assert store.demote(ref, b"z" * 40).level is Level.FEATURES
    trace = store.demote(ref, None)
    assert trace.level is Level.TRACE and not trace.readable
    # Опущенный до следа сегмент всё ещё существует как ссылка — это понижение,
    # а не удаление: запись о том, что он был, осталась.
    assert trace.segment_id == ref.segment_id
    assert trace.born_level is Level.STREAM
    with pytest.raises(LevelError, match="опущен до следа"):
        store.get(ref)


def test_demotion_history_is_kept_because_the_index_appends(tmp_path: Path) -> None:
    """По ней потом считается «сожаление о вытеснении»."""
    from harness.core.levels import demotion_history

    store = SegmentStore(tmp_path / "seg")
    ref = store.put(Level.STREAM, b"q" * 4000).ref()
    store.demote(ref, b"q" * 300)
    store.demote(ref, b"q" * 20)
    steps = [p.level for p in demotion_history(store, ref.segment_id)]
    assert steps == [Level.STREAM, Level.KEYFRAMES, Level.FEATURES]


def test_index_survives_reopening(tmp_path: Path) -> None:
    store = SegmentStore(tmp_path / "seg")
    ref = store.put(Level.KEYFRAMES, b"k" * 500).ref()
    store.demote(ref, b"k" * 20)
    again = SegmentStore(tmp_path / "seg")
    assert again.where(ref).level is Level.FEATURES
    assert again.get(ref) == b"k" * 20


def test_segment_cannot_claim_more_detail_than_it_was_born_with() -> None:
    from harness.core.levels import Placement

    with pytest.raises(LevelError, match="подробностей стало больше"):
        Placement("abc", Level.STREAM, "abc", 1, Level.FEATURES)


def test_shared_blob_is_not_deleted_out_from_under_another_branch(tmp_path: Path) -> None:
    """Адресация по содержимому означает, что блок может быть общим.

    Удалить его при понижении одного сегмента значило бы обрушить чужую ветку —
    «делим префикс физически», обращённое во вред.
    """
    store = SegmentStore(tmp_path / "seg")
    shared = b"s" * 700
    a = store.put(Level.KEYFRAMES, shared).ref()
    # Второй сегмент, размещённый на том же блоке: другая личность, тот же контент.
    from harness.core.levels import Placement, content_id
    b = Placement("другая-личность", Level.KEYFRAMES, content_id(shared),
                  len(shared), Level.KEYFRAMES)
    store._note(b)
    store.demote(a, None)
    assert store.get(b.ref()) == shared


# --- критерий 5: диск останавливает 2–3 и не трогает нулевой ---------------


def test_criterion_5_full_disk_blocks_upper_levels_only() -> None:
    ceiling = Ceiling(cap_mb=100.0, reserve_mb=10.0)
    assert ceiling.blocks(Level.STREAM, used_mb=96.0)
    assert ceiling.blocks(Level.KEYFRAMES, used_mb=96.0)
    # Нулевой не блокируется никогда и ни при каком заполнении.
    assert not ceiling.blocks(Level.TRACE, used_mb=96.0)
    assert not ceiling.blocks(Level.TRACE, used_mb=10_000.0)
    assert ceiling.state(96.0) == "запись уровней 2–3 заблокирована"
    assert ceiling.state(85.0) == "фоновое понижение при следующем сне"


def test_criterion_5_eviction_has_no_path_to_level_zero(tmp_path: Path) -> None:
    """Разделение архитектурное: хранилище сегментов про нулевой уровень не знает."""
    from harness.core.levels import LADDER, Placement, SegmentRef

    store = SegmentStore(tmp_path / "seg")
    # Положить причинную запись сюда нельзя вообще: у хранилища нет такого пути.
    with pytest.raises(ReserveError, match="отдельном зарезервированном"):
        store.put(Level.TRACE, "причинная запись".encode("utf-8"))

    # И понизить сегмент, уже доехавший до следа, тоже нельзя: дальше некуда.
    ref = store.put(Level.FEATURES, b"f" * 100).ref()
    store.demote(ref, None)
    assert store.where(ref).level is Level.TRACE
    with pytest.raises(ReserveError, match="не понижается"):
        store.demote(ref, b"lower still")

    # У лестницы нет ступени, ведущей из нулевого уровня.
    assert Level.TRACE not in LADDER
    assert not Level.TRACE.evictable
    # И у хранилища нет ни одного атрибута, ведущего к резерву нулевого уровня.
    assert not any("trace" in name.lower() or "reserve" in name.lower()
                   for name in vars(store))


def test_hours_left_is_reported_not_percentage() -> None:
    """Проценты ни о чём не говорят, часы говорят обо всём (STORAGE.md, 5)."""
    ceiling = Ceiling(cap_mb=100.0)
    assert ceiling.hours_left(used_mb=20.0, mb_per_hour=4.0) == 20.0
    # Скорость записи неизвестна — выдумывать её нельзя.
    assert ceiling.hours_left(used_mb=20.0, mb_per_hour=0.0) is None


def test_trace_reserve_reports_exhaustion_without_deleting(tmp_path: Path) -> None:
    reserve = TraceReserve(tmp_path / "trace", reserve_mb=0.001)
    (tmp_path / "trace" / "entries.jsonl").write_bytes(b"x" * 5000)
    assert reserve.exhausted
    # Файл на месте: при исчерпании резерва система сообщает, а не чистит.
    assert (tmp_path / "trace" / "entries.jsonl").stat().st_size == 5000


# --- критерий 6: чтение v1 с запросом v2 падает внятно ---------------------


def _make_v1_branch(root: Path) -> Path:
    """Ветка в формате v1: как её писал прежний код, без слоя и без среза."""
    path = root / "branches" / "000-v1branch"
    path.mkdir(parents=True)
    meta = BranchMeta("000-v1branch", PROFILE.structure_hash, PROFILE.as_dict(),
                      None, None, None, "старый корпус", format=FORMAT_V1,
                      read_only=True)
    (path / Journal.META).write_text(
        json.dumps(meta.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    line = {"seq": 0, "kind": "action", "stamp": {"t_self": 1, "t_world": 1,
                                                  "t_content": None},
            "actor": "agent", "profile_hash": PROFILE.profile_hash,
            "structure_hash": PROFILE.structure_hash,
            "event": {"code": "delivered"}, "prev": "0" * 32, "digest": "aa"}
    (path / Journal.ENTRIES).write_text(
        json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_criterion_6_v1_refuses_a_metric_that_needs_v2(tmp_path: Path) -> None:
    from harness.model.confabulation import measure

    path = _make_v1_branch(tmp_path / "journal")
    legacy = LegacyJournal(path)
    with pytest.raises(FormatError) as exc:
        measure(legacy)
    text = str(exc.value)
    assert "формате v1" in text
    assert "actor_layer" in text and "stated_reason_id" in text
    # Сообщение объясняет, что именно неизмеримо, а не просто «поле отсутствует».
    assert "метрика конфабуляции" in text
    assert "Нужен прогон в формате v2" in text


def test_criterion_6_v1_entry_has_no_layer_attribute_at_all(tmp_path: Path) -> None:
    """`AttributeError`, а не `ActorLayer.NONE`: заглушка страшнее падения."""
    path = _make_v1_branch(tmp_path / "journal")
    entries = list(LegacyJournal(path))
    assert len(entries) == 1 and isinstance(entries[0], LegacyEntry)
    with pytest.raises(AttributeError):
        entries[0].actor_layer  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        entries[0].state  # type: ignore[attr-defined]


def test_criterion_6_v2_reader_refuses_a_v1_branch(tmp_path: Path) -> None:
    path = _make_v1_branch(tmp_path / "journal")
    with pytest.raises(FormatError, match="формате v1"):
        Journal(path, mode="r")
    # И наоборот: v1-читатель не берётся за v2.
    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True):
        pass
    v2 = sorted((tmp_path / "s" / "journal" / "branches").iterdir())[0]
    with pytest.raises(FormatError, match="а не v1"):
        LegacyJournal(v2)


def test_open_branch_picks_the_reader_from_the_data(tmp_path: Path) -> None:
    path = _make_v1_branch(tmp_path / "journal")
    assert isinstance(open_branch(path), LegacyJournal)
    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True):
        pass
    v2 = sorted((tmp_path / "s" / "journal" / "branches").iterdir())[0]
    assert isinstance(open_branch(v2), Journal)


def test_v1_branch_is_read_only_in_the_data(tmp_path: Path) -> None:
    """Пометка стоит в самих данных, а не в документации."""
    path = _make_v1_branch(tmp_path / "journal")
    meta = BranchMeta.from_dict(json.loads((path / Journal.META).read_text()))
    assert meta.read_only and meta.is_v1
    assert LegacyJournal(path).mode == "r"
    assert not hasattr(LegacyJournal(path), "append")


def test_new_branches_are_v2_and_carry_the_format(tmp_path: Path) -> None:
    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        assert rec.journal.meta.format == FORMAT
        e = rec.journal.append(Kind.NOTE, Stamp(1, 1), Actor.HUMAN, ActorLayer.HUMAN,
                               event={"code": "note"})
    assert e.payload()["format"] == FORMAT


# --- критерий 7: Hypothesis с test=None попадает в deferred ----------------


def test_criterion_7_question_is_legal_and_deferred() -> None:
    from harness.model.beliefs import Hypothesis, Origin, Provenance, State

    q = Hypothesis.question("ENT_7A3F|made_by|?", 0.3,
                            Provenance(Origin.HUNCH, "b", 1),
                            because="создателя нельзя наблюдать: его тут нет",
                            reopens_on=("встреча с наблюдаемым деятелем",))
    assert q.test is None and q.is_question
    assert q.state is State.DEFERRED
    assert q.as_dict()["state"] == "deferred"


def test_criterion_7_three_outcomes_not_two() -> None:
    from harness.model.beliefs import Hypothesis, Origin, Provenance, State

    pv = Provenance(Origin.HUNCH, "b", 1)
    h = Hypothesis("c", "нажать и посмотреть", 0.5, pv)
    # Гипотеза с тестом, который ещё не проводили, ни в один исход не попадает:
    # иначе перечень непонятого забился бы обычной очередью дел.
    assert h.state is None
    assert h.resolved(True).state is State.VERIFIED
    assert h.resolved(False).state is State.REFUTED
    assert {s.value for s in State} == {"verified", "refuted", "deferred"}


def test_criterion_7_question_reopens_when_a_test_appears() -> None:
    from harness.model.beliefs import BeliefError, Hypothesis, Origin, Provenance

    q = Hypothesis.question("c", 0.4, Provenance(Origin.HUNCH, "b", 1),
                            because="читать я пока не умею",
                            reopens_on=("умение читать",))
    with pytest.raises(BeliefError, match="нельзя подтвердить"):
        q.confirm(True, Provenance(Origin.EXPERIENCE, "b", 2))
    reopened = q.with_test("прочитать надпись")
    assert not reopened.is_question
    assert reopened.reopens_on == ("умение читать",)
    b = reopened.confirm(True, Provenance(Origin.EXPERIENCE, "b", 3))
    assert b.n_experience == 1


def test_criterion_7_question_without_a_reason_is_refused() -> None:
    """Отсутствие теста законно, отсутствие причины — нет."""
    from harness.model.beliefs import BeliefError, Hypothesis, Origin, Provenance

    with pytest.raises(BeliefError, match="перечнем того, что"):
        Hypothesis("c", None, 0.5, Provenance(Origin.HUNCH, "b", 1))


# --- срез состояния: None это не ноль --------------------------------------


def test_state_snapshot_absence_is_not_zero(tmp_path: Path) -> None:
    """`prediction_error = None` и `= 0.0` — разные утверждения."""
    absent = StateSnapshot()
    perfect = StateSnapshot(prediction_error=0.0)
    assert absent.is_empty and not perfect.is_empty
    assert absent.as_dict()["prediction_error"] is None
    assert perfect.as_dict()["prediction_error"] == 0.0

    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        rec.journal.append(Kind.GOAL, Stamp(1, 1), Actor.AGENT, ActorLayer.DRIVE,
                           event={"code": "set"})
        rec.journal.append(Kind.GOAL, Stamp(2, 2), Actor.AGENT, ActorLayer.DRIVE,
                           state=perfect, event={"code": "set"})
    with Session.open(tmp_path / "s") as s:
        errors = [e.state.prediction_error for e in s.journal.entries([Kind.GOAL])]
    assert errors == [None, 0.0]


def test_state_snapshot_keeps_all_keys_so_absence_is_visible() -> None:
    """Отсутствующий ключ читатель примет за старый формат, `null` — не примет."""
    keys = set(StateSnapshot().as_dict())
    assert keys == {"prediction_error", "drives", "mood", "goal_id",
                    "goal_transition", "stated_reason_id"}


def test_state_snapshot_refuses_impossible_values() -> None:
    with pytest.raises(JournalError, match="настроение вне"):
        StateSnapshot(mood=(2.0, 0.0))
    with pytest.raises(JournalError, match="valence, arousal"):
        StateSnapshot(mood=(0.1,))  # type: ignore[arg-type]
    with pytest.raises(JournalError, match="отрицательная"):
        StateSnapshot(prediction_error=-0.1)


def test_wall_clock_is_an_anchor_not_a_fourth_clock(tmp_path: Path) -> None:
    """Он не проверяется на монотонность и не участвует в порядке записей."""
    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        rec.journal.append(Kind.NOTE, Stamp(1, 1), Actor.HUMAN, ActorLayer.HUMAN,
                           event={"code": "note"}, wall_clock=1000.0)
        # Настенное время пошло назад — журнал это принимает: на разных машинах и
        # после правки системного времени так и бывает, а порядок задают три часов.
        rec.journal.append(Kind.NOTE, Stamp(2, 2), Actor.HUMAN, ActorLayer.HUMAN,
                           event={"code": "note"}, wall_clock=500.0)
    with Session.open(tmp_path / "s") as s:
        s.journal.verify()
        clocks = [e.wall_clock for e in s.journal.entries([Kind.NOTE])]
    assert clocks == [1000.0, 500.0]


def test_rebuild_does_not_depend_on_wall_clock(tmp_path: Path) -> None:
    """Иначе пересборка перестала бы быть воспроизводимой между машинами."""
    from harness.model.rebuild import rebuild_from_journal

    prints = []
    for wall in (1_000.0, 9_999_999.0):
        root = tmp_path / f"s{int(wall)}"
        with Recorder(root, profile=PROFILE, source="t", synthetic=True) as rec:
            from harness.core.action import Action
            for i in range(5):
                rec.journal.append(
                    Kind.ACTION, Stamp(i + 1, i + 1), Actor.AGENT, ActorLayer.DRIVE,
                    action=Action.key("OUT_0A11", 100), wall_clock=wall + i,
                    event={"code": "delivered", "responded": i % 2 == 0})
            prints.append(rebuild_from_journal(rec.journal).beliefs.fingerprint())
    assert prints[0] == prints[1]


# --- линия -----------------------------------------------------------------


def test_lineage_opens_a_new_root_and_prints_what_leaked(tmp_path: Path) -> None:
    from harness.core.lineage import Lineage, new_lineage, read_lineage

    lin, journal = new_lineage(tmp_path / "lin12", PROFILE,
                               reason="смена формата записи v1 → v2",
                               world_seed=5)
    journal.close()
    assert lin.id.startswith("LIN_")
    assert (tmp_path / "lin12" / Lineage.MANIFEST_FILE).exists()
    # Манифест непереносимого печатается всегда, и оператор в нём есть.
    what = {lk.what for lk in lin.leaks}
    assert "оператор" in what and "формат журнала" in what
    assert "версия кода" in what and "сид и состояние мира" in what
    text = lin.as_text()
    assert "v1 → v2" in text and "манифест непереносимого" in text
    assert read_lineage(tmp_path / "lin12").id == lin.id


def test_lineage_defaults_everything_to_fresh() -> None:
    from harness.core.lineage import MANIFEST_ITEMS, Inherit, LineageManifest

    man = LineageManifest()
    assert set(man.items) == set(MANIFEST_ITEMS)
    assert all(it.mode is Inherit.FRESH for it in man.items.values())
    assert man.inherited == []


def test_lineage_inheritance_needs_a_source() -> None:
    from harness.core.lineage import Inherit, LineageError, ManifestItem

    with pytest.raises(LineageError, match="не наследование, а надежда"):
        ManifestItem(Inherit.AS_TESTIMONY)
    with pytest.raises(LineageError, match="либо с нуля"):
        ManifestItem(Inherit.FRESH, ref="откуда-то")


def test_lineage_id_lands_in_every_record(tmp_path: Path) -> None:
    from harness.core.lineage import new_lineage

    lin, journal = new_lineage(tmp_path / "lin", PROFILE, reason="проверка",
                               world_seed=1)
    journal.append(Kind.GOAL, Stamp(1, 1), Actor.AGENT, ActorLayer.DRIVE,
                   event={"code": "set"})
    journal.close()
    reopened = Journal.open_latest(tmp_path / "lin" / "journal")
    assert all(e.lineage_id == lin.id for e in reopened)
    assert all(e.branch_id == reopened.meta.branch_id for e in reopened)


def test_lineage_refuses_a_dirty_root(tmp_path: Path) -> None:
    from harness.core.lineage import LineageError, new_lineage

    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "чужое.txt").write_text("тут уже живут", encoding="utf-8")
    with pytest.raises(LineageError, match="прибавлять, а не"):
        new_lineage(tmp_path / "busy", PROFILE, reason="поверх чужого")
