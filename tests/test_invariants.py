"""По одному тесту на каждый инвариант из CLAUDE.md.

«Каждый инвариант из списка выше покрыт тестом» — правило работы, а не пожелание.
Тесты здесь проверяют не то, что нужный текст написан в документации, а то, что
нарушение невозможно или падает громко.

Инвариант 6 (файрвол восприятия) в вехе 0 не может быть проверен по существу:
большой модели нет вообще. Вместо вакуумно зелёного теста стоит растяжка — она
падает в тот момент, когда планировщик появляется, и требует прибора к нему.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from harness.core.action import Action, ActionError, Reversibility
from harness.core.clocks import ClockError, Clocks, Stamp
from harness.core.journal import Actor, Journal, JournalError, Kind, TamperError
from harness.core.profile import MILESTONE_0, Profile, ProfileError
from harness.core.symbols import SymbolError, Symbolizer, is_symbol
from harness.session import Recorder, Session

SRC = Path(__file__).resolve().parent.parent / "src"


# --- 1. Журнал только дозаписывается ---------------------------------------


def test_invariant_1_journal_is_append_only(tmp_path: Path) -> None:
    """Правка записи рвёт цепочку хешей и обнаруживается точно."""
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="test",
                  synthetic=True) as rec:
        for i in range(5):
            rec.record_frame(np.full((16, 16), i * 10, dtype=np.uint8))
        entries_path = Path(rec.journal.path) / Journal.ENTRIES

    with Session.open(tmp_path / "s") as s:
        s.journal.verify()          # целый журнал проходит

    lines = entries_path.read_text(encoding="utf-8").splitlines()
    doctored = json.loads(lines[2])
    doctored["stamp"]["t_world"] = 999           # «уточнили» время задним числом
    lines[2] = json.dumps(doctored, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))
    entries_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with Session.open(tmp_path / "s") as s:
        with pytest.raises(TamperError, match="хеш не сходится"):
            s.journal.verify()


def test_invariant_1_deleted_line_is_detected(tmp_path: Path) -> None:
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="test",
                  synthetic=True) as rec:
        for i in range(5):
            rec.record_frame(np.full((16, 16), i, dtype=np.uint8))
        entries_path = Path(rec.journal.path) / Journal.ENTRIES

    lines = entries_path.read_text(encoding="utf-8").splitlines()
    del lines[3]
    entries_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with Session.open(tmp_path / "s") as s:
        with pytest.raises(TamperError):
            s.journal.verify()


def test_invariant_1_beliefs_are_rebuildable(session: Session) -> None:
    """Производное состояние выводится из журнала, а не хранится отдельно.

    В вехе 0 «убеждение» одно: карта тела, то есть какие выходы отвечали. Она
    собирается из записей действий — и должна собираться дважды одинаково.
    """
    def rebuild(s: Session) -> dict[str, int]:
        out: dict[str, int] = {}
        for _, act in s.actions():
            key = act.output or f"move:{act.dx},{act.dy}"
            out[key] = out.get(key, 0) + 1
        return out

    first = rebuild(session)
    second = rebuild(session)
    assert first == second
    assert first, "в корпусе нет ни одного действия — проверять нечего"


# --- 2. Три часов и хеш профиля в каждой записи -----------------------------


def test_invariant_2_every_entry_has_stamp_and_hash(session: Session) -> None:
    n = 0
    for e in session.journal:
        assert isinstance(e.stamp, Stamp)
        assert e.stamp.t_self >= 0 and e.stamp.t_world >= 0
        assert e.profile_hash and e.structure_hash
        n += 1
    assert n > 0


def test_invariant_2_entry_without_stamp_is_impossible(tmp_path: Path) -> None:
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="test",
                  synthetic=True) as rec:
        with pytest.raises(JournalError, match="без трёх часов"):
            rec.journal.append(Kind.NOTE, None, Actor.NONE)  # type: ignore[arg-type]


def test_invariant_2_t_content_may_be_none() -> None:
    """None у времени содержимого — значение, а не ошибка."""
    assert Stamp(1, 2, None).t_content is None
    with pytest.raises(ClockError):
        Stamp(1, 2, -0.5)


def test_invariant_2_clocks_do_not_go_back() -> None:
    c = Clocks()
    c.set_world(10)
    with pytest.raises(ClockError, match="назад"):
        c.set_world(9)


# --- 3. Мир никогда не ставится на паузу -----------------------------------


def test_invariant_3_stop_does_not_pause_journal(tmp_path: Path) -> None:
    """СТОП отключает эффекторы, а запись продолжается."""
    from harness.inject.base import Injector, InjectionSink, NullDevice
    from harness.inject.stop import StopSwitch

    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="test",
                  synthetic=True) as rec:
        stop = StopSwitch(rec.journal)
        inj = Injector(InjectionSink(NullDevice(), sleep=lambda _: None), rec.journal,
                       stop=stop)
        rec.record_frame(np.zeros((8, 8), dtype=np.uint8))
        stop.engage("тест", rec.clocks.stamp())
        out = inj.submit(Action.key("OUT_0A11", 100), rec.clocks.stamp())
        assert out.stopped and not out.delivered
        # мир идёт: кадры продолжают писаться после стопа
        rec.record_frame(np.ones((8, 8), dtype=np.uint8))
        rec.record_frame(np.full((8, 8), 2, dtype=np.uint8))

    with Session.open(tmp_path / "s") as s:
        s.journal.verify()
        kinds = [e.kind for e in s.journal]
        assert Kind.STOP in kinds
        # после стопа есть кадры
        after_stop = kinds[kinds.index(Kind.STOP):]
        assert Kind.FRAME in after_stop


def test_invariant_3_no_blocking_sleep_in_hold() -> None:
    """Удержание нарезано на срезы: основной цикл не блокируется целиком.

    Проверяем не «нет sleep», а то, что sleep вызывается многими короткими
    порциями, — именно это даёт прерываемость.
    """
    from harness.inject.base import InjectionSink, NullDevice

    slept: list[float] = []
    sink = InjectionSink(NullDevice(), slice_ms=5.0, sleep=slept.append)
    completed, _ = sink.hold(["OUT_0A11"], 100, abort=lambda: False)
    assert completed
    assert len(slept) >= 20, f"удержание не нарезано: {len(slept)} вызовов сна"
    assert max(slept) <= 0.0051


# --- 4. Агент не получает координат, названий, разметки ---------------------


def test_invariant_4_action_rejects_key_names() -> None:
    with pytest.raises(ActionError, match="непрозрачный идентификатор"):
        Action.key("W", 100)
    with pytest.raises(ActionError, match="непрозрачный идентификатор"):
        Action.key("Space", 100)


def test_invariant_4_mask_rejects_key_names() -> None:
    from harness.inject.mask import InputMask, MaskError

    m = InputMask()
    with pytest.raises(MaskError, match="не выход"):
        m.block("F5")


def test_invariant_4_percept_carries_no_labels() -> None:
    from harness.agentside.perception import Percept, PerceptError

    ok = Percept(Stamp(1, 1, None), np.zeros((4, 4), dtype=np.uint8),
                 symbols=("SYM_1A2B",))
    ok.assert_opaque()
    with pytest.raises(PerceptError, match="не символ"):
        Percept(Stamp(1, 1, None), np.zeros((4, 4), dtype=np.uint8),
                symbols=("Здоровье",))


def test_invariant_4_body_exposes_no_meaning() -> None:
    """У тела есть адреса выходов и ни одного слова о том, что они делают."""
    from harness.agentside.body import Body

    body = Body(["OUT_0A11", "OUT_0B22"], submit=lambda a: a)
    text = json.dumps({"outputs": list(body.outputs)}, ensure_ascii=False)
    for forbidden in ("W", "Space", "Escape", "прыжок", "jump", "attack"):
        assert forbidden not in text
    assert not hasattr(body, "names") and not hasattr(body, "keymap")


# --- 5. Весь текст хешируется в непрозрачный символ ------------------------


def test_invariant_5_symbolizer_has_no_inverse() -> None:
    sym = Symbolizer("s0")
    s = sym.symbolize("Здоровье: 20")
    assert is_symbol(s)
    for attr in ("decrypt", "reverse", "table", "lookup", "inverse"):
        assert not hasattr(sym, attr), f"у символизатора есть {attr}: он перестал быть односторонним"


def test_invariant_5_same_text_same_symbol_different_text_differs() -> None:
    sym = Symbolizer("s0")
    assert sym.symbolize("Здоровье") == sym.symbolize("  Здоровье  ")
    assert sym.symbolize("Опыт: 27") != sym.symbolize("Опыт: 28")


def test_invariant_5_salt_change_changes_all_symbols() -> None:
    a, b = Symbolizer("s0"), Symbolizer("s1")
    assert a.symbolize("Голод") != b.symbolize("Голод")


def test_invariant_5_journal_refuses_plain_text_in_perception(tmp_path: Path) -> None:
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="test",
                  synthetic=True) as rec:
        rec.record_perception({"symbols": ["SYM_1A2B"], "count": 1})
        with pytest.raises(SymbolError, match="читаемый текст"):
            rec.record_perception({"symbols": ["Здоровье"]})


def test_invariant_5_decryption_table_lives_only_in_debug(corpus: Path) -> None:
    from harness.debug.channel import DebugChannel

    table = DebugChannel(corpus / "debug", mode="r").symbol_table()
    assert table, "в отладочном потоке нет таблицы расшифровки"
    # Ни одна настоящая надпись не встречается в журнале
    journal_text = (next((corpus / "journal" / "branches").iterdir())
                    / Journal.ENTRIES).read_text(encoding="utf-8")
    for symbol, text in table.items():
        assert text not in journal_text, f"надпись {text!r} утекла в журнал"


# --- 6. Файрвол восприятия -------------------------------------------------


def test_invariant_6_tripwire_no_planner_without_firewall_instrument() -> None:
    """Растяжка, а не проверка.

    В вехе 0 большой модели нет, поэтому проверять, что она отвечает только на
    «что я вижу», буквально не на чем. Тест падает в тот момент, когда в дереве
    появляется планировщик или клиент модели, и требует вместе с ним прибор:
    запись сырого запроса и сырого ответа плюс проверку ответа на императивы.
    Это то, чего не хватало в дизайне пульта (DESIGN-REVIEW-CONSOLE.md, пункт 2).
    """
    suspicious: list[str] = []
    for path in SRC.rglob("*.py"):
        name = path.stem.lower()
        if any(k in name for k in ("planner", "vlm", "llm", "planning", "model_client")):
            suspicious.append(str(path.relative_to(SRC)))
        text = path.read_text(encoding="utf-8")
        for marker in ("anthropic", "openai", "def ask_model", "class Planner"):
            if marker in text:
                suspicious.append(f"{path.relative_to(SRC)}: {marker}")

    firewall = list(SRC.rglob("*firewall*.py"))
    assert not suspicious or firewall, (
        "появился планировщик или клиент большой модели: "
        f"{suspicious}. Инвариант 6 требует прибора на файрвол — модуля "
        "harness/*/firewall*.py, который записывает сырой вопрос и сырой ответ "
        "и проверяет ответ на «что делать». Без него нарушение файрвола "
        "невидимо, а веха 1 начинается именно с него"
    )


# --- 7. У каждого убеждения есть происхождение -----------------------------


def test_invariant_7_reversibility_carries_mu_sigma_n() -> None:
    """В вехе 0 единственная оценка — обратимость; у неё μ, σ, n обязательны."""
    r = Reversibility()
    assert (r.mu, r.sigma, r.n) == (0.0, 1.0, 0)
    assert not r.is_known
    r2 = r.observe(True).observe(False)
    assert r2.n == 2 and 0 < r2.mu < 1 and r2.sigma > 0


def test_invariant_7_unknown_cannot_masquerade_as_known() -> None:
    with pytest.raises(ActionError, match="незнание выглядит как знание"):
        Reversibility(mu=0.9, sigma=0.1, n=0)


def test_invariant_7_action_records_provenance_of_outcome(session: Session) -> None:
    """У каждой записи действия есть исход и источник: кто и через что."""
    seen = 0
    for entry, _ in session.actions():
        assert entry.event.get("code") in {"delivered", "masked", "stopped", "failed"}
        assert "device" in entry.event
        assert entry.actor in (Actor.AGENT, Actor.HUMAN)
        seen += 1
    assert seen > 0


# --- 8. Действие — это (key, duration_ms, modifiers) -----------------------


def test_invariant_8_no_discrete_press() -> None:
    with pytest.raises(ActionError, match="дискретное событие"):
        Action.key("OUT_0A11", 0)


def test_invariant_8_duration_is_integer_ms() -> None:
    with pytest.raises(ActionError, match="целым числом"):
        Action.key("OUT_0A11", 12.5)  # type: ignore[arg-type]


def test_invariant_8_modifiers_are_part_of_action() -> None:
    a = Action.key("OUT_0A11", 340, modifiers=("MOD_1B2C",))
    assert a.modifiers == ("MOD_1B2C",)
    assert a.duration_ms == 340
    assert "340" in str(a)


def test_invariant_8_journal_roundtrip_keeps_duration(session: Session) -> None:
    for _, act in session.actions():
        assert act.duration_ms > 0 or act.kind.value == "nothing"


# --- 9. У каждого действия есть оценка обратимости -------------------------


def test_invariant_9_reversibility_is_mandatory_field() -> None:
    a = Action.key("OUT_0A11", 100)
    assert a.reversibility is not None
    assert a.caution == 1.0, "незнание обратимости обязано давать максимум осторожности"


def test_invariant_9_caution_comes_from_reversibility_only() -> None:
    """Осторожность выводится из «умею ли откатить», а не из списка запретов."""
    known_safe = Action.key("OUT_0A11", 100, reversibility=Reversibility(1.0, 0.0, 20))
    known_bad = Action.key("OUT_0A11", 100, reversibility=Reversibility(0.0, 0.0, 20))
    assert known_safe.caution < known_bad.caution
    assert known_safe.caution == 0.0


def test_invariant_9_no_forbidden_list_anywhere() -> None:
    """В коде нет чёрного списка «опасных» выходов: осторожность считается."""
    for path in SRC.rglob("*.py"):
        if "debug" in path.parts:
            continue          # отладочная раскладка — законное место для имён
        text = path.read_text(encoding="utf-8")
        for marker in ("FORBIDDEN", "DANGEROUS_KEYS", "BLACKLIST", "NEVER_PRESS"):
            assert marker not in text, f"{path.relative_to(SRC)}: список запретов {marker}"


# --- 10. Самоотчёты ни на что не влияют автоматически ---------------------


def test_invariant_10_note_entries_change_nothing(tmp_path: Path) -> None:
    """Пометка в журнале не меняет ни профиля, ни хеша, ни производных величин."""
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="test",
                  synthetic=True) as rec:
        before = rec.profile.profile_hash
        rec.record_note("мне кажется, я научился ходить")
        rec.record_note("СТОП")            # даже если текст похож на команду
        after = rec.profile.profile_hash
    assert before == after

    with Session.open(tmp_path / "s") as s:
        s.journal.verify()
        notes = [e for e in s.journal if e.kind is Kind.NOTE]
        assert len(notes) == 2
        # Ни один вид записи, кроме PROFILE_CHANGE, не меняет профиль
        assert not any(e.kind is Kind.PROFILE_CHANGE for e in s.journal)


def test_invariant_10_no_code_path_reads_note_text() -> None:
    """Ни одна функция не читает текст пометок и речи как управляющий сигнал.

    Ищем в дереве обращения к тексту записей NOTE. Разрешено только выводить его
    человеку (CLI). Любое сравнение текста с чем-либо — повод падать.

    `harness.debug` исключён: это сторона исследователя, где читаемый текст
    законен по определению — там лежит таблица расшифровки, и сравнивать надписи
    между собой она обязана, иначе не найдёт коллизии символов. Инвариант 10
    про другое: про то, что слова агента не меняют его параметров.
    """
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if "debug" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # шаблон: event["text"] == ... / .get("text") in ...
            if isinstance(node, ast.Compare):
                src = ast.dump(node)
                if "'text'" in src and any(
                        isinstance(op, (ast.Eq, ast.In, ast.NotEq)) for op in node.ops):
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, (
        "текст самоотчёта участвует в сравнении: " + ", ".join(offenders)
        + ". Реакция допустима только на поведение и объективные величины")


# --- 11. Структурные переключатели форкают журнал -------------------------


def test_invariant_11_structural_change_requires_fork(tmp_path: Path) -> None:
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="test",
                  synthetic=True) as rec:
        rec.record_frame(np.zeros((8, 8), dtype=np.uint8))
        structural = MILESTONE_0.with_structural(frame_format="rgb8")
        with pytest.raises(JournalError, match="нужен fork"):
            rec.change_parameters(structural, reason="попытка подменить структуру")


def test_invariant_11_parameters_change_without_fork(tmp_path: Path) -> None:
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="test",
                  synthetic=True) as rec:
        branch_before = rec.journal.meta.branch_id
        tuned = MILESTONE_0.with_parameters(capture_fps=60.0)
        rec.change_parameters(tuned, reason="подняли частоту")
        assert rec.journal.meta.branch_id == branch_before
        assert tuned.structure_hash == MILESTONE_0.structure_hash
        assert tuned.profile_hash != MILESTONE_0.profile_hash


def test_invariant_11_fork_creates_new_branch(tmp_path: Path) -> None:
    from harness.core.journal import branch_chain

    rec = Recorder(tmp_path / "s", profile=MILESTONE_0, source="test", synthetic=True)
    rec.record_frame(np.zeros((8, 8), dtype=np.uint8))
    first = rec.journal.meta.branch_id
    rec.fork(MILESTONE_0.with_structural(audio_channels=2, text_symbolized=False),
             reason="проверяем читаемый текст")
    second = rec.journal.meta.branch_id
    rec.record_frame(np.ones((8, 8), dtype=np.uint8))
    rec.close()

    assert first != second
    chain = branch_chain(tmp_path / "s" / "journal")
    assert len(chain) == 2
    assert chain[1].parent == first
    assert chain[1].forked_at_seq is not None


def test_invariant_11_one_branch_one_structure(tmp_path: Path) -> None:
    """Две структуры в одной ветке ловятся проверкой журнала."""
    rec = Recorder(tmp_path / "s", profile=MILESTONE_0, source="test", synthetic=True)
    rec.record_frame(np.zeros((8, 8), dtype=np.uint8))
    path = Path(rec.journal.path) / Journal.ENTRIES
    rec.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    d = json.loads(lines[-1])
    d["structure_hash"] = "deadbeef" * 4
    d["digest"] = __import__("harness.core.journal", fromlist=["x"]).entry_digest(
        {k: v for k, v in d.items() if k not in ("prev", "digest")}, d["prev"])
    lines[-1] = json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with Session.open(tmp_path / "s") as s:
        with pytest.raises(TamperError, match="structure_hash"):
            s.journal.verify()


def test_invariant_11_parameters_and_structural_never_overlap() -> None:
    with pytest.raises(ProfileError, match="и в parameters, и в structural"):
        Profile("плохой", parameters={"x": 1}, structural={"x": 2})
    assert not set(MILESTONE_0.parameters) & set(MILESTONE_0.structural)


# --- 12. Отладочный канал строго отделён ----------------------------------


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports_of(path: Path, module: str) -> set[str]:
    """Какие модули harness импортирует данный файл, с разрешением относительных."""
    out: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module.rsplit(".", 1)[0] if "." in module else module
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("harness"):
                    out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                up = node.level - 1
                base = base[:len(base) - up] if up else base
                prefix = ".".join(base)
                target = f"{prefix}.{node.module}" if node.module else prefix
            elif node.module and node.module.startswith("harness"):
                target = node.module
            else:
                continue
            out.add(target)
            for a in node.names:
                out.add(f"{target}.{a.name}")
    return out


def _reachable(start: str) -> set[str]:
    """Все модули harness, достижимые по импортам из данного."""
    by_name: dict[str, Path] = {_module_name(p): p for p in SRC.rglob("*.py")}
    seen: set[str] = set()
    queue = [start]
    while queue:
        cur = queue.pop()
        if cur in seen:
            continue
        seen.add(cur)
        path = by_name.get(cur)
        if path is None:
            continue
        for imp in _imports_of(path, cur):
            # импорт может указывать на модуль или на имя внутри модуля
            for candidate in (imp, imp.rsplit(".", 1)[0]):
                if candidate in by_name and candidate not in seen:
                    queue.append(candidate)
    return seen


def test_invariant_12_agentside_cannot_reach_debug() -> None:
    """Из агентской стороны нет пути импорта к отладочному потоку.

    Проверяется транзитивно: не «не импортирует напрямую», а «не достаёт через
    любую цепочку посредников».
    """
    reach = _reachable("harness.agentside")
    leaked = sorted(m for m in reach if m.startswith("harness.debug"))
    assert not leaked, (
        f"агентская сторона достаёт до отладочного потока: {leaked}. "
        "Путь надо разорвать: передавайте нужное снаружи функцией, "
        "как это сделано с резолвером скан-кодов")


def test_invariant_12_inject_cannot_reach_debug() -> None:
    """Инъекция ввода тоже не знает настоящих имён клавиш.

    Таблица скан-кодов живёт в `harness.debug.keymap`, а устройство получает
    только односторонний резолвер.
    """
    reach = _reachable("harness.inject")
    leaked = sorted(m for m in reach if m.startswith("harness.debug"))
    assert not leaked, f"инъекция достаёт до отладочного потока: {leaked}"


def test_invariant_12_truth_is_not_in_journal(corpus: Path) -> None:
    """Истина синтетики есть в отладочном потоке и отсутствует в журнале."""
    from harness.corpus.synthetic import load_frame_truth

    truth = load_frame_truth(corpus)
    assert truth and "cam_dx" in truth[0]

    journal_text = (next((corpus / "journal" / "branches").iterdir())
                    / Journal.ENTRIES).read_text(encoding="utf-8")
    for forbidden in ("cam_x", "cam_dx", "hud", "bearing_deg", "sound_world_xy"):
        assert forbidden not in journal_text, f"{forbidden} утекло в журнал"


def test_invariant_12_debug_stream_is_separate_file(corpus: Path) -> None:
    assert (corpus / "debug" / "truth.jsonl").exists()
    assert (corpus / "debug" / "symbols.jsonl").exists()
    journal_files = list((corpus / "journal").rglob("*"))
    assert not any("truth" in p.name for p in journal_files)
