"""Конвейер убеждений, открытые вопросы, полный набор операций сна. TASK-12.

Главное, что здесь проверяется, — два различения, которые размываются сами, без всякой
попытки их убрать:

- **свидетельство против опыта.** Одна удобная строка в `Belief.observe` повышала
  происхождение любого подтверждённого утверждения до `experience`, и защита от
  загрязнения отваливалась на первой успешной проверке. Ни один тест на это не падал;
- **перечень непонятого против очереди дел.** Условие `not h.checked` считало вопросы
  ожидающими проверки, и показатель «открытых вопросов» мерил длину очереди.
"""

from __future__ import annotations

import pytest

from harness.core.branches import Ledger
from harness.core.clocks import Stamp
from harness.core.journal import Actor, ActorLayer, Kind
from harness.core.profile import from_schema
from harness.model.beliefs import (TRANSITIONS, Belief, BeliefError, BeliefStore,
                                   Hypothesis, Origin, Provenance, Stage)
from harness.model.beliefs import Testimony as Claimed   # имя без префикса Test:
from harness.model.beliefs import merge_testimony, pipeline_arbitration
from harness.model.beliefs import transition_allowed
from harness.model.questions import (Missing, OpenQuestions, from_journal,
                                     journal_question, matches, token)
from harness.session import Recorder, Session

PROFILE = from_schema("конвейер", capture_width=32, capture_height=32)
BRANCH = "тест-конвейера"


def _prov(origin: Origin = Origin.EXPERIENCE, seq: int = 1) -> Provenance:
    src = "self" if origin is not Origin.TESTIMONY else "сосед"
    return Provenance(origin, BRANCH, seq, src, 1.0 if origin is not Origin.TESTIMONY
                      else 0.6)


# --- четыре состояния и переходы ------------------------------------------------


def test_four_stages_and_no_fifth() -> None:
    """Положений ровно четыре, и у каждого объявлен смысл."""
    from harness.model.beliefs import STAGE_MEANING

    assert {str(s) for s in Stage} == {"hypothesis", "verified", "refuted", "deferred"}
    assert set(STAGE_MEANING) == set(Stage)
    assert all(v for v in STAGE_MEANING.values())


def test_hypothesis_in_work_has_a_stage_but_no_outcome() -> None:
    """Тест есть, не выполнен: положение `hypothesis`, исхода нет.

    Это и есть различение, которое размывается: `state is None` означает «исхода нет»,
    и подменить его на `deferred` значит записать «проверить нечем» там, где просто не
    дошли руки.
    """
    h = Hypothesis("что-то", "нажать и посмотреть", 0.5, _prov())
    assert h.stage is Stage.HYPOTHESIS
    assert h.state is None
    assert not h.is_question


def test_question_is_deferred_and_not_in_the_queue() -> None:
    """Вопрос отложен, и в очередь дел он не попадает."""
    store = BeliefStore(BRANCH)
    store.add_hypothesis(Hypothesis.question(
        "кто это сделал", 0.4, _prov(Origin.HUNCH), because="деятеля не видел",
        reopens_on=(token(Missing.PLACE),)))
    store.add_hypothesis(Hypothesis("дело", "нажать", 0.5, _prov()))

    assert len(store.deferred()) == 1
    assert len(store.unchecked_hypotheses()) == 1
    # Прежнее условие считало обе: вопрос тоже не `checked`.
    assert sum(1 for h in store.hypotheses.values() if not h.checked) == 2
    assert store.stats()["by_stage"]["deferred"] == 1
    assert store.stats()["by_stage"]["hypothesis"] == 1


def test_transitions_are_declared_and_enforced() -> None:
    """Разрешённых переходов три, обратных нет ни одного."""
    assert len(TRANSITIONS) == 3
    assert transition_allowed(Stage.DEFERRED, Stage.HYPOTHESIS)
    assert transition_allowed(Stage.HYPOTHESIS, Stage.VERIFIED)
    assert not transition_allowed(Stage.HYPOTHESIS, Stage.DEFERRED)
    assert not transition_allowed(Stage.VERIFIED, Stage.HYPOTHESIS)


def test_resolved_twice_is_refused() -> None:
    """Разрешённую гипотезу нельзя разрешить снова: это не разрешённый переход."""
    h = Hypothesis("что-то", "нажать", 0.5, _prov()).resolved(True)
    assert h.stage is Stage.VERIFIED
    with pytest.raises(BeliefError, match="не разрешён"):
        h.resolved(False)


def test_every_transition_has_a_counter() -> None:
    """Инвариант 26: у каждого перехода счётчик, и ветка без срабатываний — дефект."""
    ledger = Ledger()
    arb = ledger.add(pipeline_arbitration())
    assert len(arb.branches) == len(TRANSITIONS)
    assert len(arb.dead) == len(TRANSITIONS), "пока ничего не срабатывало"

    q = Hypothesis.question("загадка", 0.5, _prov(Origin.HUNCH),
                            because="нечем", reopens_on=(token(Missing.SKILL),))
    h = q.with_test("научился — попробовать", arb=arb)
    h.resolved(True, arb=arb)
    h.resolved(False, arb=arb)
    assert not arb.dead, f"мёртвые переходы: {[b.name for b in arb.dead]}"


# --- свидетельство не становится опытом ---------------------------------------


def test_confirmed_testimony_stays_testimony() -> None:
    """Пересказ, проверенный своими руками, остаётся пересказом по происхождению.

    Сдвиг числа: по прежнему правилу происхождение сменили бы все 40 пересказов корпуса
    (`tools/measure_beliefs.py`), по новому — ни один.
    """
    heard = Belief("ENT_0001|dyn|что-то", 0.6, 0.5, 1, _prov(Origin.TESTIMONY))
    assert heard.is_hearsay

    checked = heard.observe(True, _prov(Origin.EXPERIENCE, seq=100))
    assert checked.provenance.origin is Origin.TESTIMONY, "опытом стать не мог"
    assert checked.n_experience == 1, "но своя проверка засчитана"
    assert not checked.is_hearsay, "слухом он быть перестал"
    assert checked.checked_at == (100,), "эпизод проверки записан отдельно"
    assert checked.verified_testimony
    assert checked.confidence > heard.confidence


def test_hunch_does_become_experience() -> None:
    """Догадка была своя — проверка делает её опытом. Правка запретила ровно одно."""
    hunch = Belief("ENT_0002|dyn|догадка", 0.5, 0.5, 1, _prov(Origin.HUNCH))
    after = hunch.observe(True, _prov(Origin.EXPERIENCE, seq=7))
    assert after.provenance.origin is Origin.EXPERIENCE
    assert after.checked_at == (7,)


def test_origin_never_downgrades() -> None:
    """Опыт не понижается до пересказа, сколько бы чужого ни пришло потом."""
    own = Belief("ENT_0003|dyn|своё", 1.0, 0.1, 1, _prov(Origin.EXPERIENCE), 1)
    after = own.observe(True, _prov(Origin.TESTIMONY, seq=9))
    assert after.provenance.origin is Origin.EXPERIENCE
    assert after.n_experience == 1 and after.n == 2


def test_hypothesis_from_testimony_confirms_as_testimony() -> None:
    """Гипотеза, выросшая из чужого слова, подтверждается, не меняя происхождения."""
    h = Hypothesis("чужое утверждение", "проверить самому", 0.6,
                   _prov(Origin.TESTIMONY))
    b = h.confirm(True, _prov(Origin.EXPERIENCE, seq=42))
    assert b.provenance.origin is Origin.TESTIMONY
    assert b.n_experience == 1 and b.checked_at == (42,)


def test_merge_testimony_reports_the_queue_not_the_questions() -> None:
    """`pending_recheck` считает очередь дел, а вопросы в неё не входят."""
    store = BeliefStore(BRANCH)
    store.add_hypothesis(Hypothesis.question("загадка", 0.5, _prov(Origin.HUNCH),
                                             because="нечем"))
    info = merge_testimony(store, [Claimed("ENT_0009|dyn|слух", source="сосед",
                                             trust=0.5, branch=BRANCH, seq=3)])
    assert info["added"] == 1
    assert info["pending_recheck"] == 1, "одна гипотеза с тестом, вопрос не в счёт"


# --- переоткрытие отложенного --------------------------------------------------


def test_matching_rule_is_strict_in_one_direction() -> None:
    """Безымянная нехватка закрывается любым таким; именованная — только точным."""
    assert matches(token(Missing.PLACE), [token(Missing.PLACE, "PL_1")])
    assert not matches(token(Missing.PLACE, "PL_1"), [token(Missing.PLACE)])
    assert not matches(token(Missing.SKILL), [token(Missing.PLACE, "PL_1")])


def test_register_refuses_a_hypothesis_with_a_test() -> None:
    """В перечень непонятого нельзя положить дело, до которого не дошли руки."""
    reg = OpenQuestions()
    with pytest.raises(BeliefError, match="не очередь дел"):
        reg.ask(Hypothesis("дело", "нажать", 0.5, _prov()), 1)


def test_reopening_closes_the_loop() -> None:
    """Появилось место — вопрос вернулся в работу и разрешился задним числом."""
    reg = OpenQuestions()
    reg.ask(Hypothesis.question(
        "кто оставил артефакт", 0.4, _prov(Origin.HUNCH),
        because="деятеля не наблюдал", reopens_on=(token(Missing.PLACE),)), 10)

    ledger = Ledger()
    arb = ledger.add(pipeline_arbitration())
    got = reg.offer([token(Missing.PLACE, "PL_7F3A")], seq=100,
                    build_test=lambda q, by: f"дойти до {by}", arb=arb)
    assert len(got) == 1 and got[0].by == "место:PL_7F3A"

    q = reg.questions["кто оставил артефакт"]
    assert q.hypothesis.stage is Stage.HYPOTHESIS, "вопрос стал гипотезой"
    reg.resolve("кто оставил артефакт", outcome=True, seq=150, arb=arb)
    assert q.hypothesis.stage is Stage.VERIFIED
    assert q.waited == 140
    assert reg.mean_wait() == 140


def test_resolution_without_reopening_is_refused() -> None:
    """Исход, взявшийся без найденного теста, — не разрешение, а выдумка."""
    reg = OpenQuestions()
    reg.ask(Hypothesis.question("загадка", 0.5, _prov(Origin.HUNCH),
                                because="нечем"), 5)
    with pytest.raises(BeliefError, match="минуя переоткрытие"):
        reg.resolve("загадка", outcome=True, seq=9)


def test_false_offers_and_coverage_are_reported() -> None:
    """Инвариант 31: у проверки есть доля ложных и покрытие, и оба видны."""
    reg = OpenQuestions()
    reg.ask(Hypothesis.question("нужна табличка", 0.3, _prov(Origin.HUNCH),
                                because="читать не умею",
                                reopens_on=(token(Missing.READING),)), 1)
    reg.ask(Hypothesis.question("слепой вопрос", 0.2, _prov(Origin.HUNCH),
                                because="не знаю, чего не хватает"), 2)

    got = reg.offer([token(Missing.READING, "SYM_1")], seq=50,
                    build_test=lambda q, by: None)
    assert got == [], "тест не построился — переоткрытия нет"
    assert reg.false_offer_share() == 1.0
    assert reg.coverage() == 0.5, "один вопрос из двух объявил нехватку"
    assert "ложных предложений" in reg.report()
    assert "переоткрыты не будут никогда" in reg.report()


def test_blind_question_is_never_reopened() -> None:
    """Вопрос без объявленной нехватки не вернётся ни от какой возможности."""
    reg = OpenQuestions()
    reg.ask(Hypothesis.question("почему мир такой", 0.2, _prov(Origin.HUNCH),
                                because="не знаю даже того, чего не хватает"), 1)
    everything = [token(k) for k in Missing] + [token(k, "X") for k in Missing]
    assert reg.offer(everything, seq=9, build_test=lambda q, by: "любой тест") == []
    assert reg.open_questions()[0].blind


# --- вопросы выводимы из журнала ------------------------------------------------


def test_questions_survive_a_rebuild(tmp_path) -> None:
    """Перечень непонятого выводим из следа, а не живёт сбоку (инвариант 1)."""
    h = Hypothesis.question("кто оставил артефакт", 0.4,
                            _prov(Origin.HUNCH), because="деятеля не наблюдал",
                            reopens_on=(token(Missing.PLACE),))
    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        journal_question(rec.journal, rec.clocks.stamp(), h)

    with Session.open(tmp_path / "s") as s:
        reg = from_journal(s.journal)
        from harness.model.rebuild import rebuild_from_journal

        rebuilt = rebuild_from_journal(s.journal)

    assert len(reg.open_questions()) == 1
    assert reg.open_questions()[0].hypothesis.reopens_on == (token(Missing.PLACE),)
    assert rebuilt.questions_asked == 1
    assert len(rebuilt.beliefs.deferred()) == 1


def test_vitals_count_open_questions(tmp_path) -> None:
    """Показатель 11 считается по журналу и объявляет единицу «вопрос»."""
    from harness.model.vitals import from_journal as vitals_from_journal

    h = Hypothesis.question("загадка", 0.5, _prov(Origin.HUNCH), because="нечем",
                            reopens_on=(token(Missing.SKILL),))
    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        journal_question(rec.journal, rec.clocks.stamp(), h)
    with Session.open(tmp_path / "s") as s:
        v = vitals_from_journal(s.journal, profile=s.profile, skip=("beliefs",))

    open_q = v.get("questions_open")
    assert open_q is not None and open_q.value == 1
    assert open_q.independence.unit == "вопрос"
    cov = v.get("questions_coverage")
    assert cov is not None and cov.value == 1.0


# --- часть 2: операции сна -----------------------------------------------------


def test_recall_is_a_transaction_not_a_read() -> None:
    """Извлечение меняет карточку: растёт обращение и ранг. Но не `mu`."""
    store = BeliefStore(BRANCH)
    store.learn_affordance("ENT_1", "k1", True, _prov())
    before_mu = store.entities["ENT_1"].affordances["k1"].mu
    before_rank = store.entities["ENT_1"].rank()

    got = store.recall("ENT_1", "k1", seq=50)
    assert got is not None
    for _ in range(4):
        store.recall("ENT_1", "k1", seq=60)

    e = store.entities["ENT_1"]
    assert e.recalls == 5 and e.last_recall_seq == 60
    assert e.rank() > before_rank, "часто извлекаемое держится подробнее"
    assert e.affordances["k1"].mu == before_mu, "извлечение — не наблюдение"


def test_recall_of_nothing_is_none() -> None:
    store = BeliefStore(BRANCH)
    assert store.recall("ENT_NONE", "k", seq=1) is None


def test_split_is_proposed_only_when_saturated() -> None:
    """Середина при большом n — две сущности; при малом — «ещё не знаю»."""
    from harness.model.consolidation import Consolidator

    con = Consolidator(from_schema("сон", split_min_observations=8,
                                   split_middle_band=0.1))
    store = BeliefStore(BRANCH)
    for i in range(10):                        # чередование: ровно половина
        store.learn_affordance("ENT_A", "k", i % 2 == 0, _prov(seq=i + 1))
    store.learn_affordance("ENT_B", "k", True, _prov(seq=100))
    store.learn_affordance("ENT_B", "k", False, _prov(seq=101))

    got = con.propose_splits(store)
    ids = {p.entity for p in got}
    assert "ENT_A" in ids, "насыщенная середина — признак двух сущностей"
    assert "ENT_B" not in ids, "два наблюдения — это неопределённость"


def test_sleep_runs_every_declared_operation(tmp_path) -> None:
    """Полный набор операций: каждая либо сработала, либо доложила ноль."""
    from harness.model.consolidation import SLEEP_OPERATIONS, Consolidator

    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        rec.journal.append(Kind.GOAL, Stamp(1, 1), Actor.AGENT, ActorLayer.DRIVE,
                           event={"code": "set"})
    with Session.open(tmp_path / "s") as s:
        con = Consolidator(PROFILE)
        rebuilt, report = con.run(s.journal, Stamp(2, 2), live=BeliefStore(BRANCH))

    assert len(SLEEP_OPERATIONS) == 5
    d = report.as_dict()
    for key in ("merges_proposed", "splits_proposed", "edges_recomputed",
                "scripts_mined", "ranked", "demoted", "drift"):
        assert key in d, f"операция {key} не докладывается вовсе"
    assert report.drift is not None, "дрейф считается, если есть с чем сравнивать"


def test_missing_recompute_is_loud() -> None:
    """Граф без пересчёта — не «нечего делать», а отсутствующая операция."""
    from harness.model.consolidation import Consolidator

    con = Consolidator(PROFILE)
    with pytest.raises(AttributeError, match="не выполнена"):
        con.recompute_edges(object())


def test_drift_is_measured_not_assumed(tmp_path) -> None:
    """Дрейф памяти — число, а не «совпало/нет»: живое против пересобранного."""
    from harness.model.rebuild import drift

    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        rec.journal.append(Kind.GOAL, Stamp(1, 1), Actor.AGENT, ActorLayer.DRIVE,
                           event={"code": "set"})
    live = BeliefStore(BRANCH)
    live.learn_affordance("ENT_PHANTOM", "k", True, _prov())

    with Session.open(tmp_path / "s") as s:
        got = drift(live, s.journal)

    assert got["drift"] == 1.0, "утверждение есть в живом и отсутствует в следе"
    assert got["only_live"] == ["ENT_PHANTOM|afford|k"]
    assert got["unit"] == "утверждение" and got["n_claims"] == 1
    assert not got["same_fingerprint"]


# --- закладки и карантин (инвариант 18) ----------------------------------------


def test_bookmark_requires_a_reason() -> None:
    """Закладка без причины не ставится: список причин — окно в состояние агента."""
    from harness.model.bookmarks import BookmarkError, Bookmarks

    marks = Bookmarks.from_profile(PROFILE)
    with pytest.raises(BookmarkError, match="без причины"):
        marks.place("SEG_1", "   ")


def test_bookmark_budget_is_limited() -> None:
    """Предел закладок — условие выбора, а не препятствие."""
    from harness.model.bookmarks import BookmarkError, Bookmarks

    marks = Bookmarks(per_sleep=2)
    marks.place("SEG_1", "ещё не понял")
    marks.place("SEG_2", "видел один раз")
    assert marks.left == 0
    with pytest.raises(BookmarkError, match="больше нет"):
        marks.place("SEG_3", "тоже интересно")


def test_bookmark_expires_next_sleep() -> None:
    """Протухает к следующему сну — иначе держит место вечно."""
    from harness.model.bookmarks import Bookmarks

    marks = Bookmarks(per_sleep=5)
    marks.place("SEG_1", "модель сломалась, не знаю почему")
    assert marks.protected("SEG_1").startswith("закладка")
    gone = marks.wake()
    assert [b.segment for b in gone] == ["SEG_1"]
    assert marks.protected("SEG_1") == "", "после сна защиты нет"


def test_renewal_needs_a_new_reason() -> None:
    """Продление той же причиной — инерция, а не решение."""
    from harness.model.bookmarks import BookmarkError, Bookmarks

    marks = Bookmarks(per_sleep=5)
    marks.place("SEG_1", "ещё не понял")
    with pytest.raises(BookmarkError, match="заново указанной причины"):
        marks.renew("SEG_1", "ещё не понял")
    b = marks.renew("SEG_1", "теперь понял половину, держи остальное")
    assert b.renewals == 1


def test_quarantine_is_not_demotable_by_the_agent() -> None:
    """Карантинный класс агенту недоступен: понижение отказывается и пишет причину."""
    from harness.model.bookmarks import Bookmarks, Quarantine, demotable

    marks = Bookmarks(per_sleep=5)
    marks.quarantine("SEG_DEATH", Quarantine.DEATH)
    allowed, refused = demotable(["SEG_DEATH", "SEG_ORDINARY"], marks)
    assert allowed == ["SEG_ORDINARY"]
    assert len(refused) == 1 and "смерть" in refused[0].why
    assert marks.stats()["refusals"] == 1


def test_every_quarantine_class_says_why() -> None:
    from harness.model.bookmarks import WHY_QUARANTINE, Quarantine

    assert set(WHY_QUARANTINE) == set(Quarantine)
    assert all(v for v in WHY_QUARANTINE.values())


def test_demotion_goes_through_bookmarks(tmp_path) -> None:
    """Сон понижает только то, что разрешили закладки и карантин."""
    from harness.model.bookmarks import Bookmarks, Quarantine
    from harness.model.consolidation import Consolidator

    marks = Bookmarks(per_sleep=5)
    marks.quarantine("SEG_SPIKE", Quarantine.ERROR_SPIKE)
    marks.place("SEG_MINE", "я это ещё не понял")

    with Recorder(tmp_path / "s", profile=PROFILE, source="t", synthetic=True) as rec:
        rec.journal.append(Kind.GOAL, Stamp(1, 1), Actor.AGENT, ActorLayer.DRIVE,
                           event={"code": "set"})
    with Session.open(tmp_path / "s") as s:
        con = Consolidator(PROFILE)
        _, report = con.run(s.journal, Stamp(2, 2), marks=marks,
                            segments=["SEG_SPIKE", "SEG_MINE", "SEG_PLAIN"])

    # Закладка протухла на входе в сон, поэтому SEG_MINE понижается; карантин — нет.
    assert "SEG_SPIKE" not in report.demoted
    assert "SEG_PLAIN" in report.demoted
    assert report.bookmarks_expired == 1
    assert any("карантин" in r["why"] for r in report.demote_refused)
