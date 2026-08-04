"""Самоотчёт и показатели: слова агента отдельно, объективные числа отдельно.

Главное здесь — не то, что отчёт собирается, а то, что он ни на что не влияет.
Инвариант 10 до появления этого модуля проверял отсутствующую вещь: самоотчётов не
было, и тесты смотрели на пометки исследователя. Теперь есть чему испортить
эксперимент, и проверяется именно оно: отчёт — сток, пересборка его пропускает, ни
один агентский модуль его не читает.

Показатели проверяются с другой стороны: что они посчитаны по журналу и что
отсутствующий показатель отсутствует явно, с причиной, а не в виде правдоподобного
нуля.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from harness.behaviour import selfreport
from harness.behaviour.selfreport import (CODES, VOCABULARY, Line, SelfReport,
                                          check_opaque, compose, journal_report)
from harness.core.action import (Action, Kind as ActionKind, action_key,
                                 output_id)
from harness.core.clocks import Stamp
from harness.core.journal import Actor, Kind
from harness.core.profile import MILESTONE_0, from_schema
from harness.core.symbols import SymbolError
from harness.model import vitals
from harness.model.beliefs import Origin, Provenance
from harness.model.drives import DRIVE_NAMES, EMOTIONS, Mood, Motivation, emotion_label
from harness.model.rebuild import BodyMap, rebuild_from_journal
from harness.session import Recorder, Session

SRC = Path(__file__).resolve().parent.parent / "src" / "harness"


def _profile(**kw: object):
    base: dict[str, object] = dict(capture_width=32, capture_height=32)
    base.update(kw)
    return from_schema("ТЕСТ-отчёт", **base)


def _body() -> BodyMap:
    """Карта тела с одним живым выходом, одним молчащим и одним непонятным.

    Собирается тем же путём, каким её собирает пересборка — через `fact()`, а не
    отдельным тестовым способом: иначе тест проверял бы свою реализацию карты тела,
    а не настоящую.
    """
    body = BodyMap(live_min_responses=2, silent_min_deliveries=3)
    body.set_background_rate(0.0)
    live, silent, unclear = (output_id("OUT", i) for i in (1, 2, 3))
    for seq in range(3):
        f = body.fact(live, seq)
        f.tries += 1
        f.delivered += 1
        f.responded += 1
        g = body.fact(silent, seq)
        g.tries += 1
        g.delivered += 1
    h = body.fact(unclear, 4)               # одна попытка — «не знаю», а не «молчит»
    h.tries += 1
    h.delivered += 1
    h.responded += 1
    return body


# --- отчёт как сток ---------------------------------------------------------


def test_report_lines_are_codes_from_a_closed_set() -> None:
    """Строка с кодом вне схемы не уходит в журнал: это была бы свободная фраза."""
    r = SelfReport(1, 1, [Line("я_молодец", 1.0, "experience")])
    with pytest.raises(ValueError, match="вне схемы"):
        r.as_event()


def test_report_rejects_screen_text() -> None:
    """Расшифрованная надпись в отчёте падает так же громко, как в восприятии."""
    r = SelfReport(1, 1, [Line(selfreport.I_LEARNED, "Открыть инвентарь", "experience")])
    with pytest.raises(SymbolError):
        r.as_event()


def test_report_accepts_its_own_vocabulary_and_composite_keys() -> None:
    """Своё слово и составной ключ убеждения — не надпись с экрана.

    Проверка не «всё, что похоже на слово, запрещено»: имена драйвов и виды целей
    агент не с экрана взял. Проверяется, что словарь именно закрыт.
    """
    check_opaque("curiosity")
    check_opaque("любопытство")
    check_opaque(f"ENT_1C90|afford|{action_key(output_id('OUT', 2), 200)}")
    check_opaque("untried:12")
    check_opaque("unknown_rate:0.375")
    with pytest.raises(SymbolError):
        check_opaque("inventory|open")


def test_every_code_used_by_compose_is_declared() -> None:
    """Ни одна строка, которую compose умеет выдать, не выпадает из схемы."""
    report = compose(stamp=Stamp(5, 5), body=_body(), outputs=8,
                    motivation=Motivation(_profile()),
                    did={"babble": 12, "plan": 2})
    report.as_event()                      # падает, если код вне CODES
    assert {ln.code for ln in report.lines} <= CODES


def test_emotion_labels_are_the_declared_set() -> None:
    """`emotion_label` не возвращает ничего, кроме объявленных пяти ярлыков.

    Иначе словарь отчёта разошёлся бы с реальностью и в отчёт попало бы слово,
    которого в словаре нет — падение на пустом месте.
    """
    seen = set()
    for v in np.linspace(-1.0, 1.0, 21):
        for a in np.linspace(-1.0, 1.0, 21):
            for high in (False, True):
                seen.add(emotion_label(Mood(float(v), float(a)), error_high=high))
    assert seen <= set(EMOTIONS), f"ярлык вне набора: {seen - set(EMOTIONS)}"
    assert set(DRIVE_NAMES) <= VOCABULARY


def test_invariant_10_rebuild_ignores_self_reports(tmp_path: Path) -> None:
    """Слова агента о себе не создают убеждений и не меняют карту тела.

    Самый опасный обход инварианта 10 — не «агент приказал себе», а «агент сказал
    о себе, пересборка поверила». Здесь один и тот же журнал пересобирается до и
    после добавления отчётов: отпечаток убеждений и карта тела обязаны совпасть.
    """
    profile = _profile()
    with Recorder(tmp_path / "s", profile=profile, source="test", synthetic=True) as rec:
        out = output_id("OUT", 1)
        for i in range(4):
            rec.journal.append(Kind.ACTION, Stamp(i, i), Actor.AGENT,
                               action=Action(ActionKind.KEY, 200, output=out),
                               event={"code": "delivered", "responded": True})
        before = rebuild_from_journal(rec.journal)

        # Отчёт утверждает, что выход молчит и что он откатываемый. Ни то, ни другое
        # не должно попасть в модель.
        report = SelfReport(9, 9, [Line(selfreport.I_LEARNED, out, "experience"),
                                   Line("body_silent", 1, "experience")])
        journal_report(rec.journal, report, Stamp(9, 9))
        after = rebuild_from_journal(rec.journal)

    assert after.self_reports == 1, "запись отчёта не дошла до журнала"
    assert after.beliefs.fingerprint() == before.beliefs.fingerprint()
    assert after.body.as_dict() == before.body.as_dict()


def test_invariant_10_no_agent_side_module_reads_the_report() -> None:
    """Отчёт не читает никто, кроме пульта: проверяется обходом импортов.

    Тем же способом, которым проверяется изоляция отладочного канала. Список
    разрешённых читателей закрыт: CLI и сам модуль. Если отчёт понадобится
    поведению — это не «добавить импорт», а разговор про инвариант 10.
    """
    allowed = {"cli.py"}
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.name == "selfreport.py" or path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any("selfreport" in n for n in names):
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not offenders, (
        "самоотчёт читается кодом: " + ", ".join(offenders)
        + ". Реакция допустима только на поведение и объективные величины")


def test_invariant_10_writing_reports_does_not_change_behaviour(tmp_path: Path) -> None:
    """Круг с самоотчётами идёт точно так же, как круг без них.

    Это проверка не механизма, а результата, и она сильнее всех остальных: два
    прогона с одним seed, в одном агент отчитывается о себе каждый круг, в другом
    молчит. Тело, цели и настроение обязаны совпасть до последнего числа. Если
    когда-нибудь совпадать перестанут — где-то появилось чтение отчёта.
    """
    from harness.behaviour.babbling import Babbler, run_babbling
    from harness.behaviour.goals import GoalStack, candidates_from_body, choose
    from harness.corpus.world import InteractiveWorld
    from harness.vision.predict import PredictionError

    def run(with_reports: bool) -> dict:
        profile = from_schema("ТЕСТ-отчёт-круг", capture_width=160, capture_height=96,
                              babble_rate=0.9, babble_repeats=3)
        world = InteractiveWorld(profile, seed=11)
        path = tmp_path / ("with" if with_reports else "without")
        with Recorder(path, profile=profile, source="t", synthetic=True) as rec:
            babbler = Babbler(profile, world.outputs, journal=rec.journal, rng_seed=11)
            motivation = Motivation(profile)
            stack = GoalStack(profile, journal=rec.journal)
            error = PredictionError(profile)
            branch = rec.journal.meta.branch_id
            for _ in range(4):
                summary = error.summary()
                motivation.update(error_mean=summary["mean"],
                                  error_sigma=summary["sigma"],
                                  error_now=summary["last"])
                if stack.active is None:
                    cands = candidates_from_body(
                        babbler.body, world.outputs,
                        caution_threshold=float(
                            profile.parameters["irreversibility_threshold"]))
                    if cands:
                        stack.push(choose(cands, motivation, top=1)[0], motivation,
                                   rec.journal.seq, branch, rec.clocks.stamp(),
                                   budget_ticks=50)
                run_babbling(world, babbler, steps=40, clocks=rec.clocks, error=error)
                stack.tick(rec.clocks.stamp())
                if with_reports:
                    stamp = rec.clocks.stamp()
                    report = compose(stamp=stamp, body=babbler.body, goals=stack,
                                     motivation=motivation, error=error,
                                     outputs=len(world.outputs))
                    journal_report(rec.journal, report, stamp)
            return {"body": babbler.body.as_dict(), "goals": stack.stats(),
                    "mood": motivation.mood.as_dict(),
                    "error": error.summary()}

    with_reports, without = run(True), run(False)
    # Сначала — что сравнивать вообще есть что: совпадение двух пустых прогонов
    # ничего не проверяет.
    assert with_reports["body"]["live"] > 0, "лепет не открыл ни одного выхода"
    assert with_reports["goals"]["goals"] > 0, "ни одной цели не поставлено"
    assert with_reports == without, (
        "прогон с самоотчётами разошёлся с прогоном без них — значит отчёт где-то "
        "читается. Реакция допустима только на поведение и объективные величины")


def test_report_says_what_it_does_not_know() -> None:
    """Перечень незнания есть и он не пуст, когда есть чего не знать."""
    body = _body()
    report = compose(stamp=Stamp(1, 1), body=body, outputs=8)
    unknown = [ln.value for ln in report.by_code(selfreport.I_DO_NOT_KNOW)]
    assert any(str(v).startswith("untried:") for v in unknown), unknown
    assert report.value(selfreport.BODY_KNOWN) == round(3 / 8, 3)


def test_report_is_silent_where_it_has_nothing_to_say() -> None:
    """Ноль и «нечего сказать» — разные вещи. Пустой агент не отчитывается нулями."""
    report = compose(stamp=Stamp(0, 0))
    assert report.lines == []
    assert report.value(selfreport.GOAL_PASSED) is None


# --- показатели -------------------------------------------------------------


def _journal_with_life(tmp_path: Path):
    """Журнал с попытками, целью, шагом плана, сном и разрывом захвата."""
    profile = _profile()
    with Recorder(tmp_path / "s", profile=profile, source="test", synthetic=True) as rec:
        # Кадр и разрыв — через Recorder, у него свои часы. Остальное дописывается
        # руками позже по t_self: журнал не отматывает время назад.
        rec.record_frame(np.zeros((32, 32), dtype=np.uint8))
        rec.record_gap("frames_dropped", {"missed": 3})
        out = output_id("OUT", 1)
        for i in range(100, 106):
            rec.journal.append(Kind.ACTION, Stamp(i, i), Actor.AGENT,
                               action=Action(ActionKind.KEY, 200, output=out),
                               event={"code": "delivered", "responded": i % 2 == 0})
        rec.journal.append(Kind.GOAL, Stamp(107, 107), Actor.AGENT,
                           event={"code": "set", "id": "G1", "kind": "learn_output"})
        rec.journal.append(Kind.GOAL, Stamp(108, 108), Actor.AGENT,
                           event={"code": "passed", "id": "G1"})
        rec.journal.append(Kind.GOAL, Stamp(109, 109), Actor.AGENT,
                           event={"code": "abandoned", "id": "G2", "spent_ticks": 40})
        rec.journal.append(Kind.PLAN, Stamp(110, 110), Actor.AGENT,
                           event={"code": "plan_step", "agreed": True})
        rec.journal.append(Kind.PLAN, Stamp(111, 111), Actor.AGENT,
                           event={"code": "plan_step", "agreed": False})
        rec.journal.append(Kind.SLEEP, Stamp(112, 112), Actor.NONE,
                           event={"code": "consolidation"})
    return profile


def test_vitals_are_derived_from_the_journal(tmp_path: Path) -> None:
    """Показатели считаются по журналу и совпадают с тем, что в него положено."""
    profile = _journal_with_life(tmp_path)
    with Session.open(tmp_path / "s") as s:
        v = vitals.from_journal(s.journal, profile=profile)

    assert v.value("actions") == 6
    assert v.value("responded_share") == round(3 / 6, 4)
    assert v.value("goals_set") == 1
    assert v.value("competence") == round(1 / 2, 4)      # одна прошла, одна брошена
    assert v.value("ticks_to_abandon") == 40.0
    assert v.value("plan_steps") == 2
    assert v.value("plan_agreement") == 0.5
    assert v.value("sleeps") == 1
    assert v.value("capture_gaps") == 1
    assert v.value("body_live") == 1


def test_vitals_name_what_is_missing_instead_of_guessing(tmp_path: Path) -> None:
    """Отсутствующий показатель отсутствует явно и с причиной.

    Правдоподобное число вместо реального — худшее, что может случиться: оно
    сломает эксперимент незаметно. Поэтому ошибка предсказания в сводке пустая, а
    не «примерно по числу всплесков».
    """
    profile = _journal_with_life(tmp_path)
    with Session.open(tmp_path / "s") as s:
        v = vitals.from_journal(s.journal, profile=profile)

    absent = {x.code: x for x in v.absent()}
    assert "prediction_error_mean" in absent
    for x in v.absent():
        assert len(x.note) > 20, f"{x.code}: пустое место без объяснения"
    assert all(x.source for x in v.vitals)


def test_vital_without_source_is_impossible() -> None:
    """Число, про которое нельзя сказать, откуда оно, в сводку не попадает."""
    with pytest.raises(ValueError, match="без источника"):
        vitals.Vital("competence", 0.5, "доля", "")
    with pytest.raises(ValueError, match="без объяснения"):
        vitals.Vital("competence", None, "доля", "журнал")


def test_vitals_on_empty_journal_do_not_pretend(tmp_path: Path) -> None:
    """На пустом журнале доли отсутствуют, а не равны нулю."""
    with Recorder(tmp_path / "s", profile=MILESTONE_0, source="t", synthetic=True):
        pass
    with Session.open(tmp_path / "s") as s:
        v = vitals.from_journal(s.journal, profile=MILESTONE_0)
    assert v.value("masked_share") is None
    assert v.value("competence") is None
    assert v.get("masked_share").note                    # причина указана
