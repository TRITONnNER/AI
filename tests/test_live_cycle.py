"""Арена, живой цикл и прогон М4 по живому материалу. TASK-22.

Две половины задачи проверяются здесь раздельно:

- **М5** — арена с мгновенным респавном и цикл без паузы мира. Идёт на синтетике, зрения не
  требует, поэтому проверяется целиком.
- **TASK-22, части 1–3** — прогон по живым записям. Живых записей в контейнере нет, и
  проверяется то, что можно: список проверок объявлен заранее, отказ при пустом корпусе
  громкий, а механика прогона работает на записи, подставленной вместо живой.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.core.profile import from_schema

# --- арена -------------------------------------------------------------------


def test_arena_respawns_without_stopping_the_world(tmp_path: Path) -> None:
    """Респавн не останавливает мир: часы мира идут через границу эпизода.

    Проверяется по журналу и по часам мира, а не по отсутствию вызова `pause`: паузы в коде
    может не быть, а мир всё равно простоит, если между эпизодами он не тикает.
    """
    from harness.behaviour.arena import Arena, reach_place
    from harness.core.action import Action
    from harness.core.journal import Kind
    from harness.session import Recorder, Session

    prof = from_schema("АРЕНА", capture_width=64, capture_height=48)
    with Recorder(tmp_path / "s", profile=prof, source="arena", synthetic=True) as rec:
        arena = Arena(reach_place(prof, distance_px=1e9), journal=rec.journal, seed=1)
        arena.start(rec.clocks.stamp())
        first = arena.world
        ticks = 0
        for _ in range(30):
            arena.step(Action.key(arena.world.outputs[0], 100), rec.clocks.stamp())
            ticks += 1
        arena.scenario = arena.scenario.__class__(
            arena.scenario.name, arena.scenario.build, arena.scenario.goal,
            budget_ticks=10, why=arena.scenario.why)
        ep = arena.respawn(rec.clocks.stamp())
        assert ep.index == 1, "новый эпизод не начат"
        assert arena.world is not first, "мир не пересоздан: это не респавн"
        for _ in range(5):
            arena.step(Action.key(arena.world.outputs[0], 100), rec.clocks.stamp())
            ticks += 1
        arena.close("прервано", rec.clocks.stamp())

    with Session.open(tmp_path / "s") as s:
        marks = [e for e in s.journal if e.kind is Kind.GOAL]
        codes = [e.event.get("code") for e in marks]
    # Обе границы каждого эпизода в журнале: тайная перезагрузка невозможна.
    assert codes.count("episode_start") == 2
    assert codes.count("episode_end") == 2
    # Инвариант 17: у возрождения есть инициатор, и это не человек.
    from harness.core.journal import Actor, ActorLayer

    assert all(e.actor is Actor.AGENT for e in marks)
    assert all(e.actor_layer is ActorLayer.DRIVE for e in marks)


def test_arena_prefers_the_goal_over_the_budget(tmp_path: Path) -> None:
    """Цель, достигнутая последним тиком бюджета, записывается как достигнутая.

    Обратный порядок проверок увёл бы данные о достижимости цели в пессимизм: эпизоды,
    закончившиеся успехом на последнем тике, считались бы «не успел».
    """
    from harness.behaviour.arena import Arena, Scenario

    class World:
        t_world = 0
        outputs = ("OUT_0A11",)
        last_action_changed = True

        def step(self, action):        # noqa: ANN001 — мир поддельный, подпись мира
            self.t_world += 1
            return None

    def goal(world):                   # noqa: ANN001
        return (lambda: world.t_world >= 3), "три тика мира"

    arena = Arena(Scenario("предел", lambda seed: World(), goal, budget_ticks=3,
                           why="проверка порядка исходов"), seed=0)
    arena.start()
    for _ in range(3):
        arena.step("шаг")
    assert arena.finished() == "достигнута", (
        "исход выбран по бюджету при достигнутой цели: порядок проверок обратный")


def test_scenario_without_declared_purpose_is_refused() -> None:
    """Сценарий без объявленного смысла не собирается: он набирал бы данные ни о чём."""
    from harness.behaviour.arena import ArenaError, Scenario

    with pytest.raises(ArenaError):
        Scenario("безымянный", lambda seed: None, lambda w: (lambda: True, "тест"),
                 budget_ticks=5)


# --- живой цикл --------------------------------------------------------------


def test_live_cycle_never_pauses_the_world(tmp_path: Path) -> None:
    """Оборотов цикла ровно столько же, сколько тиков мира. Инвариант 3 числом."""
    from harness.behaviour.arena import reach_place
    from harness.livecycle import run

    prof = from_schema("М5", capture_width=64, capture_height=48)
    got = run(tmp_path / "s", scenario=reach_place(prof), profile=prof,
              seconds=0.4, seed=1, hz_scale=40.0)
    assert got.loops > 10, "цикл не покрутился: проверять нечего"
    assert got.loops == got.world_ticks
    assert got.world_never_paused


def test_subsumption_actually_fires(tmp_path: Path) -> None:
    """Нижний контур перебивает верхний **не ноль раз**.

    Ветка с нулём срабатываний — дефект (инвариант 26). При коротком поиске планировщик
    успевает закончить внутри среза, и субсумпция не проверяется вовсе: «не сработало,
    потому что не понадобилось» неотличимо от «не работает».
    """
    from harness.behaviour.arena import reach_place
    from harness.livecycle import run

    prof = from_schema("М5", capture_width=64, capture_height=48)
    got = run(tmp_path / "s", scenario=reach_place(prof), profile=prof,
              seconds=0.6, seed=2, hz_scale=40.0)
    assert got.preempted > 0, (
        "рефлекс ни разу не перебил планировщика — субсумпция не проверена этим прогоном")
    planner = next(c for c in got.contours if c["name"] == "planner")
    assert planner["preempted"] > 0
    assert planner["slices"] > planner["runs"], (
        "планировщик заканчивал за один срез: долгая работа не проверена")


def test_every_delivered_action_carries_its_initiator(tmp_path: Path) -> None:
    """У каждого доставленного действия есть слой, и он тот, чей контур ответил."""
    from harness.behaviour.arena import reach_place
    from harness.core.journal import ActorLayer, Kind
    from harness.livecycle import LAYER_OF, run
    from harness.session import Session

    prof = from_schema("М5", capture_width=64, capture_height=48)
    got = run(tmp_path / "s", scenario=reach_place(prof), profile=prof,
              seconds=0.5, seed=3, hz_scale=40.0)
    assert got.layers_complete
    assert str(ActorLayer.NONE) not in got.by_layer
    with Session.open(tmp_path / "s") as s:
        actions = [e for e in s.journal if e.kind is Kind.ACTION]
    assert actions, "действий не записано"
    for e in actions:
        contour = e.event.get("contour", "")
        assert e.actor_layer is LAYER_OF[contour], (
            f"действие от контура «{contour}» записано слоем {e.actor_layer}")


def test_confabulation_no_longer_equals_the_reflex_share(tmp_path: Path) -> None:
    """Метрика перестала совпадать с долей действий рефлекса. TASK-24, направление A.

    До правки объяснение планировщика стояло до следующего объяснения и накрывало все
    действия подряд: «объяснено не тем слоем» было тождественно «начато не планировщиком»,
    и метрика совпадала с долей рефлекса до 0.12 п.п. Теперь объяснение прикрепляется
    только под живым ожиданием — там, где планировщик выдал команду и ждёт исполнения, —
    и расхождение возникает между его убеждением и фактом.

    Проверяется **разница**, а не значение: значение зависит от частот, а требование
    задачи было именно про несовпадение двух величин.
    """
    from harness.behaviour.arena import reach_place
    from harness.livecycle import run

    prof = from_schema("М5", capture_width=64, capture_height=48)
    got = run(tmp_path / "s", scenario=reach_place(prof), profile=prof,
              seconds=0.8, seed=4, hz_scale=40.0)
    conf = got.confabulation
    assert conf.get("share_by_explanation") is not None, (
        f"метрика не посчиталась: {conf.get('absent_reason')}")
    reflex = got.by_layer.get("reflex", 0) / max(1, got.delivered)
    gap = abs(conf["share_by_explanation"] - reflex)
    assert gap > 0.05, (
        f"доля расхождений {conf['share_by_explanation']:.3f} снова совпала с долей "
        f"рефлекса {reflex:.3f} (разница {gap:.3f}). Значит постановка опять вырождена, и "
        "это надо сказать в MEASUREMENT.md, а не подгонять порог")
    # И то, из-за чего разница появилась: объяснение накрывает **не все** действия.
    assert 0 < got.under_expectation < got.delivered
    assert got.expectations > 0 and got.expectations_expired > 0


def test_explanation_is_attached_only_under_a_live_expectation(tmp_path: Path) -> None:
    """Действие вне ожидания планировщика остаётся **без** объяснения.

    Это и есть переставленная ссылка: планировщик заявляет о себе там, где выдал команду и
    ждёт её исполнения, а не там, где просто говорил последним.
    """
    from harness.behaviour.arena import reach_place
    from harness.core.journal import Kind
    from harness.livecycle import run
    from harness.session import Session

    prof = from_schema("М5", capture_width=64, capture_height=48)
    got = run(tmp_path / "s", scenario=reach_place(prof), profile=prof,
              seconds=0.8, seed=5, hz_scale=40.0)
    with Session.open(tmp_path / "s") as s:
        actions = [e for e in s.journal if e.kind is Kind.ACTION]
    with_reason = [e for e in actions if e.state.stated_reason_id]
    without = [e for e in actions if not e.state.stated_reason_id]
    assert with_reason and without, (
        "либо все действия объяснены, либо ни одно: ссылка стоит не по ожиданию")
    # Под ожиданием бывают действия **обоих** слоёв — иначе расхождение было бы
    # невозможно по построению, и метрика опять мерила бы тождество.
    layers = {str(e.actor_layer) for e in with_reason}
    assert len(layers) >= 2, f"под ожиданием только один слой: {layers}"
    assert len(with_reason) == got.under_expectation


# --- TASK-22: прогон по живому материалу --------------------------------------


def test_m4_live_refuses_loudly_on_an_empty_corpus(tmp_path: Path) -> None:
    """Пустой корпус — громкий отказ с командами, а не ноль.

    И вместе с отказом печатается **вся объявленная опись**: список проверок зафиксирован
    до появления записей, и это единственное, что мешает выбрать проверки под данные.
    """
    from harness.m4live import CHECKS, run

    rep = run(tmp_path / "нет-корпуса")
    assert rep.refused
    assert "harness ingest" in rep.refused
    text = rep.render_text()
    for check in CHECKS:
        assert check.key in text, f"проверка {check.key} не названа в отказе"
        assert check.synthetic in text


def test_every_declared_check_names_its_unit_and_why_it_matters() -> None:
    """У каждой проверки описи есть единица независимости и причина, почему она важна.

    Проверка без единицы не создаётся (инвариант 22); проверка без объявленной важности
    через месяц неотличима от собранной для полноты.
    """
    from harness.m4live import CHECKS, VERDICTS

    assert len(CHECKS) >= 10
    keys = [c.key for c in CHECKS]
    assert len(keys) == len(set(keys)), "ключи проверок повторяются"
    for c in CHECKS:
        assert c.unit in ("домен", "запись", "корпус", "источник отпечатка"), c.unit
        assert c.synthetic and c.matters and c.what
    assert VERDICTS == ("совпало", "разошлось", "нечем проверить")


def test_m4_live_runs_end_to_end_on_a_stand_in_recording(tmp_path: Path) -> None:
    """Механика прогона работает на записи, подставленной вместо живой.

    Живых записей в контейнере нет; синтетическая запись здесь — **не** замена живого
    материала, а способ проверить, что прогон не падает и заполняет опись. Числа из этого
    прогона результатом не являются и в отчёты не идут.
    """
    from harness.corpus.live import write_regions, Region
    from harness.m4live import run
    from harness.session import Recorder

    prof = from_schema("ПОДСТАВНАЯ", capture_width=64, capture_height=48)
    corpus = tmp_path / "corpus"
    path = corpus / "stillness-подставная"
    with Recorder(path, profile=prof, source="synthetic", synthetic=True) as rec:
        rng = np.random.default_rng(5)
        base = rng.integers(0, 255, size=(48, 64), dtype=np.uint8)
        for i in range(12):
            rec.record_frame(base if i % 3 else np.roll(base, i, axis=1))
    write_regions(path, [Region(0, 0, 10, 64, "screen")], note="подставная разметка")

    rep = run(corpus)
    assert not rep.refused
    keys = {f.key for f in rep.findings}
    # Все проверки, которые можно заполнить по одной записи, заполнены.
    for key in ("iou", "trivial", "signal", "decided", "integrity", "fitness", "era",
                "places", "clock_gap", "identity_live", "marked"):
        assert key in keys, f"проверка {key} не заполнена"
    assert all(f.verdict in ("совпало", "разошлось", "нечем проверить")
               for f in rep.findings)
    # Опись — не починка: у каждого расхождения есть текст причины, и ни одного действия.
    for f in rep.diverged:
        assert f.why or f.got is not None


def test_live_fingerprint_recovers_identity_on_a_quiet_stretch() -> None:
    """Живой отпечаток дробит тождество, а пересмотр его собирает: ошибка падает до нуля.

    Это и есть ответ части 2 в проверяемой форме: механика слияния не зависит от того,
    приходит отпечаток от скаляра или от пикселей. На дизеринге ±1 уровень шесть областей
    дают больше шести карточек, и сон сводит их обратно к шести.
    """
    from harness.m4live import identity_on_sources, quiet_run

    rng = np.random.default_rng(7)
    base = rng.integers(0, 255, size=(90, 160), dtype=np.uint8)
    frames = [base.copy() for _ in range(12)]
    for f in frames[1:]:
        f[:] = np.clip(f.astype(np.int16) + rng.integers(-1, 2, f.shape),
                       0, 255).astype(np.uint8)
    assert quiet_run(frames)[1] >= 8, "отрезок не спокойный: проверять нечего"

    got = identity_on_sources(frames)
    live = next(c for c in got["cases"] if c["source"] == "live")
    assert live["error_before"] > 0, (
        "живой отпечаток не раздробил тождество — тогда проверять нечего, "
        "и постановка не воспроизводит живой случай")
    assert live["error_after"] == 0, (
        f"пересмотр не собрал карточки обратно: ошибка {live['error_after']}")
    assert got["n_sources"] == 3


def test_a_moving_screen_gives_no_identity_answer_instead_of_a_wrong_one() -> None:
    """Нет спокойного отрезка — «нечем проверить», а не число.

    Первая редакция брала подряд идущие кадры любого отрезка и получила 33 карточки при
    шести сущностях. Число выглядело результатом и им не было: на движущемся экране
    фиксированная область — не одна и та же вещь.
    """
    from harness.m4live import identity_on_sources

    rng = np.random.default_rng(11)
    frames = [rng.integers(0, 255, size=(90, 160), dtype=np.uint8) for _ in range(12)]
    got = identity_on_sources(frames)
    assert got["n_sources"] == 2, "живой источник взялся там, где встреч не существует"
    assert any("нечем проверить" in r for r in got["rows"])
    assert got["quiet_len"] < 8


# --- TASK-29, D2: смерть верхнего контура --------------------------------------


def test_agent_keeps_acting_when_the_planner_dies(tmp_path: Path) -> None:
    """Планировщик недоступен — агент продолжает на рефлексах, а мир не встаёт.

    Контур снимается целиком, а не «отвечает пусто»: недоступный сервис не присылает
    пустой ответ, он не присылает ничего. Пустой ответ был бы другим, более мягким
    механизмом, чем бывает в жизни.
    """
    from harness.behaviour.arena import reach_place
    from harness.livecycle import run

    prof = from_schema("М5", capture_width=64, capture_height=48)
    got = run(tmp_path / "s", scenario=reach_place(prof), profile=prof,
              seconds=0.8, seed=4, hz_scale=40.0, kill_planner_at=200)
    assert got.planner_died_at == 200
    assert got.delivered_after_death > 0, (
        "после смерти верхнего контура агент не сделал ничего — значит он остановился "
        "вместе с планировщиком, а не продолжил на рефлексах")
    assert got.world_never_paused, "смерть контура не отменяет инвариант 3"


def test_the_drop_is_visible_without_a_single_word_from_the_agent(tmp_path: Path) -> None:
    """«Стал глупее» — объективная величина, а не самоотчёт (инвариант 10).

    Доля действий, за которые планировщик считает себя причиной, обращается в ноль. Это
    видно по журналу и по счётчикам, и ни одно слово агента в проверку не входит.
    """
    from harness.behaviour.arena import reach_place
    from harness.livecycle import run

    prof = from_schema("М5", capture_width=64, capture_height=48)
    got = run(tmp_path / "s", scenario=reach_place(prof), profile=prof,
              seconds=0.8, seed=4, hz_scale=40.0, kill_planner_at=200)
    assert got.planner_share_before and got.planner_share_before > 0.1, (
        f"до смерти контура планировщик вёл {got.planner_share_before} действий — "
        "падать нечему, и замер вырожден")
    assert got.planner_share_after == 0.0
    assert got.noticed_by_expectations


def test_noticing_has_both_error_rates(tmp_path: Path) -> None:
    """Инвариант 32: и ложная тревога, и ложное подтверждение, обе с объявленным способом.

    **Ложная тревога:** контур снят на первом обороте — планировщик не успел подействовать
    ни разу, замечать нечего. **Ложное подтверждение:** контур жив весь прогон — «заметил»
    обязан молчать, и это опаснее тревоги, потому что закрывает вопрос.
    """
    from harness.behaviour.arena import reach_place
    from harness.livecycle import run

    prof = from_schema("М5", capture_width=64, capture_height=48)
    early = run(tmp_path / "early", scenario=reach_place(prof), profile=prof,
                seconds=0.8, seed=4, hz_scale=40.0, kill_planner_at=1)
    assert not early.noticed_by_expectations, (
        "ложная тревога: планировщик не действовал ни разу, а падение «замечено»")
    assert early.delivered_after_death > 0, "и при этом агент всё равно действовал"

    alive = run(tmp_path / "alive", scenario=reach_place(prof), profile=prof,
                seconds=0.8, seed=4, hz_scale=40.0)
    assert alive.planner_died_at is None
    assert not alive.noticed_by_expectations, (
        "ложное подтверждение: контур жив, а падение «замечено»")
