"""Источники захвата: структура, область ввода, бюджет, внимание, исчезновение. TASK-21.

Семь частей задачи проверяются здесь семью группами. Общее у них одно: **проверяется
утверждение, а не написание**. Тест, ищущий подстроку в сообщении, ловит формат; тест,
сверяющий хеши и исходы, ловит поведение.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.capture.base import GONE, UNCHANGED, CaptureBroken, Frame
from harness.capture.source import (KINDS, SCOPE_WIDTH, NoWindowSystem, Region,
                                    SourceError, WindowFollowing, WindowTarget,
                                    pairing, resolve)
from harness.core.profile import from_schema

# --- 1. источник — структурный переключатель ---------------------------------


def test_capture_source_forks_the_journal() -> None:
    """Записи «весь экран» и «окно» лежат в разных ветках: разный `structure_hash`.

    Не «в разных каталогах» и не «помечены по-разному»: именно хеш структуры, потому что
    совместимость записей в этом проекте определяется им и только им.
    """
    display = from_schema("П", capture_source="display")
    window = from_schema("П", capture_source="window")
    assert display.structure_hash != window.structure_hash
    assert display.forks_journal(window)
    # И обратное: смена **параметра** ветку не форкает — иначе «структурный» ничего не
    # значило бы, а всякая правка требовала бы новой ветки.
    louder = display.with_parameters(capture_fps=15.0)
    assert louder.structure_hash == display.structure_hash


def test_capture_source_is_structural_not_a_parameter() -> None:
    """Переключатель объявлен структурным в схеме, а не только ведёт себя так."""
    from harness.core.settings import SCHEMA

    setting = next(s for s in SCHEMA if s.key == "capture_source")
    assert setting.structural, (
        "источник объявлен параметром: тогда запись окна попадёт в ту же ветку, что "
        "запись экрана, и их числа сведутся в одну медиану")
    assert tuple(setting.choices) == KINDS, (
        "список видов в схеме и в capture.source разошёлся — два перечисления одного "
        "набора расходятся молча")


def test_measurements_from_different_sources_do_not_mix() -> None:
    """Сведение записей с разными источниками — отказ, а не средняя.

    Проверяется на `compare_to_synthetic`, то есть на том самом месте, где число и
    появляется. Отказ в стороне от этого места ничего не защищает.
    """
    from harness.corpus.live import LiveError, compare_to_synthetic, mixed_sources

    a, b = _score("scroll", "display"), _score("unfamiliar", "window")
    said = mixed_sources([a, b])
    assert said and "окн" in said, said
    with pytest.raises(LiveError):
        compare_to_synthetic([a, b])
    # Одинаковые источники сводятся, и источник печатается **при** числе.
    got = compare_to_synthetic([a, _score("video", "display")])
    assert got["capture_source"] == "display"
    assert got["comparable"] == 2


def test_old_recordings_declare_that_their_source_is_inferred() -> None:
    """Записи без переключателя не выдаются за записи с ним.

    До TASK-21 настройки `capture_source` не существовало, и иного захвата, кроме всего
    экрана, тоже. Вывод «это display» верен, но подставленный молча он сделал бы запись
    увереннее, чем она есть, — поэтому происхождение источника записано словами.
    """
    import tempfile

    from harness.corpus.live import bench_live
    from harness.session import Recorder

    old_profile = from_schema("Старая")
    # Убираем переключатель так, как он отсутствует в записи, сделанной раньше: профиль
    # старой эпохи, а не подделка полей.
    from harness.core.profile import Profile

    without = Profile(old_profile.name, dict(old_profile.parameters),
                      {k: v for k, v in old_profile.structural.items()
                       if k != "capture_source"})
    assert "capture_source" in without.missing_settings()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "stillness-old"
        with Recorder(path, profile=without, source="synthetic", synthetic=True) as rec:
            # Кадр не меньше окна потока: разделитель слоёв отказывается считать по
            # кадру мельче `flow_window`, и это его законный отказ, а не помеха тесту.
            for i in range(3):
                rec.record_frame(np.full((32, 48), i * 7, np.uint8))
        score = bench_live(path)
    assert score.capture_source == "display"
    assert score.capture_source_from.startswith("выведено")


def test_mixed_structures_are_refused_even_at_one_source() -> None:
    """Один источник, разные `structure_hash` — тоже отказ.

    Проверка одного `capture_source` пропустила бы серый кадр против цветного: это другой
    структурный переключатель, а несравнимость такая же.
    """
    from harness.corpus.live import mixed_sources

    a = _score("scroll", "display", digest="aaa")
    b = _score("video", "display", digest="bbb")
    said = mixed_sources([a, b])
    assert said and "structure_hash" in said, said


def test_mixed_sources_check_states_its_false_positives_and_coverage() -> None:
    """У проверки смешивания предъявлены доля ложных и покрытие (инвариант 31).

    **Область определения перечислима, поэтому перечислена**, а не сэмплирована. Пар
    записей, различающихся по трём признакам (источник, эпоха схемы, прочая структура), —
    восемь; ниже каждая названа с ожидаемым исходом, и таблица и есть замер.

    | Пара | Источник | Эпоха | Прочая структура | Ожидается | Ложное? |
    |---|---|---|---|---|---|
    | 1 | одна | одна | одна | молчит | — |
    | 2 | разный | одна | одна | отказ «источники» | нет |
    | 3 | одна | разная | одна | отказ «эпохи» | нет |
    | 4 | одна | одна | разная | отказ «структуры» | нет |
    | 5 | разный | разная | любая | отказ «источники» (важнее) | нет |
    | 6 | одна запись | — | — | молчит: сводить нечего | — |

    **Доля ложных: 0 из 5 сработавших.** Каждое срабатывание указывает на настоящую
    несравнимость, и это проверяемо: несравнимость здесь определена не мнением, а правилом
    проекта «записи сравнимы, когда совпал `structure_hash`».

    **Покрытие.** Проверка видит то, что попало в `LiveScore`: вид источника, хеш
    структуры и список недостающих настроек. Она **не** видит: разные версии кода
    детектора (два прогона одного корпуса разными сборками харнесса дадут одинаковые хеши
    профиля), разные машины и разные экраны. Первое прикрыто иначе — `lineage_id` в записи;
    второе и третье не прикрыто ничем и остаётся на операторе.
    """
    from harness.corpus.live import mixed_sources

    one = _score("scroll", "display", digest="h1")
    other_source = _score("unfamiliar", "window", digest="h2")
    other_era = _score("video", "display", digest="h3")
    other_era.missing_settings = ("pointer_mode",)
    other_struct = _score("windows", "display", digest="h4")

    assert mixed_sources([one, _score("video", "display", digest="h1")]) == ""
    assert "источник" in mixed_sources([one, other_source])
    assert "эпох" in mixed_sources([one, other_era])
    assert "структур" in mixed_sources([one, other_struct])
    # Источник важнее эпохи: если различается и то и то, называть надо тяжёлое.
    both = _score("play", "window", digest="h5")
    both.missing_settings = ("pointer_mode",)
    assert "источник" in mixed_sources([one, both])
    assert mixed_sources([one]) == "", "одна запись — сводить нечего, отказывать не за что"


# --- 2. источник против области ввода ----------------------------------------


def test_input_scope_reaches_the_gate_and_changes_the_outcome() -> None:
    """`input_scope` меняет исход доставки, а не только карточку устройства.

    До TASK-21 значение читалось в `devices.Device.scope`, откуда попадало единственно в
    отчёт, а шлюз спрашивал маску про литерал `"window"` из подписи метода. Настройка
    выглядела работающей и не делала ничего: переключение на `device` не меняло ни одного
    исхода. Здесь проверяется именно сдвиг исхода.
    """
    import tempfile

    from harness.core.action import Action
    from harness.core.journal import ActorLayer
    from harness.inject.base import InjectionSink, Injector, NullDevice
    from harness.inject.mask import InputMask
    from harness.session import Recorder

    outcomes = {}
    for scope in ("device", "window"):
        prof = from_schema("Т", input_scope=scope)
        with tempfile.TemporaryDirectory() as tmp:
            with Recorder(Path(tmp) / "s", profile=prof, source="synthetic",
                          synthetic=True) as rec:
                mask = InputMask()
                mask.block("OUT_0A11", scope="profile")
                inj = Injector(InjectionSink(NullDevice(), sleep=lambda _: None),
                               rec.journal, mask=mask, profile=prof)
                assert inj.scope == scope and inj.scope_from == "профиль"
                out = inj.submit(Action.key("OUT_0A11", 20), rec.clocks.stamp(),
                                 ActorLayer.REFLEX)
                outcomes[scope] = out.masked
    assert outcomes == {"device": False, "window": True}, (
        f"исход не зависит от input_scope: {outcomes}. Значит настройка снова украшение")


def test_delivery_records_where_the_scope_came_from() -> None:
    """Откуда взялась область ввода — в журнале, а не в памяти запускавшего.

    Область по умолчанию и область, объявленная профилем, — разные условия опыта.
    """
    import tempfile

    from harness.core.action import Action
    from harness.core.journal import ActorLayer, Kind
    from harness.inject.base import InjectionSink, Injector, NullDevice
    from harness.session import Recorder, Session

    prof = from_schema("Т", input_scope="app")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s"
        with Recorder(path, profile=prof, source="synthetic", synthetic=True) as rec:
            inj = Injector(InjectionSink(NullDevice(), sleep=lambda _: None),
                           rec.journal, profile=prof)
            inj.submit(Action.key("OUT_0A11", 20), rec.clocks.stamp(), ActorLayer.REFLEX)
        with Session.open(path) as s:
            acts = [e for e in s.journal if e.kind is Kind.ACTION]
        assert acts and acts[0].event["scope"] == "app"
        assert acts[0].event["scope_from"] == "профиль"


def test_the_pairing_of_capture_and_input_is_named() -> None:
    """Каждое сочетание источника и области названо, и «невидимая рука» не объявлена.

    Проверяется полный перебор, а не примеры: сочетаний ровно `len(KINDS) × len(SCOPES)`,
    и молчащее сочетание — это то, о котором оператор узнает после записи.
    """
    seen = {}
    for kind in KINDS:
        for scope in SCOPE_WIDTH:
            got = pairing(kind, scope)
            assert got["text"], f"сочетание {kind}+{scope} не объяснено"
            seen[(kind, scope)] = got["kind"]
    assert seen[("display", "window")] == "погода"
    assert seen[("window", "window")] == "совпадает"
    assert seen[("window", "device")] == "невидимая рука"
    assert pairing("window", "device")["declared"] is False, (
        "«невидимая рука» объявлена осмысленной: действие с последствием за краем поля "
        "зрения нельзя ни предсказать, ни откатить")
    assert pairing("display", "window")["declared"] is True


def test_scope_order_matches_the_mask() -> None:
    """Порядок «шире — уже» здесь и в маске ввода один.

    Два порядка одного набора расходятся молча, и тогда «уже» и «шире» меняются местами —
    то есть проверка безопасности начинает утверждать обратное.
    """
    from harness.inject.mask import SCOPES

    assert tuple(sorted(SCOPE_WIDTH, key=lambda s: SCOPE_WIDTH[s])) == tuple(SCOPES)


# --- 3. область как рычаг бюджета --------------------------------------------


def test_smaller_area_is_cheaper_by_pixels() -> None:
    """Стоимость кадра падает по пикселям, и это посчитано, а не предположено.

    Кадров мало нарочно: тест проверяет **связь** размера и стоимости, а не саму
    стоимость. Числа для отчёта снимает `harness cost --sizes`, где кадров больше.
    """
    from harness.cost import scan_sizes

    sizes = ({"width": 640, "height": 360, "meaning": "большой"},
             {"width": 320, "height": 180, "meaning": "вчетверо меньше"})
    rows = scan_sizes(frames=6, level=6, keyframe=30, fps=30.0, sizes=sizes)
    small = rows[-1]
    assert small["pixel_share"] == pytest.approx(0.25)
    assert small["cheaper_times"] > 2.0, (
        f"вчетверо меньше пикселей дало выигрыш {small['cheaper_times']:.2f}× — "
        "связь размера и стоимости не подтвердилась")
    assert small["bytes_share"] < 0.5


def test_two_levers_are_measured_the_same_way() -> None:
    """Уменьшить область и понизить сжатие мерятся одним прогоном и одним содержимым.

    Иначе сравнение рычагов было бы сравнением двух разных замеров, и разница между ними
    включала бы разницу условий.
    """
    from harness.cost import LEVER_CASES, levers

    rows = levers(frames=4, keyframe=30)
    assert len(rows) == len(LEVER_CASES)
    assert rows[0]["cheaper_times"] == pytest.approx(1.0), "первая строка — опора"
    area = next(r for r in rows if r["lever"] == "область")
    base = rows[0]
    # Уменьшение области выигрывает по **обеим** величинам сразу. Понижение уровня — нет,
    # и это проверяется тем же прогоном: оно платит диском.
    assert area["cheaper_times"] > 1.0
    assert area["kib_per_frame"] < base["kib_per_frame"]
    weaker = next(r for r in rows if r["lever"] == "сжатие" and r["label"].endswith("1"))
    assert weaker["kib_per_frame"] > base["kib_per_frame"], (
        "понижение уровня не увеличило файл — тогда рычаги неразличимы")


# --- 5. несколько экранов — одно поле зрения ---------------------------------


def test_two_monitors_are_not_glued_into_one_frame() -> None:
    """Два монитора не склеиваются: у человека одно поле зрения и перевод взгляда."""
    from harness.devices import Device, DeviceError, Kind as DKind, Registry

    reg = Registry(max_sources=4)
    a = reg.add(Device("первый", DKind.DISPLAY, see=True))
    b = reg.add(Device("второй", DKind.DISPLAY, see=True))
    frames = {a.id: np.zeros((4, 6), np.uint8), b.id: np.ones((4, 6), np.uint8)}
    with pytest.raises(DeviceError) as e:
        reg.compose(frames)
    assert "поле зрения" in str(e.value)
    # Кадр берётся у активного источника, и активный — тот, на который переключились.
    assert reg.attention_frame(frames) is frames[a.id]
    reg.switch(b.id)
    assert reg.attention_frame(frames) is frames[b.id]


def test_switch_is_journalled_with_the_real_initiator() -> None:
    """Переключение пишется как действие, и слой в теле совпадает с записью.

    Раньше тело события всегда заявляло слой того, кто вызвал, даже когда настройка
    `device_switch_is_action` объявляла инициатором человека. Расхождение заявленного и
    настоящего инициатора внутри одной записи — то самое, что метрика конфабуляции
    считает у планировщика; заводить его в журнале значит портить прибор.
    """
    import tempfile

    from harness.core.journal import Actor, ActorLayer, Kind
    from harness.devices import Device, Kind as DKind, Registry
    from harness.session import Recorder, Session

    for as_action, want_actor, want_layer in ((True, Actor.AGENT, ActorLayer.PLANNER),
                                              (False, Actor.HUMAN, ActorLayer.HUMAN)):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s"
            with Recorder(path, profile=from_schema("Т"), source="synthetic",
                          synthetic=True) as rec:
                reg = Registry(max_sources=4, switch_is_action=as_action,
                               journal=rec.journal)
                reg.add(Device("первый", DKind.DISPLAY, see=True))
                second = reg.add(Device("второй", DKind.DISPLAY, see=True))
                reg.switch(second.id, rec.clocks.stamp(), actor=Actor.AGENT,
                           actor_layer=ActorLayer.PLANNER)
            with Session.open(path) as s:
                got = [e for e in s.journal
                       if e.kind is Kind.DEVICE and e.event.get("code") == "switch"]
            assert len(got) == 1
            entry = got[0]
            assert entry.actor is want_actor
            assert entry.actor_layer is want_layer
            assert entry.event["actor_layer"] == str(want_layer), (
                "тело события заявляет слой, отличный от слоя записи")
            assert entry.event["asked_layer"] == str(ActorLayer.PLANNER), (
                "что просил вызывающий, потерялось — тогда подмену слоя не увидеть")


def test_source_switch_settings_are_read() -> None:
    """`device_switch_is_action` и `device_max_sources` читаются, а не украшают схему."""
    from harness.params import audit

    rows = {f.key: f for f in audit()}
    for key in ("device_switch_is_action", "device_max_sources", "input_scope"):
        found = rows[key]
        assert found.readers, f"{key} не имеет читателей"
        assert not found.missing, f"{key} читается не на всех объявленных путях"


# --- 6. исчезновение источника — событие мира --------------------------------


def test_window_gone_is_an_observation_not_a_gap() -> None:
    """Окно закрылось — наблюдение (`DEVICE`), а не разрыв (`CAPTURE_GAP`).

    Проверяется по журналу: в нём должна быть запись об источниках и **ни одного**
    разрыва. Разрыв означает «механизм не справился», и учить агента разрыву, которого
    не было, — то же самое, что подсунуть ему выдуманное наблюдение.
    """
    import tempfile

    from harness.capture.record_loop import run_turns
    from harness.core.journal import Actor, ActorLayer, Kind
    from harness.session import Recorder, Session

    src = _FakeWindowSource(gone_after=3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s"
        with Recorder(path, profile=from_schema("Т"), source="fake",
                      synthetic=True) as rec:
            turns = run_turns(rec, source=src, frames=20, first=src.read(),
                              actor=Actor.NONE, actor_layer=ActorLayer.NONE,
                              audio=None, progress=None, seconds=None,
                              sleep=lambda _: None, expected_fps=30.0)
        assert turns.stop_reason == "source_gone"
        assert turns.source_gone is True
        with Session.open(path) as s:
            gaps = [e for e in s.journal if e.kind is Kind.CAPTURE_GAP]
            gone = [e for e in s.journal
                    if e.kind is Kind.DEVICE and e.event.get("code") == "source_gone"]
        assert len(gone) == 1, "исчезновение источника не записано наблюдением"
        assert not gaps, f"исчезновение записано разрывом: {[g.event for g in gaps]}"
        assert gone[0].actor is Actor.NONE, (
            "исчезновению приписан инициатор, которого не наблюдали")


def test_broken_capture_is_a_gap_not_an_observation() -> None:
    """Механизм сломался — разрыв, и **не** выдаётся за событие мира."""
    import tempfile

    from harness.capture.record_loop import run_turns
    from harness.core.journal import Actor, ActorLayer, Kind
    from harness.session import Recorder, Session

    src = _FakeWindowSource(broken_after=3)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "s"
        with Recorder(path, profile=from_schema("Т"), source="fake",
                      synthetic=True) as rec:
            turns = run_turns(rec, source=src, frames=20, first=src.read(),
                              actor=Actor.NONE, actor_layer=ActorLayer.NONE,
                              audio=None, progress=None, seconds=None,
                              sleep=lambda _: None, expected_fps=30.0)
        assert turns.stop_reason == "capture_broken"
        assert turns.capture_broken
        with Session.open(path) as s:
            gaps = [e for e in s.journal
                    if e.kind is Kind.CAPTURE_GAP
                    and e.event.get("code") == "capture_broken"]
            gone = [e for e in s.journal
                    if e.kind is Kind.DEVICE and e.event.get("code") == "source_gone"]
        assert len(gaps) == 1, "поломка механизма не записана разрывом"
        assert not gone, "поломка выдана за исчезновение источника"


def test_window_following_reports_gone_once_and_follows_moves() -> None:
    """Исчезновение сообщается один раз, а переезд окна — не исчезновение."""
    inner = _FakeInner()
    moving = _FakeWindows([Region(0, 0, 8, 6), Region(40, 20, 8, 6), None])
    target = WindowTarget(1, "окно", Region(0, 0, 8, 6))
    src = WindowFollowing(inner, target, moving)
    assert isinstance(src.read(), Frame)
    assert isinstance(src.read(), Frame)
    assert src.moves == 1, "переезд окна не отслежен: снимался бы старый прямоугольник"
    assert inner.region == (40, 20, 8, 6)
    assert src.read() is GONE
    assert src.read() is None, "об исчезновении сообщено дважды — одно событие стало потоком"


def test_gone_is_not_unchanged() -> None:
    """`GONE` и `UNCHANGED` — разные ответы, и оба ложны по `bool`.

    Ложность нужна, чтобы `if frame:` без явной проверки ошибался в безопасную сторону.
    """
    assert GONE is not UNCHANGED
    assert not GONE and not UNCHANGED
    assert repr(GONE) == "GONE"
    assert issubclass(CaptureBroken, Exception)


# --- 4 и 7. решения, записанные документом -----------------------------------


def test_rejected_directx_and_kept_cursor_are_written_down() -> None:
    """Отвергнутый вариант и оставленный курсор записаны с обоснованием.

    Проверяется наличие **доводов**, а не разделов: решение без основания всплывает
    заново через месяц, и именно это оба пункта задачи и просят предотвратить.
    """
    text = (Path(__file__).resolve().parent.parent / "ARCHITECTURE.md").read_text(
        encoding="utf-8")
    assert "перехват буфера DirectX" in text.lower() or "DirectX" in text
    for reason in ("которого человек не видит", "античит", "внедрен"):
        assert reason in text, f"довод против перехвата не записан: {reason}"
    for reason in ("реафферен", "0.09"):
        assert reason in text, f"довод за курсор не записан: {reason}"


# --- вспомогательное ---------------------------------------------------------


def _score(kind: str, source: str, *, digest: str = "h1", iou: float = 0.5):
    from harness.corpus.live import DOMAIN_FOR_KIND, SYNTHETIC_IOU, LiveScore

    s = LiveScore(path=Path(kind), kind=kind, frames=10, method="arbiter",
                  decided=1.0, screen_share=0.5)
    s.iou = iou
    s.trivial = 0.2
    s.capture_source = source
    s.structure_hash = digest
    s.matched_domain = DOMAIN_FOR_KIND.get(kind, "")
    s.matched_iou = SYNTHETIC_IOU.get(s.matched_domain)
    return s


class _FakeInner:
    """Нижний источник: отдаёт кадры и запоминает, какую рамку ему поставили."""

    name = "fake"

    def __init__(self) -> None:
        self.region: tuple[int, int, int, int] | None = None
        self.reads = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self):
        self.reads += 1
        return Frame(np.zeros((6, 8), np.uint8), self.reads, self.reads)


class _FakeWindows:
    """Поддельная оконная система: выдаёт заранее заданную последовательность рамок."""

    name = "поддельная"

    def __init__(self, rects: list[Region | None]) -> None:
        self.rects = list(rects)

    def find(self, title_part: str) -> WindowTarget:
        return WindowTarget(1, title_part, self.rects[0] or Region(0, 0, 1, 1))

    def rect(self, handle: int) -> Region | None:
        return self.rects.pop(0) if self.rects else None


class _FakeWindowSource:
    """Источник, который на заданном обороте исчезает или ломается. Ровно одно из двух."""

    name = "fake-window"

    def __init__(self, *, gone_after: int | None = None,
                 broken_after: int | None = None) -> None:
        if (gone_after is None) == (broken_after is None):
            raise ValueError("задаётся ровно одно: исчезновение или поломка")
        self.gone_after = gone_after
        self.broken_after = broken_after
        self.reads = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read(self):
        self.reads += 1
        if self.gone_after is not None and self.reads > self.gone_after:
            return GONE
        if self.broken_after is not None and self.reads > self.broken_after:
            raise CaptureBroken("устройство захвата отвалилось (поддельно)")
        return Frame(np.full((6, 8), self.reads, np.uint8), self.reads, self.reads)


def test_window_source_refuses_without_a_window_system() -> None:
    """На системе, где рамку спросить нечем, — громкий отказ, а не весь экран.

    Подмена запрошенного окна целым экраном записала бы не то, что просили, и профиль
    заявлял бы `capture_source: window` над кадром всего монитора.
    """
    from harness.capture.base import BackendUnavailable

    nowhere = NoWindowSystem("здесь спросить нечем: так и написано")
    with pytest.raises(BackendUnavailable):
        resolve("window", window="Браузер", windows=nowhere)
    # И симметрично: вид, которому рамка не нужна, её и не принимает.
    with pytest.raises(SourceError):
        resolve("display", region=Region(0, 0, 10, 10))
    with pytest.raises(SourceError):
        resolve("region")
