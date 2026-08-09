"""Путь «действие дошло до устройства».

Все остальные тесты инъекции проверяют ветки отказа: маску, СТОП, отсутствие
устройства. Успешная доставка — отдельная ветка, и её надо проверять именно
потому, что она самая незаметная: если модификаторы окажутся зажаты в неверном
порядке или не отпущены, игра будет вести себя странно, а журнал будет выглядеть
идеально.

Здесь стоит поддельное устройство, записывающее вызовы. Оно не подменяет
результат — оно и есть предмет проверки: важно, *что* харнесс приказал сделать.
Дошло ли это до настоящей игры, проверяется только на живой машине, и это
записано в `docs/ARCHITECTURE-HARNESS.md`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.core.action import Action, Reversibility
from harness.core.profile import MILESTONE_0
from harness.core.journal import ActorLayer
from harness.inject.base import Injector, InjectionSink
from harness.inject.mask import InputMask
from harness.inject.stop import StopSwitch
from harness.session import Recorder, Session


class FakeDevice:
    """Записывает приказы вместо того, чтобы их исполнять."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.closed = False

    def key_down(self, output: str) -> None:
        self.calls.append(("down", output))

    def key_up(self, output: str) -> None:
        self.calls.append(("up", output))

    def mouse_move(self, dx: int, dy: int) -> None:
        self.calls.append(("move", (dx, dy)))

    def close(self) -> None:
        self.closed = True


def _rec(tmp_path: Path) -> Recorder:
    return Recorder(tmp_path / "s", profile=MILESTONE_0, source="test", synthetic=True)


def test_modifiers_are_pressed_first_and_released_last(tmp_path: Path) -> None:
    dev = FakeDevice()
    with _rec(tmp_path) as rec:
        inj = Injector(InjectionSink(dev, sleep=lambda _: None), rec.journal)
        out = inj.submit(Action.key("OUT_0A11", 40, modifiers=("MOD_1B2C", "MOD_3D4E")),
                         rec.clocks.stamp(), ActorLayer.REFLEX)
    assert out.delivered and not out.masked and not out.stopped
    assert dev.calls == [
        ("down", "MOD_1B2C"), ("down", "MOD_3D4E"), ("down", "OUT_0A11"),
        ("up", "OUT_0A11"), ("up", "MOD_3D4E"), ("up", "MOD_1B2C"),
    ], "модификаторы обязаны зажиматься первыми и отпускаться последними"


def test_delivered_action_is_journaled_with_device(tmp_path: Path) -> None:
    dev = FakeDevice()
    with _rec(tmp_path) as rec:
        inj = Injector(InjectionSink(dev, sleep=lambda _: None), rec.journal)
        inj.submit(Action.key("OUT_0A11", 120,
                              reversibility=Reversibility(0.9, 0.05, 12)),
                   rec.clocks.stamp(), ActorLayer.REFLEX)

    with Session.open(tmp_path / "s") as s:
        (entry, act), = list(s.actions())
        assert entry.event["code"] == "delivered"
        assert entry.event["device"] == "fake"
        assert entry.event["reason"] is None
        assert not act.masked
        assert act.duration_ms == 120
        assert act.reversibility.n == 12, "оценка обратимости не должна теряться при записи"


def test_mouse_move_reaches_device(tmp_path: Path) -> None:
    dev = FakeDevice()
    with _rec(tmp_path) as rec:
        inj = Injector(InjectionSink(dev, sleep=lambda _: None), rec.journal)
        out = inj.submit(Action.mouse(15, -7, 16), rec.clocks.stamp(), ActorLayer.REFLEX)
    assert out.delivered
    assert dev.calls == [("move", (15, -7))]


def test_nothing_is_delivered_and_recorded(tmp_path: Path) -> None:
    """Осознанное бездействие доходит до устройства как пауза и пишется в журнал."""
    dev = FakeDevice()
    with _rec(tmp_path) as rec:
        inj = Injector(InjectionSink(dev, sleep=lambda _: None), rec.journal)
        out = inj.submit(Action.nothing(50), rec.clocks.stamp(), ActorLayer.REFLEX)
    assert out.delivered
    assert dev.calls == [], "бездействие не должно шевелить устройством"
    with Session.open(tmp_path / "s") as s:
        (_, act), = list(s.actions())
        assert act.kind.value == "nothing"
        assert act.caution == 0.0, "ничего не делать — обратимо по определению"


def test_masked_action_never_touches_device(tmp_path: Path) -> None:
    """Заглушённая попытка не доходит до устройства, но доходит до журнала."""
    dev = FakeDevice()
    mask = InputMask()
    mask.block("OUT_0A11", scope="app")
    with _rec(tmp_path) as rec:
        inj = Injector(InjectionSink(dev, sleep=lambda _: None), rec.journal, mask=mask)
        out = inj.submit(Action.key("OUT_0A11", 60), rec.clocks.stamp(),
                         ActorLayer.REFLEX, scope="window")
    assert out.masked and not out.delivered
    assert dev.calls == [], "маска пропустила приказ до устройства"
    with Session.open(tmp_path / "s") as s:
        (_, act), = list(s.actions())
        assert act.masked and "mask:window" in act.mask_reason


def test_masked_modifier_blocks_whole_action(tmp_path: Path) -> None:
    """Если заглушён модификатор, действие не выполняется целиком.

    Иначе агент получил бы `OUT` без `MOD` — то есть другое действие, чем
    задумал, и журнал бы этого не объяснил.
    """
    dev = FakeDevice()
    mask = InputMask()
    mask.block("MOD_1B2C", scope="window")
    with _rec(tmp_path) as rec:
        inj = Injector(InjectionSink(dev, sleep=lambda _: None), rec.journal, mask=mask)
        out = inj.submit(Action.key("OUT_0A11", 60, modifiers=("MOD_1B2C",)),
                         rec.clocks.stamp(), ActorLayer.REFLEX)
    assert out.masked and dev.calls == []
    assert "MOD_1B2C" in out.reason


def test_stop_before_submit_never_touches_device(tmp_path: Path) -> None:
    dev = FakeDevice()
    with _rec(tmp_path) as rec:
        stop = StopSwitch(rec.journal)
        stop.engage("аварийно", rec.clocks.stamp())
        inj = Injector(InjectionSink(dev, sleep=lambda _: None), rec.journal, stop=stop)
        out = inj.submit(Action.key("OUT_0A11", 500), rec.clocks.stamp(), ActorLayer.REFLEX)
    assert out.stopped and dev.calls == []
    assert out.reason.startswith("stop:")


def test_stop_release_lets_actions_through_again(tmp_path: Path) -> None:
    dev = FakeDevice()
    with _rec(tmp_path) as rec:
        stop = StopSwitch(rec.journal)
        inj = Injector(InjectionSink(dev, sleep=lambda _: None), rec.journal, stop=stop)
        stop.engage("пауза", rec.clocks.stamp())
        assert inj.submit(Action.key("OUT_0A11", 30), rec.clocks.stamp(),
                          ActorLayer.REFLEX).stopped
        stop.release(rec.clocks.stamp())
        assert inj.submit(Action.key("OUT_0A11", 30), rec.clocks.stamp(),
                          ActorLayer.REFLEX).delivered

    with Session.open(tmp_path / "s") as s:
        codes = [e.event.get("code") for e in s.journal]
        assert "stop" in codes and "resume" in codes
        assert [c for c in codes if c in ("stopped", "delivered")] == ["stopped", "delivered"]


def test_watchdog_reset_after_release(tmp_path: Path) -> None:
    """После снятия стопа сторожевой таймер считает заново, а не срабатывает сразу."""
    from harness.core.clocks import Stamp
    from harness.inject.watchdog import Watchdog

    clock = [0.0]
    wd = Watchdog(still_seconds=1.0, threshold=0.002, now=lambda: clock[0])
    still = np.zeros((16, 16), dtype=np.uint8)
    for _ in range(20):
        clock[0] += 0.1
        wd.feed(still, Stamp(1, 1))
    assert wd.tripped is not None

    wd.reset()
    clock[0] += 0.1
    assert wd.feed(still, Stamp(2, 2)) is None, "после reset таймер обязан начать заново"
    assert wd.tripped is None


def test_journal_records_every_attempt_exactly_once(tmp_path: Path) -> None:
    """Сколько попыток сделано, столько записей действий в журнале."""
    dev = FakeDevice()
    mask = InputMask()
    mask.block("OUT_0B22", scope="window")
    with _rec(tmp_path) as rec:
        stop = StopSwitch(rec.journal)
        inj = Injector(InjectionSink(dev, sleep=lambda _: None), rec.journal,
                       mask=mask, stop=stop)
        inj.submit(Action.key("OUT_0A11", 20), rec.clocks.stamp(), ActorLayer.REFLEX)
        inj.submit(Action.key("OUT_0B22", 20), rec.clocks.stamp(), ActorLayer.REFLEX)
        stop.engage("тест", rec.clocks.stamp())
        inj.submit(Action.key("OUT_0A11", 20), rec.clocks.stamp(), ActorLayer.REFLEX)

    with Session.open(tmp_path / "s") as s:
        s.journal.verify()
        codes = [e.event["code"] for e, _ in s.actions()]
    assert codes == ["delivered", "masked", "stopped"]


def test_unknown_output_is_rejected_by_body() -> None:
    from harness.agentside.body import Body

    body = Body(["OUT_0A11"], submit=lambda a: a)
    with pytest.raises(ValueError, match="не принадлежит этому телу"):
        body.press("OUT_9999", 100)
    with pytest.raises(ValueError, match="не принадлежит этому телу"):
        body.press("OUT_0A11", 100, modifiers=("MOD_0000",))
