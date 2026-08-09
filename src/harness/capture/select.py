"""Какой механизм захвата взять на этой машине и **почему именно этот**.

`TASK-08`, часть 1. Прежняя логика выбора была одной строкой в таблице: одна оконная
система — один backend. Из-за этого на Windows брался mss, хотя dxcam стоял рядом
установленный и работающий, — и `doctor` печатал «есть», не сказав, что взял худший
из двух доступных.

Отчёт «есть» без объяснения, почему не взят более подходящий механизм, — дефект
вывода, а не мелочь оформления. Оператор видит зелёную строку и идёт записывать; то,
что игра не попадёт в кадр, выясняется через десятки минут записи.

## Порядок предпочтения и чем он обоснован

На Windows **Desktop Duplication (dxcam) первым**, mss запасным:

- mss идёт через GDI BitBlt: нагрузка на процессор выше, и на 1080p замер дал
  **19.9 кадр/с при заявленных 30**. Критерий М0 требует устойчивых 30+;
- и главное: через BitBlt **полноэкранные DirectX-приложения не захватываются**.
  Вместо игры в кадр попадает рабочий стол или чернота, и происходит это молча.

Поэтому mss на Windows — не равноправная альтернатива, а запасной путь с
предупреждением. Предупреждение печатается всегда, когда он активен.

На X11 и macOS выбор один — mss. Быстрые пути (PipeWire, ScreenCaptureKit) не
написаны, и притворяться, что выбор есть, было бы ложью в отчёте.

## Почему выбор проверяется запуском, а не только наличием пакета

`import dxcam` проходит и там, где `dxcam.create()` падает: нет адаптера DirectX 11,
устарел драйвер, сеанс без рабочего стола. Наличие пакета — не наличие механизма,
поэтому кандидат считается годным, только если **запустился**, и причина отказа
каждого непрошедшего попадает в отчёт.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable

from ..machine import Machine, Session
from .base import BackendUnavailable


@dataclass(frozen=True, slots=True)
class Backend:
    """Кандидат: чем он хорош, чем плох и что нужно, чтобы он заработал."""

    name: str
    module: str                  # пакет, без которого не заработает
    why: str                     # почему он в этом порядке
    caveat: str = ""             # чего он не умеет; печатается, когда он активен

    @property
    def installed(self) -> bool:
        return importlib.util.find_spec(self.module) is not None


DXCAM = Backend(
    "screen_dxcam", "dxcam",
    "Desktop Duplication: видит полноэкранные DirectX-приложения и держит 30+ "
    "кадров при 1080p")
MSS = Backend(
    "screen_mss", "mss",
    "GDI BitBlt через mss: работает везде, где есть рабочий стол",
    caveat="полноэкранные игры и приложения DirectX через этот механизм **не "
           "захватываются** — в кадр попадёт рабочий стол или чернота, и молча. "
           "Замер на 1080p дал 19.9 кадр/с при заявленных 30")
MSS_PLAIN = Backend(
    "screen_mss", "mss",
    "mss через X11: единственный написанный механизм для этой оконной системы")

#: Кандидаты по оконной системе, в порядке предпочтения.
CANDIDATES: dict[Session, tuple[Backend, ...]] = {
    Session.X11: (MSS_PLAIN,),
    Session.WAYLAND: (),          # нужен PipeWire; backend не написан
    Session.WINDOWS: (DXCAM, MSS),
    Session.MACOS: (MSS_PLAIN,),
    Session.NONE: (),
}


@dataclass(slots=True)
class Choice:
    """Что выбрано, что было ещё, и почему не оно."""

    chosen: Backend | None
    considered: tuple[Backend, ...]
    rejected: list[tuple[str, str]] = field(default_factory=list)
    source: Any = None            # запущенный источник, если открывали

    @property
    def alternatives(self) -> list[str]:
        return [b.name for b in self.considered
                if self.chosen is None or b.name != self.chosen.name]

    @property
    def caveat(self) -> str:
        return self.chosen.caveat if self.chosen else ""

    def why_text(self) -> str:
        """Одной строкой: выбор, причина, альтернативы. Для отчёта `doctor`."""
        if self.chosen is None:
            return "механизма нет: " + ("; ".join(f"{n} — {w}" for n, w in self.rejected)
                                        or "кандидатов для этой системы нет")
        rows = [f"{self.chosen.name}: {self.chosen.why}"]
        for name, why in self.rejected:
            rows.append(f"не взят {name}: {why}")
        others = [b.name for b in self.considered if b.name != self.chosen.name
                  and not any(n == b.name for n, _ in self.rejected)]
        if others:
            rows.append("в запасе: " + ", ".join(others))
        return "; ".join(rows)

    def as_dict(self) -> dict[str, Any]:
        return {"chosen": self.chosen.name if self.chosen else None,
                "considered": [b.name for b in self.considered],
                "rejected": [{"name": n, "why": w} for n, w in self.rejected],
                "caveat": self.caveat, "why": self.why_text()}


def _factory(name: str) -> Callable[..., Any]:
    from .screen import MacScreenCapture, ScreenCapture, WindowsScreenCapture

    return {"screen_dxcam": WindowsScreenCapture,
            "screen_mss": ScreenCapture,
            "screen_sck": MacScreenCapture}[name]


def plan(machine: Machine) -> Choice:
    """Что бы выбрали, **не запуская**. Для отчёта на машине без экрана.

    Отдельно от `open_screen`, потому что «что бы взяли» и «что взялось» — разные
    утверждения, и второе требует экрана. На машине без экрана честно доступно
    только первое, и подменять его вторым нельзя.
    """
    considered = CANDIDATES[machine.session]
    rejected: list[tuple[str, str]] = []
    for b in considered:
        if not b.installed:
            rejected.append((b.name, f"нет пакета {b.module}"))
            continue
        return Choice(chosen=b, considered=considered, rejected=rejected)
    return Choice(chosen=None, considered=considered, rejected=rejected)


def open_screen(machine: Machine, *, gray: bool = True,
                region: tuple[int, int, int, int] | None = None) -> Choice:
    """Открыть первый кандидат, который **запустился**. Остальные — с причиной.

    Возвращает `Choice` с уже запущенным источником в `source`. Закрывать его —
    вызывающему: здесь нельзя решить, когда он больше не нужен.
    """
    considered = CANDIDATES[machine.session]
    rejected: list[tuple[str, str]] = []
    if not considered:
        return Choice(chosen=None, considered=considered,
                      rejected=[("нет кандидатов",
                                 f"для оконной системы {machine.session} механизм "
                                 "захвата не написан")])
    for b in considered:
        if not b.installed:
            rejected.append((b.name, f"нет пакета {b.module}"))
            continue
        source = _factory(b.name)(region, gray=gray)
        try:
            source.start()
        except BackendUnavailable as e:
            rejected.append((b.name, str(e)))
            continue
        return Choice(chosen=b, considered=considered, rejected=rejected,
                      source=source)
    return Choice(chosen=None, considered=considered, rejected=rejected)
