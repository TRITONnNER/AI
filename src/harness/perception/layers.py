"""Слои восприятия: система отсчёта × дальность, каждый со своей частотой.

Из архитектуры и из замысла: «нужно делать послойно — прямо перед лицом (условно
интерфейс), передний план, средний, задний». Слои различаются не «важностью», а двумя
вещами сразу:

| Слой | Система отсчёта | Что от него нужно | Частота |
|---|---|---|---|
| экранный | прибит к экрану | что здесь не мир, а интерфейс | 30 Гц |
| ближний | относительно меня | успею ли, что наплывает | 20 Гц |
| средний | граф мест | где я и как отсюда куда-то | 2 Гц |
| дальний | ориентиры и свет | что вообще происходит вокруг | 0.2 Гц |

## Почему это не «просто разные функции»

Три причины, и все три проверяются.

**Частоты разные, и это экономия, а не аккуратность.** «Горы никуда не денутся»: считать
дальний слой двадцать раз в секунду — выбросить работу. Экономия измерена, а не заявлена:
`savings()` показывает долю обновлений, которых удалось не делать.

**Нижние слои перебивают верхние.** Это субсумпция из архитектуры, и здесь она означает
конкретное: пока средний слой пересобирает граф мест, ближний обязан успеть посчитать
тау. Поэтому слои живут контурами в `behaviour/contours.py`, где уровень задаёт порядок
обслуживания, а работа нарезается на срезы.

**У каждого слоя свой ответ, и он доступен в любой момент.** Не «слой посчитал и отдал»,
а «у слоя есть последний известный ответ» — потому что мир не ставится на паузу
(инвариант 3), и спросить можно между обновлениями.

## Чего здесь нет

Ни одного нового способа что-то увидеть. Каждый слой собран из уже измеренных частей:
экранный — из арбитра трёх признаков (`vision/selfworld.py`), ближний — из тау
(`vision/layers.py`, медианная ошибка 3–4 %), средний — из графа мест, дальний — из
уровня и контраста вида плюс грубого отпечатка. Это сборка, а не изобретение: слой,
который считал бы что-то ещё не измеренное, врал бы своим существованием.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.profile import Profile
from ..core.symbols import assert_known_words

SCREEN = "screen"
NEAR = "near"
MID = "mid"
FAR = "far"
# Порядок — от близкого к далёкому, он же порядок субсумпции: чем ближе, тем срочнее.
ORDER = (SCREEN, NEAR, MID, FAR)

# Слова, которые слои имеют право сказать. Набор закрыт: это собственные понятия
# восприятия — имена слоёв и имена признаков, которыми решён экранный слой, — а не
# надписи с экрана. Всё, чего здесь нет, падает как утечка надписи (инвариант 5).
VOCABULARY = frozenset(ORDER) | frozenset({
    "merged", "parallax", "stillness", "coupling", "no_signal",
})


@dataclass(slots=True)
class LayerAnswer:
    """Последнее, что слой знает. Отдельный тип, чтобы «когда» нельзя было потерять.

    `at_frame` обязателен: ответ дальнего слоя может быть пятисекундной давности, и
    решение по нему без знания давности — решение по устаревшему. Ноль означал бы
    «только что», поэтому `None` для «ещё не считал» — отдельное значение.
    """

    name: str
    at_frame: int
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"layer": self.name, "at_frame": self.at_frame, **self.payload}


class PerceptionLayer:
    """Один слой: своя частота, свой ответ, своя цена.

    Наследники обязаны реализовать `look`. Базовый класс не делает вид, что умеет
    смотреть: `look` здесь поднимает исключение, а не возвращает пустой ответ.
    """

    name = "base"

    def __init__(self, profile: Profile, hz: float, *, level: int) -> None:
        self.profile = profile
        self.hz = float(hz)
        self.level = int(level)
        self.updates = 0
        self.frames_seen = 0
        self._answer: LayerAnswer | None = None

    # --- то, что реализует наследник ----------------------------------------

    def look(self, frame: np.ndarray, at_frame: int) -> dict[str, Any]:
        raise NotImplementedError(
            f"слой {self.name} не умеет смотреть: `look` не реализован. Пустой ответ "
            "вместо этого был бы хуже — по нему не видно, что слоя нет")

    # --- общее --------------------------------------------------------------

    def update(self, frame: np.ndarray, at_frame: int) -> LayerAnswer:
        payload = self.look(frame, at_frame)
        self.updates += 1
        self._answer = LayerAnswer(self.name, at_frame, payload)
        return self._answer

    def answer(self) -> LayerAnswer | None:
        """Последний известный ответ. `None` — ещё не смотрел, и это не то же, что ноль."""
        return self._answer

    def age(self, at_frame: int) -> int | None:
        """Сколько кадров назад ответ получен. `None`, если ответа ещё нет."""
        return None if self._answer is None else at_frame - self._answer.at_frame

    def stats(self) -> dict[str, Any]:
        return {"layer": self.name, "hz": self.hz, "level": self.level,
                "updates": self.updates}


class ScreenLayer(PerceptionLayer):
    """Экранный слой: что прибито к экрану, а что мир.

    Собран из арбитра трёх признаков. Арбитр — накопитель: он копит свидетельства по
    кадрам и решает один раз, поэтому здесь каждый кадр только скармливается, а ответ
    берётся из накопленного. Так и должно быть: «это интерфейс» — вывод по истории
    движения, а не по одному кадру.
    """

    name = SCREEN

    def __init__(self, profile: Profile, hz: float, *, level: int = 0) -> None:
        super().__init__(profile, hz, level=level)
        from ..vision.selfworld import LayerArbiter

        self._arbiter = LayerArbiter(profile)

    def look(self, frame: np.ndarray, at_frame: int) -> dict[str, Any]:
        self._arbiter.feed(frame)
        verdict = self._arbiter.result()
        from ..vision.selfworld import SCREEN as SCREEN_LABEL

        screen = float((verdict.labels == SCREEN_LABEL).mean())
        return {"screen_fraction": round(screen, 4),
                "signal": verdict.signal,
                "disagreement_fraction": round(verdict.disagreement_fraction, 4)}

    @property
    def arbiter(self) -> Any:
        return self._arbiter


class NearLayer(PerceptionLayer):
    """Ближний слой: успею ли. Тау, а не расстояние.

    Единственная величина здесь — время до контакта в кадрах, посчитанное по скорости
    роста видимого размера. Расстояния нет и не будет: «оценка многих вещей происходит
    временем».
    """

    name = NEAR

    def __init__(self, profile: Profile, hz: float, *, level: int = 1) -> None:
        super().__init__(profile, hz, level=level)
        from ..vision.layers import TimeToContact

        self._tau = TimeToContact(profile)

    def look(self, frame: np.ndarray, at_frame: int) -> dict[str, Any]:
        looming = self._tau.feed(frame)
        if looming is None:
            # Ничего не растёт или данных мало. Это не «контакта не будет».
            return {"size": None, "growth": None, "tau_steps": None, "n": 0}
        return {"size": round(looming.size, 2),
                "growth": round(looming.growth, 4),
                "tau_steps": (None if looming.tau_steps is None
                              else round(looming.tau_steps, 2)),
                "n": looming.n}


class MidLayer(PerceptionLayer):
    """Средний слой: граф мест и время между ними.

    Обновляется редко не из экономии, а по существу: место меняется медленнее, чем
    кадр. Двадцать обновлений в секунду дали бы двадцать записей об одном и том же.
    """

    name = MID

    def __init__(self, profile: Profile, hz: float, *, level: int = 2,
                 graph: Any = None) -> None:
        super().__init__(profile, hz, level=level)
        from ..model.places import PlaceGraph

        self.graph = graph if graph is not None else PlaceGraph.from_profile(profile)
        self.seconds_per_frame = 1.0 / max(1e-9, float(profile.parameters["capture_fps"]))

    def look(self, frame: np.ndarray, at_frame: int) -> dict[str, Any]:
        place = self.graph.see(frame, at_frame,
                              seconds_per_seq=self.seconds_per_frame, mode="unknown")
        stats = self.graph.stats()
        return {"place": place, "places": stats["places"],
                "leaving": stats["leaving"], "lost": stats["lost"]}


class FarLayer(PerceptionLayer):
    """Дальний слой: ориентиры, свет, погода. Самый редкий и самый грубый.

    Считает две вещи: уровень и контраст вида (это и есть «свет») и грубый отпечаток по
    уменьшенной сетке (это и есть «ориентиры» — силуэт того, что вокруг, без деталей).
    Мельче здесь не нужно: деталь на дальнем плане меняется медленнее, чем мы успеваем
    её забыть, а грубость даёт устойчивость к тому, что вблизи всё время шевелится.
    """

    name = FAR

    def __init__(self, profile: Profile, hz: float, *, level: int = 3) -> None:
        super().__init__(profile, hz, level=level)
        grid = int(profile.structural["place_grid"])
        coarsen = max(1, int(profile.parameters["periphery_coarsening"]))
        self.grid = max(2, grid // coarsen)
        self.levels = int(profile.structural["place_levels"])
        # Размытие берётся оттуда же, откуда сетка: ориентир дальнего слоя и отпечаток
        # места считаются одной функцией, и разойтись они не должны — иначе «тот же вид»
        # у графа мест и у ориентиров означало бы разное.
        self.blur_px = int(profile.structural["place_blur_px"])

    def look(self, frame: np.ndarray, at_frame: int) -> dict[str, Any]:
        from ..model.places import view

        v = view(frame, grid=self.grid, levels=self.levels, blur_px=self.blur_px)
        return {"level": round(v.level, 2), "contrast": round(v.contrast, 2),
                "landmark": list(v.cells)}


@dataclass(slots=True)
class PerceptionStack:
    """Четыре слоя вместе плюс часы, кому когда пора.

    Кадр даётся стеку, а стек решает, кого обновлять. Решает по частотам из профиля, а
    не по важности: важность здесь не при чём — редкий слой не менее важен, он просто
    меняется медленнее.
    """

    layers: dict[str, PerceptionLayer]
    clock: Any
    frames: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    cascade: Any = None
    blocked: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_profile(cls, profile: Profile, *, graph: Any = None,
                     cascade: Any = None) -> "PerceptionStack":
        from ..vision.layers import LayerClock
        from .cascade import Cascade

        p = profile.parameters
        layers: dict[str, PerceptionLayer] = {
            SCREEN: ScreenLayer(profile, float(p["layer_screen_hz"])),
            NEAR: NearLayer(profile, float(p["layer_near_hz"])),
            MID: MidLayer(profile, float(p["layer_mid_hz"]), graph=graph),
            FAR: FarLayer(profile, float(p["layer_far_hz"])),
        }
        return cls(layers=layers, clock=LayerClock.from_profile(profile),
                   cascade=cascade if cascade is not None else Cascade.from_profile(profile))

    def feed(self, frame: Any, *, error: float | None = None,
             t_self: int = 0) -> list[LayerAnswer]:
        """Дать кадр. Обновятся только те слои, которым пора **и кому позволено**.

        Два разных «не обновился», и они считаются раздельно:

        - `skipped` — слою не пора по его частоте. Так было всегда;
        - `blocked` — каскад запер ступень: кадр не менялся или совпал с
          предсказанием. Это новое, и это ответ на то, что отметки «без изменений»
          доходили до журнала и не доходили до восприятия.

        `frame` может быть отметкой `UNCHANGED` от источника — тогда смотреть
        нечего ни одному слою, и это не потеря кадра, а ответ.
        """
        from ..capture.base import UNCHANGED
        from .cascade import CHANGE, LAYER_STAGE

        self.frames += 1
        verdict = None
        if self.cascade is not None:
            verdict = self.cascade.admit(frame, error=error, t_self=t_self)

        # Кадра нет вовсе — обновлять нечего ни при каком вердикте. Часы при этом не
        # идут: неподвижный экран не должен продвигать очередь слоёв, иначе после
        # минуты неподвижности всем «пора» разом.
        # `frames_seen` считает **обороты, предложенные слою**, а не кадры, на которые
        # он посмотрел: иначе две ветки запирания считались бы по-разному, и доля
        # запертого зависела бы от того, кто именно запер.
        if frame is UNCHANGED:
            for name in ORDER:
                self.layers[name].frames_seen += 1
                self.blocked[name] = self.blocked.get(name, 0) + 1
            return []

        top = None if verdict is None else verdict.top
        if top is not None and top <= CHANGE:
            for name in ORDER:
                self.layers[name].frames_seen += 1
                self.blocked[name] = self.blocked.get(name, 0) + 1
            return []

        due = set(self.clock.tick())
        out: list[LayerAnswer] = []
        for name in ORDER:
            layer = self.layers[name]
            layer.frames_seen += 1
            if top is not None and LAYER_STAGE[name] > top:
                self.blocked[name] = self.blocked.get(name, 0) + 1
            elif name in due:
                with (self.cascade.timed(LAYER_STAGE[name]) if self.cascade is not None
                      else nullcontext()):
                    out.append(layer.update(frame, self.frames))
            else:
                self.skipped[name] = self.skipped.get(name, 0) + 1
        if self.cascade is not None:
            self.cascade.settle()
        return out

    def answer(self, name: str) -> LayerAnswer | None:
        return self.layers[name].answer()

    def contours(self, frames: Callable[[], np.ndarray | None]) -> dict[str, Any]:
        """Слои как контуры для планировщика контуров (`behaviour/contours.py`).

        `frames` отдаёт текущий кадр или `None`, если кадра нет. Возвращается словарь
        «имя слоя → Start», который скармливается `Scheduler.add` с уровнем слоя. Так
        нижние слои получают право перебивать верхние, а работа нарезается срезами.
        """
        from ..behaviour.contours import fixed

        made: dict[str, Any] = {}
        for name in ORDER:
            layer = self.layers[name]

            def start(layer: PerceptionLayer = layer) -> Any:
                frame = frames()
                if frame is None:
                    return fixed(None)()
                answer = layer.update(frame, self.frames)
                return fixed(answer)()

            made[name] = start
        return made

    def as_perception(self) -> dict[str, Any]:
        """Сводка для журнала. Проверяется на читаемый текст, как всё, что уйдёт агенту."""
        payload = {"frames": self.frames,
                   "layers": {name: (None if (a := self.layers[name].answer()) is None
                                     else a.as_dict())
                              for name in ORDER}}
        assert_known_words(
            [v for name in ORDER
             if (a := self.layers[name].answer()) is not None
             for v in a.payload.values() if isinstance(v, str)],
            vocabulary=VOCABULARY, path="perception.layers")
        return payload

    def stats(self) -> dict[str, Any]:
        return {"frames": self.frames,
                "layers": [self.layers[n].stats() for n in ORDER],
                "skipped": dict(sorted(self.skipped.items())),
                "blocked": dict(sorted(self.blocked.items())),
                "clock": self.clock.summary(),
                "cascade": None if self.cascade is None else self.cascade.stats()}
