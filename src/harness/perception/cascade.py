"""Каскад восприятия: четыре ступени, каждая дороже предыдущей и реже её.

Ключевое отношение, ради которого каскад существует: **ступень 3 работает в
100–300 раз реже ступени 0**. Не держится — архитектура не работает, потому что
бюджет тогда не сходится ни при каком размере модели.

| Ступень | Что делает | Частота | Кто её исполняет |
|---|---|---|---|
| 0. Изменение | разность на уменьшенной копии | 60+ Гц | `ChangeDetector`, здесь |
| 1. Поток | поток, разделение движения, глубина | 20–30 Гц | `ScreenLayer`, `NearLayer` |
| 2. Отпечаток | отпечаток вида и ориентиры | 5–10 Гц | `MidLayer`, `FarLayer` |
| 3. Большая модель | «что я вижу» по требованию | 0.2–0.5 Гц | `PerceptionFirewall` |

Ступени 1–3 здесь **не реализуются заново**. Каскад — это гейт и учёт: он решает,
кому на этом кадре вообще позволено работать, и запоминает, сколько это стоило.
Слой, который считал бы что-то ещё не измеренное, врал бы своим существованием.

## Отметка «без изменений» — данные, а не пропуск

Источник (Desktop Duplication) сам говорит `UNCHANGED`, когда экран не менялся, и
до сих пор это доходило только до журнала: восприятие такую отметку не видело
вообще. Между тем в опорной записи неподвижности **318 изменившихся кадров из 1574
оборотов** — четыре кадра из пяти не несут ничего, и на каждом из них работали все
слои.

Поэтому здесь два входа в ступень 0, а не один:

- источник сказал `UNCHANGED` — ответ бесплатный, разность не считается вовсе;
- источник дал кадр — считается разность на уменьшенной копии.

Оба дают один и тот же исход `changed=False`, и выше нулевой ступени тогда не
запускается ничто. Различие между ними сохраняется в счётчиках (`free` против
`measured`): «нам сказали» и «мы проверили» — разные основания, и смешивать их
значило бы потерять, какой источник сколько экономит.

## Ошибка предсказания управляет расходом

До сих пор ошибка предсказания управляла обучением и не управляла ценой взгляда.
Здесь она управляет: **совпало с предсказанием — ступени выше первой не
запускаются**. Порог отдельный (`cascade_predicted_error`), потому что вопрос
другой: «удивил ли мир настолько, чтобы смотреть внимательнее» — не то же самое,
что «удивил ли настолько, чтобы тратить квоту большой модели».

Порог ступени 3 не заводится заново: он уже есть — `model_min_novelty` и
`model_min_gap_cycles`, и решает по ним `AskGate` из `describers.py`. Второй порог
о том же был бы двумя валютами вместо одной.

## Чем распределяется бюджет между ступенями

Величина — **сужение sigma на миллисекунду вычислений**, и считается она
**предельной**, а не средней: сужение, случившееся на кадре, приписывается той
ступени, до которой каскад на этом кадре дошёл. Это ровно тот вопрос, на который
бюджет и отвечает: «если позволить каскаду пройти на ступень глубже, сколько
сужения sigma это купит за миллисекунду». Средняя приписала бы глубокой ступени
работу мелких, которые всё равно отработали бы.

Sigma каскад сам не считает: её отдаёт `sigma_probe` — функция, которую передаёт
владелец модели. Без неё величина равна `None`, а не нулю: «не измеряли» и
«не сузилась» — разные утверждения, и второе здесь было бы неправдой.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..capture.base import UNCHANGED, Unchanged
from ..core.profile import Profile
from .imagecodec import downscale

#: Номера ступеней. Именами пользуется отчёт, номерами — сравнение «выше/ниже».
CHANGE = 0
FLOW = 1
FINGERPRINT = 2
MODEL = 3
STAGES = (CHANGE, FLOW, FINGERPRINT, MODEL)

#: Как ступень называется в отчёте. Слова свои, с экрана сюда ничего не попадает.
STAGE_NAMES = {CHANGE: "change", FLOW: "flow", FINGERPRINT: "fingerprint",
               MODEL: "model"}

#: Какой слой стека к какой ступени относится. Таблица объявлена явно, потому что
#: уровни слоёв (`PerceptionLayer.level`) — это порядок субсумпции, а не ступени
#: каскада, и совпадение чисел там случайно. Дальний слой считает грубый отпечаток
#: и ориентиры, то есть работу ступени 2; большой модели в стеке нет вообще.
LAYER_STAGE = {"screen": FLOW, "near": FLOW, "mid": FINGERPRINT, "far": FINGERPRINT}


@dataclass(frozen=True, slots=True)
class Verdict:
    """Что каскаду позволено на этом кадре.

    `top` — самая глубокая разрешённая ступень. `changed` и `surprised` держатся
    отдельно от неё, потому что «кадр не менялся» и «кадр изменился, но совпал с
    предсказанием» упираются в разные ступени и лечатся разным.
    """

    top: int
    changed: bool
    surprised: bool | None
    change_fraction: float | None
    measured: bool

    def allows(self, stage: int) -> bool:
        return stage <= self.top

    def as_dict(self) -> dict[str, Any]:
        return {"top": self.top, "top_name": STAGE_NAMES[self.top],
                "changed": self.changed, "surprised": self.surprised,
                "change_fraction": (None if self.change_fraction is None
                                    else round(self.change_fraction, 6)),
                "measured": self.measured}


@dataclass(slots=True)
class StageCost:
    """Счёт по одной ступени: сколько раз звали, сколько это стоило.

    `ms_per_call` — то, чего в таблице ступеней не хватало: стоимость нулевой
    измерена (0.5 мс), у остальных стояло «измерить».
    """

    stage: int
    calls: int = 0
    total_ms: float = 0.0
    blocked: int = 0
    sigma_narrowed: float = 0.0
    sigma_ms: float = 0.0
    sigma_frames: int = 0

    @property
    def ms_per_call(self) -> float | None:
        """`None`, а не ноль: ступень, которую ни разу не звали, ничего не стоила."""
        return None if not self.calls else self.total_ms / self.calls

    @property
    def sigma_per_ms(self) -> float | None:
        """Сужение sigma на миллисекунду. `None`, если sigma не измерялась."""
        if not self.sigma_frames or self.sigma_ms <= 0.0:
            return None
        return self.sigma_narrowed / self.sigma_ms

    def as_dict(self) -> dict[str, Any]:
        ms = self.ms_per_call
        per = self.sigma_per_ms
        return {"stage": self.stage, "name": STAGE_NAMES[self.stage],
                "calls": self.calls, "blocked": self.blocked,
                "total_ms": round(self.total_ms, 3),
                "ms_per_call": None if ms is None else round(ms, 4),
                "sigma_frames": self.sigma_frames,
                "sigma_per_ms": None if per is None else round(per, 8)}


class ChangeDetector:
    """Ступень 0: изменился ли кадр. Разность на уменьшенной копии.

    Уменьшение обязательно, и не ради скорости самой по себе: полная разность на
    1080p стоит столько же, сколько ступень 1, и гейт перестал бы окупаться. При
    `max_side` 128 сравнивается около шестнадцати тысяч ячеек вместо двух миллионов.

    Порог двойной, и это не перестраховка. `pixel_delta` отсекает шум кодека и
    дрожание одного пикселя; `threshold` отвечает на другой вопрос — какая доля
    кадра должна пошевелиться, чтобы это считалось изменением. Мигающий курсор —
    ровно тот случай, когда первый порог пройден, а второй нет.
    """

    def __init__(self, *, max_side: int = 128, pixel_delta: int = 8,
                 threshold: float = 0.002) -> None:
        if max_side <= 0:
            raise ValueError("max_side должен быть положительным")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("порог доли изменившихся пикселей вне [0, 1]")
        self.max_side = int(max_side)
        self.pixel_delta = int(pixel_delta)
        self.threshold = float(threshold)
        self._previous: np.ndarray | None = None

    @classmethod
    def from_profile(cls, profile: Profile) -> "ChangeDetector":
        p = profile.parameters
        return cls(max_side=int(p["cascade_change_max_side_px"]),
                   pixel_delta=int(p["cascade_change_pixel_delta"]),
                   threshold=float(p["cascade_change_threshold"]))

    def small(self, frame: np.ndarray) -> np.ndarray:
        """Уменьшенная копия в int16: разность считается со знаком, а не по модулю uint8."""
        return downscale(frame, self.max_side).astype(np.int16)

    def look(self, frame: np.ndarray) -> tuple[bool, float]:
        """Изменился ли кадр и какая доля ячеек пошевелилась.

        Первый кадр объявляется изменившимся: сравнивать не с чем, и «не менялся»
        было бы утверждением о том, чего мы не видели.
        """
        small = self.small(frame)
        previous, self._previous = self._previous, small
        if previous is None or previous.shape != small.shape:
            # Смена формы кадра — тоже изменение, и притом самое крупное: сравнивать
            # разноразмерные копии нечем, а объявить «то же самое» было бы ложью.
            return True, 1.0
        moved = np.abs(small - previous) > self.pixel_delta
        fraction = float(moved.mean())
        return fraction > self.threshold, fraction

    def forget(self) -> None:
        """Забыть опорный кадр: следующий будет считаться изменившимся.

        Нужно после разрыва в записи. Сравнивать кадр с тем, что было до пропуска,
        значило бы измерять длину пропуска, а не изменение экрана.
        """
        self._previous = None


class LogBuffer:
    """Логарифмический буфер: каждый кадр за секунду, каждый десятый за десять,
    каждый сотый за сто.

    Зачем не обычное окно. Тау, параллакс, сопоставление последовательностей и поиск
    скриптов смотрят на разные горизонты: тау — на доли секунды, скрипт — на минуты.
    Ровное окно на минуту при 30 к/с — это 1800 кадров, то есть память и перебор
    растут линейно по горизонту. Здесь горизонт растёт, а память нет: каждая полоса
    хранит своё число кадров, и полос столько, сколько десятичных порядков нужно.

    Полоса `i` берёт каждый `decimation**i`-й кадр. Ёмкость у всех одна, поэтому
    полоса `i` покрывает `capacity * decimation**i` кадров: три полосы по 30 кадров
    при прореживании 10 — это секунда, десять секунд и сто секунд за 90 кадров
    вместо трёх тысяч.
    """

    def __init__(self, *, capacity: int = 30, tiers: int = 3,
                 decimation: int = 10) -> None:
        if capacity <= 0 or tiers <= 0 or decimation <= 1:
            raise ValueError("ёмкость и число полос положительны, прореживание больше 1")
        self.capacity = int(capacity)
        self.tiers = int(tiers)
        self.decimation = int(decimation)
        self._tiers: list[list[tuple[int, Any]]] = [[] for _ in range(self.tiers)]
        self.seen = 0

    @classmethod
    def from_profile(cls, profile: Profile) -> "LogBuffer":
        p = profile.parameters
        return cls(capacity=int(p["cascade_buffer_capacity"]),
                   tiers=int(p["cascade_buffer_tiers"]),
                   decimation=int(p["cascade_buffer_decimation"]))

    def put(self, at_frame: int, item: Any) -> None:
        """Положить кадр. В какие полосы он попадёт — решает его номер."""
        self.seen += 1
        for i in range(self.tiers):
            every = self.decimation ** i
            if at_frame % every:
                continue
            tier = self._tiers[i]
            tier.append((at_frame, item))
            if len(tier) > self.capacity:
                del tier[0]

    def tier(self, i: int) -> list[tuple[int, Any]]:
        """Полоса целиком, от старого к новому."""
        return list(self._tiers[i])

    def span(self, i: int) -> int:
        """Сколько кадров покрывает полоса: от первого до последнего, а не сколько хранит."""
        tier = self._tiers[i]
        return 0 if len(tier) < 2 else tier[-1][0] - tier[0][0]

    def horizon(self) -> int:
        """Самый дальний кадр, до которого буфер дотягивается."""
        return max((self.span(i) for i in range(self.tiers)), default=0)

    def at_least(self, frames_back: int) -> tuple[int, Any] | None:
        """Самый свежий кадр, отстоящий не менее чем на `frames_back`.

        Именно это спрашивают параллакс и сопоставление последовательностей: не
        «кадр номер такой-то», а «что было достаточно давно, чтобы picture успела
        измениться». Возвращается `None`, если буфер так далеко не достаёт, — и это
        честный ответ, а не ближайший имеющийся.
        """
        newest = max((t[-1][0] for t in self._tiers if t), default=None)
        if newest is None:
            return None
        want = newest - int(frames_back)
        best: tuple[int, Any] | None = None
        for tier in self._tiers:
            for at, item in tier:
                if at <= want and (best is None or at > best[0]):
                    best = (at, item)
        return best

    def stats(self) -> dict[str, Any]:
        held = sum(len(t) for t in self._tiers)
        return {"seen": self.seen, "held": held,
                "horizon_frames": self.horizon(),
                "tiers": [{"tier": i, "every": self.decimation ** i,
                           "held": len(self._tiers[i]), "span_frames": self.span(i)}
                          for i in range(self.tiers)]}


class Cascade:
    """Гейт и учёт над четырьмя ступенями.

    Каскад ничего не смотрит сам, кроме нулевой ступени. Он отвечает на два вопроса:
    кому позволено работать на этом кадре (`admit`) и сколько это стоило (`spend`).
    """

    def __init__(self, *, detector: ChangeDetector, buffer: LogBuffer,
                 predicted_error: float = 0.15, enabled: bool = True,
                 ask_gate: Any = None,
                 sigma_probe: Callable[[], float] | None = None) -> None:
        self.detector = detector
        self.buffer = buffer
        self.predicted_error = float(predicted_error)
        self.enabled = bool(enabled)
        self.ask_gate = ask_gate
        self.sigma_probe = sigma_probe
        self.costs = {s: StageCost(s) for s in STAGES}
        self.frames = 0
        self.free_unchanged = 0        # источник сам сказал «без изменений»
        self.measured_unchanged = 0    # разность показала, что кадр тот же
        self.quiet = 0                 # изменился, но совпал с предсказанием
        self.reached = {s: 0 for s in STAGES}
        self._sigma_before: float | None = None
        self._last: Verdict | None = None
        self._frame_ms = 0.0           # потрачено на текущем кадре, всеми ступенями

    @classmethod
    def from_profile(cls, profile: Profile, *, ask_gate: Any = None,
                     sigma_probe: Callable[[], float] | None = None) -> "Cascade":
        p = profile.parameters
        return cls(detector=ChangeDetector.from_profile(profile),
                   buffer=LogBuffer.from_profile(profile),
                   predicted_error=float(p["cascade_predicted_error"]),
                   enabled=bool(profile.structural["cascade_enabled"]),
                   ask_gate=ask_gate, sigma_probe=sigma_probe)

    # --- решение -------------------------------------------------------------

    def admit(self, frame: np.ndarray | Unchanged, *, error: float | None = None,
              t_self: int = 0) -> Verdict:
        """Что позволено на этом кадре. Стоимость самой ступени 0 сюда же и пишется.

        `error` — ошибка предсказания. `None` означает «ещё не считали», и тогда
        ступени выше первой не запираются: запереть их на нехватке величины значило
        бы объявить мир предсказанным, ни разу его не предсказав.
        """
        self.frames += 1
        started = time.perf_counter()

        if frame is UNCHANGED:
            # Ответ источника. Разность не считается: платить за проверку того, что
            # нам уже сказали, — ровно та работа, ради отказа от которой гейт и стоит.
            self.free_unchanged += 1
            changed, fraction, measured = False, None, False
        else:
            changed, fraction = self.detector.look(frame)  # type: ignore[arg-type]
            measured = True
            if not changed:
                self.measured_unchanged += 1

        cost = self.costs[CHANGE]
        cost.calls += 1
        stage0_ms = (time.perf_counter() - started) * 1000.0
        cost.total_ms += stage0_ms
        self._frame_ms = stage0_ms

        if not self.enabled:
            # Выключенный каскад пропускает всё: это контрольный прогон, а не режим
            # «как было». Разность всё равно считается, иначе сравнить не с чем.
            verdict = Verdict(MODEL, changed, None, fraction, measured)
            return self._finish(verdict, frame, t_self)

        if not changed:
            verdict = Verdict(CHANGE, False, None, fraction, measured)
            return self._finish(verdict, frame, t_self)

        surprised = None if error is None else error >= self.predicted_error
        if surprised is False:
            self.quiet += 1
            verdict = Verdict(FLOW, True, False, fraction, measured)
            return self._finish(verdict, frame, t_self)

        top = FINGERPRINT
        if self.ask_gate is None or self.ask_gate.should_ask(error, t_self):
            top = MODEL
        verdict = Verdict(top, True, surprised, fraction, measured)
        return self._finish(verdict, frame, t_self)

    def _finish(self, verdict: Verdict, frame: np.ndarray | Unchanged,
                t_self: int) -> Verdict:
        self.reached[verdict.top] += 1
        for stage in STAGES:
            if stage > verdict.top:
                self.costs[stage].blocked += 1
        if frame is not UNCHANGED:
            self.buffer.put(self.frames, frame)
        self._sigma_before = None if self.sigma_probe is None else float(self.sigma_probe())
        self._last = verdict
        return verdict

    # --- учёт ----------------------------------------------------------------

    def spend(self, stage: int, ms: float) -> None:
        """Записать, что ступень отработала и сколько это стоило."""
        if stage not in self.costs:
            raise ValueError(f"нет такой ступени: {stage}")
        cost = self.costs[stage]
        cost.calls += 1
        cost.total_ms += float(ms)
        self._frame_ms += float(ms)

    def timed(self, stage: int) -> "_Timed":
        """`with cascade.timed(FLOW): ...` — то же, что `spend`, но без ручного таймера."""
        return _Timed(self, stage)

    def settle(self) -> None:
        """Закрыть кадр: приписать сужение sigma той ступени, до которой дошли.

        Зовётся после того, как разрешённые ступени отработали. Без вызова сужение
        не приписывается никому — и это правильнее, чем приписать наугад.
        """
        if self._last is None or self.sigma_probe is None or self._sigma_before is None:
            return
        after = float(self.sigma_probe())
        narrowed = self._sigma_before - after
        cost = self.costs[self._last.top]
        # Миллисекунды берутся те, что потрачены **на этом кадре**, а не средние по
        # ступени: сужение купила эта работа. Средняя приписала бы кадру чужой расход
        # и тем сильнее врала бы, чем неравномернее ступень работает.
        cost.sigma_narrowed += narrowed
        cost.sigma_frames += 1
        cost.sigma_ms += self._frame_ms
        self._sigma_before = None

    # --- отчёт ---------------------------------------------------------------

    @property
    def ratio(self) -> float | None:
        """Во сколько раз каскад доходит до ступени 3 реже, чем до нулевой.

        Это и есть проверяемое отношение архитектуры: ожидание 100–300.

        Считается по **решению гейта** (`reached`), а не по фактическим вызовам
        (`costs[MODEL].calls`), и это разные величины: гейт пропустил кадр наверх, а
        отработала ли там модель — за пределами его видимости. Вызовы отвечают на
        другой вопрос, «сколько это стоило», и живут в `StageCost`.

        Нулевая ступень идёт на каждом кадре по построению, поэтому знаменателем
        отношения стоит число кадров. `None` — до модели не дошли ни разу, и это не
        отношение «бесконечность», а отсутствие величины.
        """
        deep = self.reached[MODEL]
        return None if not deep else self.frames / deep

    def share_reaching(self, stage: int) -> float:
        """Доля кадров, на которых каскад дошёл до ступени не ниже указанной."""
        if not self.frames:
            return 0.0
        return sum(self.reached[s] for s in STAGES if s >= stage) / self.frames

    def stats(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "enabled": self.enabled,
            "unchanged": {"free": self.free_unchanged,
                          "measured": self.measured_unchanged,
                          "share": round((self.free_unchanged + self.measured_unchanged)
                                         / self.frames, 4) if self.frames else 0.0},
            "quiet": self.quiet,
            "reached": {STAGE_NAMES[s]: self.reached[s] for s in STAGES},
            "share_reaching": {STAGE_NAMES[s]: round(self.share_reaching(s), 6)
                               for s in STAGES},
            "ratio_change_to_model": (None if self.ratio is None
                                      else round(self.ratio, 1)),
            "stages": [self.costs[s].as_dict() for s in STAGES],
            "buffer": self.buffer.stats(),
        }


@dataclass(slots=True)
class _Timed:
    cascade: Cascade
    stage: int
    started: float = 0.0

    def __enter__(self) -> "_Timed":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.cascade.spend(self.stage, (time.perf_counter() - self.started) * 1000.0)
