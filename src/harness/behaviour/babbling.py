"""Лепет: агент открывает своё тело нажатиями.

Он не знает ни одного названия клавиши, ни одной функции, ни того, сколько
выходов вообще что-то делают. Он знает только, что выходы адресуемы. Всё
остальное добывается пробами — и это единственный способ, которым карта тела
может возникнуть, не будучи подсказкой.

Три вещи, которые делают лепет осмысленным, а не случайным перебором:

1. **Длительность — часть пробы.** Один и тот же выход, зажатый на 40 мс и на
   400 мс, может дать разное. Поэтому проба — это `(выход, удержание)`, и
   удержания перебираются тоже (инвариант 8).
2. **Обратимость проверяется попыткой откатить, а не берётся из таблицы.**
   После пробы, что-то изменившей, агент пытается вернуть как было — тем же
   выходом или другим. Получилось — обратимость растёт, не получилось — падает.
   Осторожность выводится отсюда и только отсюда (инварианты 9 и 10).
3. **Пауза после необратимого.** Если откатить не удалось, лепет замирает на
   `pause_after_irreversible_ms`: последствие надо увидеть, а не затоптать
   следующей пробой.

Порядок проб не случаен: сначала непробованное (там больше всего неизвестного),
потом то, где мало повторов, и в конце сочетания. Осторожные выходы отодвигаются
в хвост — но не исключаются: исключить значило бы завести список запретов.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

from ..core.action import Action, Reversibility
from ..core.clocks import Stamp
from ..core.journal import (Actor, ActorLayer, Journal, Kind as EntryKind,
                            StateSnapshot)
from ..core.profile import Profile
from ..model.rebuild import BodyMap

# Потолок темпа исследования. Не «на всякий случай»: без него коэффициент усиления
# при скуке мог бы дать сотню проб за такт, и один такт съел бы весь бюджет прогона.
PACE_MAX = 4.0


@dataclass(frozen=True, slots=True)
class Probe:
    """Одна проба: что нажать и как долго."""

    output: str
    duration_ms: int
    modifiers: tuple[str, ...] = ()
    why: str = ""              # почему выбрана именно эта проба; для журнала

    def to_action(self, reversibility: Reversibility) -> Action:
        return Action.key(self.output, self.duration_ms, self.modifiers,
                          reversibility=reversibility)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Что вышло. `changed` и `undone` — объективные наблюдения, не мнения."""

    probe: Probe
    delivered: bool
    changed: bool
    undone: bool | None        # None — откат не пробовали
    error_value: float | None = None
    undo_output: str | None = None      # чем пробовали откатить
    undo_was_known: bool = False        # это был уже найденный способ или догадка

    def as_event(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": "delivered" if self.delivered else "failed",
                             "responded": self.changed, "why": self.probe.why}
        if self.undone is not None:
            d["undone"] = self.undone
            d["undo_output"] = self.undo_output
            d["undo_was_known"] = self.undo_was_known
        if self.error_value is not None:
            d["prediction_error"] = round(self.error_value, 6)
        return d


class Babbler:
    """Выбирает пробы и обновляет карту тела по результатам.

    Ничего не исполняет сам: исполнение передаётся снаружи функцией. Так лепет
    остаётся проверяемым офлайн — по интерактивному миру, без устройства и без игры.
    """

    def __init__(self, profile: Profile, outputs: Sequence[str], *,
                 body: BodyMap | None = None, journal: Journal | None = None,
                 rng_seed: int = 0,
                 state_extra: Callable[[], dict[str, Any]] | None = None) -> None:
        import random

        p = profile.parameters
        self.profile = profile
        self.outputs = tuple(dict.fromkeys(outputs))
        if not self.outputs:
            raise ValueError("нечего пробовать: тело без выходов")
        self.body = body or BodyMap(
            live_min_responses=int(p["body_live_min_responses"]),
            # Порог «молчит» — это и есть `babble_repeats`: столько раз лепет
            # пробует выход, прежде чем считать его молчащим. Отдельная ручка на то
            # же самое разошлась бы с ней при первой же настройке, и вывод «молчит»
            # стал бы недостижим — на замере так и было: пробовалось два раза,
            # требовалось три, и пятнадцать молчащих выходов из двадцати четырёх
            # навсегда остались неясными.
            silent_min_deliveries=int(p["babble_repeats"]),
            response_sigmas=float(p["body_excess_sigmas"]))
        self.journal = journal
        # Чем дополнить срез состояния: драйвы, настроение, активная цель. Лепет их
        # не знает — их знает тот, кто его вызывает, и отдаёт замыканием.
        self._state_extra = state_extra
        self.hold_min = int(p["babble_hold_min_ms"])
        self.hold_max = int(p["babble_hold_max_ms"])
        if self.hold_min > self.hold_max:
            raise ValueError("babble_hold_min_ms больше babble_hold_max_ms")
        self.repeats = int(p["babble_repeats"])
        self.combo_after = int(p["babble_combo_after"])
        self.caution_threshold = float(p["irreversibility_threshold"])
        self.rate = float(p["babble_rate"])
        self._pace: float | None = None   # первый такт не пропускается
        self.paced_out = 0          # тактов пропущено темпом: видно в отчёте
        self.pause_ms = int(p["pause_after_irreversible_ms"])
        self._rng = random.Random(rng_seed)
        self._pause_left_ms = 0
        self.probes_done = 0
        self.irreversible_hits = 0
        # Что уже пробовали в качестве отката для каждого выхода и что сработало.
        # Это тоже добытое знание, а не таблица: «X откатывается Y» выясняется
        # только тем, что Y однажды вернул мир как было.
        self._undo_tried: dict[str, set[str]] = {}
        self.inverse_found: dict[str, str] = {}
        # Выходы, для которых догадки кончились: все живые выходы уже пробованы
        # как откат и ни один не вернул мир. Осторожность у них остаётся
        # максимальной — и это верный итог, а не недоработка.
        self._undo_exhausted: set[str] = set()

    # --- выбор пробы --------------------------------------------------------

    def _holds(self) -> tuple[int, ...]:
        """Три удержания: короткое, среднее, длинное. Больше не нужно — разница
        между 300 и 320 мс не выучивается, а перебор растёт линейно."""
        mid = (self.hold_min + self.hold_max) // 2
        return (self.hold_min, mid, self.hold_max)

    def pace(self, explore_rate: float | None = None) -> int:
        """Сколько проб на этом такте. Темп исследования — ось модуляции.

        Число, а не «да/нет», потому что ось двусторонняя: страх тормозит
        исследование, скука его ускоряет. Доля тактов выразить второе не может —
        быстрее, чем каждый такт, доля не бывает, и «скука ускоряет лепет» при
        базовом темпе 1.0 оказалось бы невыполнимым требованием. Поэтому 0.5 — проба
        через такт, 2.0 — две пробы за такт.

        Отдельный метод, а не значение `None` из `next_probe`: `None` там уже значит
        «идёт пауза после необратимого», и совмещение двух смыслов в одном ответе
        сразу же дало ложное «при высокой осторожности лепет встал совсем» — вызов из
        одной точки не мог отличить «темп пропустил такт» от «лепет остановился».

        Счётчик, а не случайность: прогон обязан воспроизводиться от сида, и «страх
        притормозил исследование» должно читаться в числе проб, а не в шуме. Первый
        такт не пропускается никогда — пропущенный первый такт неотличим от
        неработающего лепета.

        До этого ручка `babble_rate` не читалась ни одной строкой кода: объявленный и
        никем не спрошенный темп.
        """
        rate = max(0.0, min(PACE_MAX, self.rate if explore_rate is None
                            else float(explore_rate)))
        if self._pace is None:
            # Первый такт не пропускается никогда, и остаток подобран так, чтобы
            # дальше темп шёл ровно `rate`: иначе фиксированный старт завышал бы
            # долю проб на малых темпах (при 0.5 выходило 0.75).
            self._pace = 1.0 - rate
        self._pace += rate
        times = int(self._pace)
        self._pace -= times
        if times == 0:
            self.paced_out += 1
        return times

    def next_probe(self, *, caution_threshold: float | None = None) -> Probe | None:
        """Что пробовать дальше. `None`, если идёт пауза после необратимого."""
        if self._pause_left_ms > 0:
            return None
        threshold = self.caution_threshold if caution_threshold is None else caution_threshold

        untried = [o for o in self.outputs if o not in self.body.outputs]
        if untried:
            out = untried[0]
            return Probe(out, self._holds()[0], why="ни разу не пробован")

        # Недобранные повторы: выход, отвечавший не всегда, надо перепроверить.
        thin = [o for o in self.outputs
                if self.body.outputs[o].delivered < self.repeats]
        if thin:
            out = min(thin, key=lambda o: (self.body.outputs[o].delivered, o))
            n = self.body.outputs[out].delivered
            return Probe(out, self._holds()[n % 3], why=f"повтор {n + 1} из {self.repeats}")

        # Живые выходы с неизвестной обратимостью — самое ценное непознанное.
        #
        # Из перебора выпадают два вида выходов, и по разным причинам:
        #
        # - те, для кого догадки об откате кончились (`_undo_exhausted`);
        # - те, кто перестал реагировать (`no_change_streak`). Ломать можно только
        #   то, что ещё не сломано: такой выход отвечал раньше, а теперь не даёт
        #   ничего, и попытки отката по нему не случаются вообще. На замере
        #   именно на этом застревал весь лепет — 1170 проб из 1266 уходили в
        #   один такой выход, потому что его счётчик поисков не мог вырасти.
        #
        # Порядок — по числу уже потраченных проб: оно растёт всегда, поэтому
        # голодания не бывает ни при какой форме мира.
        stale = 3
        unknown = [o for o in self.outputs
                   if self.body.outputs[o].state == "live"
                   and not self.body.outputs[o].reversibility.is_known
                   and o not in self._undo_exhausted
                   and self.body.outputs[o].no_change_streak < stale]
        if unknown:
            out = min(unknown, key=lambda o: (self.body.outputs[o].delivered,
                                              self.body.outputs[o].undo_searches, o))
            return Probe(out, self._holds()[1], why="обратимость неизвестна")

        # Сочетания — только после того, как одиночные разобраны.
        live = sorted(o for o in self.outputs if self.body.outputs[o].state == "live")
        if len(self.body.outputs) >= self.combo_after and len(live) >= 2:
            safe = [o for o in live
                    if self.body.outputs[o].reversibility.caution < threshold]
            if len(safe) >= 2:
                a, b = self._rng.sample(safe, 2)
                return Probe(a, self._holds()[1], modifiers=(b,),
                             why="сочетание двух живых выходов")

        # Всё разобрано: продолжаем уточнять то, где разброс больше. Осторожные
        # уходят в хвост, но не исключаются — списка запретов здесь нет.
        candidates = sorted(
            self.outputs,
            key=lambda o: (self.body.outputs[o].reversibility.caution >= threshold,
                           self.body.outputs[o].reversibility.sigma * -1,
                           self.body.outputs[o].delivered, o))
        out = candidates[0]
        return Probe(out, self._rng.choice(self._holds()), why="уточнение")

    def reversibility_of(self, output: str) -> Reversibility:
        f = self.body.outputs.get(output)
        return f.reversibility if f else Reversibility()

    def undo_candidate(self, output: str) -> str | None:
        """Чем попробовать откатить последствие выхода.

        Порядок догадок — от самой дешёвой к самой дорогой:

        1. Уже найденная обратная пара, если она есть.
        2. Тот же выход: для переключателя это верно, и проверить дешевле всего.
        3. Другой живой выход, ещё не пробованный как откат для этого.

        `None` значит «идей больше нет», и тогда выход остаётся с неизвестной или
        низкой обратимостью — то есть осторожным. Это честный итог: агент не
        нашёл способа откатить, а не «выход запрещён».
        """
        known = self.inverse_found.get(output)
        if known is not None:
            return known
        tried = self._undo_tried.setdefault(output, set())
        if output not in tried:
            return output

        # Догадка о симметрии: если Y однажды вернул то, что сделал `output`, то,
        # возможно, `output` вернёт то, что делает Y. Это именно догадка и она
        # проверяется пробой, а не принимается за факт, — но проверять её стоит
        # раньше случайного перебора, потому что она чаще верна.
        for other, inverse in self.inverse_found.items():
            if inverse == output and other not in tried:
                return other

        live = [o for o in self.outputs
                if o != output and self.body.outputs.get(o) is not None
                and self.body.outputs[o].state == "live" and o not in tried]
        # Сначала те, чью обратимость мы уже видели: они предсказуемее, и если
        # такой выход вернёт мир, пара найдена надёжнее.
        if not live:
            self._undo_exhausted.add(output)
            return None
        live.sort(key=lambda o: (not self.body.outputs[o].reversibility.is_known, o))
        return live[0]

    def note_undo_attempt(self, output: str, undo_output: str, worked: bool) -> None:
        """Запомнить, что пробовали и чем кончилось."""
        self._undo_tried.setdefault(output, set()).add(undo_output)
        if worked:
            self.inverse_found[output] = undo_output
            # Обратимость симметрична: если Y вернул последствие X, то X, скорее
            # всего, вернёт последствие Y. Это догадка, и она будет проверена
            # отдельной пробой, а не принята за факт.
            self._undo_tried.setdefault(undo_output, set())

    # --- результат ----------------------------------------------------------

    def absorb(self, result: ProbeResult, stamp: Stamp | None = None) -> None:
        """Записать результат пробы: в карту тела и, если есть, в журнал."""
        probe = result.probe
        self.probes_done += 1
        for out in (*probe.modifiers, probe.output):
            f = self.body.fact(out, stamp.t_self if stamp else 0)
            f.tries += 1
            if not result.delivered:
                continue
            f.delivered += 1
            f.durations_ms.append(probe.duration_ms)
            if result.changed:
                f.responded += 1
                f.no_change_streak = 0
            else:
                f.no_change_streak += 1
            if result.undone is None:
                continue
            if result.undone:
                # Способ найден: это и есть свидетельство обратимости.
                f.reversibility = f.reversibility.observe(True)
                f.undo_method = result.undo_output
            elif result.undo_was_known:
                # Известный способ перестал работать — вот это уже про мир.
                f.reversibility = f.reversibility.observe(False)
            else:
                # Догадка не сработала. Это про догадку, а не про мир: оценка
                # обратимости остаётся неизвестной, то есть осторожность —
                # максимальной. Смешивать одно с другим значило бы считать
                # необратимым всё, к чему не сразу подобрался ключ.
                f.undo_searches += 1

        if result.undone is False and result.undo_was_known:
            self.irreversible_hits += 1
            self._pause_left_ms = self.pause_ms
        elif result.undone is False:
            # Не нашли способ — тоже повод притормозить, но короче: последствие
            # уже случилось и его надо разглядеть.
            self._pause_left_ms = max(self._pause_left_ms, self.pause_ms // 3)

        if self.journal is not None and stamp is not None:
            action = probe.to_action(self.reversibility_of(probe.output))
            # Лепет — не рефлекс, не навык и не план: ни один из них ещё не
            # существует. Он происходит потому, что любопытство вне коридора и
            # больше ответить нечем, поэтому инициатор — контур драйвов. Если
            # когда-нибудь появится отдельный контур лепета, изменится набор
            # слоёв (структурное изменение), а не эта атрибуция.
            # Срез состояния, а не только событие. Ошибка предсказания лежала
            # **только** в событии (`ProbeResult.as_event`), и из-за этого
            # `vitals.prediction_error_mean` докладывал «ни одна запись не несёт
            # ошибки предсказания» по журналу, каждая запись которого её несла:
            # читатели построены на `StateSnapshot`, а число было положено рядом.
            # Величина в записи есть, а измерить её нельзя — худший вид расхождения,
            # потому что данные выглядят целыми.
            #
            # Драйвы, настроение и цель приходят от вызывающего (`state_extra`): лепет
            # про них не знает и знать не должен — иначе `behaviour/babbling` начнёт
            # импортировать мотивацию, и слой перестанет быть проверяемым отдельно.
            extra = dict(self._state_extra() if self._state_extra else {})
            state = StateSnapshot(prediction_error=result.error_value,
                                 drives=extra.get("drives", {}),
                                 mood=extra.get("mood"),
                                 goal_id=extra.get("goal_id"))
            self.journal.append(EntryKind.ACTION, stamp, Actor.AGENT,
                                ActorLayer.DRIVE, action=action, state=state,
                                event={**result.as_event(), "device": "babble",
                                       "latency_ms": float(probe.duration_ms)})

    def tick_pause(self, dt_ms: float) -> None:
        self._pause_left_ms = max(0, self._pause_left_ms - int(dt_ms))

    @property
    def paused(self) -> bool:
        return self._pause_left_ms > 0

    # --- сводка -------------------------------------------------------------

    def progress(self) -> dict[str, Any]:
        """«118 из 512 выходов проверено» — та же сводка, что в пульте."""
        st = self.body.stats()
        return {
            "outputs_total": len(self.outputs),
            "outputs_touched": st["known_outputs"],
            "live": st["live"], "silent": st["silent"],
            "unclear": st["unclear"],
            "untried": len(self.outputs) - st["known_outputs"],
            "background_rate": round(self.body.background_rate, 4),
            "probes_done": self.probes_done,
            "irreversible_hits": self.irreversible_hits,
            "unknown_reversibility": st["unknown_reversibility"],
            "not_undoable": self.body.dangerous(self.caution_threshold),
            "inverse_pairs": dict(sorted(self.inverse_found.items())),
            "undo_search_exhausted": sorted(self._undo_exhausted),
            "paused_ms": self._pause_left_ms,
        }


def run_babbling(world: Any, babbler: Babbler, *, steps: int,
                 clocks: Any, journal: Journal | None = None,
                 undo_with: Callable[[str], str | None] | None = None,
                 error: Any = None,
                 explore_rate: float | None = None,
                 caution_threshold: float | None = None) -> dict[str, Any]:
    """Прогон лепета по миру. Проверяемо офлайн, без игры и без устройства.

    `undo_with` — чем агент пытается откатить последствие. По умолчанию он
    пробует тот же выход ещё раз: это самая простая догадка «нажму опять, вдруг
    вернётся», и для переключателя она верна, а для необратимого — нет. Именно
    так обратимость и выясняется: попыткой, а не таблицей.

    ## Последствие определяется по пикселям, а не по состоянию мира

    Раньше здесь читалось `Observation.changed` — то есть изменилось ли внутреннее
    состояние мира. Это истина, которой у агента нет: он видит только пиксели.
    В игре ошибка не проявлялась, потому что там изменение состояния всегда видно
    в кадре, — ровно такие ошибки и живут годами.

    И сравнивать надо не с нулём, а с тем, как мир меняется **сам**. В видео
    картинка идёт без всяких действий; в браузере крутится анимация. Сравнение с
    нулём объявило бы живыми все выходы подряд — что на замере и произошло: в
    домене «видео» из двенадцати молчащих выходов не нашлось ни одного.

    Поэтому фон измеряется на шагах без действия, и последствием считается
    превышение над фоном на `babble_response_sigmas` сигм.
    """
    from ..core.action import Action as _Action
    from ..vision.selfworld import frame_energy

    sigmas = float(babbler.profile.parameters["babble_response_sigmas"])
    floor = float(babbler.profile.parameters["babble_response_floor"])
    idle: list[float] = []
    # Порог врёт и сам по себе: анимация интерфейса или идущее видео иногда дают
    # превышение над фоном без всякого действия. Насколько часто — измеримо тем же
    # тестом по паре кадров без действия, и без этой величины нельзя отличить
    # «ответил один раз» от «порог сработал один раз».
    idle_tests = 0
    false_alarms = 0

    def background() -> tuple[float, float]:
        if len(idle) < 3:
            return 0.0, 0.0
        arr = idle[-64:]
        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / len(arr)
        return mean, var ** 0.5

    def differs(change: float) -> bool:
        """Отличается ли изменение кадра от того, как мир меняется сам.

        Проверка двусторонняя, и это не формальность. В видео нажатие «пауза»
        *уменьшает* изменение кадра до нуля — последствие налицо, а односторонняя
        проверка «стало больше фона» его не увидит. На замере из-за этого в домене
        «видео» не находилось ни одного живого выхода из четырёх.
        """
        mean, sigma = background()
        # Пол нужен: при побитово неподвижном фоне сигма равна нулю, и любое
        # дрожание в один уровень яркости стало бы «последствием».
        return abs(change - mean) > max(floor, sigmas * max(sigma, 1e-4))

    def slots() -> Iterator[bool]:
        """Такты и пробы внутри такта. Темп может дать больше одной пробы за такт.

        Отдельный генератор, чтобы тело пробы осталось нетронутым: переписывать
        семьдесят строк ради двух осей модуляции — это заводить новые ошибки в
        механизме, который уже замерен.
        """
        for _ in range(steps):
            babbler.tick_pause(1000.0
                               / float(babbler.profile.parameters["capture_fps"]))
            times = babbler.pace(explore_rate)
            if times == 0:
                yield False
                continue
            for _ in range(times):
                yield True

    for probing in slots():
        # Две оси модуляции читаются здесь: темп исследования решает, сколько проб
        # на такте, порог необратимости — что именно выбрать. Без этих двух строк
        # обе оси считались бы реализованными и не влияли бы ни на что.
        probe = (babbler.next_probe(caution_threshold=caution_threshold)
                 if probing else None)
        if probe is None:
            world.step(None, with_audio=False)      # мир идёт и во время паузы
            clocks.tick_self()
            clocks.set_world(clocks.t_world + 1)
            continue

        idle_a = world.step(None, with_audio=False).frame
        clocks.tick_self()
        clocks.set_world(clocks.t_world + 1)
        before = world.step(None, with_audio=False).frame
        clocks.tick_self()
        clocks.set_world(clocks.t_world + 1)
        # Два шага без действия подряд дают замер фона: столько мир меняется сам.
        # Этот же замер проверяется тем же порогом — до того, как попадёт в фон,
        # иначе проба проверяла бы саму себя.
        sample_idle = frame_energy(idle_a, before)
        if len(idle) >= 3:
            idle_tests += 1
            if differs(sample_idle):
                false_alarms += 1
            babbler.body.set_background_rate(false_alarms / idle_tests)
        idle.append(sample_idle)

        action = probe.to_action(babbler.reversibility_of(probe.output))
        after = world.step(action, with_audio=False)
        clocks.tick_self()
        clocks.set_world(clocks.t_world + 1)
        changed = differs(frame_energy(before, after.frame))

        err = None
        if error is not None:
            sample = error.feed(after.frame)
            err = sample.value if sample else None

        undone: bool | None = None
        undo_output = None
        undo_known = False
        # Обратимость проверяется только у одиночной пробы. У сочетания мир
        # изменили два выхода, и откат одного из них мир не вернёт — а вина за
        # это досталась бы одному, хотя виноваты оба. На замере именно так
        # обратимость движения падала с 1.0 до 0.33 при верно найденной паре.
        # Обратимость сочетания — отдельный вопрос, и он требует отдельной пробы.
        if changed and not probe.modifiers:
            undo_known = probe.output in babbler.inverse_found
            undo_output = (babbler.undo_candidate(probe.output) if undo_with is None
                           else undo_with(probe.output))
            if undo_output is not None:
                world.step(_Action.key(undo_output, probe.duration_ms), with_audio=False)
                clocks.tick_self()
                clocks.set_world(clocks.t_world + 1)
                restored = world.step(None, with_audio=False).frame
                clocks.tick_self()
                clocks.set_world(clocks.t_world + 1)
                # Откат удался, если расхождение с тем, что было до пробы, не
                # больше того, как мир меняется сам. Побитового совпадения не
                # будет никогда: интерфейс анимирован, а в видео картинка идёт.
                undone = not differs(frame_energy(before, restored))
                babbler.note_undo_attempt(probe.output, undo_output, undone)

        babbler.absorb(ProbeResult(probe, True, changed, undone, err,
                                   undo_output=undo_output,
                                   undo_was_known=undo_known),
                       clocks.stamp())

    return babbler.progress()
