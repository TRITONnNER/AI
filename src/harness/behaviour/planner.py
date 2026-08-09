"""Планировщик: цепочка действий, выбранная воображением.

Последняя большая недостающая часть архитектуры. Всё, на чём она стоит, уже есть:
цели с объективным тестом, навыки со статистикой, модель перехода из журнала и
размыкатель эффекторов. Планировщик их соединяет и ничего своего не выдумывает.

Как это устроено и почему именно так.

**Поиск идёт по модели перехода, а не по правилам мира.** Правил агент не знает.
Поэтому расширение узла — это вопрос модели «что будет, если нажать это здесь», и
там, где модель отвечает «не знаю», ветка не строится вообще. План через незнание
был бы обещанием, которое нечем сдержать.

**Стоимость — измеренные секунды.** `mu_seconds` рёбер графа мест, то есть сколько
проход занимал на самом деле. Никакой стоимости по умолчанию у неизвестного
перехода нет: у него нет стоимости, потому что его не наблюдали.

**Осторожность не смешивается со стоимостью.** Прибавить к времени «штраф за
необратимость» значило бы придумать секунды, которых никто не измерял, и после этого
время в плане перестало бы значить время. Поэтому осторожность — отдельный признак
плана (`risky`, `max_caution`), и планировщик ищет дважды: сначала маршрут без
неоткатываемых шагов, и только если такого нет — обычный, помеченный опасным.

**Планировщик прерываем.** Это генератор, а не функция: мир не ставится на паузу
(инвариант 3). После каждого расширения он отдаёт управление, а лучший план на
текущий момент лежит в `best` и доступен в любой миг.

**План — гипотеза, пока не применён.** Ровно как навык. Модель может ошибаться, и
поэтому исполнение сверяет каждый шаг с предсказанием: разошлось — план брошен,
расхождение записано, и это же расхождение есть ошибка предсказания, то есть та
самая единая валюта. Плана, который «сработал по модели», недостаточно ни для чего.

**Награды здесь нет.** Ни очков, ни оценки полезности. План либо приводит туда, где
тест цели проходит, либо не приводит. Это проверяется тестом, который читает AST
модуля и ищет соответствующие идентификаторы.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..core.action import (UNKNOWN, Action, Reversibility, action_key,
                          macro_key, parse_action_key, parse_any_key)
from ..core.branches import Arbitration
from ..core.clocks import Stamp
from ..core.journal import (Actor, ActorLayer, Journal, Kind as EntryKind,
                          StateSnapshot)
from ..core.profile import Profile
from ..model.drives import Modulation
from ..model.forward import ForwardModel
from .goals import Goal
from .imagination import Loop, Mode
from .skills import Step


class PlanError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PlanStep:
    """Шаг плана: что нажать, куда это по модели ведёт и насколько это надёжно.

    Шаг бывает одиночным нажатием или **цепочкой** — навыком из библиотеки. Во втором
    случае `output` и `duration_ms` описывают первое нажатие, а `chain` — все ключи
    действий по порядку. Так сделано затем, чтобы весь прежний код, который смотрит на
    `output`, продолжал работать и не начал молча делать половину шага: `actions()`
    возвращает столько действий, сколько их в шаге, и вызывающий обязан выполнить все.
    """

    output: str
    duration_ms: int
    expected_place: str
    p: float                     # доля переходов, закончившихся именно там
    seconds: float
    n: int                       # на скольких наблюдениях держится
    caution: float
    # Ключи всех действий шага. Пустой кортеж — одиночное нажатие.
    chain: tuple[str, ...] = ()

    @property
    def is_macro(self) -> bool:
        return len(self.chain) > 1

    def key(self) -> str:
        """Ключ шага: тот же, под которым он лежит в модели перехода."""
        if self.is_macro:
            return macro_key(list(self.chain))
        return action_key(self.output, self.duration_ms)

    def to_step(self) -> Step:
        return Step(self.output, self.duration_ms)

    def actions(self, reversibility: Reversibility = UNKNOWN) -> tuple[Action, ...]:
        """Действия шага по порядку. У одиночного шага одно, у навыка — все.

        Обратимость передаётся снаружи и одна на все действия шага: она про выход, а
        планировщик знает только осторожность цепочки в целом. Разбирать её по звеньям
        здесь значило бы выдумать то, чего в модели нет.
        """
        if not self.is_macro:
            return (Action.key(self.output, self.duration_ms,
                               reversibility=reversibility),)
        out: list[Action] = []
        for one in self.chain:
            parsed = parse_action_key(one)
            if parsed is None:               # ключ проверен при сборке макроса
                raise PlanError(f"в шаге плана не ключ действия: {one!r}")
            output, ms, mods = parsed
            out.append(Action.key(output, ms, mods, reversibility=reversibility))
        return tuple(out)

    def as_dict(self) -> dict[str, Any]:
        return {"output": self.output, "duration_ms": self.duration_ms,
                "expected_place": self.expected_place, "p": round(self.p, 4),
                "seconds": round(self.seconds, 3), "n": self.n,
                "caution": round(self.caution, 4),
                "chain": list(self.chain), "is_macro": self.is_macro}


@dataclass(slots=True)
class Plan:
    """Цепочка шагов до места, где тест цели должен пройти.

    `is_guess` — плана касается то же правило, что навыка: пока он не применён в
    мире, он остаётся предположением, каким бы уверенным ни выглядел по модели.
    """

    goal_id: str
    steps: list[PlanStep]
    target: str
    seconds: float = 0.0
    confidence: float = 0.0          # произведение p по шагам: где тоньше, там и рвётся
    max_caution: float = 0.0
    risky: bool = False
    expansions: int = 0
    # Самое слабое звено: сколько раз наблюдён наименее проверенный шаг. Уверенность
    # как произведение долей ничего не говорит о том, на чём эти доли держатся.
    min_step_n: int = 0
    applied: int = 0
    worked: int = 0
    reason: str = ""

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def is_guess(self) -> bool:
        return self.applied == 0

    def outputs(self) -> list[str]:
        return [s.output for s in self.steps]

    def as_dict(self) -> dict[str, Any]:
        return {"goal_id": self.goal_id, "target": self.target,
                "steps": [s.as_dict() for s in self.steps],
                "length": self.length, "seconds": round(self.seconds, 3),
                "confidence": round(self.confidence, 4),
                "max_caution": round(self.max_caution, 4), "risky": self.risky,
                "min_step_n": self.min_step_n,
                "expansions": self.expansions, "is_guess": self.is_guess,
                "applied": self.applied, "worked": self.worked,
                "reason": self.reason}


@dataclass(slots=True)
class Progress:
    """Что планировщик успел. Отдаётся после каждого расширения."""

    expansions: int
    frontier: int
    visited: int
    best: Plan | None
    done: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"expansions": self.expansions, "frontier": self.frontier,
                "visited": self.visited, "done": self.done,
                "best": None if self.best is None else self.best.as_dict()}


class Planner:
    """Поиск цепочки по модели перехода. Генератор, потому что прерываемый.

    Обход — по наименьшему измеренному времени (Дейкстра по `mu_seconds`). Это не
    выбор из вкусовых соображений: время — единственная величина в модели, которую
    агент действительно измерял, и потому единственная, по которой можно упорядочить
    маршруты, ничего не придумав.
    """

    def __init__(self, profile: Profile, model: ForwardModel, *,
                 hold_ms: int | None = None) -> None:
        p = profile.parameters
        self.model = model
        self.max_expansions = int(p["plan_max_expansions"])
        self.max_length = int(p["plan_max_length"])
        # Базовая длина хранится отдельно от текущей: модуляция домножает базу, а не
        # уже сдвинутое значение, иначе горизонт съезжает от такта к такту сам собой.
        self.base_max_length = self.max_length
        self.min_step_p = float(p["plan_min_step_p"])
        self.min_step_n = int(p["plan_min_step_n"])
        self.caution_threshold = float(p["irreversibility_threshold"])
        # Удержание шага берётся из наблюдённого перехода. `hold_ms` остаётся
        # только для планов, где длительность взять неоткуда, — таких быть не должно.
        self.hold_ms = int(hold_ms if hold_ms is not None else p["babble_hold_min_ms"])
        self.best: Plan | None = None
        self.expansions = 0

    # --- поиск --------------------------------------------------------------

    def search(self, goal: Goal, start: str, target: str) -> Iterator[Progress]:
        """Искать маршрут `start` → `target`. Отдаёт управление после каждого шага.

        Два прохода в одном генераторе. Сначала ищется маршрут, не наступающий на
        то, что агент не умеет откатывать; если такого нет — ищется любой, и он
        помечается опасным. Это не запрет и не список исключений: осторожность
        определяется как «я не умею это откатить» (инвариант 9), и она меняет
        порядок предпочтения, а не список возможного.

        Два прохода внутри, а не снаружи, потому что вызывающему нельзя оставлять
        выбор «а вызвать ли второй раз»: забытый второй вызов выглядел бы как
        «маршрута нет», хотя маршрут есть.
        """
        yield from self._pass(goal, start, target, avoid_risky=True)
        if self.best is None:
            # Бюджет второго прохода свой: первый мог израсходовать весь, и тогда
            # «маршрута нет» означало бы «не успел посмотреть», а это разные вещи.
            self.expansions = 0
            yield from self._pass(goal, start, target, avoid_risky=False)

    def _pass(self, goal: Goal, start: str, target: str,
              *, avoid_risky: bool) -> Iterator[Progress]:
        if start == target:
            self.best = Plan(goal.id, [], target, reason="уже на месте")
            yield Progress(0, 0, 0, self.best, done=True)
            return

        # (секунды, длина, место, шаги) — длина в ключе, чтобы порядок был
        # определён при равном времени и результат не зависел от порядка словаря.
        frontier: list[tuple[float, int, str, tuple[PlanStep, ...]]] = [
            (0.0, 0, start, ())]
        # Место запоминается вместе с длиной пути, а не только со временем. Иначе
        # обход по времени и предел длины ссорятся: место, впервые достигнутое
        # дешёвым, но длинным путём, закрывает себя для короткого — а из длинного
        # дальше идти уже нельзя, предел исчерпан. На замере это стоило 307
        # недостижимых целей из 1741: обход не находил того, что в графе есть.
        best_at: dict[tuple[str, int], float] = {(start, 0): 0.0}
        visited = 0

        while frontier and self.expansions < self.max_expansions:
            seconds, length, place, steps = heapq.heappop(frontier)
            visited += 1
            if length >= self.max_length:
                yield Progress(self.expansions, len(frontier), visited, self.best)
                continue

            for key in self.model.actions_from(place):
                keys = parse_any_key(key)
                if keys is None:
                    continue                    # не действие — повторить нечем
                first = parse_action_key(keys[0])
                if first is None:
                    continue
                output, duration_ms = first[0], first[1]
                # Два разных «цепочки», и путать их нельзя. `macro` — ключи
                # действий внутри одного шага (навык из библиотеки). `route` —
                # последовательность шагов плана. До исправления оба лежали в одной
                # переменной `chain`, и у **второго** исхода того же действия
                # `macro` оказывался кортежем `PlanStep`: шаг становился макросом из
                # шагов плана, `key()` падал на разборе, а `actions()` пытался
                # исполнить объект вместо нажатия. Одиночные исходы это не задевало,
                # поэтому ошибка ждала первого места с двумя исходами — то есть
                # ровно записи петель.
                macro = keys if len(keys) > 1 else ()
                pred = self.model.predict(place, key)
                if pred is None:
                    continue                    # «не знаю» ветку не строит
                caution = self.model.caution(key)
                if avoid_risky and caution >= self.caution_threshold:
                    continue
                # Все наблюдённые исходы, а не только самый частый: отброшенная
                # ветка бывает единственной, ведущей к цели.
                for outcome, p in pred.outcomes():
                    if p < self.min_step_p:
                        continue                # слишком ненадёжный шаг
                    if outcome.n < self.min_step_n:
                        continue                # наблюдён слишком мало раз
                    self.expansions += 1
                    # Удержание берётся из наблюдения, а не из настройки: план
                    # обязан повторять то действие, на котором модель училась.
                    step = PlanStep(output, duration_ms, outcome.dst, p,
                                    outcome.mu_seconds, outcome.n, caution,
                                    chain=macro)
                    route = steps + (step,)
                    total = seconds + outcome.mu_seconds
                    dst = outcome.dst

                    if dst == target:
                        candidate = self._plan(goal, route, target)
                        if (self.best is None
                                or candidate.seconds < self.best.seconds):
                            self.best = candidate
                        continue

                    seen_at = (dst, length + 1)
                    known = best_at.get(seen_at)
                    if known is None or total < known:
                        best_at[seen_at] = total
                        heapq.heappush(frontier, (total, length + 1, dst, route))
                yield Progress(self.expansions, len(frontier), visited, self.best)
                if self.expansions >= self.max_expansions:
                    break
            yield Progress(self.expansions, len(frontier), visited, self.best)

        done = Progress(self.expansions, len(frontier), visited, self.best, done=True)
        if self.best is None:
            done = Progress(self.expansions, len(frontier), visited, None, done=True)
        yield done

    def _plan(self, goal: Goal, chain: Sequence[PlanStep], target: str) -> Plan:
        seconds = sum(s.seconds for s in chain)
        confidence = 1.0
        for s in chain:
            confidence *= s.p
        max_caution = max((s.caution for s in chain), default=0.0)
        risky = max_caution >= self.caution_threshold
        min_n = min((s.n for s in chain), default=0)
        return Plan(goal.id, list(chain), target, seconds, confidence, max_caution,
                    risky, self.expansions, min_step_n=min_n,
                    reason=("маршрут найден" if not risky else
                            "маршрут найден, но в нём есть неоткатываемый шаг"))

    # --- удобная обёртка ----------------------------------------------------

    def modulate(self, modulation: Modulation) -> None:
        """Принять две оси модуляции: длину горизонта и порог необратимости.

        Горизонт задан в секундах, а план мерится шагами, поэтому переносится
        отношением к базе, а не самим числом: сжался горизонт вдвое — вдвое короче
        допустимый план. Прямой перевод секунд в шаги требовал бы знать, сколько
        секунд занимает шаг, а это свойство мира, которого агенту никто не сообщал.

        Осторожность идёт как есть: она и там, и здесь доля необратимости.

        Метод существует затем, что без него оси мертвы. Прежняя редакция считала
        горизонт «реализованной осью», а планировщик брал длину прямо из профиля и
        про настроение не знал.
        """
        base_h = float(modulation.base.get("horizon_s", 0.0))
        if base_h > 1e-9:
            scale = modulation.horizon_s / base_h
            self.max_length = max(1, int(round(self.base_max_length * scale)))
        self.caution_threshold = float(modulation.caution_threshold)

    def plan(self, goal: Goal, start: str, target: str) -> Plan | None:
        """Досчитать поиск до конца. Для офлайновых прогонов и тестов.

        В основном цикле так вызывать нельзя: это блокирующий вызов, а мир не
        ставится на паузу. Там нужен `search` через планировщика-контур.
        """
        for _ in self.search(goal, start, target):
            pass
        return self.best


# ---------------------------------------------------------------------------
# Разведка, из которой получается модель, по которой можно планировать
# ---------------------------------------------------------------------------
#
# Это не украшение планировщика, а условие его работы, и оно измерено.
#
# Случайный лепет по непрерывному миру даёт граф мест, в котором почти каждый
# переход наблюдён **один раз**. Такой переход выглядит достоверным на 100 %: доля
# «привело туда» равна единице по построению, потому что делить не на что. План из
# таких шагов выглядит уверенным и не доходит никогда: замер по 27 планам — 0
# дошедших. Планы, у которых самый слабый шаг наблюдён 6–7 раз, доходят в 89 %.
#
# Разница не в планировщике, а в том, что случайная разведка не повторяется: чтобы
# переход подтвердился, надо вернуться в то же место и нажать то же самое. Поэтому
# разведка здесь предпочитает **дожать неподтверждённое** тому, чтобы пробовать
# новое, и только потом расширяется.
#
# Никакого знания о мире в этом нет: выбор делается по собственной статистике
# переходов, то есть по журналу.


def choose_probe(model: ForwardModel, place: str | None,
                 outputs: Sequence[str], *, hold_ms: int,
                 min_n: int = 2, inverse: Mapping[str, str] | None = None,
                 last_output: str | None = None) -> tuple[str, int]:
    """Что пробовать здесь и сейчас, чтобы модель стала пригодной для плана.

    Порядок предпочтения, и каждый пункт — про честность модели, а не про новизну:

    1. **Подтвердить здесь.** Переход отсюда, наблюдённый меньше `min_n` раз: пока
       он не подтверждён, планировать по нему нельзя.
    2. **Расширить здесь.** Выход, которым отсюда ещё не уходили: без этого модель
       знает про место один выход из шестнадцати.
    3. **Вернуться, если делать здесь больше нечего.** Незнакомое место соблазняет
       уходить всё дальше, и разведка не возвращается никогда: замер по 2500 шагов
       без возврата — ни одного подтверждённого перехода и ни одного плана. Возврат
       делается обратной парой, найденной лепетом (`inverse_found`), то есть
       собственным знанием «чем откатывается это», а не встроенным «назад».

       Порядок здесь важен, и он тоже стоил замера: если возврат идёт **до**
       расширения, разведка запирается в пинг-понге между двумя местами — на прогоне
       она подтвердила один переход 738 раз и не открыла ни одного нового места.
    4. **Меньше всего пробованное.** Не «первый по списку»: с ним разведка жала одно
       и то же. Не случайный: прогон должен быть воспроизводим.
    """
    def least_used() -> str:
        """Выход, которым пользовались меньше всего — по всей модели, а не здесь.

        Нужно для мест, про которые неизвестно ничего: брать первый по списку
        значило бы жать одно и то же и уходить всё дальше в незнакомое. Замер: с
        «первым по списку» разведка ни разу не возвращалась туда, где уже что-то
        подтверждено, и планировать было не от чего.
        """
        used: dict[str, int] = {o: 0 for o in outputs}
        for (_src, key), outcomes in model.transitions.items():
            parsed = parse_action_key(key)
            if parsed is None or parsed[0] not in used:
                continue
            used[parsed[0]] += sum(o.n for o in outcomes)
        return min(outputs, key=lambda o: (used[o], o))

    def thin_transitions(where: str) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        for key in model.actions_from(where):
            pred = model.predict(where, key)
            if pred is None:
                continue
            weakest = min(o.n for o, _ in pred.outcomes())
            if weakest < min_n:
                out.append((weakest, key))
        return sorted(out)

    def anything_thin_anywhere() -> bool:
        return any(o.n < min_n
                   for outcomes in model.transitions.values() for o in outcomes)

    def step_back() -> tuple[str, int] | None:
        """Обратная пара, найденная лепетом. Не «назад», а «чем это откатывается»."""
        if inverse is None or last_output is None:
            return None
        back = inverse.get(last_output)
        return (back, hold_ms) if back else None

    if place is None or not model.actions_from(place):
        if anything_thin_anywhere():
            back = step_back()
            if back is not None:
                return back
        return least_used(), hold_ms

    thin = thin_transitions(place)
    if thin:
        parsed = parse_action_key(thin[0][1])
        if parsed is not None:
            return parsed[0], parsed[1]

    seen_outputs = set()
    for key in model.actions_from(place):
        parsed = parse_action_key(key)
        if parsed is not None:
            seen_outputs.add(parsed[0])
    fresh = [o for o in outputs if o not in seen_outputs]
    if fresh:
        return fresh[0], hold_ms

    # Здесь всё подтверждено и всё пробовано — значит пора туда, где нет.
    if anything_thin_anywhere():
        back = step_back()
        if back is not None:
            return back
    return least_used(), hold_ms


# ---------------------------------------------------------------------------
# Замыкание петли: прийти в известное иначе, а не уйти в неизвестное
# ---------------------------------------------------------------------------
#
# Зачем это отдельно от `choose_probe`. Прежняя разведка подтверждает переходы,
# пробует непробованное и возвращается обратной парой — то есть **той же дорогой,
# которой пришла**. Из-за этого подтверждённый подграф остаётся почти цепью со
# степенью 1–2, а цепь двудольна: все обходы между парой вершин имеют фиксированную
# чётность, поэтому достижимое за два шага **никогда** не достижимо за три. Замер
# это и показал: целей, достижимых и за два, и за три шага, ровно ноль при 54
# двухшаговых и 50 трёхшаговых путях. Сравнение планов при равной сложности при
# такой форме графа неосуществимо в принципе, и виноват не планировщик и не
# представление места, а то, что разведка ни разу не вернулась другой дорогой.
#
# Величина, которую надо максимизировать, — **прирост степени узлов**, а не число
# новых узлов. Отсюда две поправки к порядку предпочтений:
#
# 1. Возврат обратной парой ставится **последним**, а не третьим. Обратная пара по
#    построению не может добавить ребро: она ведёт туда, откуда пришли, тем же
#    ребром, которое уже есть.
# 2. Появляется предпочтение **недозамкнутого места**: если известен маршрут в
#    место со степенью ниже порога, у которого есть непробованные выходы, — идти
#    туда первым шагом этого маршрута. Ребро в известное место от нового
#    предшественника поднимает степень сразу двух узлов.
#
# Чего здесь нет: попыток угадать, куда ведёт непробованный выход. Этого знать
# нельзя, и подставить сюда догадку значило бы планировать по незнанию.


def degrees(graph: Any) -> dict[str, int]:
    """Степень каждого места по рёбрам, которые куда-то ведут. Петли не считаются.

    Петля — законное наблюдение («нажал и остался»), но степени она не добавляет:
    для маршрута она бесполезна, а цепь остаётся цепью при любом числе петель.
    """
    out: dict[str, set[str]] = {}
    for edge in graph.edges.values():
        if edge.dst == edge.src:
            continue
        out.setdefault(edge.src, set()).add(edge.dst)
        out.setdefault(edge.dst, set()).add(edge.src)
    return {k: len(v) for k, v in out.items()}


def undersewn(graph: Any, model: ForwardModel, outputs: Sequence[str], *,
              bar: int) -> list[tuple[int, int, str]]:
    """Места, куда стоит вернуться: мало рёбер и есть что попробовать.

    Возвращает `(степень, сколько непробовано, место)`, отсортированное так, что
    первым идёт самое недозамкнутое. Место без непробованных выходов в список не
    попадает: прийти туда можно, но добавить ребро оттуда нечем.
    """
    deg = degrees(graph)
    out: list[tuple[int, int, str]] = []
    for place in graph.places:
        seen: set[str] = set()
        for key in model.actions_from(place):
            parsed = parse_action_key(key)
            if parsed is not None:
                seen.add(parsed[0])
        untried = len([o for o in outputs if o not in seen])
        if untried and deg.get(place, 0) < bar:
            out.append((deg.get(place, 0), -untried, place))
    return sorted(out)


#: Ветки разведки с замыканием, в порядке приоритета. Объявлены списком, потому что
#: инвариант 26 требует счётчик на каждой, а счётчик по необъявленному списку врёт
#: умолчанием: ветка, которой нет в перечне, не попадёт в отчёт о нулевых
#: срабатываниях, и мёртвый код снова окажется невидимым.
CLOSING_BRANCHES = (
    ("нет модели здесь", "место незнакомо: решать нечем, отдаём прежней разведке"),
    ("подтвердить здесь", "переход наблюдён меньше min_n раз — планировать по нему нельзя"),
    ("непробованное здесь", "выход, которым отсюда не уходили, при недозамкнутом месте"),
    ("идти в недозамкнутое", "маршрут в место с малой степенью: прийти в известное иначе"),
    ("непробованное всё же", "дозамкнутых целей нет или до них нет маршрута"),
    ("прежняя разведка", "обратная пара и наименее пробованный выход"),
)


def closing_arbitration() -> Arbitration:
    """Счётчики для разведки с замыканием. Инвариант 26."""
    return Arbitration.of("разведка с замыканием", CLOSING_BRANCHES)


def choose_closing_probe(model: ForwardModel, graph: Any, place: str | None,
                         outputs: Sequence[str], *, hold_ms: int,
                         min_n: int = 2, degree_bar: int = 2,
                         inverse: Mapping[str, str] | None = None,
                         last_output: str | None = None,
                         arb: Arbitration | None = None) -> tuple[str, int]:
    """Разведка, которая старается прийти в известное иначе.

    Порядок предпочтения, и каждый пункт про степень, а не про новизну:

    1. **Подтвердить здесь.** Как и раньше: пока переход не наблюдён `min_n` раз,
       планировать по нему нельзя, и никакая топология этого не заменит.
    2. **Попробовать непробованное здесь.** Единственный способ добавить ребро из
       этого места.
    3. **Идти в недозамкнутое место** — то, у которого мало рёбер и есть что
       попробовать, — первым шагом известного маршрута. Это и есть «вернуться
       другой дорогой»: маршрут ведёт в известное место, но приходим мы туда от
       другого предшественника.
    4. **Обратная пара** — последней, а не третьей. Она не может поднять степень.
    5. Меньше всего пробованное, если ничего из перечисленного не нашлось.
    """
    def hit(name: str) -> None:
        if arb is not None:
            arb.hit(name)

    if place is None or not model.actions_from(place):
        hit("нет модели здесь")
        return choose_probe(model, place, outputs, hold_ms=hold_ms, min_n=min_n,
                            inverse=inverse, last_output=last_output)

    # 1 и 2 — как в `choose_probe`, повторно, потому что порядок дальше другой.
    for key in model.actions_from(place):
        pred = model.predict(place, key)
        if pred is None:
            continue
        if min(o.n for o, _ in pred.outcomes()) < min_n:
            parsed = parse_action_key(key)
            if parsed is not None:
                hit("подтвердить здесь")
                return parsed[0], parsed[1]

    seen_outputs: set[str] = set()
    for key in model.actions_from(place):
        parsed = parse_action_key(key)
        if parsed is not None:
            seen_outputs.add(parsed[0])
    fresh = [o for o in outputs if o not in seen_outputs]
    here_degree = degrees(graph).get(place, 0)
    # Ключевое место всей поправки, и оно стоило отдельного замера.
    #
    # Первая версия ставила «идти в недозамкнутое» **после** «пробовать
    # непробованное здесь» — и оказалась мёртвым кодом: на замере пункт
    # «непробованное здесь» срабатывал в 1500 шагах из 1500. Причина простая и
    # неустранимая порядком: каждый шаг уводит в новое место, где непробовано все
    # шестнадцать выходов, поэтому непробованное **не кончается никогда**, и до
    # возврата дело не доходит. Числа не сдвинулись ни на что: доля узлов степени
    # 3+ осталась 4 %, узлов «и за 2, и за 3 шага» — ноль.
    #
    # Поэтому условие не «есть ли что пробовать», а «дозамкнуто ли **здесь**».
    # Свежее место имеет степень 1 (только ребро, которым пришли) и потому
    # исследуется. Как только у него появились и вход, и выход — степень 2, то есть
    # ровно цепь, — разведка уходит замыкать, а не углубляться. Это и есть
    # «максимизировать прирост степени, а не число новых узлов».
    if fresh and here_degree < degree_bar:
        hit("непробованное здесь")
        return fresh[0], hold_ms

    # 3. Куда идти, чтобы поднять степень. Маршрут берётся из графа — это тот же
    # Дейкстра по измеренным секундам, которым ходит планировщик.
    for _deg, _untried, target in undersewn(graph, model, outputs, bar=degree_bar):
        if target == place:
            continue
        route = graph.route(place, target)
        if not route:
            continue
        first = parse_action_key(route[0].mode)
        if first is None:
            continue
        # Не разворачиваться немедленно: шаг, отменяющий последнее действие, ведёт
        # туда, откуда мы пришли, и ребра не добавляет.
        if (inverse is not None and last_output is not None
                and inverse.get(last_output) == first[0] and len(route) == 1):
            continue
        hit("идти в недозамкнутое")
        return first[0], first[1]

    # 4. Дозамкнутых целей нет или до них нет маршрута — тогда всё-таки вглубь.
    if fresh:
        hit("непробованное всё же")
        return fresh[0], hold_ms

    # 5 и 6 — как раньше: обратная пара, потом меньше всего пробованное.
    hit("прежняя разведка")
    return choose_probe(model, place, outputs, hold_ms=hold_ms, min_n=min_n,
                        inverse=inverse, last_output=last_output)


def probe_chooser(profile: Profile) -> Callable[..., tuple[str, int]]:
    """Какая разведка включена профилем. Переключатель структурный: он форкает журнал.

    Смешивать в одной ветке опыт двух разведок нельзя — у графа будет разная форма,
    и любое сравнение по нему станет сравнением двух разных экспериментов.
    """
    share = float(profile.structural["explore_closing_share"])
    if share <= 0.0:
        return choose_probe
    bar = int(profile.parameters["explore_close_degree_bar"])
    # Доля отрабатывается счётчиком, а не случайностью: прогон обязан
    # воспроизводиться от сида, и кривая баланса не должна зависеть от того, какому
    # генератору достался этот вызов. Накопитель тот же, что у темпа лепета.
    state = {"credit": 0.0, "closing": 0, "deep": 0}

    def chooser(model: ForwardModel, graph: Any, place: str | None,
                outputs: Sequence[str], **kw: Any) -> tuple[str, int]:
        state["credit"] += share
        if state["credit"] >= 1.0:
            state["credit"] -= 1.0
            state["closing"] += 1
            return choose_closing_probe(model, graph, place, outputs,
                                        degree_bar=bar, **kw)
        state["deep"] += 1
        kw.pop("graph", None)
        return choose_probe(model, place, outputs, **kw)

    chooser.counts = state          # type: ignore[attr-defined]
    return chooser


# ---------------------------------------------------------------------------
# Репетиция плана воображением
# ---------------------------------------------------------------------------


def rehearse(plan: Plan, model: ForwardModel, loop: Loop, start: str, *,
             stamp_of: Callable[[int], Stamp] | None = None) -> tuple[bool, str]:
    """Прогнать план через размыкатель: те же действия, эффекторы отключены.

    Зачем репетировать то, что уже посчитано поиском. Поиск работает с числами
    модели, а репетиция — с тем же самым циклом действия, через который идёт и
    настоящее действие. Если план невыразим действиями (шаг ведёт не туда, куда
    обещал поиск), это выяснится здесь, а не в мире. И записи `THOUGHT` в журнале
    появляются именно отсюда: продуманное отличимо от сделанного (инвариант 1).
    """
    place = start
    actions = [Action.key(s.output, s.duration_ms) for s in plan.steps]
    steps = loop.imagine(actions, stamp_of)
    for i, (ps, st) in enumerate(zip(plan.steps, steps)):
        if st.mode is Mode.ACT:
            raise PlanError("репетиция пошла с подключёнными эффекторами: "
                            "мысль стала бы поступком")
        key = action_key(ps.output, ps.duration_ms)
        pred = model.predict(place, key)
        if pred is None:
            return False, f"шаг {i}: модель разучилась предсказывать {key}"
        if pred.likely.dst != ps.expected_place:
            return False, (f"шаг {i}: поиск обещал {ps.expected_place}, "
                           f"модель говорит {pred.likely.dst}")
        place = pred.likely.dst
    return place == plan.target, ("репетиция сошлась" if place == plan.target
                                  else f"репетиция привела в {place}, а не в "
                                       f"{plan.target}")


# ---------------------------------------------------------------------------
# Исполнение с проверкой каждого шага
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Execution:
    """Чем кончилось применение плана. Все поля — наблюдения, не мнения."""

    plan: Plan
    steps_done: int = 0
    surprises: int = 0                    # шагов, где мир не совпал с моделью
    goal_passed: bool = False
    abandoned_at: int | None = None
    reason: str = ""
    places: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"goal_id": self.plan.goal_id, "steps_done": self.steps_done,
                "steps_total": self.plan.length, "surprises": self.surprises,
                "goal_passed": self.goal_passed, "abandoned_at": self.abandoned_at,
                "reason": self.reason, "places": list(self.places)}


def execute(plan: Plan, *, act: Callable[[Action], str],
            goal: Goal | None = None, journal: Journal | None = None,
            stamp_of: Callable[[int], Stamp] | None = None,
            model: ForwardModel | None = None,
            reason_id: str | None = None) -> Execution:
    """Применить план в мире, сверяя каждый шаг с предсказанием.

    `act` исполняет одно действие и возвращает место, в котором мир оказался. Шаг из
    навыка исполняется целиком, и сверяется только его итог: обещание модели было про
    конец цепочки. Первое расхождение с предсказанием прекращает исполнение: дальше план построен на
    состоянии, которого нет, и продолжать значит действовать вслепую.

    Расхождение — не поломка, а данные. Оно записывается в журнал и есть та же
    ошибка предсказания, что и везде: планировщик не отдельный механизм со своей
    валютой, он работает на общей.

    `reason_id` — ссылка на реплику, в которой планировщик объяснил, зачем он это
    делает (`state_reason`). Исполнение её не читает и не сравнивает: сопоставление
    заявленной причины с настоящим инициатором — работа метрик, а не контроллера
    (`ARCHITECTURE.md`, правила модулей). Здесь только проставляется ссылка.
    """
    ex = Execution(plan)
    for i, step in enumerate(plan.steps):
        # Шаг может быть цепочкой (навык из библиотеки). Тогда выполняются все
        # действия по порядку, а сверка с предсказанием — одна, после последнего:
        # модель обещала место в конце цепочки, а не после каждого её звена.
        place = ""
        for action in step.actions():
            place = act(action)
        ex.steps_done += 1
        ex.places.append(place)
        agreed = place == step.expected_place
        if not agreed:
            ex.surprises += 1
        if journal is not None and stamp_of is not None:
            journal.append(
                EntryKind.PLAN, stamp_of(i), Actor.AGENT, ActorLayer.PLANNER,
                state=StateSnapshot(goal_id=plan.goal_id,
                                    stated_reason_id=reason_id),
                event={"code": "plan_step", "goal": plan.goal_id, "index": i,
                       "expected": step.expected_place, "observed": place,
                       "agreed": agreed, "p": round(step.p, 4),
                       "actions": len(step.chain) or 1})
        if not agreed:
            ex.abandoned_at = i
            ex.reason = (f"шаг {i}: ожидалось {step.expected_place}, "
                         f"оказалось {place}. План брошен: дальше он опирается на "
                         "состояние, которого нет")
            plan.applied += 1
            return ex

    plan.applied += 1
    if goal is not None:
        ex.goal_passed = bool(goal.test())
    else:
        ex.goal_passed = bool(ex.places and ex.places[-1] == plan.target)
    if ex.goal_passed:
        plan.worked += 1
        ex.reason = "план дошёл до цели, тест цели пройден"
    else:
        ex.reason = ("план исполнен целиком, но тест цели не прошёл: модель ведёт "
                     "не туда, куда думает цель")
    return ex


# ---------------------------------------------------------------------------
# Полный круг: план → исполнение → расхождение → новый план
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Attempt:
    """Одна попытка добраться: план, исполнение и чем кончилось."""

    plan: Plan | None
    execution: Execution | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"plan": None if self.plan is None else self.plan.as_dict(),
                "execution": None if self.execution is None else self.execution.as_dict(),
                "reason": self.reason}


@dataclass(slots=True)
class Journey:
    """Чем кончилась попытка дойти до места. Все величины — наблюдения."""

    target: str
    attempts: list[Attempt] = field(default_factory=list)
    arrived: bool = False
    ticks: int = 0
    replans: int = 0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"target": self.target, "arrived": self.arrived,
                "attempts": [a.as_dict() for a in self.attempts],
                "tries": len(self.attempts), "replans": self.replans,
                "ticks": self.ticks, "reason": self.reason}


def travel(profile: Profile, goal: Goal, target: str, *,
           where: Callable[[], str | None],
           act: Callable[[Action], str],
           rebuild: Callable[[], ForwardModel],
           journal: Journal | None = None,
           stamp_of: Callable[[int], Stamp] | None = None) -> Journey:
    """Дойти до места: планировать, идти, при расхождении планировать заново.

    Здесь замыкается то, чего не хватало: модель мира предсказывает, планировщик
    строит цепочку, исполнение сверяет предсказание с миром, а расхождение не
    игнорируется и не сглаживается — оно прекращает план и заставляет строить новый
    по обновлённой модели. Бюджет цели тратится в её же циклах, и когда он кончился,
    цель бросается с причиной (правило 2 из `goals.py`).

    `where` возвращает место, в котором мы сейчас, или `None` при потере
    ориентации. `rebuild` пересобирает модель — после каждого расхождения она
    другая, потому что журнал вырос.
    """
    replan = bool(profile.parameters["plan_replan_on_surprise"])
    journey = Journey(target)

    while journey.ticks < goal.budget_ticks:
        here = where()
        if here is None:
            journey.reason = "потеря ориентации: планировать не от чего"
            return journey
        if here == target:
            journey.arrived = True
            journey.reason = "на месте"
            return journey

        model = rebuild()
        planner = Planner(profile, model)
        plan = planner.plan(goal, here, target)
        journey.ticks += 1
        if plan is None:
            journey.attempts.append(Attempt(None, None,
                                            "плана нет: из этого места модель не "
                                            "знает ни одного пути к цели"))
            journey.reason = ("плана нет. Это не поломка планировщика: из места, "
                              "из которого ни разу не уходили, вести некуда, и "
                              "правильный ответ — идти исследовать")
            return journey

        ex = execute(plan, act=act, goal=goal, journal=journal, stamp_of=stamp_of,
                     model=model)
        journey.ticks += ex.steps_done
        journey.attempts.append(Attempt(plan, ex, ex.reason))
        if ex.goal_passed:
            journey.arrived = True
            journey.reason = "дошёл"
            return journey
        if ex.surprises and not replan:
            journey.reason = ("расхождение с моделью, перепланирование выключено "
                              "профилем")
            return journey
        if ex.surprises:
            journey.replans += 1
            continue
        # План исполнен целиком, а тест не прошёл: модель ведёт не туда, куда
        # думает цель. Второй раз тот же план строить бессмысленно.
        journey.reason = ("план исполнен, тест цели не прошёл: модель и цель "
                          "расходятся в том, что считать целью")
        return journey

    journey.reason = f"бюджет цели кончился: {goal.budget_ticks} циклов"
    return journey


# ---------------------------------------------------------------------------
# Планировщик как контур
# ---------------------------------------------------------------------------


def contour(profile: Profile, model: ForwardModel, goal: Goal, start: str,
            target: str) -> Callable[[], Iterator[Progress]]:
    """Обёртка планировщика для `contours.Scheduler`.

    Планировщик — верхний контур (~0.5 Гц из архитектуры), и он обязан отдавать
    лучший ответ в любой момент. Здесь это выполняется буквально: каждое
    расширение — отдельный `yield`, и нижние контуры успевают вклиниться.
    """
    def start_work() -> Iterator[Progress]:
        planner = Planner(profile, model)
        yield from planner.search(goal, start, target)

    return start_work
