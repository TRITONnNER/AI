"""Драйвы, настроение, эмоция как модулятор порогов.

Инвариант 10 здесь главный: «самоотчёты агента ни на что не влияют
автоматически. Реакция только на поведение и объективные величины». Поэтому ни
одна величина в этом модуле не читает ни слова агента о себе. Всё выводится из
трёх объективных источников:

1. ошибка предсказания и её статистика,
2. состояние ресурсов,
3. состояние собственной памяти (сколько непроверенного, сколько неоткатываемого).

Эмоция — не таблица «страх → горизонт три секунды». Таблица была бы
захардкоженной константой поведения, а по правилам их быть не должно. Вместо неё
непрерывная модуляция **семи** осей от валентности и возбуждения, с
коэффициентами из профиля. Название эмоции («досада», «скука») существует только
как ярлык для исследователя и ни на что не влияет: его не читает ни одна ветка
кода, и это проверяется тестом.

Аллостаз, а не гомеостаз: драйв смотрит на горизонт `drive_horizon_s` вперёд и
реагирует на прогноз, а не на текущее отклонение. Иначе агент всегда опаздывает.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from ..core.profile import Profile

# Шесть драйвов — те же, что в пульте. Набор закрыт: новый драйв меняет форму
# мотивации, то есть это структурное изменение, а не добавление строки.
DRIVE_NAMES = ("integrity", "energy", "curiosity", "competence", "order", "social")


class DriveOrigin(StrEnum):
    """Откуда взялся драйв. Различие обязательное, а не описательное.

    `AUDIT.md`, часть 2 нашла здесь расхождение имени с механизмом: `MIND.md` и
    `ROADMAP.md` (М6) требуют, чтобы драйвы **обнаруживались** через корреляцию
    областей интерфейса с пережитым, а в коде шесть драйвов заданы литералами. Сама
    машинерия при этом верная — аллостаз, обновление от объективных величин, — но
    **набор задан, а не выведен**.

    Решение по этому расхождению — третий исход из трёх возможных: не «код прав» и не
    «переписать М6 сейчас», а **два механизма перестают делить одно имя**. Драйв,
    заданный исследователем, и драйв, выведенный из корреляций, — разные вещи, и
    запись обязана их различать. Поэтому происхождение — поле, а не комментарий:
    доля выведенных драйвов становится числом в отчёте (сейчас 0 из 6) и не может
    молча выглядеть как выполненное требование.

    Ровно так же устроено происхождение убеждений (инвариант 7): утверждение без
    происхождения в хранилище не попадает.
    """

    GIVEN = "given"            # задан исследователем: заглушка до М6
    DISCOVERED = "discovered"  # выведен из корреляции интерфейса с пережитым

# Ярлыки эмоций. Набор закрыт и существует только для человека. Он объявлен
# константой не для того, чтобы по нему что-то ветвилось, а наоборот: самоотчёт
# использует его как словарь допустимых слов, и всё, чего в словаре нет, считается
# утечкой надписи. Тест проверяет и то, что `emotion_label` не возвращает ничего,
# кроме этих пяти, и то, что ни одна ветка кода с ними не сравнивается.
EMOTIONS = ("страх", "досада", "удовлетворение", "скука", "любопытство")


@dataclass(slots=True)
class Drive:
    """Один драйв: где он сейчас, куда его тянет и что будет через горизонт."""

    name: str
    value: float
    target: float
    corridor: float = 0.15          # полуширина коридора гомеостаза
    forecast: float = 0.0           # где окажется через горизонт, если не вмешиваться
    # Происхождение обязательно и по умолчанию `given`: драйв, про который не
    # сказано, откуда он взялся, — это заданный драйв, притворяющийся выведенным.
    origin: DriveOrigin = DriveOrigin.GIVEN
    # Чем именно он выведен, если выведен. У заданного — пусто, и это единственная
    # законная пустота: у выведенного здесь стоит область интерфейса, с которой
    # нашлась корреляция, и без неё он не собирается.
    grounded_in: str | None = None

    def __post_init__(self) -> None:
        if self.origin is DriveOrigin.DISCOVERED and not self.grounded_in:
            raise ValueError(
                f"драйв {self.name!r} объявлен выведенным, но не сказано, из чего. "
                "Выведенный драйв без источника корреляции неотличим от заданного, "
                "а вся разница между М6 и заглушкой — именно в этом")

    @property
    def deficit(self) -> float:
        """Насколько драйв вне коридора. Ноль — внутри, положительное — снаружи."""
        d = abs(self.value - self.target)
        return max(0.0, d - self.corridor)

    @property
    def forecast_deficit(self) -> float:
        """То же, но по прогнозу: аллостаз реагирует на это, а не на текущее."""
        d = abs(self.forecast - self.target)
        return max(0.0, d - self.corridor)

    @property
    def in_corridor(self) -> bool:
        return self.deficit == 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": round(self.value, 4),
                "target": self.target, "corridor": self.corridor,
                "forecast": round(self.forecast, 4),
                "deficit": round(self.deficit, 4),
                "forecast_deficit": round(self.forecast_deficit, 4),
                "origin": str(self.origin), "grounded_in": self.grounded_in}


@dataclass(frozen=True, slots=True)
class Mood:
    """Настроение: валентность и возбуждение, оба в [−1, 1].

    Не «эмоция агента», а два числа, выведенных из объективных величин. Знак
    валентности — направление ошибки предсказания относительно её обычного
    уровня: мир идёт лучше ожидаемого или хуже. Возбуждение — изменчивость.
    """

    valence: float = 0.0
    arousal: float = 0.0

    def blend(self, target: Mood, inertia: float) -> Mood:
        k = 1.0 - max(0.0, min(1.0, inertia))
        return Mood(_clip(self.valence + (target.valence - self.valence) * k),
                    _clip(self.arousal + (target.arousal - self.arousal) * k))

    def as_dict(self) -> dict[str, float]:
        return {"valence": round(self.valence, 4), "arousal": round(self.arousal, 4)}


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def emotion_label(mood: Mood, *, error_high: bool = False) -> str:
    """Ярлык для исследователя. Ни на что не влияет и никем не читается.

    Существует ровно для того, чтобы человек, глядя на пульт, понимал, что
    происходит. Любая попытка завязать поведение на это имя — нарушение
    инварианта 10 в обход: поведение обязано зависеть от чисел, а не от слова.
    """
    if error_high and mood.arousal > 0.5:
        return "страх"
    if mood.arousal > 0.3 and mood.valence < -0.1:
        return "досада"
    if mood.valence > 0.25:
        return "удовлетворение"
    if mood.arousal < -0.2:
        return "скука"
    return "любопытство"


#: Семь осей модуляции из `MIND.md`. Набор закрыт и объявлен здесь, потому что
#: «сколько осей доходит до поведения» — величина, которую надо считать, а не
#: пересчитывать глазами по коду.
#:
#: Ось живёт, только если её читает потребитель. Прежняя редакция считала три оси
#: реализованными, но `modulation()` вызывался ровно в одном месте — в печати отчёта,
#: — а планировщик и лепет брали порог из профиля напрямую. То есть эмоция не
#: модулировала ничего, и три «реализованные» оси были мёртвым кодом. Поймал это не
#: разбор кода, а поиск потребителей: сам по себе модуль читался правильно.
AXES = ("attention_windows", "horizon_s", "caution_threshold", "explore_rate",
        "reflex_priority", "task_resource_share", "return_inertia")

#: Кто читает каждую ось. `None` — потребителя нет, ось объявлена и мертва.
#:
#: Таблица нужна затем, что «ось реализована» и «ось влияет на поведение» —
#: разные утверждения, и первое без второго уже один раз обмануло аудит: три оси
#: считались реализованными, а `modulation()` вызывался только при печати отчёта.
#: Здесь названо, кто именно читает; тест на каждую названную ось предъявляет сдвиг
#: числа в поведении потребителя (инвариант 25), а ось без потребителя докладывается
#: как незакрытая — числом «осей с потребителем: N из 7», а не молчанием.
CONSUMERS: Mapping[str, str | None] = {
    # Нарезка окон внимания существует (`vision.layers.windows`), но её никто не
    # вызывает: внимание как контур — часть М5. Ось объявлена, потребителя нет.
    "attention_windows": None,
    "horizon_s": "behaviour.planner.Planner.plan (глубина поиска)",
    "caution_threshold": "behaviour.babbling.Babbler.next_probe",
    "explore_rate": "behaviour.babbling.Babbler.next_probe (темп)",
    "reflex_priority": "behaviour.contours.Scheduler (право перебивать)",
    "task_resource_share": "core.resources.ResourceGovernor.admit",
    "return_inertia": "behaviour.goals.GoalStack.push (бюджет цели)",
}


def axes_with_consumer() -> tuple[str, ...]:
    """Оси, у которых есть кто читающий. Остальные — заявленные и мёртвые."""
    return tuple(a for a in AXES if CONSUMERS.get(a))


@dataclass(frozen=True, slots=True)
class Modulation:
    """Что эмоция сделала с порогами. Все семь — производные, не настройки.

    Семь осей — требование `MIND.md`: ширина внимания, длина горизонта, порог
    необратимости, темп исследования, право рефлексов перебивать планировщик, доля
    ресурсов на задачу, инерция возврата. Каждая здесь непрерывна и выведена из
    валентности и возбуждения с коэффициентом из профиля; таблицы «страх → горизонт
    три секунды» нет и быть не может, она была бы захардкоженной константой
    поведения.

    Базовые значения хранятся рядом со сдвинутыми не для красоты отчёта: без них
    нельзя отличить «ось не сдвинулась, потому что настроение ровное» от «ось не
    сдвинулась, потому что её никто не считает».
    """

    attention_windows: int
    horizon_s: float
    caution_threshold: float
    explore_rate: float
    reflex_priority: float
    task_resource_share: float
    return_inertia: float
    base: Mapping[str, float] = field(default_factory=dict)

    @property
    def babble_rate(self) -> float:
        """Прежнее имя темпа исследования. Лепет — его единственный потребитель."""
        return self.explore_rate

    def value(self, axis: str) -> float:
        if axis not in AXES:
            raise KeyError(f"нет оси {axis!r}; есть {AXES}")
        return float(getattr(self, axis))

    def shifted(self) -> tuple[str, ...]:
        """Оси, которые настроение действительно сдвинуло. Пустой набор — тоже ответ."""
        return tuple(a for a in AXES
                     if abs(self.value(a) - float(self.base.get(a, 0.0))) > 1e-9)

    def as_dict(self) -> dict[str, Any]:
        digits = {"attention_windows": 0, "horizon_s": 3}
        return {
            **{a: (int(self.value(a)) if a == "attention_windows"
                   else round(self.value(a), digits.get(a, 4))) for a in AXES},
            "changed": {a: round(self.value(a) - float(self.base.get(a, 0.0)),
                                 digits.get(a, 4)) for a in AXES},
            "shifted": list(self.shifted()),
        }


@dataclass(slots=True)
class Motivation:
    """Драйвы, настроение и модуляция вместе. Обновляется объективными величинами."""

    profile: Profile
    drives: dict[str, Drive] = field(default_factory=dict)
    mood: Mood = Mood()
    ticks: int = 0
    _error_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.drives:
            self.drives = given_drives()

    @property
    def discovered(self) -> int:
        """Сколько драйвов выведено из опыта. До М6 — ноль, и это видно в отчёте."""
        return sum(1 for d in self.drives.values()
                   if d.origin is DriveOrigin.DISCOVERED)

    # --- обновление ---------------------------------------------------------

    def update(self, *, error_mean: float, error_sigma: float, error_now: float,
               load: float = 0.0, unknown_reversibility: int = 0,
               hearsay: int = 0, beliefs: int = 0, testimonies: int = 0) -> Mood:
        """Один такт контура драйвов. Все аргументы — объективные величины.

        `load` — доля занятых ресурсов, `unknown_reversibility` — сколько выходов
        с неизвестной обратимостью, `hearsay` — сколько убеждений держится только
        на чужих словах.
        """
        p = self.profile.parameters
        rate = float(p["drive_return_rate"])
        horizon_ticks = max(1.0, float(p["drive_horizon_s"]) * float(p["hz_drives"]))

        self.ticks += 1
        self._error_history.append(error_now)
        if len(self._error_history) > 256:
            self._error_history.pop(0)

        # Целевая валентность: мир идёт лучше обычного — плюс, хуже — минус.
        # Нормируем на разброс, потому что «хуже обычного» зависит от сцены.
        if error_sigma > 1e-9:
            target_valence = _clip((error_mean - error_now) / (3.0 * error_sigma))
        else:
            target_valence = 0.0
        volatility = _volatility(self._error_history)
        target_arousal = _clip(volatility / max(1e-9, error_mean + error_sigma) - 0.35)

        self.mood = self.mood.blend(Mood(target_valence, target_arousal),
                                    float(p["mood_inertia"]))

        # Драйвы: каждый тянется к своей цели и толкается объективной величиной.
        pushes = {
            "energy": -load * 0.02,
            # Скучно — любопытство растёт; удивляет постоянно — падает.
            "curiosity": (error_mean - error_now) * 0.5,
            # Компетентность растёт, когда ошибка падает со временем.
            "competence": _trend(self._error_history) * -2.0,
            # Целостность падает от того, чего я не умею откатывать.
            "integrity": -0.004 * min(10, unknown_reversibility),
            # Порядок падает от непроверенного чужого.
            "order": -0.01 * (hearsay / max(1, beliefs)) if beliefs else 0.0,
            "social": 0.01 * min(5, testimonies) - 0.002,
        }
        for name, drive in self.drives.items():
            push = pushes.get(name, 0.0)
            drive.value = _clip(drive.value + (drive.target - drive.value) * rate + push,
                                0.02, 1.0)
            # Прогноз: линейная экстраполяция того же движения на горизонт.
            step = (drive.target - drive.value) * rate + push
            drive.forecast = _clip(drive.value + step * horizon_ticks, 0.0, 1.0)
        return self.mood

    # --- следствия ----------------------------------------------------------

    def modulation(self, *, error_high: bool = False) -> Modulation:
        """Пороги после модуляции настроением. Семь осей, все читаемые."""
        p = self.profile.parameters
        base: dict[str, float] = {
            "attention_windows": float(p["attention_windows"]),
            "horizon_s": float(p["drive_horizon_s"]),
            "caution_threshold": float(p["irreversibility_threshold"]),
            "explore_rate": float(p["babble_rate"]),
            "reflex_priority": float(p["emotion_reflex_priority_base"]),
            "task_resource_share": float(p["emotion_task_share_base"]),
            # Инерция возврата — это бюджет тиков на цель, и он выводится там же,
            # где его считает GoalStack: горизонт на частоту навыков. Дублировать
            # формулу нельзя, но и импортировать поведение в модель нельзя, поэтому
            # тест сверяет, что два места дают одно число.
            "return_inertia": float(
                max(1, int(float(p["drive_horizon_s"]) * float(p["hz_skills"])))),
        }
        if not bool(self.profile.structural["emotion_enabled"]):
            # Выключено — значит все семь равны базе. Не «ноль» и не «None»: ось
            # существует, просто настроение её не двигает, и `shifted()` пуст.
            return Modulation(int(base["attention_windows"]), base["horizon_s"],
                              base["caution_threshold"], base["explore_rate"],
                              base["reflex_priority"], base["task_resource_share"],
                              base["return_inertia"], base=base)

        a, v = self.mood.arousal, self.mood.valence
        neg_v, pos_a = max(0.0, -v), max(0.0, a)

        # Возбуждение сужает внимание: при быстрых изменениях следить за пятью
        # местами разом дороже, чем за одним. Округление вниз, но не ниже одного
        # окна: ноль окон — это не «узкое внимание», это слепота.
        windows = max(1, int(base["attention_windows"]
                             * (1.0 - float(p["emotion_attention_gain"]) * pos_a)))
        # Возбуждение сжимает горизонт: когда всё быстро меняется, планировать на
        # минуту вперёд бессмысленно.
        horizon = base["horizon_s"] * (1.0 - float(p["emotion_horizon_gain"]) * pos_a)
        # Отрицательная валентность поднимает осторожность: дела идут хуже
        # ожидаемого — цена ошибки выше.
        caution = _clip(base["caution_threshold"]
                        + float(p["emotion_caution_gain"]) * neg_v, 0.0, 1.0)
        # Скука ускоряет исследование, страх тормозит.
        explore = base["explore_rate"] * (1.0 + float(p["emotion_babble_gain"])
                                          * max(0.0, -a))
        explore *= (1.0 - (0.8 if error_high else 0.0))
        # Возбуждение отдаёт право хода рефлексам: чем быстрее мир, тем меньше
        # смысла ждать планировщик. Это не отключение планировщика, а порог, ниже
        # которого нижний контур не перебивает верхний.
        reflex = _clip(base["reflex_priority"]
                       + float(p["emotion_reflex_gain"]) * pos_a, 0.0, 1.0)
        # Плохо идущие дела сужают долю ресурсов на одну задачу: когда прогноз
        # рушится, вкладываться в одну ставку дороже.
        share = _clip(base["task_resource_share"]
                      - float(p["emotion_task_share_gain"]) * neg_v, 0.05, 1.0)
        # Инерция возврата: при высоком возбуждении цель бросается быстрее. Это
        # бюджет тиков, поэтому множитель, а не сдвиг.
        inertia = max(1.0, base["return_inertia"]
                      * (1.0 - float(p["emotion_inertia_gain"]) * pos_a))
        return Modulation(windows, max(0.5, horizon), caution,
                          # Потолок 4, а не 1: темп — число проб за такт, и скука
                          # обязана уметь ускорять, а не только упираться в базу.
                          _clip(explore, 0.0, 4.0), reflex, share, inertia,
                          base=base)

    def dominant(self) -> Drive:
        """Какой драйв сильнее всего требует внимания — по прогнозу, не по сейчас."""
        return max(self.drives.values(),
                   key=lambda d: (d.forecast_deficit, d.deficit, d.name))

    def goal_pressure(self) -> dict[str, float]:
        """Давление каждого драйва с весами из профиля. Из этого рождаются цели."""
        p = self.profile.parameters
        wc = float(p["weight_curiosity"])
        wh = float(p["weight_homeostasis"])
        out: dict[str, float] = {}
        for name, d in self.drives.items():
            w = wc if name == "curiosity" else wh
            out[name] = round(w * (0.5 * d.deficit + 0.5 * d.forecast_deficit), 6)
        return out

    def as_dict(self, *, error_high: bool = False) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "mood": self.mood.as_dict(),
            "emotion_label": emotion_label(self.mood, error_high=error_high),
            "drives": [d.as_dict() for d in self.drives.values()],
            # Число, из-за которого расхождение больше не может быть незаметным.
            "discovered": f"{self.discovered} из {len(self.drives)}",
            "dominant": self.dominant().name,
            "pressure": self.goal_pressure(),
            "modulation": self.modulation(error_high=error_high).as_dict(),
        }


def _volatility(xs: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    diffs = [abs(b - a) for a, b in zip(xs, xs[1:])]
    return sum(diffs) / len(diffs)


def _trend(xs: list[float], window: int = 32) -> float:
    """Наклон: падает ошибка или растёт. Знак важнее величины."""
    tail = xs[-window:]
    if len(tail) < 4:
        return 0.0
    half = len(tail) // 2
    first = sum(tail[:half]) / half
    second = sum(tail[half:]) / (len(tail) - half)
    return second - first


def given_drives() -> dict[str, Drive]:
    """Шесть заданных драйвов — заглушка на месте М6, и названа заглушкой.

    Это не «молчаливая заглушка» в запрещённом смысле: она не возвращает
    правдоподобное значение вместо настоящего, она возвращает **честно помеченное**
    значение. Разница в том, что помеченное считается: `Motivation.discovered`
    вернёт ноль, и отчёт скажет «выведено 0 из 6», а не промолчит.

    Значения и коридоры — стартовая точка, а не утверждение о человеке. Когда М6
    появится, набор будет строиться из корреляций, и эта функция останется только
    для контрольных прогонов «а что если набор дать заранее» — то есть станет
    ablation, каким и должна быть.
    """
    return {
        "integrity": Drive("integrity", 0.86, 0.85),
        "energy": Drive("energy", 0.62, 0.62),
        "curiosity": Drive("curiosity", 0.60, 0.60, corridor=0.2),
        "competence": Drive("competence", 0.55, 0.74),
        "order": Drive("order", 0.50, 0.50, corridor=0.2),
        "social": Drive("social", 0.34, 0.34, corridor=0.25),
    }


def from_profile(profile: Profile, overrides: Mapping[str, float] | None = None) -> Motivation:
    m = Motivation(profile)
    for name, value in (overrides or {}).items():
        if name not in m.drives:
            raise KeyError(f"нет драйва {name!r}; есть {DRIVE_NAMES}")
        m.drives[name].value = _clip(float(value), 0.02, 1.0)
    return m
