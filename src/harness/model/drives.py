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
непрерывная модуляция трёх порогов от валентности и возбуждения, с
коэффициентами из профиля. Название эмоции («досада», «скука») существует только
как ярлык для исследователя и ни на что не влияет: его не читает ни одна ветка
кода, и это проверяется тестом.

Аллостаз, а не гомеостаз: драйв смотрит на горизонт `drive_horizon_s` вперёд и
реагирует на прогноз, а не на текущее отклонение. Иначе агент всегда опаздывает.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..core.profile import Profile

# Шесть драйвов — те же, что в пульте. Набор закрыт: новый драйв меняет форму
# мотивации, то есть это структурное изменение, а не добавление строки.
DRIVE_NAMES = ("integrity", "energy", "curiosity", "competence", "order", "social")

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
                "forecast_deficit": round(self.forecast_deficit, 4)}


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


@dataclass(frozen=True, slots=True)
class Modulation:
    """Что эмоция сделала с порогами. Все три — производные, не настройки."""

    horizon_s: float
    caution_threshold: float
    babble_rate: float
    base_horizon_s: float
    base_caution: float
    base_babble: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon_s": round(self.horizon_s, 3),
            "caution_threshold": round(self.caution_threshold, 4),
            "babble_rate": round(self.babble_rate, 4),
            "changed": {
                "horizon_s": round(self.horizon_s - self.base_horizon_s, 3),
                "caution_threshold": round(self.caution_threshold - self.base_caution, 4),
                "babble_rate": round(self.babble_rate - self.base_babble, 4),
            },
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
            self.drives = {
                "integrity": Drive("integrity", 0.86, 0.85),
                "energy": Drive("energy", 0.62, 0.62),
                "curiosity": Drive("curiosity", 0.60, 0.60, corridor=0.2),
                "competence": Drive("competence", 0.55, 0.74),
                "order": Drive("order", 0.50, 0.50, corridor=0.2),
                "social": Drive("social", 0.34, 0.34, corridor=0.25),
            }

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
        """Пороги после модуляции настроением."""
        p = self.profile.parameters
        base_h = float(p["drive_horizon_s"])
        base_c = float(p["irreversibility_threshold"])
        base_b = float(p["babble_rate"])
        if not bool(self.profile.structural["emotion_enabled"]):
            return Modulation(base_h, base_c, base_b, base_h, base_c, base_b)

        hg = float(p["emotion_horizon_gain"])
        cg = float(p["emotion_caution_gain"])
        bg = float(p["emotion_babble_gain"])
        a, v = self.mood.arousal, self.mood.valence

        # Возбуждение сжимает горизонт: когда всё быстро меняется, планировать на
        # минуту вперёд бессмысленно.
        horizon = base_h * (1.0 - hg * max(0.0, a))
        # Отрицательная валентность поднимает осторожность: дела идут хуже
        # ожидаемого — цена ошибки выше.
        caution = _clip(base_c + cg * max(0.0, -v), 0.0, 1.0)
        # Скука ускоряет лепет, страх тормозит.
        babble = base_b * (1.0 + bg * max(0.0, -a)) * (1.0 - (0.8 if error_high else 0.0))
        return Modulation(max(0.5, horizon), caution, _clip(babble, 0.0, 1.0),
                          base_h, base_c, base_b)

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


def from_profile(profile: Profile, overrides: Mapping[str, float] | None = None) -> Motivation:
    m = Motivation(profile)
    for name, value in (overrides or {}).items():
        if name not in m.drives:
            raise KeyError(f"нет драйва {name!r}; есть {DRIVE_NAMES}")
        m.drives[name].value = _clip(float(value), 0.02, 1.0)
    return m
