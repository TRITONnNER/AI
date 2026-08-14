"""Режим среды: четыре признака, определяемые наблюдением. TASK-32 A, TASK-33 B.

Кросс-доменный замер дал разброс IoU 0.196–0.815 и читался как «метод где-то работает
хуже». Читался неверно. **Параллакс в мире без камеры не работает хуже — он неприменим**:
общего сдвига там нет по построению, и число, которое он всё равно выдаёт, — не плохой
ответ, а ответ не на тот вопрос. Смешивать «плохо» и «не о том» нельзя: первое чинится
порогом, второе — только отказом отвечать.

## Четыре признака, и все четыре наблюдаемы

Ни один не сообщается агенту и ни один не берётся из истины мира.

1. **Эго-движение** (`EGO_MOTION`) — существует выход, повторяемо дающий согласованный
   сдвиг всего кадра больше, чем бывает при бездействии. Множитель в схеме
   (`regime_ego_over_idle`).

   **Признак наблюдательный, и потому он шире, чем «у меня есть камера».** Замер поймал
   это на проигрывателе видео: перемотка translates всю картинку, и признак отвечает «да»
   там, где камеры нет вовсе. Это не ошибка детектора — параллаксу нужно именно то, что
   признак и проверяет: чтобы мои действия давали согласованный сдвиг. Различить «двигаю
   себя» и «двигаю содержимое согласованно» наблюдением нельзя, и притворяться, что можно,
   было бы подсказкой.
2. **Мир идёт без меня** (`WORLD_ALONE`) — при полном бездействии кадр меняется, и меняется
   **сравнимо** с тем, как от моих действий. Две мерки, обе из схемы: абсолютный пол
   (`regime_alone_margin`) и доля от вызванного мной изменения (`regime_alone_ratio`).

   Одной абсолютной мерки не хватило, и это тоже нашёл замер: у всех пяти доменов
   обрамление анимировано на 10–20 % кадра, поэтому по абсолютной мерке «мир идёт сам»
   находилось везде, включая игру, где сам мир не делает ничего.
3. **Обратимость** (`REVERSIBLE`) — существует действие с сигнатурой «возвращает
   состояние»: после него вид совпадает с видом до предыдущего действия. Это тот же
   поиск обратных пар, которым занимается лепет, и здесь он ровно так же наблюдателен.
4. **Внешний судья** (`JUDGE`) — определяется ли успех наблюдением. **Здесь и сейчас этот
   признак не определяется, и это сказано полем, а не тишиной**: у всех пяти синтетических
   доменов канала оценки нет вовсе, поэтому детектор, который всегда отвечает «нет», не
   проверен ничем (инвариант 27 — замер вакуумен по свойству мира). Признак объявлен,
   умеет быть `None` = «не определено», и станет измеримым, когда появится домен с каналом
   оценки — это направление D той же задачи.

## Почему признак — не настройка

Настройка «в этом мире есть камера» была бы подсказкой (инвариант 4): агент обязан
выяснить это сам. Поэтому режим **определяется прогоном**, а его результат — запись с `n`
и происхождением, как у любого другого убеждения.

## Опорный уровень домена (TASK-33 B)

Все три ошибки первой редакции детектора были одной ошибкой — **сравнением с нулём**.
Поэтому опорный уровень здесь не число в сравнении, а измеряемый объект: `Baseline`,
снятый с записи неподвижности, с `n` и единицей. У каждого признака в `Evidence.reference`
сказано, **с чем именно сравнивали**.

**Нет записи неподвижности — нет ответа.** Признак, которому фон нужен, отвечает «не
определено», а не подбирает порог: порог, снятый с другого домена, был бы подсказкой об
этом (инвариант 4). То же и при слишком короткой записи (`regime_baseline_min_frames`):
одна пара кадров даёт медиану, равную самой себе, и разброс, равный нулю.

**И «не определено» появилось там, где раньше молча стояло «нет».** Если ни одно действие
не изменило вид, возвращать было нечего — это «проверить было нечем», а не «обратимых
действий нет». Прежняя редакция отвечала «нет», и на двух доменах из пяти этот ответ
случайно совпадал с истиной: проверка выглядела безошибочной (0 из 5) за счёт клеток,
которые она вообще не измеряла.

## Что даёт режим

`applicable(mechanism, regime)` отвечает, применим ли механизм, и **чего именно не
хватает**, если нет. Механизм, объявленный неприменимым, не выдаёт числа: он докладывает
«неприменимо» и называет отсутствующий признак. Это не отказ от измерения, а отказ от
подмены: число, полученное там, где вопрос не имеет смысла, портит сводку по всем доменам
разом — именно так разброс 0.196–0.815 и получился.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from ..vision.selfworld import estimate_global_shift

#: Признаки режима. Набор закрыт: пятый признак — это изменение схемы записи, а не
#: строка на месте вызова.
EGO_MOTION = "эго-движение"
WORLD_ALONE = "мир идёт без меня"
REVERSIBLE = "обратимость"
JUDGE = "внешний судья"
FEATURES = (EGO_MOTION, WORLD_ALONE, REVERSIBLE, JUDGE)

#: Механизм → какие признаки ему нужны, и почему. Таблица объявлена здесь, потому что
#: «механизм неприменим» — утверждение о механизме, а не о домене, и жить оно должно
#: рядом с механизмом, а не в скрипте разбора.
MECHANISMS: dict[str, dict[str, Any]] = {
    "параллакс": {
        "needs": (EGO_MOTION, f"не {WORLD_ALONE}"),
        "why": "признак отделяет то, что сместилось вместе со всем кадром. Нужны **два**"
               " условия, и второе выяснилось замером: мало того, чтобы мои действия "
               "давали сдвиг, — надо ещё, чтобы мир не ехал сам между кадрами. Иначе "
               "сдвиг между двумя кадрами вызван не мной, а признак всё равно приписывает "
               "его моему движению. Именно так проигрыватель видео и дал IoU 0.196",
    },
    "граф мест": {
        "needs": (EGO_MOTION,),
        "why": "узел графа — вид места, ребро — переход между видами. Если вид не "
               "меняется от моих действий, все наблюдения сложатся в одно место, и граф "
               "будет говорить «я всё время здесь» — верно и бесполезно",
    },
    "тау": {
        "needs": (EGO_MOTION,),
        "why": "время до контакта считается по скорости расширения приближающегося. Без "
               "эго-движения ничто не приближается **из-за меня**, и тау мерит движение "
               "содержимого, выдавая его за сближение",
    },
    "неподвижность": {
        "needs": (WORLD_ALONE,),
        "why": "признак отделяет то, что не меняется, когда меняется остальное. В мире, "
               "который сам ничего не делает, «остальное» не меняется тоже, и признаку "
               "нечего противопоставить неподвижному",
    },
    "оценка обратимости": {
        "needs": (REVERSIBLE,),
        "why": "осторожность определена как «не умею откатить». Там, где откатить нельзя "
               "ничего, величина вырождается в константу и решений не меняет",
    },
    "цель с внешним судьёй": {
        "needs": (JUDGE,),
        "why": "тест такой цели — предсказание чужой оценки. Нет канала оценки — нет и "
               "того, что предсказывать",
    },
}


class RegimeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Baseline:
    """Опорный уровень домена: что бывает, когда агент не делает ничего.

    Все три ошибки первой редакции детектора были одной и той же ошибкой — **сравнением с
    нулём**. Признак спрашивал «сдвиг больше нуля?», «кадр меняется?», «вид совпал точно?»
    — и в домене, который идёт сам, каждый из этих вопросов отвечает «да» без всякого
    участия агента. Поэтому опорный уровень здесь — отдельный измеряемый объект, а не
    число, вписанное в сравнение: у него есть `n`, единица и происхождение, как у всякого
    убеждения.

    `frames == 0` означает «записи неподвижности нет». Тогда признаки, которым она нужна,
    отвечают «не определено» — и это **не** «нет». Подобрать порог вместо отсутствующего
    фона нельзя: это и было бы подсказкой о домене, которую агент не выяснял.
    """

    frames: int                 # сколько пар кадров без действия
    diff_med: float             # медианное изменение кадра при бездействии
    diff_min: float             # самая тихая пара: тише в этом домене не бывает
    diff_max: float
    shift_med: float            # медианный общий сдвиг при бездействии, px
    shift_max: float
    sim_med: float              # сходство отпечатков двух соседних кадров без действия
    sim_min: float
    #: Сходство отпечатков через **тот же промежуток**, что у проверки обратимости: между
    #: видом до действия и видом после двух действий проходит два шага мира. Мерка через
    #: один шаг была бы строже настоящего фона ровно на один шаг дрейфа — и в едущем
    #: домене «вид вернулся» не находилось бы никогда.
    sim_gap_med: float = 1.0
    sim_gap_min: float = 1.0
    gap: int = 2

    @property
    def present(self) -> bool:
        return self.frames > 0

    def enough(self, min_frames: int) -> bool:
        """Хватает ли записи. Мало кадров — это тоже «не определено», а не «фон нулевой».

        Одна пара кадров даёт медиану, равную самой себе, и разброс, равный нулю: по такой
        записи любой признак «подтверждается» с уверенностью, которой нет.
        """
        return self.frames >= max(1, int(min_frames))

    def as_dict(self) -> dict[str, Any]:
        return {"frames": self.frames, "unit": "пара кадров без действия",
                "diff_med": round(self.diff_med, 6),
                "diff_min": round(self.diff_min, 6),
                "diff_max": round(self.diff_max, 6),
                "shift_med": round(self.shift_med, 3),
                "shift_max": round(self.shift_max, 3),
                "sim_med": round(self.sim_med, 4),
                "sim_min": round(self.sim_min, 4),
                "sim_gap_med": round(self.sim_gap_med, 4),
                "sim_gap_min": round(self.sim_gap_min, 4),
                "gap": self.gap}


#: Пустая запись неподвижности: домен, где бездействие наблюдать не пришлось.
NO_BASELINE = Baseline(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2)

#: Какому признаку нужен опорный уровень. `JUDGE` его не требует: канал чужой оценки — это
#: не измерение кадра, и фон тут не при чём. Различие объявлено таблицей, а не тем, что до
#: одного из признаков проверка не дотянулась.
NEEDS_BASELINE = {EGO_MOTION: True, WORLD_ALONE: True, REVERSIBLE: True, JUDGE: False}


@dataclass(frozen=True, slots=True)
class Evidence:
    """Один признак: ответ, на чём он основан и сколько наблюдений за ним.

    `value is None` — «не определено», и это **не** то же самое, что `False`. «Обратимых
    действий нет» и «обратимость проверить было нечем» ведут к разным решениям: в первом
    случае механизм неприменим, во втором неизвестно, применим ли, и это надо доизмерить.

    `reference` — с чем именно сравнивали, словами и числом. Признак без опорного уровня
    сравнивался бы с нулём, а это и есть та ошибка, из-за которой эго-движение находилось
    в проигрывателе видео.
    """

    feature: str
    value: bool | None
    n: int
    unit: str
    detail: str
    reference: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "value": self.value, "n": self.n,
                "unit": self.unit, "detail": self.detail,
                "reference": self.reference}


@dataclass(slots=True)
class Regime:
    """Режим среды: четыре признака с основаниями и опорный уровень домена."""

    evidence: dict[str, Evidence] = field(default_factory=dict)
    baseline: Baseline = NO_BASELINE

    def add(self, ev: Evidence) -> Evidence:
        if ev.feature not in FEATURES:
            raise RegimeError(f"нет такого признака режима: {ev.feature!r}")
        self.evidence[ev.feature] = ev
        return ev

    def has(self, feature: str) -> bool | None:
        if feature not in FEATURES:
            raise RegimeError(f"нет такого признака режима: {feature!r}")
        ev = self.evidence.get(feature)
        return None if ev is None else ev.value

    def as_dict(self) -> dict[str, Any]:
        return {f: (self.evidence[f].as_dict() if f in self.evidence else None)
                for f in FEATURES}


@dataclass(frozen=True, slots=True)
class Verdict:
    """Применим ли механизм. `applicable is None` — «неизвестно», а не «нет»."""

    mechanism: str
    applicable: bool | None
    missing: tuple[str, ...]
    undetermined: tuple[str, ...]

    @property
    def line(self) -> str:
        if self.applicable:
            return f"{self.mechanism}: применим"
        if self.applicable is None:
            return (f"{self.mechanism}: неизвестно, применим ли — не определено: "
                    f"{', '.join(self.undetermined)}")
        return f"{self.mechanism}: неприменимо — нет: {', '.join(self.missing)}"

    def as_dict(self) -> dict[str, Any]:
        return {"mechanism": self.mechanism, "applicable": self.applicable,
                "missing": list(self.missing),
                "undetermined": list(self.undetermined), "line": self.line}


def applicable(mechanism: str, regime: Regime) -> Verdict:
    """Применим ли механизм в этом режиме, и чего не хватает, если нет."""
    spec = MECHANISMS.get(mechanism)
    if spec is None:
        raise RegimeError(f"нет такого механизма: {mechanism!r}. "
                          f"Есть: {sorted(MECHANISMS)}")
    missing, undet = [], []
    for need in spec["needs"]:
        # Требование может быть отрицательным: «не мир идёт без меня». Записывается
        # строкой с приставкой «не », а не отдельным полем: набор признаков закрыт, и
        # плодить зеркальные признаки значило бы удваивать таблицу истины ни за что.
        negated = need.startswith("не ")
        feature = need[3:] if negated else need
        got = regime.has(feature)
        if got is None:
            undet.append(need)
            continue
        satisfied = (not got) if negated else got
        if not satisfied:
            missing.append(need)
    if missing:
        return Verdict(mechanism, False, tuple(missing), tuple(undet))
    if undet:
        return Verdict(mechanism, None, (), tuple(undet))
    return Verdict(mechanism, True, (), ())


# --- определение признаков наблюдением --------------------------------------


def measure_baseline(*, step: Callable[[Any], np.ndarray], frames: int,
                     print_of: Callable[[np.ndarray], tuple[int, ...]],
                     tolerance: int, gap: int = 2) -> Baseline:
    """Запись неподвижности: что бывает в этом домене, когда агент не делает ничего.

    Три величины, и каждая — опора для своего признака:

    - **изменение кадра** — опора для «мир идёт без меня»;
    - **общий сдвиг** — опора для эго-движения: сдвиг бывает и без действий, если едет
      содержимое;
    - **сходство отпечатков** — опора для обратимости: «вид вернулся» значит «похож не
      меньше, чем два соседних кадра похожи сами по себе». В неподвижном домене это
      требование строгое (сходство 1.0), в едущем — мягче ровно на то, насколько домен
      едет. Ни то, ни другое не подбирается: и там и там мерка снята с домена.

    `frames == 0` — законный вызов: он возвращает пустую запись, и все зависящие признаки
    станут «не определено».
    """
    frames = max(0, int(frames))
    gap = max(1, int(gap))
    if frames == 0:
        return NO_BASELINE
    from ..model.places import similarity

    diffs: list[float] = []
    shifts: list[float] = []
    sims: list[float] = []
    prints: list[tuple[int, ...]] = []
    prev = step(None)
    prints.append(print_of(prev))
    for _ in range(frames):
        cur = step(None)
        prints.append(print_of(cur))
        a, b = prev.astype(np.float64), cur.astype(np.float64)
        diffs.append(float(np.abs(b - a).mean()))
        sh = estimate_global_shift(a, b)
        shifts.append(float((sh.dy ** 2 + sh.dx ** 2) ** 0.5))
        sims.append(similarity(prints[-2], prints[-1], tolerance=tolerance))
        prev = cur
    gapped = [similarity(prints[i], prints[i + gap], tolerance=tolerance)
              for i in range(len(prints) - gap)] or list(sims)
    return Baseline(len(diffs), float(np.median(diffs)), min(diffs), max(diffs),
                    float(np.median(shifts)), max(shifts),
                    float(np.median(sims)), min(sims),
                    float(np.median(gapped)), min(gapped), gap)


def detect(*, step: Callable[[Any], np.ndarray], outputs: Iterable[str],
           profile: Any, hold_ms: int = 200, idle_frames: int = 12,
           probes_per_output: int = 3, judge: bool | None = None,
           baseline: Baseline | None = None) -> Regime:
    """Определить режим, действуя и наблюдая. Истины мира здесь нет ни в одном виде.

    `step(action_or_None) -> frame` — единственный доступ к среде: подать действие (или
    ничего) и получить кадр. Ни маски, ни состояния, ни имён.

    `judge` — есть ли канал чужой оценки. Аргумент, а не определение по кадру: канал
    оценки приходит извне восприятия (речь оператора, разметка домена), и выдумывать его
    по пикселям значило бы завести молчаливую заглушку. `None` — «не определено», и
    именно так и остаётся на всех нынешних доменах.

    ## Две ошибки первой редакции, обе найдены прогоном

    **Эго-движение сравнивалось с нулём, а не с бездействием.** В проигрывателе видео
    содержимое едет само, фазовая корреляция находит сдвиг 32 пикселя без всякого
    действия, и детектор объявлял эго-движение там, где камеры нет вовсе. Теперь сдвиг
    после удержания сравнивается со сдвигом при бездействии, и во сколько раз —
    объявлено в схеме (`regime_ego_over_idle`).

    **Возврат вида проверялся средней разностью пикселей.** У всех доменов обрамление
    анимировано (10–20 % кадра), средняя разность из-за этого никогда не падает до нуля,
    и «вид вернулся» не находилось нигде. Теперь возврат проверяется **отпечатком вида**
    — тем самым, которым агент узнаёт места, — и мигающий индикатор его не сбивает.
    """
    from ..core.action import Action
    from ..model.places import fingerprint, similarity

    p = profile.parameters
    min_shift = float(p["flow_min_global_shift"])
    margin = float(p["regime_alone_margin"])
    over_idle = float(p["regime_ego_over_idle"])
    alone_ratio = float(p["regime_alone_ratio"])
    same_view = float(p["place_same_similarity"])
    min_baseline = int(p["regime_baseline_min_frames"])
    reference = str(profile.structural["regime_view_reference"])
    grid = int(profile.structural["place_grid"])
    levels = int(profile.structural["place_levels"])
    blur = int(profile.structural["place_blur_px"])
    tol = int(profile.structural["place_level_tolerance"])

    def print_of(frame: np.ndarray) -> tuple[int, ...]:
        return fingerprint(frame, grid=grid, levels=levels, blur_px=blur)

    # 1. Запись неподвижности — опорный уровень домена. Либо передана снаружи (например,
    #    снята из живой записи), либо снимается здесь же бездействием.
    bl = (baseline if baseline is not None
            else measure_baseline(step=step, frames=idle_frames, print_of=print_of,
                                  tolerance=tol))
    grounded = bl.enough(min_baseline)

    # «Вид вернулся» сравнивается с фоном домена, а не с постоянным порогом. Планка —
    # сходство отпечатков через тот же промежуток, что у самой проверки: между видом до
    # действия и видом после двух действий проходит два шага мира, и в едущем домене за
    # эти два шага картинка уезжает сама. В неподвижном домене планка получается строгой
    # (сходство 1.0), в едущем — мягче ровно на дрейф домена. Ни то, ни другое не
    # подобрано: и там и там мерка снята с этого домена.
    #
    # Точка отсчёта для «вид вернулся» — выбор из трёх, объявленный структурной настройкой
    # `regime_view_reference`, потому что от него зависит ответ признака.
    #
    # Замер по пяти доменам (`tools/measure_regime.py`) даёт обе ошибки для каждого:
    #
    # | отсчёт | ложных тревог | ложных подтверждений |
    # |---|---|---|
    # | `criterion` — критерий тождества вида агента | 1 из 5 | 0 из 5 |
    # | `background` — фон домена через тот же промежуток | 0 из 5 | 1 из 5 |
    # | `looser` — меньшее из двух | 1 из 5 | 0 из 5 |
    #
    # **Замер не различает три отсчёта**: у каждого ровно одна ошибка из пяти доменов, и
    # различаются они только тем, на каком домене ошибаются. Говорить о победителе по этим
    # числам нельзя, и потому выбор оставлен настройкой, а умолчанием стоит `criterion` —
    # по смыслу, а не по числам: это `place_same_similarity`, то же самое отношение,
    # которым агент узнаёт места. «Вид вернулся» и «это то же место» — один вопрос, и
    # отвечать на него двумя разными мерками было бы хуже любой из них. Фон домена мерит
    # **дрейф**, а промах отката — неточность действия, и дрейфом она не описывается.
    if bl.frames and reference == "background":
        view_bar = bl.sim_gap_med
    elif bl.frames and reference == "looser":
        view_bar = min(same_view, bl.sim_gap_med)
    else:
        view_bar = same_view

    def same(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
        return similarity(a, b, tolerance=tol) >= view_bar

    idle_mean = bl.diff_med
    idle_shift_med = bl.shift_med

    # 2. Эго-движение: **существует ли выход**, повторяемо дающий сдвиг больше, чем
    #    бывает при бездействии. Доля по всем выходам здесь неверная мерка: проводка
    #    разрежена, и один двигающий выход из шести дал бы долю 0.17 при любом пороге.
    #
    # Опора — не только медиана бездействия, но и **самый большой** сдвиг без действия:
    # если содержимое дёргается, медиана его недооценивает. `flow_min_global_shift` —
    # разрешение самой оценки потока, а не мерка домена, и потому остаётся полом.
    need = max(min_shift, idle_shift_med * over_idle, bl.shift_max)
    per_output: dict[str, float] = {}
    per_output_diff: dict[str, float] = {}
    for out in outputs:
        got: list[float] = []
        diffs: list[float] = []
        for _ in range(probes_per_output):
            before = step(None)
            after = step(Action.key(out, hold_ms))
            b64, a64 = before.astype(np.float64), after.astype(np.float64)
            sh = estimate_global_shift(b64, a64)
            got.append(float((sh.dy ** 2 + sh.dx ** 2) ** 0.5))
            diffs.append(float(np.abs(a64 - b64).mean()))
        per_output[out] = float(np.median(got))
        per_output_diff[out] = float(np.median(diffs))
    best = max(per_output.values()) if per_output else 0.0
    ego = best >= need
    movers = sorted(o for o, v in per_output.items() if v >= need)

    # «Мир идёт без меня» — величина **относительная**: изменение при бездействии против
    # наибольшего изменения, которое я умею вызвать. Абсолютного пола мало (см. докстринг).
    acted_diff = [d for d in per_output_diff.values() if d is not None]
    acted_max = max(acted_diff) if acted_diff else 0.0
    ratio = idle_mean / acted_max if acted_max > 1e-9 else 0.0
    alone = idle_mean > margin and ratio >= alone_ratio

    # 3. Обратимость: существует ли действие, возвращающее **вид**. Пары «сделал A —
    #    сделал B — тот же вид, что до A»; вид сравнивается отпечатком, а не пикселями.
    reversible = False
    pairs = 0
    found: tuple[str, str] | None = None
    outs = list(outputs)
    for a in outs:
        base = print_of(step(None))
        after_a = print_of(step(Action.key(a, hold_ms)))
        if same(base, after_a):
            continue                    # действие вид не изменило: возвращать нечего
        for b in outs:
            after_b = print_of(step(Action.key(b, hold_ms)))
            pairs += 1
            if same(after_b, base):
                reversible, found = True, (a, b)
                break
        if reversible:
            break

    # Без записи неподвижности сравнивать не с чем, и тогда признак отвечает «не
    # определено» — а не подбирает порог. Это единственный честный ответ: пороги, снятые
    # с другого домена, были бы подсказкой о **этом** (инвариант 4), а сравнение с нулём
    # уже трижды дало неверный признак.
    no_base = ("записи неподвижности нет (кадров без действия "
               f"{bl.frames}, нужно {min_baseline}): сравнивать не с чем")
    regime = Regime(baseline=bl)
    regime.add(Evidence(
        EGO_MOTION, ego if grounded else None, len(per_output), "выход",
        (f"наибольший медианный сдвиг по выходу {best:.1f} px против порога {need:.1f} px"
         + (f"; двигают: {', '.join(movers)}" if movers else "")) if grounded else no_base,
        reference=(f"бездействие: медиана {idle_shift_med:.1f} px, наибольший "
                   f"{bl.shift_max:.1f} px, множитель {over_idle:g}, пол оценки потока "
                   f"{min_shift:g} px" if grounded else "нет")))
    regime.add(Evidence(
        WORLD_ALONE, alone if grounded else None, bl.frames, "кадр без действия",
        (f"изменение при бездействии {idle_mean:.3f}, доля от вызванного мной "
         f"{ratio:.3f}") if grounded else no_base,
        reference=(f"пол {margin} и порог доли {alone_ratio}; самая тихая пара кадров "
                   f"домена {bl.diff_min:.3f}, самая шумная {bl.diff_max:.3f}"
                   if grounded else "нет")))
    # Ни одно действие не изменило вид — значит возвращать было нечего, и это «не
    # определено», а не «обратимых действий нет». Различие то же, что между «проверял и не
    # нашёл» и «проверить было нечем», и оно ведёт к разным решениям.
    if not grounded:
        rev_value, rev_detail = None, no_base
    elif pairs == 0:
        rev_value = None
        rev_detail = ("ни одно действие не изменило вид: возвращать было нечего. Это не "
                      "«обратимых действий нет», а «проверить было нечем»")
    else:
        rev_value = reversible
        rev_detail = (f"вид вернуло {found[1]} после {found[0]}" if found
                      else "ни одно действие не вернуло вид")
    regime.add(Evidence(
        REVERSIBLE, rev_value, pairs, "пара действий", rev_detail,
        reference=(f"сходство отпечатков не ниже {view_bar:.3f} (отсчёт "
                   f"«{reference}»): критерий тождества вида {same_view:g}, фон домена "
                   f"через {bl.gap} шага без действия {bl.sim_gap_med:.3f}"
                   if grounded else "нет")))
    regime.add(Evidence(JUDGE, judge, 0 if judge is None else 1, "канал оценки",
                        "канала чужой оценки в этой среде нет вовсе" if judge is None
                        else f"канал оценки объявлен: {judge}",
                        reference="опорный уровень не нужен: канал оценки — не измерение "
                                  "кадра"))
    return regime


def report(regime: Regime) -> dict[str, Any]:
    """Режим и приговоры по всем механизмам — то, что уходит в запись и в отчёт."""
    verdicts = {name: applicable(name, regime).as_dict() for name in MECHANISMS}
    return {
        "features": regime.as_dict(),
        "baseline": regime.baseline.as_dict(),
        "mechanisms": verdicts,
        "applicable": sorted(k for k, v in verdicts.items() if v["applicable"] is True),
        "inapplicable": sorted(k for k, v in verdicts.items()
                               if v["applicable"] is False),
        "undetermined": sorted(k for k, v in verdicts.items()
                               if v["applicable"] is None),
    }
