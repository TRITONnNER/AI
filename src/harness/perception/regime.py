"""Режим среды: четыре признака, определяемые наблюдением. TASK-32, направление A.

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
class Evidence:
    """Один признак: ответ, на чём он основан и сколько наблюдений за ним.

    `value is None` — «не определено», и это **не** то же самое, что `False`. «Обратимых
    действий нет» и «обратимость проверить было нечем» ведут к разным решениям: в первом
    случае механизм неприменим, во втором неизвестно, применим ли, и это надо доизмерить.
    """

    feature: str
    value: bool | None
    n: int
    unit: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"feature": self.feature, "value": self.value, "n": self.n,
                "unit": self.unit, "detail": self.detail}


@dataclass(slots=True)
class Regime:
    """Режим среды: четыре признака с основаниями."""

    evidence: dict[str, Evidence] = field(default_factory=dict)

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


def detect(*, step: Callable[[Any], np.ndarray], outputs: Iterable[str],
           profile: Any, hold_ms: int = 200, idle_frames: int = 12,
           probes_per_output: int = 3, judge: bool | None = None) -> Regime:
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
    grid = int(profile.structural["place_grid"])
    levels = int(profile.structural["place_levels"])
    blur = int(profile.structural["place_blur_px"])
    tol = int(profile.structural["place_level_tolerance"])

    def print_of(frame: np.ndarray) -> tuple[int, ...]:
        return fingerprint(frame, grid=grid, levels=levels, blur_px=blur)

    def same(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
        return similarity(a, b, tolerance=tol) >= same_view

    # 1. Бездействие: и насколько меняется кадр, и какой при этом бывает сдвиг. Второе
    #    нужно как **база сравнения** для эго-движения: см. докстринг.
    idle_diff: list[float] = []
    idle_shift: list[float] = []
    prev = step(None)
    for _ in range(idle_frames):
        cur = step(None)
        idle_diff.append(float(np.abs(cur.astype(np.float64)
                                      - prev.astype(np.float64)).mean()))
        sh = estimate_global_shift(prev.astype(np.float64), cur.astype(np.float64))
        idle_shift.append(float((sh.dy ** 2 + sh.dx ** 2) ** 0.5))
        prev = cur
    idle_mean = float(np.median(idle_diff)) if idle_diff else 0.0
    idle_shift_med = float(np.median(idle_shift)) if idle_shift else 0.0

    # 2. Эго-движение: **существует ли выход**, повторяемо дающий сдвиг больше, чем
    #    бывает при бездействии. Доля по всем выходам здесь неверная мерка: проводка
    #    разрежена, и один двигающий выход из шести дал бы долю 0.17 при любом пороге.
    need = max(min_shift, idle_shift_med * over_idle)
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

    regime = Regime()
    regime.add(Evidence(EGO_MOTION, ego, len(per_output), "выход",
                        f"наибольший медианный сдвиг по выходу {best:.1f} px против "
                        f"порога {need:.1f} px (бездействие даёт {idle_shift_med:.1f} px, "
                        f"множитель {over_idle:g})"
                        + (f"; двигают: {', '.join(movers)}" if movers else "")))
    regime.add(Evidence(WORLD_ALONE, alone, len(idle_diff), "кадр без действия",
                        f"изменение при бездействии {idle_mean:.3f} (пол {margin}), доля "
                        f"от вызванного мной {ratio:.3f} против порога {alone_ratio}"))
    regime.add(Evidence(REVERSIBLE, reversible, pairs, "пара действий",
                        f"вид вернуло {found[1]} после {found[0]}" if found
                        else "ни одно действие не вернуло вид"))
    regime.add(Evidence(JUDGE, judge, 0 if judge is None else 1, "канал оценки",
                        "канала чужой оценки в этой среде нет вовсе" if judge is None
                        else f"канал оценки объявлен: {judge}"))
    return regime


def report(regime: Regime) -> dict[str, Any]:
    """Режим и приговоры по всем механизмам — то, что уходит в запись и в отчёт."""
    verdicts = {name: applicable(name, regime).as_dict() for name in MECHANISMS}
    return {
        "features": regime.as_dict(),
        "mechanisms": verdicts,
        "applicable": sorted(k for k, v in verdicts.items() if v["applicable"] is True),
        "inapplicable": sorted(k for k, v in verdicts.items()
                               if v["applicable"] is False),
        "undetermined": sorted(k for k, v in verdicts.items()
                               if v["applicable"] is None),
    }
