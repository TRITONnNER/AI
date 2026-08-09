"""Показатели: одно место, где видно, сколько агент прожил и чему научился.

Зачем отдельный модуль, если у каждого узла уже есть свой `stats()`. Потому что
девять сводок — это не сводка. Пока показатели разбросаны по `BodyMap.stats()`,
`GoalStack.stats()`, `PlaceGraph.stats()` и остальным, ответить на вопрос «сколько
уже готово и что изменилось за час» нельзя: у каждой сводки своя единица, свой
момент времени и свой набор полей. Здесь всё сведено к одной форме — код, число,
единица, источник, — и посчитано **из журнала**, а не из живых объектов в памяти.

Из журнала — это не педантизм, а инвариант 1: журнал источник истины, всё остальное
из него выводимо. Показатель, который можно получить только из работающего процесса,
нельзя ни перепроверить, ни сравнить с прошлым прогоном. Поэтому `from_journal`
принимает журнал и профиль, и больше ничего.

## Чего здесь принципиально нет

Показателя, которого в журнале нет, здесь нет тоже — вместо него строка со значением
`None` и причиной. Это главное правило модуля: посчитать ошибку предсказания «примерно
по числу всплесков» было бы правдоподобным числом вместо реального, то есть ровно тем,
что запрещено в правилах работы. Ошибка предсказания в журнал пока не пишется, и в
отчёте так и сказано.

## Отношение к самоотчёту

Показатели — объективная колонка. Самоотчёт (`behaviour/selfreport.py`) — колонка со
словами агента. Их удобно смотреть рядом (`harness report`), и именно рядом, а не
вместе: если бы самоотчёт участвовал в расчёте показателя, инвариант 10 был бы нарушен
самым незаметным способом — через сводку, на которую потом смотрит человек и меняет
профиль. Поэтому здесь нет ни одного обращения к `SELF_REPORT`, кроме подсчёта того,
сколько таких записей есть.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..core.journal import Journal, Kind as EntryKind
from ..core.profile import Profile
from .rebuild import Rebuilt, rebuild_from_journal


@dataclass(frozen=True, slots=True)
class Vital:
    """Один показатель: что, сколько, в чём и откуда.

    `unit` и `source` обязательны и не для красоты. Единица — потому что «12» без
    единицы нельзя сравнить ни с прошлым прогоном, ни с порогом. Источник — потому
    что только по нему видно, посчитан показатель по журналу или взят из воздуха;
    если источник пустой, показателя быть не должно.
    """

    code: str
    value: float | int | None
    unit: str
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError(
                f"показатель {self.code} без источника. Число, про которое нельзя "
                "сказать, откуда оно, хуже отсутствующего числа")
        if self.value is None and not self.note:
            raise ValueError(
                f"показатель {self.code} отсутствует без объяснения. Пустое место в "
                "сводке обязано быть подписано, иначе оно читается как ноль")

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "value": self.value, "unit": self.unit,
                "source": self.source, "note": self.note}


@dataclass(slots=True)
class Vitals:
    """Все показатели вместе плюс пересобранная модель, из которой часть посчитана."""

    vitals: list[Vital] = field(default_factory=list)
    rebuilt: Rebuilt | None = None

    def add(self, code: str, value: float | int | None, unit: str, source: str,
            note: str = "") -> None:
        self.vitals.append(Vital(code, value, unit, source, note))

    def get(self, code: str) -> Vital | None:
        return next((v for v in self.vitals if v.code == code), None)

    def value(self, code: str) -> float | int | None:
        v = self.get(code)
        return v.value if v else None

    def absent(self) -> list[Vital]:
        """Показатели, которых нет. Самая полезная часть для планирования работы."""
        return [v for v in self.vitals if v.value is None]

    def as_dict(self) -> dict[str, Any]:
        return {"vitals": [v.as_dict() for v in self.vitals],
                "absent": [v.code for v in self.absent()]}

    def render_text(self, *, width: int = 78) -> str:
        lines: list[str] = []
        for v in self.vitals:
            if v.value is None:
                shown = "—"
            elif isinstance(v.value, float):
                shown = f"{v.value:.4g}"
            else:
                shown = str(v.value)
            head = f"  {v.code:<28} {shown:>10} {v.unit:<10}"
            lines.append(head.rstrip())
            if v.note:
                lines.append(f"      {v.note[:width - 6]}")
        return "\n".join(lines)


def _share(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def from_journal(journal: Journal, *, profile: Profile,
                 skip: Iterable[str] = ()) -> Vitals:
    """Посчитать показатели по журналу. Единственный способ их получить.

    `skip` — коды показателей, которые не считать: пересборка убеждений по большому
    журналу небесплатна, и вызывающему иногда нужны только счётчики записей.
    """
    skipped = set(skip)
    out = Vitals()
    p = profile.parameters

    by_kind: dict[str, int] = {}
    frames_missed = 0
    gaps = 0
    actions = 0
    masked = 0
    delivered = 0
    responded = 0
    goals_set = 0
    goals_passed = 0
    goals_abandoned = 0
    abandon_spent: list[int] = []
    plan_steps = 0
    plan_agreed = 0
    sleeps = 0
    last_sleep_self: int | None = None
    breaches: dict[str, int] = {}
    first_self = last_self = None
    first_world = last_world = None
    # Ошибка предсказания и настроение приходят из среза состояния записи, если
    # тот, кто писал, их измерял. Список, а не сумма: без числа наблюдений среднее
    # ничего не значит, а нулевая длина списка — это «не измерялось», а не ноль.
    errors: list[float] = []
    valences: list[float] = []
    layers: dict[str, int] = {}

    for e in journal:
        kind = str(e.kind)
        by_kind[kind] = by_kind.get(kind, 0) + 1
        if first_self is None:
            first_self, first_world = e.stamp.t_self, e.stamp.t_world
        last_self, last_world = e.stamp.t_self, e.stamp.t_world
        layers[str(e.actor_layer)] = layers.get(str(e.actor_layer), 0) + 1
        if e.state.prediction_error is not None:
            errors.append(float(e.state.prediction_error))
        if e.state.mood is not None:
            valences.append(float(e.state.mood[0]))

        if e.kind is EntryKind.CAPTURE_GAP:
            gaps += 1
            frames_missed += int(e.event.get("missed", 0))
        elif e.kind is EntryKind.ACTION:
            actions += 1
            if e.action is not None and e.action.masked:
                masked += 1
            if e.event.get("code") == "delivered":
                delivered += 1
            if e.event.get("responded") is True:
                responded += 1
        elif e.kind is EntryKind.GOAL:
            code = e.event.get("code")
            if code == "set":
                goals_set += 1
            elif code == "passed":
                goals_passed += 1
            elif code == "abandoned":
                goals_abandoned += 1
                abandon_spent.append(int(e.event.get("spent_ticks", 0)))
        elif e.kind is EntryKind.PLAN:
            plan_steps += 1
            if e.event.get("agreed") is True:
                plan_agreed += 1
        elif e.kind is EntryKind.SLEEP:
            sleeps += 1
            last_sleep_self = e.stamp.t_self
        elif e.kind is EntryKind.RESOURCE:
            code = str(e.event.get("code", "?"))
            breaches[code] = breaches.get(code, 0) + 1

    frames = by_kind.get(str(EntryKind.FRAME), 0)
    J = "журнал"

    # --- сколько прожито ----------------------------------------------------
    out.add("entries", sum(by_kind.values()), "записей", J)
    out.add("lived_self", (last_self - first_self) if first_self is not None else None,
            "циклов", J, "" if first_self is not None else "журнал пуст")
    out.add("lived_world", (last_world - first_world) if first_world is not None else None,
            "тиков", J, "" if first_world is not None else "журнал пуст")
    out.add("frames", frames, "кадров", J)

    # --- целостность записи -------------------------------------------------
    # Не «показатель качества», а условие осмысленности всех остальных: по журналу с
    # дырами слои, сдвиг и ошибка считаются по разным моментам времени.
    out.add("capture_gaps", gaps, "разрывов", f"{J}: capture_gap")
    out.add("frames_lost_share", _share(frames_missed, frames + frames_missed),
            "доля", f"{J}: capture_gap.missed",
            "" if frames or frames_missed else "кадров не было")

    # --- тело ---------------------------------------------------------------
    out.add("actions", actions, "попыток", f"{J}: action")
    out.add("masked_share", _share(masked, actions), "доля", f"{J}: action.masked",
            "" if actions else "попыток действия не было")
    out.add("delivered_share", _share(delivered, actions), "доля",
            f"{J}: action.code", "" if actions else "попыток действия не было")
    out.add("responded_share", _share(responded, delivered), "доля",
            f"{J}: action.responded",
            "" if delivered else "ни одна попытка не доехала до устройства")

    # --- цели: это и есть компетентность -----------------------------------
    # Компетентность считается по пройденным тестам целей, а не по словам агента о
    # себе и не по числу нажатий: тест цели — единственная объективная проверка
    # «получилось», какая в системе есть (инвариант 10).
    out.add("goals_set", goals_set, "целей", f"{J}: goal.set")
    out.add("competence", _share(goals_passed, goals_passed + goals_abandoned),
            "доля", f"{J}: goal.passed / (passed+abandoned)",
            "" if goals_passed + goals_abandoned else "ни одна цель ещё не закрылась")
    out.add("goals_abandoned", goals_abandoned, "целей", f"{J}: goal.abandoned")
    out.add("ticks_to_abandon",
            round(sum(abandon_spent) / len(abandon_spent), 2) if abandon_spent else None,
            "циклов", f"{J}: goal.spent_ticks",
            "" if abandon_spent else "брошенных целей ещё нет")

    # --- планы --------------------------------------------------------------
    out.add("plan_steps", plan_steps, "шагов", f"{J}: plan")
    out.add("plan_agreement", _share(plan_agreed, plan_steps), "доля",
            f"{J}: plan.agreed", "" if plan_steps else "планы ещё не исполнялись")

    # --- сон и сверка с реальностью -----------------------------------------
    out.add("sleeps", sleeps, "прогонов", f"{J}: sleep")
    if last_sleep_self is not None and last_self is not None:
        out.add("since_sleep_self", last_self - last_sleep_self, "циклов",
                f"{J}: sleep")
    else:
        out.add("since_sleep_self", None, "циклов", f"{J}: sleep",
                "консолидация ещё не запускалась")

    # --- вмешательства и слова ----------------------------------------------
    out.add("interventions", by_kind.get(str(EntryKind.INTERVENTION), 0),
            "вмешательств", f"{J}: intervention")
    out.add("thoughts", by_kind.get(str(EntryKind.THOUGHT), 0), "записей",
            f"{J}: thought")
    out.add("self_reports", by_kind.get(str(EntryKind.SELF_REPORT), 0), "отчётов",
            f"{J}: self_report")
    out.add("testimonies", by_kind.get(str(EntryKind.TESTIMONY), 0), "свидетельств",
            f"{J}: testimony")
    out.add("stops", by_kind.get(str(EntryKind.STOP), 0), "остановов",
            f"{J}: stop")
    out.add("resource_breaches", sum(breaches.values()), "упоров",
            f"{J}: resource")

    # --- то, что видно только после пересборки ------------------------------
    if "beliefs" not in skipped:
        rebuilt = rebuild_from_journal(
            journal,
            trust_human=float(p["testimony_trust_human"]),
            live_min_responses=int(p["body_live_min_responses"]),
            silent_min_deliveries=int(p["babble_repeats"]))
        out.rebuilt = rebuilt
        # Поля берутся по имени без `get(..., 0)` намеренно: переименованное поле
        # обязано ронять сводку, а не показывать ноль. Ноль в сводке читается как
        # измеренный факт, и отличить его от опечатки потом невозможно.
        bs = rebuilt.beliefs.stats()
        body = rebuilt.body.stats()
        total = int(bs["beliefs"])
        out.add("beliefs", total, "убеждений", "пересборка журнала")
        out.add("hearsay_share", _share(int(bs["hearsay"]), total), "доля",
                "пересборка: provenance",
                "" if total else "убеждений ещё нет")
        out.add("body_known", int(body["known_outputs"]), "выходов",
                "пересборка: карта тела")
        out.add("body_live", int(body["live"]), "выходов",
                "пересборка: карта тела")
        out.add("body_unclear", int(body["unclear"]), "выходов",
                "пересборка: карта тела",
                "выход отвечал, но реже, чем требует порог — не молчит и не жив")
        out.add("undo_unknown", int(body["unknown_reversibility"]), "выходов",
                "пересборка: обратимость",
                "чего агент не умеет откатывать: осторожность здесь максимальна")

    # --- чего в журнале нет -------------------------------------------------
    # Ниже — не заготовки под будущее, а честное перечисление: показатель назван,
    # значение отсутствует, причина указана. Так видно, чего в системе нет, вместо
    # правдоподобного числа на этом месте.
    # Ошибка предсказания и настроение теперь есть в формате записи (срез
    # состояния, v2), но только если писавший их измерял. Поэтому здесь два разных
    # исхода, и путать их нельзя: посчитанное среднее — или названная причина, по
    # которой считать нечего.
    if errors:
        out.add("prediction_error_mean", sum(errors) / len(errors), "0..1",
                f"срез состояния: {len(errors)} записей",
                "средняя ошибка предсказания по тем записям, где она измерялась. "
                "Записи без неё в среднее не входят: ноль там означал бы идеальное "
                "предсказание, а не отсутствие предсказателя")
    else:
        out.add("prediction_error_mean", None, "0..1", "нет источника",
                "ни одна запись не несёт ошибки предсказания в срезе состояния. "
                "Поле в формате есть (v2), измерять его пока некому: предсказателя "
                "в этом прогоне не было")
    if valences:
        out.add("mood_valence", sum(valences) / len(valences), "-1..1",
                f"срез состояния: {len(valences)} записей",
                "средняя валентность по записям, где настроение снималось. Оно "
                "выведено из объективных величин, а не заявлено агентом")
    else:
        out.add("mood_valence", None, "-1..1", "нет источника",
                "настроение не снималось ни в одной записи. Поле в формате есть, "
                "но драйвов в этом прогоне не было — а подставить сюда нуль значило "
                "бы сообщить о ровном настроении там, где его никто не мерил")
    out.add("places", None, "мест", "нет источника",
            "граф мест в журнал не пишется отдельным видом записи: он строится по "
            "кадрам, а кадры в журнале лежат ссылками на хранилище")
    # Распределение по слою-инициатору: без него доля конфабуляции обманчива.
    for name in sorted(layers):
        out.add(f"layer_{name}", layers[name], "записей",
                "actor_layer записей",
                f"сколько записей начал слой {name}. Рядом с метрикой конфабуляции "
                "это обязательно: 20 % расхождений при девяноста процентах записей "
                "от планировщика и при девяноста от рефлекса — разные прогоны")

    return out
