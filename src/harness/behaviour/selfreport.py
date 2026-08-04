"""Самоотчёт агента: что он делал, почему, что его удивило и как он себя чувствует.

Из замысла: «отдавать отчёт своим действиям». И оттуда же — предупреждение, которое
важнее самой возможности: «самоотчёты агента ни на что не влияют автоматически.
Реакция только на поведение и объективные величины. Если код позволит завязать
последствия на слова агента, эксперимент испорчен навсегда» (инвариант 10).

До этого модуля инвариант 10 проверял отсутствующую вещь: самоотчётов не было, и
проверялось, что пометки исследователя ни на что не влияют. Теперь самоотчёт есть, и
инвариант стал проверкой того, что действительно может испортиться.

## Как устроена невозможность влияния

Не договорённостью, а тремя механизмами, каждый со своим тестом:

1. **Отчёт — сток.** Он строится из того, что у агента уже есть, и не возвращает
   ничего, чем можно управлять: `SelfReport` — набор чисел и непрозрачных символов, а
   не решение. Ни одна функция здесь не решает, что делать дальше.
2. **Пересборка отчёты пропускает.** Как и `THOUGHT`: запись вида `SELF_REPORT` не
   создаёт убеждений и не меняет карту тела. Сказать «я умею откатывать этот выход» и
   тем самым сделать его откатываемым — невозможно.
3. **Никто его не читает.** Ни один модуль агентской стороны не импортирует этот; его
   читают пульт, CLI и исследователь. Проверяется обходом графа импортов — тем же
   способом, которым проверяется изоляция отладочного канала.

## Из чего он собран

Только из того, что агенту законно доступно: карта тела, убеждения с происхождением,
цели с их тестами, настроение, ошибка предсказания, граф мест. Ни координат, ни имён
клавиш, ни расшифрованного текста — всё, что попадает в отчёт, проходит ту же проверку
на непрозрачность, что и восприятие.

Отдельно про «чего я не знаю». Это не украшение: перечень незнания — самая полезная
часть отчёта для исследователя, потому что он показывает, куда агент *не* смотрел, а
такие места не видны ни по одному счётчику успехов.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..core.clocks import Stamp
from ..core.journal import Actor, Journal, Kind as EntryKind
from ..core.symbols import assert_no_plain_text
from ..model.drives import DRIVE_NAMES, EMOTIONS, emotion_label
from ..model.rebuild import BodyMap


@dataclass(frozen=True, slots=True)
class Line:
    """Одна строка отчёта: код, величина, происхождение.

    Код, а не фраза, и это не педантизм. Фраза на естественном языке была бы текстом,
    который так и тянет разобрать и на что-нибудь завязать; код разобрать нечем, кроме
    таблицы в этом файле, и она известна только исследователю. Заодно отчёт остаётся
    сравнимым между прогонами: строки считаются, а не читаются.
    """

    code: str
    value: float | int | str | bool | None = None
    provenance: str = ""          # experience | testimony | hunch | none

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "value": self.value,
                "provenance": self.provenance}


# Коды строк отчёта. Набор закрыт: новая строка — это изменение схемы отчёта, а не
# свободный текст.
WHAT_I_DID = "what_i_did"                 # чем занимался: доля проб, планов, сна
WHY = "why"                               # какой драйв давил сильнее всего
GOAL_ACTIVE = "goal_active"               # какая цель активна и её тест
GOAL_PASSED = "goal_passed"               # сколько целей прошло тест
GOAL_ABANDONED = "goal_abandoned"         # сколько брошено по бюджету
SURPRISED_BY = "surprised_by"             # где ошибка предсказания дала всплеск
I_LEARNED = "i_learned"                   # что стало опытом (с происхождением)
I_HEARD = "i_heard"                       # что осталось чужим словом
I_GUESS = "i_guess"                       # что осталось догадкой
I_CANNOT_UNDO = "i_cannot_undo"           # чего не умею откатить
I_DO_NOT_KNOW = "i_do_not_know"           # чего не знаю: непробованное, неясное
FEELING = "feeling"                       # настроение и ярлык эмоции
BODY_KNOWN = "body_known"                 # какая часть тела открыта
PLACES_KNOWN = "places_known"             # сколько мест и подтверждённых переходов

CODES = frozenset({
    WHAT_I_DID, WHY, GOAL_ACTIVE, GOAL_PASSED, GOAL_ABANDONED, SURPRISED_BY,
    I_LEARNED, I_HEARD, I_GUESS, I_CANNOT_UNDO, I_DO_NOT_KNOW, FEELING,
    BODY_KNOWN, PLACES_KNOWN,
    # Числовые строки той же схемы: величина без собственного смысла кроме числа.
    "valence", "arousal", "body_silent", "error_mean", "goal_kind",
    "transitions_confirmed",
})

# Слова, которые в отчёте допустимы, потому что они не с экрана. Это собственный
# словарь агента: имена драйвов, ярлыки эмоций, виды целей, названия состояний
# выхода и служебные части составных ключей убеждений. Набор закрыт — ровно чтобы
# было чем отличить своё слово от подсмотренной надписи.
VOCABULARY = frozenset(DRIVE_NAMES) | frozenset(EMOTIONS) | CODES | frozenset({
    "learn_output", "undo_output", "verify_hearsay", "reach_place", "reduce_error",
    "live", "silent", "unclear", "untried", "unknown_rate",
    "afford", "dyn", "experience", "testimony", "hunch", "none",
    # Чем агент был занят. Тоже закрытый набор: занятие, которого нет в списке,
    # означает новый контур, а не новое слово в отчёте.
    "babble", "plan", "rehearse", "travel", "sleep", "watch",
})

_SEPARATORS = re.compile(r"[|@:/,]")
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")


def check_opaque(value: str, *, path: str = "") -> None:
    """Проверить, что в строке отчёта нет надписи с экрана.

    Проверка не та же, что на границе восприятия, и это не поблажка. На границе
    допустимы только символы: там всё приходит с экрана, значит всё подозрительно.
    Здесь строки составные (`ENT_1C90|afford|OUT_2C@200`) и содержат слова, которых
    на экране быть не может — имена драйвов, виды целей. Поэтому строка режется на
    части, и каждая часть обязана быть либо числом, либо словом из закрытого
    словаря, либо символом — последнее проверяется той же функцией, что и восприятие.

    Смысл именно в закрытости словаря: расшифрованная надпись в словарь не входит и
    падает громко, как и должна.
    """
    for token in _SEPARATORS.split(value):
        token = token.strip()
        if not token or token in VOCABULARY or _NUMBER.match(token):
            continue
        assert_no_plain_text(token, path=path or "self_report")


@dataclass(slots=True)
class SelfReport:
    """Отчёт целиком. Ничего, кроме строк, чисел и непрозрачных символов."""

    stamp_self: int
    stamp_world: int
    lines: list[Line] = field(default_factory=list)

    def by_code(self, code: str) -> list[Line]:
        return [ln for ln in self.lines if ln.code == code]

    def value(self, code: str) -> Any:
        found = self.by_code(code)
        return found[0].value if found else None

    def as_dict(self) -> dict[str, Any]:
        return {"t_self": self.stamp_self, "t_world": self.stamp_world,
                "lines": [ln.as_dict() for ln in self.lines]}

    def as_event(self) -> dict[str, Any]:
        """То, что уйдёт в журнал. Проверяется на читаемый текст, как восприятие."""
        for ln in self.lines:
            if ln.code not in CODES:
                raise ValueError(
                    f"строка отчёта с кодом {ln.code!r} вне схемы. Набор кодов закрыт: "
                    "новая строка — изменение схемы отчёта, а не свободный текст")
            if isinstance(ln.value, str):
                check_opaque(ln.value, path=f"self_report.{ln.code}")
        return {"code": "self_report", **self.as_dict()}


def compose(*, stamp: Stamp, body: BodyMap | None = None,
            beliefs: Any = None, goals: Any = None, motivation: Any = None,
            error: Any = None, places: Any = None,
            model: Any = None, outputs: int = 0,
            did: dict[str, int] | None = None,
            error_high: bool = False) -> SelfReport:
    """Собрать отчёт из того, что у агента уже есть.

    Все части необязательны: агент на второй минуте жизни не имеет ни графа мест, ни
    целей, и отчёт об этом так и скажет — отсутствием строк, а не нулями. Ноль и
    «нечего сказать» — разные вещи, и путать их в отчёте так же нельзя, как в модели.
    """
    lines: list[Line] = []

    # --- чем был занят ------------------------------------------------------
    for name, count in sorted((did or {}).items()):
        lines.append(Line(WHAT_I_DID, f"{name}:{int(count)}", "experience"))

    # --- что делал и почему -------------------------------------------------
    if motivation is not None:
        dominant = motivation.dominant()
        lines.append(Line(WHY, dominant.name, "experience"))
        mood = motivation.mood
        lines.append(Line(FEELING, emotion_label(mood, error_high=error_high),
                          "experience"))
        lines.append(Line("valence", round(mood.valence, 3), "experience"))
        lines.append(Line("arousal", round(mood.arousal, 3), "experience"))

    # --- тело ---------------------------------------------------------------
    if body is not None:
        known = len(body.outputs)
        live = body.by_state("live")
        silent = body.by_state("silent")
        unclear = body.by_state("unclear")
        lines.append(Line(BODY_KNOWN,
                          round(known / outputs, 3) if outputs else known,
                          "experience"))
        for out in live:
            lines.append(Line(I_LEARNED, out, "experience"))
        for out in body.dangerous(1.0):
            lines.append(Line(I_CANNOT_UNDO, out, "experience"))
        for out in unclear:
            lines.append(Line(I_DO_NOT_KNOW, out, "experience"))
        if outputs and known < outputs:
            lines.append(Line(I_DO_NOT_KNOW, f"untried:{outputs - known}", "none"))
        lines.append(Line("body_silent", len(silent), "experience"))

    # --- убеждения: опыт, чужое слово, догадка ------------------------------
    if beliefs is not None:
        for belief in beliefs.beliefs():
            code = I_LEARNED
            if getattr(belief, "is_hearsay", False):
                code = I_HEARD
            elif str(getattr(belief.provenance, "origin", "")) == "hunch":
                code = I_GUESS
            lines.append(Line(code, belief.claim,
                              str(getattr(belief.provenance, "origin", ""))))

    # --- цели ---------------------------------------------------------------
    if goals is not None:
        active = getattr(goals, "active", None)
        if active is not None:
            lines.append(Line(GOAL_ACTIVE, active.target, "experience"))
            lines.append(Line("goal_kind", active.kind, "experience"))
        stats = goals.stats()
        lines.append(Line(GOAL_PASSED, int(stats.get("passed", 0)), "experience"))
        lines.append(Line(GOAL_ABANDONED, int(stats.get("abandoned", 0)),
                          "experience"))

    # --- что удивило --------------------------------------------------------
    if error is not None:
        summary = error.summary()
        lines.append(Line(SURPRISED_BY, len(getattr(error, "spikes", [])),
                          "experience"))
        lines.append(Line("error_mean", round(float(summary.get("mean", 0.0)), 6),
                          "experience"))

    # --- места --------------------------------------------------------------
    if places is not None:
        lines.append(Line(PLACES_KNOWN, len(places), "experience"))
    if model is not None:
        confirmed = sum(1 for outs in model.transitions.values()
                        for o in outs if o.n >= 2)
        lines.append(Line("transitions_confirmed", confirmed, "experience"))
        lines.append(Line(I_DO_NOT_KNOW, f"unknown_rate:{model.unknown_rate:.3f}",
                          "none"))

    return SelfReport(stamp.t_self, stamp.t_world, lines)


def journal_report(journal: Journal, report: SelfReport, stamp: Stamp) -> Any:
    """Записать отчёт в журнал видом `SELF_REPORT`.

    Отдельный вид записи, а не `NOTE` и не `PERCEPTION`, ровно по той же причине, по
    которой воображаемое пишется `THOUGHT`: из журнала должно быть видно, что это
    слова агента о себе, а не наблюдение и не пометка человека. Пересборка такие
    записи пропускает — сказанное о себе не становится знанием о мире.
    """
    return journal.append(EntryKind.SELF_REPORT, stamp, Actor.AGENT,
                          event=report.as_event())
