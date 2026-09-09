"""Критерий «запуск состоялся». SPEC-FULL, A4.

Семь пунктов, и все семь обязательны. Не выполнено — блок B не начинается.

Смысл этого модуля в том, чтобы критерий перестал быть прозой. Прозаический критерий
проверяется чтением, а чтение уже один раз объявило «узлы вышли на полку» там, где ось
была неверной (вырождение 13.7), и «отпечаток различает шум» там, где дело было в
способе кормления. Здесь каждый пункт считается по journal'у и по замерам, и у каждого
есть исход.

## Четыре исхода, а не два

| Исход | Что значит |
|---|---|
| `выполнен` | посчитано и сошлось |
| `не выполнен` | посчитано и не сошлось |
| `не проверен` | данных нет; **не** засчитывается |
| `выполнен вакуумно` | сошлось потому, что механизм не работал |

Четвёртый исход добавлен не для полноты. Замер каскада (`MEASUREMENT.md`, 38) дал ровно
его: пункт 5 формально выполнен во всех 60 прогонах, а на четырёх доменах из пяти
ступень 3 недостижима по построению, и ноль там означает «механизм не работал». Это то
самое «сошлось», которое инвариант 32 называет опаснее ложной тревоги: тревогу
проверяют, а совпадение закрывают. Критерий, у которого нет такого исхода, объявил бы
запуск состоявшимся.

`не проверен` тоже не мелочь. Соблазн подставить правдоподобное значение там, где данных
нет, — худшее, что здесь может произойти: пункт выглядел бы пройденным, и никто больше
не вернулся бы к нему. Поэтому пункт без данных называет, **каких именно** данных не
хватает и какой командой они добываются.

## Пороги живут в схеме

`launch_hour_seconds`, `launch_max_deep_share`, `launch_max_new_per_observation` —
настройки, а не числа в этом файле (инвариант 23). Иначе они не попадут в
`profile_hash`, и два прогона критерия окажутся несравнимы между собой.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

MET = "выполнен"
FAILED = "не выполнен"
UNCHECKED = "не проверен"
VACUOUS = "выполнен вакуумно"

#: Пункты критерия — дословно из спецификации, A4.
POINTS: tuple[tuple[int, str], ...] = (
    (1, "час непрерывной работы на настоящем экране без вмешательства"),
    (2, "мир ни разу не остановлен: обороты равны тикам мира"),
    (3, "число узлов графа мест насыщается по числу наблюдений"),
    (4, "после сна число сущностей падает за счёт слияния, а не растёт"),
    (5, "доля кадров, дошедших до ступени 3, ниже потолка"),
    (6, "расход укладывается в объявленный бюджет"),
    (7, "поставлена и закрыта хотя бы одна цель с пройденным тестом"),
)


@dataclass(frozen=True, slots=True)
class Point:
    """Один пункт критерия: исход, число, единица и объяснение.

    `value` и `unit` обязательны вместе с исходом: «выполнен» без числа — это опять
    проза, только с галочкой.
    """

    number: int
    name: str
    verdict: str
    value: Any = None
    unit: str = ""
    why: str = ""
    fix: str = ""            # чем добыть данные, если пункт не проверен

    @property
    def counts(self) -> bool:
        """Засчитывается ли пункт. Вакуумный — **нет**, и в этом весь смысл исхода."""
        return self.verdict == MET

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"point": self.number, "name": self.name,
                               "verdict": self.verdict, "counts": self.counts,
                               "value": self.value, "unit": self.unit, "why": self.why}
        if self.fix:
            out["fix"] = self.fix
        return out


@dataclass(frozen=True, slots=True)
class Verdict:
    """Все семь пунктов вместе."""

    points: tuple[Point, ...]

    @property
    def launched(self) -> bool:
        """Запуск состоялся. Все семь обязательны — «шесть из семи» не бывает."""
        return len(self.points) == len(POINTS) and all(p.counts for p in self.points)

    @property
    def met(self) -> int:
        return sum(1 for p in self.points if p.counts)

    def as_dict(self) -> dict[str, Any]:
        return {"launched": self.launched, "met": self.met, "of": len(POINTS),
                "points": [p.as_dict() for p in self.points],
                "blocks_b": not self.launched}

    def text(self) -> str:
        mark = {MET: "✓", FAILED: "✗", UNCHECKED: "?", VACUOUS: "∅"}
        lines = [f"Запуск состоялся: {'да' if self.launched else 'нет'} "
                 f"({self.met} из {len(POINTS)})"]
        for p in self.points:
            value = "" if p.value is None else f"  [{p.value}{' ' + p.unit if p.unit else ''}]"
            lines.append(f"  {mark.get(p.verdict, '?')} {p.number}. {p.name}{value}")
            if p.why:
                lines.append(f"      {p.why}")
            if p.fix:
                lines.append(f"      добыть: {p.fix}")
        if not self.launched:
            lines.append("\nБлок B не начинается: критерий A4 требует все семь пунктов.")
        return "\n".join(lines)


def _entries(session: Path, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Записи журнала сессии как словари. Пусто — журнала нет или он пуст."""
    wanted = None if kinds is None else set(kinds)
    out: list[dict[str, Any]] = []
    for path in sorted(session.glob("journal/branches/*/entries.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if wanted is None or rec.get("kind") in wanted:
                out.append(rec)
    return out


def _meta(session: Path) -> dict[str, Any]:
    path = session / "session.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _measurement(measurements: Path, name: str) -> dict[str, Any] | None:
    path = measurements / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# --- пункты -----------------------------------------------------------------


def _point_1(session: Path, meta: dict[str, Any], *, hour_seconds: float) -> Point:
    name = POINTS[0][1]
    if not meta:
        return Point(1, name, UNCHECKED, why="сессии нет: нечего читать",
                     fix="harness record --seconds 3600")
    if meta.get("synthetic", True):
        return Point(1, name, UNCHECKED, value=None,
                     why="запись синтетическая, а пункт требует настоящего экрана",
                     fix="harness record на машине с дисплеем")
    frames = _entries(session, ["frame"])
    if not frames:
        return Point(1, name, UNCHECKED, why="в журнале нет кадров",
                     fix="harness record --seconds 3600")
    stamps = [f.get("stamp", {}) for f in frames]
    turns = len(stamps)
    fps = float(meta.get("profile", {}).get("parameters", {}).get("capture_fps", 0.0))
    seconds = turns / fps if fps > 0 else 0.0
    interventions = len(_entries(session, ["intervention"]))
    if interventions:
        return Point(1, name, FAILED, value=interventions, unit="вмешательств",
                     why="работа прерывалась исследователем")
    if seconds < hour_seconds:
        return Point(1, name, FAILED, value=round(seconds), unit="с",
                     why=f"меньше объявленного часа ({hour_seconds:.0f} с)")
    return Point(1, name, MET, value=round(seconds), unit="с",
                 why="без вмешательств")


def _point_2(session: Path, meta: dict[str, Any]) -> Point:
    name = POINTS[1][1]
    frames = _entries(session, ["frame"])
    if not frames:
        return Point(2, name, UNCHECKED, why="в журнале нет кадров",
                     fix="harness record")
    turns = len(frames)
    worlds = [int(f.get("stamp", {}).get("t_world", 0)) for f in frames]
    span = max(worlds) - min(worlds) + 1 if worlds else 0
    gaps = len(_entries(session, ["capture_gap"]))
    if span != turns:
        return Point(2, name, FAILED, value=f"{turns} против {span}",
                     unit="оборотов против тиков",
                     why="мир уходил вперёд без оборота: где-то была пауза")
    if gaps:
        return Point(2, name, FAILED, value=gaps, unit="разрывов захвата",
                     why="кадры терялись: обороты и тики совпали лишь по счёту")
    return Point(2, name, MET, value=turns, unit="оборотов",
                 why="обороты равны тикам мира, разрывов нет")


def _point_3(measurements: Path, *, max_new: float) -> Point:
    name = POINTS[2][1]
    data = _measurement(measurements, "recognition")
    source = "recognition.json"
    if data is None:
        data = _measurement(measurements, "fingerprint")
        source = "fingerprint.json"
    if data is None:
        return Point(3, name, UNCHECKED, why="замера насыщения нет",
                     fix="python3 tools/measure_recognition.py")
    tails: list[float] = []
    for row in data.get("rows", []):
        tail = row.get("new_per_observation_tail")
        if isinstance(tail, (int, float)):
            tails.append(float(tail))
    for row in data.get("saturation", []) if isinstance(data.get("saturation"), list) else []:
        tail = row.get("new_per_observation_tail")
        if isinstance(tail, (int, float)):
            tails.append(float(tail))
    if not tails:
        return Point(3, name, UNCHECKED, why=f"в {source} нет хвоста кривой насыщения",
                     fix="python3 tools/measure_recognition.py")
    worst = max(tails)
    if worst > max_new:
        return Point(3, name, FAILED, value=round(worst, 4), unit="новых узлов на наблюдение",
                     why=f"порог {max_new}; кривая не выходит на полку ({source})")
    return Point(3, name, MET, value=round(worst, 4), unit="новых узлов на наблюдение",
                 why=f"худшее сочетание укладывается в {max_new} ({source})")


def _point_4(session: Path) -> Point:
    name = POINTS[3][1]
    sleeps = _entries(session, ["sleep"])
    if not sleeps:
        return Point(4, name, UNCHECKED, why="снов в журнале нет",
                     fix="прогон с консолидацией: harness live")
    grew: list[str] = []
    merged = 0
    for s in sleeps:
        e = s.get("event", {})
        before, after = e.get("entities_before"), e.get("entities_after")
        if before is None or after is None:
            continue
        merged += len(e.get("merges_applied", []) or [])
        if after > before:
            grew.append(f"{before}→{after}")
    if grew:
        return Point(4, name, FAILED, value=len(grew), unit="снов с ростом",
                     why=f"число сущностей выросло: {', '.join(grew[:3])}")
    if not merged:
        return Point(4, name, VACUOUS, value=0, unit="слияний",
                     why="сущностей не прибавилось, но и слияний не было: "
                         "падать было нечему, и пункт сошёлся сам собой")
    return Point(4, name, MET, value=merged, unit="слияний",
                 why="падение идёт за счёт слияния")


def _point_5(measurements: Path, *, ceiling: float) -> Point:
    name = POINTS[4][1]
    data = _measurement(measurements, "cascade")
    if data is None:
        return Point(5, name, UNCHECKED, why="замера каскада нет",
                     fix="python3 tools/measure_cascade.py")
    outcome = data.get("outcome", {})
    over = int(outcome.get("runs_over_launch_ceiling", 0))
    share = outcome.get("model_share_median_all")
    vacuous_on = outcome.get("launch_ceiling_met_but_vacuous_on") or []
    if over:
        return Point(5, name, FAILED, value=over, unit="прогонов выше потолка",
                     why=f"потолок {ceiling}")
    if vacuous_on:
        return Point(5, name, VACUOUS, value=share, unit="доля",
                     why=f"на доменах {', '.join(vacuous_on)} ступень недостижима по "
                         "построению: ноль означает «механизм не работал»")
    return Point(5, name, MET, value=share, unit="доля",
                 why=f"ниже потолка {ceiling} и не вакуумно")


def _point_6(session: Path) -> Point:
    name = POINTS[5][1]
    meta = _meta(session)
    if not meta:
        return Point(6, name, UNCHECKED, why="сессии нет", fix="harness record")
    breaches = _entries(session, ["resource"])
    codes = [b.get("event", {}).get("code") for b in breaches]
    money = [c for c in codes if c in {"spend_cap", "model_rate"}]
    if money:
        return Point(6, name, FAILED, value=len(money), unit="упоров в бюджет",
                     why=f"упоры: {', '.join(sorted(set(money)))}")
    return Point(6, name, MET, value=0, unit="упоров в бюджет",
                 why="в объявленный бюджет уложились")


def _point_7(session: Path) -> Point:
    name = POINTS[6][1]
    goals = _entries(session, ["goal"])
    if not goals:
        return Point(7, name, UNCHECKED, why="целей в журнале нет",
                     fix="прогон с драйвами: harness live")
    passed = [g for g in goals if g.get("event", {}).get("code") == "passed"]
    if not passed:
        codes = sorted({g.get("event", {}).get("code") for g in goals})
        return Point(7, name, FAILED, value=0, unit="закрытых целей",
                     why=f"цели есть, пройденных нет: {', '.join(c for c in codes if c)}")
    return Point(7, name, MET, value=len(passed), unit="закрытых целей",
                 why="тест удовлетворения пройден")


def evaluate(session: Path | None = None, *, measurements: Path | None = None,
             profile: Any = None) -> Verdict:
    """Проверить критерий запуска. Отсутствие данных — исход, а не исключение."""
    session = Path(session) if session is not None else Path("НЕТ")
    # Корень репозитория от этого файла: src/harness/launch.py → на три уровня вверх.
    root = Path(__file__).resolve().parent.parent.parent
    measurements = (Path(measurements) if measurements is not None
                    else root / "docs" / "measurements")

    if profile is not None:
        p = profile.parameters
        hour = float(p["launch_hour_seconds"])
        ceiling = float(p["launch_max_deep_share"])
        max_new = float(p["launch_max_new_per_observation"])
    else:
        from .core.profile import from_schema

        p = from_schema("критерий-запуска").parameters
        hour = float(p["launch_hour_seconds"])
        ceiling = float(p["launch_max_deep_share"])
        max_new = float(p["launch_max_new_per_observation"])

    meta = _meta(session)
    points = (
        _point_1(session, meta, hour_seconds=hour),
        _point_2(session, meta),
        _point_3(measurements, max_new=max_new),
        _point_4(session),
        _point_5(measurements, ceiling=ceiling),
        _point_6(session),
        _point_7(session),
    )
    return Verdict(points)
