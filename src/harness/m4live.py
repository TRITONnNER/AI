"""Прогон М4 по живому материалу: три вопроса и опись расхождений. TASK-22.

Записывает оператор; здесь только прогон. Главное свойство этого модуля — **список
проверок объявлен до появления записей**. Это не оформление: замер, чей состав выбирают
после того, как увидели данные, ничего не опровергает. Список ниже (`CHECKS`) зафиксирован
до первой живой записи, и каждая проверка объявляет, чего от неё ждут и по какому основанию.

## Три части задачи и почему они здесь вместе

1. **Разделение себя и мира** — тот же код, без правок, против опорного числа 0.667 и
   ожидания 0.51 с границами из `MEASUREMENT.md` §18. Считает `corpus.live`; здесь только
   собрано и напечатано.
2. **Тождество карточек на живых отпечатках** — та же механика слияния и расщепления, что в
   TASK-14, но с третьим источником отпечатка: значения приходят из настоящих пикселей.
   Истина остаётся у исследователя, как и там.
3. **Опись расхождений** — то, чего синтетика не показывала. **Инвентаризация, а не
   починка.** Чинить следующей задачей, зная полный список: правка, сделанная по первому
   найденному расхождению, обычно ломает второе, а второго ещё никто не видел.

## Почему опись — отдельная вещь, а не «остальные числа»

У каждой проверки есть значение, **полученное на синтетике**, и оно записано здесь заранее.
Расхождение — это не «плохое число», а разница между тем, что показывала синтетика, и тем,
что показал экран. Без записанного ожидания живое число сравнивать не с чем, и любой его
исход выглядел бы приемлемым.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .corpus.live import (KINDS, LiveError, LiveScore, bench_live,
                          compare_to_synthetic, mixed_sources, refute)


class M4Error(RuntimeError):
    """Прогнать нечего или нечем. Текст обязан говорить, что сделать."""


@dataclass(frozen=True, slots=True)
class Check:
    """Одна проверка описи: что смотрим, что показала синтетика, чем это мерено."""

    key: str
    what: str                    # что проверяется, словами
    synthetic: str               # что показала синтетика — записано до живых записей
    unit: str                    # единица независимости
    matters: str                 # почему расхождение здесь важно


#: **Опись объявлена заранее.** Двенадцать проверок; ни одна не добавлена после того, как
#: живые числа стали известны, и это проверяется тестом по дате коммита не хуже, чем
#: совестью: список лежит в коде, а код в журнале правок.
CHECKS: tuple[Check, ...] = (
    Check("iou", "IoU разделения себя и мира против сопоставимого домена",
          "медиана 0.667 по пяти доменам, разброс 0.196–0.815", "домен",
          "это то число, ради которого весь корпус и записывается"),
    Check("trivial", "перебит ли тривиальный ответ «весь кадр — экран»",
          "на синтетике истина занимает 5–40 % кадра, тривиальный ответ низок", "запись",
          "не перебил — метод не нашёл ничего, и IoU сравнивать не с чем"),
    Check("signal", "каким признаком решено: параллакс, неподвижность, связь или все",
          "игра и документ — все три; рабочий стол и видео — неподвижность", "запись",
          "признак, выигравший не тот, что в сопоставимом домене, означает, что метод "
          "работает по другой причине, и число сравнимо только по виду"),
    Check("decided", "доля пикселей, про которые метод хоть что-то решил",
          "на синтетике 96–100 %", "запись",
          "низкая доля решённых при высоком IoU — это IoU по кусочку кадра"),
    Check("disagreement", "доля пикселей, где применимые признаки ответили по-разному",
          "0.1 % в игре, 91 % в мире с глубиной", "запись",
          "расхождение признаков — встроенный сторож: он показывает, что кто-то врёт, "
          "не говоря кто"),
    Check("places", "сколько мест находит граф по отпечаткам живых кадров",
          "на синтетике места растут с каждым шагом, фронтир не кончается (§13.2)",
          "запись",
          "одно место на всю запись означает, что отпечаток не различает вида; "
          "место на каждый кадр — что различает шум"),
    Check("identity_live", "ошибка тождества до и после пересмотра на живом отпечатке",
          "на двух синтетических источниках ошибка падает или остаётся нулевой",
          "источник отпечатка",
          "механика тождества не должна зависеть от качества отпечатка; если зависит, "
          "живая колонка выродится"),
    Check("integrity", "цела ли запись: цепочка хешей, кадры, часы",
          "на синтетике цела всегда: правка любой строки обнаруживается", "запись",
          "испорченную запись нельзя ни чинить, ни считать: её числа ничего не значат"),
    Check("fitness", "годна ли запись для того, ради чего делалась",
          "у stillness обязательны отметки «без изменений»; в первом замере их было ноль",
          "запись",
          "целая и негодная запись — обычный исход, и путать её с испорченной нельзя"),
    Check("era", "совпадает ли профиль записи с текущей схемой",
          "синтетические записи пишутся текущей схемой всегда", "запись",
          "запись другой эпохи сравнима только после решения, чем были недостающие "
          "настройки"),
    Check("clock_gap", "расходятся ли часы агента и мира",
          "на синтетике t_self и t_world идут вместе: один оборот — один тик", "запись",
          "разрыв часов на живом — свойство машины, и он влияет на все временные оценки"),
    Check("marked", "сколько записей размечено вручную",
          "на синтетике истина известна всегда, разметка не нужна", "корпус",
          "без разметки IoU не считается вовсе, и это не «плохо», а «нечем сравнить»"),
)

BY_KEY: dict[str, Check] = {c.key: c for c in CHECKS}


@dataclass(slots=True)
class Finding:
    """Что показала одна проверка на одной записи. **Ни одной правки по итогам.**"""

    key: str
    where: str                   # имя записи или «корпус»
    got: Any
    verdict: str                 # "совпало" | "разошлось" | "нечем проверить"
    why: str = ""

    def as_dict(self) -> dict[str, Any]:
        c = BY_KEY[self.key]
        return {"key": self.key, "what": c.what, "synthetic": c.synthetic,
                "unit": c.unit, "matters": c.matters,
                "where": self.where, "got": self.got,
                "verdict": self.verdict, "why": self.why}


#: Исходы проверки. Их три, а не два, по той же причине, по которой у конвейера убеждений
#: три исхода: «нечем проверить» — законный ответ, и терять его нельзя.
VERDICTS: tuple[str, ...] = ("совпало", "разошлось", "нечем проверить")


@dataclass(slots=True)
class Report:
    """Итог прогона: три части задачи плюс опись. Печатается и пишется в json."""

    corpus: Path
    sessions: list[str] = field(default_factory=list)
    layers: dict[str, Any] = field(default_factory=dict)      # часть 1
    identity: dict[str, Any] = field(default_factory=dict)    # часть 2
    findings: list[Finding] = field(default_factory=list)     # часть 3
    refused: str = ""

    @property
    def diverged(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict == "разошлось"]

    @property
    def unchecked(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict == "нечем проверить"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus": str(self.corpus), "sessions": self.sessions,
            "refused": self.refused,
            "checks_declared": [
                {"key": c.key, "what": c.what, "synthetic": c.synthetic,
                 "unit": c.unit, "matters": c.matters} for c in CHECKS],
            "layers": self.layers, "identity": self.identity,
            "findings": [f.as_dict() for f in self.findings],
            "diverged": len(self.diverged), "unchecked": len(self.unchecked),
            "note": "опись, а не починка: расхождения перечислены и не исправлены. "
                    "Правка по первому найденному обычно ломает второе, а второго ещё "
                    "никто не видел",
        }

    def render_text(self) -> str:
        rows: list[str] = []
        if self.refused:
            rows.append(self.refused)
            rows.append("")
            rows.append("Опись проверок объявлена заранее и ждёт записей — "
                        f"{len(CHECKS)} проверок:")
            for c in CHECKS:
                rows.append(f"  {c.key:<14} {c.what}")
                rows.append(f"  {'':<14} на синтетике: {c.synthetic} "
                            f"(единица — {c.unit})")
            return "\n".join(rows)

        rows.append(f"Корпус: {self.corpus} — записей {len(self.sessions)}")
        rows.append("")
        rows.append("Часть 1. Разделение себя и мира")
        if self.layers.get("verdict"):
            rows.append(f"  {self.layers['verdict']}")
        for row in self.layers.get("per_session", []):
            rows.append(f"  {row}")
        rows.append("")
        rows.append("Часть 2. Тождество карточек по источникам отпечатка")
        for row in self.identity.get("rows", []):
            rows.append(f"  {row}")
        if self.identity.get("why"):
            rows.append(f"  {self.identity['why']}")
        rows.append("")
        rows.append(f"Часть 3. Опись: расхождений {len(self.diverged)}, "
                    f"нечем проверить {len(self.unchecked)}, "
                    f"всего проверок {len(self.findings)}")
        for f in self.findings:
            mark = {"разошлось": "РАСХОЖДЕНИЕ", "совпало": "совпало",
                    "нечем проверить": "нечем"}[f.verdict]
            rows.append(f"  [{mark:^12}] {f.key:<14} {f.where:<22} {f.got}")
            if f.why:
                rows.append(f"  {'':<16} {f.why}")
        rows.append("")
        rows.append("Ничего из перечисленного здесь не исправлено намеренно: первый "
                    "прогон — опись. Чинить следующей задачей, зная полный список.")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Часть 2: третий источник отпечатка — живые пиксели
# ---------------------------------------------------------------------------


class LiveFingerprint:
    """Отпечаток из настоящих кадров. Третий источник рядом с `Quantized` и `Exact`.

    Зачем он нужен именно таким. Механика тождества проверяется на мире с **известной
    истиной**: столько-то сущностей, столько-то встреч каждой. Истину даёт постановка, а
    отпечаток — то, чем сущность опознаётся. На синтетике отпечаток берётся от скаляра, и
    его свойства известны по построению; на живом экране они неизвестны, и именно это надо
    измерить.

    Поэтому сущностям сопоставляются **области настоящего кадра**, а встречам — те же
    области в других кадрах. Слипание и дробление тогда возникают не потому, что их
    подстроили, а потому что так выглядят настоящие пиксели: две области экрана могут дать
    один отпечаток, а одна область в разных кадрах — разные.

    **Кадры берутся из спокойного отрезка записи, и это не удобство.** Первая редакция
    брала подряд идущие кадры любого отрезка, и на движущемся мире получила 33 карточки
    при шести сущностях. Число выглядело результатом и им не было: на прокручивающемся
    экране фиксированная полоса — **не одна и та же вещь**, её содержимое меняется целиком,
    и «шесть сущностей по восемь встреч» в такой постановке просто ложь о мире. Замер был
    вакуумен по свойству постановки (инвариант 27), а не показывал дефект механики.
    Поэтому встречи берутся там, где область действительно та же: на отрезке, где экран
    почти не менялся. Если такого отрезка в записи нет, ответ — «нечем проверить», и это
    само по себе находка про запись.

    `near` сравнивает отпечатки по числу совпавших ячеек сетки: доля несовпавших ячеек не
    больше `tolerance`. Скалярная корзина `fingerprint_bucket` тут не годится — она про
    числа, а здесь сетка уровней.
    """

    name = "live"

    def __init__(self, frames: list[np.ndarray], *, regions: int = 6,
                 tolerance: float = 0.1) -> None:
        if not frames:
            raise M4Error("живых кадров нет: отпечаток строить нечем")
        self.frames = frames
        self.regions = int(regions)
        self.tolerance = float(tolerance)
        self._prints: dict[str, tuple[int, ...]] = {}

    def region_of(self, index: int) -> tuple[int, int, int, int]:
        """Какая часть кадра отведена сущности. Полосы по вертикали, по числу сущностей.

        Полосы, а не случайные прямоугольники: положение областей должно быть
        воспроизводимо между прогонами, иначе два прогона одного корпуса дадут разные
        числа, и сравнить их будет нельзя.
        """
        h, w = self.frames[0].shape[:2]
        band = max(1, h // max(1, self.regions))
        top = (index % self.regions) * band
        return (0, min(top, h - band), w, band)

    def __call__(self, payload: Any) -> str:
        """`payload` — `(индекс сущности, индекс встречи)`. Возвращает отпечаток строкой."""
        from .model.places import fingerprint

        idx, sighting = (int(payload[0]), int(payload[1]))
        frame = self.frames[sighting % len(self.frames)]
        left, top, width, height = self.region_of(idx)
        patch = frame[top:top + height, left:left + width]
        cells = fingerprint(patch)
        key = "".join(f"{c:x}" if 0 <= c < 16 else "z" for c in cells)
        self._prints[key] = cells
        return key

    def near(self, a: str, b: str) -> bool:
        """Близки ли отпечатки: доля несовпавших ячеек не больше порога."""
        if a == b:
            return True
        ca, cb = self._prints.get(a), self._prints.get(b)
        if ca is None or cb is None or len(ca) != len(cb):
            return False
        differ = sum(1 for x, y in zip(ca, cb) if x != y)
        return (differ / max(1, len(ca))) <= self.tolerance


def frames_of(path: Path, *, limit: int = 120) -> list[np.ndarray]:
    """Кадры записи для построения отпечатков. Не больше `limit`: нужен вид, не корпус."""
    from .session import Session

    out: list[np.ndarray] = []
    with Session.open(path) as s:
        for _, image in s:
            out.append(np.asarray(image))
            if len(out) >= limit:
                break
    return out


#: Насколько кадр должен быть похож на предыдущий, чтобы отрезок считался спокойным. Доля
#: пикселей, изменившихся больше чем на `QUIET_LEVEL` уровней яркости.
#:
#: Числа объявлены здесь, а не подобраны: 2 уровня — предел дизеринга и субпиксельного
#: сглаживания (TASK-11, замер порчи), 2 % площади — курсор с запасом (он занимает 0.09 %).
#: Отрезок, где меняется меньше, — это «та же картинка», и именно там область экрана
#: остаётся одной и той же вещью.
QUIET_LEVEL = 2
QUIET_SHARE = 0.02


def quiet_run(frames: list[np.ndarray]) -> tuple[int, int]:
    """Самый длинный спокойный отрезок: `(начало, длина)`.

    Спокойный — значит соседние кадры почти совпадают. Нужен затем, что на движущемся
    экране фиксированная область не является одной и той же вещью, и «встречи одной
    сущности» там были бы выдумкой (см. `LiveFingerprint`).
    """
    if not frames:
        return (0, 0)
    best_start = best_len = 0
    start = 0
    run = 1
    for i in range(1, len(frames)):
        a, b = frames[i - 1], frames[i]
        if a.shape != b.shape:
            changed = 1.0
        else:
            diff = np.abs(a.astype(np.int16) - b.astype(np.int16)) > QUIET_LEVEL
            changed = float(diff.mean())
        if changed <= QUIET_SHARE:
            run += 1
        else:
            if run > best_len:
                best_start, best_len = start, run
            start, run = i, 1
    if run > best_len:
        best_start, best_len = start, run
    return (best_start, best_len)


def identity_on_sources(live_frames: list[np.ndarray] | None,
                        *, sightings: int = 8) -> dict[str, Any]:
    """Ошибка тождества до и после пересмотра, **по каждому источнику отдельно**.

    Мир один и тот же для всех источников — иначе разница между колонками включала бы
    разницу миров. Истина известна по построению и в механику не попадает.
    """
    from .core.profile import from_schema
    from .model.beliefs import BeliefStore, EntityKind, Origin, Provenance, entity_id
    from .model.identity import Exact, Quantized, revise

    n_true = 6
    profile = from_schema("тождество-живое", split_min_observations=6,
                          split_middle_band=0.12, identity_min_shared_keys=1,
                          identity_min_confidence=0.6, identity_max_passes=6)
    sources: list[Any] = [Quantized(bucket=0.1), Exact()]
    quiet = (0, 0)
    if live_frames:
        quiet = quiet_run(live_frames)
        if quiet[1] >= sightings:
            start, length = quiet
            sources.append(LiveFingerprint(live_frames[start:start + length],
                                           regions=n_true))

    rows: list[str] = []
    cases: list[dict[str, Any]] = []
    for fp in sources:
        store = BeliefStore("м4-живое")
        truth: dict[str, str] = {}
        seq = 1
        for t in range(n_true):
            outcome = t % 2 == 0
            for i in range(sightings):
                # Полезная нагрузка отличается по источнику, потому что источники берут
                # разное: скаляр или (сущность, встреча). Мир при этом один: те же
                # `n_true` сущностей с теми же исходами действия.
                payload = ((t, i) if fp.name == "live" else [0.3 + t * 0.7])
                print_ = fp(payload)
                ent = entity_id(f"{fp.name}|{print_}")
                store.touch(ent, str(EntityKind.THING), seq, fingerprint=print_,
                            key="нажать", outcome=outcome, source=fp.name)
                store.learn_affordance(ent, "нажать", outcome,
                                       Provenance(Origin.EXPERIENCE, "м4-живое", seq),
                                       kind=str(EntityKind.THING))
                truth.setdefault(ent, f"сущность-{t}")
                if truth[ent] != f"сущность-{t}":
                    truth[ent] = "слипшаяся"
                seq += 1
        glued = sum(1 for v in truth.values() if v == "слипшаяся")
        rep = revise(store, profile=profile, source=fp.name, fp=fp)
        case = rep.as_dict()
        case.update({"true_entities": n_true, "glued_before": glued,
                     "error_before": abs(rep.before - n_true),
                     "error_after": abs(rep.after - n_true)})
        cases.append(case)
        rows.append(f"{fp.name:<10} карточек {rep.before:>3} → {rep.after:<3} "
                    f"(истина {n_true}), слияний {rep.merges}, расщеплений {rep.splits}, "
                    f"ошибка {case['error_before']} → {case['error_after']}")
    if live_frames and quiet[1] < sightings:
        rows.append(
            f"live       нечем проверить: самый длинный спокойный отрезок записи — "
            f"{quiet[1]} кадров, а встреч одной сущности нужно {sightings}. На "
            "движущемся экране фиксированная область — не одна и та же вещь, и «встречи "
            "одной сущности» были бы выдумкой")
    return {
        "cases": cases, "rows": rows,
        "quiet_start": quiet[0], "quiet_len": quiet[1],
        "quiet_rule": (f"кадр считается тем же, если больше {QUIET_SHARE:.0%} пикселей "
                       f"не изменились сильнее {QUIET_LEVEL} уровней яркости"),
        "unit": "карточка", "unit_for_method": "источник отпечатка",
        "n_sources": len(sources),
        "why": ("единица для утверждения о механике — источник отпечатка, и их "
                f"{len(sources)}: внутри источника все карточки опознаны одной функцией, "
                "то есть независимое наблюдение о методе там одно"),
    }


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------


def run(corpus: Path, *, method: str = "arbiter",
        on_step: Callable[[str], None] | None = None) -> Report:
    """Прогнать всё, что М4 умеет, по живому корпусу. Отказ — если записей нет."""
    from .corpus.live import LiveError as _LE
    from .session import Session

    corpus = Path(corpus)
    rep = Report(corpus=corpus)
    sessions = sorted(p for p in corpus.glob("*") if (p / "session.json").exists()) \
        if corpus.exists() else []
    rep.sessions = [p.name for p in sessions]
    if not sessions:
        rep.refused = (
            f"в {corpus} нет ни одной записи харнесса. Записать и принять:\n"
            "  harness record --plan            # что записывать и какими командами\n"
            "  harness ingest ЗАПИСЬ --corpus " + str(corpus) + " --kind ВИД\n"
            "Прогонять нечего — и это единственный честный ответ, а не ноль.")
        return rep

    # --- часть 1 ------------------------------------------------------------
    scores: list[LiveScore] = []
    for path in sessions:
        if on_step is not None:
            on_step(f"разделение слоёв: {path.name}")
        try:
            scores.append(bench_live(path, method=method))
        except _LE as e:
            rep.findings.append(Finding("iou", path.name, None, "нечем проверить",
                                        f"прогон не состоялся: {e}"))
    if scores:
        mixed = mixed_sources(scores)
        if mixed:
            rep.layers = {"verdict": "сведение отказано: " + mixed, "per_session": []}
        else:
            cmp = compare_to_synthetic(scores)
            rep.layers = {
                "verdict": cmp["verdict"],
                "reference": cmp.get("reference_text", ""),
                "expectation": cmp.get("expectation_text", ""),
                "per_session": [
                    f"{s.kind:<12} IoU "
                    + ("не считается: разметки нет" if s.iou is None
                       else f"{s.iou:.3f} (домен «{s.matched_domain}» даёт "
                            f"{s.matched_iou:.3f})")
                    for s in scores],
                "checks": cmp.get("checks", []),
            }

    # --- опись по каждой записи --------------------------------------------
    for path, score in zip(sessions, scores):
        rep.findings.extend(_findings_for(path, score))

    # --- часть 2 ------------------------------------------------------------
    if on_step is not None:
        on_step("тождество на трёх источниках отпечатка")
    frames = frames_of(sessions[0])
    rep.identity = identity_on_sources(frames)
    live_case = next((c for c in rep.identity["cases"] if c["source"] == "live"), None)
    if live_case is None:
        rep.findings.append(Finding(
            "identity_live", "корпус",
            f"спокойный отрезок {rep.identity.get('quiet_len', 0)} кадров",
            "нечем проверить",
            "в записи нет отрезка, где область экрана остаётся той же вещью достаточно "
            "долго. Это находка про запись, а не про механику: считать по движущемуся "
            "экрану значило бы выдумать встречи"))
    else:
        worse = live_case["error_after"] > live_case["error_before"]
        rep.findings.append(Finding(
            "identity_live", "корпус",
            f"ошибка {live_case['error_before']} → {live_case['error_after']}",
            "разошлось" if worse else "совпало",
            "пересмотр увеличил ошибку тождества на живом отпечатке" if worse else
            "пересмотр не увеличил ошибку, как и на синтетических источниках"))

    # --- опись по корпусу ---------------------------------------------------
    marked = sum(1 for s in scores if s.iou is not None)
    rep.findings.append(Finding(
        "marked", "корпус", f"{marked} из {len(sessions)}",
        "нечем проверить" if marked == 0 else "совпало",
        "без разметки IoU не считается вовсе — это «нечем сравнить», а не «плохо»"
        if marked == 0 else ""))
    return rep


def _findings_for(path: Path, score: LiveScore) -> list[Finding]:
    """Опись по одной записи. Ни одной правки: только запись того, что видно."""
    from .session import Session

    out: list[Finding] = []
    name = path.name

    # IoU и тривиальный ответ — через тот же `refute`, что и в `bench-live`: два места,
    # решающих одно, разошлись бы.
    check = refute(score)
    out.append(Finding("iou", name,
                       None if score.iou is None else round(score.iou, 4),
                       "нечем проверить" if score.iou is None
                       else ("разошлось" if check["outcome"].startswith("опровергнуто")
                             else "совпало"),
                       check.get("why", "")))
    out.append(Finding("trivial", name,
                       None if score.trivial is None else round(score.trivial, 4),
                       "нечем проверить" if score.trivial is None
                       else ("совпало" if score.beats_trivial else "разошлось"),
                       "" if score.beats_trivial else
                       "тривиальный ответ не перебит с запасом"))

    expected_signal = _EXPECTED_SIGNAL.get(score.matched_domain, "")
    out.append(Finding("signal", name, score.signal,
                       "нечем проверить" if not expected_signal
                       else ("совпало" if score.signal == expected_signal
                             else "разошлось"),
                       f"в домене «{score.matched_domain}» синтетика решала признаком "
                       f"«{expected_signal}»" if expected_signal else
                       "сопоставимого домена нет"))
    out.append(Finding("decided", name, round(score.decided, 4),
                       "совпало" if score.decided >= 0.96 else "разошлось",
                       "на синтетике решено 96–100 % пикселей"))
    out.append(Finding("disagreement", name,
                       None if score.disagreement is None
                       else round(score.disagreement, 4),
                       "нечем проверить" if score.disagreement is None
                       else "совпало",
                       "признак был один: сравнивать голоса не с чем"
                       if score.disagreement is None else
                       "число записано; порога здесь нет намеренно — синтетика дала "
                       "от 0.1 % до 91 %, и это разброс, а не граница"))

    with Session.open(path) as s:
        report = s.verify()
        prof = s.profile
        places, clock_gap = _places_and_clock(s)
    out.append(Finding("integrity", name, report.get("problems") or "цела",
                       "разошлось" if report.get("problems") else "совпало",
                       "; ".join(report.get("problems", []))[:200]))
    marks = report.get("unchanged_marks", 0)
    fit_bad = score.kind == "stillness" and marks == 0
    out.append(Finding("fitness", name, f"отметок «без изменений» {marks}",
                       "разошлось" if fit_bad else "совпало",
                       "запись неподвижности без отметок мерила мигающий курсор, а не фон"
                       if fit_bad else ""))
    era = prof.era()
    out.append(Finding("era", name, era,
                       "совпало" if era.startswith("профиль полон") else "разошлось",
                       "" if era.startswith("профиль полон") else
                       "запись другой эпохи схемы: сравнима после решения, чем были "
                       "недостающие настройки"))
    out.append(Finding("places", name, places,
                       "разошлось" if places <= 1 else "совпало",
                       "одно место на всю запись означает, что отпечаток не различает "
                       "вида" if places <= 1 else ""))
    out.append(Finding("clock_gap", name, clock_gap,
                       "совпало" if clock_gap == 0 else "разошлось",
                       "" if clock_gap == 0 else
                       "часы агента и мира расходятся: это влияет на все временные оценки"))
    return out


#: Каким признаком синтетика решала в каждом домене. Из `benchmark.EXPECTATION`, но в
#: машинной форме: там текст для человека, здесь значение для сверки.
_EXPECTED_SIGNAL: dict[str, str] = {
    "game": "merged", "document": "merged",
    "desktop": "stillness", "video": "stillness", "depth": "merged",
}


def _places_and_clock(session: Any) -> tuple[int, int]:
    """Сколько мест по отпечаткам и на сколько расходятся часы. Оба числа — из журнала."""
    from .model.places import PlaceGraph, fingerprint

    graph = PlaceGraph()
    gaps = 0
    seq = 0
    for entry in session.journal:
        stamp = getattr(entry, "stamp", None)
        if stamp is not None and getattr(stamp, "t_self", None) is not None:
            if int(stamp.t_self) != int(stamp.t_world):
                gaps += 1
    for _, image in session:
        graph.observe(fingerprint(np.asarray(image)), seq, seconds_per_seq=1.0 / 30.0)
        seq += 1
    return len(graph.places), gaps


def write_json(rep: Report, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep.as_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path
