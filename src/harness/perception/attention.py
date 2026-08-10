"""Внимание как бюджет дефицитного ресурса. TASK-24, направление C.

До этой правки `attention_windows: 2` было константой в схеме **без потребителя**: ось
модуляции объявлена, нарезка окон существует (`vision.layers.windows`), а спрашивать окна
некому. Здесь появляется тот, кто их распределяет.

## Два источника претензий, и они не равны в правах

**Снизу — заметность.** Резкое изменение, движение, звук вне поля зрения, всплеск ошибки
предсказания. Такая претензия имеет право **перебить** уже выданное окно и не спрашивает
разрешения у планировщика: пока планировщик решает, стоит ли смотреть, смотреть уже поздно.
Это та же субсумпция, что в контурах, только про внимание.

**Сверху — релевантность цели.** Планировщик просит окно на то, что ему нужно для текущей
цели. Такая претензия ждёт своей очереди и уступает заметности.

Асимметрия не вкусовая. Заметность — про события, которые уже случились и могут не
повториться; релевантность — про то, что будет нужно и подождёт. Дать им равные права
значило бы пропускать однократные события ради предсказуемых.

## Арбитраж: ожидаемая польза, делённая на стоимость

**Польза** — ожидаемое снижение `sigma` по величине, которая влияет на решение. Не «сколько
нового», не «насколько интересно»: снижение разброса у величины, от которой зависит выбор.
Область, про которую всё уже известно (`sigma` мала), пользы не даёт, сколько бы она ни
мигала. Область, про которую ничего не известно, но её значение ни на что не влияет, тоже:
`sigma` там большая, а вес в решении нулевой.

**Стоимость** — время и одно окно из бюджета. Оба слагаемых объявлены: окно занято, пока
претензия обслуживается, и это время нельзя потратить на другое.

## Чего здесь нет

**Награды за интересное.** Любопытство в проекте выражено через ошибку предсказания, а не
через отдельную оценку «интересности»; внимание распределяется по снижению разброса, и это
та же единая валюта.

**Памяти о том, что уже смотрели.** Претензии оцениваются заново каждый такт: область, где
`sigma` не снизилась, сама перестанет запрашиваться, потому что ожидаемая польза считается
по наблюдённому снижению (`Source.observed_gain`). Отдельного списка «уже смотрел» нет —
он был бы вторым состоянием о том же и разошёлся бы с первым.

## Окно занято, пока наблюдение не окончено

В первой редакции выданные окна сбрасывались каждый такт. Ветка перебивания при этом была
**недостижима**, а не редка: перебить можно только уже выданное окно, а уже выданных к началу
такта не оставалось ни одного. Замер показал ноль срабатываний на всех трёх бюджетах — по
инварианту 26 это дефект, и он структурный, а не настроечный.

Теперь занятость переживает такт: окно освобождает `report`, то есть окончание наблюдения.
Наблюдение, длящееся дольше такта, держит окно, и именно его заметность имеет право
перебить. У перебивания появляется цена: прерванное наблюдение не даёт ничего (`cut_short`),
и эта цена считается отдельным числом — иначе право перебивать выглядело бы бесплатным.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: Откуда пришла претензия. Набор закрыт: третий источник — это изменение схемы записи.
FROM_BELOW = "снизу"      # заметность: событие уже случилось
FROM_ABOVE = "сверху"     # релевантность: понадобится для цели
ORIGINS = (FROM_BELOW, FROM_ABOVE)

#: Признаки заметности. Перечислены, потому что «резкое изменение» и «звук вне поля зрения»
#: приходят из разных модулей и меряются разным; сваливать их в одно число «заметность»
#: значило бы потерять, что именно сработало.
SALIENCE = ("резкое изменение", "движение", "звук вне поля зрения",
            "всплеск ошибки предсказания")


class AttentionError(ValueError):
    pass


@dataclass(slots=True)
class Source:
    """Источник, на который можно направить внимание, и что он давал раньше.

    `sigma` — текущий разброс величины, которую этот источник уточняет. `weight` — насколько
    эта величина влияет на решение: ноль означает «уточнять нечего для выбора», и такой
    источник пользы не даёт даже при огромном разбросе.

    `observed_gain` — среднее наблюдённое снижение `sigma` за окно. Именно оно, а не
    обещание, входит в ожидаемую пользу после первого же обслуживания: источник, ничего не
    давший, сам уходит вниз очереди без отдельного списка «уже смотрел».
    """

    id: str
    sigma: float
    weight: float = 1.0
    cost_s: float = 0.01
    grants: int = 0
    gains: list[float] = field(default_factory=list)

    @property
    def observed_gain(self) -> float | None:
        """Среднее снижение `sigma` за окно. `None` — ещё не смотрели ни разу."""
        return None if not self.gains else sum(self.gains) / len(self.gains)

    @property
    def empty(self) -> bool:
        """Смотрели и не получили ничего. Считается по наблюдению, а не по обещанию."""
        return bool(self.gains) and (self.observed_gain or 0.0) <= 1e-9

    def expected_gain(self) -> float:
        """Ожидаемое снижение разброса по величине, влияющей на решение.

        До первого обслуживания — оптимистичная оценка «половина разброса»: не потому, что
        так бывает, а потому, что неизвестный источник обязан быть попробован. Оптимизм
        живёт ровно одно обслуживание: дальше в дело идёт наблюдённое.
        """
        base = self.sigma * 0.5 if self.observed_gain is None else self.observed_gain
        return base * self.weight

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "sigma": round(self.sigma, 5),
                "weight": self.weight, "grants": self.grants,
                "observed_gain": (None if self.observed_gain is None
                                  else round(self.observed_gain, 6)),
                "empty": self.empty}


@dataclass(frozen=True, slots=True)
class Claim:
    """Претензия на окно внимания: откуда, на что и почему."""

    source: str
    origin: str
    reason: str                     # признак заметности или имя цели
    urgency: float = 1.0            # во сколько раз претензия торопится; только для «снизу»

    def __post_init__(self) -> None:
        if self.origin not in ORIGINS:
            raise AttentionError(f"нет такого источника претензии: {self.origin!r}")
        if self.origin == FROM_BELOW and self.reason not in SALIENCE:
            raise AttentionError(
                f"претензия снизу с признаком {self.reason!r}: признаки заметности "
                f"перечислены ({list(SALIENCE)}), и новый — это изменение схемы, а не "
                "строка на месте вызова")


@dataclass(slots=True)
class Grant:
    """Выданное окно: кому, по какой претензии и что из этого вышло."""

    source: str
    origin: str
    reason: str
    score: float
    preempted: str = ""            # кого выселили, если выселяли
    gain: float | None = None      # снижение sigma; None — ещё не отчитались
    ticks_held: int = 0            # сколько тактов окно занято этим наблюдением
    cut_short: bool = False        # наблюдение прервали заметностью, пользы не будет

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "origin": self.origin, "reason": self.reason,
                "score": round(self.score, 6), "preempted": self.preempted,
                "gain": None if self.gain is None else round(self.gain, 6),
                "ticks_held": self.ticks_held, "cut_short": self.cut_short}


@dataclass(slots=True)
class Attention:
    """Распределитель окон. Бюджет — `attention_windows` из профиля.

    Единица дефицита — **окно на такт**: за такт выдаётся не больше `windows` окон, и это
    то, чего не хватает. Считать дефицит во времени было бы точнее физически и бесполезно
    практически: окна нарезаются кадром, и дробить их нечем.
    """

    windows: int
    #: Считать ли пользу на стоимость. `False` — **контроль**: окна раздаются по порядку
    #: поступления, а заметность по-прежнему идёт первой. Режим объявлен полем, а не
    #: подменой метода в замере: контроль, живущий в измерительном скрипте, отличается от
    #: боевого кода не только тем, чем задумано.
    arbitrate: bool = True
    #: Бюджет, объявленный профилем, до модуляции настроением. Хранится затем, что
    #: «сузилось внимание» — утверждение об отличии от базы, а не о самом числе.
    base_windows: int = 0
    sources: dict[str, Source] = field(default_factory=dict)
    #: Занятые окна: наблюдения, начатые и ещё не окончившиеся. Переживают такт.
    granted: list[Grant] = field(default_factory=list)
    history: list[Grant] = field(default_factory=list)
    #: Накопленное снижение sigma против числа потраченных окон. Нужно затем, что итоговое
    #: снижение на насыщённом прогоне от порядка не зависит (см. `MEASUREMENT.md`, 13.6), и
    #: различать расстановки можно только по кривой, а не по её концу.
    curve: list[tuple[int, float]] = field(default_factory=list)
    #: Счётчики для отчёта. Ветка без срабатываний — дефект (инвариант 26), поэтому они
    #: считаются, а не выводятся по логам.
    requests: int = 0
    from_below: int = 0
    from_above: int = 0
    preemptions: int = 0
    refused: int = 0
    busy_skipped: int = 0          # претензия на источник, который уже под наблюдением
    cut_short: int = 0             # наблюдений прервано заметностью
    total_gain: float = 0.0

    @classmethod
    def from_profile(cls, profile: Any, sources: Iterable[Source] = (), *,
                     arbitrate: bool = True) -> "Attention":
        """Бюджет — из профиля. Единственное место чтения `attention_windows`."""
        got = cls(windows=int(profile.parameters["attention_windows"]),
                  arbitrate=arbitrate)
        got.base_windows = got.windows
        for source in sources:
            got.add(source)
        return got

    def modulate(self, modulation: Any) -> int:
        """Принять ось модуляции: возбуждение сужает внимание.

        Без этого метода ось `attention_windows` осталась бы мёртвой в том же смысле, в
        каком были мёртвы горизонт и осторожность до `Planner.modulate`: величина считается,
        бюджет берётся прямо из профиля, и настроение ни на что не влияет. Округление вниз с
        полом в одно окно: ноль окон означал бы «внимания нет вовсе», а это не сужение, а
        отключение восприятия.
        """
        self.windows = max(1, int(round(float(modulation.attention_windows))))
        return self.windows

    def add(self, source: Source) -> Source:
        if source.id in self.sources:
            raise AttentionError(f"источник {source.id} уже есть")
        self.sources[source.id] = source
        return source

    # --- арбитраж -----------------------------------------------------------

    def score(self, claim: Claim) -> float:
        """Ожидаемая польза, делённая на стоимость. Одно число, и оно объяснимо.

        Претензия снизу умножается на `urgency`: событие, которое может не повториться,
        стоит дороже равного по пользе, но повторяемого. Множитель приходит от того, кто
        заметил, а не выбирается здесь: заметность мерится в модулях восприятия.
        """
        source = self.sources.get(claim.source)
        if source is None:
            raise AttentionError(f"претензия на неизвестный источник {claim.source}")
        if not self.arbitrate:
            # Контроль: все претензии равны, порядок решает поступление. Не «арбитража
            # нет вовсе» — заметность всё равно идёт первой, — а «арбитраж без пользы».
            return 1.0
        gain = source.expected_gain()
        cost = max(1e-9, source.cost_s)
        weight = claim.urgency if claim.origin == FROM_BELOW else 1.0
        return gain * weight / cost

    def step(self, claims: Iterable[Claim]) -> list[Grant]:
        """Раздать окна на этот такт. Возвращает **занятые** окна — и старые, и новые.

        Порядок: сначала претензии снизу, потом сверху — и это **не** сортировка по
        оценке. Заметность имеет право перебить уже выданное окно, поэтому она
        рассматривается первой независимо от чисел; внутри каждой группы порядок по
        оценке. Сортировать всё одной оценкой значило бы дать планировщику право
        перекупить прерывание, а прерывание снизу разрешения не спрашивает.

        Окна, начатые на прошлых тактах и ещё не окончившиеся, остаются занятыми: их
        освобождает `report`, а не наступление следующего такта. Возвращаются они тоже —
        наблюдателю нужно знать, какие наблюдения ещё живы, потому что прерванное окно из
        этого списка исчезает, и это единственный способ узнать о прерывании.
        """
        for grant in self.granted:
            grant.ticks_held += 1
        watched = {g.source for g in self.granted}

        below = [c for c in claims if c.origin == FROM_BELOW]
        above = [c for c in claims if c.origin == FROM_ABOVE]
        below.sort(key=self.score, reverse=True)
        above.sort(key=self.score, reverse=True)
        self.requests += len(below) + len(above)
        self.from_below += len(below)
        self.from_above += len(above)

        for claim in below + above:
            if claim.source in watched:
                # Уже смотрим. Второе окно на тот же источник — не удвоенная польза, а
                # потраченный бюджет: наблюдение одно.
                self.busy_skipped += 1
                continue
            if len(self.granted) < self.windows:
                self._grant(claim)
                watched.add(claim.source)
                continue
            if claim.origin == FROM_BELOW:
                victim = self._weakest_from_above()
                if victim is not None:
                    self.granted.remove(victim)
                    watched.discard(victim.source)
                    victim.cut_short = True
                    self.cut_short += 1
                    self.preemptions += 1
                    self._grant(claim, preempted=victim.source)
                    watched.add(claim.source)
                    continue
            self.refused += 1
        return list(self.granted)

    def _weakest_from_above(self) -> Grant | None:
        """Кого выселять: слабейшее окно, выданное сверху. Окна снизу не выселяются.

        Иначе два прерывания подряд отменяли бы друг друга, и ни одно не обслуживалось бы:
        это ровно тот случай, когда «право перебивать» превращается в отсутствие работы.
        """
        candidates = [g for g in self.granted if g.origin == FROM_ABOVE]
        return min(candidates, key=lambda g: g.score) if candidates else None

    def _grant(self, claim: Claim, *, preempted: str = "") -> Grant:
        grant = Grant(source=claim.source, origin=claim.origin, reason=claim.reason,
                      score=self.score(claim), preempted=preempted)
        self.granted.append(grant)
        self.history.append(grant)
        self.sources[claim.source].grants += 1
        return grant

    def report(self, source_id: str, *, sigma_after: float) -> float:
        """Чем кончилось наблюдение: насколько снизился разброс. Освобождает окно.

        Отчёт обязателен и приходит **снаружи**: внимание не знает, что даёт наблюдение, и
        придумать снижение за наблюдателя значило бы завести пользу, которой никто не мерил.

        Отчёт по источнику, окно которого не занято, — отказ, а не ноль. Так бывает ровно в
        одном случае: наблюдение прервали заметностью, а наблюдатель об этом не узнал. Тихо
        зачесть такой отчёт значило бы приписать пользу прерванному наблюдению и стереть
        цену перебивания.
        """
        source = self.sources.get(source_id)
        if source is None:
            raise AttentionError(f"отчёт по неизвестному источнику {source_id}")
        open_grant = next((g for g in self.granted if g.source == source_id), None)
        if open_grant is None:
            raise AttentionError(
                f"отчёт по источнику {source_id}, окно которого не занято: наблюдение либо "
                "уже закрыто, либо прервано заметностью — проверьте, что вернул step()")
        gain = max(0.0, source.sigma - float(sigma_after))
        source.gains.append(gain)
        source.sigma = float(sigma_after)
        open_grant.gain = gain
        self.granted.remove(open_grant)
        self.total_gain += gain
        self.curve.append((len(self.history), self.total_gain))
        return gain

    # --- сводка -------------------------------------------------------------

    def windows_to_gain(self, target: float) -> int | None:
        """Сколько окон потрачено до достижения накопленного снижения `target`.

        `None` — не достигнуто за прогон. Это и есть неразложимый показатель арбитража:
        итоговое снижение на насыщённом прогоне одинаково при любом порядке, а **когда**
        оно набрано — нет. Порог задаёт вызывающий: потолок доступного снижения знает мир,
        а не распределитель окон.
        """
        for spent, got in self.curve:
            if got >= target:
                return spent
        return None

    def summary(self) -> dict[str, Any]:
        """Три показателя, названные в задаче, плюс то, без чего они обманывают."""
        served = [g for g in self.history if g.gain is not None]
        gains = [g.gain or 0.0 for g in served]
        wasted = [g for g in served if (g.gain or 0.0) <= 1e-9]
        below_served = [g for g in served if g.origin == FROM_BELOW]
        return {
            "windows": self.windows,
            "arbitrate": self.arbitrate,
            "unit": "окно на такт",
            "requests": self.requests,
            "from_below": self.from_below, "from_above": self.from_above,
            "granted": len(self.history), "served": len(served),
            "refused": self.refused, "preemptions": self.preemptions,
            "busy_skipped": self.busy_skipped,
            # Цена права перебивать: наблюдения, прерванные заметностью и не давшие ничего.
            "cut_short": self.cut_short,
            "cut_share": (self.cut_short / len(self.history)) if self.history else None,
            "total_gain": round(self.total_gain, 6),
            "curve": [[spent, round(got, 6)] for spent, got in self.curve],
            # 1. Доля запросов, потраченных на прерывания снизу.
            "below_share_of_grants": (len(below_served) / len(served)
                                      if served else None),
            # 2. Снижение sigma на запрос.
            "gain_per_grant": (sum(gains) / len(gains)) if gains else None,
            # 3. Сколько раз внимание ушло на источник, не давший ничего.
            "wasted_grants": len(wasted),
            "wasted_share": (len(wasted) / len(served)) if served else None,
            "empty_sources": sorted(s.id for s in self.sources.values() if s.empty),
            "sources": [s.as_dict() for s in self.sources.values()],
        }
