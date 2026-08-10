"""Драйвы из корреляции областей интерфейса с событиями. Начало М6, TASK-24, B.

Агент не получает разметки «это полоска здоровья». У него есть: экранный слой, отделённый
восприятием, и журнал того, что он делал. Из этих двух надо получить утверждение вида
«область A убывает при событиях класса X» — и не получить его там, где связи нет.

## Что здесь считается областью и почему сетка, а не сегментация

Область — ячейка сетки поверх экранного слоя. Сегментация «настоящих» элементов интерфейса
была бы догадкой о структуре, которой у агента нет: он не знает ни что такое «полоска», ни
где она кончается. Сетка не знает тоже, и это её достоинство — она не привносит структуры,
которой нет в пикселях. Ячейка попадает в наблюдение, только если она **устойчива**: лежит в
экранном слое и её содержимое меняется. Неподвижная и неизменная ячейка (фон панели) не
несёт информации; ячейка вне экранного слоя — это мир, а не интерфейс.

## Что считается классом событий

Класс — это выход, который агент удерживал (`OUT_1A2B`), то есть непрозрачный символ из его
собственного репертуара. Не «необратимое действие» и не «удар»: смыслов у агента нет, а
собственные выходы у него есть. Именно поэтому связь выучиваема: обе стороны корреляции
доступны ему без подсказок.

## Как считается связь и почему именно так

Для каждой пары (ячейка, класс) сравниваются два распределения изменений ячейки:
**после событий класса** и **во всех остальных тиках**. Величина связи — разность средних,
делённая на разброс фона (`d` Коэна). Порог и минимальное число событий — из схемы
(`link_min_effect`, `link_min_events`), а не из кода анализа: порог, живущий в анализе, не
попадает в `profile_hash` и не сравним между прогонами (инвариант 23).

**Знак важен.** «Убывает при X» и «растёт при X» — разные утверждения, и связь несёт знак.
Модуль ищет именно убывание: драйв — это дефицит, и полоска, которая от действия растёт, не
дефицит, а награда. Награды в проекте нет, поэтому растущие связи записываются, но драйва из
них не делается.

## Аллостаз, а не гомеостаз

Обнаруженный драйв получает `forecast` — где величина окажется через `drive_horizon_s`, если
ничего не делать, — и решения принимаются по `forecast_deficit`, а не по текущему значению.
Прогноз линейный по последнему наклону: сложнее пока нечем, а линейный уже отличает «падает
и упадёт» от «упало и стоит».
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .beliefs import BeliefStore, Hypothesis, Origin, Provenance
from .drives import Drive, DriveOrigin


class GaugeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Cell:
    """Ячейка сетки: где она и насколько устойчива."""

    row: int
    col: int
    top: int
    left: int
    height: int
    width: int

    @property
    def id(self) -> str:
        """Непрозрачный идентификатор области. Координат в нём нет (инвариант 4).

        Агенту достаётся именно это имя: по нему он может ссылаться на область, но не
        может узнать, где она на экране. Исследователь сопоставляет имя с координатами
        через отладочный поток.
        """
        return f"AREA_{self.row}_{self.col}"

    def value(self, frame: np.ndarray, mask: np.ndarray | None = None) -> float:
        """Уровень области: средняя яркость **по пикселям экранного слоя**.

        Маска обязательна по смыслу, хотя и необязательна по подписи. Ячейка сетки почти
        всегда содержит и интерфейс, и просвечивающий мир; мир под мышью смещается, и его
        размах яркости на порядок больше, чем вклад тонкой полоски. Средняя по всей ячейке
        поэтому измеряет движение мира, а не состояние интерфейса — и первая редакция
        именно так и не нашла связь, которая в мире заведомо была: сигнал полоски утонул в
        сотне уровней мирового шума.

        Агент к этому моменту экранный слой уже отделил, так что смотреть только на его
        пиксели — не привилегия, а использование того, что у него есть.
        """
        patch = frame[self.top:self.top + self.height,
                      self.left:self.left + self.width]
        if patch.size == 0:
            return 0.0
        if mask is None:
            return float(patch.mean())
        m = mask[self.top:self.top + self.height, self.left:self.left + self.width]
        return float(patch[m].mean()) if bool(m.any()) else 0.0


def grid_cells(shape: tuple[int, int], *, rows: int, cols: int) -> list[Cell]:
    """Сетка по кадру. Ячейки одинаковые: неравные вносили бы структуру, которой нет."""
    h, w = shape
    ch, cw = max(1, h // rows), max(1, w // cols)
    return [Cell(r, c, r * ch, c * cw, ch, cw)
            for r in range(rows) for c in range(cols)]


@dataclass(slots=True)
class Watcher:
    """Копит уровни областей и то, какие классы событий случались на каждом тике.

    Хранит не кадры, а по одному числу на ячейку: кадры за час — гигабайты, а связь
    считается по уровням. Это не оптимизация, а условие применимости: копить кадры в
    рабочем контексте нельзя (он эфемерный), а уровни можно.
    """

    cells: list[Cell]
    #: Маска экранного слоя от восприятия. Уровни считаются только по её пикселям, если
    #: `use_mask`. Выключение — законный прогон, а не поблажка: см. `use_mask`.
    mask: np.ndarray | None = None
    #: Считать уровень только по пикселям экранного слоя или по всей ячейке.
    #:
    #: **Оба прогона нужны, и разница между ними — измеряемая величина.** По экранному слою
    #: правильнее по смыслу: наблюдать состояние интерфейса надо по интерфейсу, а не по
    #: миру, просвечивающему сквозь ту же ячейку. По всей ячейке — так, как было бы без
    #: разделения слоёв вовсе, то есть контроль на то, много ли разделение даёт.
    #:
    #: Замер (`tools/measure_drives.py`, часть 2): по маске находится 6 связей при лучшей
    #: величине d = 10.6, по всей ячейке — 5 при той же величине. Разделение слоёв
    #: добавляет одну связь и не меняет силы главной; ложных связей на контрольном
    #: (молчащем) классе событий ноль в обоих прогонах.
    use_mask: bool = True
    #: Только устойчивые ячейки участвуют в корреляции: те, что задевают экранный слой и
    #: меняются. Заполняется `settle`.
    stable: list[str] = field(default_factory=list)
    series: dict[str, list[float]] = field(default_factory=dict)
    events: list[tuple[str, ...]] = field(default_factory=list)
    ticks: int = 0

    def settle(self, screen_mask: np.ndarray, *, min_screen: float = 0.2) -> list[str]:
        """Какие ячейки наблюдаются: те, что **задевают экранный слой**.

        `screen_mask` приходит от восприятия (`vision.selfworld`), а не от истины мира.
        Здесь она аргумент нарочно: модуль не должен уметь спросить истину даже случайно.

        Критерий один — положение, и это исправление, а не упрощение. Первая редакция
        требовала ещё и чтобы уровень ячейки **менялся во время прогрева**, и на этом
        отсекала ровно те области, ради которых всё делается: полоска за прогрев не
        убывала ни разу (событий класса не было), её уровень был константой, и ячейка
        объявлялась неустойчивой. Критерий, зависящий от того, случилось ли за окном
        наблюдения интересующее событие, отбрасывает область тем чаще, чем реже событие, —
        то есть работает против собственной цели.

        Постоянство уровня отсекается позже и там, где ему место: в `find_links` пара с
        нулевым разбросом фона пропускается, потому что делить не на что.

        `min_screen = 0.2` — доля пикселей ячейки в экранном слое. **Не 0.5**: тонкая
        полоска в восемь пикселей высотой не даёт большинства ни в одной ячейке сетки, и
        требовать большинства значит видеть только толстые элементы. Цена низкого порога —
        ложные связи, и она измеряется на мире без полоски, где находить нечего.
        """
        self.mask = screen_mask
        keep: list[str] = []
        for cell in self.cells:
            patch = screen_mask[cell.top:cell.top + cell.height,
                                cell.left:cell.left + cell.width]
            if patch.size and float(patch.mean()) >= min_screen:
                keep.append(cell.id)
        self.stable = keep
        self.series = {cid: [] for cid in keep}
        return keep

    def observe_levels(self, levels: dict[str, float], classes: Iterable[str]) -> None:
        """Один тик, когда уровни известны напрямую, без кадра.

        Нужно для проверки **механики** корреляции против заведомо известной истины: там
        уровни задаются постановкой, и рисовать ради них пиксели значило бы смешать проверку
        корреляции с проверкой восприятия. Восприятие проверяется отдельно — и, как
        выяснилось замером, эту связь оно не пропускает.
        """
        for cid in self.stable:
            if cid not in levels:
                raise GaugeError(f"уровень области {cid} не задан: подставить нечего")
            self.series[cid].append(float(levels[cid]))
        self.events.append(tuple(classes))
        self.ticks += 1

    def observe(self, frame: np.ndarray, classes: Iterable[str]) -> None:
        """Один тик: уровни устойчивых областей и классы событий этого тика."""
        by_id = {c.id: c for c in self.cells}
        mask = self.mask if self.use_mask else None
        for cid in self.stable:
            self.series[cid].append(by_id[cid].value(frame, mask))
        self.events.append(tuple(classes))
        self.ticks += 1


@dataclass(frozen=True, slots=True)
class Link:
    """Найденная связь: область, класс события, знак и сила. С `n` при числе."""

    area: str
    event_class: str
    effect: float                # d Коэна: (среднее после события − фон) / разброс фона
    n_events: int
    mean_after: float
    mean_background: float

    @property
    def falls(self) -> bool:
        return self.effect < 0

    def as_dict(self) -> dict[str, Any]:
        return {"area": self.area, "event_class": self.event_class,
                "effect": round(self.effect, 4), "n_events": self.n_events,
                "unit": "событие класса",
                "falls": self.falls,
                "mean_after": round(self.mean_after, 4),
                "mean_background": round(self.mean_background, 4)}

    def claim(self) -> str:
        """Как связь звучит утверждением. Ни одного смысла: только символы и знак."""
        verb = "убывает" if self.falls else "растёт"
        return f"область {self.area} {verb} при событиях класса {self.event_class}"


def find_links(watcher: Watcher, *, profile: Any,
               horizon_ticks: int = 3) -> list[Link]:
    """Связи между областями и классами событий. Порог — из профиля.

    `horizon_ticks` — сколько тиков после события считать «после». Не ноль: эффект может
    проявиться не в тот же тик, а через кадр-два, и требовать мгновенности значило бы
    находить только самые быстрые связи.
    """
    p = profile.parameters
    min_effect = float(p["link_min_effect"])
    min_events = int(p["link_min_events"])

    classes: set[str] = set()
    for tick in watcher.events:
        classes.update(tick)
    out: list[Link] = []
    for area in watcher.stable:
        xs = watcher.series[area]
        if len(xs) < 3:
            continue
        deltas = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
        for cls in sorted(classes):
            after_idx: set[int] = set()
            for t, tick in enumerate(watcher.events):
                if cls in tick:
                    # **Смещение на единицу, и оно не косметическое.** `deltas[i]` — это
                    # изменение между тиками `i` и `i+1`, то есть изменение, наблюдаемое
                    # **на** тике `i+1`. Событие на тике `t` меняет уровень уже на этом
                    # тике, поэтому его след лежит в `deltas[t-1]`. Первая редакция
                    # начинала окно с `deltas[t]` и пропускала главный след целиком: на
                    # мире с точно известной связью и падением в 76 уровней яркости
                    # детектор находил ноль связей из двух.
                    for k in range(max(0, t - 1),
                                   min(t - 1 + horizon_ticks, len(deltas))):
                        after_idx.add(k)
            after = [deltas[i] for i in sorted(after_idx)]
            background = [d for i, d in enumerate(deltas) if i not in after_idx]
            n_events = sum(1 for tick in watcher.events if cls in tick)
            if n_events < min_events or len(after) < 2 or len(background) < 2:
                continue
            sd = statistics.pstdev(background)
            if sd < 1e-9:
                # Фон без разброса: делить не на что. Не «связь бесконечна», а «сравнивать
                # нечем» — такая пара пропускается, и это видно по отсутствию строки.
                continue
            effect = (statistics.fmean(after) - statistics.fmean(background)) / sd
            if abs(effect) >= min_effect:
                out.append(Link(area=area, event_class=cls, effect=effect,
                                n_events=n_events,
                                mean_after=statistics.fmean(after),
                                mean_background=statistics.fmean(background)))
    out.sort(key=lambda link: abs(link.effect), reverse=True)
    return out


def hypothesis_of(link: Link, *, branch: str, seq: int) -> Hypothesis:
    """Связь как гипотеза: у неё есть тест, и тест выражен в действиях агента.

    Тест — «удержать этот выход и посмотреть на область»: то, что агент умеет сделать сам.
    Гипотеза без теста была бы вопросом (законное состояние), но здесь тест строится, и
    объявлять её вопросом значило бы прятать проверяемое в непроверяемое.
    """
    return Hypothesis(
        claim=link.claim(),
        test=f"удержать {link.event_class} и сравнить уровень {link.area} до и после",
        prior=min(0.9, 0.5 + abs(link.effect) / 10.0),
        provenance=Provenance(Origin.EXPERIENCE, branch, seq))


def drive_of(link: Link, *, series: list[float], profile: Any) -> Drive:
    """Драйв из связи. **Только из убывающей**: драйв — это дефицит.

    Растущая связь тоже находка, но драйва из неё не делается: «растёт, когда я делаю X» —
    это награда, а награды в проекте нет (см. `ROADMAP.md`, М6 и правило про отсутствие
    оценки полезности).

    Аллостаз: `forecast` — линейная экстраполяция на `drive_horizon_s`, а решения снаружи
    принимаются по `forecast_deficit`. Гомеостаз реагировал бы на текущее значение и
    начинал бы действовать, когда уже поздно.
    """
    if not link.falls:
        raise GaugeError(
            f"связь {link.claim()} растущая: драйв из неё не делается. Растущая связь — "
            "это награда, а не дефицит, и превращать её в драйв значило бы завести "
            "оценку полезности, которой в проекте нет")
    p = profile.parameters
    horizon_s = float(p["drive_horizon_s"])
    hz = float(p["hz_drives"])
    horizon_ticks = max(1.0, horizon_s * hz)

    scale = max(1e-9, max(series) - min(series)) if series else 1.0
    value = (series[-1] - min(series)) / scale if series else 0.0
    # Наклон по последней четверти: по всей серии он размазался бы старым поведением.
    tail = series[-max(3, len(series) // 4):] if len(series) >= 3 else series
    slope = ((tail[-1] - tail[0]) / max(1, len(tail) - 1)) / scale if len(tail) > 1 else 0.0
    forecast = max(0.0, min(1.0, value + slope * horizon_ticks))
    return Drive(name=f"дефицит:{link.area}", value=value, target=1.0,
                 forecast=forecast, origin=DriveOrigin.DISCOVERED,
                 grounded_in=f"{link.area}@{link.event_class}")


def ground(watcher: Watcher, *, profile: Any, store: BeliefStore,
           seq: int = 1) -> dict[str, Any]:
    """Полный проход: связи → гипотезы → драйвы. Возвращает числа для отчёта.

    Гипотезы кладутся в хранилище с происхождением `experience`: связь найдена в опыте, а
    не услышана. Убеждением она станет после проверки действием — этого здесь нет
    намеренно, проверка живёт в конвейере убеждений и вызывается сном.
    """
    links = find_links(watcher, profile=profile)
    drives: list[Drive] = []
    for link in links:
        store.add_hypothesis(hypothesis_of(link, branch=store.branch, seq=seq))
        if link.falls:
            drives.append(drive_of(link, series=watcher.series[link.area],
                                   profile=profile))
    return {
        "links": [link.as_dict() for link in links],
        "drives": [d.as_dict() for d in drives],
        "stable_areas": len(watcher.stable),
        "ticks": watcher.ticks,
        "unit": "связь (область × класс)",
        "note": ("порог связи и минимальное число событий — из схемы "
                 "(link_min_effect, link_min_events), а не из кода анализа"),
    }
