"""Модель перехода: что будет, если нажать это здесь.

Планировать нельзя без предсказания. Но предсказание должно браться из опыта, а не
из правил мира: правил агент не знает и знать не должен. Поэтому здесь ровно одно —
статистика по наблюдённым переходам «(место, выход) → место», выведенная из графа
мест, который сам выведен из журнала.

Три вещи, которые делают эту модель моделью, а не таблицей догадок.

1. **«Не знаю» — законный ответ.** `predict` возвращает `None` там, где перехода не
   наблюдали. Правдоподобное значение вместо реального — худшее, что здесь может
   произойти: план построится, окажется неверным, и причина будет неотличима от
   ошибки планировщика. Доля незнания видна в `stats()` и обязана быть на виду.

2. **Переход вероятностный, а не однозначный.** Одно и то же нажатие в одном и том
   же месте уводит по-разному: мир не обязан быть детерминированным, а место — это
   отпечаток вида, и разные состояния мира могут выглядеть одинаково. Поэтому у
   предсказания есть `p` — доля переходов, закончившихся в этом месте, — и список
   остальных исходов.

3. **Время измеренное, а не придуманное.** `mu_seconds` и `sigma` берутся из рёбер
   графа, то есть из того, сколько проход занимал на самом деле. Никакой стоимости
   «по умолчанию» у неизвестного перехода нет — у него нет стоимости вообще.

Чего в модели нет: координат, направлений, расстояний, представления о том, что
такое «вперёд». Только «отсюда этим — туда, за столько секунд, столько раз».
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..core.action import parse_action_key
from .places import PlaceGraph, Traversal
from .rebuild import BodyMap


@dataclass(frozen=True, slots=True)
class Outcome:
    """Один наблюдённый исход перехода."""

    dst: str
    n: int
    mu_seconds: float
    sigma: float

    def as_dict(self) -> dict[str, Any]:
        return {"dst": self.dst, "n": self.n,
                "mu_seconds": round(self.mu_seconds, 3),
                "sigma": round(self.sigma, 3)}


@dataclass(frozen=True, slots=True)
class Prediction:
    """Что модель говорит про «нажать этот выход в этом месте».

    `p` — доля переходов, закончившихся в самом частом исходе. Единица означает
    «всегда приводило сюда», а не «обязано привести»: `n` рядом, и по нему видно,
    на чём держится уверенность.
    """

    src: str
    action: str                  # ключ действия: выход вместе с удержанием
    likely: Outcome
    alternatives: tuple[Outcome, ...] = ()
    total_n: int = 0

    @property
    def p(self) -> float:
        return self.likely.n / self.total_n if self.total_n else 0.0

    @property
    def is_guess(self) -> bool:
        """Один наблюдённый проход — это догадка, а не знание."""
        return self.total_n < 2

    @property
    def certain(self) -> bool:
        return not self.alternatives and self.total_n >= 2

    def outcomes(self) -> list[tuple[Outcome, float]]:
        """Все наблюдённые исходы с их долями, самый частый первым.

        Планировщику нужны все, а не только самый частый. Одно и то же нажатие в
        одном и том же месте уводит по-разному, и ветка, отброшенная как «не самая
        вероятная», может быть единственной, ведущей к цели: на замере из-за этого
        не находилось 893 маршрута из 3205, причём каждый раз спотыкались об один
        честный переход 50/50.
        """
        total = self.total_n or 1
        return [(o, o.n / total) for o in (self.likely, *self.alternatives)]

    def as_dict(self) -> dict[str, Any]:
        return {"src": self.src, "action": self.action,
                "likely": self.likely.as_dict(), "p": round(self.p, 4),
                "total_n": self.total_n, "is_guess": self.is_guess,
                "alternatives": [o.as_dict() for o in self.alternatives]}


@dataclass(slots=True)
class ForwardModel:
    """Переходы, сгруппированные по (место, выход). Пересобираема из графа мест."""

    transitions: dict[tuple[str, str], list[Outcome]] = field(default_factory=dict)
    body: BodyMap | None = None
    asked: int = 0
    unknown: int = 0

    # --- сборка -------------------------------------------------------------

    @classmethod
    def from_graph(cls, graph: PlaceGraph, body: BodyMap | None = None) -> "ForwardModel":
        """Собрать из графа мест. Ребро графа уже несёт то, чем его прошли."""
        model = cls(body=body)
        for edge in graph.edges.values():
            model.observe(edge)
        return model

    def observe(self, edge: Traversal) -> None:
        """Добавить ребро графа как наблюдённый переход.

        Ребро берётся только если его `mode` — полный ключ действия
        (`OUT_xx@200`). Переход, про который неизвестно, **чем именно** его прошли,
        для планирования бесполезен: повторить его нечем. И длительность здесь не
        придирка — нажатие на 40 мс и на 200 мс это разные действия (инвариант 8), а
        подставить длительность «по умолчанию» значило бы предсказывать последствие
        другого действия. На замере это стоило первого же шага плана: модель обещала
        место, наблюдённое при 200 мс, планировщик жал 40 мс и попадал не туда.
        """
        if not edge.mode or parse_action_key(edge.mode) is None:
            return
        key = (edge.src, edge.mode)
        outcomes = self.transitions.setdefault(key, [])
        for i, o in enumerate(outcomes):
            if o.dst == edge.dst:
                outcomes[i] = Outcome(o.dst, o.n + edge.n,
                                      edge.mu_seconds, edge.sigma)
                break
        else:
            outcomes.append(Outcome(edge.dst, edge.n, edge.mu_seconds, edge.sigma))

    # --- предсказание -------------------------------------------------------

    def actions_from(self, place: str) -> list[str]:
        """Какими действиями пробовали уходить отсюда. Ключи, не выходы."""
        return sorted({key for (src, key) in self.transitions if src == place})

    # Прежнее имя оставлено бы соблазном забыть про длительность, поэтому его нет.

    def predict_key(self, place: str, key: str) -> "Prediction | None":
        return self.predict(place, key)

    def predict(self, place: str, key: str) -> Prediction | None:
        """Что будет от этого действия здесь. `None` — не знаю.

        «Не знаю» — не то же, что «ничего не будет».
        """
        self.asked += 1
        outcomes = self.transitions.get((place, key))
        if not outcomes:
            self.unknown += 1
            return None
        ordered = sorted(outcomes, key=lambda o: (-o.n, o.dst))
        total = sum(o.n for o in ordered)
        return Prediction(place, key, ordered[0], tuple(ordered[1:]), total)

    def caution(self, key: str) -> float:
        """Осторожность по действию: «я не умею это откатить», а не список запретов.

        Обратимость известна про выход, а не про длительность, поэтому ключ
        разбирается и берётся выход. Без карты тела осторожность максимальна:
        незнание обратимости — это и есть максимальная осторожность (инвариант 9).
        """
        if self.body is None:
            return 1.0
        parsed = parse_action_key(key)
        output = parsed[0] if parsed else key
        facts = self.body.outputs.get(output)
        return 1.0 if facts is None else facts.reversibility.caution

    # --- сводка -------------------------------------------------------------

    @property
    def unknown_rate(self) -> float:
        return self.unknown / self.asked if self.asked else 0.0

    def stats(self) -> dict[str, Any]:
        places = {src for (src, _) in self.transitions}
        certain = sum(1 for k in self.transitions if len(self.transitions[k]) == 1)
        return {"pairs": len(self.transitions), "places": len(places),
                "single_outcome_pairs": certain,
                "asked": self.asked, "unknown": self.unknown,
                "unknown_rate": round(self.unknown_rate, 4)}

    def as_dict(self) -> dict[str, Any]:
        return {"transitions": {f"{src}|{key}": [o.as_dict() for o in outs]
                                for (src, key), outs in sorted(self.transitions.items())},
                **self.stats()}


def build(graph: PlaceGraph, body: BodyMap | None = None,
          skip: Iterable[str] = ()) -> ForwardModel:
    """Модель из графа, с возможностью не брать некоторые выходы.

    `skip` нужен не для запретов, а для честных ablation-прогонов: «а если бы этого
    выхода не было». Список запрещённых действий проектом не предусмотрен. Сравнение
    идёт по выходу, а не по ключу действия: выключается выход целиком, со всеми
    длительностями.
    """
    skipped = set(skip)
    model = ForwardModel(body=body)
    for edge in graph.edges.values():
        parsed = parse_action_key(edge.mode)
        output = parsed[0] if parsed else edge.mode
        if output in skipped or edge.mode in skipped:
            continue
        model.observe(edge)
    return model
