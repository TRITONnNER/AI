"""Граф мест.

Из архитектуры: «пространство — граф мест. Узел: отпечаток вида. Ребро:
`(mode, mu_seconds, sigma, n, label)`. Никакой метрики глобально; локально в
ближнем слое — относительная геометрия».

Поэтому здесь нет ни одной координаты. Узел не знает, где он находится, — он
знает только, как выглядит. Ребро не знает расстояния — оно знает, сколько
секунд ходьбы туда уходило и с каким разбросом. «Направление» тоже отсутствует:
его агент выведет сам из того, каким действием он проходил ребро.

Отпечаток вида: кадр сжимается до сетки, яркость приводится к среднему по кадру
и квантуется. Приведение к среднему нужно, чтобы место оставалось тем же местом
при выключенном свете: темнее — не значит другое место. Сравнение — по доле
совпавших ячеек, то есть по расстоянию Хэмминга, а не по пикселям: место
узнаётся с другого угла, а попиксельно оно тогда совсем другое.

Экранный слой из отпечатка исключается, если маска известна: интерфейс одинаков
всюду и, попав в отпечаток, склеил бы все места в одно.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

GRID = 8                  # сторона сетки отпечатка
LEVELS = 4                # на сколько уровней квантуется ячейка


def fingerprint(frame: np.ndarray, *, exclude: np.ndarray | None = None,
                grid: int = GRID, levels: int = LEVELS) -> tuple[int, ...]:
    """Отпечаток вида: кортеж уровней по сетке.

    `exclude` — маска пикселей, которые в отпечаток не входят (экранный слой).
    Ячейка, полностью попавшая в маску, получает уровень −1: «не знаю». Это
    отдельное значение, а не ноль, иначе «тут интерфейс» стало бы «тут темно».
    """
    gray = frame if frame.ndim == 2 else frame[:, :, :3].mean(axis=2)
    f = gray.astype(np.float64)
    if exclude is not None:
        if exclude.shape != f.shape:
            raise ValueError(f"маска {exclude.shape} не по кадру {f.shape}")
        keep = ~exclude
    else:
        keep = np.ones(f.shape, dtype=bool)

    h, w = f.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    cells: list[float] = []
    valid: list[bool] = []
    for i in range(grid):
        for j in range(grid):
            block = f[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            mask = keep[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if mask.sum() == 0:
                cells.append(0.0)
                valid.append(False)
            else:
                cells.append(float(block[mask].mean()))
                valid.append(True)

    arr = np.asarray(cells)
    ok = np.asarray(valid)
    if ok.any():
        mean = arr[ok].mean()
        spread = arr[ok].std() or 1.0
        # Приведение к среднему и разбросу: место остаётся собой при смене
        # освещения, но перестаёт быть собой при смене вида.
        z = (arr - mean) / spread
    else:
        z = arr
    q = np.clip(((z + 2.0) / 4.0 * levels).astype(int), 0, levels - 1)
    return tuple(int(v) if ok[k] else -1 for k, v in enumerate(q))


def similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Доля совпавших ячеек. Ячейки «не знаю» не участвуют ни за, ни против."""
    if len(a) != len(b):
        raise ValueError("отпечатки разной длины: сетки не совпадают")
    pairs = [(x, y) for x, y in zip(a, b) if x >= 0 and y >= 0]
    if not pairs:
        return 0.0
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def place_id(fp: tuple[int, ...]) -> str:
    h = hashlib.blake2b(bytes((v + 1) & 0xFF for v in fp), digest_size=2).hexdigest().upper()
    return f"PLACE_{h}"


@dataclass(slots=True)
class Place:
    """Узел: отпечаток вида и когда его видели. Никаких координат."""

    id: str
    fingerprint: tuple[int, ...]
    first_seq: int
    last_seq: int
    visits: int = 1
    # Несколько отпечатков на одно место: то же место с другого угла выглядит
    # иначе, и держать один «правильный» вид было бы неверно.
    variants: list[tuple[int, ...]] = field(default_factory=list)

    def best_similarity(self, fp: tuple[int, ...]) -> float:
        return max(similarity(fp, self.fingerprint),
                   *(similarity(fp, v) for v in self.variants)) if self.variants \
            else similarity(fp, self.fingerprint)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "visits": self.visits, "first_seq": self.first_seq,
                "last_seq": self.last_seq, "variants": len(self.variants) + 1}


@dataclass(slots=True)
class Traversal:
    """Ребро: как проходили и сколько это заняло. `(mode, mu, sigma, n, label)`."""

    src: str
    dst: str
    mode: str                  # каким действием прошли; для агента непрозрачно
    mu_seconds: float = 0.0
    sigma: float = 0.0
    n: int = 0
    label: str = ""

    def observe(self, seconds: float) -> None:
        self.n += 1
        if self.n == 1:
            self.mu_seconds, self.sigma = seconds, 0.0
            return
        prev = self.mu_seconds
        self.mu_seconds += (seconds - prev) / self.n
        # Онлайновая дисперсия по Уэлфорду: без хранения всех проходов.
        self.sigma = (((self.n - 2) * self.sigma ** 2
                       + (seconds - prev) * (seconds - self.mu_seconds))
                      / max(1, self.n - 1)) ** 0.5

    @property
    def confidence(self) -> float:
        """Толщина ребра в пульте: чем больше проходов и меньше разброс, тем выше."""
        if self.n == 0:
            return 0.0
        return min(1.0, self.n / 10.0) * (1.0 / (1.0 + self.sigma))

    def as_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "mode": self.mode,
                "mu_seconds": round(self.mu_seconds, 3), "sigma": round(self.sigma, 3),
                "n": self.n, "label": self.label,
                "confidence": round(self.confidence, 4)}


class PlaceGraph:
    """Граф мест. Строится по записи офлайн и живёт в среднем слое (2 Гц)."""

    def __init__(self, *, same_place_similarity: float = 0.82,
                 variant_similarity: float = 0.6) -> None:
        if not 0.0 < variant_similarity < same_place_similarity <= 1.0:
            raise ValueError("порог варианта должен быть ниже порога того же места")
        self.same = same_place_similarity
        self.variant = variant_similarity
        self.places: dict[str, Place] = {}
        self.edges: dict[tuple[str, str, str], Traversal] = {}
        self.current: str | None = None
        self._since_seq: int = 0
        self._lost = 0

    # --- узлы ---------------------------------------------------------------

    def recognize(self, fp: tuple[int, ...]) -> tuple[str | None, float]:
        """Найти самое похожее место. `None`, если ничего похожего нет."""
        best, score = None, 0.0
        for p in self.places.values():
            s = p.best_similarity(fp)
            if s > score:
                best, score = p.id, s
        return best, score

    def observe(self, fp: tuple[int, ...], seq: int, *, seconds_per_seq: float = 1.0,
                mode: str = "unknown") -> str:
        """Увидеть вид. Возвращает место, в котором мы теперь.

        Три исхода: то же место, известное другое место (тогда появляется или
        уточняется ребро), совсем новое место.
        """
        best, score = self.recognize(fp)

        if best is not None and score >= self.same:
            place = self.places[best]
            place.visits += 1
            place.last_seq = max(place.last_seq, seq)
            if score < 1.0 and score >= self.variant and len(place.variants) < 16:
                place.variants.append(fp)
        else:
            pid = place_id(fp)
            if pid in self.places:
                # Коллизия отпечатков: разные виды дали один идентификатор.
                # Не сливаем молча — добавляем как вариант и отмечаем.
                self.places[pid].variants.append(fp)
                place = self.places[pid]
            else:
                place = Place(pid, fp, first_seq=seq, last_seq=seq)
                self.places[pid] = place

        if self.current is not None and self.current != place.id:
            seconds = max(0.0, (seq - self._since_seq) * seconds_per_seq)
            key = (self.current, place.id, mode)
            edge = self.edges.get(key)
            if edge is None:
                edge = Traversal(self.current, place.id, mode)
                self.edges[key] = edge
            edge.observe(seconds)

        if self.current != place.id:
            self._since_seq = seq
        self.current = place.id
        return place.id

    def lost(self) -> None:
        """Потеря ориентации: вид ни на что не похож.

        Это не ошибка и не повод что-то придумывать. Считается отдельно, потому
        что частая потеря ориентации означает, что порог узнавания подобран не
        так, — и это надо видеть, а не сглаживать.
        """
        self._lost += 1
        self.current = None

    # --- выборки ------------------------------------------------------------

    def neighbours(self, place: str) -> list[Traversal]:
        return [e for e in self.edges.values() if e.src == place]

    def __len__(self) -> int:
        return len(self.places)

    def edges_from_to(self, src: str, dst: str) -> list[Traversal]:
        return [e for k, e in self.edges.items() if k[0] == src and k[1] == dst]

    def route(self, src: str, dst: str) -> list[Traversal] | None:
        """Путь с наименьшим ожидаемым временем. Дейкстра по `mu_seconds`.

        Никакой геометрии: только времена проходов, которые агент сам измерил.
        Ребро, пройденное один раз, считается по своему `mu`, но его `confidence`
        низкая — выбор маршрута по ненадёжным рёбрам должен быть виден.
        """
        if src not in self.places or dst not in self.places:
            return None
        import heapq

        best: dict[str, float] = {src: 0.0}
        prev: dict[str, Traversal] = {}
        heap: list[tuple[float, str]] = [(0.0, src)]
        while heap:
            cost, node = heapq.heappop(heap)
            if node == dst:
                break
            if cost > best.get(node, float("inf")):
                continue
            for edge in self.neighbours(node):
                nxt = cost + max(0.001, edge.mu_seconds)
                if nxt < best.get(edge.dst, float("inf")):
                    best[edge.dst] = nxt
                    prev[edge.dst] = edge
                    heapq.heappush(heap, (nxt, edge.dst))
        if dst not in prev and dst != src:
            return None
        path: list[Traversal] = []
        node = dst
        while node != src:
            edge = prev[node]
            path.append(edge)
            node = edge.src
        return list(reversed(path))

    def stats(self) -> dict[str, Any]:
        edges = list(self.edges.values())
        return {"places": len(self.places), "edges": len(edges),
                "lost": self._lost, "current": self.current,
                "mean_visits": round(
                    sum(p.visits for p in self.places.values()) / len(self.places), 2)
                if self.places else 0.0,
                "mean_edge_confidence": round(
                    sum(e.confidence for e in edges) / len(edges), 4) if edges else 0.0,
                "one_shot_edges": sum(1 for e in edges if e.n == 1)}

    def as_dict(self) -> dict[str, Any]:
        return {"places": [p.as_dict() for _, p in sorted(self.places.items())],
                "edges": [e.as_dict() for _, e in sorted(self.edges.items())],
                **self.stats()}

    def __iter__(self) -> Iterator[Place]:
        return iter(self.places.values())
