"""Сходятся ли `mu` двух маршрутов между одними и теми же местами.

`TASK-04-PLACEGRAPH.md`, часть 2. Главный замер задачи, потому что до сих пор `mu`
графа мест **не с чем было сверять вообще**: пока подтверждённый подграф оставался
цепью, между парой мест существовал ровно один путь, и время прохода нечем было
проверить, кроме себя самого.

Что именно проверяется: **узнаёт ли отпечаток вида то же место с другой стороны.**
Само существование пар с двумя маршрутами доказывает это слабо — пары могли возникнуть
и при слиянии разных мест в один отпечаток. Количественная сверка `mu` проверяет
по-настоящему: если два маршрута ведут в **одно** место, время прохода по ним связано
измеримо, а если отпечаток слил два разных места, связи не будет.

Критерии зафиксированы **до прогона** в `MEASUREMENT.md`, раздел 10.2:

    se = sqrt(sigma_A² / n_A + sigma_B² / n_B)
    z  = |mu_A − mu_B| / se
    пара согласована при z < 2

| Доля пар с z<2 | Знаки разницы | Вывод |
|---|---|---|
| ≥ 0.8 | — | граф мест работает |
| < 0.8 | сбалансированы | `sigma` занижена |
| < 0.8 | перекошены | расхождение систематическое: подозрение на слияние мест |

`n` ведётся по **непересекающимся по рёбрам** парам (инвариант 22, применённый к
собственному замеру): 26 пар, делящих рёбра, — не 26 независимых наблюдений. При менее
чем пяти непересекающихся парах исход — «замер не поставлен», независимо от чисел.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from math import comb, sqrt
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harness.behaviour.babbling import Babbler, run_babbling          # noqa: E402
from harness.behaviour.planner import choose_closing_probe            # noqa: E402
from harness.core.action import Action, action_key                    # noqa: E402
from harness.core.clocks import Clocks                               # noqa: E402
from harness.core.profile import from_schema                         # noqa: E402
from harness.corpus.world import InteractiveWorld                    # noqa: E402
from harness.model.forward import ForwardModel                       # noqa: E402
from harness.model.places import PlaceGraph                          # noqa: E402
from harness.model.units import pseudoreplication, share             # noqa: E402

Z_BAR = 2.0                 # порог согласия, объявлен заранее
MIN_INDEPENDENT = 5         # ниже этого замер не поставлен, объявлено заранее


# ---------------------------------------------------------------------------
# Пара маршрутов
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RouteStat:
    """Один маршрут: чем прошли, сколько это заняло и на скольких наблюдениях."""

    keys: tuple[str, ...]           # рёбра как (src, dst, mode)
    mu: float
    sigma: float
    n: int
    steps: int

    def as_dict(self) -> dict[str, Any]:
        return {"steps": self.steps, "mu": round(self.mu, 4),
                "sigma": round(self.sigma, 4), "n": self.n}


@dataclass(frozen=True, slots=True)
class Pair:
    """Два подтверждённых маршрута между одними и теми же местами."""

    seed: int
    src: str
    dst: str
    a: RouteStat
    b: RouteStat

    @property
    def diff(self) -> float:
        """Разница со знаком: положительная — второй маршрут дольше."""
        return self.b.mu - self.a.mu

    @property
    def se(self) -> float:
        """Объединённая ошибка разницы. Ноль означает, что sigma нулевые у обоих."""
        return sqrt(self.a.sigma ** 2 / max(1, self.a.n)
                    + self.b.sigma ** 2 / max(1, self.b.n))

    @property
    def z(self) -> float | None:
        """`None`, когда разброс нулевой у обоих: делить не на что.

        Это не «идеальное согласие»: нулевая `sigma` при малом `n` означает, что
        разброс не измерен, а не что его нет. Такие пары считаются отдельно и в
        долю согласившихся не входят ни в ту, ни в другую сторону.
        """
        if self.se <= 0:
            return None
        return abs(self.diff) / self.se

    @property
    def agrees(self) -> bool | None:
        z = self.z
        return None if z is None else z < Z_BAR

    @property
    def edges(self) -> frozenset[str]:
        return frozenset(self.a.keys) | frozenset(self.b.keys)

    def as_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "src": self.src, "dst": self.dst,
                "a": self.a.as_dict(), "b": self.b.as_dict(),
                "diff": round(self.diff, 4),
                "se": round(self.se, 4),
                "z": None if self.z is None else round(self.z, 3),
                "agrees": self.agrees,
                "step_diff": self.b.steps - self.a.steps}


# ---------------------------------------------------------------------------
# Статистика без внешних зависимостей
# ---------------------------------------------------------------------------


def binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Точный двусторонний биномиальный тест. Нужен для проверки перекоса знаков."""
    if n == 0:
        return 1.0
    def pmf(x: int) -> float:
        return comb(n, x) * p ** x * (1 - p) ** (n - x)
    obs = pmf(k)
    return min(1.0, sum(pmf(x) for x in range(n + 1) if pmf(x) <= obs + 1e-12))


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Ранговая корреляция. `None`, когда пар меньше трёх или все ранги равны."""
    if len(xs) < 3:
        return None
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    rx, ry = rank(xs), rank(ys)
    if len(set(rx)) < 2 or len(set(ry)) < 2:
        return None
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def edge_disjoint(pairs: list[Pair]) -> list[Pair]:
    """Максимальное (жадно) множество пар без общих рёбер.

    Жадность от пар с наименьшим числом рёбер: короткие пары блокируют меньше, и
    так их влезает больше. Точное максимальное независимое множество здесь не нужно
    — нужна честная нижняя оценка числа независимых наблюдений.
    """
    used: set[str] = set()
    out: list[Pair] = []
    for pair in sorted(pairs, key=lambda p: (len(p.edges), p.seed, p.src, p.dst)):
        if pair.edges & used:
            continue
        used |= pair.edges
        out.append(pair)
    return out


# ---------------------------------------------------------------------------
# Сбор графа и поиск пар
# ---------------------------------------------------------------------------


def build_graph(seed: int, *, steps: int, babble: int, closing: bool,
                variable_cost: bool = False, bounded: bool = False,
                jitter: float | None = None,
                extent: int | None = None) -> tuple[PlaceGraph, ForwardModel, int]:
    """Собрать граф разведкой. Условия те же, что в замере степеней.

    `variable_cost` и `bounded` — структурные переключатели мира из `TASK-05`. По
    умолчанию оба выключены: это прежний мир, контроль, на котором замер обязан
    остаться вакуумным.
    """
    extra: dict[str, Any] = {}
    if jitter is not None:
        extra["world_cost_jitter"] = jitter
    if extent is not None:
        extra["world_extent_px"] = extent
    profile = from_schema("сверка-mu", capture_width=320, capture_height=180,
                          babble_repeats=3, place_record_loops=True,
                          macro_max_length=0, explore_closing_share=(1.0 if closing else 0.0),
                          world_variable_cost=variable_cost,
                          world_bounded=bounded, **extra)
    min_n = int(profile.parameters["plan_min_step_n"])
    bw = InteractiveWorld(profile, seed=seed, n_outputs=16)
    bab = Babbler(profile, bw.outputs, rng_seed=seed)
    run_babbling(bw, bab, steps=babble, clocks=Clocks())
    inverse = dict(bab.inverse_found)

    world = InteractiveWorld(profile, seed=seed, n_outputs=16)
    graph = PlaceGraph.from_profile(profile)
    first = world.step(None, with_audio=False)
    graph.see(first.frame, first.t_world, seconds_per_seq=1 / 30.0, mode="start")
    state: dict[str, Any] = {"last": None}

    def step(out: str, ms: int = 200) -> str:
        obs = world.step(Action.key(out, ms), with_audio=False)
        state["last"] = out
        # Графу передаются **такты мира**, а не счётчик шагов. Без этого время в
        # графе осталось бы тождественным топологии даже при переменной стоимости в
        # мире: шаг всегда один, а такты — столько, сколько переход стоил. Это и
        # есть та связь, из-за отсутствия которой замер TASK-04 был вакуумен.
        return graph.see(obs.frame, obs.t_world, seconds_per_seq=1 / 30.0,
                         mode=action_key(out, ms))

    model = ForwardModel.from_graph(graph, bab.body)
    from harness.behaviour.planner import choose_probe
    for i in range(steps):
        if i % 25 == 0:
            model = ForwardModel.from_graph(graph, bab.body)
        if closing:
            out, ms = choose_closing_probe(model, graph, graph.current, world.outputs,
                                           hold_ms=200, min_n=min_n, inverse=inverse,
                                           last_output=state["last"])
        else:
            out, ms = choose_probe(model, graph.current, world.outputs, hold_ms=200,
                                   min_n=min_n, inverse=inverse,
                                   last_output=state["last"])
        step(out, ms)
    return graph, ForwardModel.from_graph(graph, bab.body), min_n


def one_step_pairs(graph: PlaceGraph, min_n: int, seed: int) -> list[Pair]:
    """Пары мест, между которыми есть два разных подтверждённых **ребра**.

    Это самый прямой случай: два разных действия ведут из одного места в одно и то
    же. Время прохода у них измерялось независимо, поэтому сверка `mu` здесь
    проверяет именно узнавание места, а не арифметику сложения рёбер.
    """
    by_pair: dict[tuple[str, str], list] = {}
    for key, edge in graph.edges.items():
        if edge.n < min_n or edge.dst == edge.src:
            continue
        by_pair.setdefault((edge.src, edge.dst), []).append((key, edge))
    out: list[Pair] = []
    for (src, dst), items in sorted(by_pair.items()):
        if len(items) < 2:
            continue
        # Берём два наиболее наблюдённых: у остальных `sigma` держится на воздухе.
        items.sort(key=lambda x: (-x[1].n, x[0]))
        (ka, ea), (kb, eb) = items[0], items[1]
        out.append(Pair(
            seed, src, dst,
            RouteStat((str(ka),), ea.mu_seconds, ea.sigma, ea.n, 1),
            RouteStat((str(kb),), eb.mu_seconds, eb.sigma, eb.n, 1)))
    return out


def multi_step_pairs(graph: PlaceGraph, min_n: int, seed: int, *,
                     max_depth: int = 4) -> list[Pair]:
    """Пары, между которыми есть два разных подтверждённых **пути** разной длины.

    Здесь `mu` маршрута складывается из рёбер, а `sigma` — по независимости
    (корень из суммы квадратов). Это допущение, и оно объявлено: если проходы по
    рёбрам зависимы, объединённый разброс занижен, и часть расхождений окажется
    следствием допущения, а не свойства графа. Поэтому одношаговые пары считаются
    отдельно — на них этого допущения нет.
    """
    adj: dict[str, list[tuple[str, Any]]] = {}
    for key, edge in graph.edges.items():
        if edge.n < min_n or edge.dst == edge.src:
            continue
        adj.setdefault(edge.src, []).append((str(key), edge))

    # Все простые пути до заданной глубины. Граф мал, перечисление дешевле хитрости.
    paths: dict[tuple[str, str], list[RouteStat]] = {}
    for start in adj:
        stack = [(start, (), (start,), 0.0, 0.0)]
        while stack:
            node, keys, seen, mu, var = stack.pop()
            if len(keys) >= max_depth:
                continue
            for key, edge in adj.get(node, ()):
                if edge.dst in seen:
                    continue
                nk = keys + (key,)
                nmu = mu + edge.mu_seconds
                nvar = var + edge.sigma ** 2
                paths.setdefault((start, edge.dst), []).append(
                    RouteStat(nk, nmu, sqrt(nvar), 0, len(nk)))
                stack.append((edge.dst, nk, seen + (edge.dst,), nmu, nvar))

    # `n` маршрута — самое слабое звено: маршрут держится на нём.
    n_of = {str(k): e.n for k, e in graph.edges.items()}
    out: list[Pair] = []
    for (src, dst), routes in sorted(paths.items()):
        fixed = [RouteStat(r.keys, r.mu, r.sigma,
                           min(n_of[k] for k in r.keys), r.steps) for r in routes]
        # Разные длины, иначе это одношаговый случай или два параллельных ребра.
        by_len: dict[int, RouteStat] = {}
        for r in fixed:
            if r.steps not in by_len or r.n > by_len[r.steps].n:
                by_len[r.steps] = r
        if len(by_len) < 2:
            continue
        lens = sorted(by_len)
        a, b = by_len[lens[0]], by_len[lens[-1]]
        out.append(Pair(seed, src, dst, a, b))
    return out


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Verdict:
    kind: str                       # one_step | multi_step
    pairs: list[Pair] = field(default_factory=list)
    independent: list[Pair] = field(default_factory=list)

    @property
    def scorable(self) -> list[Pair]:
        return [p for p in self.independent if p.agrees is not None]

    @property
    def unmeasured_spread(self) -> int:
        return sum(1 for p in self.independent if p.agrees is None)

    def report(self) -> dict[str, Any]:
        total, indep = len(self.pairs), len(self.independent)
        scorable = self.scorable
        agreed = sum(1 for p in scorable if p.agrees)
        signs = [p.diff for p in scorable if p.diff != 0]
        positive = sum(1 for d in signs if d > 0)
        sign_p = binomial_two_sided(positive, len(signs))
        rho = spearman([float(p.b.steps - p.a.steps) for p in scorable],
                       [p.diff for p in scorable])

        if len(scorable) < MIN_INDEPENDENT:
            outcome = "замер не поставлен"
            why = (f"непересекающихся пар с измеренным разбросом {len(scorable)}, "
                   f"порог {MIN_INDEPENDENT} объявлен заранее. Доля «сколько из "
                   f"{len(scorable)} согласовались» результатом не является")
        else:
            frac = agreed / len(scorable)
            if frac >= 0.8:
                outcome = "согласуются в пределах разброса"
                why = ("граф мест работает: первое подтверждение представления "
                       "за весь проект")
            elif sign_p >= 0.05:
                outcome = "расходятся случайно"
                why = ("sigma занижена: разброс не отражает реальной "
                       f"неопределённости. Знаки сбалансированы, p={sign_p:.3f}")
            else:
                outcome = "расходятся систематически"
                why = ("отпечаток сливает разные места либо ребро меряет не время "
                       f"прохода. Знаки перекошены, p={sign_p:.4f}. "
                       + ("Связь знака с числом промежуточных узлов есть "
                          f"(ρ={rho:.2f}) — похоже на постоянную надбавку на "
                          "переход, а не на слияние мест"
                          if rho is not None and abs(rho) > 0.5 else
                          "Связи знака с числом промежуточных узлов нет"
                          + (f" (ρ={rho:.2f})" if rho is not None else
                             " (ρ не считается)")
                          + " — накопление надбавки этим не объясняется"))

        return {
            "kind": self.kind,
            "pairs_total": total,
            "pairs_independent": indep,
            "pseudoreplication": round(pseudoreplication(total, indep), 2)
            if indep else None,
            "scorable": len(scorable),
            "unmeasured_spread": self.unmeasured_spread,
            "agreed": agreed,
            "agreed_share": round(agreed / len(scorable), 4) if scorable else None,
            "sign_positive": positive,
            "sign_total": len(signs),
            "sign_p": round(sign_p, 4),
            "spearman_steps_vs_diff": None if rho is None else round(rho, 3),
            "outcome": outcome,
            "why": why,
            "seeds": sorted({p.seed for p in self.independent}),
            "examples": [p.as_dict() for p in self.independent[:8]],
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="3 5 7 11 13 17 19 23")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--babble", type=int, default=600)
    ap.add_argument("--variable-cost", action="store_true",
                    help="переменная стоимость прохода (структурный переключатель)")
    ap.add_argument("--bounded", action="store_true",
                    help="ограниченная область (структурный переключатель)")
    ap.add_argument("--jitter", type=float, default=None,
                    help="разброс стоимости; 0 обязан обнулить sigma")
    ap.add_argument("--extent", type=int, default=None)
    ap.add_argument("--no-closing", action="store_true",
                    help="прежняя разведка: пар почти не будет, это и есть смысл")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split()]
    one = Verdict("one_step")
    multi = Verdict("multi_step")
    for seed in seeds:
        graph, _model, min_n = build_graph(seed, steps=args.steps,
                                           babble=args.babble,
                                           closing=not args.no_closing,
                                           variable_cost=args.variable_cost,
                                           bounded=args.bounded,
                                           jitter=args.jitter,
                                           extent=args.extent)
        o = one_step_pairs(graph, min_n, seed)
        m = multi_step_pairs(graph, min_n, seed)
        one.pairs += o
        multi.pairs += m
        print(f"  сид {seed:>3}: мест {len(graph):>3}, пар с двумя рёбрами {len(o):>3}, "
              f"пар с двумя путями разной длины {len(m):>3}", flush=True)

    one.independent = edge_disjoint(one.pairs)
    multi.independent = edge_disjoint(multi.pairs)

    out = {}
    for v in (one, multi):
        r = v.report()
        out[v.kind] = r
        title = ("два разных ребра между одними местами" if v.kind == "one_step"
                 else "два пути разной длины между одними местами")
        print(f"\n── {title} ──")
        print(f"  пар всего {r['pairs_total']}, непересекающихся по рёбрам "
              f"{r['pairs_independent']}"
              + (f" (псевдорепликация ×{r['pseudoreplication']})"
                 if r['pseudoreplication'] else ""))
        print(f"  из них с измеренным разбросом {r['scorable']}, "
              f"с нулевой sigma у обоих {r['unmeasured_spread']}")
        if r["agreed_share"] is not None:
            print(f"  согласовались (z<{Z_BAR}): {r['agreed']}/{r['scorable']} = "
                  f"{r['agreed_share']:.0%}")
            print(f"  знаки: {r['sign_positive']} из {r['sign_total']} "
                  f"положительных, p={r['sign_p']}")
            print(f"  связь знака с длиной: ρ={r['spearman_steps_vs_diff']}")
        print(f"  ИСХОД: {r['outcome']}")
        print(f"    {r['why']}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
