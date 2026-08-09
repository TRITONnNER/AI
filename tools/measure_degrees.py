"""Форма подтверждённого подграфа: степени, чётность, пары с двумя маршрутами.

`TASK-03-MEASUREMENT-LOOPS.md`, часть 1. Проверяет три из четырёх критериев
готовности сразу, потому что они про одно и то же — про форму графа:

1. доля узлов со степенью 3+ (по рёбрам, которые куда-то ведут; петли степени не
   добавляют);
2. есть ли узел, достижимый **и за два, и за три** подтверждённых шага. Это ровно
   то, чего на цепи не бывает: цепь двудольна, чётность обхода фиксирована;
3. есть ли пара узлов с двумя различными подтверждёнными маршрутами — первая
   настоящая проверка графа мест, потому что до сих пор `mu` и `sigma` не с чем
   было сверять.

Замер на восьми сидах, петли записываются, `python3 tools/measure_degrees.py`:

    прежняя разведка:      доля 3+ медиана 4.0%, узлов «и за 2, и за 3» 0, пар 4
    разведка с замыканием: доля 3+ медиана 0.0%, узлов «и за 2, и за 3» 10, пар 26

Результат смешанный, и это надо читать буквально. Критерии 2 и 3 достигнуты: то,
чего было ноль, стало десять, а пар с двумя маршрутами стало 26 вместо 4 и на трёх
сидах вместо одного. Критерий 1 **не достигнут и ухудшился**: медиана доли узлов
степени 3+ упала с 4 % до 0 %.

Почему так, и почему это не противоречие. Ненаправленная степень 3+ и наличие цикла
нечётной длины в **направленном** подграфе — разные вещи. У прежней разведки на сидах
5 и 17 доля 3+ была 32 % и 39 %, а узлов «и за 2, и за 3» — ноль: рёбра шли туда и
обратно, чётность не ломалась. Замыкание ломает чётность, но исследует меньше, и
подтверждённый подграф выходит мельче (сид 13: 44 узла против 29, сид 19: 44 против
25), поэтому доля узлов высокой степени падает.

Вывод по гигиене измерений, тот же, что и во всём `MEASUREMENT.md`: **критерий 1 был
косвенным показателем для критериев 2–3, и он за целью не следит.** Мерить надо то,
ради чего всё делалось, — возможность сравнить планы при равной сложности, — а не
удобный прокси.
"""
import sys, statistics
from collections import Counter
sys.path.insert(0, "src")

from harness.behaviour.babbling import Babbler, run_babbling
from harness.behaviour.planner import choose_closing_probe, choose_probe
from harness.core.action import Action, action_key
from harness.core.clocks import Clocks
from harness.core.profile import from_schema
from harness.corpus.world import InteractiveWorld
from harness.model.forward import ForwardModel
from harness.model.places import PlaceGraph

def run(seed, loops, steps=1500, closing=False):
    profile = from_schema("степени", capture_width=320, capture_height=180,
                          babble_repeats=3, place_record_loops=loops,
                          macro_max_length=0)
    min_n = int(profile.parameters["plan_min_step_n"])
    bw = InteractiveWorld(profile, seed=seed, n_outputs=16)
    bab = Babbler(profile, bw.outputs, rng_seed=seed)
    run_babbling(bw, bab, steps=600, clocks=Clocks())
    inverse = dict(bab.inverse_found)
    world = InteractiveWorld(profile, seed=seed, n_outputs=16)
    graph = PlaceGraph.from_profile(profile)
    graph.see(world.step(None, with_audio=False).frame, 0, seconds_per_seq=1/30.0, mode="start")
    st = {"seq": 0, "last": None}
    def step(out, ms=200):
        obs = world.step(Action.key(out, ms), with_audio=False)
        st["seq"] += 1; st["last"] = out
        return graph.see(obs.frame, st["seq"], seconds_per_seq=1/30.0, mode=action_key(out, ms))
    model = ForwardModel.from_graph(graph, bab.body)
    for i in range(steps):
        if i % 25 == 0:
            model = ForwardModel.from_graph(graph, bab.body)
        if closing:
            out, ms = choose_closing_probe(model, graph, graph.current, world.outputs,
                                          hold_ms=200, min_n=min_n, inverse=inverse,
                                          last_output=st["last"])
        else:
            out, ms = choose_probe(model, graph.current, world.outputs, hold_ms=200,
                                  min_n=min_n, inverse=inverse, last_output=st["last"])
        step(out, ms)
    # Пары, достижимые и за два, и за три шага — то, чего сейчас ноль.
    conf_edges = {}
    for e in graph.edges.values():
        if e.n >= min_n and e.dst != e.src:
            conf_edges.setdefault(e.src, set()).add(e.dst)
    def at_depth(src, d):
        cur = {src}
        for _ in range(d):
            cur = {y for x in cur for y in conf_edges.get(x, ())}
        return cur
    both = 0
    two_and_three = []
    for src in conf_edges:
        a, b = at_depth(src, 2), at_depth(src, 3)
        common = (a & b) - {src}
        if common:
            both += 1
            two_and_three.append((src, sorted(common)[0]))
    # Пары с двумя различными маршрутами: у них mu можно сравнить.
    multi = 0
    for src in conf_edges:
        for dst in conf_edges[src]:
            ways = [e for e in graph.edges.values()
                    if e.src == src and e.dst == dst and e.n >= min_n]
            if len(ways) > 1:
                multi += 1
    # Подтверждённый подграф: рёбра, наблюдённые не меньше min_n раз, без петель.
    conf = {}
    for e in graph.edges.values():
        if e.n >= min_n and e.dst != e.src:
            conf.setdefault(e.src, set()).add(e.dst)
            conf.setdefault(e.dst, set()).add(e.src)
    deg = Counter(len(v) for v in conf.values())
    nodes = len(conf)
    high = sum(c for d, c in deg.items() if d >= 3)
    return {"seed": seed, "places": len(graph), "conf_nodes": nodes,
            "deg": dict(sorted(deg.items())),
            "share_deg3plus": high / nodes if nodes else 0.0,
            "both_2_and_3": both, "multi_route_pairs": multi}

SEEDS = (3, 5, 7, 11, 13, 17, 19, 23)
for closing in (False, True):
    name = "разведка с замыканием" if closing else "прежняя разведка"
    print(f"══ {name}, петли записываются ══")
    shares, both_all, multi_all = [], 0, 0
    for seed in SEEDS:
        r = run(seed, True, closing=closing)
        shares.append(r["share_deg3plus"])
        both_all += r["both_2_and_3"]; multi_all += r["multi_route_pairs"]
        print(f"  сид {seed:>3}: мест {r['places']:>3}, подграф {r['conf_nodes']:>3}, "
              f"степени {r['deg']}, 3+ {r['share_deg3plus']:.0%}, "
              f"узлов «и за 2, и за 3» {r['both_2_and_3']}, "
              f"пар с двумя маршрутами {r['multi_route_pairs']}")
    print(f"  ИТОГО доля узлов степени 3+: медиана {statistics.median(shares):.1%}; "
          f"узлов, достижимых и за 2, и за 3 шага: {both_all}; "
          f"пар с двумя маршрутами: {multi_all}\n")
