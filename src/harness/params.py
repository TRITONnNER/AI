"""Детектор мёртвых и однобоких параметров. Инвариант 30.

Четыре случая подряд, когда настройка существует и ничего не делает:

| Параметр | Что было |
|---|---|
| `babble_rate` | не читался ни одной строкой |
| три оси модуляции эмоций | единственный потребитель — печать отчёта |
| `capture_width`, `capture_height` | читались только синтетическими мирами |
| `audio_sync_tolerance_ms` | лежал в профиле, не проверялся нигде |

Это закономерность, а не совпадения, и цена известна: живая запись несла профиль,
заявляющий кадр **в 36 раз меньше настоящего**. Данные целы, но запись врала о себе, и
всякий расчёт по её профилю ошибался ровно в отношение площадей. Нашлось случайно —
через сомнение в коэффициенте сжатия.

## Три диагноза, и средний опаснее крайнего

- **мёртвый** — читателей нет вообще. Дефект, но виден по нулю;
- **однобокий** — читатели есть, но не покрывают все пути, где параметр объявлен
  применимым. **Опаснее мёртвого**: параметр выглядит работающим, работает на одном
  пути и молча врёт на другом. Так и вышло с размером кадра;
- **только отчёт** — единственный потребитель печатает величину. Читателем не
  считается: печать ничего не меняет в поведении, а именно поведение параметр и
  обещает. Так три оси модуляции эмоций считались реализованными.

## Почему прежней проверки не хватило

`tests/test_wiring.py` искал имя настройки текстом по всему `src/`. Такой поиск находит
упоминание в строке документации, не различает чтение и запись и — главное — не знает,
**на каком пути** параметр прочитан. `capture_width` он нашёл в синтетических мирах и
успокоился.

## Два способа, и оба нужны

- **статически** (`read_sites`) — обход AST: где именно берётся значение по ключу, в
  какой функции и в каком модуле. Видит пути, которые в прогоне не проходились;
- **в прогоне** (`counting_profile`) — счётчики обращений к словарю настроек. Ловит
  чтения через переменную и вычисленный ключ, которых статика не видит.

Каждый вывод помечен способом, которым получен. Совпадение способов усиливает вывод,
расхождение — само по себе находка: значит один из способов слеп в этом месте.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping

from .core.settings import (GROUP_APPLIES, SCHEMA, RunPath, Setting, applies_of)

#: Путь исполнения. Словарь объявлен в схеме (`core/settings.RunPath`): применимость —
#: это заявление о настройке, а детектор только сверяет заявленное с найденным. Имя
#: `Path_` оставлено местным, чтобы не путать с `pathlib.Path` в этом же модуле.
Path_ = RunPath


#: Какие пути обслуживает модуль. Порядок важен: первое совпадение выигрывает, поэтому
#: частное идёт перед общим.
#:
#: **Модуль может обслуживать несколько путей, и это не исключение, а норма.** Первая
#: редакция приписывала каждому модулю ровно один путь и получила 27 «однобоких», из
#: которых больше двадцати были её собственной ошибкой: `session.py` — общий для живой
#: записи и генератора корпуса, `core/blobstore.py` и `core/resources.py` — вообще для
#: всего. Детектор, кричащий ложно, хуже отсутствующего: настоящие находки тонут, и
#: смотреть в него перестают.
LIVE_SYNTH = (Path_.LIVE, Path_.SYNTHETIC)
EVERY = (Path_.LIVE, Path_.SYNTHETIC, Path_.AGENT)

MODULE_PATHS: tuple[tuple[str, tuple[Path_, ...]], ...] = (
    # Только синтетика: миры и их генераторы.
    ("corpus/", (Path_.SYNTHETIC,)),
    ("benchmark.py", (Path_.SYNTHETIC,)),
    # Колония экземпляров сидит на двух путях сразу, и это проверено по коду, а не
    # приписано по имени файла: `spawn` заводит миры по сидам (синтетика), а `deliver`
    # складывает `Testimony` в `BeliefStore` через `merge_testimony` — это конвейер
    # убеждений, то есть агентский стек. Первая редакция считала модуль чисто
    # синтетическим и объявила однобокими три работающих параметра обмена
    # свидетельствами.
    ("instances.py", (Path_.SYNTHETIC, Path_.AGENT)),
    # Только живая запись: захват экрана, инъекция ввода, осмотр машины.
    ("capture/", (Path_.LIVE,)),
    ("selftest.py", (Path_.LIVE,)),
    ("progress.py", (Path_.LIVE,)),
    ("doctor.py", (Path_.LIVE,)),
    ("machine.py", (Path_.LIVE,)),
    ("inject/", (Path_.LIVE,)),
    # Общая инфраструктура записи: через неё пишет и живой захват, и генератор.
    ("session.py", LIVE_SYNTH),
    ("devices.py", LIVE_SYNTH),
    ("debug/", LIVE_SYNTH),
    ("core/blobstore.py", LIVE_SYNTH),
    ("core/journal.py", EVERY),
    ("core/levels.py", EVERY),
    ("core/resources.py", EVERY),
    ("core/clocks.py", EVERY),
    ("core/profile.py", EVERY),
    ("core/settings.py", EVERY),
    # Печать: читателем не считается.
    ("cli.py", (Path_.REPORT,)),
    # Агентский стек.
    ("perception/", (Path_.AGENT,)),
    ("vision/", (Path_.AGENT,)),
    ("behaviour/", (Path_.AGENT,)),
    ("model/", (Path_.AGENT,)),
    ("core/", (Path_.AGENT,)),
    ("params.py", (Path_.REPORT,)),
)

#: Функции, чей результат уходит в печать и никуда больше. Чтение внутри такой функции
#: читателем не считается (инвариант 30): печать ничего не меняет в поведении.
REPORT_FUNCS: frozenset[str] = frozenset({
    "as_dict", "render_text", "report", "text", "summary", "why_text",
    "as_debug_dict", "describe", "plan_text", "verdict", "table", "render_markdown",
})

# Применимость объявлена в схеме: `core/settings.GROUP_APPLIES` для групп,
# `Setting.applies_to` для отдельных настроек, `applies_of` сводит одно с другим.
# Здесь таблица не дублируется — два места правды разошлись бы молча.


def paths_of(module: str) -> tuple[Path_, ...]:
    """Какие пути обслуживает модуль. Первое совпадение по `MODULE_PATHS`."""
    for prefix, paths in MODULE_PATHS:
        if module.startswith(prefix) or module == prefix.rstrip("/"):
            return paths
    return (Path_.AGENT,)


class Diagnosis(StrEnum):
    """Отсортированы по тяжести: первое тяжелее последнего."""

    ONE_SIDED = "однобокий"      # опаснее мёртвого: выглядит работающим
    DEAD = "мёртвый"
    REPORT_ONLY = "только отчёт"
    PLANNED = "задел"            # объявлен заранее, веха названа
    OK = "работает"


#: Порядок тяжести для сортировки отчёта.
SEVERITY: dict[Diagnosis, int] = {
    Diagnosis.ONE_SIDED: 0, Diagnosis.DEAD: 1, Diagnosis.REPORT_ONLY: 2,
    Diagnosis.PLANNED: 3, Diagnosis.OK: 4,
}


@dataclass(frozen=True, slots=True)
class Site:
    """Одно место чтения: модуль, функция, обслуживаемые пути, читатель ли это."""

    module: str
    function: str
    paths: tuple[Path_, ...]
    line: int

    @property
    def report_only(self) -> bool:
        return (self.function in REPORT_FUNCS
                or tuple(self.paths) == (Path_.REPORT,))

    def as_dict(self) -> dict[str, Any]:
        return {"module": self.module, "function": self.function,
                "paths": [str(p) for p in self.paths], "line": self.line,
                "report_only": self.report_only}


# ---------------------------------------------------------------------------
# Способ первый: статический обход
# ---------------------------------------------------------------------------


class _Reads(ast.NodeVisitor):
    """Собирает чтения по строковому ключу с именем объемлющей функции."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.paths = paths_of(module)
        self.func: list[str] = []
        self.found: list[tuple[str, Site]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.func.append(node.name)
        self.generic_visit(node)
        self.func.pop()

    visit_AsyncFunctionDef = visit_FunctionDef       # type: ignore[assignment]

    def _note(self, key: str, line: int) -> None:
        self.found.append((key, Site(self.module, self.func[-1] if self.func else "",
                                     self.paths, line)))

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # `p["ключ"]`, `profile.parameters["ключ"]`, `d["ключ"]`
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            self._note(node.slice.value, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # `p.get("ключ", ...)` — тоже чтение
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            self._note(node.args[0].value, node.lineno)
        self.generic_visit(node)


def read_sites(root: Path | None = None) -> dict[str, list[Site]]:
    """Где какой ключ читается. Статически, по AST — не текстовым поиском.

    Текстовый поиск, стоявший здесь раньше, находил имя настройки в строке
    документации и не отличал чтение от упоминания. Он же пропустил однобокость
    размера кадра: имя нашлось в синтетическом мире, и проверка успокоилась.
    """
    root = Path(root) if root else Path(__file__).resolve().parent
    out: dict[str, list[Site]] = {}
    keys = {s.key for s in SCHEMA}
    for file in sorted(root.rglob("*.py")):
        rel = file.relative_to(root).as_posix()
        if rel == "core/settings.py":
            continue                     # объявление, а не чтение
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        v = _Reads(rel)
        v.visit(tree)
        for key, site in v.found:
            if key in keys:
                out.setdefault(key, []).append(site)
    return out


# ---------------------------------------------------------------------------
# Способ второй: счётчики в прогоне
# ---------------------------------------------------------------------------


class CountingDict(dict):
    """Словарь настроек, считающий обращения. Ловит то, чего не видит статика.

    Чтение через переменную (`key = "capture_fps"; p[key]`) и через цикл по ключам
    статическому обходу не видно вовсе. Счётчик видит любое обращение — но только на
    том пути, который в прогоне пройден, поэтому способы дополняют друг друга, а не
    заменяют.
    """

    def __init__(self, data: Mapping[str, Any], counts: dict[str, int]) -> None:
        super().__init__(data)
        self._counts = counts

    def __getitem__(self, key: str) -> Any:
        self._counts[key] = self._counts.get(key, 0) + 1
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:      # type: ignore[override]
        self._counts[key] = self._counts.get(key, 0) + 1
        return super().get(key, default)


def counting_profile(profile: Any) -> tuple[Any, dict[str, int]]:
    """Тот же профиль, но каждое чтение настройки считается.

    Профиль неизменяем (`frozen`), поэтому словари подменяются через
    `object.__setattr__`: это не обход запрета на мутацию настроек, а замена
    контейнера на такой же по содержимому. Значения не меняются, хеши остаются те же.
    """
    counts: dict[str, int] = {}
    object.__setattr__(profile, "parameters",
                       CountingDict(profile.parameters, counts))
    object.__setattr__(profile, "structural",
                       CountingDict(profile.structural, counts))
    return profile, counts


def runtime_probe(*, steps: int = 40, seed: int = 7,
                  root: Path | None = None) -> dict[str, int]:
    """Короткий прогон со счётчиками: сколько раз какая настройка прочитана.

    Что этот способ даёт и чего не даёт, сказать надо прямо, иначе им же и соврут.

    **Даёт:** чтение через вычисленный ключ. `p[name]` в цикле по именам статике не
    видно вовсе, а счётчику видно любое обращение. Настройка, которую статика назвала
    мёртвой, а счётчик прочёл, — не мёртвая, и находка здесь именно в расхождении.

    **Не даёт:** путь. Счётчик знает, что ключ взяли, но не знает, из живого захвата
    его взяли или из синтетического мира. Однобокость — свойство путей, и установить её
    счётчиком нельзя. Поэтому диагноз ставит статика, а счётчик её проверяет.

    Прогон **синтетический и агентский**: живой путь требует экрана и звука, которых у
    прогона нет, и подсовывать сюда заглушку было бы хуже всего — счётчик показал бы
    чтения, которых на настоящей машине не случилось.
    """
    from .behaviour.babbling import Babbler, run_babbling          # noqa: PLC0415
    from .core.clocks import Clocks                                # noqa: PLC0415
    from .core.profile import from_schema                          # noqa: PLC0415
    from .corpus.world import InteractiveWorld                     # noqa: PLC0415
    from .model.forward import ForwardModel                        # noqa: PLC0415
    from .model.places import PlaceGraph                           # noqa: PLC0415
    from .session import Recorder                                  # noqa: PLC0415

    profile, counts = counting_profile(from_schema("проверка чтений"))
    world = InteractiveWorld(profile, seed=seed, n_outputs=12)
    babbler = Babbler(profile, world.outputs, rng_seed=seed)
    run_babbling(world, babbler, steps=steps, clocks=Clocks())

    graph = PlaceGraph.from_profile(profile)
    obs = world.step(None, with_audio=False)
    graph.see(obs.frame, 0, seconds_per_seq=1 / 30.0, mode="start")
    ForwardModel.from_graph(graph, babbler.body)

    if root is not None:
        with Recorder(root, profile=profile, source="проверка чтений",
                      synthetic=True) as rec:
            rec.record_frame(obs.frame)
    return counts


# ---------------------------------------------------------------------------
# Диагноз
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Finding:
    key: str
    diagnosis: Diagnosis
    sites: tuple[Site, ...]
    applies: tuple[Path_, ...]
    missing: tuple[Path_, ...] = ()      # пути, где читателя нет
    planned: str = ""
    runtime_reads: int | None = None     # None — прогон не делался

    @property
    def readers(self) -> tuple[Site, ...]:
        """Только настоящие читатели: печать читателем не считается."""
        return tuple(s for s in self.sites if not s.report_only)

    @property
    def disagreement(self) -> str:
        """Расхождение способов, если оно есть. Само по себе находка.

        Расхождение означает, что один из двух способов в этом месте слеп, и знать,
        который, важнее самого диагноза: «статика молчит, прогон читает» — это чтение
        через вычисленный ключ, то есть диагноз «мёртвый» неверен; «статика видит,
        прогон не читает» — это всего лишь путь, который прогон не проходил, и
        дефектом не является.
        """
        if self.runtime_reads is None:
            return ""
        if self.runtime_reads > 0 and not self.readers:
            return (f"статика читателей не нашла, прогон прочёл {self.runtime_reads} "
                    "раз: ключ вычисляется, диагноз «мёртвый» здесь неверен")
        if self.runtime_reads == 0 and self.readers:
            return "прогон этот путь не проходил — не дефект, а неполнота прогона"
        return ""

    @property
    def static_blind(self) -> bool:
        """Статика читателей не нашла, а прогон прочёл. Единственное расхождение,
        которое означает ошибку способа, а не неполноту прогона."""
        return bool(self.runtime_reads) and not self.readers

    @property
    def method(self) -> str:
        """Каким способом получен вывод. Требование части 2: помечать каждый."""
        if self.runtime_reads is None:
            return "статика"
        if self.disagreement:
            return "расхождение способов"
        if self.runtime_reads > 0:
            return "статика и прогон"
        return "статика (прогон молчит)"

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "diagnosis": str(self.diagnosis),
                "applies": [str(p) for p in self.applies],
                "missing": [str(p) for p in self.missing],
                "readers": [s.as_dict() for s in self.readers],
                "report_only": [s.as_dict() for s in self.sites if s.report_only],
                "planned": self.planned, "runtime_reads": self.runtime_reads,
                "method": self.method, "disagreement": self.disagreement}


def diagnose(setting: Setting, sites: Iterable[Site],
             runtime_reads: int | None = None) -> Finding:
    all_sites = tuple(sites)
    applies = applies_of(setting)
    # Настройка, **объявленная** отчётной, читателем считает и печать. Это не поддавка:
    # инвариант 23 требует держать порог замера в схеме, иначе он не попадёт в
    # `profile_hash` и прогоны станут несравнимыми; инвариант 30 требует читателя на
    # объявленном пути. Оба выполнимы одновременно ровно так — путь объявлен отчётным.
    # Диагноз «только отчёт» остаётся для случая, ради которого он и заведён: настройка
    # объявлена меняющей поведение, а потребитель у неё один — печать.
    report_declared = tuple(applies) == (Path_.REPORT,)
    readers = tuple(s for s in all_sites
                    if report_declared or not s.report_only)
    covered = {p for s in readers for p in s.paths}
    missing = tuple(p for p in applies if p not in covered)

    if setting.planned:
        # Задел объявлен явно, с указанием, чего не хватает. Это законный третий
        # исход, и он **виден**: параметр, ждущий вехи, отличается от забытого при
        # рефакторинге только тем, что об этом написано.
        return Finding(setting.key, Diagnosis.PLANNED, all_sites, applies,
                       missing, setting.planned, runtime_reads)
    if not readers:
        if all_sites:
            return Finding(setting.key, Diagnosis.REPORT_ONLY, all_sites, applies,
                           applies, "", runtime_reads)
        return Finding(setting.key, Diagnosis.DEAD, all_sites, applies, applies,
                       "", runtime_reads)
    if missing:
        return Finding(setting.key, Diagnosis.ONE_SIDED, all_sites, applies,
                       missing, "", runtime_reads)
    return Finding(setting.key, Diagnosis.OK, all_sites, applies, (), "",
                   runtime_reads)


def audit(root: Path | None = None,
          runtime: Mapping[str, int] | None = None) -> list[Finding]:
    """Полный разбор. Отсортирован по тяжести: сначала однобокие, потом мёртвые."""
    sites = read_sites(root)
    out = [diagnose(s, sites.get(s.key, ()),
                    None if runtime is None else int(runtime.get(s.key, 0)))
           for s in SCHEMA]
    out.sort(key=lambda f: (SEVERITY[f.diagnosis], f.key))
    return out


def counts(findings: Iterable[Finding]) -> dict[str, int]:
    got: dict[str, int] = {str(d): 0 for d in Diagnosis}
    for f in findings:
        got[str(f.diagnosis)] += 1
    return got


def defects(findings: Iterable[Finding]) -> list[Finding]:
    """Мёртвые, однобокие и «только отчёт». Заделы дефектом не считаются."""
    bad = {Diagnosis.ONE_SIDED, Diagnosis.DEAD, Diagnosis.REPORT_ONLY}
    return [f for f in findings if f.diagnosis in bad]


def defect_count(findings: Iterable[Finding]) -> int:
    """Одно число для постоянных показателей: сколько параметров не работают."""
    return len(defects(findings))


def render_table(findings: Iterable[Finding], *, only_defects: bool = False) -> str:
    """Таблица «параметр → читатели → диагноз», по тяжести. Часть 2 TASK-10."""
    rows = list(findings)
    total = len(rows)
    if only_defects:
        rows = defects(rows)
        if not rows:
            return (f"дефектных параметров нет: {total} настроек, у каждой есть "
                    "читатель, и он покрывает все объявленные пути")
    out: list[str] = []
    head = f"{'диагноз':<14} {'параметр':<30} {'способ':<22} читатели"
    out.append(head)
    out.append("-" * len(head))
    for f in rows:
        readers = ", ".join(f"{s.module}:{s.function}" for s in f.readers) or "—"
        out.append(f"{str(f.diagnosis):<14} {f.key:<30} {f.method:<22} {readers}")
        if f.missing and f.diagnosis is not Diagnosis.OK:
            out.append(f"{'':<14} нет читателя на: "
                       + ", ".join(str(p) for p in f.missing))
        if f.planned:
            out.append(f"{'':<14} задел: {f.planned}")
        if f.disagreement:
            out.append(f"{'':<14} {f.disagreement}")
    got = counts(rows)
    out.append("")
    out.append("итого: " + ", ".join(f"{k} {v}" for k, v in got.items() if v))
    out.append(f"дефектов (мёртвые + однобокие + только отчёт): {defect_count(rows)}")
    return "\n".join(out)
