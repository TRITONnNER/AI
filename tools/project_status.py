#!/usr/bin/env python3
"""Состояние проекта, посчитанное по коду и тестам, а не написанное руками.

Зачем машинно. Написанный руками список «что готово» расходится с кодом на второй
неделе, и расхождение незаметно: текст выглядит так же убедительно, когда врёт.
Здесь каждая строка выведена из чего-то проверяемого:

- **инварианты** — из тестов, которые их проверяют, и из того, прошли ли они;
- **настройки** — из схемы: сколько объявлено, сколько структурных, сколько ещё не
  реализовано (`planned`) и почему;
- **части** — из того, есть ли модуль, есть ли у него тесты и есть ли замер;
- **чего нет** — из явного списка в этом файле, но с обязательной причиной; строка
  без причины считается ошибкой и роняет прогон.

    python3 tools/project_status.py            # напечатать
    python3 tools/project_status.py --md FILE  # записать docs/STATUS.md
    python3 tools/project_status.py --json     # машинно
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "harness"
TESTS = ROOT / "tests"
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Двенадцать инвариантов и то, чем каждый проверяется
# ---------------------------------------------------------------------------
#
# Ключ — номер из CLAUDE.md. Покрытие считается по соглашению об именах:
# `test_invariant_{номер}_...`. Соглашение, а не список имён, потому что тестов на
# один инвариант бывает несколько, и список пришлось бы править при каждом новом —
# то есть он бы отставал. Инвариант, у которого не нашлось ни одного теста, попадает
# в отчёт как непроверенный: инвариант без теста — это пожелание.

INVARIANTS: dict[int, str] = {
    1: "Журнал только дозаписывается и никогда не редактируется",
    2: "Каждая запись штампуется хешем профиля и тремя часами",
    3: "Мир никогда не ставится на паузу",
    4: "Агент не получает координат, имён клавиш, меток, счёта",
    5: "Весь текст с экрана хешируется в непрозрачный символ",
    6: "Файрвол восприятия: модель отвечает только «что я вижу»",
    7: "У каждого убеждения есть происхождение",
    8: "Действие — это (выход, длительность, модификаторы)",
    9: "У каждого действия есть оценка обратимости",
    10: "Самоотчёты агента ни на что не влияют автоматически",
    11: "Структурные переключатели форкают журнал",
    12: "Отладочный канал строго отделён от агентского кода",
}


# ---------------------------------------------------------------------------
# Части проекта: модуль, тесты, чем замерено
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Part:
    name: str
    modules: tuple[str, ...]
    tests: tuple[str, ...]          # подстроки имён тестов
    measured: str = ""              # чем и как замерено; пусто — замера нет
    absent: str = ""                # чего именно нет; пусто — есть всё


PARTS: tuple[Part, ...] = (
    Part("Журнал и цепочка хешей", ("core/journal.py",), ("journal", "tamper"),
         "правка любой строки обнаруживается на своей записи"),
    Part("Хранение кадров и звука", ("core/blobstore.py",), ("blob", "shard", "frames"),
         "кадр+дельта+zlib, промотка к произвольному кадру 7.6 мс"),
    Part("Профиль и два хеша", ("core/profile.py", "core/settings.py"),
         ("profile", "schema", "setting"),
         "135 настроек: единицы, границы, форк журнала"),
    Part("Ресурсы: память, диск, расход, темп",
         ("core/resources.py",), ("resource", "governor", "cap"),
         "упор действует, журнал не обрезается"),
    Part("Устройства ввода-вывода", ("devices.py",), ("device",),
         "склейка источников, задержки, «видеть ≠ управлять»"),
    Part("Захват экрана", ("capture/screen.py", "capture/base.py"), ("capture", "backend"),
         "", "нет дисплея: проверено только честное отказывание"),
    Part("Инъекция ввода", ("inject/base.py", "inject/uinput_device.py"),
         ("inject", "delivery", "mask", "stop"),
         "СТОП рвёт удержание внутри кадра; маска говорит выходами",
         "нет /dev/uinput и нет игры: живая доставка не проверена"),
    Part("Сторожевой таймер", ("inject/watchdog.py",), ("watchdog",),
         "замершая картинка жмёт СТОП"),
    Part("Синтетический корпус", ("corpus/synthetic.py", "corpus/world.py"),
         ("corpus", "world", "synthetic"),
         "истина в отладочном потоке, проверяемом код её не видит"),
    Part("Разделение себя и мира", ("vision/selfworld.py",), ("selfworld", "layer", "signal"),
         "три признака, IoU 0.81 при точности 1.00 в игре"),
    Part("Ошибка предсказания", ("vision/predict.py",), ("predict", "error", "spike"),
         "всплески на смене режима, сравнение с опорным уровнем"),
    Part("Убеждения и происхождение", ("model/beliefs.py", "model/rebuild.py"),
         ("belief", "provenance", "rebuild", "body"),
         "две пересборки дают один отпечаток; свидетельство ≠ опыт"),
    Part("Граф мест", ("model/places.py",), ("place", "graph", "route", "loop",
                                             "refine", "split", "band"),
         "без координат, маршрут по измеренному времени; «нажал и остался» — тоже "
         "ребро (без него разведка запиралась на одном нажатии: 1 место за 1500 "
         "шагов), место делится по расхождению своих же предсказаний"),
    Part("Модель перехода", ("model/forward.py",), ("model_", "forward", "transition"),
         "«не знаю» — законный ответ; все исходы, а не самый частый"),
    Part("Драйвы, настроение, эмоция", ("model/drives.py",), ("drive", "mood", "emotion"),
         "аллостаз с горизонтом; ярлык эмоции ни на что не влияет"),
    Part("Сон и консолидация", ("model/consolidation.py",), ("sleep", "consolidat"),
         "не выдумывает, предупреждает о просрочке сверки с реальностью"),
    Part("Лепет: открытие тела", ("behaviour/babbling.py",), ("babbl", "body", "probe"),
         "9/9 живых, 15/15 молчащих, 8/8 обратных пар, необратимый найден"),
    Part("Цели с тестом и бюджетом", ("behaviour/goals.py",), ("goal",),
         "круг замкнут: давление ставит цель, пробы закрывают, отказ по бюджету"),
    Part("Библиотека навыков и цепочки", ("behaviour/skills.py",),
         ("skill", "macro", "recorder", "chain"),
         "макросы добыты из журнала; цепочка — один шаг плана со своей статистикой: "
         "при той же глубине поиска достижимо 21 место против 12, доля дошедших "
         "94–96 % не изменилась, цена — рёбер втрое больше"),
    Part("Контуры и субсумпция", ("behaviour/contours.py",), ("contour", "subsum"),
         "прерываемо, лучший ответ доступен всегда"),
    Part("Размыкатель эффекторов", ("behaviour/imagination.py",),
         ("imagin", "breaker", "thought"),
         "мысль не может стать поступком: исключение, а не тихое нажатие"),
    Part("Планировщик", ("behaviour/planner.py",), ("plan", "travel", "rehears"),
         "планы от трёх шагов: 224 из 239 одной попыткой (94 %), 238 из 239 с "
         "перепланированием. Из 19 провалов 8 — первое удивление, которого модель "
         "знать не могла, 11 — честный проигрыш вероятности"),
    Part("Файрвол восприятия", ("perception/firewall.py",), ("firewall", "question"),
         "императивы и оценки перехватываются, описание проходит"),
    Part("Сервисы описания", ("perception/describers.py", "perception/imagecodec.py"),
         ("describ", "quota", "cache", "png", "downscale", "gate"),
         "квота, кэш по побитовому отпечатку, цепочка, громкий отказ без ключа",
         "ни один сервис на этой машине не настроен: ответ модели не проверен"),
    Part("Доменная независимость", ("benchmark.py", "corpus/domains.py"),
         ("domain", "cross", "bench"),
         "4 мира × 6 сидов проходят все"),
    Part("Оптический поток и дальность", ("vision/flow.py",),
         ("flow", "depth_is", "single_run", "opencv", "effort", "scale"),
         "три метода на одном мире, по 25 прогонов: 0.49 / 0.50 / 0.54 при случайном "
         "0.33, разброс 0.14–0.20. Найдена и вторая поправка в мире: на текстуре "
         "одного масштаба большое смещение не находится в принципе",
         "надёжной дальности нет ни у одного метода; нужно послойное разделение "
         "движения"),
    Part("Слои взгляда: тау и частоты", ("vision/layers.py",),
         ("tau", "looming", "layer_clock", "fove", "look", "attention"),
         "тау предсказывает время до контакта с медианной ошибкой 3–4 % без "
         "расстояний; частоты слоёв экономят 56 % обновлений, фовеация — 87 %",
         "дальность по фазовой корреляции — один из трёх методов, все неотличимы: "
         "см. «Оптический поток и дальность»"),
    Part("Мир с глубиной", ("corpus/depth.py",), ("depth_world", "plane", "tau"),
         "три плана с параллаксом 2.4/1.0/0.15 и наплывающий предмет; истина по "
         "пикселям и настоящее время до контакта",
         "разделение слоёв на нём проваливается объявленно: у мира три движения"),
    Part("Самоотчёт агента", ("behaviour/selfreport.py",),
         ("selfreport", "report_", "_report"),
         "прогон с отчётами и без них совпадает до последнего числа; пересборка "
         "записи SELF_REPORT пропускает; словарь отчёта закрыт"),
    Part("Показатели по журналу", ("model/vitals.py",), ("vital",),
         "27 показателей из журнала, у каждого единица и источник; отсутствующие "
         "названы с причиной вместо правдоподобного нуля"),
    Part("Экземпляры", ("instances.py",), ("instance",),
         "обмен только свидетельствами, не опытом"),
    Part("Отладочный канал", ("debug/channel.py", "debug/keymap.py"), ("debug",),
         "недостижим из агентского кода — проверено обходом графа импортов"),
)


# ---------------------------------------------------------------------------
# Чего нет — с обязательной причиной
# ---------------------------------------------------------------------------

MISSING: tuple[tuple[str, str], ...] = (
    ("Живая запись с экрана", "нет дисплея в этом окружении"),
    ("Инъекция ввода в игру", "нет /dev/uinput и нет игры"),
    ("Петлевой звук и горячая клавиша СТОПа", "нет звукового устройства"),
    ("Видеокодек для кадров", "нет ffmpeg; предел шардов измерен — 106 ГиБ/час на 1080p"),
    ("Ответ настоящей большой модели", "адаптеры есть, ни один сервис здесь не настроен"),
    ("Обученные сети, weights/", "предсказатель аналитический; обучение не начато"),
    ("Пользы от второго деления места",
     "механизм есть и проверен на синтетике, но на интерактивном мире второй "
     "признак не понадобился ни разу: числа при пределе 1 и 3 совпадают до плана. "
     "Нужен мир, где состояние различимо только двумя признаками сразу"),
    ("Различения двух состояний мира, неразличимых из места агента",
     "11 провалов из 19 на замере — честный проигрыш вероятности: у пары два "
     "исхода, разделить их нечем ни по яркости, ни по ячейке. `travel` их "
     "добирает перепланированием (238 из 239), но одна попытка — нет"),
    ("Связи между библиотекой навыков и цепочками в графе",
     "цепочки записываются подряд идущими окнами, а не отбираются по повторяемости "
     "из библиотеки: `mine` находит повторяющиеся, а в граф идут все прошедшие"),
    ("Слои восприятия отдельными контурами",
     "PerceptionLayer из архитектуры не реализован; ручки помечены planned"),
    ("Разделение слоёв там, где камера не движется вовсе",
     "остаётся слабейший признак; меняющееся обрамление находится на 0.38–0.54"),
    ("Надёжной дальности (ближний/средний/дальний план)",
     "пять попыток, три метода. По 25 прогонов на метод: баланс 0.49 / 0.50 / 0.54 "
     "при случайном 0.33 — то есть частично находится, но разброс 0.14–0.20 и "
     "отдельные прогоны от 0.25 до 0.83. Различить методы по такому разбросу нельзя. "
     "Нужно послойное разделение движения: совместная оценка «где какой слой» и «как "
     "он движется». Таблицы — в harness.vision.flow"),
    ("Разделение слоёв в мире с глубиной",
     "метод стоит на допущении «у мира одно движение»; при трёх планах признаки "
     "расходятся на 91 % пикселей, и сторож расхождений это показывает"),
)


# ---------------------------------------------------------------------------
# Сбор
# ---------------------------------------------------------------------------


def test_names() -> dict[str, str]:
    """Имя теста → файл. Читается из AST, а не запуском: дешевле и надёжнее."""
    out: dict[str, str] = {}
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    out[node.name] = path.name
    return out


def run_tests() -> tuple[int, int, str]:
    """Прогнать тесты. Возвращает (прошло, упало, строка итога)."""
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=no"],
                          cwd=ROOT, capture_output=True, text=True)
    tail = (proc.stdout or "").strip().splitlines()
    line = tail[-1] if tail else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", line)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", line)) else 0
    return passed, failed, line


def settings_summary() -> dict[str, object]:
    from harness.core.settings import SCHEMA

    src_text = "\n".join(
        p.read_text(encoding="utf-8") for p in SRC.rglob("*.py")
        if p.name != "settings.py")
    unread = [s.key for s in SCHEMA
              if not s.planned and not re.search(rf'["\']{re.escape(s.key)}["\']', src_text)]
    return {
        "total": len(SCHEMA),
        "structural": sum(1 for s in SCHEMA if s.structural),
        "groups": sorted({s.group for s in SCHEMA}),
        "planned": [(s.key, s.planned) for s in SCHEMA if s.planned],
        "declared_but_unread": unread,
    }


def part_rows(names: dict[str, str]) -> list[dict[str, object]]:
    rows = []
    for part in PARTS:
        exists = [m for m in part.modules if (SRC / m).exists()]
        lines = sum((SRC / m).read_text(encoding="utf-8").count("\n") for m in exists)
        hits = sorted(n for n in names
                      if any(t in n for t in part.tests))
        rows.append({
            "name": part.name,
            "modules": list(part.modules),
            "missing_modules": [m for m in part.modules if m not in exists],
            "lines": lines,
            "tests": len(hits),
            "measured": part.measured,
            "absent": part.absent,
        })
    return rows


def collect(*, with_tests: bool = True) -> dict[str, object]:
    names = test_names()
    invariants = []
    for num, text in sorted(INVARIANTS.items()):
        prefix = f"test_invariant_{num}_"
        mine = sorted(n for n in names if n.startswith(prefix))
        invariants.append({"n": num, "text": text, "tests": mine,
                           "covered": bool(mine),
                           "files": sorted({names[n] for n in mine})})
    data: dict[str, object] = {
        "invariants": invariants,
        "invariants_covered": sum(1 for i in invariants if i["covered"]),
        "settings": settings_summary(),
        "parts": part_rows(names),
        "tests_total": len(names),
        "missing": [{"what": w, "why": y} for w, y in MISSING],
        "figures": sorted(p.name for p in (ROOT / "docs" / "figures").glob("*.png")),
    }
    if with_tests:
        passed, failed, line = run_tests()
        data["tests_passed"] = passed
        data["tests_failed"] = failed
        data["tests_line"] = line
    return data


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------


def render_markdown(d: dict) -> str:
    out: list[str] = []
    w = out.append
    w("# Состояние проекта")
    w("")
    w("Этот файл **сгенерирован** `tools/project_status.py` по коду и тестам. "
      "Править его руками бессмысленно: он перезапишется. Смысл именно в этом — "
      "написанный руками список «что готово» расходится с кодом незаметно.")
    w("")
    if "tests_passed" in d:
        w(f"Тестов: **{d['tests_passed']} прошло**, {d['tests_failed']} упало "
          f"(`{d['tests_line']}`).")
    else:
        w(f"Тестов в наборе: **{d['tests_total']}** (не запускались).")
    w("")

    w("## Инварианты из CLAUDE.md")
    w("")
    w(f"Покрыто тестом: **{d['invariants_covered']} из {len(d['invariants'])}**.")
    w("")
    w("| № | Инвариант | Тестов | Где |")
    w("|---|---|---|---|")
    for i in d["invariants"]:
        mark = str(len(i["tests"])) if i["covered"] else "**НЕТ**"
        where = ", ".join(f"`{f}`" for f in i["files"]) or "—"
        w(f"| {i['n']} | {i['text']} | {mark} | {where} |")
    w("")

    s = d["settings"]
    w("## Настройки")
    w("")
    w(f"Объявлено **{s['total']}**, из них структурных (форкают журнал) "
      f"**{s['structural']}**, групп {len(s['groups'])}.")
    w("")
    if s["declared_but_unread"]:
        w("**Объявлены и нигде не читаются** — это ложь о возможностях, "
          "починить обязательно:")
        w("")
        for k in s["declared_but_unread"]:
            w(f"- `{k}`")
    else:
        w("Каждая настройка либо читается кодом, либо помечена как ещё не "
          "реализованная. Проверяется тестом.")
    w("")
    if s["planned"]:
        w(f"Помечено «ещё не реализовано» ({len(s['planned'])}):")
        w("")
        w("| Настройка | Чего не хватает |")
        w("|---|---|")
        for k, why in s["planned"]:
            w(f"| `{k}` | {why} |")
        w("")

    w("## Части")
    w("")
    w("| Часть | Строк | Тестов | Чем замерено | Чего не хватает |")
    w("|---|---|---|---|---|")
    for p in d["parts"]:
        gap = p["absent"] or ("**модулей нет**" if p["missing_modules"] else "—")
        measured = p["measured"] or "**замера нет**"
        w(f"| {p['name']} | {p['lines']} | {p['tests']} | {measured} | {gap} |")
    w("")

    w("## Чего нет и почему")
    w("")
    w("| Чего нет | Почему |")
    w("|---|---|")
    for m in d["missing"]:
        w(f"| {m['what']} | {m['why']} |")
    w("")

    w("## Картинки")
    w("")
    w("Все рисуются по настоящим прогонам: `python3 tools/make_figures.py docs/figures`.")
    w("")
    for f in d["figures"]:
        w(f"- `docs/figures/{f}`")
    w("")
    return "\n".join(out)


def render_text(d: dict) -> str:
    out = []
    w = out.append
    if "tests_passed" in d:
        w(f"тесты: {d['tests_passed']} прошло, {d['tests_failed']} упало")
    total_tests = sum(len(i["tests"]) for i in d["invariants"])
    w(f"инварианты: {d['invariants_covered']}/{len(d['invariants'])} покрыты, "
      f"всего {total_tests} тестов на инварианты")
    bad = [i for i in d["invariants"] if not i["covered"]]
    for i in bad:
        w(f"   БЕЗ ТЕСТА: инвариант {i['n']} — {i['text']}")
    s = d["settings"]
    w(f"настройки: {s['total']} объявлено, {s['structural']} структурных, "
      f"{len(s['planned'])} ещё не реализованы")
    for k in s["declared_but_unread"]:
        w(f"   ОБЪЯВЛЕНА И НЕ ЧИТАЕТСЯ: {k}")
    w("")
    name_w = max(len(p["name"]) for p in d["parts"])
    w(f"{'часть'.ljust(name_w)}  строк  тестов  замер")
    w(f"{'-' * name_w}  -----  ------  -----")
    for p in d["parts"]:
        mark = "есть" if p["measured"] else "НЕТ "
        w(f"{p['name'].ljust(name_w)}  {p['lines']:5}  {p['tests']:6}  {mark}"
          + (f"  · {p['absent']}" if p["absent"] else ""))
    w("")
    w("чего нет:")
    for m in d["missing"]:
        w(f"   {m['what']} — {m['why']}")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md", type=Path, default=None, help="записать markdown в файл")
    ap.add_argument("--json", action="store_true", help="машинно читаемо")
    ap.add_argument("--no-tests", action="store_true", help="не запускать тесты")
    args = ap.parse_args(argv[1:])

    data = collect(with_tests=not args.no_tests)

    # Строка «чего нет» без причины — ошибка отчёта, а не мелочь: список без причин
    # превращается в отговорку.
    for m in data["missing"]:
        if not m["why"]:
            print(f"нет причины у пункта «{m['what']}»", file=sys.stderr)
            return 2

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_text(data))
    if args.md:
        args.md.write_text(render_markdown(data), encoding="utf-8")
        print(f"\nзаписано: {args.md}")
    bad = [i for i in data["invariants"] if not i["covered"]]
    unread = data["settings"]["declared_but_unread"]
    return 1 if (bad or unread or data.get("tests_failed")) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
