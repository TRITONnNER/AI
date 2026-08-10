"""Замер по TASK-15: сколько тестов зелены только потому, что идут в контейнере.

Первый прогон на машине оператора уронил пять тестов, и ни один не был про код: они
проверяли окружение, а не поведение. Отсюда вопрос, на который ответа не было: **сколько
ещё таких?** 636 зелёных рядом с пятью такими падениями означают, что доля ложно-зелёных
неизвестна (инвариант 31).

Способ измерить один: прогнать тот же набор в **другом объявленном окружении** и
посмотреть, у каких тестов изменился исход. Тест, чей исход зависит от машины, а
зависимость не объявлена, — дефект того же класса, что и пять найденных.

Окружения подменяет плагин `tools/envplug.py`: `machine.detect` и наличие пакетов
захвата. Переменные окружения не подменяются намеренно — на Windows их нет вовсе, и
подмена их ничего не значит; ровно эта ошибка и сидела в пяти тестах.

**Единица независимости — тест.** Не прогон: один прогон даёт по одному наблюдению на
каждый тест, и все они разные. Не окружение: окружений пять, но утверждение делается о
тестах.

Запуск: `python3 tools/measure_env_dependence.py`. Идёт около восьми минут (пять полных
прогонов набора). Результат — `docs/measurements/env_dependence.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
# Корень тоже: `tools` — пакет, и при запуске файлом первым в пути оказывается `tools/`,
# а не репозиторий. Без этой строки замер падает `ModuleNotFoundError: tools`.
sys.path.insert(0, str(ROOT))

from tools.envplug import ENVIRONMENTS      # noqa: E402

#: Состояние до правок TASK-15. Текущим кодом не воспроизводится: тесты уже исправлены, а
#: три копии знания о Wayland сведены в одну. Записано здесь потому, что это и есть
#: результат — доля ложно-зелёных **была** и стала известна.
BEFORE: dict[str, Any] = {
    "на машине оператора": {
        "окружение": "Windows 10, дисплей есть, dxcam и mss установлены",
        "упало": 5, "прошло": 636,
        "почему": "три теста сравнивали поведение с наличием DISPLAY, ждали отказа от "
                  "установленного пакета и ждали кода 2 от прошедшей записи; два "
                  "выставляли XDG_SESSION_TYPE, чего detect на Windows не читает",
    },
    "в подменённых окружениях": {
        "окружение": "windows / x11 / macos, подменено плагином",
        "упало": 6, "прошло": 637,
        "почему": "к пяти добавился шестой: `harness doctor` падал с трассировкой, "
                  "когда установленный пакет захвата бросал не BackendUnavailable, а "
                  "своё исключение. Это дефект кода, а не теста, и нашёл его замер",
    },
}


#: Чего замер не видит по устройству, а не по нехватке данных. Пустой тест исхода не
#: меняет: он зелен во всех окружениях сразу, потому что не проверяет ничего. Такие нашла
#: статическая проверка `tests/test_environment_rules.py`, и это разделение двух
#: инструментов, а не оговорка.
BLIND_SPOT: dict[str, Any] = {
    "что": "пустой тест — утверждение под условием о машине, недостижимым в этом "
           "окружении. Исход не меняется, потому что проверки не было ни в одном",
    "кем найдено": "статическая проверка объявления окружения",
    "сколько": 3,
    "какие": [
        "test_operator.py::test_doctor_says_which_mechanism_and_why — утверждение под "
        "`if check.state is YES`, а в контейнере графической сессии нет: не выполнялось "
        "ни разу",
        "test_operator.py::test_blocking_and_later_are_separate_and_cannot_be_mixed — "
        "утверждение под `if uinput.state is not YES`",
        "test_operator.py::test_every_missing_check_carries_a_command_to_copy — цикл "
        "только по нехваткам: там, где всё зелёное, витков нет",
    ],
}


def run(env: str, *, quiet: bool = True) -> dict[str, Any]:
    """Один полный прогон набора в объявленном окружении."""
    cmd = [sys.executable, "-m", "pytest", "-p", "tools.envplug", "-q",
           "--no-header", "-rf"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "HARNESS_FAKE_ENV": env,
                               "PYTHONPATH": str(ROOT / "src")})
    out = proc.stdout + proc.stderr
    failed = sorted({m.group(1) for m in
                     re.finditer(r"^FAILED (\S+)", out, re.MULTILINE)})
    tail = re.search(r"(\d+) passed", out)
    passed = int(tail.group(1)) if tail else 0
    skipped = re.search(r"(\d+) skipped", out)
    if not quiet:
        print(out[-2000:])
    return {"env": env, "passed": passed, "failed": failed,
            "skipped": int(skipped.group(1)) if skipped else 0,
            "total": passed + len(failed) + (int(skipped.group(1)) if skipped else 0)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", default=" ".join(ENVIRONMENTS),
                    help="какие окружения прогнать")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "env_dependence.json")
    args = ap.parse_args(argv[1:])

    envs = args.envs.split()
    runs = []
    for env in envs:
        got = run(env)
        runs.append(got)
        print(f"{env:<10} прошло {got['passed']}, упало {len(got['failed'])}, "
              f"пропущено {got['skipped']}"
              + (f" — {', '.join(x.split('::')[-1] for x in got['failed'])}"
                 if got["failed"] else ""))

    base = next((r for r in runs if r["env"] == "container"), runs[0])
    # Число собранных тестов обязано совпасть во всех окружениях: подменяется машина, а
    # не набор. Расхождение означает, что файлы правились **во время** замера, и сравнение
    # исходов сравнивает разные наборы. Первый прогон этого замера так и вышел — 643
    # против 649, — и молча дал бы «доля ложно-зелёных 0.31 %» из воздуха.
    totals = {r["env"]: r["total"] for r in runs}
    if len(set(totals.values())) > 1 and len(runs) > 1:
        raise SystemExit(
            "набор изменился между прогонами: " + ", ".join(
                f"{k}={v}" for k, v in totals.items())
            + ".\nЗамер сравнивает исходы одного и того же набора; правки во время "
              "прогона делают сравнение бессмысленным. Повторите на неизменных файлах")
    # Ложно-зелёный — тест, зелёный в контейнере и не зелёный в другом объявленном
    # окружении. Пропуск ложно-зелёным **не является**: пропущенный тест не утверждает,
    # что проверка прошла, — он говорит, что не выполнялась.
    flipped = sorted({t for r in runs for t in r["failed"]}
                     - set(base["failed"]))
    total = base["total"] or 1
    data = {
        "runs": runs,
        "environments": {k: v for k, v in ENVIRONMENTS.items() if k in envs},
        "before": BEFORE,
        "blind_spot": BLIND_SPOT,
        "flipped": flipped,
        "false_green_share": round(len(flipped) / total, 6),
        "coverage": {
            "tests_run": base["total"],
            "environments": len(envs),
            "note": "покрытие — по числу тестов, исполненных в каждом окружении. "
                    "Замер видит только то, что запускалось: тест, пропущенный во всех "
                    "окружениях, о своей зависимости не сообщает никак",
        },
        "unit": "тест",
        "why_not_env_vars": "переменные окружения не подменяются: на Windows их нет, и "
                            "подмена их ничего не меняет. Подменяется machine.detect — "
                            "единственное место, где проект отвечает, что за машина",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print()
    print(f"тестов в прогоне: {base['total']}, окружений: {len(envs)}")
    print(f"меняют исход между окружениями: {len(flipped)}"
          + (f" — {flipped}" if flipped else ""))
    print(f"доля ложно-зелёных: {data['false_green_share']:.2%}")
    print(f"было до правок: {BEFORE['в подменённых окружениях']['упало']} из "
          f"{BEFORE['в подменённых окружениях']['упало'] + BEFORE['в подменённых окружениях']['прошло']}")
    # Путь печатается **после** попытки укоротить, а не вместо неё: при относительном
    # `--out` (`docs/measurements/...`) `relative_to` роняло прогон уже после записи файла,
    # то есть замер был сделан, а сообщение о нём — нет. Печать не имеет права ронять то,
    # что уже посчитано.
    try:
        shown = args.out.resolve().relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"записано: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
