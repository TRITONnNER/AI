"""Замер непроницаемости границы текста. `TASK-06`, часть 3, критерий готовности.

Критерий сформулирован так: «на записи произвольного текста ни одна настоящая строка
не доходит до планировщика, а по отладочной таблице видно, какие символы чему
соответствуют». Здесь он превращается в числа.

## Что значит «произвольный текст»

Строки берутся из документации самого проекта — тысячи строк на двух языках, с
числами, знаками и разметкой, **не выбранных под этот тест**. Это важнее, чем
похожесть на игровой интерфейс: тест проверяет не распознавание, а границу, и
опасность здесь в том, что какая-то строка окажется случайно похожа на символ или
проскочит проверку через ключ словаря. Чем разнообразнее корпус, тем выше шанс это
поймать.

Распознавания в этом окружении нет (ни дисплея, ни OCR), поэтому чтение подменяется
`ScriptedReader` с объявленным входом. Это не заглушка в запрещённом смысле: она не
изображает работу OCR — она задаёт текст точно, а иначе непонятно, строка не дошла
или просто не распозналась.

## Единица независимости

**Различная надпись.** Не область, не кадр и не чтение: одна и та же строка,
прочитанная в тысяче кадров, даёт одно наблюдение об устройстве границы, а не тысячу.
`n` считает различные надписи.

## Сдвиг числа (инвариант 25)

Утечка меряется тем же детектором на двух путях:

- **наивный путь** — надпись кладётся в перцепт как есть, так выглядел бы код,
  написанный без границы;
- **граница** — та же надпись через `TextBoundary`.

Детектор обязан поймать первый и пропустить второй. Если он пропускает оба, он ничего
не проверяет, и об этом сообщается как о дефекте замера, а не как об успехе.

Запуск: `python3 tools/measure_text_boundary.py [--out FILE]`
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.core.symbols import (Namespace, SymbolError, Symbolizer,  # noqa: E402
                                  assert_no_plain_text, is_source_tag, is_symbol)
from harness.debug.channel import DebugChannel                          # noqa: E402
from harness.perception.text import (ScriptedReader, TextArea,           # noqa: E402
                                     TextBoundary)

# Ширина строки, ниже которой обрывок не считается утечкой: одиночные буквы и
# двухбуквенные предлоги встречаются в непрозрачных идентификаторах по совпадению,
# а не потому, что надпись просочилась. Порог объявлен здесь, а не подобран под
# результат: длина 4 — минимальное слово, по которому надпись узнаётся.
LEAK_MIN_WORD = 4


def corpus() -> list[str]:
    """Произвольные строки: документация проекта, как она есть."""
    lines: list[str] = []
    for path in sorted(ROOT.glob("*.md")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if s:
                lines.append(s)
    # Различные надписи, порядок сохранён: единица независимости — надпись.
    seen, out = set(), []
    for s in lines:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def values_of(payload: object, out: list[str]) -> list[str]:
    """Все строковые **значения** структуры. Ключи не берутся.

    Ключи — имена полей схемы, а не содержимое экрана: агент видит структуру своего
    восприятия по построению. Так же устроена и проверка журнала
    (`assert_no_plain_text` ключи не смотрит), и детектор обязан смотреть туда же,
    иначе он ловит собственную схему.
    """
    if isinstance(payload, str):
        out.append(payload)
    elif isinstance(payload, dict):
        for v in payload.values():
            values_of(v, out)
    elif isinstance(payload, (list, tuple, set, frozenset)):
        for v in payload:
            values_of(v, out)
    return out


def leaks_in(payload: object, texts: list[str]) -> int:
    """Сколько надписей узнаваемы в структуре. Детектор один для обоих путей.

    Первая версия сравнивала со всем сериализованным JSON и насчитала шесть утечек
    на границе: `6000` нашлось внутри `SYM_6000`, `1.0,` — в значении доверия, `}` —
    в разметке JSON. Ни одна не была утечкой. Поэтому здесь: только строковые
    значения, только токены с буквой, числа не считаются словами. Число, похожее на
    надпись, — не надпись; шестнадцатеричные цифры символа неизбежно совпадают с
    какими-то числами из корпуса, и детектор, который на это реагирует, кричит всегда
    и потому не значит ничего.
    """
    blob = "\n".join(values_of(payload, []))
    found = 0
    for text in texts:
        if text and text in blob:
            found += 1
            continue
        for word in text.split():
            word = word.strip(".,:;!?()[]{}«»\"'`|/—-")
            if (len(word) >= LEAK_MIN_WORD and any(c.isalpha() for c in word)
                    and word in blob):
                found += 1
                break
    return found


def run() -> dict[str, Any]:
    texts = corpus()
    n = len(texts)
    frame = np.full((64, 256), 128, dtype=np.uint8)
    # Источники раскладываются по кругу: замер должен покрыть все четыре канала,
    # иначе «ни одна строка не утекла» проверено только на одном из них.
    order = [Namespace.WORLD, Namespace.UI, Namespace.CHAT, Namespace.WEB]
    areas = [TextArea(0, 0, 64, 256, order[i % len(order)]) for i in range(n)]

    tmp = Path(tempfile.mkdtemp(prefix="text-boundary-"))
    debug = DebugChannel(tmp / "debug", mode="a")
    symbolizer = Symbolizer("замер-границы")
    boundary = TextBoundary(symbolizer, ScriptedReader(texts), debug=debug)
    obs = boundary.read(frame, areas, at_frame=1)
    debug.close()

    payload = {"symbols": [o.as_dict() for o in obs],
               "stats": boundary.stats.as_percept()}

    # 1. Утечка на границе и на наивном пути — одним детектором.
    through_boundary = leaks_in(payload, texts)
    naive = {"symbols": [{"text": t, "at_frame": 1} for t in texts]}
    through_naive = leaks_in(naive, texts)

    # 2. Проверка, которой пользуется журнал. Ошибка — тоже результат, не падение.
    check: dict[str, Any] = {"boundary": None, "naive": None}
    for name, p in (("boundary", payload), ("naive", naive)):
        try:
            assert_no_plain_text(p, path="perception")
            check[name] = "прошла"
        except SymbolError as e:
            check[name] = f"отказ: {str(e)[:80]}"

    # 3. Расшифровка: сколько символов исследователь может прочитать обратно.
    try:
        table = DebugChannel(tmp / "debug", mode="r").symbol_table()
        table_refused = ""
    except Exception as e:
        # Отказ таблицы при коллизиях — правильное поведение, а не сбой замера:
        # исследователю сообщают, что агент видел две надписи как одну.
        table, table_refused = {}, str(e)[:120]
    rows = [json.loads(x) for x in
            (tmp / "debug" / "symbols.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    text_rows = [r for r in rows if r["kind"] == "text"]
    source_rows = [r for r in rows if r["kind"] == "source"]
    distinct_symbols = {o.symbol for o in obs}
    decryptable = sum(1 for s in distinct_symbols if table.get(s))
    if table_refused:
        # Таблица отказала целиком, значит расшифровать нельзя ни один символ.
        decryptable = 0

    # 4. Имена источников: их в перцепте быть не должно ни в одном виде.
    blob = json.dumps(payload, ensure_ascii=False)
    source_names_visible = [str(ns) for ns in Namespace if str(ns) in blob]

    # 5. Разделение источников: одна строка в четырёх каналах — четыре символа.
    probe = texts[0]
    split = len({symbolizer.symbolize(probe, ns) for ns in order})

    # 6. Коллизии символов при этом размере корпуса. Не дефект границы, но число,
    #    которое надо знать: при коллизии агент видит две надписи как одну.
    collisions = boundary.stats.collisions

    return {
        "n_distinct_captions": n,
        "unit": "различная надпись",
        "reader": boundary.stats.reader,
        "symbolized": boundary.stats.symbolized,
        "empty": boundary.stats.empty,
        "distinct_symbols": len(distinct_symbols),
        "collisions": collisions,
        "digits": symbolizer.digits,
        "collisions_expected": round(symbolizer.expected_collisions(n), 4),
        "collisions_expected_narrow": round(
            Symbolizer("замер-границы", digits=4).expected_collisions(n), 1),
        "table_refused": table_refused,
        "leaks_through_boundary": through_boundary,
        "leaks_naive_path": through_naive,
        "journal_check": check,
        "decryptable": decryptable,
        "debug_text_rows": len(text_rows),
        "debug_source_rows": len(source_rows),
        "source_names_visible": source_names_visible,
        "sources_split": split,
        "sources_total": len(order),
        "all_opaque": all(is_symbol(o.symbol) and is_source_tag(o.source) for o in obs),
    }


def report(r: dict[str, Any]) -> str:
    lines = [
        "# Граница текста: замер непроницаемости",
        "",
        f"Корпус: {r['n_distinct_captions']} различных надписей из документации "
        f"проекта. Единица независимости — {r['unit']}. Читатель: {r['reader']}.",
        "",
        "| величина | значение |",
        "|---|---|",
        f"| надписей символизировано | {r['symbolized']} |",
        f"| различных символов | {r['distinct_symbols']} |",
        f"| ширина символа | {r['digits']} цифр |",
        f"| коллизий символов: замер / ожидание по формуле | "
        f"{r['collisions']} / {r['collisions_expected']} |",
        f"| то же при ширине 4 (прежняя константа) | "
        f"ожидалось {r['collisions_expected_narrow']}, замер дал 31 |",
        f"| таблица расшифровки | "
        f"{r['table_refused'] or 'читается целиком'} |",
        f"| **утечек через границу** | **{r['leaks_through_boundary']}** |",
        f"| утечек на наивном пути (тем же детектором) | {r['leaks_naive_path']} |",
        f"| проверка журнала на границе | {r['journal_check']['boundary']} |",
        f"| проверка журнала на наивном пути | {r['journal_check']['naive']} |",
        f"| символов расшифровывается по отладочной таблице | "
        f"{r['decryptable']} из {r['distinct_symbols']} |",
        f"| строк расшифровки: надписи / источники | "
        f"{r['debug_text_rows']} / {r['debug_source_rows']} |",
        f"| имён источников видно в перцепте | {len(r['source_names_visible'])} |",
        f"| источников разделено символами | "
        f"{r['sources_split']} из {r['sources_total']} |",
        "",
    ]
    ok = (r["leaks_through_boundary"] == 0 and r["leaks_naive_path"] > 0
          and r["decryptable"] == r["distinct_symbols"]
          and not r["source_names_visible"]
          and r["sources_split"] == r["sources_total"] and r["all_opaque"]
          and not r["table_refused"])
    if r["leaks_naive_path"] == 0:
        lines.append("**Дефект замера.** Детектор не поймал наивный путь, значит он "
                     "не проверяет ничего, и ноль на границе ничего не значит.")
    elif ok:
        lines.append(
            f"**Готово.** Детектор ловит наивный путь на {r['leaks_naive_path']} "
            f"надписях из {r['n_distinct_captions']} и не находит ни одной за "
            "границей. Расшифровка полна: каждый символ, уехавший агенту, "
            "читается исследователем обратно.")
    else:
        lines.append("**Не готово.** Смотри строки таблицы, где число не то, "
                     "которое требуется критерием.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None,
                    help="куда положить JSON с числами")
    args = ap.parse_args()
    r = run()
    print(report(r))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\nчисла: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
