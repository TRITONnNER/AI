#!/usr/bin/env python3
"""Проверка механической целостности выгрузки проекта из Claude Design.

Скрипт не читает смысл документа. Он проверяет только то, что проверяется
формально: все ли части на месте, совпадают ли разделы с манифестом, не
осталось ли заглушек, сходится ли блок самопроверки с фактом.

Использование:
    python3 tools/check_design_export.py docs/design-draft/

Код возврата: 0 — ошибок нет, 1 — есть ошибки, 2 — не удалось разобрать ввод.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# --- разметка, которую обещал промт выгрузки -------------------------------

RE_MANIFEST_OPEN = re.compile(r"^===\s*MANIFEST\s*===\s*$")
RE_MANIFEST_CLOSE = re.compile(r"^===\s*END\s+MANIFEST\s*===\s*$")
RE_TOTAL_PARTS = re.compile(r"^TOTAL_PARTS:\s*(\d+)\s*$")
RE_SECTIONS_COUNT = re.compile(r"^SECTIONS:\s*(\d+)\s*$")
# "3. Словарь терминов | 1 | ПОЛНЫЙ"  (во второй колонке допускается "часть 1")
RE_MANIFEST_ROW = re.compile(
    r"^\s*(\d+)[.)]\s*(.+?)\s*\|\s*(?:часть\s*)?(\d+)\s*\|\s*([А-ЯA-Z]+)\s*$",
    re.IGNORECASE,
)

RE_PART_OPEN = re.compile(r"^===\s*ЧАСТЬ\s+(\d+)\s*/\s*(\d+)\s*===\s*$")
RE_PART_CLOSE = re.compile(r"^===\s*КОНЕЦ\s+ЧАСТИ\s+(\d+)\s*/\s*(\d+)\s*===\s*$")

RE_SELFCHECK_OPEN = re.compile(r"^===\s*SELF-CHECK\s*===\s*$")
RE_SELFCHECK_CLOSE = re.compile(r"^===\s*END\s+SELF-CHECK\s*===\s*$")
RE_SELFCHECK_ROW = re.compile(r"^([A-Z_]+(?:\s*/\s*[A-Z_]+)*):\s*(.*)$")

RE_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
RE_FENCE = re.compile(r"^\s*(```|~~~)")

VALID_STATUSES = {"ПОЛНЫЙ", "ЧАСТИЧНЫЙ", "ПУСТОЙ"}

# Заглушки. Ключ — регулярное выражение, значение — что именно не так.
PLACEHOLDERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<!\.)\.\.\.(?!\.)"), "троеточие вместо текста"),
    (re.compile(r"…"), "троеточие вместо текста"),
    (re.compile(r"\bTODO\b|\bFIXME\b|\bTBD\b|\bXXX\b", re.IGNORECASE), "маркер незаконченного"),
    (re.compile(r"и\s+так\s+далее|и\s+т\.\s*д\.|и\s+прочее", re.IGNORECASE), "перечисление свёрнуто"),
    (re.compile(r"аналогично\s+(выше|предыдущ)", re.IGNORECASE), "ссылка вместо текста"),
    # предлог между словами допускается: «см. в обсуждении», «смотри у нас выше»
    (re.compile(r"(см\.|смотри)\s+(?:[а-я]{1,3}\s+){0,2}(выше|ниже|обсуждени|переписк)",
                re.IGNORECASE), "ссылка вместо текста"),
    (re.compile(r"(описан|указан|рассмотрен)[а-я]*\s+(выше|ранее|ниже)", re.IGNORECASE), "ссылка вместо текста"),
    (re.compile(r"как\s+обсуждалось", re.IGNORECASE), "ссылка вместо текста"),
    (re.compile(r"стандартн[а-я]+\s+подход", re.IGNORECASE), "решение подменено отговоркой"),
    (re.compile(r"(здесь|тут|там|дальше)\s+(всё|все)\s+как\s+обычно",
                re.IGNORECASE), "решение подменено отговоркой"),
]

# «НЕ ОБСУЖДАЛОСЬ» — законная формулировка из промта, не заглушка.
RE_LEGIT_EMPTY = re.compile(r"НЕ\s+ОБСУЖДАЛОСЬ")

MIN_PART_WORDS = 250  # ниже этого часть подозрительно коротка


@dataclass
class Problem:
    severity: str  # "ERROR" | "WARN"
    where: str
    message: str


@dataclass
class ManifestRow:
    number: int
    title: str
    part: int
    status: str


@dataclass
class Heading:
    level: int
    text: str
    line: int
    body_chars: int = 0
    has_children: bool = False


@dataclass
class Part:
    number: int
    declared_total: int
    path: Path
    first_line: int
    headings: list[Heading] = field(default_factory=list)
    word_count: int = 0
    closed: bool = False


def normalize(text: str) -> str:
    """Приводит заголовок к виду, по которому его можно сравнивать."""
    text = unicodedata.normalize("NFKD", text).casefold()
    text = re.sub(r"^[\d.)\s]+", "", text)          # ведущая нумерация
    text = re.sub(r"[^\w\s]+", " ", text)            # знаки препинания
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def titles_match(a: str, b: str) -> bool:
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def iter_code_mask(lines: list[str]) -> list[bool]:
    """Отмечает строки внутри блоков кода — в них заглушки не ищем."""
    mask = [False] * len(lines)
    inside = False
    for i, line in enumerate(lines):
        if RE_FENCE.match(line):
            inside = not inside
            mask[i] = True
            continue
        mask[i] = inside
    return mask


# --- разбор манифеста ------------------------------------------------------


def parse_manifest(files: list[Path], problems: list[Problem]) -> tuple[list[ManifestRow], int | None, int | None]:
    rows: list[ManifestRow] = []
    total_parts: int | None = None
    declared_sections: int | None = None
    found_in: Path | None = None

    for path in files:
        lines = read_lines(path)
        inside = False
        closed = False
        for lineno, line in enumerate(lines, 1):
            if RE_MANIFEST_OPEN.match(line):
                if found_in is not None and found_in != path:
                    problems.append(
                        Problem("ERROR", f"{path.name}:{lineno}",
                                f"второй манифест — первый был в {found_in.name}; "
                                "оставьте только исправленный")
                    )
                inside = True
                found_in = path
                continue
            if RE_MANIFEST_CLOSE.match(line):
                inside = False
                closed = True
                continue
            if not inside:
                continue

            if m := RE_TOTAL_PARTS.match(line):
                total_parts = int(m.group(1))
            elif m := RE_SECTIONS_COUNT.match(line):
                declared_sections = int(m.group(1))
            elif m := RE_MANIFEST_ROW.match(line):
                status = m.group(4).upper()
                if status not in VALID_STATUSES:
                    problems.append(
                        Problem("ERROR", f"{path.name}:{lineno}",
                                f"статус «{status}» не из набора "
                                f"{'/'.join(sorted(VALID_STATUSES))}")
                    )
                rows.append(ManifestRow(int(m.group(1)), m.group(2), int(m.group(3)), status))
            elif line.strip():
                problems.append(
                    Problem("WARN", f"{path.name}:{lineno}",
                            f"строка манифеста не разобрана: {line.strip()[:80]}")
                )
        if inside and not closed:
            problems.append(Problem("ERROR", path.name, "манифест не закрыт строкой === END MANIFEST ==="))

    if found_in is None:
        problems.append(Problem("ERROR", "-", "манифест не найден ни в одном файле"))
        return rows, total_parts, declared_sections

    if total_parts is None:
        problems.append(Problem("ERROR", found_in.name, "в манифесте нет TOTAL_PARTS"))
    if declared_sections is None:
        problems.append(Problem("ERROR", found_in.name, "в манифесте нет SECTIONS"))
    elif declared_sections != len(rows):
        problems.append(
            Problem("ERROR", found_in.name,
                    f"SECTIONS обещает {declared_sections} разделов, перечислено {len(rows)}")
        )

    numbers = [r.number for r in rows]
    expected = list(range(1, len(rows) + 1))
    if numbers != expected:
        problems.append(
            Problem("ERROR", found_in.name,
                    f"нумерация разделов не сплошная: {numbers}")
        )

    return rows, total_parts, declared_sections


# --- разбор частей ---------------------------------------------------------


def parse_parts(files: list[Path], problems: list[Problem]) -> list[Part]:
    parts: list[Part] = []

    for path in files:
        lines = read_lines(path)
        code_mask = iter_code_mask(lines)
        current: Part | None = None

        for lineno, line in enumerate(lines, 1):
            if m := RE_PART_OPEN.match(line):
                if current is not None:
                    problems.append(
                        Problem("ERROR", f"{path.name}:{lineno}",
                                f"часть {m.group(1)} началась, пока часть "
                                f"{current.number} не закрыта")
                    )
                    parts.append(current)
                current = Part(int(m.group(1)), int(m.group(2)), path, lineno)
                continue

            if m := RE_PART_CLOSE.match(line):
                if current is None:
                    problems.append(
                        Problem("ERROR", f"{path.name}:{lineno}",
                                f"закрытие части {m.group(1)} без открытия")
                    )
                    continue
                if int(m.group(1)) != current.number:
                    problems.append(
                        Problem("ERROR", f"{path.name}:{lineno}",
                                f"часть открыта как {current.number}, закрыта как {m.group(1)}")
                    )
                current.closed = True
                parts.append(current)
                current = None
                continue

            if current is None:
                continue

            if code_mask[lineno - 1]:
                continue

            if hm := RE_HEADING.match(line):
                level = len(hm.group(1))
                for prev in reversed(current.headings):
                    if prev.level < level:
                        prev.has_children = True
                        break
                    if prev.level >= level:
                        break
                current.headings.append(Heading(level, hm.group(2), lineno))
            elif line.strip():
                current.word_count += len(line.split())
                if current.headings:
                    current.headings[-1].body_chars += len(line.strip())

        if current is not None:
            problems.append(
                Problem("ERROR", f"{path.name}:{current.first_line}",
                        f"часть {current.number} не закрыта строкой "
                        f"=== КОНЕЦ ЧАСТИ {current.number}/{current.declared_total} ===")
            )
            parts.append(current)

    return parts


def check_part_sequence(parts: list[Part], manifest_total: int | None, problems: list[Problem]) -> None:
    if not parts:
        problems.append(Problem("ERROR", "-", "ни одной части не найдено"))
        return

    totals = {p.declared_total for p in parts}
    if len(totals) > 1:
        problems.append(
            Problem("ERROR", "-", f"части заявляют разное общее число: {sorted(totals)}")
        )
    total = parts[0].declared_total

    if manifest_total is not None and manifest_total != total:
        problems.append(
            Problem("ERROR", "-",
                    f"манифест обещает {manifest_total} частей, части заявляют {total}")
        )

    seen: dict[int, Part] = {}
    for p in parts:
        if p.number in seen:
            problems.append(
                Problem("ERROR", f"{p.path.name}:{p.first_line}",
                        f"часть {p.number} встречается второй раз "
                        f"(первая — {seen[p.number].path.name})")
            )
        else:
            seen[p.number] = p

    for expected in range(1, total + 1):
        if expected not in seen:
            problems.append(Problem("ERROR", "-", f"часть {expected}/{total} отсутствует"))

    for p in parts:
        if p.word_count < MIN_PART_WORDS:
            problems.append(
                Problem("WARN", f"{p.path.name}:{p.first_line}",
                        f"часть {p.number} короткая — {p.word_count} слов; "
                        "возможно, модель свернула содержание")
            )


def check_sections(rows: list[ManifestRow], parts: list[Part], problems: list[Problem]) -> None:
    by_number = {p.number: p for p in parts}

    for row in rows:
        target = by_number.get(row.part)
        if target is None:
            problems.append(
                Problem("ERROR", "-",
                        f"раздел {row.number} «{row.title}» обещан в части "
                        f"{row.part}, а такой части нет")
            )
            continue

        found_here = any(titles_match(row.title, h.text) for h in target.headings)
        if found_here:
            continue

        elsewhere = [p.number for p in parts
                     if any(titles_match(row.title, h.text) for h in p.headings)]
        if elsewhere:
            problems.append(
                Problem("ERROR", "-",
                        f"раздел {row.number} «{row.title}» обещан в части "
                        f"{row.part}, найден в части {elsewhere[0]}")
            )
        else:
            problems.append(
                Problem("ERROR", "-",
                        f"раздел {row.number} «{row.title}» не найден ни в одной части")
            )

    for p in parts:
        for h in p.headings:
            if h.has_children or h.body_chars > 0:
                continue
            problems.append(
                Problem("ERROR", f"{p.path.name}:{h.line}",
                        f"заголовок «{h.text}» пустой: ни текста, ни подразделов")
            )


def check_placeholders(files: list[Path], problems: list[Problem]) -> int:
    count = 0
    for path in files:
        lines = read_lines(path)
        code_mask = iter_code_mask(lines)
        for lineno, line in enumerate(lines, 1):
            if code_mask[lineno - 1] or RE_LEGIT_EMPTY.search(line):
                continue
            for pattern, why in PLACEHOLDERS:
                if pattern.search(line):
                    count += 1
                    problems.append(
                        Problem("ERROR", f"{path.name}:{lineno}",
                                f"{why}: {line.strip()[:90]}")
                    )
                    break
    return count


def check_schema_units(files: list[Path], problems: list[Problem]) -> None:
    """Таблица со колонкой «тип» обязана иметь колонку с единицами измерения."""
    for path in files:
        for lineno, line in enumerate(read_lines(path), 1):
            if not line.strip().startswith("|"):
                continue
            cells = [normalize(c) for c in line.strip().strip("|").split("|")]
            if not any(c == "тип" or c.startswith("тип ") for c in cells):
                continue
            if any("едини" in c for c in cells):
                continue
            problems.append(
                Problem("WARN", f"{path.name}:{lineno}",
                        "в схеме данных нет колонки с единицами измерения")
            )


def check_selfcheck(files: list[Path], parts: list[Part], rows: list[ManifestRow],
                    placeholder_count: int, problems: list[Problem]) -> None:
    values: dict[str, str] = {}
    found = False

    for path in files:
        lines = read_lines(path)
        inside = False
        for lineno, line in enumerate(lines, 1):
            if RE_SELFCHECK_OPEN.match(line):
                inside = True
                found = True
                continue
            if RE_SELFCHECK_CLOSE.match(line):
                inside = False
                continue
            if inside and (m := RE_SELFCHECK_ROW.match(line.strip())):
                values[re.sub(r"\s+", "", m.group(1))] = m.group(2).strip()

    if not found:
        problems.append(Problem("ERROR", "-", "блок === SELF-CHECK === отсутствует"))
        return

    def as_int(key: str) -> int | None:
        raw = values.get(key)
        if raw is None:
            problems.append(Problem("ERROR", "SELF-CHECK", f"нет поля {key}"))
            return None
        m = re.search(r"\d+", raw)
        if m is None:
            problems.append(
                Problem("ERROR", "SELF-CHECK", f"в поле {key} нет числа: «{raw}»")
            )
            return None
        return int(m.group(0))

    if (claimed := as_int("PARTS_EMITTED")) is not None:
        actual = len({p.number for p in parts})
        if claimed != actual:
            problems.append(
                Problem("ERROR", "SELF-CHECK",
                        f"PARTS_EMITTED = {claimed}, фактически частей {actual}")
            )

    if (claimed := as_int("SECTIONS_EMITTED")) is not None and claimed != len(rows):
        problems.append(
            Problem("ERROR", "SELF-CHECK",
                    f"SECTIONS_EMITTED = {claimed}, в манифесте {len(rows)} разделов")
        )

    if (claimed := as_int("PLACEHOLDERS")) is not None:
        if claimed != 0:
            problems.append(
                Problem("ERROR", "SELF-CHECK",
                        f"модель сама признаёт {claimed} заглушек — выгрузка неполная")
            )
        elif placeholder_count:
            problems.append(
                Problem("ERROR", "SELF-CHECK",
                        f"PLACEHOLDERS = 0, а найдено {placeholder_count}")
            )

    dropped = values.get("DROPPED", "")
    if dropped and normalize(dropped) not in {"нет", "ничего", "none", "no"}:
        problems.append(
            Problem("WARN", "SELF-CHECK",
                    f"модель признаёт потерянное содержание: {dropped[:120]}")
        )

    conflicts = values.get("CONFLICTS", "")
    if conflicts and (m := re.search(r"\d+", conflicts)) and int(m.group(0)) > 0:
        problems.append(
            Problem("WARN", "SELF-CHECK",
                    f"расхождений с CLAUDE.md заявлено {m.group(0)} — разобрать вручную")
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="каталог с частями выгрузки или один файл")
    args = ap.parse_args()

    if args.path.is_dir():
        # README — описание каталога, а не часть выгрузки
        files = [p for p in sorted(args.path.glob("*.md")) if p.stem.casefold() != "readme"]
        if not files:
            print(f"в {args.path} пока нет частей выгрузки — только описание каталога.\n"
                  "Промт выгрузки: docs/PROMPT-EXPORT-DESIGN.md", file=sys.stderr)
            return 2
    elif args.path.is_file():
        files = [args.path]
    else:
        print(f"не найдено: {args.path}", file=sys.stderr)
        return 2

    problems: list[Problem] = []
    rows, manifest_total, _ = parse_manifest(files, problems)
    parts = parse_parts(files, problems)

    check_part_sequence(parts, manifest_total, problems)
    if rows:
        check_sections(rows, parts, problems)
    placeholder_count = check_placeholders(files, problems)
    check_schema_units(files, problems)
    check_selfcheck(files, parts, rows, placeholder_count, problems)

    errors = [p for p in problems if p.severity == "ERROR"]
    warns = [p for p in problems if p.severity == "WARN"]

    print(f"файлов: {len(files)}   частей: {len({p.number for p in parts})}   "
          f"разделов в манифесте: {len(rows)}   слов: {sum(p.word_count for p in parts)}")
    if rows:
        by_status = {s: sum(1 for r in rows if r.status == s) for s in sorted(VALID_STATUSES)}
        print("статусы разделов: " + "  ".join(f"{k}={v}" for k, v in by_status.items()))
    print()

    for group, label in ((errors, "ОШИБКА"), (warns, "ВНИМАНИЕ")):
        for p in group:
            print(f"[{label}] {p.where}: {p.message}")
    if problems:
        print()

    if errors:
        print(f"выгрузка неполная: ошибок {len(errors)}, предупреждений {len(warns)}")
        return 1

    print(f"механика в порядке; предупреждений {len(warns)}. "
          "Смысл проверяется по docs/DESIGN-REVIEW-CHECKLIST.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
