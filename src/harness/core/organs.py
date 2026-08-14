"""Органы: конкретные ресурсы машины, а не абстрактная загрузка. TASK-32, направление B.

«Это его тело, и органы у него разного размера». Абстрактная «загрузка 60 %» не говорит
ничего о том, что можно сделать: 60 % чего именно, и сколько осталось в тех единицах, в
которых считается решение. Поэтому здесь — конкретные величины с единицами и **источником**
каждой.

## Правило отказа

Величина, которую нечем прочитать, возвращается как `None` с названной причиной. **Ноль
здесь запрещён**: «видеопамяти свободно 0» и «видеопамять читать нечем» ведут к
противоположным решениям, и подменять второе первым — та самая молчаливая заглушка, которая
ломает эксперимент незаметно.

## Почему это доступно агенту

Свободная память — не подсказка о мире (инвариант 4), а свойство собственного тела: агент
имеет право знать, сколько у него места, ровно как он знает, что у него есть выходы. Ни
одно имя файла, ни один путь и ни одно имя процесса сюда не попадают — только числа и
единицы.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Organ:
    """Один ресурс: сколько всего, сколько свободно, чем прочитано.

    `total is None` и `free is None` означают «прочитать нечем», и тогда обязателен
    `why` — иначе незнание выглядит как знание.
    """

    name: str
    unit: str
    total: float | None
    free: float | None
    source: str
    why: str = ""

    def __post_init__(self) -> None:
        if (self.total is None or self.free is None) and not self.why:
            raise ValueError(
                f"орган {self.name}: величина не прочитана, а причина не названа. "
                "Ноль вместо причины читался бы как «ничего не осталось»")

    @property
    def known(self) -> bool:
        return self.total is not None and self.free is not None

    @property
    def free_share(self) -> float | None:
        if not self.known or not self.total:
            return None
        return max(0.0, min(1.0, float(self.free or 0.0) / float(self.total)))

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "unit": self.unit, "total": self.total,
                "free": self.free, "source": self.source, "why": self.why,
                "known": self.known,
                "free_share": (None if self.free_share is None
                               else round(self.free_share, 4))}


def ram(proc: str = "/proc/meminfo") -> Organ:
    """Оперативная память: всего и доступно. Источник — `/proc/meminfo`.

    `MemAvailable`, а не `MemFree`: свободная память в Linux почти всегда мала, потому что
    ядро держит кэш, и решение по `MemFree` отказывалось бы работать на здоровой машине.
    """
    try:
        text = Path(proc).read_text(encoding="ascii")
    except OSError as exc:
        return Organ("оперативная память", "МиБ", None, None, proc,
                     why=f"нечем прочитать: {exc.__class__.__name__}")
    got: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            try:
                got[parts[0][:-1]] = float(parts[1]) / 1024.0     # КиБ → МиБ
            except ValueError:
                continue
    total, avail = got.get("MemTotal"), got.get("MemAvailable")
    if total is None or avail is None:
        return Organ("оперативная память", "МиБ", None, None, proc,
                     why="в meminfo нет MemTotal или MemAvailable")
    return Organ("оперативная память", "МиБ", total, avail, proc)


def disk(path: str | Path) -> Organ:
    """Диск по конкретному пути: тот, куда пишется журнал, а не «диск вообще»."""
    target = Path(path)
    probe = target if target.exists() else target.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return Organ("диск", "МиБ", None, None, str(probe),
                     why=f"нечем прочитать: {exc.__class__.__name__}")
    return Organ("диск", "МиБ", usage.total / 1024 ** 2, usage.free / 1024 ** 2,
                 str(probe))


def vram(run: Any = None) -> Organ:
    """Видеопамять: всего и свободно. Источник — `nvidia-smi`, если он есть.

    Отсутствие видеокарты и отсутствие утилиты — разные причины, и обе называются. Ни в
    одном случае не возвращается ноль: ноль означал бы «видеопамять кончилась», и агент
    отказался бы от того, что на самом деле возможно.
    """
    runner = run or subprocess.run
    cmd = ("nvidia-smi", "--query-gpu=memory.total,memory.free",
           "--format=csv,noheader,nounits")
    try:
        done = runner(cmd, capture_output=True, text=True, timeout=5.0, check=False)
    except FileNotFoundError:
        return Organ("видеопамять", "МиБ", None, None, "nvidia-smi",
                     why="nvidia-smi не установлен: видеокарты может не быть вовсе, а "
                         "может не быть только утилиты — это разные причины, и обе не ноль")
    except Exception as exc:                       # noqa: BLE001 — причина важнее типа
        return Organ("видеопамять", "МиБ", None, None, "nvidia-smi",
                     why=f"опрос не удался: {exc.__class__.__name__}")
    if getattr(done, "returncode", 1) != 0:
        return Organ("видеопамять", "МиБ", None, None, "nvidia-smi",
                     why=f"nvidia-smi ответил кодом {done.returncode}")
    line = (done.stdout or "").strip().splitlines()
    if not line:
        return Organ("видеопамять", "МиБ", None, None, "nvidia-smi",
                     why="nvidia-smi ответил пусто")
    try:
        total_s, free_s = line[0].split(",")[:2]
        return Organ("видеопамять", "МиБ", float(total_s), float(free_s), "nvidia-smi")
    except ValueError:
        return Organ("видеопамять", "МиБ", None, None, "nvidia-smi",
                     why=f"ответ не разобрался: {line[0][:40]!r}")


def rss(pid: int | None = None) -> Organ:
    """Сколько памяти машины ещё **не занято мной**.

    Величина названа так нарочно. Читать её как «сколько свободно» нельзя: свободное на
    машине считает `ram()`, а здесь — своя доля от тела. Числа те же по форме, смысл
    разный, и путать их значило бы дважды посчитать одно.
    """
    p = os.getpid() if pid is None else pid
    path = f"/proc/{p}/statm"
    try:
        pages = int(Path(path).read_text(encoding="ascii").split()[1])
    except (OSError, IndexError, ValueError) as exc:
        return Organ("память не под мной", "МиБ", None, None, path,
                     why=f"нечем прочитать: {exc.__class__.__name__}")
    mb = pages * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2
    # «Всего» здесь — оперативная память машины: моя доля от тела, а не от себя самой.
    whole = ram()
    return Organ("память не под мной", "МиБ", whole.total,
                 max(0.0, (whole.total or mb) - mb), path,
                 why="" if whole.known else "объём машины неизвестен")


def read_all(*, journal_path: str | Path = ".", run: Any = None) -> dict[str, Any]:
    """Все органы разом — то, что уходит в отчёт и в срез состояния.

    `unknown` печатается отдельным числом: сколько органов прочитать не удалось. Иначе
    отчёт с тремя `None` выглядел бы так же, как отчёт со тремя нулями.
    """
    organs = [ram(), disk(journal_path), vram(run=run), rss()]
    return {
        "organs": [o.as_dict() for o in organs],
        "known": sum(1 for o in organs if o.known),
        "unknown": sum(1 for o in organs if not o.known),
        "unknown_why": {o.name: o.why for o in organs if not o.known},
        "unit": "орган",
    }
