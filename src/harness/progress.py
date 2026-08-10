"""Ход записи: что видит оператор, пока идёт запись.

`TASK-16`. `harness record --seconds 60` печатал три строки и молчал всю минуту.
Оператор дважды прервал запись, приняв её за зависшую, — оба раза `KeyboardInterrupt`
пришёл внутри `zlib.compress`, то есть **в момент нормальной работы**. Живого корпуса у
проекта нет именно из-за этого: единственный шаг, который может сделать только человек,
не выполняется, потому что человеку не показывают, что шаг идёт.

Поэтому здесь не украшение вывода, а часть механизма записи, и у неё есть свойство,
которого нет ни у одной другой печати в проекте: **этот вывод сам меняет содержимое
экрана**, а на записи вида «неподвижность» изменение экрана — измеряемая величина.

Выбор объявлен, а не сделан молча:

- по умолчанию ход идёт **строкой на месте** (`\\r`, без перевода): для записей, где
  оператор что-то делает, вклад терминала в изменения кадра пренебрежим рядом с тем, что
  делает он сам;
- для записи **неподвижности** ход идёт **в файл** рядом с сессией, а на экран печатается
  одна строка о том, что ход пишется туда. Опорный уровень «всё, что меняется здесь,
  меняется само» нельзя мерить, добавляя в кадр собственную мигающую строку. Верхняя
  оценка вклада — арифметическая: при обновлении раз в секунду и захвате 30 кадр/с
  меняющаяся строка делает изменившимся один оборот из тридцати, то есть до 3.3 % кадров
  вместо «без изменений». Это и есть порядок величины, которую записывают, — то есть
  испорчен был бы именно замер, а не оформление;
- `none` — не печатать ничего. Нужен для прогонов из сценария, где вывод перехватывается.

Частота обновления — настройка профиля (`progress_min_interval_s`), а не константа: она
меняет и удобство, и вклад в кадр, поэтому обязана попадать в `profile_hash`.

Строка собирается **чистой функцией** `render_line`, а канал вывода отделён от сборки:
иначе проверять ход записи можно было бы только глазами на машине с дисплеем, а этого
здесь нет.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

#: Каналы вывода хода. Набор закрыт: «ещё один похожий» ничего не проверяет, а
#: недостающий виден сразу — отказом по неизвестному имени.
CHANNELS: tuple[str, ...] = ("line", "file", "none")

#: Виды записи, для которых ход пишется в файл, а не на экран. Причина — в докстринге
#: модуля: на этих записях изменение экрана и есть измеряемая величина.
#:
#: Список, а не одно имя: `stillness` сегодня единственная такая запись, но свойство
#: («мерится отсутствие изменений») принадлежит виду записи, а не имени, и второй такой
#: вид добавится строкой здесь, а не правкой условия в трёх местах.
FILE_KINDS: tuple[str, ...] = ("stillness",)


def channel_for(kind: str | None, *, default: str) -> tuple[str, str]:
    """Куда писать ход и **почему**. Возвращает (канал, причина).

    Причина возвращается вместе с выбором и печатается оператору: выбор, сделанный
    молча, оператор считает поломкой — ровно как молчащую запись.
    """
    if default not in CHANNELS:
        raise ValueError(f"неизвестный канал хода {default!r}; набор: {CHANNELS}")
    if default == "none":
        return "none", "ход не печатается: так велено ключом"
    if kind in FILE_KINDS:
        return "file", (
            f"вид записи «{kind}»: на ней мерится отсутствие изменений экрана, и "
            "мигающая строка в терминале попала бы в кадр как изменение")
    return default, ("ход идёт строкой на месте" if kind is None else
                     f"вид записи «{kind}»: вклад терминала в изменения кадра "
                     "пренебрежим рядом с действиями оператора")


def _hms(seconds: float) -> str:
    """Секунды как м:сс. Часы появляются только когда они есть."""
    s = max(0, int(round(seconds)))
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


@dataclass(frozen=True, slots=True)
class Sample:
    """Одно наблюдение хода записи. Всё, что нужно строке, и ничего лишнего."""

    elapsed_s: float
    total_s: float
    written: int
    unchanged: int
    total_turns: int
    disk_bytes: int

    @property
    def turns(self) -> int:
        return self.written + self.unchanged

    @property
    def share(self) -> float:
        """Доля пройденного. По оборотам, а не по времени.

        Обороты — то, чем задана длина записи (`--seconds` переводится в кадры на
        старте), и по ним доля не соврёт на машине, отдающей 28 кадров вместо тридцати.
        Время при этом печатается тоже: оператор ждёт секунды, а не обороты.
        """
        return 0.0 if self.total_turns <= 0 else min(1.0, self.turns / self.total_turns)

    @property
    def gib_per_hour(self) -> float | None:
        """Расход в ГиБ/ч по факту, а не по константе. `None` — мерить пока нечем."""
        if self.elapsed_s <= 0 or self.disk_bytes <= 0:
            return None
        return self.disk_bytes / self.elapsed_s * 3600.0 / (1 << 30)

    @property
    def left_s(self) -> float | None:
        """Оценка остатка по достигнутому темпу оборотов. `None` — темпа ещё нет.

        По достигнутому, а не по заявленной частоте: заявленная — цель цикла, и на
        машине, отдающей 20 кадров вместо тридцати, оценка по ней врала бы в полтора
        раза именно там, где оператор решает, ждать ли ему дальше.
        """
        if self.turns <= 0 or self.elapsed_s <= 0 or self.total_turns <= 0:
            return None
        per_turn = self.elapsed_s / self.turns
        return max(0.0, (self.total_turns - self.turns) * per_turn)


def render_line(s: Sample) -> str:
    """Строка хода. Чистая функция: то же наблюдение — та же строка.

    Порядок величин — от того, что оператор спрашивает первым: сколько ещё ждать. Дальше
    то, что он спросит, если ответ его удивит: сколько кадров, сколько статики, сколько
    это стоит на диске.
    """
    gib = s.gib_per_hour
    left = s.left_s
    return (
        f"{_hms(s.elapsed_s)} из {_hms(s.total_s)} · {s.share * 100:4.0f}% · "
        f"кадров {s.written}, без изменений {s.unchanged} · "
        f"{s.disk_bytes / (1 << 20):.1f} МиБ"
        + (f", {gib:.1f} ГиБ/ч" if gib is not None else ", расход пока не измерен")
        + (f" · осталось ~{_hms(left)}" if left is not None else " · остаток неизвестен")
    )


def render_final(s: Sample, *, path: Path, interrupted: bool,
                 audio_note: str = "") -> str:
    """Итог. Те же числа, что в ходе, плюс путь — и отметка, если прервано.

    Прерванная запись говорит **на чём именно** прервана: «прервано на 4 с из 60»
    отличает частичную запись от полной, и это то различие, за которым оператор пришёл.
    """
    head = (f"прервано на {_hms(s.elapsed_s)} из {_hms(s.total_s)}"
            if interrupted else f"записано за {_hms(s.elapsed_s)}")
    gib = s.gib_per_hour
    return (f"{head}: {s.written} изменившихся кадров и {s.unchanged} отметок "
            f"«без изменений», {s.disk_bytes / (1 << 20):.1f} МиБ"
            + (f" ({gib:.1f} ГиБ/ч)" if gib is not None else "")
            + f" → {path}"
            + (f" ({audio_note})" if audio_note else ""))


class Progress:
    """Печать хода записи. Не чаще, чем велит профиль.

    Устройство разделено нарочно: `Sample` — наблюдение, `render_line` — строка,
    `Progress` — канал и частота. Так ход проверяется офлайн, без дисплея и без
    терминала, а на машине оператора остаётся ровно то же поведение.
    """

    def __init__(self, *, total_turns: int, fps: float, path: Path,
                 channel: str = "line", why: str = "",
                 min_interval_s: float = 1.0,
                 out: TextIO | None = None,
                 now: Callable[[], float] = time.monotonic) -> None:
        if channel not in CHANNELS:
            raise ValueError(f"неизвестный канал хода {channel!r}; набор: {CHANNELS}")
        self.total_turns = int(total_turns)
        self.fps = float(fps)
        self.path = Path(path)
        self.channel = channel
        self.why = why
        self.min_interval_s = float(min_interval_s)
        self._now = now
        self._out = out
        self.started = now()
        # Первое обновление печатается **всегда**, а не через секунду ожидания: молчание
        # в начале — ровно то, что оператор дважды принял за зависание. Отсюда не 0.0, а
        # минус бесконечность: «с прошлой печати прошло сколько угодно».
        self._last = float("-inf")
        self.updates = 0            # сколько раз строка обновлена
        self.suppressed = 0         # сколько обновлений подавлено частотой
        # Файл хода лежит **рядом** с сессией, а не внутри: каталог сессии обязан быть
        # пустым на старте записи, и посторонний файл в нём сделал бы повторный запуск
        # невозможным по той же причине, из-за которой оператор уже напоролся.
        self.log_path = self.path.parent / f"{self.path.name}-progress.log"
        self._log: TextIO | None = None

    @classmethod
    def from_profile(cls, profile: Any, *, total_turns: int, path: Path,
                     kind: str | None = None, channel: str | None = None,
                     out: TextIO | None = None,
                     now: Callable[[], float] = time.monotonic) -> Progress:
        """Собрать ход по профилю записи. Обе настройки читаются **здесь**.

        Читаются здесь, а не в разборе аргументов: `cli` — путь отчёта, и настройка,
        прочитанная только там, детектором объявляется «только отчёт», то есть ручкой без
        читателя на живом пути (инвариант 30). Это не бухгалтерия ради детектора:
        настройка хода записи и должна читаться там, где ход делается.
        """
        p = profile.parameters
        default = str(channel or p["progress_channel"])
        chosen, why = channel_for(kind, default=default)
        return cls(total_turns=total_turns, fps=float(p["capture_fps"]), path=path,
                   channel=chosen, why=why,
                   min_interval_s=float(p["progress_min_interval_s"]),
                   out=out, now=now)

    # --- канал ------------------------------------------------------------

    @property
    def total_s(self) -> float:
        return 0.0 if self.fps <= 0 else self.total_turns / self.fps

    def open(self) -> str:
        """Открыть канал и вернуть строку, которую надо сказать оператору сразу.

        Строка есть всегда, кроме `none`: молчание в начале — это ровно то, что оператор
        принял за зависание.
        """
        if self.channel == "none":
            return ""
        if self.channel == "file":
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("a", encoding="utf-8")
            return (f"ход записи пишется в {self.log_path}, а не на экран — {self.why}\n"
                    f"смотреть на ходу: tail -f {self.log_path}")
        return f"идёт запись {_hms(self.total_s)} — {self.why}"

    def _write(self, text: str, *, final: bool = False) -> None:
        if self.channel == "none":
            return
        if self.channel == "file" and self._log is not None:
            self._log.write(text + "\n")
            self._log.flush()
            return
        stream = self._out
        if stream is None:
            import sys

            stream = sys.stderr
        # Строка обновляется **на месте**: возврат каретки без перевода. Перевод строки
        # только у итога — иначе за минуту записи набегает шестьдесят строк, и то, что
        # оператор искал (идёт ли запись), тонет в том, что он уже видел.
        stream.write(("\r" + text.ljust(96)) if not final else ("\r" + text + "\n"))
        stream.flush()

    # --- ход --------------------------------------------------------------

    def sample(self, *, written: int, unchanged: int, disk_bytes: int) -> Sample:
        return Sample(elapsed_s=self._now() - self.started, total_s=self.total_s,
                      written=written, unchanged=unchanged,
                      total_turns=self.total_turns, disk_bytes=int(disk_bytes))

    def update(self, *, written: int, unchanged: int,
               disk_bytes: int | Callable[[], int], force: bool = False) -> str | None:
        """Обновить строку хода, если пора. Возвращает напечатанное либо `None`.

        Возвращает строку, а не печатает молча: так проверка видит, что именно оператор
        прочитал, и не полагается на перехват потока.

        `disk_bytes` принимает и функцию: подсчёт байтов на диске — обход каталога, и на
        частоте оборотов (30 Гц) он стоил бы дороже самой записи. Функция вызывается
        только тогда, когда строка действительно печатается, то есть раз в секунду.
        """
        elapsed = self._now() - self.started
        if not force and elapsed - self._last < self.min_interval_s:
            self.suppressed += 1
            return None
        self._last = elapsed
        got = disk_bytes() if callable(disk_bytes) else disk_bytes
        s = self.sample(written=written, unchanged=unchanged, disk_bytes=got)
        line = render_line(s)
        self.updates += 1
        self._write(line)
        return line

    def finish(self, *, written: int, unchanged: int,
               disk_bytes: int | Callable[[], int],
               interrupted: bool = False, audio_note: str = "") -> str:
        got = disk_bytes() if callable(disk_bytes) else disk_bytes
        s = self.sample(written=written, unchanged=unchanged, disk_bytes=got)
        line = render_final(s, path=self.path, interrupted=interrupted,
                            audio_note=audio_note)
        if self.channel == "file" and self._log is not None:
            self._log.write(line + "\n")
            self._log.flush()
        else:
            self._write(line, final=True)
        self.close()
        return line

    def close(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def stats(self) -> dict[str, Any]:
        """Чем этот ход был. Для журнала и для проверок."""
        return {"channel": self.channel, "why": self.why,
                "updates": self.updates, "suppressed": self.suppressed,
                "min_interval_s": self.min_interval_s,
                "log": str(self.log_path) if self.channel == "file" else None}


def dir_bytes(root: Path) -> int:
    """Сколько сейчас занимает каталог. Настоящие байты, а не оценка по константе."""
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
