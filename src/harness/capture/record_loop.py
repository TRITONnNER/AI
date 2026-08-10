"""Оборот записи: один цикл «спросить экран — записать — показать ход».

`TASK-16`. Цикл вынут из `cli.cmd_record` не ради красоты раскладки, а потому что его
надо **проверять офлайн**. Два его свойства проверяются только тем, что цикл можно
позвать без дисплея:

- ход записи печатается, пока запись идёт (оператор дважды прервал запись, приняв
  молчание за зависание);
- `Ctrl+C` в любой момент оставляет **годную** запись с отметкой «прервано на N из M», а
  не трассировку из середины `zlib.compress`.

Прерывание перехватывается **здесь**, внутри открытой сессии. Снаружи — уже поздно:
`with Recorder(...)` закрывает хранилища на выходе, и запись в журнал после этого
невозможна. Ровно поэтому отметка о прерывании и не появлялась.

`KeyboardInterrupt` наследуется от `BaseException`, а не от `Exception`, и ловится
отдельным `except`: `except Exception` его не видит, и это не оплошность Python, а
защита от кода, который глотает Ctrl+C. Здесь он не глотается — он записывается и
возвращается вызывающему исходом `interrupted`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .base import UNCHANGED
from .screen import AudioOverflow


#: Чем кончилась запись. Названия печатаются оператору, поэтому лежат рядом с исходами, а
#: не собираются в месте печати: «дошла до срока» и «дошла до числа кадров» — разные
#: события, и путать их значит повторять пункт 1 задачи TASK-17.
STOP_REASONS: dict[str, str] = {
    "deadline": "истёк срок в секундах",
    "frames": "набрано заданное число кадров",
    "interrupted": "прервано оператором",
    "source_ended": "источник кадров кончился",
}


@dataclass(slots=True)
class Turns:
    """Чем кончился цикл. Числа те же, что оператор видел в ходе записи."""

    written: int = 0            # изменившихся кадров
    unchanged: int = 0          # отметок «без изменений»
    audio_blocks: int = 0
    interrupted: bool = False
    source_ended: bool = False
    stop_reason: str = "frames"
    elapsed_s: float = 0.0
    #: Разбивка времени по стадиям, наносекунды. `TASK-17`, пункт 2: «узкое место где-то
    #: в записи на диск» — не диагноз, и разбивка снимается **на машине оператора**.
    capture_ns: int = 0          # ожидание кадра от источника
    record_ns: int = 0           # всё, что делает Recorder: кодирование, сжатие, запись
    audio_ns: int = 0
    #: Сколько оборотов ожидалось к сроку по заявленной частоте. Печатается рядом с тем,
    #: сколько получилось: недобор частоты — свойство машины, и он обязан быть виден
    #: числом, а не растянутой записью.
    expected_turns: int = 0

    @property
    def turns(self) -> int:
        return self.written + self.unchanged

    def stages(self) -> dict[str, Any]:
        """Разбивка на оборот, в миллисекундах. Для печати и для журнала.

        На оборот, а не на всю запись: оператор сравнивает с периодом кадра (33 мс при 30
        кадр/с), и сумма по всей записи этого сравнения не даёт.
        """
        n = max(1, self.turns)
        return {"turns": self.turns, "elapsed_s": round(self.elapsed_s, 3),
                "capture_ms": self.capture_ns / n / 1e6,
                "record_ms": self.record_ns / n / 1e6,
                "audio_ms": self.audio_ns / n / 1e6,
                "expected_turns": self.expected_turns,
                "stop_reason": self.stop_reason}


def run_turns(rec: Any, *, source: Any, frames: int, first: Any,
              actor: Any, actor_layer: Any,
              audio: Any = None,
              progress: Any = None,
              seconds: float | None = None,
              now: Callable[[], float] | None = None,
              expected_fps: float = 0.0,
              disk_bytes: Callable[[], int] | None = None) -> Turns:
    """Записать `frames` оборотов **или** `seconds` секунд — что кончится раньше.

    `first` — первый кадр, взятый до открытия записи: по нему строился профиль, и
    выбрасывать его значило бы потерять кадр, за который уже заплачено.

    **`seconds` — срок, а не пожелание.** До TASK-17 длина записи задавалась только числом
    кадров: `--seconds 10` превращалось в 300 кадров, и на машине, отдающей 6.5 кадр/с,
    запись шла 46 секунд. Оператор просил десять секунд и получал минуту, причём молча.
    Теперь срок обрывает запись, а недобор частоты виден числом: `expected_turns` против
    `turns`.

    Отметка о прерывании пишется до выхода: сессия закрывается вызывающим (через `with`),
    и к этому моменту в журнале уже стоит, на какой секунде из какой человек остановился.
    """
    import time

    clock = now or time.monotonic
    started = clock()
    deadline = None if seconds is None else started + float(seconds)
    t = Turns()
    try:
        while t.turns < frames:
            if deadline is not None and clock() >= deadline:
                t.stop_reason = "deadline"
                break
            if progress is not None:
                # Ход обновляется каждый оборот, а печатается не чаще, чем велит
                # профиль. Байты диска — функцией: обход каталога делается только когда
                # строка действительно печатается, а не тридцать раз в секунду.
                progress.update(written=t.written, unchanged=t.unchanged,
                                disk_bytes=disk_bytes or (lambda: 0))
            if t.turns == 0:
                frame = first
            else:
                c0 = clock()
                frame = source.read()
                t.capture_ns += int((clock() - c0) * 1e9)
            if frame is UNCHANGED:
                # Экран не менялся. **Не пропуск.** На записи «неподвижность» это
                # основной исход: считать его потерей значило бы объявить сломанной
                # запись, прошедшую идеально.
                if t.written == 0:
                    # До первого кадра ссылаться не на что — а первый кадр Desktop
                    # Duplication отдаёт всегда. Значит это поломка.
                    rec.record_gap("unchanged_before_first_frame",
                                   {"turns": t.unchanged + 1})
                    t.unchanged += 1
                    continue
                rec.record_unchanged(actor=actor, actor_layer=actor_layer)
                t.unchanged += 1
                continue
            if frame is None:
                rec.record_gap("source_ended", {"after_frames": t.written})
                t.source_ended = True
                t.stop_reason = "source_ended"
                break
            block = None
            if audio is not None:
                try:
                    got = audio.read()
                    block = None if got is None else got.samples
                except AudioOverflow as over:
                    # Переполнение — разрыв синхронизации, и он идёт в журнал.
                    block = over.block.samples
                    rec.record_gap("audio_overflow", {"after_frames": t.written})
                except Exception as e:
                    rec.record_gap("audio_failed", {"after_frames": t.written,
                                                    "reason": str(e)[:120]})
                    audio = None
            if block is not None:
                t.audio_blocks += 1
            r0 = clock()
            rec.record_frame(frame.image, t_world=frame.t_world,
                             audio=block, actor=actor, actor_layer=actor_layer)
            t.record_ns += int((clock() - r0) * 1e9)
            t.written += 1
    except KeyboardInterrupt:
        t.interrupted = True
        t.stop_reason = "interrupted"
        elapsed = 0.0
        total = 0.0
        if progress is not None:
            elapsed = progress.sample(written=t.written, unchanged=t.unchanged,
                                      disk_bytes=0).elapsed_s
            total = progress.total_s
        rec.record_interrupted(elapsed_s=elapsed, total_s=total,
                               written=t.written, unchanged=t.unchanged)
    t.elapsed_s = clock() - started
    # Сколько оборотов машина обязана была успеть к этому моменту по заявленной частоте.
    # Число нужно именно рядом с достигнутым: `capture_fps` — цель цикла, и расхождение
    # цели с действительностью есть свойство машины, которое обязано быть видно.
    t.expected_turns = (int(round(t.elapsed_s * expected_fps)) if expected_fps > 0
                        else frames)
    if progress is not None:
        # Ход этой записи — часть записи: канал, частота, сколько раз печаталось. Иначе
        # вопрос «почему на этой записи изменившихся кадров 3 %» останется без ответа, а
        # ответом может быть именно мигавшая в терминале строка.
        rec.record_intervention("progress", progress.stats())
    return t


def dir_bytes(root: Path) -> int:
    """Сколько занимает каталог записи. Ре-экспорт: у цикла и хода один ответ на это."""
    from ..progress import dir_bytes as _impl

    return _impl(root)
