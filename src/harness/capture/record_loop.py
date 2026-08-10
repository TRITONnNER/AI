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


@dataclass(slots=True)
class Turns:
    """Чем кончился цикл. Числа те же, что оператор видел в ходе записи."""

    written: int = 0            # изменившихся кадров
    unchanged: int = 0          # отметок «без изменений»
    audio_blocks: int = 0
    interrupted: bool = False
    source_ended: bool = False

    @property
    def turns(self) -> int:
        return self.written + self.unchanged


def run_turns(rec: Any, *, source: Any, frames: int, first: Any,
              actor: Any, actor_layer: Any,
              audio: Any = None,
              progress: Any = None,
              disk_bytes: Callable[[], int] | None = None) -> Turns:
    """Записать `frames` оборотов. Возвращает числа, а не печатает их.

    `first` — первый кадр, взятый до открытия записи: по нему строился профиль, и
    выбрасывать его значило бы потерять кадр, за который уже заплачено.

    Отметка о прерывании пишется до выхода: сессия закрывается вызывающим (через `with`),
    и к этому моменту в журнале уже стоит, на какой секунде из какой человек остановился.
    """
    t = Turns()
    try:
        while t.turns < frames:
            if progress is not None:
                # Ход обновляется каждый оборот, а печатается не чаще, чем велит
                # профиль. Байты диска — функцией: обход каталога делается только когда
                # строка действительно печатается, а не тридцать раз в секунду.
                progress.update(written=t.written, unchanged=t.unchanged,
                                disk_bytes=disk_bytes or (lambda: 0))
            frame = first if t.turns == 0 else source.read()
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
            rec.record_frame(frame.image, t_world=frame.t_world,
                             audio=block, actor=actor, actor_layer=actor_layer)
            t.written += 1
    except KeyboardInterrupt:
        t.interrupted = True
        elapsed = 0.0
        total = 0.0
        if progress is not None:
            elapsed = progress.sample(written=t.written, unchanged=t.unchanged,
                                      disk_bytes=0).elapsed_s
            total = progress.total_s
        rec.record_interrupted(elapsed_s=elapsed, total_s=total,
                               written=t.written, unchanged=t.unchanged)
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
