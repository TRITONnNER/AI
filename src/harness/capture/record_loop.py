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
#: не собираются в месте печати: «дошла до срока» и «дошла до числа оборотов» — разные
#: события, и путать их значит повторять пункт 1 задачи TASK-17.
STOP_REASONS: dict[str, str] = {
    "deadline": "истёк срок в секундах",
    "turns": "набрано заданное число оборотов",
    "interrupted": "прервано оператором",
    "source_ended": "источник кадров кончился",
    "stopped": "остановлено из панели",
}

#: Исходы, при которых запись остановил **человек**. Их два, потому что нажать можно в двух
#: местах — `Ctrl+C` в терминале и «Прервать» в панели, — но означают они одно, и отметка в
#: журнале у них одна: `record_interrupted`. Разделять их значило бы заводить два класса
#: одного события и потом сравнивать записи, сделанные «по-разному».
BY_HUMAN: tuple[str, ...] = ("interrupted", "stopped")

#: **Что считает предел записи.** Оборот — один взгляд на экран, а не один изменившийся
#: кадр. Отметка «без изменений» — тоже оборот: она занимает то же время и даёт ту же
#: запись журнала со своими часами.
#:
#: Решение объявлено здесь потому, что альтернатива хуже, а не потому, что так вышло. Если
#: считать только изменившиеся кадры, то запись неподвижного экрана — та, ради которой всё
#: и делается, — не кончится никогда: экран не меняется, счётчик не растёт. Это ровно
#: обратная поломка к той, что нашлась в TASK-17, и обе от одной причины: предел и срок
#: путали друг с другом.
#:
#: При **срочной** записи (`--seconds`) предел оборотов не применяется вовсе: срок сам
#: решает, когда кончить. Иначе выходило то, что и вышло у оператора: 300 отметок за
#: секунду, и «десять секунд» закончились на первой.
TURNS_MEAN = ("оборот — один взгляд на экран; отметка «без изменений» тоже оборот, "
              "потому что занимает то же время и даёт запись журнала со своими часами")


@dataclass(slots=True)
class Turns:
    """Чем кончился цикл. Числа те же, что оператор видел в ходе записи."""

    written: int = 0            # изменившихся кадров
    unchanged: int = 0          # отметок «без изменений»
    audio_blocks: int = 0
    interrupted: bool = False
    source_ended: bool = False
    stop_reason: str = "turns"
    elapsed_s: float = 0.0
    #: Сколько времени цикл **ждал** между оборотами, чтобы не обогнать целевую частоту.
    #: Без ожидания цикл крутится вхолостую на полной скорости процессора: у оператора
    #: вышло 246 кадр/с при цели 30, то есть восьмикратный перебор. Хуже, чем сожжённый
    #: процессор: интервалы между кадрами становятся случайными, а на них стоят оптический
    #: поток, tau и все временные оценки.
    idle_ns: int = 0
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
                "idle_ms": self.idle_ns / n / 1e6,
                "idle_share": (self.idle_ns / (self.elapsed_s * 1e9)
                               if self.elapsed_s > 0 else 0.0),
                "achieved_fps": (self.turns / self.elapsed_s
                                 if self.elapsed_s > 0 else 0.0),
                # Доля оборотов, на которых экран изменился. Без неё две записи с одной
                # машины несравнимы: 6.5 кадр/с на меняющемся экране и 246 кадр/с на
                # неподвижном — это не разные машины, а разные экраны.
                "change_share": (self.written / self.turns if self.turns else None),
                "expected_turns": self.expected_turns,
                "stop_reason": self.stop_reason}

    def busy_ms(self) -> float:
        """Работа на оборот без ожидания: захват плюс запись. Это и есть то, что
        сравнивается с бюджетом кадра."""
        n = max(1, self.turns)
        return (self.capture_ns + self.record_ns + self.audio_ns) / n / 1e6


def run_turns(rec: Any, *, source: Any, frames: int | None, first: Any,
              actor: Any, actor_layer: Any,
              audio: Any = None,
              progress: Any = None,
              seconds: float | None = None,
              now: Callable[[], float] | None = None,
              sleep: Callable[[float], None] | None = None,
              expected_fps: float = 0.0,
              pace: bool = True,
              should_stop: Callable[[], bool] | None = None,
              disk_bytes: Callable[[], int] | None = None) -> Turns:
    """Записать `seconds` секунд **либо** `frames` оборотов. Ровно одно из двух.

    `first` — первый кадр, взятый до открытия записи: по нему строился профиль, и
    выбрасывать его значило бы потерять кадр, за который уже заплачено.

    **Срок и предел оборотов не совмещаются.** TASK-17 сделал срок в печати, но оставил
    предел в условии выхода, и `--seconds 10` кончилась за секунду с причиной «набрано
    заданное число кадров»: 29 кадров и 271 отметка «без изменений» дали 300 оборотов =
    10 × 30. Поэтому при заданном `seconds` вызывающий передаёт `frames=None`, и выход
    только по времени. Что считает предел, когда он задан, — в `TURNS_MEAN`.

    **Цикл держит целевую частоту** (`expected_fps`), а не крутится на полной скорости
    процессора. Без ожидания у оператора вышло 246 кадр/с при цели 30, и дело не в
    сожжённом процессоре: интервалы между кадрами становятся случайными, а на них стоят
    оптический поток, tau и все временные оценки. Ожидание считается (`idle_ns`) и
    печатается долей — холостая доля, которой не видно, однажды окажется нулём, и это
    надо будет заметить.

    Отметка о прерывании пишется до выхода: сессия закрывается вызывающим (через `with`),
    и к этому моменту в журнале уже стоит, на какой секунде из какой человек остановился.
    """
    import time

    clock = now or time.monotonic
    naptime = sleep or time.sleep
    started = clock()
    deadline = None if seconds is None else started + float(seconds)
    if frames is None and deadline is None:
        raise ValueError("не задано ни срока, ни числа оборотов: запись не кончится")
    # Период кадра по цели. Ноль означает «не ограничивать» — так прогоняются замеры, где
    # нужна полная скорость, и это объявлено ключом `pace`, а не выведено из молчания.
    period = (1.0 / float(expected_fps)) if (pace and expected_fps > 0) else 0.0
    t = Turns()
    try:
        while frames is None or t.turns < frames:
            if deadline is not None and clock() >= deadline:
                t.stop_reason = "deadline"
                break
            if should_stop is not None and should_stop():
                # Остановка снаружи — то же, что `Ctrl+C`: решение человека, а не поломка.
                # Поэтому и отметка та же, и путь тот же; кнопка «Прервать» в панели не
                # заводит второго класса события.
                t.stop_reason = "stopped"
                t.interrupted = True
                _mark_interrupted(rec, t, progress, clock, started)
                break
            if period:
                # Ждём до начала следующего оборота по расписанию от старта, а не «период
                # после предыдущего»: второе накапливает отставание, и за минуту запись
                # уезжает на секунды. Если отстаём — не спим вовсе, но и не догоняем
                # рывком: пропущенный оборот уже пропущен, и врать о нём нечем.
                due = started + t.turns * period
                wait = due - clock()
                if wait > 0:
                    if deadline is not None and due >= deadline:
                        t.stop_reason = "deadline"
                        break
                    naptime(wait)
                    t.idle_ns += int(wait * 1e9)
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
        _mark_interrupted(rec, t, progress, clock, started)
    t.elapsed_s = clock() - started
    # Сколько оборотов машина обязана была успеть к этому моменту по заявленной частоте.
    # Число нужно именно рядом с достигнутым: `capture_fps` — цель цикла, и расхождение
    # цели с действительностью есть свойство машины, которое обязано быть видно.
    t.expected_turns = (int(round(t.elapsed_s * expected_fps)) if expected_fps > 0
                        else (frames or t.turns))
    if progress is not None:
        # Ход этой записи — часть записи: канал, частота, сколько раз печаталось. Иначе
        # вопрос «почему на этой записи изменившихся кадров 3 %» останется без ответа, а
        # ответом может быть именно мигавшая в терминале строка.
        rec.record_intervention("progress", progress.stats())
    return t


def _mark_interrupted(rec: Any, t: Turns, progress: Any,
                      clock: Callable[[], float], started: float) -> None:
    """Отметка «прервано на N из M». Одна на оба способа остановки.

    Пишется **внутри** открытой сессии: снаружи `with` уже закрыл хранилища, и записать
    некуда — ровно поэтому отметки и не было до TASK-16.
    """
    elapsed = clock() - started
    total = 0.0
    if progress is not None:
        total = progress.total_s
    reason = "Ctrl+C" if t.stop_reason == "interrupted" else "кнопка «Прервать» в панели"
    rec.record_interrupted(elapsed_s=elapsed, total_s=total,
                           written=t.written, unchanged=t.unchanged, reason=reason)


def dir_bytes(root: Path) -> int:
    """Сколько занимает каталог записи. Ре-экспорт: у цикла и хода один ответ на это."""
    from ..progress import dir_bytes as _impl

    return _impl(root)
