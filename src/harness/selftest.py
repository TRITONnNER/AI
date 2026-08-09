"""`harness selftest`: десять секунд захвата и проверка, что получилось не пустышка.

`TASK-07`, часть 3. `doctor` отвечает на вопрос «можно ли записывать», а этот —
на вопрос «а записалось ли». Это разные вопросы: разрешения могут быть выданы,
пакеты стоять, место быть, — и при этом кадры приходить одинаковыми, звук моно, а
воспроизведение падать на первом же обращении.

## Что именно проверяется и почему каждое

1. **Кадры приходят, частота не ниже заявленной.** Иначе запись на три минуты
   окажется записью на двадцать секунд, растянутой по метке времени.
2. **Кадры не чёрные и не одинаковые.** Чёрные — это macOS без разрешения и Wayland
   через XWayland. Одинаковые — это backend, отдающий один и тот же буфер: формально
   кадры есть, фактически записан один. Проверяется по содержимому, а не по факту
   возврата.
3. **Записи журнала создаются, поля заполнены, `actor_layer = human`.** В живой
   записи действует оператор, и это обязано быть видно: без слоя-инициатора живой
   корпус неотличим по атрибуции от синтетического, где кадры не начаты никем.
4. **Звук пишется, каналы различаются.** Моно не годится вовсе: пеленг считается из
   разницы каналов, и два одинаковых канала — это моно, записанное как стерео.
5. **Воспроизведение из журнала работает.** Запись, которую нельзя прочитать обратно,
   не запись. Проверяется тем же кодом, которым потом читает замер.

## Почему «одинаковые кадры» — отказ, а не предупреждение

На неподвижном рабочем столе кадры могут совпадать почти полностью — но не побитово:
курсор, часы, антиалиасинг. Побитовое совпадение **всех** кадров за десять секунд на
настоящем экране означает, что кадры не обновляются. Поэтому сравнение точное и
только между соседними: «ни один кадр не отличается от предыдущего» — отказ, а
«некоторые совпадают» — норма.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class Step:
    """Один шаг проверки. `where` — не «что-то сломалось», а где именно."""

    name: str
    ok: bool
    detail: str
    where: str = ""             # заполняется только при провале

    def __post_init__(self) -> None:
        if not self.ok and not self.where:
            raise ValueError(
                f"шаг {self.name!r} провалился и не сказал, где именно. Задача "
                "команды — назвать место, иначе оператор идёт читать код")

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail,
                "where": self.where}


@dataclass(slots=True)
class Result:
    steps: list[Step] = field(default_factory=list)
    session: Path | None = None

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    @property
    def failed(self) -> list[Step]:
        return [s for s in self.steps if not s.ok]

    def add(self, name: str, ok: bool, detail: str, where: str = "") -> Step:
        step = Step(name, ok, detail, where)
        self.steps.append(step)
        return step

    def render_text(self) -> str:
        rows = [f"  [{'да ' if s.ok else 'НЕТ'}] {s.name}: {s.detail}"
                for s in self.steps]
        if self.ok:
            rows += ["", f"Всё прошло. Запись лежит в {self.session}.",
                     "Дальше: harness record --plan"]
        else:
            first = self.failed[0]
            rows += ["", f"Сломалось: {first.name}.",
                     f"Где именно: {first.where}"]
        return "\n".join(rows)

    def as_dict(self) -> dict[str, Any]:
        return {"steps": [s.as_dict() for s in self.steps], "ok": self.ok,
                "session": str(self.session) if self.session else None}


def _spread(frames: list[np.ndarray]) -> float:
    return max(float(f.std()) for f in frames) if frames else 0.0


def _all_identical(frames: Iterable[np.ndarray]) -> bool:
    """Ни один кадр не отличается от предыдущего. Именно это — отказ."""
    prev = None
    seen_change = False
    for f in frames:
        if prev is not None and not np.array_equal(prev, f):
            seen_change = True
            break
        prev = f
    return not seen_change


def run(root: Path, *, seconds: float = 10.0, with_audio: bool = True,
        source: Any = None, audio_source: Any = None) -> Result:
    """Захватить, записать, прочитать обратно. `source` подменяется в тестах.

    Порядок шагов — порядок зависимостей: пока кадры не оказались настоящими, нет
    смысла проверять журнал, а пока журнал не записан, нечего воспроизводить. Первый
    провал прекращает проверку: дальнейшие «НЕТ» были бы следствиями, и оператор
    чинил бы последствие вместо причины.
    """
    from .capture.base import BackendUnavailable
    from .core.journal import Actor, ActorLayer
    from .core.profile import MILESTONE_0
    from .session import Recorder, Session as ReadSession

    res = Result()
    profile = MILESTONE_0
    fps = float(profile.parameters["capture_fps"])
    want = max(1, int(fps * seconds))

    # --- 1. Захват -----------------------------------------------------------
    if source is None:
        from .capture.screen import ScreenCapture
        source = ScreenCapture(gray=profile.structural["frame_format"] == "gray8")
    try:
        source.start()
    except BackendUnavailable as e:
        res.add("захват запускается", False, "backend отказался",
                where=f"{getattr(source, 'name', 'источник')}: {e}\n"
                      f"             сначала: harness doctor")
        return res

    frames: list[np.ndarray] = []
    started = time.monotonic()
    try:
        while len(frames) < want and time.monotonic() - started < seconds * 3:
            f = source.read()
            if f is None:
                break
            frames.append(f.image)
    finally:
        source.stop()
    elapsed = max(1e-6, time.monotonic() - started)
    got_fps = len(frames) / elapsed

    if not frames:
        res.add("кадры приходят", False, "ни одного кадра",
                where="источник запустился и не отдал ни одного кадра — это не "
                      "нехватка прав, а поломка backend'а")
        return res
    res.add("кадры приходят", True, f"{len(frames)} кадров за {elapsed:.1f} с")

    # Частота: допускается недобор до 20 %, ниже — отказ. Порог объявлен здесь, а
    # не подобран: 20 % — это разница между «система занята» и «кадры теряются».
    floor = fps * 0.8
    ok_rate = got_fps >= floor
    res.add("частота не ниже заявленной", ok_rate,
            f"{got_fps:.1f} кадр/с при заявленных {fps:g}",
            where="" if ok_rate else
            (f"получено {got_fps:.1f} кадр/с, порог {floor:.1f}. Захват не "
             f"успевает: закройте тяжёлые окна или уменьшите capture_fps в профиле"))
    if not ok_rate:
        return res

    spread = _spread(frames)
    peak = max(int(f.max()) for f in frames)
    if peak == 0 or spread < 1.0:
        res.add("кадры не чёрные", False,
                f"разброс яркости {spread:.2f}, максимум {peak}",
                where="кадры пустые. На macOS это отсутствие разрешения «Запись "
                      "экрана», под Wayland — захват через XWayland. "
                      "Точную причину скажет harness doctor")
        return res
    res.add("кадры не чёрные", True, f"разброс яркости {spread:.1f}")

    if _all_identical(frames):
        res.add("кадры не одинаковые", False,
                f"все {len(frames)} кадров совпадают побитово",
                where="backend отдаёт один и тот же буфер: формально кадры есть, "
                      "фактически записан один. Подвигайте мышью и повторите; если "
                      "повторится — backend захвата неисправен")
        return res
    res.add("кадры не одинаковые", True, "соседние кадры различаются")

    # --- 2. Звук -------------------------------------------------------------
    audio: np.ndarray | None = None
    if with_audio:
        if audio_source is None:
            from .capture.screen import LoopbackAudio
            audio_source = LoopbackAudio(
                rate=int(profile.parameters["audio_rate"]),
                channels=int(profile.structural["audio_channels"]),
                block_ms=float(profile.parameters["audio_block_ms"]))
        try:
            audio_source.start()
            block = audio_source.read()
            audio_source.stop()
        except Exception as e:                  # sounddevice бросает своё
            res.add("звук пишется", False, "петлевой вход не открылся",
                    where=f"{e}\n             звук записи не блокирует: повторите "
                          f"с --no-audio, а вход настройте по harness doctor")
            return res
        audio = None if block is None else block.samples
        if audio is None or audio.shape[0] == 0:
            res.add("звук пишется", False, "блок пустой",
                    where="вход открылся и отдал ноль отсчётов: скорее всего "
                          "выбран не тот источник (нужен monitor, не микрофон)")
            return res
        res.add("звук пишется", True, f"{audio.shape[0]} отсчётов, "
                                      f"{audio.shape[1]} канала")
        if audio.shape[1] < 2:
            res.add("каналы различаются", False, f"каналов {audio.shape[1]}",
                    where="моно записано как стерео. Пеленг считается из разницы "
                          "каналов, и на моно он не считается вовсе")
            return res
        differ = not np.array_equal(audio[:, 0], audio[:, 1])
        res.add("каналы различаются", differ,
                "каналы разные" if differ else "каналы совпадают побитово",
                where="" if differ else
                "оба канала одинаковы: это моно, продублированное в два канала. "
                "Обычно так ведёт себя микрофон, выбранный вместо петлевого входа")
        if not differ:
            return res

    # --- 3. Журнал -----------------------------------------------------------
    root = Path(root)
    try:
        with Recorder(root, profile=profile, source=getattr(source, "name", "?"),
                      synthetic=False, note="harness selftest") as rec:
            for i, image in enumerate(frames):
                rec.record_frame(image, t_world=i,
                                 audio=audio if i == 0 else None,
                                 actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN)
    except Exception as e:
        res.add("записи журнала создаются", False, "запись не удалась",
                where=f"{type(e).__name__}: {e}")
        return res
    res.session = root

    with ReadSession(root) as sess:
        entries = list(sess.journal.frames())
        if len(entries) != len(frames):
            res.add("записи журнала создаются", False,
                    f"кадров {len(frames)}, записей {len(entries)}",
                    where="часть кадров не доехала до журнала: смотрите записи "
                          "gap в журнале — там причина отказа")
            return res
        res.add("записи журнала создаются", True, f"{len(entries)} записей")

        bad = [x for x in entries
               if str(x.actor_layer) != "human" or str(x.actor) != "human"]
        res.add("actor_layer = human", not bad,
                f"все {len(entries)} записей от человека" if not bad
                else f"{len(bad)} записей не от человека",
                where="" if not bad else
                "в живой записи действует оператор, и слой-инициатор обязан быть "
                "human: без него живой корпус неотличим по атрибуции от "
                "синтетического")
        if bad:
            return res

        empty = [x for x in entries if x.frame is None or x.stamp is None]
        res.add("поля записей заполнены", not empty,
                "ссылка на кадр и три часа на месте" if not empty
                else f"{len(empty)} записей без ссылки или без штампа",
                where="" if not empty else
                "запись без ссылки на кадр или без штампа времени — это запись, "
                "из которой ничего не восстановить")
        if empty:
            return res

        # --- 4. Воспроизведение ---------------------------------------------
        try:
            first = sess.image(0)
        except Exception as e:
            res.add("воспроизведение работает", False, "чтение упало",
                    where=f"{type(e).__name__}: {e}")
            return res
        if first is None or not np.array_equal(first, frames[0]):
            res.add("воспроизведение работает", False,
                    "прочитанный кадр не совпал с записанным",
                    where="хранилище кадров вернуло не то, что в него положили — "
                          "это порча данных, а не настройка")
            return res
        res.add("воспроизведение работает", True,
                "первый кадр прочитан и совпал с записанным")
    return res
