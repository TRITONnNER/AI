"""`harness selftest`: десять секунд захвата и проверка, что получилось не пустышка.

`TASK-07` часть 3, `TASK-08` части 3 и 3a. `doctor` отвечает на вопрос «можно ли
записывать», а этот — «а записалось ли». Это разные вопросы: разрешения могут быть
выданы, пакеты стоять, место быть, — и при этом кадры приходить одинаковыми, звук
складываться в моно на уровне драйвера, а воспроизведение падать на первом обращении.

## Все проверки, а не до первого отказа

Первый прогон на машине оператора остановился на строке «частота не ниже заявленной»
и не выполнил ничего дальше. Различие кадров, различие каналов звука, запись журнала,
воспроизведение и расход места **от частоты захвата не зависят** — заход стоил
оператору десяти минут ради одной строки.

Правило: выполняются все проверки, между которыми нет зависимости **по данным**.
Останов допустим там, где следующая проверка физически невозможна: воспроизводить
нечего, пока журнал не записан. Всё остальное считается и печатается.

## Частота — это два числа, а не одно

Desktop Duplication не выдаёт неизменённый кадр вовсе. Поэтому «сколько кадров в
секунду» распадается:

- **обороты цикла в секунду** — как часто мы спрашивали экран;
- **изменившиеся кадры в секунду** — как часто он отвечал новым содержимым.

На неподвижном экране второе законно равно нулю. Считать это отказом значило бы
объявить сломанной запись «неподвижность», которая именно неподвижностью и является.

## «Кадры не одинаковые» — проверка не универсальная

На записи неподвижности кадры **обязаны** быть одинаковыми. Поэтому условие зависит
от того, чего мы ждём: `expect_change=False` для статичной сцены. И отдельно: если
источник сам сообщил «без изменений», одинаковость — его честный ответ, а не
поломанный буфер. Поломанный буфер — это когда источник **утверждает**, что отдал
новый кадр, а кадр тот же.
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
    skipped: list[tuple[str, str]] = field(default_factory=list)
    #: Измеренный расход места. Первое такое число, полученное на живом захвате.
    bytes_per_hour: float | None = None

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps) and not self.skipped

    @property
    def failed(self) -> list[Step]:
        return [s for s in self.steps if not s.ok]

    def add(self, name: str, ok: bool, detail: str, where: str = "") -> Step:
        step = Step(name, ok, detail, where)
        self.steps.append(step)
        return step

    def skip(self, name: str, why: str) -> None:
        """Проверка невозможна без предыдущей. Не «прошла» и не «упала»."""
        self.skipped.append((name, why))

    def summary(self) -> str:
        passed = sum(1 for s in self.steps if s.ok)
        rows = [f"Прошло {passed} из {len(self.steps)}"
                + (f", пропущено {len(self.skipped)}" if self.skipped else "") + "."]
        if self.failed:
            first = self.failed[0]
            rows.append(f"Чинить первым: {first.name}.")
            rows.append(f"Где именно: {first.where}")
            others = [s.name for s in self.failed[1:]]
            if others:
                rows.append("Не прошли также: " + ", ".join(others)
                            + ". Возможно, это следствия первого.")
        elif self.skipped:
            rows.append("Пропущено без причины для отказа: "
                        + "; ".join(f"{n} — {w}" for n, w in self.skipped))
        else:
            rows.append(f"Всё прошло. Запись лежит в {self.session}.")
            rows.append("Дальше: harness record --plan")
        return "\n".join(rows)

    def render_text(self) -> str:
        rows = [f"  [{'да ' if s.ok else 'НЕТ'}] {s.name}: {s.detail}"
                for s in self.steps]
        rows += [f"  [ -- ] {n}: {w}" for n, w in self.skipped]
        return "\n".join(rows + ["", self.summary()])

    def as_dict(self) -> dict[str, Any]:
        return {"steps": [s.as_dict() for s in self.steps], "ok": self.ok,
                "skipped": [{"name": n, "why": w} for n, w in self.skipped],
                "bytes_per_hour": self.bytes_per_hour,
                "session": str(self.session) if self.session else None}


def _spread(frames: list[np.ndarray]) -> float:
    return max(float(f.std()) for f in frames) if frames else 0.0


def _all_identical(frames: Iterable[np.ndarray]) -> bool:
    """Ни один кадр не отличается от предыдущего. Именно это — отказ."""
    prev = None
    for f in frames:
        if prev is not None and not np.array_equal(prev, f):
            return False
        prev = f
    return True


def _dir_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


@dataclass(slots=True)
class Capture:
    """Что дал захват. Два счётчика, а не один: обороты и изменившиеся кадры."""

    frames: list[np.ndarray] = field(default_factory=list)
    turns: int = 0               # сколько раз спросили экран
    unchanged: int = 0           # сколько раз экран ответил «то же самое»
    elapsed: float = 0.0
    ended: bool = False          # источник кончился раньше срока
    mechanism: str = "?"

    @property
    def turns_per_s(self) -> float:
        return self.turns / max(1e-6, self.elapsed)

    @property
    def changed_per_s(self) -> float:
        return len(self.frames) / max(1e-6, self.elapsed)


def _capture(source: Any, *, seconds: float, want: int) -> Capture:
    from .capture.base import UNCHANGED

    got = Capture(mechanism=getattr(source, "name", "?"))
    started = time.monotonic()
    deadline = started + seconds
    while len(got.frames) + got.unchanged < want and time.monotonic() < deadline:
        f = source.read()
        got.turns += 1
        if f is UNCHANGED:
            got.unchanged += 1
            continue
        if f is None:
            got.ended = True
            break
        got.frames.append(f.image)
    got.elapsed = max(1e-6, time.monotonic() - started)
    return got


def run(root: Path, *, seconds: float = 10.0, with_audio: bool = True,
        expect_change: bool = True,
        source: Any = None, audio_source: Any = None) -> Result:
    """Захватить, записать, прочитать обратно. `source` подменяется в тестах."""
    from .capture.base import BackendUnavailable
    from .core.journal import Actor, ActorLayer
    from .core.profile import MILESTONE_0
    from .machine import detect
    from .session import Recorder, Session as ReadSession

    res = Result()
    profile = MILESTONE_0
    fps = float(profile.parameters["capture_fps"])
    want = max(1, int(fps * seconds))
    machine = detect()
    alt = ""

    # --- 1. Захват. Без кадров дальше нечего проверять --------------------------
    if source is None:
        from .capture.select import open_screen

        choice = open_screen(machine, gray=profile.structural["frame_format"] == "gray8")
        if choice.source is None:
            res.add("захват запускается", False, "ни один механизм не запустился",
                    where=f"{choice.why_text()}\n             сначала: harness doctor")
            for name in ("кадры приходят", "кадры не чёрные", "журнал", "звук",
                         "воспроизведение", "расход места"):
                res.skip(name, "нечего проверять: захват не запустился")
            return res
        source = choice.source
        alt = choice.caveat
    else:
        try:
            source.start()
        except BackendUnavailable as e:
            res.add("захват запускается", False, "backend отказался",
                    where=f"{getattr(source, 'name', 'источник')}: {e}\n"
                          f"             сначала: harness doctor")
            return res

    try:
        got = _capture(source, seconds=seconds, want=want)
    finally:
        source.stop()

    if not got.frames:
        res.add("кадры приходят", False,
                f"оборотов {got.turns}, кадров 0, «без изменений» {got.unchanged}",
                where="источник запустился и не отдал ни одного кадра. Первый кадр "
                      "приходит всегда, даже на неподвижном экране, поэтому это "
                      "поломка механизма, а не отсутствие изменений")
        for name in ("журнал", "воспроизведение", "расход места"):
            res.skip(name, "нечего записывать: кадров нет")
        return res

    res.add("кадры приходят", True,
            f"{len(got.frames)} изменившихся и {got.unchanged} без изменений "
            f"за {got.elapsed:.1f} с, механизм {got.mechanism}")

    # --- 2. Частота: два числа. Ни одно не блокирует остальные проверки --------
    floor = fps * 0.8
    ok_rate = got.turns_per_s >= floor
    # Причина в коде называется прежде совета оператору. Первая редакция советовала
    # «закрыть тяжёлые окна», тогда как дело было в выбранном механизме: оператор,
    # последовавший совету, получил бы те же двадцать кадров.
    if alt:
        advice = (f"активен {got.mechanism}, и он не держит заявленной частоты: "
                  f"{alt}. Сначала проверьте механизм — harness doctor покажет, "
                  f"какой выбран и почему. Настройки и окна — только если механизм "
                  f"верный")
    else:
        advice = (f"механизм {got.mechanism} — единственный для этой системы, "
                  f"поэтому дело не в выборе. Закройте тяжёлые окна или уменьшите "
                  f"capture_fps в профиле")
    res.add("частота оборотов не ниже заявленной", ok_rate,
            f"{got.turns_per_s:.1f} оборот/с при заявленных {fps:g}",
            where="" if ok_rate else
            f"получено {got.turns_per_s:.1f} оборот/с, порог {floor:.1f}. {advice}")

    # Изменившиеся кадры — отдельное число, и ноль здесь законен на статичной сцене.
    res.add("изменившиеся кадры считаются отдельно", True,
            f"{got.changed_per_s:.1f} изменённых кадр/с"
            + (" (на неподвижном экране это законно мало)" if not expect_change
               else ""))

    # --- 3. Содержимое: не чёрное, не залитое ---------------------------------
    spread = _spread(got.frames)
    peak = max(int(f.max()) for f in got.frames)
    res.add("кадры не чёрные", not (peak == 0 or spread < 1.0),
            f"разброс яркости {spread:.1f}, максимум {peak}",
            where="" if not (peak == 0 or spread < 1.0) else
            "кадры пустые. На macOS это отсутствие разрешения «Запись экрана», "
            "под Wayland — захват через XWayland. Точную причину скажет harness doctor")

    # --- 4. Одинаковость: условие зависит от вида сцены -----------------------
    identical = _all_identical(got.frames)
    if not expect_change:
        res.add("кадры одинаковы, как и ожидалось", True,
                "сцена статична: совпадение кадров здесь — правильный ответ"
                if identical else "часть кадров всё же различается — тоже норма")
    elif got.unchanged and identical:
        res.add("кадры не одинаковые", True,
                f"источник сам сообщил «без изменений» {got.unchanged} раз — это его "
                "честный ответ, а не повторённый буфер")
    else:
        res.add("кадры не одинаковые", not identical,
                "соседние кадры различаются" if not identical
                else f"все {len(got.frames)} кадров совпадают побитово",
                where="" if not identical else
                "источник утверждает, что отдал новый кадр, а кадр тот же: это "
                "повторённый буфер. Подвигайте мышью и повторите; если повторится — "
                "механизм захвата неисправен")

    # --- 5. Звук. От кадров не зависит ----------------------------------------
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
            audio = None if block is None else block.samples
        except Exception as e:                  # sounddevice бросает своё
            res.add("звук пишется", False, "петлевой вход не открылся",
                    where=f"{e}\n             звук записи не блокирует: остальные "
                          f"проверки ниже выполнены. Вход настроить поможет "
                          f"harness doctor")
            res.skip("каналы различаются", "звука нет")
        if audio is not None and audio.shape[0]:
            res.add("звук пишется", True,
                    f"{audio.shape[0]} отсчётов, {audio.shape[1]} канала")
            if audio.shape[1] < 2:
                res.add("каналы различаются", False, f"каналов {audio.shape[1]}",
                        where="моно записано как стерео. Пеленг считается из разницы "
                              "каналов, и на моно он не считается вовсе")
            else:
                differ = not np.array_equal(audio[:, 0], audio[:, 1])
                res.add("каналы различаются", differ,
                        "каналы разные" if differ else "каналы совпадают побитово",
                        where="" if differ else
                        "оба канала одинаковы. Это либо микрофон вместо петлевого "
                        "входа, либо стерео микшер, складывающий каналы в моно на "
                        "уровне драйвера — второе встречается у Realtek. Проверьте "
                        "в свойствах устройства запись, что выбран monitor/loopback, "
                        "и что у него два канала")
        elif audio is not None:
            res.add("звук пишется", False, "блок пустой",
                    where="вход открылся и отдал ноль отсчётов: скорее всего выбран "
                          "не тот источник (нужен monitor, не микрофон)")
            res.skip("каналы различаются", "звука нет")

    # --- 6. Журнал. Нужны кадры ----------------------------------------------
    root = Path(root)
    try:
        with Recorder(root, profile=profile, source=got.mechanism,
                      synthetic=False, note="harness selftest") as rec:
            for i, image in enumerate(got.frames):
                rec.record_frame(image, t_world=i,
                                 audio=audio if i == 0 else None,
                                 actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN)
            for _ in range(got.unchanged):
                # Отметки «без изменений» пишутся тоже: иначе журнал соврёт о том,
                # сколько времени длилась запись.
                rec.record_unchanged(actor=Actor.HUMAN, actor_layer=ActorLayer.HUMAN)
    except Exception as e:
        res.add("записи журнала создаются", False, "запись не удалась",
                where=f"{type(e).__name__}: {e}")
        for name in ("actor_layer = human", "поля записей заполнены",
                     "воспроизведение работает", "расход места"):
            res.skip(name, "журнал не записан")
        return res
    res.session = root

    with ReadSession(root) as sess:
        entries = list(sess.journal.frames())
        expected = len(got.frames) + got.unchanged
        res.add("записи журнала создаются", len(entries) == expected,
                f"{len(entries)} записей при {len(got.frames)} кадрах и "
                f"{got.unchanged} отметках «без изменений»",
                where="" if len(entries) == expected else
                f"ожидалось {expected}, записалось {len(entries)}: смотрите записи "
                f"gap в журнале — там причина отказа")

        bad = [x for x in entries
               if str(x.actor_layer) != "human" or str(x.actor) != "human"]
        res.add("actor_layer = human", not bad,
                f"все {len(entries)} записей от человека" if not bad
                else f"{len(bad)} записей не от человека",
                where="" if not bad else
                "в живой записи действует оператор, и слой-инициатор обязан быть "
                "human: без него живой корпус неотличим по атрибуции от "
                "синтетического")

        empty = [x for x in entries if x.frame is None or x.stamp is None]
        res.add("поля записей заполнены", not empty,
                "ссылка на кадр и три часа на месте" if not empty
                else f"{len(empty)} записей без ссылки или без штампа",
                where="" if not empty else
                "запись без ссылки на кадр или без штампа времени — это запись, из "
                "которой ничего не восстановить")

        # --- 7. Воспроизведение. Нужен журнал --------------------------------
        try:
            first = sess.image(0)
            same = first is not None and np.array_equal(first, got.frames[0])
            res.add("воспроизведение работает", same,
                    "первый кадр прочитан и совпал с записанным" if same
                    else "прочитанный кадр не совпал с записанным",
                    where="" if same else
                    "хранилище кадров вернуло не то, что в него положили — это порча "
                    "данных, а не настройка")
        except Exception as e:
            res.add("воспроизведение работает", False, "чтение упало",
                    where=f"{type(e).__name__}: {e}")

    # --- 8. Расход места: измеренный, а не расчётный --------------------------
    used = _dir_bytes(root)
    per_frame = used / max(1, len(entries))
    per_hour = per_frame * fps * 3600.0
    res.bytes_per_hour = per_hour
    res.add("расход места измерен", True,
            f"{used / (1 << 20):.1f} МиБ за {got.elapsed:.1f} с "
            f"({per_frame / 1024:.1f} КиБ на запись) → "
            f"{per_hour / (1 << 30):.1f} ГиБ/ч при {fps:g} кадр/с")
    return res
