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


#: Куда складывается последний замер расхода. Рядом с записями оператора, а не в
#: коде: константа в коде — это то, с чего начался разбор в TASK-08 и TASK-09.
MEASURED_FILE = "storage-measured.json"


@dataclass(frozen=True, slots=True)
class Measured:
    """Измеренный расход. С отметкой времени, содержимого и механизма.

    `doctor` берёт **это**, а не константу, если файл есть. Пока замера не было, он
    так и говорит — «оценка, замера не было», — а не выдаёт расчёт за факт: в
    TASK-08 расчёт по выдуманной константе разошёлся с действительностью в 25 раз, в
    TASK-09 — в 14, и оба раза оператор узнавал об этом позже, чем мог бы.
    """

    bytes_per_frame: float
    frame_w: int
    frame_h: int
    fps: float
    compression: float
    mechanism: str
    entries: int
    at_unix: float = 0.0

    @property
    def gib_per_hour(self) -> float:
        return self.bytes_per_frame * self.fps * 3600.0 / (1 << 30)

    def as_dict(self) -> dict[str, Any]:
        return {"bytes_per_frame": round(self.bytes_per_frame, 1),
                "frame_w": self.frame_w, "frame_h": self.frame_h,
                "fps": self.fps, "compression": round(self.compression, 1),
                "mechanism": self.mechanism, "entries": self.entries,
                "at_unix": self.at_unix,
                "gib_per_hour": round(self.gib_per_hour, 3)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Measured":
        return cls(float(d["bytes_per_frame"]), int(d["frame_w"]), int(d["frame_h"]),
                   float(d["fps"]), float(d["compression"]), str(d["mechanism"]),
                   int(d["entries"]), float(d.get("at_unix", 0.0)))

    def save(self, root: Path) -> Path:
        import json

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / MEASURED_FILE
        path.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2)
                        + "\n", encoding="utf-8")
        return path

    @staticmethod
    def load(root: Path) -> "Measured | None":
        import json

        path = Path(root) / MEASURED_FILE
        if not path.exists():
            return None
        try:
            return Measured.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, KeyError):
            # Испорченный файл — это отсутствие замера, а не замер. Молча брать из
            # него числа нельзя, а падать не за что: доктор скажет «замера не было».
            return None


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
    measured: "Measured | None" = None

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
                "measured": self.measured.as_dict() if self.measured else None,
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
    # Звук пишется **всю сессию**, а не берётся разово. Первая редакция читала один
    # блок для проверки и печатала «звук пишется: 960 отсчётов, 2 канала» на сессии
    # длиной 6.7 с, где блоков должно быть около 335. Проверка подтверждала факт и
    # молчала про непрерывность — то есть проверяла не то, что важно.
    audio: list[np.ndarray] = field(default_factory=list)
    audio_rate: int = 0
    audio_overflows: int = 0
    audio_error: str = ""
    first_frame_ns: int = 0
    first_audio_ns: int = 0

    @property
    def turns_per_s(self) -> float:
        return self.turns / max(1e-6, self.elapsed)

    @property
    def audio_samples(self) -> int:
        return sum(int(b.shape[0]) for b in self.audio)

    @property
    def audio_seconds(self) -> float:
        return self.audio_samples / max(1, self.audio_rate)

    @property
    def audio_coverage(self) -> float:
        """Доля длительности сессии, покрытая звуком. Это и есть непрерывность."""
        return self.audio_seconds / max(1e-6, self.elapsed)

    @property
    def sync_offset_ms(self) -> float:
        """Расхождение начала звука с началом кадров. Профиль допускает 50 мс."""
        if not (self.first_frame_ns and self.first_audio_ns):
            return 0.0
        return abs(self.first_audio_ns - self.first_frame_ns) / 1e6

    @property
    def changed_per_s(self) -> float:
        return len(self.frames) / max(1e-6, self.elapsed)


def _drain_audio(audio_source: Any, got: Capture) -> None:
    """Выгрести всё, что накопилось в звуковом буфере. По одному блоку недостаточно.

    Блоков в секунду больше, чем кадров (50 против 30 при блоке 20 мс), поэтому цикл,
    читающий по одному блоку на кадр, отстаёт и в конце переполняет буфер. Отставание
    выглядело бы как «звука меньше, чем сессии» — то есть как дефект захвата там, где
    дефект в темпе чтения.
    """
    if audio_source is None:
        return
    block_len = getattr(audio_source, "block", 0) or 1
    for _ in range(8):                       # предохранитель от бесконечного цикла
        if getattr(audio_source, "pending", block_len) < block_len:
            break
        try:
            block = audio_source.read()
        except Exception as e:
            if type(e).__name__ == "AudioOverflow":
                got.audio_overflows += 1
                block = getattr(e, "block", None)
            else:
                got.audio_error = str(e)
                return
        if block is None:
            return
        got.audio.append(block.samples)
        got.audio_rate = int(block.rate)
        if not got.first_audio_ns:
            got.first_audio_ns = int(block.monotonic_ns)


def _capture(source: Any, *, seconds: float, want: int,
             audio_source: Any = None) -> Capture:
    from .capture.base import UNCHANGED

    got = Capture(mechanism=getattr(source, "name", "?"))
    started = time.monotonic()
    deadline = started + seconds
    while len(got.frames) + got.unchanged < want and time.monotonic() < deadline:
        f = source.read()
        got.turns += 1
        _drain_audio(audio_source, got)
        if f is UNCHANGED:
            got.unchanged += 1
            continue
        if f is None:
            got.ended = True
            break
        if not got.first_frame_ns:
            got.first_frame_ns = int(f.monotonic_ns)
        got.frames.append(f.image)
    got.elapsed = max(1e-6, time.monotonic() - started)
    _drain_audio(audio_source, got)
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

    # Звук открывается **до** захвата: он пишется всю сессию, а не берётся разово.
    if with_audio and audio_source is None:
        from .capture.screen import LoopbackAudio

        audio_source = LoopbackAudio(
            rate=int(profile.parameters["audio_rate"]),
            channels=int(profile.structural["audio_channels"]),
            block_ms=float(profile.parameters["audio_block_ms"]))
    audio_started = False
    audio_open_error = ""
    if with_audio and audio_source is not None:
        try:
            audio_source.start()
            audio_started = True
        except Exception as e:                  # sounddevice бросает своё
            audio_open_error = str(e)

    try:
        got = _capture(source, seconds=seconds, want=want,
                       audio_source=audio_source if audio_started else None)
    finally:
        source.stop()
        if audio_started:
            try:
                audio_source.stop()
            except Exception:
                pass

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
    #
    # Мерится **длительность**, а не факт. Прежняя проверка печатала «звук пишется:
    # 960 отсчётов, 2 канала» на сессии длиной 6.7 с — то есть один блок из
    # трёхсот тридцати пяти, — и ничего не говорила о непрерывности. Это был дефект
    # проверки, а не захвата: блок читался ровно один раз, для галочки.
    audio: np.ndarray | None = None
    if with_audio:
        if not audio_started:
            res.add("звук пишется", False, "петлевой вход не открылся",
                    where=f"{audio_open_error or 'причина не названа источником'}\n"
                          f"             звук записи не блокирует: остальные проверки "
                          f"ниже выполнены. Вход настроить поможет harness doctor")
            for name in ("покрытие звука", "каналы различаются",
                         "звук сошёлся с кадрами по времени"):
                res.skip(name, "звука нет")
        elif not got.audio:
            res.add("звук пишется", False,
                    f"вход открылся и не отдал ни одного блока"
                    + (f": {got.audio_error}" if got.audio_error else ""),
                    where="скорее всего выбран не тот источник: нужен monitor или "
                          "loopback, микрофон пишет комнату, а не то, что слышит "
                          "агент. Список входов покажет harness doctor")
            for name in ("покрытие звука", "каналы различаются",
                         "звук сошёлся с кадрами по времени"):
                res.skip(name, "блоков звука нет")
        else:
            audio = np.concatenate(got.audio, axis=0)
            res.add("звук пишется", True,
                    f"{len(got.audio)} блоков, {got.audio_samples} отсчётов, "
                    f"{got.audio_seconds:.1f} с при {got.audio_rate} Гц")

            # Покрытие: доля длительности сессии, у которой есть звук. Порог 0.9 —
            # объявлен здесь: ниже этого запись уже не «со звуком», а «с фрагментами».
            cover = got.audio_coverage
            floor_cover = 0.9
            res.add("покрытие звука", cover >= floor_cover,
                    f"{cover:.0%} длительности сессии"
                    + (f", переполнений буфера {got.audio_overflows}"
                       if got.audio_overflows else ""),
                    where="" if cover >= floor_cover else
                    (f"звук покрывает {cover:.0%} сессии при пороге "
                     f"{floor_cover:.0%}: {got.audio_seconds:.1f} с звука на "
                     f"{got.elapsed:.1f} с записи. Либо вход отдаёт с перебоями, "
                     f"либо цикл не успевает выгребать буфер — во втором случае "
                     f"переполнения будут видны в строке выше"))

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
                        "в свойствах устройства записи, что выбран monitor/loopback "
                        "и что у него два канала")

            # Сверка по времени. `audio_sync_tolerance_ms` был объявлен в профиле и
            # не проверялся ни одной строкой — то есть обещал точность, которую
            # никто не подтверждал.
            tol = float(profile.parameters["audio_sync_tolerance_ms"])
            off = got.sync_offset_ms
            res.add("звук сошёлся с кадрами по времени", off <= tol,
                    f"расхождение начала {off:.0f} мс при допуске {tol:g} мс",
                    where="" if off <= tol else
                    (f"звук начался на {off:.0f} мс позже или раньше кадров при "
                     f"допуске {tol:g} мс. Пеленг считается по разнице каналов, а "
                     f"привязка события к кадру — по времени, и при таком сдвиге "
                     f"звук относится к другому кадру"))

    # --- 6. Журнал. Нужны кадры ----------------------------------------------
    root = Path(root)
    # Профиль записи строится **по кадру**, а не по пожеланию: экран отдаёт монитор
    # целиком, и заявить 320×180 над кадрами 1080p значит записать ложь о себе.
    written_profile = profile.for_frame(got.frames[0])
    try:
        with Recorder(root, profile=written_profile, source=got.mechanism,
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
    h, w = got.frames[0].shape[:2]
    raw = h * w * (3 if got.frames[0].ndim == 3 else 1)
    # Степень сжатия печатается рядом: без неё «49.3 КиБ на запись» нечем поверить, и
    # первый же разбор поделил это на размер кадра из профиля — а профиль врал.
    res.add("расход места измерен", True,
            f"{used / (1 << 20):.1f} МиБ за {got.elapsed:.1f} с "
            f"({per_frame / 1024:.1f} КиБ на запись при кадре {w}×{h}, "
            f"сырой {raw / 1024:.0f} КиБ — сжатие {raw / max(1.0, per_frame):.0f}×) → "
            f"{per_hour / (1 << 30):.1f} ГиБ/ч при {fps:g} кадр/с")
    res.measured = Measured(bytes_per_frame=per_frame, frame_w=w, frame_h=h,
                            fps=fps, compression=raw / max(1.0, per_frame),
                            mechanism=got.mechanism, entries=len(entries),
                            at_unix=time.time())
    # Замер переживает запуск: следующий `harness doctor` возьмёт его вместо
    # константы и скажет, откуда число.
    from .machine import state_dir

    res.measured.save(state_dir())
    return res
