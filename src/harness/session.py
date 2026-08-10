"""Сессия: журнал, кадры, звук и отладочный поток в одном каталоге.

Раскладка на диске:

    <сессия>/
      session.json                 что за сессия, чем записана, синтетика или нет
      journal/branches/NNN-xxxxxxxx/{branch.json, entries.jsonl}
      frames/{index.jsonl, shard-00000.bin, ...}
      audio/{index.jsonl, shard-00000.bin, ...}
      debug/{truth.jsonl, symbols.jsonl}

Почему `debug/` внутри каталога сессии, если инвариант 12 требует изоляции:
изоляция не про расстояние на диске, а про отсутствие пути из агентского кода.
Держать истину рядом с записью удобно исследователю и не даёт им разъехаться при
копировании. Путь недостижим потому, что `harness.agentside` не импортирует
`harness.debug` — это и проверяется тестом. Расстояние в файловой системе от
такой проверки не защищает: подняться на каталог вверх умеет любой код.

`Recorder` пишет, `Session` читает. Читающий объект не открывает ни один файл на
запись, поэтому воспроизведение не может испортить запись.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np

from .core.action import Action
from .core.blobstore import AudioStore, BlobRef, FrameStore
from .core.clocks import Clocks, Stamp
from .core.journal import (Actor, ActorLayer, Entry, Journal, Kind as EntryKind,
                           StateSnapshot, branch_chain)
from .core.profile import Profile, short

if TYPE_CHECKING:  # только для аннотации: ограничитель необязателен
    from .core.resources import ResourceGovernor

SESSION_META = "session.json"


class SessionError(RuntimeError):
    pass


#: Код записи о прерывании записи оператором. Вид записи — `INTERVENTION`, а не
#: `CAPTURE_GAP`: разрыв означает **потерю** (кадры были, но не записались), а здесь
#: ничего не потеряно — человек решил остановиться, и это его действие, `Actor.HUMAN`.
#: Смешать одно с другим значило бы считать нажатие Ctrl+C поломкой захвата.
INTERRUPTED = "record_interrupted"


def next_free_path(root: Path) -> Path:
    """Свободный путь рядом: `сессия`, `сессия-2`, `сессия-3`, …

    Нужен для отказа, который **называет команду**: «каталог не пуст» без готовой строки
    оператор читает как «делай что-нибудь», и в прошлый раз это кончилось тем, что он
    не сделал ничего.
    """
    root = Path(root)
    if not root.exists() or not any(root.iterdir()):
        return root
    for n in range(2, 1000):
        candidate = root.parent / f"{root.name}-{n}"
        if not candidate.exists() or not any(candidate.iterdir()):
            return candidate
    raise SessionError(f"рядом с {root} тысяча занятых имён — назовите путь сами")


def describe_existing(root: Path) -> str:
    """Что уже лежит в каталоге и что с этим делать. Для внятного отказа.

    Три разных случая, и путать их нельзя: годная запись, прерванная запись и посторонние
    файлы. Прежний отказ говорил одно и то же на все три — «не пуст», — и оператор,
    прервавший запись на середине, получал его как приговор своей записи, хотя запись
    была цела и годна к приёму.
    """
    root = Path(root)
    free = next_free_path(root)
    if not (root / SESSION_META).exists():
        listed = sorted(p.name for p in root.iterdir())[:5]
        return (f"{root} не пуст, и это не сессия: {', '.join(listed)}"
                + (" …" if len(listed) == 5 else "")
                + f"\nЗапись не пишется поверх чужих файлов. Свободный путь рядом:"
                  f"\n  harness record {free}")
    try:
        with Session.open(root) as s:
            entries = len(list(s.journal))
            stop = s.interrupted
    except Exception as e:                     # битая или недописанная запись
        return (f"{root}: сессия есть, но не открывается ({type(e).__name__}: {e}).\n"
                f"Проверить: harness verify {root}\n"
                f"Писать новую: harness record {free}")
    what = ("прервана: " + stop["note"]) if stop else "дописана до конца"
    # Длина в предлагаемой команде — та, которую оператор **уже просил**, а не круглое
    # число из головы: прерванная запись знает, на сколько её заводили, и «допишите то,
    # что не дописалось» — это готовая команда, а не совет подумать. Там, где длины нет,
    # ключа в команде тоже нет: выдуманное число в строке для копирования — тот же обман,
    # что оценка, выданная за замер.
    again = f"harness record {free}"
    if stop and float(stop.get("total_s") or 0) > 0:
        again += f" --seconds {float(stop['total_s']):.0f}"
    return (f"{root}: здесь уже лежит сессия, {entries} записей, {what}.\n"
            "Сессии не дописываются поверх: журнал только дозаписывается внутри своей "
            "линии, а вторая запись — это другая линия.\n"
            f"Эта запись годна к приёму: harness ingest {root} --corpus "
            f"{root.parent / 'corpus'} --kind ВИД\n"
            f"Писать следующую: {again}")


#: Поля профиля, описывающие **свойство самой записи**, и место, где оно сверяется с
#: записью. Правило общее (TASK-10, часть 4): расхождение — отказ, а не предупреждение.
#:
#: Предупреждение здесь не годится по опыту: живая запись оператора уже унесла профиль,
#: заявляющий кадр 320×180 при кадрах 1920×1080, и предупреждение никто бы не прочёл, а
#: посчитанное по такому профилю ошибалось в тридцать шесть раз. Заявление о себе — не
#: пожелание; либо оно верно, либо записи нет.
#:
#: `capture_fps` в список не входит, и это решение, а не пропуск: частота кадров —
#: **цель для цикла, а не свойство записи**. Машина, отдающая 28 кадров вместо
#: тридцати, работает нормально, и отказ на этом остановил бы любую настоящую запись.
#: Достигнутая частота наблюдаема по трём часам самой записи, и `selftest` её считает и
#: печатает; профилю тут верить не нужно.
RECORD_CLAIMS: tuple[tuple[str, str], ...] = (
    ("capture_width", "первый кадр, живая запись: Recorder._check_declared"),
    ("capture_height", "первый кадр, живая запись: Recorder._check_declared"),
    ("frame_format", "первый кадр, любой путь: Recorder._check_declared"),
    ("audio_channels", "первый блок, любой путь: AudioStore.append_block"),
    ("audio_rate", "открытие устройства: LoopbackAudio.start"),
)


@dataclass(frozen=True, slots=True)
class SessionMeta:
    created_unix: float
    profile: dict[str, Any]
    source: str
    synthetic: bool
    platform: str
    harness_version: str
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "created_unix": self.created_unix,
            "profile": self.profile,
            "source": self.source,
            "synthetic": self.synthetic,
            "platform": self.platform,
            "harness_version": self.harness_version,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SessionMeta:
        return cls(float(d["created_unix"]), dict(d["profile"]), str(d["source"]),
                   bool(d["synthetic"]), str(d.get("platform", "?")),
                   str(d.get("harness_version", "?")), d.get("note"))


class Recorder:
    """Запись сессии. Всё, что пишется, проходит через журнал."""

    def __init__(self, root: str | Path, *, profile: Profile, source: str,
                 synthetic: bool, note: str | None = None,
                 governor: "ResourceGovernor | None" = None,
                 devices: Any = None, check_resources_every: int = 30,
                 lineage_id: str | None = None) -> None:
        from . import __version__

        self.root = Path(root)
        self.profile = profile
        if self.root.exists() and any(self.root.iterdir()):
            # Отказ **осматривает** каталог и называет команду. Прежний говорил только
            # «не пуст», и оператор, прервавший запись, получал этот отказ на повторную
            # попытку — то есть на ровном месте вторично упирался в то же место.
            raise SessionError(f"{self.root} не пуст.\n" + describe_existing(self.root))
        self.root.mkdir(parents=True, exist_ok=True)
        meta = SessionMeta(time.time(), profile.as_dict(), source, synthetic,
                           platform.platform(), __version__, note)
        (self.root / SESSION_META).write_text(
            json.dumps(meta.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        # Линия объявляется при создании и потом неизменна (`BranchMeta` frozen).
        # `None` означает «линии нет, это пробный прогон» — и означает это **явно**:
        # пустая строка, стоявшая здесь раньше, проходила любую проверку на
        # присутствие поля и слила бы разные линии в одну. Запись с чужой машины —
        # отдельная линия по построению, и её идентификатор ставит `harness record`.
        self.journal = Journal.create(self.root / "journal", profile,
                                      reason=f"новая сессия, источник {source}",
                                      lineage_id=lineage_id)

        p = profile.parameters
        self.frames = FrameStore(self.root / "frames", mode="a",
                                 shard_bytes=int(p.get("frame_shard_bytes", 64 << 20)),
                                 compress_level=int(p.get("frame_compress_level", 6)),
                                 keyframe_interval=int(p.get("frame_keyframe_interval", 30)))
        self.audio = AudioStore(self.root / "audio", mode="a",
                                shard_bytes=int(p.get("frame_shard_bytes", 64 << 20)),
                                compress_level=int(p.get("frame_compress_level", 6)))
        self.clocks = Clocks()
        self._audio_channels = int(profile.structural.get("audio_channels", 2))
        self._synthetic = bool(synthetic)
        self._expect_shape: tuple[int, int] | None = None
        self.governor = governor
        self._check_every = max(1, int(check_resources_every))
        self._frames_written = 0
        self._drop_tolerance = int(profile.parameters["capture_drop_tolerance"])
        self._last_frame_world: int | None = None
        # Ссылка на последний записанный кадр: её переиспользует `record_unchanged`.
        # Хранится ссылка, а не изображение: содержимое дублировать не надо, оно уже
        # лежит в хранилище и адресуется по содержимому.
        self._last_frame_ref: BlobRef | None = None
        self.unchanged_written = 0
        self._gaps_noticed = 0
        self._frames_refused = 0

        if devices is not None:
            # Состав устройств пишется в журнал, а не только в session.json:
            # журнал — источник истины, и «чем это было записано» — часть опыта.
            self.journal.append(EntryKind.DEVICE, self.clocks.stamp(), Actor.HUMAN,
                                ActorLayer.HUMAN,
                                event={"code": "registry", **devices.summary()})

    # --- сверка профиля с записью -------------------------------------------

    def _check_declared(self, image: np.ndarray) -> None:
        """Сверить с первым кадром всё, что профиль о кадре заявляет. `RECORD_CLAIMS`.

        Вызывается один раз, на первом кадре: свойства кадра в пределах сессии
        неизменны (размер и формат — структурные), а проверять каждый кадр значило бы
        платить за это тридцать раз в секунду.
        """
        # Формат — на **обоих** путях. Читателем `frame_format` до этого был только
        # живой захват (`selftest`, `harness record`), а синтетические миры рисуют кадр
        # `(H, W)` uint8 всегда, чем бы профиль ни объявлял. Профиль с `rgb8` молча
        # получал серую запись — та же ложь записи о себе, что и с размером кадра, но
        # в другой оси. Теперь такой профиль получает отказ.
        fmt = str(self.profile.structural["frame_format"])
        want_dims = {"gray8": 2, "rgb8": 3}.get(fmt)
        if want_dims is not None and int(image.ndim) != want_dims:
            raise SessionError(
                f"профиль заявляет frame_format={fmt} (это {want_dims} измерения), а "
                f"кадр пришёл с формой {tuple(int(x) for x in image.shape)}. "
                "Синтетические миры рисуют серый кадр всегда: для rgb8 нужен источник, "
                "который его отдаёт, иначе запись врёт о себе")
        if fmt == "rgb8" and int(image.shape[-1]) != 3:
            raise SessionError(
                f"профиль заявляет rgb8, а у кадра {int(image.shape[-1])} канала")

        # Размер — только у живых записей: у синтетических его задаёт профиль, и миры
        # его читают, а здесь его задаёт экран, и спорить с экраном бессмысленно.
        if self._synthetic:
            return
        want = (int(self.profile.parameters["capture_height"]),
                int(self.profile.parameters["capture_width"]))
        got = (int(image.shape[0]), int(image.shape[1]))
        if got != want:
            raise SessionError(
                f"кадр {got[1]}×{got[0]}, а профиль записи заявляет "
                f"{want[1]}×{want[0]}. Живой захват отдаёт монитор целиком, и "
                "уменьшать его никто не просит, поэтому профиль надо строить по "
                "кадру: profile.for_frame(первый_кадр). Иначе запись врёт о "
                "себе, и любой расчёт по её профилю ошибётся во столько раз, во "
                "сколько различаются площади")

    # --- запись -------------------------------------------------------------

    def record_frame(self, image: np.ndarray, *, t_world: int | None = None,
                     t_content: float | None = None,
                     audio: np.ndarray | None = None,
                     audio_offset_ms: float = 0.0,
                     actor: Actor = Actor.NONE,
                     actor_layer: ActorLayer = ActorLayer.NONE,
                     state: StateSnapshot | None = None,
                     wall_clock: float | None = None) -> Entry | None:
        """Кадр (и, если есть, синхронный блок звука) как одна запись журнала.

        Возвращает `None`, если ограничитель ресурсов отказал в записи: место или
        память кончились. Кадр при этом теряется, и об этом пишется запись — но
        уже записанное остаётся целым. Обратный порядок (выкинуть старое, чтобы
        записать новое) сделал бы журнал невоспроизводимым.
        """
        if self.governor is not None:
            if self._frames_written % self._check_every == 0:
                self.governor.check(self.clocks.stamp())
            from .core.resources import OP_FRAME
            admission = self.governor.admit(OP_FRAME)
            if not admission:
                # Кадр — второй уровень, сенсорная роскошь: в него отказывают. А
                # `record_gap` пишет причинную запись нулевого уровня, и она
                # проходит всегда — иначе журнал выглядел бы непрерывным там, где
                # запись оборвалась по месту, и это была бы худшая из потерь.
                self._frames_refused += 1
                self.record_gap("resource_refused",
                                {"reason": admission.reason,
                                 "frames_refused": self._frames_refused,
                                 "level_refused": 2, "level_kept": 0})
                return None

        if self._expect_shape is None:
            self._check_declared(image)
            self._expect_shape = (int(image.shape[0]), int(image.shape[1]))

        self.clocks.tick_self()
        self.clocks.set_world(self.clocks.t_world + 1 if t_world is None else t_world)
        self.clocks.set_content(t_content)
        stamp = self.clocks.stamp()

        frame_ref = self.frames.append_frame(image)
        self._last_frame_ref = frame_ref
        audio_ref: BlobRef | None = None
        if audio is not None:
            audio_ref = self.audio.append_block(audio, channels=self._audio_channels)

        event: dict[str, Any] = {"code": "frame"}
        if audio_ref is not None:
            event["audio_offset_ms"] = round(float(audio_offset_ms), 3)

        # Пропуск кадров замечается здесь, а не оставляется на совесть вызывающего.
        # Иначе запись выглядит непрерывной там, где источник отвалился, и всё, что
        # считается по соседним кадрам — сдвиг, слои, ошибка предсказания, — молча
        # считается по разным моментам времени. Норма пропуска — из профиля
        # (`capture_drop_tolerance`), потому что она зависит от источника.
        if self._last_frame_world is not None:
            missed = int(stamp.t_world) - int(self._last_frame_world) - 1
            if missed > self._drop_tolerance:
                self.record_gap("frames_dropped",
                                {"missed": missed,
                                 "tolerance": self._drop_tolerance,
                                 "from_t_world": int(self._last_frame_world),
                                 "to_t_world": int(stamp.t_world)})
                self._gaps_noticed += 1
        self._last_frame_world = int(stamp.t_world)
        self._frames_written += 1
        # Кадр не начат никаким контуром: он приходит от источника, а не от
        # решения. Поэтому слой none, и это выбранное значение, а не пропущенное.
        # `wall_clock` передаётся насквозь и по умолчанию не задаётся: системные
        # часы ставит сам журнал (`Journal.append`). Явное значение нужно ровно для
        # одного — изготовить запись «с чужой машины», у которой часы идут иначе:
        # взять такую запись негде, а проверить приём чужих записей без неё нельзя.
        return self.journal.append(EntryKind.FRAME, stamp, actor, actor_layer,
                                   frame=frame_ref, audio=audio_ref, event=event,
                                   state=state, wall_clock=wall_clock)

    def record_unchanged(self, *, t_world: int | None = None,
                         t_content: float | None = None,
                         actor: Actor = Actor.NONE,
                         actor_layer: ActorLayer = ActorLayer.NONE,
                         wall_clock: float | None = None) -> Entry:
        """Экран не изменился. **Запись, а не пропуск.**

        Desktop Duplication не выдаёт неизменённый кадр вовсе (`dxcam.grab()` отдаёт
        `None`). Механизм разумный, но у него есть ловушка, и она дорогая: первая
        запись минимального набора — «неподвижность, 60 с» — статична по замыслу.
        Наивный счётчик отрапортовал бы «потеряно 1700 кадров из 1800» на записи,
        прошедшей идеально: изменений ровно ноль, и это искомый ответ.

        Поэтому здесь пишется отметка со временем, а **ссылка на кадр берётся у
        предыдущей записи**: момент времени новый, содержимое то же. Дублировать
        пиксели незачем — хранилище адресуется по содержимому и всё равно склеило бы
        их в один сегмент, но тогда в журнале стояло бы «пришёл кадр», и «экран не
        менялся» стало бы неотличимо от «пришёл точно такой же кадр». Это разные
        утверждения: первое — про устройство механизма, второе — про мир.

        `record_gap` для этого не годится категорически: разрыв означает потерю, а
        здесь ничего не потеряно.
        """
        if self._last_frame_ref is None:
            raise SessionError(
                "«без изменений» до первого кадра: ссылаться не на что. Первый кадр "
                "приходит всегда, даже на неподвижном экране, и если его не было — "
                "это поломка захвата, а не отсутствие изменений")
        self.clocks.tick_self()
        self.clocks.set_world(self.clocks.t_world + 1 if t_world is None else t_world)
        self.clocks.set_content(t_content)
        stamp = self.clocks.stamp()
        self._last_frame_world = int(stamp.t_world)
        self.unchanged_written += 1
        return self.journal.append(
            EntryKind.FRAME, stamp, actor, actor_layer,
            frame=self._last_frame_ref,
            event={"code": "unchanged", "same_as_seq": self.journal.seq},
            wall_clock=wall_clock)

    def record_source(self, *, kind: str, region: dict[str, Any] | None,
                      why: str, window: str = "") -> Entry:
        """Чем задан источник этой записи: вид, рамка и **откуда рамка взялась**.

        Вид дублирует `capture_source` из профиля нарочно: профиль говорит, при каких
        настройках писали, а эта запись — что в действительности открылось. Расхождение
        между ними — находка, и обнаружить её можно только если записаны оба.

        Рамка сюда, а не в профиль: координаты окна зависят от того, куда его подвинули
        мышью, и форк журнала на каждое перетаскивание бессмыслен. `why` словами, потому
        что «960×540» через месяц не отличить от рамки, взятой наугад.

        Заголовок окна здесь **не пишется** — только непрозрачный идентификатор
        (инварианты 4 и 5): заголовок это текст с экрана.
        """
        return self.journal.append(EntryKind.DEVICE, self.clocks.stamp(),
                                   Actor.HUMAN, ActorLayer.HUMAN,
                                   event={"code": "source", "kind": kind,
                                          "region": region, "why": why,
                                          "window": window})

    def record_source_gone(self, *, detail: dict[str, Any]) -> Entry:
        """Источник исчез: окно закрыто, свёрнуто, монитор отключён. **Наблюдение.**

        TASK-21, часть 6. Пишется видом `DEVICE` — «смена набора источников», — а не
        `CAPTURE_GAP`. Разница по существу: разрыв означает «механизм не справился с тем,
        что было», исчезновение — «того, что было, больше нет». Первое — брак записи,
        второе — содержание записи.

        `Actor.NONE`: исчезновение не инициировал никто из тех, кого журнал различает.
        Окно закрыл человек **вне** эксперимента, монитор отключил кабель. Приписать это
        человеку-оператору значило бы заявить инициатора, которого не наблюдали, — то есть
        сделать в журнале ровно то, что метрика конфабуляции ловит у планировщика.
        """
        return self.journal.append(EntryKind.DEVICE, self.clocks.stamp(),
                                   Actor.NONE, ActorLayer.NONE,
                                   event={"code": "source_gone", **detail})

    def record_gap(self, code: str, detail: dict[str, Any]) -> Entry:
        """Пропуск кадров, рассинхрон, поломка механизма. Молчать об этом нельзя.

        **Исчезновение источника сюда не пишется**: у него своя дорога
        (`record_source_gone`). Здесь только то, где виноват механизм.
        """
        return self.journal.append(EntryKind.CAPTURE_GAP, self.clocks.stamp(),
                                   Actor.NONE, ActorLayer.NONE,
                                   event={"code": code, **detail})

    def record_perception(self, payload: dict[str, Any]) -> Entry:
        """Результат границы восприятия. Проверяется на читаемый текст журналом."""
        return self.journal.append(EntryKind.PERCEPTION, self.clocks.stamp(),
                                   Actor.NONE, ActorLayer.NONE, perception=payload)

    def record_intervention(self, code: str, detail: dict[str, Any]) -> Entry:
        """Вмешательство исследователя: браковка цели, подтверждение шага, ответ.

        Пишется отдельным видом записи, чтобы прогоны с вмешательствами не
        сравнивались с чистыми как равные (см. DESIGN-REVIEW-CONSOLE.md, пункт 6).
        """
        return self.journal.append(EntryKind.INTERVENTION, self.clocks.stamp(),
                                   Actor.HUMAN, ActorLayer.HUMAN,
                                   event={"code": code, **detail})

    def record_interrupted(self, *, elapsed_s: float, total_s: float,
                           written: int, unchanged: int, reason: str = "Ctrl+C") -> Entry:
        """Оператор остановил запись. На чём именно остановил — часть записи.

        Пишется **до** закрытия хранилищ, поэтому попадает в журнал целиком, и запись
        остаётся годной: частичный корпус — корпус, а трассировка из середины
        `zlib.compress` — не результат ни в каком виде.

        Отметка идёт в журнал, а не в `session.json`: журнал — источник истины, и всё
        остальное из него выводимо (инвариант 1). `Session.interrupted` её оттуда и
        читает, а не хранит вторую копию.
        """
        note = (f"прервано на {elapsed_s:.0f} с из {total_s:.0f} с "
                f"({written} кадров, {unchanged} без изменений)")
        return self.record_intervention(
            INTERRUPTED,
            {"elapsed_s": round(float(elapsed_s), 3),
             "total_s": round(float(total_s), 3),
             "frames_written": int(written), "unchanged": int(unchanged),
             "reason": reason, "note": note})

    def record_note(self, text: str) -> Entry:
        return self.journal.append(EntryKind.NOTE, self.clocks.stamp(), Actor.HUMAN,
                                   ActorLayer.HUMAN,
                                   event={"code": "note", "text": text})

    def change_parameters(self, new_profile: Profile, *, reason: str) -> None:
        self.journal.change_parameters(new_profile, self.clocks.stamp(), reason=reason)
        self.profile = new_profile

    def fork(self, new_profile: Profile, *, reason: str) -> Recorder:
        """Структурный переключатель сменился: новая ветка в той же сессии."""
        self.journal = self.journal.fork(new_profile, reason=reason)
        self.profile = new_profile
        return self

    def cost(self) -> dict[str, Any]:
        """Куда ушло время и сколько вышло байтов, по хранилищам. `TASK-17`, пункт 2.

        Разбивка нужна на машине оператора: живая запись дала 6.5 кадр/с против 32.5 в
        `selftest` на той же машине, и «узкое место где-то в записи на диск» — не диагноз.
        Здесь она собирается там же, где происходит, а не восстанавливается по итогам.
        """
        return {"frames": self.frames.cost.as_dict(),
                "audio": self.audio.cost.as_dict()}

    def close(self) -> None:
        self.frames.close()
        self.audio.close()
        self.journal.close()

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class FrameCursor:
    """Один кадр записи: где он в журнале, когда он был, чем он был."""

    index: int          # номер по порядку среди кадров
    entry_seq: int      # номер записи журнала
    stamp: Stamp
    frame: BlobRef
    audio: BlobRef | None


class Session:
    """Чтение записанной сессии. Основной инструмент разработки (0.4).

    Ничего не открывает на запись. Промотка — по индексу в памяти, без
    перечитывания журнала.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        meta_path = self.root / SESSION_META
        if not meta_path.exists():
            raise SessionError(f"не похоже на сессию: нет {meta_path}")
        self.meta = SessionMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        self.profile = Profile.from_dict(self.meta.profile)
        self.journal = Journal.open_latest(self.root / "journal", mode="r")
        self.frames = FrameStore(self.root / "frames", mode="r")
        audio_dir = self.root / "audio"
        self.audio = AudioStore(audio_dir, mode="r") if audio_dir.is_dir() else None
        self._cursors: list[FrameCursor] = []
        self._world_ticks: list[int] = []      # для промотки по тику, двоичным поиском
        self._build_index()
        self.position = 0

    @classmethod
    def open(cls, root: str | Path) -> Session:
        return cls(root)

    def _build_index(self) -> None:
        for e in self.journal.frames():
            if e.frame is None:
                continue
            idx = len(self._cursors)
            self._cursors.append(FrameCursor(idx, e.seq, e.stamp, e.frame, e.audio))
            self._world_ticks.append(e.stamp.t_world)

    # --- доступ -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cursors)

    def cursor(self, index: int) -> FrameCursor:
        try:
            return self._cursors[index]
        except IndexError:
            raise SessionError(f"нет кадра {index}: в записи {len(self._cursors)}") from None

    def image(self, index: int) -> np.ndarray:
        return self.frames.read(self.cursor(index).frame)

    def audio_block(self, index: int) -> np.ndarray | None:
        cur = self.cursor(index)
        if cur.audio is None or self.audio is None:
            return None
        return self.audio.read(cur.audio)

    def seek(self, index: int) -> FrameCursor:
        if not 0 <= index < len(self._cursors):
            raise SessionError(f"промотка за пределы записи: {index} из {len(self._cursors)}")
        self.position = index
        return self._cursors[index]

    def seek_world(self, t_world: int) -> FrameCursor:
        """Промотка по тикам мира: ближайший кадр не позже запрошенного тика.

        Двоичным поиском: тики мира в журнале монотонны — это гарантирует
        `Journal.append`, — поэтому линейный проход по часовой записи не нужен.
        """
        import bisect

        if not self._cursors:
            raise SessionError("в записи нет кадров")
        pos = bisect.bisect_right(self._world_ticks, t_world) - 1
        return self.seek(max(0, pos))

    def step(self, delta: int = 1) -> FrameCursor:
        return self.seek(max(0, min(len(self._cursors) - 1, self.position + delta)))

    def __iter__(self) -> Iterator[tuple[FrameCursor, np.ndarray]]:
        for c in self._cursors:
            yield c, self.frames.read(c.frame)

    def actions(self) -> Iterator[tuple[Entry, Action]]:
        for e in self.journal.entries([EntryKind.ACTION]):
            if e.action is not None:
                yield e, e.action

    def map(self, fn: Callable[..., Any], *, with_audio: bool = False) -> list[Any]:
        """Прогнать функцию по всей записи (критерий готовности 0.4).

        `fn(image, cursor)` или, при `with_audio=True`, `fn(image, audio, cursor)`.
        Результат на каждом кадре — по порядку.
        """
        out: list[Any] = []
        for c, img in self:
            out.append(fn(img, self.audio_block(c.index), c) if with_audio else fn(img, c))
        return out

    # --- проверка ------------------------------------------------------------

    def verify(self) -> dict[str, Any]:
        """Целостность записи. Всё, что не сходится, — в отчёт, не в исключение.

        Кроме порванной цепочки журнала: это `TamperError`, потому что дальше
        обсуждать нечего.
        """
        self.journal.verify()
        report: dict[str, Any] = {
            "session": str(self.root),
            "synthetic": self.meta.synthetic,
            "source": self.meta.source,
            "branch": self.journal.meta.branch_id,
            "profile_hash": short(self.profile.profile_hash),
            "structure_hash": short(self.profile.structure_hash),
            "entries": len(self.journal),
            "frames": len(self._cursors),
            "frames_on_disk": len(self.frames),
            "bytes_frames": self.frames.bytes_on_disk(),
            "bytes_audio": self.audio.bytes_on_disk() if self.audio else 0,
            "problems": [],
        }
        problems: list[str] = report["problems"]

        # Отметка «без изменений» — запись журнала вида FRAME, ссылающаяся на **уже
        # лежащий** блок: нового блока она не добавляет. Считать её кадром хранилища
        # нельзя, и до TASK-16 именно это здесь и делалось: любая запись со статикой
        # объявлялась сломанной. Первой такой была бы запись «неподвижность» — опорная
        # запись минимального набора, у которой отметок больше, чем кадров, — то есть
        # оператор получил бы приговор своей записи на самой важной из них.
        marks = sum(1 for e in self.journal.frames()
                    if e.event.get("code") == "unchanged")
        fresh = len(self._cursors) - marks
        report["unchanged_marks"] = marks
        report["frames_new"] = fresh
        if fresh != len(self.frames):
            problems.append(
                f"новых кадров в журнале {fresh} (плюс {marks} отметок «без изменений»), "
                f"в хранилище {len(self.frames)}: часть кадров записана без записи "
                "журнала или наоборот")

        # Пропуски тиков мира: в вехе 0 тик мира — номер кадра, значит шаг 1.
        gaps = []
        for a, b in zip(self._cursors, self._cursors[1:]):
            d = b.stamp.t_world - a.stamp.t_world
            if d != 1:
                gaps.append({"after_frame": a.index, "delta": d})
        if gaps:
            report["world_tick_gaps"] = gaps[:20]
            problems.append(f"разрывов в тиках мира: {len(gaps)}")

        # Рассинхрон звука: критерий 0.1 — не больше audio_sync_tolerance_ms.
        tol = float(self.profile.parameters.get("audio_sync_tolerance_ms", 50.0))
        worst = 0.0
        for e in self.journal.frames():
            off = abs(float(e.event.get("audio_offset_ms", 0.0)))
            worst = max(worst, off)
        report["worst_audio_offset_ms"] = round(worst, 3)
        report["audio_sync_tolerance_ms"] = tol
        if worst > tol:
            problems.append(f"рассинхрон звука {worst:.1f} мс при допуске {tol:.0f} мс")

        # Достигнутая частота кадров — по настенному времени самой записи, а не по
        # заявленной в профиле. `capture_fps` — цель для цикла, а не свойство записи, и
        # отказывать по нему нельзя: машина, отдавшая 28 кадров вместо тридцати, работает
        # нормально. Но всякий, кто считает по записи длительность как «кадров делить на
        # capture_fps», обязан иметь возможность увидеть настоящее число, иначе профиль
        # снова оказывается единственным источником, а он уже врал про размер кадра.
        #
        # Порога здесь нет намеренно. Он стоит в `selftest`, где оператор ещё может
        # что-то поменять; порог, придуманный на чтении, был бы числом, нарисованным
        # рукой (инвариант 23), и сравнимости между прогонами не дал бы.
        report["capture_fps_declared"] = float(
            self.profile.parameters.get("capture_fps", 0.0))
        report["capture_fps_observed"] = None
        if len(self._cursors) > 1:
            walls = [e.wall_clock for e in self.journal.frames()
                     if e.wall_clock is not None]
            if len(walls) > 1:
                span = max(walls) - min(walls)
                if span > 0:
                    report["capture_fps_observed"] = round((len(walls) - 1) / span, 3)

        # Кадры должны читаться. Проверяем выборочно: начало, середина, конец,
        # иначе проверка часовой записи сама станет часовой.
        if self._cursors:
            probe = {0, len(self._cursors) // 2, len(self._cursors) - 1}
            for i in sorted(probe):
                try:
                    self.image(i)
                except Exception as exc:
                    problems.append(f"кадр {i} не читается: {exc}")

        report["branches"] = [b.branch_id for b in branch_chain(self.root / "journal")]
        # Прерванная запись — **не** проблема целостности: она короче задуманной, но
        # цела. Поэтому отметка идёт в отчёт отдельным полем, а в `problems` не идёт:
        # иначе `harness verify` объявлял бы годный частичный корпус сломанным, и
        # оператор второй раз получал бы приговор своей записи.
        report["interrupted"] = self.interrupted
        report["ok"] = not problems
        return report

    @property
    def interrupted(self) -> dict[str, Any] | None:
        """Отметка о прерывании, если она есть. Выводится из журнала, не хранится.

        `None` — запись дописана до конца. Словарь — событие записи `INTERVENTION` с
        кодом `record_interrupted`: на какой секунде из какой, сколько кадров успело.
        """
        for e in self.journal:
            if e.kind is EntryKind.INTERVENTION and e.event.get("code") == INTERRUPTED:
                return dict(e.event)
        return None

    def close(self) -> None:
        self.frames.close()
        if self.audio:
            self.audio.close()
        self.journal.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
