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
                 lineage_id: str = "") -> None:
        from . import __version__

        self.root = Path(root)
        self.profile = profile
        if self.root.exists() and any(self.root.iterdir()):
            raise SessionError(
                f"{self.root} не пуст. Сессии не дописываются поверх чужих: "
                "запись только дозаписывается внутри своей сессии")
        self.root.mkdir(parents=True, exist_ok=True)
        meta = SessionMeta(time.time(), profile.as_dict(), source, synthetic,
                           platform.platform(), __version__, note)
        (self.root / SESSION_META).write_text(
            json.dumps(meta.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        # Линия объявляется при создании и потом неизменна (`BranchMeta` frozen).
        # Пустая линия у живой записи была бы неверной по существу: запись с чужой
        # машины — это отдельная линия по построению, а пустое поле означает «про
        # линию ничего не сказано», и от «своя линия» это неотличимо.
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

    def record_gap(self, code: str, detail: dict[str, Any]) -> Entry:
        """Пропуск кадров, рассинхрон, отвал источника. Молчать об этом нельзя."""
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

        if len(self._cursors) != len(self.frames):
            problems.append(
                f"кадров в журнале {len(self._cursors)}, в хранилище {len(self.frames)}: "
                "часть кадров записана без записи журнала или наоборот")

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
        report["ok"] = not problems
        return report

    def close(self) -> None:
        self.frames.close()
        if self.audio:
            self.audio.close()
        self.journal.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
