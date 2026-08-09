"""Журнал — источник истины.

Инвариант 1: журнал только дозаписывается и никогда не редактируется. Всё
остальное — убеждения, карточки, веса — из него выводимо и может быть
пересобрано с нуля.

Заявить это недостаточно, поэтому здесь оно ещё и проверяемо. Каждая запись
несёт хеш от своего содержимого и хеша предыдущей записи. Любая правка,
вставка или удаление строки рвёт цепочку, и `verify()` показывает, на какой
записи. Файл открывается только в режиме `"a"`; функции «переписать запись» в
этом модуле нет.

Инвариант 2: без трёх часов и хеша профиля запись физически не собирается —
`append` требует `Stamp`, а хеши берёт из профиля ветки.

Инвариант 11: ветка соответствует `structure_hash`. Попытка дописать в ветку
запись с другим `structure_hash` — ошибка, а не тихое продолжение.

Инвариант 13: у каждой записи есть `actor_layer` — какой слой её начал. Он не
имеет значения по умолчанию: `append` требует его позиционно, и запись без него
не собирается вообще. Это сделано именно типом, а не проверкой в теле функции,
потому что проверку в теле можно обойти, а отсутствующий аргумент — нельзя.

Почему `actor` и `actor_layer` оба. `actor` отвечает «чья это запись» — агента,
человека или ничья; он про происхождение и права. `actor_layer` отвечает «какой
контур внутри агента начал действие» — рефлекс на 20 Гц или планировщик на
0.5 Гц. Планировщик не имеет доступа к причинам действий рефлекса, поэтому он
**будет конфабулировать структурно**, и без обоих полей это неизмеримо: нужно и
то, что было заявлено, и то, кто действовал на самом деле.

Формат записи — версия **v2**. Записи v1 не имеют `actor_layer` и среза
состояния; они читаются отдельным типом `LegacyEntry`, у которого этих полей
нет физически, поэтому попытка спросить их даёт `AttributeError`, а не
правдоподобный `none`. См. `versions.py`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping

from .action import Action
from .blobstore import BlobRef
from .clocks import Stamp
from .profile import Profile, short
from .symbols import assert_no_plain_text

GENESIS = "0" * 32

# Версия формата записи. v1 — до появления `actor_layer` и среза состояния;
# v2 — текущая. Версия живёт **в самих данных**, а не в документации: журнал,
# про формат которого надо спросить человека, не является источником истины.
FORMAT = "v2"
FORMAT_V1 = "v1"


class JournalError(RuntimeError):
    pass


class TamperError(JournalError):
    """Журнал правили. Это не восстанавливается — только форк от целой части."""


class FormatError(JournalError):
    """Запрошено то, чего в этом формате нет. Отказ, а не деградация.

    Молчаливая деградация здесь опаснее падения: метрика посчитается, выдаст
    число, число попадёт в отчёт, и никто не узнает, что оно посчитано по полю,
    которого в записях не было.
    """


class Actor(StrEnum):
    """Чья запись. Про происхождение и права, а не про контур."""

    AGENT = "agent"
    HUMAN = "human"
    NONE = "none"


class ActorLayer(StrEnum):
    """Какой слой начал действие (инвариант 13).

    Заполняется **в момент порождения действия**, а не выводится потом. Вывести
    задним числом нельзя в принципе: по журналу видно, что нажали, но не видно,
    кто решил нажать, а именно это и есть предмет измерения конфабуляции.

    `NONE` — законное значение, но выбранное: кадр захвата не начат никаким
    слоем. Значением по умолчанию оно не является, иначе «не заполнили» стало бы
    неотличимо от «начато ничем».
    """

    REFLEX = "reflex"          # ~20 Гц: прицел, тайминг, уклонение
    SKILL = "skill"            # ~2 Гц: готовый макрос из библиотеки
    PLANNER = "planner"        # ~0.5 Гц: шаг плана
    DRIVE = "drive"            # ~0.01 Гц: смена того, что вообще делать
    INTERRUPT = "interrupt"    # сторож, СТОП, упор в ресурс — перебило всё
    HUMAN = "human"            # человек: вмешательство, свидетельство, пометка
    NONE = "none"              # ничей: кадр, служебная запись


# Слои, которые может начать только агент. Запись, где действует человек, но
# слой заявлен планировщиком, означала бы, что чужой поступок попал в биографию
# агента как свой, и метрика конфабуляции считалась бы по подделанным данным.
AGENT_ONLY_LAYERS = (ActorLayer.REFLEX, ActorLayer.SKILL, ActorLayer.PLANNER,
                     ActorLayer.DRIVE)


class Kind(StrEnum):
    """Виды записей. Набор закрыт: новый вид — это изменение схемы журнала."""

    FRAME = "frame"                  # кадр захвата (+ звук, если есть)
    ACTION = "action"                # попытка действия, в том числе заглушённая
    THOUGHT = "thought"              # действие с отключёнными эффекторами
    PERCEPTION = "perception"        # результат границы восприятия, уже в символах
    PROFILE_CHANGE = "profile_change"  # повернули ручку в parameters
    BRANCH_START = "branch_start"    # первая запись ветки
    STOP = "stop"                    # СТОП: инъекция оборвана
    RESUME = "resume"                # снятие стопа
    WATCHDOG = "watchdog"            # сторожевой таймер сработал
    RESOURCE = "resource"            # упор в предел памяти, диска, бюджета
    DEVICE = "device"                # смена набора источников или прав
    GOAL = "goal"                    # цель поставлена, пройдена, брошена
    TESTIMONY = "testimony"          # свидетельство: человек, другой экземпляр, вики
    SLEEP = "sleep"                  # прогон консолидации
    INTERVENTION = "intervention"    # вмешательство исследователя
    CAPTURE_GAP = "capture_gap"      # пропуск кадров, рассинхрон, отвал источника
    PLAN = "plan"                    # шаг плана: что ожидалось и что вышло
    SELF_REPORT = "self_report"      # слова агента о себе, ни на что не влияют
    STATED_REASON = "stated_reason"  # реплика планировщика «почему я это делаю»
    NOTE = "note"                    # пометка исследователя, ни на что не влияет


# Виды записей, которые являются **выражением**, а не показанием (инвариант 20).
# Из них не выводится ничего: ни убеждения, ни карта тела, ни метрики, кроме тех,
# что измеряют сами эти слова. Речь наружу стратегична по природе и может стать
# лживой; это ожидаемо, а не поломка, — но тогда она не имеет права влиять на
# то, что считается по поведению.
EXPRESSION_KINDS = (Kind.SELF_REPORT, Kind.STATED_REASON)


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """Срез состояния на момент записи. `None` означает «не измерялось».

    Ни одно поле не имеет числового значения по умолчанию, и это главное в этом
    классе. `prediction_error = 0.0` означает «предсказание сошлось идеально»;
    `prediction_error = None` означает «предсказателя не было». Подставить здесь
    ноль вместо `None` — ровно та молчаливая заглушка, которая ломает эксперимент
    незаметно: график ошибки предсказания стал бы идеально плоским там, где её
    просто никто не считал.

    `stated_reason_id` — ссылка на запись `STATED_REASON`, если планировщик
    объяснял. Сопоставление этой реплики с `actor_layer` и есть метрика
    конфабуляции; сама запись ничего не сопоставляет, а контроллер не имеет
    права ничего сравнивать (см. `ARCHITECTURE.md`, правила модулей).
    """

    prediction_error: float | None = None
    # Снимок драйвов: имя → значение и прогноз. Пустой словарь означает
    # «драйвов не было», а не «все по нулям».
    drives: dict[str, dict[str, float]] = field(default_factory=dict)
    mood: tuple[float, float] | None = None       # (valence, arousal)
    goal_id: str | None = None
    goal_transition: str | None = None
    stated_reason_id: str | None = None

    def __post_init__(self) -> None:
        if self.mood is not None:
            if len(self.mood) != 2:
                raise JournalError(
                    f"настроение — это (valence, arousal), пришло {self.mood!r}")
            for v in self.mood:
                if not -1.0 <= float(v) <= 1.0:
                    raise JournalError(f"настроение вне [−1, 1]: {self.mood!r}")
        if self.prediction_error is not None and self.prediction_error < 0:
            raise JournalError(
                f"ошибка предсказания отрицательная: {self.prediction_error}")

    @property
    def is_empty(self) -> bool:
        """Ничего не измерялось. Такой срез в запись не пишется вообще."""
        return (self.prediction_error is None and not self.drives
                and self.mood is None and self.goal_id is None
                and self.goal_transition is None and self.stated_reason_id is None)

    def as_dict(self) -> dict[str, Any]:
        """Все ключи всегда, даже пустые.

        Отсутствующий ключ читатель может принять за старый формат, а `null` —
        нет: `null` говорит «поле есть, величины не было».
        """
        return {"prediction_error": self.prediction_error,
                "drives": self.drives,
                "mood": None if self.mood is None else list(self.mood),
                "goal_id": self.goal_id,
                "goal_transition": self.goal_transition,
                "stated_reason_id": self.stated_reason_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> StateSnapshot:
        mood = d.get("mood")
        return cls(
            prediction_error=None if d.get("prediction_error") is None
            else float(d["prediction_error"]),
            drives=dict(d.get("drives") or {}),
            mood=None if mood is None else (float(mood[0]), float(mood[1])),
            goal_id=d.get("goal_id"),
            goal_transition=d.get("goal_transition"),
            stated_reason_id=d.get("stated_reason_id"))


EMPTY_STATE = StateSnapshot()


@dataclass(frozen=True, slots=True)
class Entry:
    """Запись журнала, формат v2.

    `actor_layer` стоит до полей со значениями по умолчанию намеренно: собрать
    запись без него нельзя даже вручную, это `TypeError` конструктора.
    """

    seq: int
    kind: Kind
    stamp: Stamp
    actor: Actor
    actor_layer: ActorLayer
    profile_hash: str
    structure_hash: str
    # Заполняются журналом, а не вызывающим: это идентификация ветки, и
    # вызывающему её знать незачем.
    lineage_id: str = ""
    branch_id: str = ""
    # Якорь реального времени. Не четвёртые часы: по нему нельзя упорядочивать
    # записи (скорость мира и скорость агента меняются независимо, а на разных
    # машинах он ещё и прыгает), и `verify` его не проверяет. Он нужен только
    # исследователю — сопоставить журнал с логами машины. Агентский код журнал
    # не читает вообще, поэтому доступа к нему у агента нет по построению.
    wall_clock: float = 0.0
    state: StateSnapshot = EMPTY_STATE
    frame: BlobRef | None = None
    audio: BlobRef | None = None
    action: Action | None = None
    perception: dict[str, Any] | None = None
    event: dict[str, Any] = field(default_factory=dict)
    prev: str = GENESIS
    digest: str = ""

    def payload(self) -> dict[str, Any]:
        """Содержимое записи без хешей цепочки — то, что хешируется."""
        d: dict[str, Any] = {
            "format": FORMAT,
            "seq": self.seq,
            "kind": str(self.kind),
            "stamp": self.stamp.as_dict(),
            "actor": str(self.actor),
            "actor_layer": str(self.actor_layer),
            "profile_hash": self.profile_hash,
            "structure_hash": self.structure_hash,
            "lineage_id": self.lineage_id,
            "branch_id": self.branch_id,
            "wall_clock": round(float(self.wall_clock), 6),
        }
        if not self.state.is_empty:
            d["state"] = self.state.as_dict()
        if self.frame is not None:
            d["frame"] = self.frame.as_dict()
        if self.audio is not None:
            d["audio"] = self.audio.as_dict()
        if self.action is not None:
            d["action"] = self.action.as_dict()
        if self.perception is not None:
            d["perception"] = self.perception
        if self.event:
            d["event"] = self.event
        return d

    def as_line(self) -> str:
        d = self.payload()
        d["prev"] = self.prev
        d["digest"] = self.digest
        return json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_line(cls, line: str) -> Entry:
        """Прочитать запись v2. Запись v1 здесь **не читается** — это отказ.

        Подставить `actor_layer = none` записи, у которой его никогда не было,
        значило бы заявить «эту запись не начал никакой слой» вместо честного
        «про эту запись такого не спрашивали». После такой подстановки метрика
        конфабуляции посчиталась бы и дала бы число, и это число было бы ложью.
        Для законного чтения v1 есть `LegacyEntry`.
        """
        d = json.loads(line)
        if "actor_layer" not in d:
            raise FormatError(
                f"запись {d.get('seq')} без actor_layer — это формат v1. "
                "Читать её как v2 нельзя: слой-инициатор задним числом не "
                "восстановим. Открывайте ветку через open_legacy()")
        return cls(
            seq=int(d["seq"]),
            kind=Kind(d["kind"]),
            stamp=Stamp.from_dict(d["stamp"]),
            actor=Actor(d["actor"]),
            actor_layer=ActorLayer(d["actor_layer"]),
            profile_hash=str(d["profile_hash"]),
            structure_hash=str(d["structure_hash"]),
            lineage_id=str(d.get("lineage_id", "")),
            branch_id=str(d.get("branch_id", "")),
            wall_clock=float(d.get("wall_clock", 0.0)),
            state=(StateSnapshot.from_dict(d["state"]) if d.get("state")
                   else EMPTY_STATE),
            frame=BlobRef.from_dict(d["frame"]) if d.get("frame") else None,
            audio=BlobRef.from_dict(d["audio"]) if d.get("audio") else None,
            action=Action.from_dict(d["action"]) if d.get("action") else None,
            perception=d.get("perception"),
            event=d.get("event", {}),
            prev=str(d.get("prev", GENESIS)),
            digest=str(d.get("digest", "")),
        )


@dataclass(frozen=True, slots=True)
class LegacyEntry:
    """Запись формата v1. Полей `actor_layer` и `state` здесь нет физически.

    Это не бедный родственник `Entry`, а другой тип, и разница принципиальна:
    обращение к `e.actor_layer` даёт `AttributeError` с указанием строки кода,
    а не `ActorLayer.NONE`, которое выглядело бы как измеренная величина.
    Инвариант «никаких молчаливых заглушек» соблюдается системой типов, а не
    внимательностью читающего.

    Для чего v1 годится: проверка модулей восприятия, которым слой-инициатор не
    нужен, — отпечатки мест, оптический поток, символизация текста.

    Для чего не годится: всё, что касается управляющего стека, целей,
    конфабуляции и атрибуции действий.
    """

    seq: int
    kind: Kind
    stamp: Stamp
    actor: Actor
    profile_hash: str
    structure_hash: str
    frame: BlobRef | None = None
    audio: BlobRef | None = None
    action: Action | None = None
    perception: dict[str, Any] | None = None
    event: dict[str, Any] = field(default_factory=dict)
    prev: str = GENESIS
    digest: str = ""

    @classmethod
    def from_line(cls, line: str) -> LegacyEntry:
        d = json.loads(line)
        if "actor_layer" in d:
            raise FormatError(
                f"запись {d.get('seq')} формата v2 читается как v1. Смешивать "
                "форматы в одном чтении нельзя: у половины записей слой-инициатор "
                "был бы, а у половины нет, и любая доля по нему стала бы ложью")
        return cls(
            seq=int(d["seq"]), kind=Kind(d["kind"]),
            stamp=Stamp.from_dict(d["stamp"]), actor=Actor(d["actor"]),
            profile_hash=str(d["profile_hash"]),
            structure_hash=str(d["structure_hash"]),
            frame=BlobRef.from_dict(d["frame"]) if d.get("frame") else None,
            audio=BlobRef.from_dict(d["audio"]) if d.get("audio") else None,
            action=Action.from_dict(d["action"]) if d.get("action") else None,
            perception=d.get("perception"), event=d.get("event", {}),
            prev=str(d.get("prev", GENESIS)), digest=str(d.get("digest", "")))


def entry_digest(payload: dict[str, Any], prev: str) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    h = hashlib.blake2b(digest_size=16)
    h.update(prev.encode("ascii"))
    h.update(b"\x00")
    h.update(blob.encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class BranchMeta:
    """Что за ветка. Версия формата и линия — тоже здесь, а не в документации."""

    branch_id: str
    structure_hash: str
    profile: dict[str, Any]
    parent: str | None
    forked_at_seq: int | None
    forked_at_stamp: dict[str, Any] | None
    fork_reason: str | None
    format: str = FORMAT_V1      # у старых веток файла без поля — это и есть v1
    lineage_id: str = ""
    # Ветка read-only на уровне данных, а не соглашения. Ставится для v1: дописать
    # в неё запись v2 значило бы получить журнал, половина которого отвечает на
    # вопрос о слое-инициаторе, а половина нет.
    read_only: bool = False

    @property
    def is_v1(self) -> bool:
        return self.format == FORMAT_V1

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "structure_hash": self.structure_hash,
            "profile": self.profile,
            "parent": self.parent,
            "forked_at_seq": self.forked_at_seq,
            "forked_at_stamp": self.forked_at_stamp,
            "fork_reason": self.fork_reason,
            "format": self.format,
            "lineage_id": self.lineage_id,
            "read_only": self.read_only,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BranchMeta:
        return cls(str(d["branch_id"]), str(d["structure_hash"]), dict(d["profile"]),
                   d.get("parent"), d.get("forked_at_seq"), d.get("forked_at_stamp"),
                   d.get("fork_reason"),
                   format=str(d.get("format", FORMAT_V1)),
                   lineage_id=str(d.get("lineage_id", "")),
                   read_only=bool(d.get("read_only", False)))


class Journal:
    """Одна ветка журнала. Дозапись, проверяемая цепочка, штамп в каждой записи."""

    ENTRIES = "entries.jsonl"
    META = "branch.json"

    def __init__(self, path: str | Path, mode: Literal["a", "r"] = "r") -> None:
        self.path = Path(path)
        self.mode = mode
        meta_path = self.path / self.META
        if not meta_path.exists():
            raise JournalError(f"нет ветки журнала: {meta_path}")
        self.meta = BranchMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        if self.meta.is_v1:
            raise FormatError(
                f"ветка {self.meta.branch_id} записана в формате v1: в её записях "
                "нет ни слоя-инициатора, ни среза состояния. Открывайте её через "
                "LegacyJournal — тогда видно, что это другой формат, а не тот же "
                "самый с пропущенными полями")
        if self.meta.read_only and mode == "a":
            raise FormatError(
                f"ветка {self.meta.branch_id} помечена read-only в самих данных")
        self._profile = Profile.from_dict(self.meta.profile)
        self._text_symbolized = bool(
            self._profile.structural.get("text_symbolized", True))
        self._fh = None
        self._seq = 0
        self._last = GENESIS
        self._last_stamp: Stamp | None = None
        self._scan_tail()
        if mode == "a":
            self._fh = (self.path / self.ENTRIES).open("a", encoding="utf-8")

    # --- создание и форк ----------------------------------------------------

    @staticmethod
    def _branch_dir(root: Path, profile: Profile, n: int) -> Path:
        return root / "branches" / f"{n:03d}-{profile.structure_hash[:8]}"

    @classmethod
    def create(cls, root: str | Path, profile: Profile, *, reason: str | None = None,
               parent: str | None = None, forked_at_seq: int | None = None,
               forked_at_stamp: Stamp | None = None,
               lineage_id: str = "") -> Journal:
        root = Path(root)
        existing = sorted((root / "branches").glob("*")) if (root / "branches").is_dir() else []
        path = cls._branch_dir(root, profile, len(existing))
        if path.exists():
            raise JournalError(f"ветка уже существует: {path}")
        path.mkdir(parents=True)
        meta = BranchMeta(path.name, profile.structure_hash, profile.as_dict(), parent,
                          forked_at_seq, forked_at_stamp.as_dict() if forked_at_stamp else None,
                          reason, format=FORMAT, lineage_id=lineage_id)
        (path / cls.META).write_text(
            json.dumps(meta.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        j = cls(path, mode="a")
        j.append(Kind.BRANCH_START, Stamp(0, 0, None), Actor.NONE, ActorLayer.NONE,
                 event={"code": "branch_start", "branch": path.name,
                        "parent": parent, "reason": reason, "format": FORMAT,
                        "lineage_id": lineage_id,
                        "structure_hash_short": short(profile.structure_hash)})
        return j

    @classmethod
    def open_latest(cls, root: str | Path, mode: Literal["a", "r"] = "r") -> Journal:
        root = Path(root)
        d = root / "branches"
        branches = [p for p in sorted(d.iterdir()) if p.is_dir()] if d.is_dir() else []
        if not branches:
            raise JournalError(f"в {root} нет ни одной ветки журнала")
        return cls(branches[-1], mode=mode)

    def fork(self, new_profile: Profile, *, reason: str) -> Journal:
        """Структурный переключатель сменился → новая ветка.

        Старая ветка закрывается на запись, но остаётся целой: она и есть
        доказательство того, что было до переключения.
        """
        if not self._profile.forks_journal(new_profile):
            raise JournalError(
                "structure_hash не изменился — форк не нужен. "
                "Для смены параметров есть change_parameters()"
            )
        stamp = self._last_stamp or Stamp(0, 0, None)
        self.append(Kind.PROFILE_CHANGE, stamp, Actor.HUMAN, ActorLayer.HUMAN,
                    event={"code": "fork_out", "reason": reason,
                           "to_structure_hash": new_profile.structure_hash,
                           "diff": self._profile.diff(new_profile)})
        self.close()
        return Journal.create(self.path.parent.parent, new_profile, reason=reason,
                              parent=self.meta.branch_id, forked_at_seq=self._seq,
                              forked_at_stamp=stamp,
                              lineage_id=self.meta.lineage_id)

    def change_parameters(self, new_profile: Profile, stamp: Stamp, *, reason: str) -> None:
        """Повернуть ручки на ходу: ветка та же, `profile_hash` дальше другой."""
        if self._profile.forks_journal(new_profile):
            raise JournalError(
                "изменился structure_hash: нужен fork(), а не change_parameters(). "
                "Смешивать структуры в одной ветке нельзя (инвариант 11)"
            )
        diff = self._profile.diff(new_profile)
        if not diff:
            raise JournalError("профиль не изменился — записывать нечего")
        self.append(Kind.PROFILE_CHANGE, stamp, Actor.HUMAN, ActorLayer.HUMAN,
                    event={"code": "parameters", "reason": reason, "diff": diff,
                           "to_profile_hash": new_profile.profile_hash})
        self._profile = new_profile
        self._text_symbolized = bool(
            new_profile.structural.get("text_symbolized", True))

    # --- чтение хвоста ------------------------------------------------------

    def _scan_tail(self) -> None:
        path = self.path / self.ENTRIES
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    e = Entry.from_line(line)
                except FormatError:
                    raise
                except Exception as exc:
                    # Битая строка обязана давать внятную ошибку, а не сырой
                    # ValueError из перечисления: открытие журнала — первое, что
                    # делает всякий читатель, и по сообщению должно быть видно, что
                    # именно испорчено и где.
                    raise TamperError(
                        f"{path}:{lineno}: запись не читается при открытии журнала: "
                        f"{exc}") from exc
                self._seq = e.seq + 1
                self._last = e.digest
                self._last_stamp = e.stamp

    # --- запись ------------------------------------------------------------

    @property
    def profile(self) -> Profile:
        return self._profile

    @property
    def seq(self) -> int:
        return self._seq

    def append(self, kind: Kind, stamp: Stamp, actor: Actor,
               actor_layer: ActorLayer, *,
               frame: BlobRef | None = None, audio: BlobRef | None = None,
               action: Action | None = None, perception: dict[str, Any] | None = None,
               event: dict[str, Any] | None = None,
               state: StateSnapshot | None = None,
               wall_clock: float | None = None) -> Entry:
        """Дописать запись. `actor_layer` — позиционный и без значения по умолчанию.

        Инвариант 13 выполняется здесь подписью функции: вызов без слоя-инициатора
        не собирается вообще. Значение по умолчанию сделало бы его тихим `none`, а
        тихий `none` неотличим от «начато ничем» — и метрика конфабуляции считалась
        бы по записям, про которые никто ничего не заявлял.

        `state` — срез состояния. `None` означает «нечего снимать»: в вехе 0 нет
        ни драйвов, ни ошибки предсказания, и подставить сюда нули значило бы
        записать, что они измерены и равны нулю.

        `wall_clock` берётся из системных часов, если не передан. Передаётся он
        только в тестах, которым нужна побайтовая воспроизводимость записи.
        """
        if self.mode != "a":
            raise JournalError("журнал открыт только на чтение")
        if not isinstance(stamp, Stamp):
            raise JournalError("запись без трёх часов невозможна (инвариант 2)")
        if not isinstance(actor_layer, ActorLayer):
            raise JournalError(
                f"слой-инициатор должен быть ActorLayer, пришло {actor_layer!r}. "
                "Строка сюда не годится: набор слоёв закрыт (инвариант 13)")
        if actor_layer in AGENT_ONLY_LAYERS and actor is not Actor.AGENT:
            raise JournalError(
                f"слой {actor_layer} заявлен, но действует {actor}. Приписать чужой "
                "поступок контуру агента нельзя: метрика конфабуляции считалась бы "
                "по подделанной атрибуции")
        if (actor is Actor.HUMAN) != (actor_layer is ActorLayer.HUMAN):
            raise JournalError(
                f"{actor} против слоя {actor_layer}: человек действует слоем human, "
                "и наоборот. Иначе вмешательство исследователя попадёт в биографию "
                "агента как его собственное решение")
        if self._last_stamp is not None:
            if stamp.t_self < self._last_stamp.t_self:
                raise JournalError(
                    f"t_self назад: {self._last_stamp.t_self} → {stamp.t_self}. "
                    "Журнал дозаписывается, время в нём не отматывается")
            if stamp.t_world < self._last_stamp.t_world:
                raise JournalError(f"t_world назад: {self._last_stamp.t_world} → {stamp.t_world}")
        if perception is not None and self._text_symbolized:
            # Инвариант 5: то, что уйдёт агенту, проходит границу без читаемого текста.
            #
            # Проверка отключается ровно одним способом — структурной ручкой
            # `text_symbolized=False`, то есть заявленным ablation-прогоном «а если
            # дать агенту читаемый язык». Ручка форкает журнал (инвариант 11), и
            # смешать такой опыт с обычным нельзя: у записей другой structure_hash.
            # Ничего другого проверку не выключает, и по умолчанию она включена.
            assert_no_plain_text(perception, path="perception")
        if action is not None and kind not in (Kind.ACTION, Kind.THOUGHT):
            raise JournalError(
                "действие пишется записью вида ACTION, а воображаемое — THOUGHT. "
                "Смешивать их в одном виде записи нельзя: тогда из журнала не "
                "отличить сделанное от продуманного")

        e = Entry(seq=self._seq, kind=kind, stamp=stamp, actor=actor,
                  actor_layer=actor_layer,
                  profile_hash=self._profile.profile_hash,
                  structure_hash=self._profile.structure_hash,
                  lineage_id=self.meta.lineage_id, branch_id=self.meta.branch_id,
                  wall_clock=time.time() if wall_clock is None else float(wall_clock),
                  state=state or EMPTY_STATE,
                  frame=frame, audio=audio, action=action, perception=perception,
                  event=dict(event or {}), prev=self._last)
        e = dataclasses.replace(e, digest=entry_digest(e.payload(), e.prev))
        self._fh.write(e.as_line() + "\n")
        self._fh.flush()
        self._seq += 1
        self._last = e.digest
        self._last_stamp = stamp
        return e

    # --- чтение ------------------------------------------------------------

    def __iter__(self) -> Iterator[Entry]:
        path = self.path / self.ENTRIES
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Entry.from_line(line)
                except Exception as exc:
                    raise TamperError(f"{path}:{lineno}: запись не читается: {exc}") from exc

    def __len__(self) -> int:
        return self._seq

    def entries(self, kinds: Iterable[Kind] | None = None) -> Iterator[Entry]:
        want = set(kinds) if kinds else None
        for e in self:
            if want is None or e.kind in want:
                yield e

    def frames(self) -> Iterator[Entry]:
        return self.entries([Kind.FRAME])

    def compatible(self) -> Iterator[Entry]:
        """Только записи, совместимые с текущей структурой (0.5)."""
        h = self._profile.structure_hash
        for e in self:
            if e.structure_hash == h:
                yield e

    # --- проверка ----------------------------------------------------------

    def verify(self) -> None:
        """Пересчитать цепочку и монотонность. Расхождение — TamperError."""
        prev = GENESIS
        expect_seq = 0
        last: Stamp | None = None
        seen_structures: set[str] = set()
        for e in self:
            if e.seq != expect_seq:
                raise TamperError(f"запись {e.seq}: ожидался seq {expect_seq} — строку удалили или вставили")
            if e.prev != prev:
                raise TamperError(f"запись {e.seq}: prev {e.prev[:8]} вместо {prev[:8]} — цепочка порвана")
            recomputed = entry_digest(e.payload(), e.prev)
            if recomputed != e.digest:
                raise TamperError(
                    f"запись {e.seq}: хеш не сходится ({e.digest[:8]} против {recomputed[:8]}) "
                    "— содержимое записи правили после записи")
            if last is not None:
                if e.stamp.t_self < last.t_self:
                    raise TamperError(f"запись {e.seq}: t_self пошёл назад")
                if e.stamp.t_world < last.t_world:
                    raise TamperError(f"запись {e.seq}: t_world пошёл назад")
            if not e.profile_hash or not e.structure_hash:
                raise TamperError(f"запись {e.seq}: нет штампа профиля (инвариант 2)")
            # `wall_clock` намеренно не проверяется на монотонность: он не часы, а
            # якорь. На разных машинах и после правки системного времени он ходит
            # назад, и требовать от него порядка значило бы ронять целый журнал
            # из-за NTP. Порядок задают три часов, и они проверены выше.
            seen_structures.add(e.structure_hash)
            prev, expect_seq, last = e.digest, e.seq + 1, e.stamp
        if len(seen_structures) > 1:
            raise TamperError(
                f"в одной ветке {len(seen_structures)} разных structure_hash: {sorted(seen_structures)}. "
                "Структурные переключатели обязаны форкать журнал (инвариант 11)")

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        first: Stamp | None = None
        last: Stamp | None = None
        masked = 0
        for e in self:
            by_kind[str(e.kind)] = by_kind.get(str(e.kind), 0) + 1
            if first is None:
                first = e.stamp
            last = e.stamp
            if e.action is not None and e.action.masked:
                masked += 1
        return {"branch": self.meta.branch_id, "entries": self._seq, "by_kind": by_kind,
                "masked_actions": masked,
                "profile_hash": short(self._profile.profile_hash),
                "structure_hash": short(self._profile.structure_hash),
                "first": str(first) if first else None, "last": str(last) if last else None}

    # --- жизненный цикл ----------------------------------------------------

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None
        self.mode = "r"

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class LegacyJournal:
    """Ветка формата v1. Только чтение, только `LegacyEntry`.

    Существует ради одного: чтобы прежний корпус не выбрасывался и при этом не
    притворялся новым. Открыть его можно, дописать — нельзя, и записи из него не
    имеют полей, которых в них не было.

    `require(field)` — то, чем пользуется код метрик. Он спрашивает у журнала, есть
    ли у него нужное поле, и получает отказ с внятной причиной вместо нуля.
    """

    #: Поля, которых в формате v1 нет, и что именно из-за этого нельзя посчитать.
    MISSING = {
        "actor_layer": "слой-инициатор не записывался: атрибуция действий, метрика "
                       "конфабуляции и единство «я» неизмеримы",
        "state": "среза состояния не записывалось: ошибка предсказания, драйвы, "
                 "настроение и переходы целей неизмеримы",
        "stated_reason_id": "реплики планировщика не записывались: сопоставлять "
                            "заявленную причину не с чем",
        "lineage_id": "линий ещё не было: принадлежность записи линии неизвестна",
        "wall_clock": "якоря реального времени не записывалось",
    }

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        meta_path = self.path / Journal.META
        if not meta_path.exists():
            raise JournalError(f"нет ветки журнала: {meta_path}")
        self.meta = BranchMeta.from_dict(
            json.loads(meta_path.read_text(encoding="utf-8")))
        if not self.meta.is_v1:
            raise FormatError(
                f"ветка {self.meta.branch_id} в формате {self.meta.format}, а не v1. "
                "Открывайте её обычным Journal — иначе потеряете слой-инициатор")
        self.mode = "r"

    def require(self, *fields: str) -> None:
        """Отказаться, если запрошенная метрика опирается на поля v2."""
        absent = [f for f in fields if f in self.MISSING]
        if absent:
            why = "; ".join(f"{f} — {self.MISSING[f]}" for f in absent)
            raise FormatError(
                f"ветка {self.meta.branch_id} записана в формате v1, "
                f"запрошено {absent}: {why}. Замер на этом корпусе поставить "
                "нельзя. Нужен прогон в формате v2")

    def __iter__(self) -> Iterator[LegacyEntry]:
        path = self.path / Journal.ENTRIES
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield LegacyEntry.from_line(line)
                except FormatError:
                    raise
                except Exception as exc:
                    raise TamperError(
                        f"{path}:{lineno}: запись не читается: {exc}") from exc

    def entries(self, kinds: Iterable[Kind] | None = None) -> Iterator[LegacyEntry]:
        want = set(kinds) if kinds else None
        for e in self:
            if want is None or e.kind in want:
                yield e

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        n = 0
        for e in self:
            by_kind[str(e.kind)] = by_kind.get(str(e.kind), 0) + 1
            n += 1
        return {"branch": self.meta.branch_id, "format": self.meta.format,
                "entries": n, "by_kind": by_kind, "read_only": True,
                "missing_fields": sorted(self.MISSING)}


def open_branch(path: str | Path) -> Journal | LegacyJournal:
    """Открыть ветку тем читателем, который соответствует её формату.

    Формат берётся из данных, а не из аргумента: иначе однажды кто-то откроет v1
    как v2, получит отказ и добавит «на всякий случай» значение по умолчанию.
    """
    meta_path = Path(path) / Journal.META
    if not meta_path.exists():
        raise JournalError(f"нет ветки журнала: {meta_path}")
    meta = BranchMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
    return LegacyJournal(path) if meta.is_v1 else Journal(path, mode="r")


def branch_chain(root: str | Path) -> list[BranchMeta]:
    """Все ветки журнала по порядку появления, с родителями и точками форка."""
    root = Path(root)
    out: list[BranchMeta] = []
    d = root / "branches"
    if not d.is_dir():
        return out
    for p in sorted(x for x in d.iterdir() if x.is_dir()):
        meta = p / Journal.META
        if meta.exists():
            out.append(BranchMeta.from_dict(json.loads(meta.read_text(encoding="utf-8"))))
    return out
