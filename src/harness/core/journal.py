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
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from .action import Action
from .blobstore import BlobRef
from .clocks import Stamp
from .profile import Profile, short
from .symbols import assert_no_plain_text

GENESIS = "0" * 32


class JournalError(RuntimeError):
    pass


class TamperError(JournalError):
    """Журнал правили. Это не восстанавливается — только форк от целой части."""


class Actor(StrEnum):
    AGENT = "agent"
    HUMAN = "human"
    NONE = "none"


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
    NOTE = "note"                    # пометка исследователя, ни на что не влияет


@dataclass(frozen=True, slots=True)
class Entry:
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

    def payload(self) -> dict[str, Any]:
        """Содержимое записи без хешей цепочки — то, что хешируется."""
        d: dict[str, Any] = {
            "seq": self.seq,
            "kind": str(self.kind),
            "stamp": self.stamp.as_dict(),
            "actor": str(self.actor),
            "profile_hash": self.profile_hash,
            "structure_hash": self.structure_hash,
        }
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
        d = json.loads(line)
        return cls(
            seq=int(d["seq"]),
            kind=Kind(d["kind"]),
            stamp=Stamp.from_dict(d["stamp"]),
            actor=Actor(d["actor"]),
            profile_hash=str(d["profile_hash"]),
            structure_hash=str(d["structure_hash"]),
            frame=BlobRef.from_dict(d["frame"]) if d.get("frame") else None,
            audio=BlobRef.from_dict(d["audio"]) if d.get("audio") else None,
            action=Action.from_dict(d["action"]) if d.get("action") else None,
            perception=d.get("perception"),
            event=d.get("event", {}),
            prev=str(d.get("prev", GENESIS)),
            digest=str(d.get("digest", "")),
        )


def entry_digest(payload: dict[str, Any], prev: str) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    h = hashlib.blake2b(digest_size=16)
    h.update(prev.encode("ascii"))
    h.update(b"\x00")
    h.update(blob.encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class BranchMeta:
    branch_id: str
    structure_hash: str
    profile: dict[str, Any]
    parent: str | None
    forked_at_seq: int | None
    forked_at_stamp: dict[str, Any] | None
    fork_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "structure_hash": self.structure_hash,
            "profile": self.profile,
            "parent": self.parent,
            "forked_at_seq": self.forked_at_seq,
            "forked_at_stamp": self.forked_at_stamp,
            "fork_reason": self.fork_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BranchMeta:
        return cls(str(d["branch_id"]), str(d["structure_hash"]), dict(d["profile"]),
                   d.get("parent"), d.get("forked_at_seq"), d.get("forked_at_stamp"),
                   d.get("fork_reason"))


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
               forked_at_stamp: Stamp | None = None) -> Journal:
        root = Path(root)
        existing = sorted((root / "branches").glob("*")) if (root / "branches").is_dir() else []
        path = cls._branch_dir(root, profile, len(existing))
        if path.exists():
            raise JournalError(f"ветка уже существует: {path}")
        path.mkdir(parents=True)
        meta = BranchMeta(path.name, profile.structure_hash, profile.as_dict(), parent,
                          forked_at_seq, forked_at_stamp.as_dict() if forked_at_stamp else None,
                          reason)
        (path / cls.META).write_text(
            json.dumps(meta.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        j = cls(path, mode="a")
        j.append(Kind.BRANCH_START, Stamp(0, 0, None), Actor.NONE,
                 event={"code": "branch_start", "branch": path.name,
                        "parent": parent, "reason": reason,
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
        self.append(Kind.PROFILE_CHANGE, stamp, Actor.HUMAN,
                    event={"code": "fork_out", "reason": reason,
                           "to_structure_hash": new_profile.structure_hash,
                           "diff": self._profile.diff(new_profile)})
        self.close()
        return Journal.create(self.path.parent.parent, new_profile, reason=reason,
                              parent=self.meta.branch_id, forked_at_seq=self._seq,
                              forked_at_stamp=stamp)

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
        self.append(Kind.PROFILE_CHANGE, stamp, Actor.HUMAN,
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
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                e = Entry.from_line(line)
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

    def append(self, kind: Kind, stamp: Stamp, actor: Actor, *,
               frame: BlobRef | None = None, audio: BlobRef | None = None,
               action: Action | None = None, perception: dict[str, Any] | None = None,
               event: dict[str, Any] | None = None) -> Entry:
        if self.mode != "a":
            raise JournalError("журнал открыт только на чтение")
        if not isinstance(stamp, Stamp):
            raise JournalError("запись без трёх часов невозможна (инвариант 2)")
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
                  profile_hash=self._profile.profile_hash,
                  structure_hash=self._profile.structure_hash,
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
