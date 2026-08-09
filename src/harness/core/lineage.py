"""Линия: новая жизнь, а не продолжение той же.

По `STORAGE.md`, раздел 6. Три слова, которые легко перепутать:

| Термин | Что это | Наследование |
|---|---|---|
| **Профиль** | набор настроек с хешем | штампуется в журнал |
| **Ветка** | продолжение той же жизни от момента | наследует всё |
| **Линия** | новая жизнь | наследует то, что явно указано |

Ветка уже есть — это `Journal.fork`. Здесь третье: `new_lineage` создаёт новый
корень журнала. Старые линии остаются нетронутыми, операция **прибавляет, а не
разрушает**.

По умолчанию обнуляется всё: убеждения, карточки, граф мест, карта действий, веса
рефлексов, модельное «я», таблица заземления символов, библиотека навыков. Что
именно переезжает, задаётся манифестом — это не кнопка сброса, а список.

**`as_testimony` — режим по умолчанию для всего, что переезжает между разными
агентами.** Убеждения предка ссылаются на эпизоды, которых у потомка нет, поэтому
входят гипотезами с доверием к источнику и требуют перепроверки. `full` — только
для продолжения того же агента.

## Манифест непереносимого

`new_lineage` не обещает чистоты. Она обнуляет всё, что может, и **печатает список
того, что обнулить не удалось**. Печатает — потому что через полгода, сравнивая
линию 12 и линию 340, надо знать, что именно сравнивается. Самая честная и
неустранимая утечка в этом списке — оператор: он помнит, что сработало, и
вмешается иначе. Лекарство одно — записать протокол вмешательств до старта линии.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

from .journal import FORMAT, FORMAT_V1, Journal, branch_chain
from .profile import Profile, short


class LineageError(RuntimeError):
    pass


class Inherit(StrEnum):
    """Как переезжает один пункт манифеста."""

    FRESH = "fresh"                  # с нуля
    AS_TESTIMONY = "as_testimony"    # гипотезами с доверием к источнику
    FULL = "full"                    # как своё; только для того же агента


# Пункты манифеста — ровно те, что перечислены в `STORAGE.md`, раздел 6. Набор
# закрыт: новый пункт означает новую подсистему, которая умеет переезжать, а не
# свободную строку в конфиге.
MANIFEST_ITEMS = (
    "perception_weights", "reflex_weights", "action_map", "place_graph",
    "beliefs", "skills", "self_model", "symbol_table", "world",
)


@dataclass(frozen=True, slots=True)
class ManifestItem:
    mode: Inherit
    ref: str | None = None           # откуда грузить, если не fresh

    def __post_init__(self) -> None:
        if self.mode is not Inherit.FRESH and not self.ref:
            raise LineageError(
                f"режим {self.mode} без ссылки на источник. «Загрузить» без указания, "
                "что именно, — это не наследование, а надежда")
        if self.mode is Inherit.FRESH and self.ref:
            raise LineageError(
                f"режим fresh со ссылкой {self.ref!r}: либо с нуля, либо из источника")

    def as_dict(self) -> dict[str, Any]:
        return {"mode": str(self.mode), "ref": self.ref}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ManifestItem:
        return cls(Inherit(d["mode"]), d.get("ref"))


FRESH = ManifestItem(Inherit.FRESH)


@dataclass(slots=True)
class LineageManifest:
    """Что переезжает в новую линию, а что начинается с нуля.

    По умолчанию всё `fresh`. Это не осторожность, а определение: линия — новая
    жизнь, и всякое исключение из этого должно быть названо вслух.
    """

    items: dict[str, ManifestItem] = field(default_factory=dict)
    world_inherited: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        unknown = set(self.items) - set(MANIFEST_ITEMS)
        if unknown:
            raise LineageError(
                f"в манифесте неизвестные пункты: {sorted(unknown)}. Набор закрыт: "
                f"допустимы {list(MANIFEST_ITEMS)}")
        for name in MANIFEST_ITEMS:
            self.items.setdefault(name, FRESH)

    def get(self, name: str) -> ManifestItem:
        if name not in self.items:
            raise LineageError(f"нет такого пункта манифеста: {name}")
        return self.items[name]

    @property
    def inherited(self) -> list[str]:
        return sorted(n for n, it in self.items.items()
                      if it.mode is not Inherit.FRESH)

    def as_dict(self) -> dict[str, Any]:
        return {"items": {n: it.as_dict() for n, it in sorted(self.items.items())},
                "world_inherited": self.world_inherited, "note": self.note}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LineageManifest:
        return cls({n: ManifestItem.from_dict(v)
                    for n, v in (d.get("items") or {}).items()},
                   bool(d.get("world_inherited", False)),
                   str(d.get("note", "")))

    def as_text(self) -> str:
        rows = [f"  {n:<20} {self.items[n].mode}"
                + (f" ← {self.items[n].ref}" if self.items[n].ref else "")
                for n in MANIFEST_ITEMS]
        head = "манифест линии (по умолчанию всё с нуля):"
        tail = (f"  мир с историей: {'да' if self.world_inherited else 'нет'}")
        return "\n".join([head, *rows, tail])


# ---------------------------------------------------------------------------
# Манифест непереносимого
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Leak:
    """Одна утечка: что не обнулилось и почему это важно знать."""

    what: str
    value: str
    why: str

    def as_dict(self) -> dict[str, Any]:
        return {"what": self.what, "value": self.value, "why": self.why}


def leaks(*, profile: Profile, world_seed: int | None,
          perception_model: str, code_version: str,
          from_format: str = FORMAT_V1, to_format: str = FORMAT,
          operator: str = "тот же") -> list[Leak]:
    """Что обнулить не удалось. Список из `STORAGE.md`, раздел 6.

    Он печатается всегда, даже когда кажется, что линия чистая. Смысл именно в
    том, чтобы чистой её никто не считал.
    """
    out = [
        Leak("модель восприятия", perception_model,
             "предобученная подложка. Не наследство от агента, но и не пустота: "
             "она уже видела мир, в который агент входит впервые"),
        Leak("сид и состояние мира",
             "новый" if world_seed is None else str(world_seed),
             "мир тот же или похожий; открытие в нём не отличить от повторения, "
             "если сид совпал с прежней линией"),
        Leak("версия кода", code_version,
             "механика изменилась между линиями, и часть разницы в поведении — "
             "это разница кода, а не разница опыта"),
        Leak("хеш профиля", short(profile.profile_hash),
             "настройки поведения. Сравнимы только линии с одним structure_hash"),
        Leak("хеш-функция символов", "стабильна",
             "оставлена стабильной ради сравнимости линий: один и тот же текст "
             "даёт один и тот же символ. Пустой стартует только таблица заземления"),
        Leak("формат журнала", f"{from_format} → {to_format}",
             "записи прежних линий отвечают на меньшее число вопросов. Замер, "
             "требующий полей v2, на них поставить нельзя, а не «можно с оговоркой»"),
        Leak("оператор", operator,
             "самая честная и неустранимая утечка. Ты помнишь, что сработало, и "
             "вмешаешься иначе. Лекарство одно: записать протокол вмешательств до "
             "старта линии и следовать ему"),
    ]
    return out


def leaks_text(items: list[Leak]) -> str:
    rows = []
    for lk in items:
        rows.append(f"  • {lk.what}: {lk.value}")
        rows.append(f"      {lk.why}")
    return "\n".join(["манифест непереносимого — что обнулить не удалось:", *rows])


# ---------------------------------------------------------------------------
# Сама линия
# ---------------------------------------------------------------------------


def lineage_id(profile: Profile, *, seed: int | None, note: str) -> str:
    """Непрозрачный идентификатор линии из того, что её отличает."""
    h = hashlib.blake2b(digest_size=4)
    h.update(profile.structure_hash.encode("ascii"))
    h.update(f"|{seed}|{note}".encode("utf-8"))
    return f"LIN_{h.hexdigest().upper()}"


@dataclass(frozen=True, slots=True)
class Lineage:
    """Заведённая линия: свой корень журнала, свой манифест, свой список утечек."""

    id: str
    root: Path
    manifest: LineageManifest
    leaks: list[Leak]
    parent: str | None
    reason: str
    format: str = FORMAT

    def as_dict(self) -> dict[str, Any]:
        return {"lineage_id": self.id, "root": str(self.root),
                "parent": self.parent, "reason": self.reason,
                "format": self.format,
                "manifest": self.manifest.as_dict(),
                "leaks": [lk.as_dict() for lk in self.leaks]}

    def as_text(self) -> str:
        head = (f"линия {self.id}"
                + (f", от линии {self.parent}" if self.parent else ", первая")
                + f"\n  причина: {self.reason}\n  формат журнала: {self.format}")
        return "\n".join([head, self.manifest.as_text(), leaks_text(self.leaks)])

    MANIFEST_FILE = "lineage.json"


def new_lineage(root: str | Path, profile: Profile, *, reason: str,
                manifest: LineageManifest | None = None,
                parent: str | None = None, world_seed: int | None = None,
                perception_model: str = "нет: в вехе 0 модели восприятия нет",
                code_version: str | None = None,
                operator: str = "тот же") -> tuple[Lineage, Journal]:
    """Завести новую линию: новый корень журнала, ничего не разрушая.

    Возвращает саму линию и открытый на дозапись журнал её первой ветки. Манифест
    и список утечек пишутся рядом с журналом файлом, а не только печатаются: через
    полгода вопрос «что именно сравнивается» будет задан к данным, а не к памяти.

    Смена формата записи — структурный переключатель, поэтому она обязана открывать
    линию, а не продолжать ветку: у записей v1 и v2 разный набор полей, и держать
    их в одной жизни значило бы иметь биографию, половина которой отвечает на
    вопрос об инициаторе, а половина нет.
    """
    from .. import __version__

    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise LineageError(
            f"{root} не пуст. Линия начинается с чистого корня: дописать её в чужой "
            "значило бы смешать две жизни, а операция обязана прибавлять, а не "
            "разрушать прежнее")
    man = manifest or LineageManifest()
    lid = lineage_id(profile, seed=world_seed, note=reason)
    found = leaks(profile=profile, world_seed=world_seed,
                  perception_model=perception_model,
                  code_version=code_version or __version__,
                  operator=operator)
    lin = Lineage(lid, root, man, found, parent, reason)

    root.mkdir(parents=True, exist_ok=True)
    (root / Lineage.MANIFEST_FILE).write_text(
        json.dumps(lin.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    journal = Journal.create(root / "journal", profile,
                             reason=f"новая линия {lid}: {reason}",
                             lineage_id=lid)
    return lin, journal


def read_lineage(root: str | Path) -> Lineage:
    path = Path(root) / Lineage.MANIFEST_FILE
    if not path.exists():
        raise LineageError(f"нет манифеста линии: {path}")
    d = json.loads(path.read_text(encoding="utf-8"))
    return Lineage(str(d["lineage_id"]), Path(d["root"]),
                   LineageManifest.from_dict(d["manifest"]),
                   [Leak(x["what"], x["value"], x["why"]) for x in d["leaks"]],
                   d.get("parent"), str(d.get("reason", "")),
                   str(d.get("format", FORMAT)))


def lineage_branches(root: str | Path) -> list[dict[str, Any]]:
    """Ветки линии с их формой и форматом. Для пульта и для отчёта."""
    return [{"branch": m.branch_id, "format": m.format,
             "lineage": m.lineage_id, "parent": m.parent,
             "forked_at_seq": m.forked_at_seq, "read_only": m.read_only}
            for m in branch_chain(Path(root) / "journal")]
