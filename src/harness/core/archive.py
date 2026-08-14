"""Холодное хранилище следа: сегменты уезжают и возвращаются. TASK-32, направление B.

При резерве 20 ГБ след кончается за ~2000 часов, и сейчас система останавливает запись и
докладывает. Это правильно (инвариант 14: нехватка места никогда не запускает удаление
следа), но это тупик, а не путь. Путь — **вывоз**: старый сегмент уходит в холодное
хранилище и возвращается по запросу.

## Чем вывоз отличается от понижения и от удаления

Три разные операции, и путать их нельзя ни в коде, ни в отчёте:

| Операция | Что происходит с содержимым | Обратимо |
|---|---|---|
| понижение (`demote`) | подробности **выброшены**: кадры стали статистикой | нет |
| вывоз (`archive`) | содержимое цело, лежит в другом месте | да, `restore` |
| удаление | содержимого нет | **запрещено** (инвариант 14) |

Вывоз **не теряет ничего**, поэтому применим и к нулевому уровню, которому понижение
недоступно. Именно это и делает его путём: место освобождается, а след остаётся целым.

## Что остаётся в горячем хранилище

Запись индекса: идентификатор сегмента, его уровень, адрес по содержимому, размер и **где
именно** он теперь лежит. Ссылка из журнала продолжает работать: читатель зовёт `open`, и
если сегмент вывезен, хранилище возвращает его, привезя обратно. Разница видна вызывающему
только по времени и по полю `cold` в отчёте — намеренно: код, который не знает про вывоз,
не должен от него ломаться.

## Чего здесь нет

**Удаления.** Ни одной ветки. `Archive` умеет положить и достать; функции «выбросить» не
существует, и это проверяется тестом, а не соглашением.

**Догадок о том, куда вывозить.** Каталог холодного хранилища задаётся снаружи. Выбирать за
оператора, какой диск считать холодным, значило бы решать за него, где у него место.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .levels import Level, LevelError, Placement, content_id


class ArchiveError(LevelError):
    pass


@dataclass(frozen=True, slots=True)
class ColdRef:
    """Где лежит вывезенный сегмент и что о нём известно, не привозя его.

    Размер и адрес по содержимому остаются в горячем индексе нарочно: «сколько места
    освободит вывоз» и «тот ли это сегмент» должны отвечаться без обращения к холодному
    хранилищу, иначе учёт места потребует того самого места.
    """

    segment_id: str
    level: Level
    content: str
    nbytes: int
    where: str                   # путь внутри холодного хранилища
    reason: str                  # почему вывезен: обязательно, как у закладки

    def as_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "level": int(self.level),
                "content": self.content, "nbytes": self.nbytes,
                "where": self.where, "reason": self.reason}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ColdRef":
        return cls(str(d["segment_id"]), Level(int(d["level"])), str(d["content"]),
                   int(d["nbytes"]), str(d["where"]), str(d.get("reason", "")))


class Archive:
    """Холодное хранилище: положить, достать, сказать, что где лежит.

    Индекс держится в холодном каталоге и **дублируется** горячим индексом хранилища
    сегментов: холодный каталог может быть отключён (внешний диск вынули), и тогда система
    обязана знать, что сегмент существует и где он, а не забыть о нём.
    """

    INDEX = "cold-index.json"

    def __init__(self, cold_root: str | Path) -> None:
        self.root = Path(cold_root)
        self.refs: dict[str, ColdRef] = {}
        self._load()

    # --- индекс -------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.root / self.INDEX

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.refs = {k: ColdRef.from_dict(v) for k, v in raw.get("refs", {}).items()}

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"refs": {k: v.as_dict() for k, v in self.refs.items()}}
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.index_path)

    @property
    def available(self) -> bool:
        """Доступно ли холодное хранилище прямо сейчас (диск подключён)."""
        return self.root.exists()

    def has(self, segment_id: str) -> bool:
        return segment_id in self.refs

    def bytes_cold(self) -> int:
        return sum(r.nbytes for r in self.refs.values())

    # --- вывоз и возврат ----------------------------------------------------

    def put(self, placement: Placement, payload: bytes, *, reason: str) -> ColdRef:
        """Вывезти сегмент. `reason` обязателен: вывоз без причины — потерянный сегмент.

        Содержимое сверяется по адресу: если байты не те, вывоз отказывается. Иначе
        холодное хранилище тихо накопило бы не то, что заявлено, и обнаружилось бы это
        через месяцы, при первом возврате.
        """
        if not reason:
            raise ArchiveError(
                f"вывоз сегмента {placement.segment_id[:8]} без причины. Причина "
                "обязательна по той же причине, что у закладки: сегмент, про который "
                "неизвестно, зачем он уехал, никто не привезёт обратно")
        got = content_id(payload)
        if got != placement.content:
            raise ArchiveError(
                f"содержимое сегмента {placement.segment_id[:8]} не сходится с адресом: "
                f"заявлено {placement.content[:12]}, посчитано {got[:12]}")
        where = f"{placement.segment_id[:2]}/{placement.segment_id}.seg"
        target = self.root / where
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        ref = ColdRef(placement.segment_id, placement.level, placement.content,
                      len(payload), where, reason)
        self.refs[ref.segment_id] = ref
        self._save()
        return ref

    def get(self, segment_id: str) -> bytes:
        """Привезти сегмент обратно. Отказ громкий и говорит, что делать."""
        ref = self.refs.get(segment_id)
        if ref is None:
            raise ArchiveError(
                f"сегмент {segment_id[:8]} в холодном хранилище не числится. Это не "
                "«его нет»: возможно, он никогда не вывозился — смотрите горячий индекс")
        if not self.available:
            raise ArchiveError(
                f"холодное хранилище {self.root} недоступно: подключите его. Сегмент "
                f"{segment_id[:8]} существует и не потерян — его просто нечем достать "
                f"сейчас (заявлено {ref.nbytes} байт по адресу {ref.content[:12]})")
        path = self.root / ref.where
        if not path.exists():
            raise ArchiveError(
                f"сегмент {segment_id[:8]} числится в холодном хранилище, но файла "
                f"{ref.where} нет. Это порча хранилища, а не нехватка места: индекс и "
                "содержимое разошлись, и придумать содержимое нечем")
        payload = path.read_bytes()
        got = content_id(payload)
        if got != ref.content:
            raise ArchiveError(
                f"привезённый сегмент {segment_id[:8]} не тот: адрес {ref.content[:12]}, "
                f"а содержимое даёт {got[:12]}")
        return payload

    def forget(self, *args: Any, **kw: Any) -> None:
        """Такой операции нет и не будет. Инвариант 14.

        Метод существует ровно затем, чтобы попытка удалить след упиралась в объяснение, а
        не в отсутствие имени: `AttributeError` читался бы как «не дописали».
        """
        raise ArchiveError(
            "удаления следа не существует. Вывоз освобождает место, ничего не теряя "
            "(`put`/`get`); понижение выбрасывает подробности и применимо к уровням 1–3; "
            "удаление запрещено инвариантом 14 и не реализовано ни для какого уровня")

    def report(self) -> dict[str, Any]:
        by_level: dict[str, int] = {}
        for r in self.refs.values():
            by_level[r.level.title] = by_level.get(r.level.title, 0) + 1
        return {"root": str(self.root), "available": self.available,
                "segments": len(self.refs), "bytes": self.bytes_cold(),
                "by_level": by_level,
                "reasons": sorted({r.reason for r in self.refs.values()}),
                "unit": "сегмент"}


def oldest_first(placements: Iterable[Placement],
                 *, keep_hot: int = 0) -> Iterator[Placement]:
    """Кого вывозить первым: самых старых, кроме последних `keep_hot`.

    «Старее» здесь — по порядку появления в индексе, а не по времени записи: времени у
    сегмента может не быть вовсе, а порядок есть всегда. Последние `keep_hot` остаются
    горячими нарочно: свежий след читают чаще всего, и вывозить его — платить временем за
    место без нужды.
    """
    items = list(placements)
    if keep_hot:
        items = items[:-keep_hot] if keep_hot < len(items) else []
    yield from items
