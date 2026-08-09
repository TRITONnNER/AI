"""Четыре уровня журнала и адресация сегментов по содержимому.

По `STORAGE.md`. Главное положение оттуда, из которого выводится всё остальное:

    Инвариант «журнал вечен» относится к причинной записи, а не к пикселям.
    Пиксели — сенсорная роскошь, они вытесняются. След того, что происходило и
    почему, не вытесняется никогда.

| Уровень | Что | Порядок объёма в час | Что теряется при сбросе |
|---|---|---|---|
| 0 след | действия, часы, профиль, инициатор, драйвы, ошибка | ~1 МБ | ничего |
| 1 признаки | эмбеддинги, сущности, маски, поток | ~100 МБ | нельзя перевоспринять |
| 2 ключевые кадры | кадры вокруг заметных событий | ~125 МБ | нельзя посмотреть |
| 3 поток | видео и звук целиком | 2–4 ГБ | нельзя пересмотреть подряд |

**Нулевой уровень живёт в отдельном зарезервированном пространстве, куда механизм
вытеснения не имеет доступа на запись.** Это архитектурное разделение, а не
политика: единственная по-настоящему необратимая операция во всей системе —
удаление журнала, и она не должна иметь возможности случиться из-за заполнения
диска. Здесь это выражено тем, что `Levels` вообще не знает пути к нулевому уровню:
у него нет ссылки, которую можно было бы передать вытеснению. Проверяется тестом.

**Адресация по содержимому.** Сегмент уровней 1–3 лежит по хешу своего содержимого.
Следствия ровно те, что в документе: ветки от общего предка делят общий префикс
физически, точка ветвления не стоит почти ничего, дедупликация встроена, а не
прикручена. Проверка — в тесте: ветка от середины журнала занимает место,
пропорциональное расхождению, а не длине журнала.

**Понижение, а не удаление.** Сегмент едет по лестнице `поток → ключевые кадры →
признаки → след`. Каждый шаг освобождает много и теряет мало. Сегмент, опущенный до
следа, всё ещё существует и всё ещё участвует в причинной истории — поэтому
`SegmentRef` несёт уровень, до которого сегмент опущен, и ссылка на понижённый
сегмент это нормальное состояние, а не ошибка.

Чего здесь нет: вытеснения, закладок, насыщения по sigma. Только структура, которая
их допускает. Ранжирование и выбор, что понижать, — следующая задача.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterator, Mapping


class LevelError(RuntimeError):
    pass


class ReserveError(LevelError):
    """Попытка тронуть нулевой уровень механизмом, которому туда нельзя."""


class Level(IntEnum):
    """Уровень журнала. Порядок значим: **понижение уменьшает номер**.

    Названия и номера по `STORAGE.md`. `TRACE = 0` не случайно ноль: он и есть
    журнал, остальное — сенсорная надстройка над ним, и чем больше номер, тем
    больше подробностей и тем дороже они стоят. Поэтому лестница вытеснения ведёт
    от трёх к нулю, а не наоборот.
    """

    TRACE = 0        # след: причинная запись
    FEATURES = 1     # признаки: эмбеддинги, сущности, маски, поток
    KEYFRAMES = 2    # ключевые кадры вокруг заметных событий
    STREAM = 3       # поток: видео и звук целиком

    @property
    def title(self) -> str:
        return {0: "след", 1: "признаки", 2: "ключевые кадры", 3: "поток"}[int(self)]

    @property
    def evictable(self) -> bool:
        """Можно ли понижать. Нулевой — нельзя, и это не настройка."""
        return self is not Level.TRACE


# Лестница понижения: с какого уровня на какой едет сегмент.
LADDER = {Level.STREAM: Level.KEYFRAMES,
          Level.KEYFRAMES: Level.FEATURES,
          Level.FEATURES: Level.TRACE}


def content_id(payload: bytes) -> str:
    """Адрес сегмента — хеш его содержимого, и ничего кроме.

    Ни времени, ни счётчика, ни имени ветки. Именно поэтому два одинаковых
    сегмента в двух ветках — это один сегмент на диске, а не два.
    """
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


@dataclass(frozen=True, slots=True)
class SegmentRef:
    """Ссылка из записи журнала на сегмент уровней 1–3. **Неизменяема навсегда.**

    Здесь лежит устойчивая личность сегмента (`segment_id`) и уровень, на котором
    он родился, — и больше ничего. Текущего уровня здесь нет намеренно, и это не
    упущение, а следствие инварианта 1.

    Почему так. Понижение меняет содержимое сегмента: вместо десяти тысяч байт
    потока остаётся восемьсот байт ключевых кадров. При настоящей адресации по
    содержимому у нового содержимого **другой адрес**. Если бы ссылка в записи
    журнала была этим адресом, понижение обязано было бы её переписать — а журнал
    дозаписывается и никогда не редактируется. Ссылка молча перестала бы
    разрешаться, то есть получилась бы ровно та тихая деградация, которой в этом
    проекте быть не должно.

    Поэтому личность и адрес разделены: `segment_id` (хеш содержимого при
    рождении) пишется в журнал один раз и живёт вечно, а куда он сейчас указывает
    и на каком уровне — спрашивается у `SegmentStore`, который ведёт свой
    дозаписываемый индекс размещений.
    """

    segment_id: str              # устойчивая личность: хеш содержимого при рождении
    born_level: Level            # с чего начинал
    born_nbytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "born_level": int(self.born_level),
                "born_nbytes": self.born_nbytes}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SegmentRef:
        return cls(str(d["segment_id"]), Level(int(d["born_level"])),
                   int(d.get("born_nbytes", 0)))


@dataclass(frozen=True, slots=True)
class Placement:
    """Где сегмент лежит **сейчас**. Живёт в индексе хранилища, не в журнале.

    `level` — уровень, до которого сегмент опущен на данный момент. Ссылка на
    понижённый сегмент законна: `STORAGE.md` прямо говорит, что это нормальное
    состояние, а не ошибка. Читатель обязан посмотреть на уровень и понять, что
    именно он получит, а не предполагать, что там кадр.
    """

    segment_id: str
    level: Level
    content: str                 # адрес текущего содержимого по содержимому
    nbytes: int
    born_level: Level

    def __post_init__(self) -> None:
        if self.level > self.born_level:
            raise LevelError(
                f"сегмент {self.segment_id[:8]} заявлен на уровне «{self.level.title}», "
                f"а родился на «{self.born_level.title}»: подробностей стало больше, "
                "чем было записано. Понижение идёт только вниз, и восстановить "
                "выброшенное нечем — придумать его значило бы придумать наблюдение")

    @property
    def demoted(self) -> bool:
        return self.level != self.born_level

    @property
    def readable(self) -> bool:
        """Есть ли ещё что читать. Опущенный до следа — только запись о том, что был."""
        return self.level is not Level.TRACE

    def ref(self) -> SegmentRef:
        """Ссылка для журнала: только то, что не меняется."""
        return SegmentRef(self.segment_id, self.born_level, self.nbytes
                          if not self.demoted else 0)

    def as_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "level": int(self.level),
                "content": self.content, "nbytes": self.nbytes,
                "born_level": int(self.born_level), "demoted": self.demoted}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Placement:
        return cls(str(d["segment_id"]), Level(int(d["level"])),
                   str(d["content"]), int(d["nbytes"]),
                   Level(int(d["born_level"])))


@dataclass(frozen=True, slots=True)
class Ceiling:
    """Потолок хранилища. Применяется к уровням 1–3 и только к ним.

    Числа по `STORAGE.md`, раздел 5: 80 % — фоновое понижение при следующем сне,
    95 % — блокируется запись уровней 2–3. Резерв нулевого уровня выделяется
    отдельно и в потолок не входит.
    """

    cap_mb: float
    demote_at: float = 0.80
    block_at: float = 0.95
    reserve_mb: float = 0.0      # резерв нулевого уровня, вне потолка

    def state(self, used_mb: float) -> str:
        if self.cap_mb <= 0:
            return "без потолка"
        share = used_mb / self.cap_mb
        if share >= self.block_at:
            return "запись уровней 2–3 заблокирована"
        if share >= self.demote_at:
            return "фоновое понижение при следующем сне"
        return "в норме"

    def blocks(self, level: Level, used_mb: float) -> bool:
        """Блокируется ли запись этого уровня прямо сейчас.

        Нулевой уровень не блокируется **никогда**: при исчерпании резерва
        система остановит запись и сообщит, но это решается не здесь, а в
        `resources.py`, и всё равно не удалением.
        """
        if level is Level.TRACE:
            return False
        if self.cap_mb <= 0:
            return False
        return used_mb / self.cap_mb >= self.block_at and level >= Level.KEYFRAMES

    def hours_left(self, used_mb: float, mb_per_hour: float) -> float | None:
        """Сколько часов записи осталось при текущих настройках.

        Проценты ни о чём не говорят, часы говорят обо всём (`STORAGE.md`, раздел
        5). `None` — когда скорость записи неизвестна: выдумывать её нельзя.
        """
        if mb_per_hour <= 0 or self.cap_mb <= 0:
            return None
        return max(0.0, (self.cap_mb - used_mb) / mb_per_hour)

    @classmethod
    def from_profile(cls, profile: Any) -> Ceiling:
        """Все четыре числа из профиля: захардкоженных констант поведения нет."""
        p = profile.parameters
        return cls(cap_mb=float(p["levels_cap_mb"]),
                   demote_at=float(p["levels_demote_fraction"]),
                   block_at=float(p["levels_block_fraction"]),
                   reserve_mb=float(p["trace_reserve_mb"]))


class SegmentStore:
    """Сегменты уровней 1–3, адресуемые по содержимому, плюс индекс размещений.

    Раскладка блоков: `<корень>/blobs/<первые два знака хеша>/<хеш>`. **Без уровня
    в пути** — иначе один и тот же байт-в-байт блок, попавший на два уровня, лежал
    бы дважды, и дедупликация перестала бы быть встроенной.

    Индекс: `<корень>/placements.jsonl`, одна строка на размещение, только
    дозапись. Текущее размещение сегмента — последняя строка про него. Понижение
    **дописывает** строку, а не правит прежнюю: история того, как сегмент ехал по
    лестнице, — тоже данные, и по ней потом считается «сожаление о вытеснении».

    Дозапись и только дозапись, как и в журнале. Функции «переписать» здесь нет.
    """

    INDEX = "placements.jsonl"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._now: dict[str, Placement] = {}
        self._load()

    # --- индекс -------------------------------------------------------------

    def _load(self) -> None:
        path = self.root / self.INDEX
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    p = Placement.from_dict(json.loads(line))
                except Exception as exc:
                    raise LevelError(
                        f"{path}:{lineno}: индекс размещений битый: {exc}") from exc
                self._now[p.segment_id] = p

    def _note(self, placement: Placement) -> Placement:
        with (self.root / self.INDEX).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(placement.as_dict(), ensure_ascii=False,
                                sort_keys=True, separators=(",", ":")) + "\n")
        self._now[placement.segment_id] = placement
        return placement

    def _blob(self, content: str) -> Path:
        return self.root / "blobs" / content[:2] / content

    # --- запись -------------------------------------------------------------

    def put(self, level: Level, payload: bytes) -> Placement:
        """Положить новый сегмент. Тот же байт-в-байт блок места не занимает."""
        if level is Level.TRACE:
            raise ReserveError(
                "нулевой уровень не лежит в хранилище сегментов и не адресуется "
                "отсюда. Он в отдельном зарезервированном пространстве, куда "
                "вытеснение не имеет доступа на запись (STORAGE.md, раздел 2)")
        cid = content_id(payload)
        path = self._blob(cid)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        # Личность сегмента — адрес его содержимого при рождении. Значит два
        # одинаковых наблюдения в двух ветках это один сегмент, а не два.
        return self._note(Placement(cid, level, cid, len(payload), level))

    def where(self, ref: SegmentRef) -> Placement:
        """Где сегмент сейчас. Отсюда, а не из записи журнала."""
        p = self._now.get(ref.segment_id)
        if p is None:
            raise LevelError(
                f"сегмент {ref.segment_id[:8]} не значится в индексе размещений. "
                "Это не вытеснение: вытеснение дописывает строку о понижении, а не "
                "убирает сегмент из индекса молча")
        return p

    def get(self, ref: SegmentRef) -> bytes:
        """Прочитать сегмент. Опущенный до следа читать нечем — это отказ."""
        placement = self.where(ref)
        if not placement.readable:
            raise LevelError(
                f"сегмент {ref.segment_id[:8]} опущен до следа: содержимого больше "
                f"нет. Родился на уровне «{ref.born_level.title}». Запись о том, что "
                "он был, осталась — это и есть понижение вместо удаления")
        path = self._blob(placement.content)
        if not path.exists():
            raise LevelError(
                f"сегмент {ref.segment_id[:8]} заявлен на уровне "
                f"«{placement.level.title}», но блока нет. Это не вытеснение — "
                "вытеснение понижает размещение, а не удаляет молча")
        return path.read_bytes()

    def demote(self, ref: SegmentRef, payload: bytes | None = None) -> Placement:
        """Опустить сегмент на одну ступень лестницы.

        `payload` — то, что остаётся на новом уровне: признаки вместо кадров, кадры
        вместо потока. `None` означает «на новом уровне не остаётся ничего», и тогда
        сегмент едет до следа сразу.

        Ссылка в журнале не трогается и трогать её не нужно: она содержит только
        личность сегмента и уровень рождения, а то, куда он опущен, живёт здесь.
        Старый блок удаляется, если на него больше никто не размещён: уровни 1–3 —
        сенсорная роскошь. Причинная запись при этом остаётся целой.
        """
        placement = self.where(ref)
        if not placement.level.evictable:
            raise ReserveError(
                "нулевой уровень не понижается: он и есть причинная запись")
        target = LADDER[placement.level]
        old_content = placement.content
        if target is Level.TRACE or payload is None:
            new = self._note(Placement(ref.segment_id, Level.TRACE, "", 0,
                                       placement.born_level))
        else:
            cid = content_id(payload)
            path = self._blob(cid)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            new = self._note(Placement(ref.segment_id, target, cid, len(payload),
                                       placement.born_level))
        self._drop_unreferenced(old_content)
        return new

    def _drop_unreferenced(self, content: str) -> None:
        """Убрать блок, на который больше никто не размещён.

        Проверка обязательна: адресация по содержимому означает, что на один блок
        могут ссылаться разные сегменты из разных ветвей. Удалить его, не спросив,
        значило бы обрушить чужую ветку — то самое «ветвление делит префикс
        физически», обращённое во вред.
        """
        if not content:
            return
        if any(p.content == content for p in self._now.values()):
            return
        path = self._blob(content)
        if path.exists():
            path.unlink()

    # --- объём --------------------------------------------------------------

    def count_at(self, level: Level) -> int:
        return sum(1 for p in self._now.values() if p.level is level)

    def bytes_at(self, level: Level) -> int:
        """Байты на уровне. Один блок, разделённый двумя сегментами, считается раз."""
        seen: set[str] = set()
        total = 0
        for p in self._now.values():
            if p.level is not level or not p.content or p.content in seen:
                continue
            seen.add(p.content)
            total += p.nbytes
        return total

    def bytes_on_disk(self) -> int:
        d = self.root / "blobs"
        if not d.is_dir():
            return 0
        return sum(p.stat().st_size for p in d.rglob("*") if p.is_file())

    def bytes_total(self) -> int:
        """Только уровни 1–3. Нулевого здесь нет, и спросить его отсюда нельзя."""
        return sum(self.bytes_at(x) for x in (Level.FEATURES, Level.KEYFRAMES,
                                              Level.STREAM))

    def demoted_count(self) -> int:
        return sum(1 for p in self._now.values() if p.demoted)

    def stats(self) -> dict[str, Any]:
        return {"levels": {x.title: {"segments": self.count_at(x),
                                     "bytes": self.bytes_at(x)}
                           for x in (Level.FEATURES, Level.KEYFRAMES, Level.STREAM)},
                "segments": len(self._now),
                "demoted": self.demoted_count(),
                "to_trace": self.count_at(Level.TRACE),
                "bytes_total": self.bytes_total(),
                "bytes_on_disk": self.bytes_on_disk(),
                "note": "нулевого уровня здесь нет: он в отдельном пространстве"}


class TraceReserve:
    """Зарезервированное пространство нулевого уровня.

    Отдельный класс, а не флаг, ровно по причине из `STORAGE.md`: разделение
    должно быть архитектурным. У `SegmentStore` нет ссылки на этот объект и нет
    способа её получить, поэтому механизм вытеснения физически не может тронуть
    причинную запись — не потому, что ему запрещено, а потому, что он про неё не
    знает.

    Единственное, что резерв умеет сообщить, — кончился он или нет. При исчерпании
    правильное поведение — остановить запись и сообщить, а не чистить.
    """

    def __init__(self, root: str | Path, *, reserve_mb: float = 1024.0) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.reserve_mb = float(reserve_mb)

    def used_mb(self) -> float:
        return sum(p.stat().st_size for p in self.root.rglob("*")
                   if p.is_file()) / (1024 * 1024)

    @property
    def exhausted(self) -> bool:
        return self.reserve_mb > 0 and self.used_mb() >= self.reserve_mb

    def hours_left(self, mb_per_hour: float) -> float | None:
        if mb_per_hour <= 0 or self.reserve_mb <= 0:
            return None
        return max(0.0, (self.reserve_mb - self.used_mb()) / mb_per_hour)

    def stats(self) -> dict[str, Any]:
        used = self.used_mb()
        return {"level": Level.TRACE.title, "used_mb": round(used, 3),
                "reserve_mb": self.reserve_mb, "exhausted": self.exhausted,
                "evictable": Level.TRACE.evictable}


# ---------------------------------------------------------------------------
# Ветвление: общий префикс делится физически
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ForkCost:
    """Чем обошлась ветка. Числа измеренные, а не оценённые."""

    shared_segments: int
    shared_bytes: int
    new_segments: int
    new_bytes: int

    @property
    def share_reused(self) -> float:
        total = self.shared_segments + self.new_segments
        return self.shared_segments / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"shared_segments": self.shared_segments,
                "shared_bytes": self.shared_bytes,
                "new_segments": self.new_segments, "new_bytes": self.new_bytes,
                "share_reused": round(self.share_reused, 4)}


def fork_cost(store: SegmentStore, level: Level,
              payloads: list[bytes]) -> ForkCost:
    """Сколько займёт запись этих сегментов, если часть из них уже лежит.

    Это и есть проверка «ветвление делит общий префикс физически»: сегменты,
    совпадающие с уже сохранёнными, не занимают ничего, потому что адрес у них тот
    же. Платишь только за расхождение.

    Числа измеренные: занятое на диске спрашивается до и после.
    """
    before = store.bytes_on_disk()
    shared = shared_bytes = new = new_bytes = 0
    for payload in payloads:
        existed = store._blob(content_id(payload)).exists()
        store.put(level, payload)
        if existed:
            shared += 1
            shared_bytes += len(payload)
        else:
            new += 1
            new_bytes += len(payload)
    grew = store.bytes_on_disk() - before
    if grew != new_bytes:
        raise LevelError(
            f"диск вырос на {grew} байт, а расхождение — {new_bytes}. Либо "
            "дедупликация не сработала, либо она сработала там, где не должна")
    return ForkCost(shared, shared_bytes, new, new_bytes)


def write_manifest(path: str | Path, data: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(dict(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def walk(store: SegmentStore) -> Iterator[Placement]:
    """Все сегменты и их текущие размещения, по личности сегмента."""
    for sid in sorted(store._now):
        yield store._now[sid]


def demotion_history(store: SegmentStore, segment_id: str) -> list[Placement]:
    """Как сегмент ехал по лестнице. Индекс дозаписывается, поэтому история есть.

    Нужна не из любопытства: по ней считается «сожаление о вытеснении» — как часто
    агент пытался открыть то, что сам же понизил.
    """
    path = store.root / SegmentStore.INDEX
    out: list[Placement] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            p = Placement.from_dict(json.loads(line))
            if p.segment_id == segment_id:
                out.append(p)
    return out
