"""Живые записи с машины оператора: приём и прогон по ним того же кода.

`TASK-03-MEASUREMENT-LOOPS.md`, часть 2. Записывает оператор — в контейнере нет ни
дисплея, ни звукового устройства, ни игры. Анализ остаётся здесь, и главное
требование к нему одно: **тот же самый код**, не новый и не подправленный.

## Зачем это вообще нужно

Все числа разделения себя и мира сняты на синтетических мирах, и пересчёт по
независимым единицам показал, что за «IoU 0.97» стоит **одна** независимая
единица: маска истины побитово одинакова при любом сиде. Хуже того, генератор и
детектор писались вместе, поэтому 0.97 отражает согласие кода с самим собой.
Живая запись — первый способ это проверить, и падение метрики на ней **ожидаемо и
является результатом, а не провалом**.

## Чего на живой записи нет, и это надо сказать прямо

**Истины.** На синтетике маска обрамления известна точно (`load_hud_mask`), на
записи настоящего экрана её нет ни в каком виде. Значит IoU и точность **не
считаются вовсе**, пока оператор не разметит хотя бы одну область — и это не
придирка: посчитать IoU против маски, полученной тем же детектором, значило бы
сравнить метод с самим собой и получить 1.00 при любом качестве.

Поэтому здесь два разных отчёта:

- **без разметки** — только то, что считается без истины: какая доля пикселей
  вообще решена, расходятся ли признаки между собой, устойчива ли маска во
  времени. Это не качество, это признаки жизни;
- **с разметкой** — IoU, полнота, точность, сравнимые с синтетическими.

Разметка живёт в **отладочном канале** (`debug/truth.jsonl` рядом с записью), а не
в журнале: это истина исследователя, и агентскому коду она недоступна по построению
(инвариант 12). Её формат — прямоугольники, потому что размечать вручную пиксели
никто не будет, и прямоугольник честнее сглаженной обводки.

## Что делает приём

`ingest` берёт каталог сессии, записанной где-то ещё, и:

1. проверяет цепочку хешей — запись, доехавшая испорченной, дальше не идёт;
2. отказывается от формата v1 громко (в нём нет слоя-инициатора);
3. отказывается принимать синтетическую запись как живую — иначе через месяц
   никто не отличит одно от другого;
4. кладёт её в корпус **как есть**, ничего не пересчитывая и не нормируя.

Ни `ingest`, ни `bench_live` не имеют ни одного параметра, подкручивающего
детектор. Если для живых записей понадобится другой порог, это будет новая строка
в `Profile` и форк журнала, а не поправка на месте.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..core.journal import FORMAT, BranchMeta, FormatError, Journal, TamperError
from ..paths import live_path, show
from ..session import SESSION_META, Session, SessionMeta

#: Шесть записей из задания. Набор закрыт: каждая проверяет своё, и «ещё одна
#: похожая» ничего не добавляет, а «без одной» оставляет дыру в сравнении.
KINDS: dict[str, dict[str, str]] = {
    "stillness": {
        "title": "неподвижность",
        "duration": "60 с",
        "why": "фоновое изменение экрана без участия. Опорный уровень: всё, что "
               "меняется здесь, меняется само, и детектор обязан это знать",
        # Порядок действий, а не пожелание: первый замер оператора дал 300 изменившихся
        # кадров и **ни одной** отметки «без изменений» при 403 МиБ за 10 с. Опорный
        # уровень мерил мигающий курсор в открытом окне записи, а не фон экрана.
        "how": "запустить, свернуть окно записи, увести курсор в угол и не трогать мышь; "
               "не смотреть на экран, чтобы не переключать окна. Если после записи "
               "отметок «без изменений» ноль — в кадре что-то мигало, и опорного уровня "
               "не получилось",
    },
    "camera_only": {
        "title": "только камера",
        "duration": "3 мин",
        "why": "чистый параллакс без перемещения. Калибровка экранного слоя: "
               "обрамление стоит, мир смещается",
        "how": "вращать камеру мышью, не идти",
    },
    "play": {
        "title": "обычная игра",
        "duration": "15–20 мин",
        "why": "основной материал",
        "how": "играть как обычно",
    },
    "menu": {
        "title": "меню и инвентарь",
        "duration": "3 мин",
        "why": "экранный слой без мирового: проверка, не выучил ли детектор "
               "«обрамление там, где ничего не движется»",
        "how": "открыть меню, инвентарь, карту; полистать",
    },
    "death": {
        "title": "смерть и возрождение",
        "duration": "до первой смерти",
        "why": "разрыв поля зрения, опустевший инвентарь, экран смерти как класс "
               "контекста",
        "how": "играть до смерти и нажать возрождение самому",
    },
    "unfamiliar": {
        "title": "незнакомое приложение",
        "duration": "5 мин",
        "why": "проверка переноса: тот же код на браузере или редакторе. Если "
               "детектор работает только в игре, он выучил игру",
        "how": "открыть браузер или редактор и обычно им пользоваться",
    },
    # Три вида, которым игра не нужна вовсе. Заведены не для удобства: первое
    # сравнение синтетики с живым важнее, чем сравнение именно в игре, а установка
    # игры — это часы работы оператора и ещё один повод не начать.
    "scroll": {
        "title": "прокрутка страницы",
        "duration": "3 мин",
        "why": "содержимое едет, обрамление стоит — та же задача, что «только "
               "камера» в игре, но без игры. Здесь эго-движения нет вообще, и "
               "детектор, выучивший «движется — значит мир», обязан споткнуться",
        "how": "открыть длинную страницу и катать её колесом вверх-вниз, окно не "
               "двигать",
    },
    "video": {
        "title": "видео в браузере",
        "duration": "3 мин",
        "why": "содержимое идёт само, без единого действия. Опорный уровень для "
               "фонового изменения: сравнение с нулём объявило бы живыми все "
               "выходы подряд, и на замере в домене «видео» так и вышло",
        "how": "включить любое видео на весь кадр и не трогать ни мышь, ни клавиши",
    },
    "windows": {
        "title": "перетаскивание окон",
        "duration": "3 мин",
        "why": "несколько областей двигаются независимо друг от друга. Метод стоит "
               "на допущении «у мира одно движение», и здесь допущение ломается "
               "нарочно — как в мире с глубиной, но на настоящем экране",
        "how": "открыть два-три окна и таскать их мышью, меняя размеры",
    },
}

#: Наборы записей. Ключ — имя набора, значение — виды в порядке записи.
#:
#: Разделение появилось затем, что прежний план требовал Minecraft для всех шести
#: записей, а четыре из шести игры не требуют. Для первого честного числа —
#: сравнения синтетики с живым — игра не нужна совсем: нужен экран, браузер и
#: двенадцать минут. Требование установить игру стоило бы недели ожидания там, где
#: достаточно четверти часа.
SETS: dict[str, dict[str, Any]] = {
    "minimal": {
        "title": "минимальный: только браузер и рабочий стол",
        "kinds": ("stillness", "scroll", "video", "windows", "unfamiliar"),
        "needs": "ничего ставить не надо — хватит браузера",
        "total": "около 15 минут записи",
        # Опорное число печатается **функцией**, а не строкой: строка здесь и была тем
        # вторым местом, где оно стояло одиноким числом без единицы усреднения.
        "enough_for": "первое сравнение синтетики с живым: IoU разделения себя и "
                      "мира на настоящем экране против опорного {ref}",
    },
    "full": {
        "title": "полный: с игрой",
        "kinds": ("stillness", "camera_only", "play", "menu", "death", "unfamiliar"),
        "needs": "нужна установленная игра с отладочным экраном (Minecraft: F3)",
        "total": "около 30 минут записи плюс установка игры",
        "enough_for": "всё, что даёт минимальный, плюс эго-движение, экранный слой "
                      "интерфейса и смерть как класс контекста",
    },
}


class LiveError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Разметка оператора: истина исследователя, недоступная агенту
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Region:
    """Прямоугольник разметки. Координаты — истина исследователя.

    Именно прямоугольник, а не маска: размечать пиксели вручную никто не будет, а
    попиксельная обводка, сделанная на глаз, была бы точностью, которой нет.
    Прямоугольник признаёт свою грубость и потому не врёт.
    """

    top: int
    left: int
    height: int
    width: int
    what: str = "screen"          # screen | world

    def as_dict(self) -> dict[str, Any]:
        return {"top": self.top, "left": self.left, "height": self.height,
                "width": self.width, "what": self.what}

    @classmethod
    def from_dict(cls, d: dict) -> Region:
        return cls(int(d["top"]), int(d["left"]), int(d["height"]), int(d["width"]),
                   str(d.get("what", "screen")))


def truth_mask(regions: Iterable[Region], shape: tuple[int, int]) -> np.ndarray:
    """Собрать маску экранного слоя из прямоугольников разметки."""
    mask = np.zeros(shape, dtype=bool)
    for r in regions:
        if r.what != "screen":
            continue
        mask[r.top:r.top + r.height, r.left:r.left + r.width] = True
    return mask


def read_regions(root: str | Path) -> list[Region]:
    """Прочитать разметку из отладочного канала записи. Пусто — законно."""
    path = Path(root) / "debug" / "regions.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Region.from_dict(x) for x in data.get("regions", [])]


def write_regions(root: str | Path, regions: Iterable[Region], *,
                  note: str = "") -> Path:
    """Записать разметку. Кладётся в `debug/`, потому что это истина исследователя."""
    d = Path(root) / "debug"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "regions.json"
    path.write_text(json.dumps(
        {"note": note, "regions": [r.as_dict() for r in regions]},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Приём записи
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Ingested:
    """Что приняли и что в этом есть."""

    path: Path
    kind: str
    frames: int
    entries: int
    branch: str
    format: str
    synthetic: bool
    source: str
    has_audio: bool
    has_debug: bool
    has_regions: bool
    actor_layers: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "kind": self.kind, "frames": self.frames,
                "entries": self.entries, "branch": self.branch,
                "format": self.format, "synthetic": self.synthetic,
                "source": self.source, "has_audio": self.has_audio,
                "has_debug": self.has_debug, "has_regions": self.has_regions,
                "actor_layers": self.actor_layers, "warnings": self.warnings}

    def text(self) -> str:
        rows = [f"принято: {self.path}",
                f"  что это: {KINDS.get(self.kind, {}).get('title', self.kind)}",
                f"  кадров {self.frames}, записей {self.entries}, "
                f"ветка {self.branch}, формат {self.format}",
                f"  источник {self.source}, звук {'есть' if self.has_audio else 'нет'}, "
                f"отладочный поток {'есть' if self.has_debug else 'нет'}, "
                f"разметка {'есть' if self.has_regions else 'нет'}",
                f"  слои-инициаторы: {self.actor_layers}"]
        for w in self.warnings:
            rows.append(f"  ! {w}")
        if not self.has_regions:
            rows.append("  IoU по этой записи не посчитается: истины нет. "
                        "Разметьте область обрамления — harness mark")
        return "\n".join(rows)


def ingest(source: str | Path, corpus: str | Path, *, kind: str,
           copy: bool = True) -> Ingested:
    """Принять запись с чужой машины в корпус. Ничего не пересчитывая.

    Проверки идут до копирования: испорченная или не того формата запись не должна
    попадать в корпус вообще, иначе через месяц по ней посчитают число.
    """
    source = Path(source)
    if kind not in KINDS:
        raise LiveError(
            f"неизвестный вид записи {kind!r}. Набор закрыт: {sorted(KINDS)}. "
            "Каждая запись проверяет своё, и «ещё одна похожая» ничего не добавляет")
    meta_path = source / SESSION_META
    if not meta_path.exists():
        raise LiveError(
            f"в {source} нет {SESSION_META}: это не сессия харнесса. Запись должна "
            "быть сделана `harness record`, иначе формат не тот и сверять нечего")
    meta = SessionMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))

    branches = sorted((source / "journal" / "branches").glob("*"))
    if not branches:
        raise LiveError(f"в {source} нет ни одной ветки журнала")
    bmeta = BranchMeta.from_dict(
        json.loads((branches[-1] / Journal.META).read_text(encoding="utf-8")))
    if bmeta.is_v1:
        raise FormatError(
            f"запись {source.name} в формате v1: в её записях нет слоя-инициатора. "
            f"Перезапишите её этой версией харнесса (формат {FORMAT}) — иначе живой "
            "корпус окажется несравнимым с синтетическим по атрибуции действий")
    if meta.synthetic:
        raise LiveError(
            f"запись {source.name} помечена синтетической (`synthetic: true`). "
            "Принять её как живую нельзя: смысл живого корпуса ровно в том, что он "
            "не порождён теми же допущениями, что детектор. Если это правда "
            "синтетика — ей место в другом каталоге")

    warnings: list[str] = []
    # Открытие тоже может упасть на битой строке, и это та же поломка: ловим её
    # здесь, а не даём вывалиться сырым исключением из перечисления.
    try:
        session = Session.open(source)
    except TamperError as e:
        raise LiveError(
            f"запись {source.name} не открывается: {e}. Она доехала испорченной, и "
            "считать по ней нельзя") from None
    with session as s:
        try:
            s.journal.verify()
        except TamperError as e:
            raise LiveError(
                f"цепочка хешей записи {source.name} не сходится: {e}. Запись "
                "доехала испорченной, и считать по ней нельзя") from None
        layers: dict[str, int] = {}
        frames = 0
        entries = 0
        for e in s.journal:
            entries += 1
            layers[str(e.actor_layer)] = layers.get(str(e.actor_layer), 0) + 1
            if e.frame is not None:
                frames += 1
        gaps = sum(1 for e in s.journal if str(e.kind) == "capture_gap")

    if frames == 0:
        raise LiveError(f"в записи {source.name} ни одного кадра: сверять нечего")
    if gaps:
        warnings.append(
            f"разрывов захвата: {gaps}. Это не отказ — по записи с дырами можно "
            "считать, но соседние кадры местами относятся к разным моментам, и "
            "признаки, стоящие на межкадровой разнице, там ослабнут")
    if layers.get("human", 0) == 0:
        warnings.append(
            "ни одной записи со слоем human. Для демонстрации это странно: "
            "действовал оператор, и в журнале это должно быть видно "
            "(`harness record --actor human`)")
    # Наличие файла — не наличие звука. `AudioStore` создаёт индекс при открытии
    # сессии, поэтому пустой файл есть **всегда**, и проверка на существование
    # отвечала «звук есть» на записи без единого блока. Это ровно та подмена
    # реального значения правдоподобным, которую правила проекта запрещают: по такой
    # записи потом искали бы пеленг, которого в ней нет.
    audio_index = source / "audio" / "index.jsonl"
    has_audio = audio_index.exists() and audio_index.stat().st_size > 0
    if not has_audio:
        warnings.append("звука нет: пеленг по разнице каналов на этой записи не "
                        "проверить")
    has_debug = (source / "debug" / "truth.jsonl").exists()
    regions = read_regions(source)

    target = Path(corpus) / f"{kind}-{source.name}"
    if copy:
        if target.exists():
            raise LiveError(
                f"{target} уже есть. Записи не дописываются поверх чужих: если это "
                "новая запись того же вида, дайте её каталогу другое имя")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
    else:
        target = source

    return Ingested(target, kind, frames, entries, bmeta.branch_id, bmeta.format,
                    meta.synthetic, meta.source, has_audio, has_debug,
                    bool(regions), layers, warnings)


# ---------------------------------------------------------------------------
# Прогон того же кода по живой записи
# ---------------------------------------------------------------------------


def _margin() -> float:
    """Запас над тривиальным ответом — из схемы, а не из кода анализа.

    Одно место чтения на два читателя (`LiveScore.beats_trivial` и `refute`): порог,
    прочитанный в двух местах по-разному, — это два порога (инвариант 23).
    """
    from ..core.profile import MILESTONE_0

    return float(MILESTONE_0.parameters["live_iou_margin_over_trivial"])


@dataclass(slots=True)
class LiveScore:
    """Результат по одной живой записи.

    `iou`, `recall`, `precision` — `None`, если разметки нет. Это не ноль и не
    «плохо»: это «сверять не с чем», и подставить сюда число нельзя ничем.
    """

    path: Path
    kind: str
    frames: int
    method: str = "arbiter"       # какой из двух существующих путей прогнан
    decided: float = 0.0          # доля пикселей, про которые метод что-то решил
    screen_share: float = 0.0     # доля пикселей, отнесённых к экранному слою
    # `None` — признак был один, сравнивать его не с чем. Это не согласие.
    disagreement: float | None = None
    decided_by: dict[str, int] = field(default_factory=dict)
    signal: str = ""              # какой признак решал: MERGED, PARALLAX, NO_SIGNAL…
    reason: str = ""
    iou: float | None = None
    recall: float | None = None
    precision: float | None = None
    truth_px: int = 0
    absent_reason: str = ""
    #: IoU тривиального ответа «весь кадр — экранный слой», посчитанный по разметке этой
    #: записи. Пол, который метод обязан перебить: не перебил — не нашёл ничего.
    trivial: float | None = None
    #: Синтетический домен, с которым эта запись сравнима, и его IoU. Сравнивать запись
    #: браузера с медианой по пяти доменам значит сравнивать её в том числе с `depth`.
    matched_domain: str = ""
    matched_iou: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "kind": self.kind, "frames": self.frames,
                "method": self.method,
                "comparable_to": method_comparable(self.method),
                "decided": round(self.decided, 4),
                "screen_share": round(self.screen_share, 4),
                "disagreement": (None if self.disagreement is None
                                 else round(self.disagreement, 4)),
                "decided_by": self.decided_by,
                "signal": self.signal, "reason": self.reason,
                "iou": None if self.iou is None else round(self.iou, 4),
                "recall": None if self.recall is None else round(self.recall, 4),
                "precision": (None if self.precision is None
                              else round(self.precision, 4)),
                "truth_px": self.truth_px,
                "trivial": None if self.trivial is None else round(self.trivial, 4),
                "beats_trivial": self.beats_trivial,
                "matched_domain": self.matched_domain,
                "matched_iou": self.matched_iou,
                "absent_reason": self.absent_reason}

    @property
    def beats_trivial(self) -> bool | None:
        """Перебил ли метод тривиальный ответ. `None` — сравнивать нечем.

        Запас над полом — настройка схемы (`live_iou_margin_over_trivial`), а не число
        здесь: порог, живущий в коде анализа, не попадает в `profile_hash` и не сравним
        между прогонами (инвариант 23).
        """
        if self.iou is None or self.trivial is None:
            return None
        return self.iou >= self.trivial + _margin()

    def text(self) -> str:
        head = (f"{KINDS.get(self.kind, {}).get('title', self.kind)}, путь "
                f"«{self.method}» ({self.frames} кадров)")
        rows = [head,
                f"  решено пикселей: {self.decided:.1%}, "
                f"отнесено к экрану: {self.screen_share:.1%}, "
                + ("расхождение признаков: не считается, признак один"
                   if self.disagreement is None
                   else f"расхождение признаков: {self.disagreement:.2%}"),
               ]
        if self.decided_by:
            rows.append(f"  чем решено: {self.decided_by}")
        if self.signal:
            rows.append(f"  признак: {self.signal} — {self.reason}")
        if self.iou is None:
            rows.append(f"  IoU: не считается — {self.absent_reason}")
        else:
            rows.append(f"  IoU {self.iou:.3f}, полнота {self.recall:.3f}, "
                        f"точность {self.precision:.3f} "
                        f"(истина: {self.truth_px} пикселей разметки)")
        return "\n".join(rows)


#: Два пути разделения слоёв, и оба нужны. Синтетические числа сняты разными из
#: них: «IoU 0.97» — одиночным разделителем по параллаксу на корпусе, медиана 0.667
#: по доменам — арбитром трёх признаков. Прогонять один и сравнивать с числом,
#: снятым другим, значило бы сравнивать разное.
METHODS = ("parallax", "arbiter")


# ---------------------------------------------------------------------------
# Опорное число синтетики: одно, с единицей усреднения при нём. TASK-11, часть 1
# ---------------------------------------------------------------------------
#
# До этой правки в проекте жили **два** опорных числа, и оба назывались «IoU на
# синтетике»: 0.97 и 0.667. Ожидание по живому (`MEASUREMENT.md`, §18) — «около 0.6» с
# границами 0.4 и 0.85 — было откалибровано против 0.97. Против 0.667 то же ожидание
# означает «ничего не изменится», а нижняя граница 0.4 попадает **внутрь** разброса самой
# синтетики: два домена из пяти дают меньше. Ожидание, которое нельзя опровергнуть, —
# не ожидание.

#: Медианы IoU по доменам: пять доменов, шесть сидов, 240 кадров, арбитр трёх признаков.
#: Перемерено при закрытии TASK-11 части 1 и совпало с записанным в `MEASUREMENT.md`.
#:
#: Здесь **числа, а не одно число**, потому что одно число без разброса и породило путаницу:
#: 0.667 при разбросе 0.196–0.815 и 0.667 при разбросе 0.66–0.67 — это разные утверждения,
#: а выглядят одинаково.
SYNTHETIC_IOU: dict[str, float] = {
    "game": 0.815, "desktop": 0.730, "document": 0.667, "video": 0.417, "depth": 0.196,
}

#: Единица усреднения опорного числа. **Домен**, и это установлено замером, а не выбрано:
#: разброс между доменами (sd 0.254) в четыре раза больше разброса внутри домена
#: (sd 0.000–0.062), то есть смена сида не добавляет наблюдения, а смена домена добавляет
#: всё. Инвариант 22: `n` считает единицы, а не события, и событий здесь было 30.
SYNTHETIC_UNIT = "домен"

#: Почему медиана, а не среднее. Не по вкусу: на пяти единицах со тяжёлым низом среднее
#: неустойчиво. Выброс `depth` (0.196) — это домен с тремя планами, где допущение метода
#: «у мира одно движение» ломается нарочно; убрать его из набора значило бы улучшить
#: число, ничего не улучшив в методе. Проверено сдвигом: без `depth` медиана уезжает на
#: +0.031, среднее — на +0.092, то есть среднее в три раза чувствительнее к одному домену.
SYNTHETIC_STATISTIC = "медиана"


def synthetic_reference() -> dict[str, Any]:
    """Опорное число синтетики **вместе с единицей усреднения**. Один источник на проект.

    Возвращает не число, а запись: значение, статистику, единицу, `n` и разброс. Функция,
    отдающая одинокий `float`, — это и есть тот способ, которым «IoU на синтетике» разошлось
    в две разные величины: получатель числа не знал, что он получил.
    """
    import statistics

    values = sorted(SYNTHETIC_IOU.values())
    return {
        "value": round(statistics.median(values), 4),
        "statistic": SYNTHETIC_STATISTIC,
        "unit": SYNTHETIC_UNIT,
        "n": len(values),
        "min": min(values), "max": max(values),
        "sd": round(statistics.stdev(values), 4),
        "by_domain": dict(SYNTHETIC_IOU),
        "method": "arbiter",
        "note": "медиана по доменам; единица усреднения — домен, n=5. Разброс "
                f"{min(values):.3f}–{max(values):.3f} шире самой медианы, и печатать "
                "медиану без него нельзя",
    }


def synthetic_text() -> str:
    """Как опорное число печатается **всегда**: значение, статистика, единица, n, разброс.

    Публиковать «IoU на синтетике 0.667» без единицы усреднения запрещено (проверяется
    тестом `test_iou_is_never_published_without_its_unit`): без единицы это число уже
    один раз означало два разных утверждения.
    """
    r = synthetic_reference()
    return (f"IoU {r['value']:.3f} ({r['statistic']} по {r['n']} доменам, единица "
            f"усреднения — {r['unit']}; разброс {r['min']:.3f}–{r['max']:.3f})")


#: Какому синтетическому домену соответствует живая запись. Сравнивать запись браузера с
#: медианой по всем доменам — значит сравнивать её в том числе с `depth`, миром из трёх
#: планов, которого на экране браузера нет вовсе.
#:
#: Соответствие объявлено здесь и является **частью ожидания**, а не удобством: живая
#: запись — это один домен, а не пять, и её место в разбросе синтетики известно заранее.
DOMAIN_FOR_KIND: dict[str, str] = {
    "stillness": "desktop",
    "scroll": "document",
    "video": "video",
    "windows": "desktop",
    "unfamiliar": "document",
    "camera_only": "game",
    "play": "game",
    "menu": "game",
    "death": "game",
}

#: Число одиночного разделителя по параллаксу на синтетическом корпусе — **с единицей и
#: `n` при нём**, потому что без них оно и означало «шесть наблюдений». Опорным для
#: ожидания не является и никогда не станет: единица одна, и сколько сидов ни прогони,
#: она останется одной — маска истины от сида не зависит.
PARALLAX_REFERENCE: dict[str, Any] = {
    "value": 0.9696, "unit": "генератор", "n": 1, "statistic": "медиана по прогонам",
    "is_reference": False,
    "why_not": "у него одна независимая единица — маска истины побитово одинакова при "
               "любом сиде, поэтому шесть сидов дают одно наблюдение, — а ожидание против "
               "числа с n=1 откалибровать нельзя",
}


def method_comparable(method: str) -> str:
    """С чем сравнимо число, снятое этим путём. Собирается, а не хранится строкой.

    Строка про арбитра содержит опорное число, и хранить её готовой значило бы держать
    второе место, где это число написано, — то есть повторить ровно ту ошибку, из-за
    которой их стало два. По той же причине число параллакса берётся из
    `PARALLAX_REFERENCE`, а не пишется здесь цифрами.
    """
    if method == "parallax":
        r = PARALLAX_REFERENCE
        return (f"сравнимо с IoU {r['value']:.4f} на синтетическом корпусе (единица "
                f"усреднения — {r['unit']}, n={r['n']}). Это число **не** опорное: "
                f"{r['why_not']}")
    return "сравнимо с опорным числом синтетики: " + synthetic_text()


#: Ожидание по живому IoU — **выведенное из замера, а не выбранное**. TASK-11, часть 1.
#:
#: Как получено. Четыре причины, по которым живое должно быть хуже синтетики, названы в
#: `MEASUREMENT.md` §18.2 словами. `tools/measure_live_penalty.py` наносит каждую из них на
#: синтетический кадр и прогоняет тот же `bench_domain`: кадр портится, истина не портится.
#: Под всеми четырьмя сразу от чистого числа домена остаётся медианно 0.762 (по пяти
#: доменам; 0.461…1.666). Ожидание — опорное число, умноженное на эту долю.
#:
#: **Доля, а не разность**: чистое число зависит от числа кадров (140 против 240 в опорном
#: замере), и разность переносилась бы неверно, а доля переносится.
#:
#: Что этот замер **опроверг по дороге**: «живое обязано быть хуже» неверно как общее
#: утверждение. На двух доменах из пяти порча IoU **подняла** — на `depth` в 1.67 раза, на
#: `video` в 1.35, — и механизм понятен: там параллакс либо ломается о три плана, либо
#: отсутствует, и шум, заглушающий его голос, улучшает ответ арбитра. Поэтому потолок
#: обоснован как подозрение, а не как обязанность (см. `refute`).
LIVE_EXPECTATION: dict[str, Any] = {
    "value": 0.51,
    "unit": "домен", "n": 5,
    "ratio": 0.7622, "ratio_range": (0.4606, 1.6656),
    "improved_domains": 2,
    "basis": "опорное число × доля, оставшаяся под всеми четырьмя названными порчами сразу",
    "tool": "tools/measure_live_penalty.py",
    "note": "граница сверху, а не предсказание: живой экран портит картинку и способами, "
            "которых в списке четырёх нет. Опровержение — не отклонение от 0.51, а выход "
            "за пол или потолок (см. refute)",
}


def expectation_text() -> str:
    """Ожидание с его происхождением. Число без происхождения здесь уже стоило калибровки."""
    e = LIVE_EXPECTATION
    ref = synthetic_reference()["value"]
    return (f"ожидание по живому IoU {e['value']:.2f} = {ref:.3f} × {e['ratio']:.3f} "
            f"(единица усреднения — {e['unit']}, n={e['n']}; доля по доменам "
            f"{e['ratio_range'][0]:.3f}…{e['ratio_range'][1]:.3f})")


def trivial_iou(truth: "np.ndarray") -> float:
    """IoU тривиального ответа «весь кадр — экранный слой». Пол, который надо перебить.

    Заменяет прежнюю нижнюю границу 0.4, и по двум причинам. Первая: 0.4 попадала **внутрь**
    разброса синтетики — два домена из пяти дают меньше, — то есть «метод не работает»
    объявлялось бы на значении, которое сама синтетика выдаёт как норму. Вторая: 0.4 —
    число, нарисованное рукой в прозе (инвариант 23), а этот пол **считается** по разметке
    той самой записи, на которой судят.

    «Весь кадр — экран» даёт IoU = |истина| / |кадр|: пересечение равно истине, объединение
    равно кадру. Метод, не перебивший этого, не нашёл ничего: то же число даёт ответ, не
    смотрящий на пиксели вовсе.
    """
    total = int(truth.size)
    return (int(truth.sum()) / total) if total else 0.0


def bench_live(path: str | Path, *, method: str = "arbiter") -> LiveScore:
    """Прогнать по живой записи **тот же** код разделения себя и мира.

    Ни одного параметра, подкручивающего детектор, здесь нет намеренно: сравнение с
    синтетикой имеет смысл только если код один и тот же. Профиль берётся из самой
    записи — тот, которым её писали.

    `method` выбирает не поведение детектора, а **какой из двух существующих путей**
    прогнать: одиночный разделитель по параллаксу или арбитра трёх признаков. Это
    не подкрутка: оба пути уже есть в коде и оба использованы в синтетических
    замерах, просто в разных.
    """
    from ..vision.selfworld import SCREEN, LayerArbiter, SelfWorldSeparator

    if method not in METHODS:
        raise LiveError(f"нет пути {method!r}; есть {list(METHODS)}")
    path = Path(path)
    kind = path.name.split("-", 1)[0]
    with Session.open(path) as s:
        sep = (SelfWorldSeparator(s.profile) if method == "parallax"
               else LayerArbiter(s.profile))
        frames = 0
        shape: tuple[int, int] | None = None
        for _, img in s:
            sep.feed(img)
            frames += 1
            if shape is None:
                shape = (int(img.shape[0]), int(img.shape[1]))
        verdict = sep.result()

    if shape is None:
        raise LiveError(f"в {path} нет кадров")
    found = verdict.pixel_mask(SCREEN)
    summary = verdict.summary()
    # У арбитра трёх признаков сводка богаче, у одиночного разделителя — беднее.
    # Читаем через `get` только те поля, которых у одного из них может не быть, и
    # не подставляем на их место числа: отсутствующее расхождение признаков — это
    # «признак был один», а не «признаки согласились».
    score = LiveScore(
        path=path, kind=kind, frames=frames, method=method,
        decided=float(summary["decided_fraction"]),
        screen_share=float(found.sum()) / float(found.size),
        disagreement=(None if "disagreement_fraction" not in summary
                      else float(summary["disagreement_fraction"])),
        decided_by={k: int(v) for k, v in (summary.get("decided_by") or {}).items()},
        signal=str(summary.get("signal", "")),
        reason=str(summary.get("reason", "")))

    regions = read_regions(path)
    if not regions:
        score.absent_reason = (
            "разметки нет. На живой записи истины не существует, а сравнивать "
            "ответ метода с маской, полученной тем же методом, значит сравнить его "
            "с самим собой и получить единицу при любом качестве")
        return score

    truth = truth_mask(regions, shape)
    if not truth.any():
        score.absent_reason = "разметка есть, но пустая: ни одной области screen"
        return score
    inter = int((found & truth).sum())
    union = int((found | truth).sum())
    score.truth_px = int(truth.sum())
    score.iou = inter / union if union else None
    score.recall = inter / int(truth.sum())
    score.precision = inter / int(found.sum()) if found.any() else None
    score.trivial = trivial_iou(truth)
    score.matched_domain = DOMAIN_FOR_KIND.get(score.kind, "")
    score.matched_iou = SYNTHETIC_IOU.get(score.matched_domain)
    return score


def compare_to_synthetic(live: list[LiveScore], *,
                         synthetic_median: float | None = None) -> dict[str, Any]:
    """Синтетика против живого. Падение — результат, а не провал.

    Считается только по записям, у которых есть разметка: остальные в сравнение не входят и
    перечисляются отдельно, чтобы средняя не улучшалась от того, что неудобные записи нечем
    проверить.

    **Сравнение доменно-сопоставленное.** Каждая живая запись сравнивается с тем
    синтетическим доменом, на который она похожа (`DOMAIN_FOR_KIND`), а не с медианой по
    всем пяти: запись браузера против `depth` — это не сравнение, а разница постановок.
    Медиана по живым записям печатается тоже, но рядом с опорным числом и его единицей.

    `synthetic_median` оставлен ключом для прогонов, где опорное число задают снаружи, но
    по умолчанию берётся из единственного источника: два места, где это число написано, — то,
    с чего началась вся путаница.
    """
    ref = synthetic_reference()
    synthetic = ref["value"] if synthetic_median is None else float(synthetic_median)
    scored = [s for s in live if s.iou is not None]
    unscored = [s.kind for s in live if s.iou is None]
    if not scored:
        return {"comparable": 0, "unscored": unscored,
                "reference": ref, "reference_text": synthetic_text(),
                "verdict": "сравнивать нечего: ни одной размеченной записи. "
                           "IoU на живом корпусе неизвестен, и это единственный "
                           "честный ответ"}
    values = sorted(s.iou for s in scored if s.iou is not None)
    median = values[len(values) // 2]

    matched = []
    for s in scored:
        matched.append({
            "kind": s.kind, "iou": round(float(s.iou), 4),
            "domain": s.matched_domain,
            "domain_iou": s.matched_iou,
            "drop": (None if s.matched_iou is None
                     else round(float(s.iou) - float(s.matched_iou), 4)),
            "trivial": None if s.trivial is None else round(s.trivial, 4),
            "beats_trivial": s.beats_trivial,
        })
    return {
        "comparable": len(scored),
        "unscored": unscored,
        "unit": "сессия",
        "live_median_iou": round(median, 4),
        "live_range": [round(values[0], 4), round(values[-1], 4)],
        "reference": ref,
        "reference_text": synthetic_text(),
        "expectation": dict(LIVE_EXPECTATION),
        "expectation_text": expectation_text(),
        "synthetic_median_iou": round(synthetic, 4),
        "drop": round(synthetic - median, 4),
        "matched": matched,
        "checks": [refute(s) for s in scored],
        "verdict": _verdict(median, synthetic, len(scored)),
    }


#: Исходы предрегистрированной проверки. Их не два: «ожидание не опровергнуто» и
#: «опровергнуто» не покрывают случая, когда сравнивать не с чем, а именно он и был у этого
#: проекта всё время.
#:
#: Пятый исход — **вакуумность по свойству записи** (инвариант 27). Пол считается по разметке
#: и на записи, где обрамление занимает почти весь кадр, оказывается выше потолка: тривиальный
#: ответ «весь кадр — экран» набирает больше, чем лучший синтетический домен. Тогда IoU на
#: этой записи не различает работающий метод от неработающего вовсе, и любое число, снятое с
#: неё, — не результат. Это первое, во что упрётся запись неподвижности рабочего стола, то
#: есть самая первая запись оператора.
OUTCOMES = ("не опровергнуто", "опровергнуто: метод не работает",
            "опровергнуто: подозрительно хорошо", "сравнивать не с чем",
            "вакуумно: тривиальный ответ выше потолка")


def refute(score: LiveScore) -> dict[str, Any]:
    """Проверить предрегистрированное ожидание на одной живой записи.

    **Обе границы выведены из опорного числа, а не нарисованы.**

    - **пол** — тривиальный ответ «весь кадр — экранный слой», посчитанный по разметке
      этой записи, плюс запас из схемы. Прежняя нижняя граница 0.4 попадала внутрь
      разброса синтетики: два домена из пяти дают меньше, то есть «метод не работает»
      объявлялось бы на значении, которое сама синтетика выдаёт как норму;
    - **потолок** — лучший синтетический домен (`max(SYNTHETIC_IOU)` = 0.815). Прежние 0.85
      стояли **выше** любого синтетического домена и потому не отличали «подозрительно
      хорошо» от «как лучшая синтетика».

    Обоснование потолка — **подозрение, а не обязанность**, и это исправлено по замеру.
    Раньше здесь стояло «живое обязано быть хуже по четырём названным причинам». Замер
    порчей (`tools/measure_live_penalty.py`) это опроверг: на двух доменах из пяти те же
    четыре порчи IoU подняли. Обязанности нет; повод искать загрязнение — есть, потому что
    объяснить превосходство над лучшей синтетикой чем-то, кроме сравнения ответа метода с
    маской, полученной тем же методом, пока нечем.

    **Вакуумность проверяется первой** (инвариант 27): если пол с запасом уже выше потолка,
    то никакое значение IoU на этой записи ничего не различает, и объявлять по нему исход
    нельзя — ни «работает», ни «не работает».
    """
    if score.iou is None or score.trivial is None:
        return {"kind": score.kind, "outcome": "сравнивать не с чем",
                "why": score.absent_reason or "IoU не посчитан"}
    ceiling = max(SYNTHETIC_IOU.values())
    margin = _margin()
    if score.trivial + margin >= ceiling:
        return {
            "kind": score.kind, "outcome": "вакуумно: тривиальный ответ выше потолка",
            "iou": round(score.iou, 4), "floor": round(score.trivial + margin, 4),
            "ceiling": round(ceiling, 4),
            "why": (f"обрамление занимает {score.trivial:.1%} кадра, поэтому тривиальный "
                    f"ответ «весь кадр — экран» набирает {score.trivial:.3f}, а с запасом "
                    f"{margin:.2f} это {score.trivial + margin:.3f} — выше потолка "
                    f"{ceiling:.3f}. IoU на такой записи не различает работающий метод от "
                    "неработающего: судить надо по полноте и точности по отдельности либо "
                    "по записи, где мира в кадре больше"),
        }
    if not score.beats_trivial:
        return {
            "kind": score.kind, "outcome": "опровергнуто: метод не работает",
            "iou": round(score.iou, 4), "floor": round(score.trivial, 4),
            "why": (f"IoU {score.iou:.3f} не перебил тривиальный ответ «весь кадр — "
                    f"экран» ({score.trivial:.3f}) с запасом. То же число даёт ответ, не "
                    "смотрящий на пиксели вовсе: переделывать надо то, что на методе "
                    "стоит, а не подкручивать порог"),
        }
    if score.iou > ceiling:
        return {
            "kind": score.kind, "outcome": "опровергнуто: подозрительно хорошо",
            "iou": round(score.iou, 4), "ceiling": round(ceiling, 4),
            "why": (f"IoU {score.iou:.3f} выше лучшего синтетического домена "
                    f"({ceiling:.3f}). Объяснить это чем-то, кроме загрязнения, пока "
                    "нечем, поэтому первое, что проверяется, — не сравнивается ли ответ "
                    "метода с маской, полученной тем же методом (детектор загрязнения, "
                    "постоянная метрика 5). Прежняя формулировка — «живое обязано быть "
                    "хуже» — снята: замер порчей поднял IoU на двух доменах из пяти"),
        }
    dom = score.matched_domain or "—"
    return {
        "kind": score.kind, "outcome": "не опровергнуто",
        "iou": round(score.iou, 4), "floor": round(score.trivial, 4),
        "ceiling": round(ceiling, 4),
        "matched": dom, "matched_iou": score.matched_iou,
        "expected": LIVE_EXPECTATION["value"],
        "why": (f"IoU {score.iou:.3f} между полом {score.trivial:.3f} и потолком "
                f"{ceiling:.3f}"
                + (f"; сопоставимый домен «{dom}» даёт {score.matched_iou:.3f}"
                   if score.matched_iou is not None else "")
                + f"; {expectation_text()}"
                + ". Дальше смотрится, какая из четырёх причин даёт основной вклад"),
    }


def _verdict(live: float, synthetic: float, n: int) -> str:
    """Описать то, что случилось, а не то, чего ждали.

    Падение метрики на живой записи ожидалось — но если она выросла, сводка не имеет права
    говорить «падение ожидалось». Ожидание в такой сводке важнее результата ровно один раз:
    когда оно не подтвердилось.
    """
    diff = live - synthetic
    head = (f"живое {live:.3f} против опорного {synthetic:.3f} "
            f"({SYNTHETIC_STATISTIC} по {len(SYNTHETIC_IOU)} доменам, единица усреднения — "
            f"{SYNTHETIC_UNIT})")
    tail = f" Единица живого — сессия, n={n}."
    if n < 3:
        tail += (" Единиц меньше трёх: это одна-две записи, и по ним нельзя "
                 "заключать ничего, кроме «код прогнался и дал число».")
    # Порог «в пределах разброса» — не 0.02 из головы, а разброс самой синтетики по
    # доменам: 0.196–0.815. Сравнивать разницу с числом, меньшим этого разброса, значит
    # объявлять сдвигом то, что внутри опорной величины неотличимо.
    band = (max(SYNTHETIC_IOU.values()) - min(SYNTHETIC_IOU.values())) / 2
    if diff < -band:
        return (f"{head}: разница {diff:+.3f}. Падение — ожидавшийся исход, и оно больше "
                f"полуразброса синтетики по доменам ({band:.3f}).{tail}")
    if diff > band:
        return (f"{head}: разница {diff:+.3f}. Метрика **выросла** сильнее полуразброса "
                f"синтетики ({band:.3f}), а ожидалось падение. Это требует объяснения, а "
                "не радости: чаще всего так бывает, когда живая запись оказалась проще "
                f"синтетической — например обрамление в ней не анимировано.{tail}")
    return (f"{head}: разница {diff:+.3f}, и она внутри полуразброса синтетики по доменам "
            f"({band:.3f}). Ни падения, ни роста заявить нельзя — опорное число само "
            f"размазано шире этой разницы.{tail}")


def plan_text(set_name: str = "minimal") -> str:
    """Что именно записать оператору. Печатается `harness record --plan`.

    По умолчанию **минимальный** набор, а не полный. Порядок по умолчанию решает,
    начнёт ли оператор вообще: план, который первой строкой требует установить игру,
    откладывается на выходные, а выходные не наступают. Полный набор доступен по
    имени и никуда не делся.
    """
    if set_name not in SETS:
        raise LiveError(f"нет набора {set_name!r}; есть {sorted(SETS)}")
    spec = SETS[set_name]
    kinds = spec["kinds"]
    rows = [f"Набор «{spec['title']}»: {len(kinds)} записей.",
            f"Что нужно: {spec['needs']}.",
            f"Сколько это: {spec['total']}.",
            "Формат один и тот же, обрабатываются одним и тем же кодом.", ""]
    for i, name in enumerate(kinds, 1):
        k = KINDS[name]
        rows.append(f"  {i}. {name:<11} {k['title']} — {k['duration']}")
        rows.append(f"     зачем: {k['why']}")
        rows.append(f"     как:   {k['how']}")
        # `--kind` стоит уже в команде записи, а не только в команде приёма: от вида
        # зависит **канал хода записи**. На записи неподвижности ход уходит в файл, потому
        # что мигающая строка в терминале попала бы в кадр как изменение, а изменение там
        # и есть измеряемая величина.
        # Пути печатаются **раскрытыми и в форме этой системы**. `~/harness-live/...` в
        # `cmd.exe` не работает дважды: тильду он не раскрывает, а наклонные ставит
        # обратные. Оператор скопировал такую строку и получил каталог `~` внутри проекта.
        rows.append(f"     команда: harness record {show(live_path(name))} "
                    f"--actor human --kind {name} "
                    f"--seconds {_seconds_of(k['duration'])}")
        rows.append(f"              harness ingest {show(live_path(name))} "
                    f"--corpus {show(live_path('corpus'))} --kind {name}")
        rows.append("")
    rows += [f"Этого набора достаточно для: "
             f"{spec['enough_for'].format(ref=synthetic_text())}.", ""]
    # Стоимость кадра снимается на **движущемся** экране, и это отдельная строка плана, а
    # не примечание. Первый прогон дал 1.37 МиБ на кадр, второй — 14 КиБ: во втором экран
    # почти не менялся (271 отметка «без изменений» из 300 оборотов), и сравнивать их
    # нельзя. Условия надо воспроизводить, а не ждать, когда они повторятся сами.
    rows += [
        "Про стоимость кадра. Она зависит от того, насколько меняется экран, и на",
        "неподвижном экране ничего не говорит. Два числа на своей машине:",
        "  harness cost                       на синтетических кадрах, дисплей не нужен",
        f"  harness record {show(live_path('video'))} --kind video --seconds 30",
        "                                     на живом экране, меняющемся целиком",
        "В конце каждой записи есть строка «на что ушло время» — сравнивайте её с",
        "бюджетом кадра, а не стадии между собой.",
        "",
    ]
    if set_name == "minimal":
        rows += [
            "Игра для этого не нужна. Полный набор — `harness record --plan "
            "--set full`;",
            "он добавляет эго-движение, интерфейс и смерть, но первое число даёт "
            "уже этот.",
            "",
        ]
    rows += [
        "Важное про IoU: на живой записи истины нет, поэтому IoU не посчитается,",
        "пока хотя бы на одной записи не отмечена область обрамления:",
        "  harness mark ЗАПИСЬ --screen ВЕРХ ЛЕВО ВЫСОТА ШИРИНА",
        "Прямоугольник грубый, и это лучше сглаженной обводки на глаз: он не",
        "притворяется точнее, чем есть.",
        "",
        # Ожидание печатается **до** записи, а не после: число, объявленное после
        # получения результата, ничего не запрещает. И сразу же — где оно неприменимо.
        f"Что ожидается: {expectation_text()}.",
        "Опровержением считается выход за пол (тривиальный ответ «весь кадр — экран»,",
        "считается по вашей же разметке) или за потолок (лучший синтетический домен",
        f"{max(SYNTHETIC_IOU.values()):.3f}). Отклонение внутри этих границ — не результат.",
        "",
        "Отдельно про «неподвижность»: если обрамление занимает почти весь кадр, то",
        "тривиальный ответ набирает больше потолка, и IoU на такой записи не различает",
        "ничего (MEASUREMENT.md, §13.3). Разметить её всё равно стоит — но первое число",
        "даст запись, где мира в кадре много: прокрутка, видео, окна.",
    ]
    if set_name == "full":
        rows += [
            "",
            "Отладочный поток (истинные координаты из отладочного экрана игры) — по",
            "возможности и в отдельный файл: он пишется в debug/ и агентскому коду",
            "недоступен.",
        ]
    return "\n".join(rows)


def _seconds_of(duration: str) -> int:
    """«3 мин» → 180. Нужно затем, чтобы в команде стояло число, а не «N».

    Оператор не должен считать в голове: `--frames N` заставляло умножать минуты на
    частоту кадров, а это ровно то место, где человек ошибается и получает запись
    на четыре секунды.
    """
    digits = "".join(c for c in duration if c.isdigit())
    if not digits:
        return 60          # «до первой смерти» и подобное: минута как минимум
    value = int(digits)
    return value * 60 if "мин" in duration else value
