"""Бюджет кадра и стоимость стадий записи. `TASK-18`, пункты 3 и 4.

Два разных вопроса, которые до этого путались.

**Первый: успеваем ли.** Ответ даёт сравнение работы на оборот с **бюджетом кадра** —
`1000 / capture_fps` миллисекунд. Прежний вывод сравнивал стадии между собой («ждал кадр
3.1, писал 0.8 → узкое место в захвате») и потому находил узкое место всегда: у двух чисел
одно непременно больше. 3.1 мс при бюджете 33 мс — это десятикратный запас, а не узкое
место.

**Второй: сколько стоит кадр, который изменился целиком.** Загадка 6.5 кадр/с не решена, а
обойдена: во втором прогоне оператора экран почти не менялся (271 отметка «без изменений» из
300 оборотов), и 14 КиБ на кадр не имеют отношения к 1.37 МиБ из первого. Условия первого
прогона надо воспроизвести, а не ждать, когда они повторятся сами. Здесь для этого есть
`scan`: кадры с **объявленной долей изменения** — от неподвижного до полностью иного, — и
разбивка стоимости на каждой. Прогоняется где угодно, дисплей не нужен, потому что мерится
не захват, а то, что делают с кадром после него.

Один источник на замер и на команду: `tools/measure_record_cost.py` и `harness cost`
зовут отсюда. Две копии этого кода разошлись бы молча — как трижды разошлось знание о
Wayland (TASK-15).
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Случаи, на которых снимается стоимость кадра. Набор закрыт и **объявлен**: каждый
#: назван тем, что он делает с кадром, а не тем, сколько процентов «должно» измениться.
#:
#: `shift` — содержимое сдвигается по горизонтали, как при прокрутке, видео и движении
#: камеры. Доля — какая часть высоты кадра сдвигается. Настоящая доля изменившихся пикселей
#: **меньше** заявленной, потому что однотонные области сдвигаются сами в себя; поэтому
#: измеренная доля печатается рядом, а не подменяется заявленной.
#:
#: `noise` — несжимаемый предел: каждый пиксель случаен. На экране такого не бывает, и
#: строка подписана именно так. Она нужна ровно затем, что у оператора вышло 1.37 МиБ на
#: кадр — то есть его кадры сжимались **почти как шум**, и без верхней границы это число
#: было бы не с чем сравнить.
CASES: tuple[dict[str, Any], ...] = (
    {"kind": "shift", "fraction": 0.0,
     "meaning": "экран не менялся: запись неподвижности"},
    {"kind": "shift", "fraction": 0.01,
     "meaning": "мигает курсор или часы: почти неподвижный экран"},
    {"kind": "shift", "fraction": 0.25,
     "meaning": "движется окно или часть страницы"},
    {"kind": "shift", "fraction": 1.0,
     "meaning": "сдвинут весь кадр: видео на весь экран, прокрутка, смена сцены"},
    {"kind": "noise", "fraction": 1.0,
     "meaning": "шум: несжимаемый предел, на экране не встречается"},
)


#: Размеры кадра, между которыми выбирает оператор. TASK-21, часть 3: захват окна — это
#: **рычаг бюджета**, и вопрос «во сколько раз он дешевле» имеет численный ответ.
#:
#: Набор закрыт и каждый размер значит что-то конкретное, а не «поменьше»: сравнение
#: «1920×1080 против 960×540» отвечает на вопрос оператора (окно в пол-экрана по каждой
#: стороне — это вчетверо меньше пикселей), а промежуточные размеры показывают, линейно ли
#: падает стоимость по пикселям или нет. Без промежуточных пришлось бы верить, что падает
#: линейно, а этого никто не мерил.
FRAME_SIZES: tuple[dict[str, Any], ...] = (
    {"width": 1920, "height": 1080, "meaning": "монитор 1080p целиком"},
    {"width": 1600, "height": 900, "meaning": "окно почти во весь экран"},
    {"width": 1280, "height": 720, "meaning": "обычное окно браузера"},
    {"width": 960, "height": 540, "meaning": "окно в половину стороны: вчетверо пикселей"},
    {"width": 640, "height": 360, "meaning": "маленькое окно: девятая часть пикселей"},
)


def budget_ms(fps: float) -> float:
    """Бюджет одного оборота при целевой частоте. Миллисекунды."""
    return 1000.0 / float(fps) if fps > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Verdict:
    """Успеваем ли в бюджет кадра — и **насколько**, а не «кто из стадий больше»."""

    busy_ms: float
    budget_ms: float
    worst_stage: str
    worst_ms: float

    @property
    def fits(self) -> bool:
        return self.budget_ms <= 0 or self.busy_ms <= self.budget_ms

    @property
    def slack_ms(self) -> float:
        return self.budget_ms - self.busy_ms

    @property
    def max_fps(self) -> float | None:
        return (1000.0 / self.busy_ms) if self.busy_ms > 0 else None

    def text(self) -> str:
        """Одна строка оператору. Сначала ответ, потом число."""
        if self.budget_ms <= 0:
            return (f"работа на оборот {self.busy_ms:.1f} мс; цель частоты не задана, "
                    "сравнивать не с чем")
        if self.fits:
            return (f"укладываемся в бюджет: {self.busy_ms:.1f} мс работы при бюджете "
                    f"{self.budget_ms:.1f} мс, запас {self.slack_ms:.1f} мс")
        return (f"бюджет кадра превышен: {self.busy_ms:.1f} мс работы при бюджете "
                f"{self.budget_ms:.1f} мс — не хватает {-self.slack_ms:.1f} мс. "
                f"Больше всего съедает «{self.worst_stage}» ({self.worst_ms:.1f} мс), "
                f"потолок такой записи "
                + (f"{self.max_fps:.1f} кадр/с" if self.max_fps else "неизвестен"))

    def as_dict(self) -> dict[str, Any]:
        return {"busy_ms": round(self.busy_ms, 3), "budget_ms": round(self.budget_ms, 3),
                "fits": self.fits, "slack_ms": round(self.slack_ms, 3),
                "worst_stage": self.worst_stage, "worst_ms": round(self.worst_ms, 3),
                "max_fps": None if self.max_fps is None else round(self.max_fps, 1)}


def verdict(stages: dict[str, float], *, fps: float) -> Verdict:
    """Сравнить работу на оборот с бюджетом кадра.

    `stages` — миллисекунды на оборот по стадиям (`Turns.stages()`). Ожидание между
    оборотами (`idle_ms`) в работу **не входит**: это добровольная пауза, чтобы не
    обогнать цель, и включать её значило бы объявлять узким местом собственное ожидание.
    """
    work = {k: float(v) for k, v in stages.items()
            if k.endswith("_ms") and k not in ("idle_ms",)}
    busy = sum(work.values())
    worst = max(work.items(), key=lambda kv: kv[1]) if work else ("нет стадий", 0.0)
    names = {"capture_ms": "ожидание кадра от экрана", "record_ms": "запись кадра",
             "audio_ms": "звук"}
    return Verdict(busy_ms=busy, budget_ms=budget_ms(fps),
                   worst_stage=names.get(worst[0], worst[0]), worst_ms=worst[1])


# ---------------------------------------------------------------------------
# Кадры с объявленной долей изменения
# ---------------------------------------------------------------------------


def desktop_frames(n: int, *, width: int, height: int, seed: int = 17) -> list[np.ndarray]:
    """Кадры синтетического рабочего стола: обои, окна, сглаженный текст.

    Однотонный или случайный кадр не годятся в обе стороны: первый сжимается в сотни раз,
    второй не сжимается вовсе, и ни один не похож на экран. TASK-09 дал по этому источнику
    65.8 КиБ на кадр против 49.3 на живом экране — расхождение известно и невелико.
    """
    from .core.profile import from_schema
    from .corpus.domains import make_domain

    p = from_schema("СТОИМОСТЬ", capture_width=width, capture_height=height)
    d = make_domain("desktop", p, seed=seed)
    return [np.ascontiguousarray(d.step(None, with_audio=False).frame) for _ in range(n)]


def with_change(base: list[np.ndarray], fraction: float, *,
                seed: int = 18) -> list[np.ndarray]:
    """Из кадров рабочего стола сделать последовательность с заданной долей изменения.

    Изменение вносится **сдвигом содержимого**, а не случайным шумом: шум несжимаем и дал
    бы завышенную стоимость, которой на экране не бывает. Сдвиг — это то, что делают
    прокрутка, видео и движение камеры, то есть настоящий источник полного изменения кадра.

    `fraction = 0` — тот же кадр (запись неподвижности), `1.0` — сдвинут весь кадр.
    """
    if not base:
        return []
    rng = np.random.default_rng(seed)
    h, w = base[0].shape[:2]
    rows = int(round(h * max(0.0, min(1.0, float(fraction)))))
    out = [base[0]]
    cur = base[0]
    for i in range(1, len(base)):
        nxt = cur.copy()
        if rows:
            src = base[i % len(base)]
            shift = int(rng.integers(3, 17))
            band = np.roll(src[:rows], shift, axis=1)
            nxt[:rows] = band
        out.append(nxt)
        cur = nxt
    return out


def noise_frames(n: int, *, width: int, height: int, seed: int = 19) -> list[np.ndarray]:
    """Кадры случайного шума: верхняя граница стоимости, а не модель экрана.

    Подписана как граница везде, где печатается. Шум не сжимается, поэтому даёт худшее
    время сжатия и худший объём; на настоящем экране так не бывает, но **близко к тому**
    бывает: у оператора вышло 1.37 МиБ на кадр 1080p, то есть сжатие в полтора раза.
    """
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (height, width), dtype=np.uint8) for _ in range(n)]


def measured_change(frames: list[np.ndarray]) -> float:
    """Какая доля пикселей действительно менялась. Заявленное надо проверять.

    Доля объявляется полосой строк, но настоящая доля отличается: сдвиг однотонной области
    не меняет ни одного пикселя. Печатается измеренное, а не заявленное, — иначе таблица
    врёт о своём же условии.
    """
    if len(frames) < 2:
        return 0.0
    diffs = [float((a != b).mean()) for a, b in zip(frames, frames[1:])]
    return sum(diffs) / len(diffs)


def stage_costs(frames: list[np.ndarray], *, level: int, keyframe: int,
                shard_bytes: int = 64 << 20) -> dict[str, Any]:
    """Прогнать кадры через настоящее хранилище и снять стоимость стадий."""
    from .core.blobstore import FrameStore

    with tempfile.TemporaryDirectory(prefix="harness-cost-") as tmp:
        store = FrameStore(Path(tmp) / "frames", mode="a", shard_bytes=shard_bytes,
                           compress_level=level, keyframe_interval=keyframe)
        t0 = time.perf_counter()
        for f in frames:
            store.append_frame(f)
        wall = time.perf_counter() - t0
        store.close()
        c = store.cost
        n = max(1, c.calls)
        per_frame_ms = c.total_ns / n / 1e6
        return {
            "compress_level": level, "keyframe_interval": keyframe,
            "frames": c.calls, "keyframes": c.keyframes,
            "delta_ms": c.delta_ns / n / 1e6,
            "compress_ms": c.compress_ns / n / 1e6,
            "write_ms": c.write_ns / n / 1e6,
            "total_ms": per_frame_ms,
            "wall_ms": wall / n * 1e3,
            "kib_per_frame": c.bytes_out / n / 1024,
            "ratio": c.bytes_in / c.bytes_out if c.bytes_out else None,
            "max_fps": (1000.0 / per_frame_ms) if per_frame_ms else None,
        }


def scan_sizes(*, frames: int, level: int, keyframe: int, fps: float,
               change: float = 1.0, sizes: tuple[dict[str, Any], ...] = FRAME_SIZES,
               seed: int = 17) -> list[dict[str, Any]]:
    """Стоимость кадра по размеру области. Ответ на «во сколько раз дешевле окно».

    Доля изменения одна для всех размеров и по умолчанию **полная**: сравнивать размеры
    надо в том случае, где бюджет и трещит. На неподвижном экране любой размер дешёв, и
    сравнение размеров там ничего не показало бы — это было бы вакуумно по свойству случая
    (инвариант 27), а не по нехватке данных.

    Возвращает строки с долей от самого большого размера **по двум величинам сразу**:
    по пикселям и по стоимости. Их отношение и есть ответ на вопрос, линейно ли падает
    стоимость: одна доля без другой позволяет считать, что линейно, не проверив.
    """
    rows: list[dict[str, Any]] = []
    for size in sizes:
        w, h = int(size["width"]), int(size["height"])
        base = desktop_frames(frames, width=w, height=h, seed=seed)
        seq = with_change(base, float(change), seed=seed + 1)
        got = stage_costs(seq, level=level, keyframe=keyframe)
        got.update({"width": w, "height": h, "pixels": w * h,
                    "meaning": size["meaning"],
                    "change_declared": change,
                    "change_measured": measured_change(seq),
                    "budget_ms": budget_ms(fps),
                    "fits_budget": got["total_ms"] <= budget_ms(fps) if fps > 0 else None})
        rows.append(got)
    if rows:
        big = rows[0]
        for r in rows:
            r["pixel_share"] = r["pixels"] / big["pixels"]
            r["cost_share"] = (r["total_ms"] / big["total_ms"]) if big["total_ms"] else None
            r["bytes_share"] = (r["kib_per_frame"] / big["kib_per_frame"]
                                if big["kib_per_frame"] else None)
            # Во сколько раз дешевле — то самое число, за которым шли. Печатается как
            # «раз», а не как доля: оператор спрашивает «во сколько раз», и переводить
            # 0.27 в 3.7 в голове — лишний шаг, на котором ошибаются.
            r["cheaper_times"] = (big["total_ms"] / r["total_ms"]
                                  if r["total_ms"] else None)
    return rows


#: Два рычага бюджета на одной опоре. TASK-21, часть 3: гипотеза задачи была, что
#: уменьшить область может оказаться дешевле, чем понизить уровень сжатия, — и это
#: проверяемо в одном прогоне, если мерить оба одинаково.
#:
#: Уровень 0 стоит здесь как **ловушка среды**, и это надо читать вместе с оговоркой.
#: В контейнере он выходит самым быстрым по времени (медиана 2.7 мс против 21.0 на уровне
#: 6, три сида) — потому что запись идёт в файловую систему контейнера, где два мегабайта
#: стоят ~1.2 мс. На диске оператора те же 2025 КиБ на кадр при 30 кадр/с — это 59 МиБ/с
#: непрерывной записи, и там рычаг развернётся. Время стадии «запись», снятое здесь, к
#: диску оператора отношения не имеет; отношения между **уровнями сжатия** и между
#: **размерами** — имеет, потому что они про процессор.
LEVER_CASES: tuple[dict[str, Any], ...] = (
    {"lever": "ничего", "width": 1920, "height": 1080, "level": 6,
     "label": "1080p, уровень 6"},
    {"lever": "сжатие", "width": 1920, "height": 1080, "level": 1,
     "label": "1080p, уровень 1"},
    {"lever": "сжатие", "width": 1920, "height": 1080, "level": 0,
     "label": "1080p, без сжатия"},
    {"lever": "область", "width": 960, "height": 540, "level": 6,
     "label": "окно 960×540, уровень 6"},
    {"lever": "оба", "width": 960, "height": 540, "level": 1,
     "label": "окно 960×540, уровень 1"},
)


def levers(*, frames: int, keyframe: int,
           cases: tuple[dict[str, Any], ...] = LEVER_CASES,
           seed: int = 17) -> list[dict[str, Any]]:
    """Уменьшить область против понизить сжатие. Одна доля изменения, одни кадры.

    Сравнение честно только при одинаковом содержимом, поэтому кадры для каждого размера
    строятся одним и тем же генератором с одним сидом, а меняются ровно две величины:
    размер и уровень. Первая строка — опора, от неё считается «во сколько раз дешевле».
    """
    rows: list[dict[str, Any]] = []
    for case in cases:
        w, h = int(case["width"]), int(case["height"])
        seq = with_change(desktop_frames(frames, width=w, height=h, seed=seed), 1.0,
                          seed=seed + 1)
        got = stage_costs(seq, level=int(case["level"]), keyframe=keyframe)
        got.update({"lever": case["lever"], "label": case["label"],
                    "width": w, "height": h, "pixels": w * h})
        rows.append(got)
    if rows:
        base = rows[0]["total_ms"]
        for r in rows:
            r["cheaper_times"] = (base / r["total_ms"]) if r["total_ms"] else None
    return rows


def scan(*, width: int, height: int, frames: int, level: int, keyframe: int,
         fps: float, cases: tuple[dict[str, Any], ...] = CASES,
         seed: int = 17) -> list[dict[str, Any]]:
    """Стоимость кадра в каждом объявленном случае.

    Это и есть ответ на «почему у меня 6.5 кадр/с»: у неподвижного экрана и у экрана,
    меняющегося целиком, стоимость различается на порядок, и оба случая надо видеть рядом.
    """
    base = desktop_frames(frames, width=width, height=height, seed=seed)
    rows = []
    for case in cases:
        if case["kind"] == "noise":
            seq = noise_frames(frames, width=width, height=height, seed=seed + 2)
        else:
            seq = with_change(base, float(case["fraction"]), seed=seed + 1)
        got = stage_costs(seq, level=level, keyframe=keyframe)
        got["kind"] = case["kind"]
        got["change_declared"] = case["fraction"]
        got["change_measured"] = measured_change(seq)
        got["meaning"] = case["meaning"]
        got["budget_ms"] = budget_ms(fps)
        got["fits_budget"] = got["total_ms"] <= budget_ms(fps) if fps > 0 else None
        rows.append(got)
    return rows
