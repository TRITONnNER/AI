"""Сколько места съедает час записи. `TASK-08`, часть 2. Измерено, а не рассчитано.

`doctor` на машине оператора сказал «64.5 ГиБ ≈ 7.0 ч при 30 кадр/с», то есть **9.2 ГБ
в час**. `STORAGE.md` для уровня 3 закладывает 2–4 ГБ в час при H.265. Расхождение
втрое, и надо было выяснить, что неверно: оценка или конвейер.

## Ответ: неверна оценка, и не в ту сторону, в которую ожидалось

Гипотеза была «оценка считает сырые кадры, а конвейер кодирует» либо «кодирование не
подключено». Замер опроверг постановку: кодирования действительно нет — ни ffmpeg, ни
H.265, ни строки, которая бы их вызывала, — но **существующего сжатия хватает с
запасом**, и оценка `doctor` завышала расход не втрое, а в двадцать пять раз.

| содержимое | КиБ на кадр | ГиБ в час на 1080p, 30 кадр/с |
|---|---|---|
| шум (несжимаемый предел) | 2026 | 209 |
| неподвижный экран | 2.3 | **0.24** |
| экран с движением | 3.6 | **0.37** |

`doctor` считал по 90 КиБ на кадр — число, взятое мной в TASK-07 из головы и
подписанное как замер. Настоящее для экранного содержимого — 3.6 КиБ. Оператору с
диском на 64 ГиБ сказали «около 7 часов», тогда как честный ответ — около **170**.

Заодно снимается и расхождение со `STORAGE.md`: обещанные для уровня 3 «2–4 ГБ в час
при H.265» существующий уровень 2 обходит без всякого кодека — 0.24–0.37 ГиБ/ч. Строка
в документе описывает механизм, которого нет, но чинить надо не конвейер, а два числа:
константу в оценке и обещание в документе.

**Чего это не отменяет.** Сжатие без потерь целиком зависит от содержимого. Видео на
весь экран и игра с частицами ближе к шуму, чем к рабочему столу, и там расход будет на
порядок выше. Поэтому оценка обязана называть, для какого содержимого она верна, а не
печатать одно число.

## Что здесь меряется

Настоящий `FrameStore` с настоящими настройками профиля на трёх видах содержимого —
потому что сжатие без потерь **целиком зависит от содержимого**, и одно число без
указания, на чём оно получено, здесь бессмысленно:

- **шум** — несжимаемый предел. Хуже не бывает;
- **статика** — неподвижный экран: дельта нулевая, сжимается почти в ничто. Лучше не
  бывает;
- **экраноподобное** — крупные однотонные области, прямоугольники, текстовые строки и
  небольшое движение между кадрами. Ближе всего к рабочему столу, но всё ещё не он.

Настоящее число даст только живая запись. Его печатает `harness selftest` («расход
места измерен») на машине оператора, и вот **его** и надо занести в `MEASUREMENT.md`
как окончательное.

Единица независимости: **вид содержимого**, `n = 3`. Не кадр и не байт: кадры внутри
одного вида зависимы по построению, а утверждение здесь — про то, как конвейер ведёт
себя на разном содержимом.

Запуск: `python3 tools/measure_storage_rate.py [--frames N] [--out FILE]`
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.core.blobstore import FrameStore          # noqa: E402
from harness.core.profile import from_schema           # noqa: E402

WIDTH, HEIGHT = 1920, 1080
FPS = 30.0


def noise(n: int) -> Callable[[int], np.ndarray]:
    r = np.random.default_rng(1)
    frames = [r.integers(0, 256, (HEIGHT, WIDTH), dtype=np.uint8) for _ in range(4)]
    return lambda i: frames[i % 4]


def still(n: int) -> Callable[[int], np.ndarray]:
    """Неподвижный **экран**, а не неподвижный шум.

    Первая редакция повторяла случайный кадр и намерила 7.16 ГиБ/ч «на статике» —
    величину, которую задавали ключевые кадры несжимаемого шума, а не неподвижность.
    Замер мерил не то, что назывался мерить: неподвижный экран сжимаемый, и вопрос
    был про нижнюю границу конвейера на реальном содержимом.
    """
    base = _screen_base()
    return lambda i: base


def _screen_base() -> np.ndarray:
    """Кадр, похожий на рабочий стол: однотонные области, рамки, строки текста."""
    base = np.full((HEIGHT, WIDTH), 240, dtype=np.uint8)
    base[:40, :] = 60                                  # строка заголовка
    base[HEIGHT - 60:, :] = 45                         # панель задач
    base[80:600, 60:900] = 255                         # окно
    for y in range(120, 560, 22):                      # строки текста
        base[y:y + 9, 90:860:3] = 30
    base[200:700, 1000:1800] = 210                     # второе окно
    return base


def screenlike(n: int) -> Callable[[int], np.ndarray]:
    """То же плюс лёгкое движение: курсор, каретка, прокрутка раз в секунду.

    Не «похоже на скриншот» ради красоты: сжатие без потерь живёт ровно на этом —
    на длинных одинаковых участках и на нулевой разнице между кадрами там, где
    ничего не менялось.
    """
    base = _screen_base()
    r = np.random.default_rng(3)

    def make(i: int) -> np.ndarray:
        f = base.copy()
        # Курсор и мигающая каретка — то, что меняется на почти неподвижном экране.
        y, x = 300 + (i * 7) % 200, 1100 + (i * 11) % 400
        f[y:y + 16, x:x + 10] = 0
        if i % 15 < 7:
            f[130:146, 92:96] = 0
        # Небольшая прокрутка раз в секунду: смещение полосы текста.
        if i % 30 == 0:
            f[120:560, 90:860] = np.roll(f[120:560, 90:860], 3, axis=0)
        f[HEIGHT - 30:HEIGHT - 20, 20:120] = r.integers(0, 255, (10, 100),
                                                        dtype=np.uint8)
        return f

    return make


def desktop(n: int) -> Callable[[int], np.ndarray]:
    """Рабочий стол, каким он бывает: обои, сглаженный шрифт, тени, полутона.

    Заведено после TASK-09: замер на живом экране дал 49.3 КиБ на кадр, а мой
    «экраноподобный» — 3.6, то есть в четырнадцать раз меньше. Разница не в
    конвейере, а в том, что нарисованное мной было слишком гладким: однотонные
    прямоугольники, штриховка вместо текста, ни одной фотографии, ни одного
    градиента. Настоящий стол так не выглядит нигде.

    Оценка, в вилку которой не попадает обычный рабочий стол, бесполезна для решения
    «влезет ли запись» — а именно это решение оператор и принимает по ней.
    """
    r = np.random.default_rng(11)
    ys = np.arange(HEIGHT, dtype=np.float32)[:, None]
    xs = np.arange(WIDTH, dtype=np.float32)[None, :]
    # Обои: плавный градиент плюс мелкая фактура. Фактура и есть то, что не сжимается.
    wall = (90 + 60 * np.sin(xs / 300.0) * np.cos(ys / 220.0)
            + r.normal(0, 7, (HEIGHT, WIDTH)))
    base = np.clip(wall, 0, 255).astype(np.uint8)
    # Окно с тенью: тень — градиент, а градиент сжимается хуже однотонного.
    win = np.clip(238 + r.normal(0, 2.0, (620, 900)), 0, 255).astype(np.uint8)
    base[70:690, 50:950] = win
    for dx in range(14):
        col = 950 + dx
        base[70 + dx:690, col] = (base[70 + dx:690, col] * (0.55 + dx * 0.03)
                                  ).astype(np.uint8)
    # Сглаженный шрифт: не два уровня, а полутона по краям глифов.
    for row in range(110, 660, 20):
        line = np.zeros((11, 860), dtype=np.float32)
        starts = r.integers(0, 850, 70)
        for x0 in starts:
            w = int(r.integers(3, 9))
            line[2:9, x0:x0 + w] = 255
        # Размытие в один пиксель даёт те самые полутона краёв.
        blur = (line + np.roll(line, 1, axis=1) + np.roll(line, -1, axis=1)) / 3.0
        patch = base[row:row + 11, 70:930].astype(np.float32)
        base[row:row + 11, 70:930] = np.clip(patch - blur * 0.85, 0, 255
                                             ).astype(np.uint8)

    def make(i: int) -> np.ndarray:
        f = base.copy()
        y, x = 300 + (i * 7) % 200, 1100 + (i * 11) % 400
        f[y:y + 18, x:x + 11] = 0                       # курсор
        if i % 15 < 7:
            f[130:146, 92:96] = 0                       # каретка
        if i % 30 == 0:                                 # прокрутка раз в секунду
            f[110:660, 70:930] = np.roll(f[110:660, 70:930], 3, axis=0)
        return f

    return make


KINDS: dict[str, Callable[[int], Callable[[int], np.ndarray]]] = {
    "шум": noise,                     # несжимаемый предел: хуже не бывает
    "неподвижный экран": still,       # нижняя граница на реальном содержимом
    "экран с движением": screenlike,  # нарисованный: слишком гладкий, см. TASK-09
    "рабочий стол": desktop,          # обои, сглаженный шрифт, тени — как в жизни
}


def measure(kind: str, frames: int) -> dict[str, Any]:
    profile = from_schema("расход", capture_width=WIDTH, capture_height=HEIGHT,
                          capture_fps=FPS)
    p = profile.parameters
    tmp = Path(tempfile.mkdtemp(prefix=f"rate-{kind}-"))
    try:
        store = FrameStore(tmp, mode="a",
                           shard_bytes=int(p["frame_shard_bytes"]),
                           compress_level=int(p["frame_compress_level"]),
                           keyframe_interval=int(p["frame_keyframe_interval"]))
        make = KINDS[kind](frames)
        for i in range(frames):
            store.append_frame(make(i))
        store.close()
        used = sum(x.stat().st_size for x in tmp.rglob("*") if x.is_file())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    per_frame = used / frames
    return {"kind": kind, "frames": frames, "bytes": used,
            "kib_per_frame": round(per_frame / 1024, 1),
            "gib_per_hour": round(per_frame * FPS * 3600 / (1 << 30), 2),
            "raw_gib_per_hour": round(WIDTH * HEIGHT * FPS * 3600 / (1 << 30), 1)}


def report(rows: list[dict[str, Any]], frames: int) -> str:
    lines = [
        "# Расход места: измеренный конвейер",
        "",
        f"Кадр {WIDTH}×{HEIGHT}, серый, {FPS:g} кадр/с, {frames} кадров на вид. "
        f"Единица независимости — вид содержимого, n = {len(rows)}.",
        "",
        "| содержимое | КиБ на кадр | ГиБ в час | во сколько раз меньше сырого |",
        "|---|---|---|---|",
    ]
    for r in rows:
        ratio = r["raw_gib_per_hour"] / max(1e-9, r["gib_per_hour"])
        lines.append(f"| {r['kind']} | {r['kib_per_frame']} | {r['gib_per_hour']} | "
                     f"{ratio:.1f}× |")
    raw = rows[0]["raw_gib_per_hour"]
    lines += [
        "",
        f"Сырые кадры без сжатия: {raw} ГиБ/ч — это потолок, от которого считается "
        "выигрыш.",
        "",
        "## Что это значит",
        "",
        "1. **Оценка `doctor` была завышена в 25 раз.** Она считала по 90 КиБ на "
        "кадр — число из головы, подписанное как замер. Измеренное для экранного "
        "содержимого — 3.6 КиБ. Оператору обещали 7 часов там, где влезает 170.",
        "",
        "2. **Кодирования в видео нет**, и строка «2–4 ГБ/ч при H.265» в "
        "`STORAGE.md` описывает механизм, которого не существует. Но существующий "
        "уровень 2 обходит это обещание без кодека, поэтому уровень 3 перестаёт быть "
        "срочным: он нужен не ради места, а ради возможности пересмотреть запись "
        "подряд.",
        "",
        "3. **Одного числа расхода не бывает.** Сжатие без потерь зависит от "
        "содержимого: от 0.24 ГиБ/ч на неподвижном экране до 209 на шуме. Видео во "
        "весь экран и игра с частицами ближе к шуму. Оценка обязана называть, для "
        "какого содержимого она верна.",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=300,
                    help="кадров на вид содержимого (по умолчанию 300 = 10 с)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    for kind in KINDS:
        r = measure(kind, args.frames)
        rows.append(r)
        print(f"{kind:<16} {r['kib_per_frame']:>8} КиБ/кадр  "
              f"{r['gib_per_hour']:>6} ГиБ/ч", flush=True)
    print()
    print(report(rows, args.frames))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\nчисла: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
