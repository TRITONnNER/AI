"""Замер по TASK-17, пункт 2: куда уходит время одного кадра записи.

Живая запись на машине оператора дала **6.5 кадр/с** против **32.5** у `selftest` на той
же машине, и объяснения не было ни у кого: «узкое место где-то в записи на диск» — не
диагноз, а пересказ симптома. Прерывания оператора оба раза приходились на
`zlib.compress` — это указание, но не доказательство: в основном потоке прерывание падает
туда, где код проводит больше всего времени, и именно это здесь и проверяется числом.

**Что здесь можно измерить, а что нельзя.** Захвата экрана в контейнере нет, поэтому
стадия «ждал кадр» здесь не мерится вовсе и в таблице отсутствует. Мерится то, что от
экрана не зависит: **разность, сжатие и запись в файл** на кадрах того же размера
(1920×1080, gray8) и того же характера (рабочий стол: обои, окна, сглаженный текст). Если
эти три стадии сами по себе не влезают в период кадра, узкое место найдено без захвата; а
если влезают — значит дело в захвате, и это тоже ответ. Разбивку на машине оператора
печатает сама `harness record`.

**Единица независимости — кадр.** Не прогон: каждый кадр отдаётся сжатию отдельно, и
время на нём — отдельное наблюдение. Не настройка: настроек несколько, и утверждение
делается о кадрах при каждой.

Запуск: `python3 tools/measure_record_cost.py`. Идёт около двух минут. Результат —
`docs/measurements/record_cost.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                              # noqa: E402

from harness.cost import (budget_ms, desktop_frames,             # noqa: E402
                          scan, stage_costs)

#: Кадр записи оператора: монитор целиком, серый. Уменьшать нельзя — стоимость сжатия
#: линейна по числу пикселей, и на 320×180 все настройки покажутся мгновенными.
W, H = 1920, 1080

#: Что перебираем. Уровень сжатия и интервал ключевых кадров — настройки схемы
#: (`frame_compress_level`, `frame_keyframe_interval`), поэтому это не «варианты кода», а
#: точки в пространстве профиля: результат применим правкой профиля, а не патчем.
LEVELS: tuple[int, ...] = (1, 3, 6, 9)
KEYFRAMES: tuple[int, ...] = (30, 300)


def delta_forms(frames: list[np.ndarray]) -> dict[str, float]:
    """Две записи одной и той же разности: прежняя через int16 и нынешняя в uint8.

    Обе времени́м здесь, в одном прогоне и на одних кадрах, а не берём прежнее число из
    истории: тогда сравнение зависело бы от того, на какой машине снимали каждое, а машина
    — главное, что в этом замере меняется.
    """
    pairs = list(zip(frames, frames[1:]))
    t0 = time.perf_counter()
    for a, b in pairs:
        ((a.astype(np.int16) - b.astype(np.int16) + 128) % 256).astype(np.uint8)
    old = (time.perf_counter() - t0) / len(pairs) * 1e3
    t0 = time.perf_counter()
    for a, b in pairs:
        (a - b + np.uint8(128))
    new = (time.perf_counter() - t0) / len(pairs) * 1e3
    return {"int16_ms": old, "uint8_ms": new,
            "speedup": old / new if new else None}


def one_setting(frames: list[np.ndarray], *, level: int, keyframe: int) -> dict[str, Any]:
    """Прогнать кадры через настоящее хранилище с этими настройками.

    Считает `harness.cost.stage_costs` — тот же код, которым считает команда `harness cost`
    на машине оператора. Две копии этого счёта разошлись бы молча, и тогда числа замера и
    числа с его машины перестали бы быть сравнимыми.
    """
    return stage_costs(frames, level=level, keyframe=keyframe)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=60, help="кадров на настройку")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "docs" / "measurements" / "record_cost.json")
    args = ap.parse_args(argv[1:])

    print(f"кадры {W}×{H} рабочего стола, по {args.frames} на настройку…")
    frames = desktop_frames(args.frames, width=W, height=H)

    forms = delta_forms(frames)
    print(f"  разность кадра: через int16 {forms['int16_ms']:.1f} мс, "
          f"в uint8 {forms['uint8_ms']:.1f} мс — быстрее в {forms['speedup']:.0f} раз")

    rows = []
    for keyframe in KEYFRAMES:
        for level in LEVELS:
            row = one_setting(frames, level=level, keyframe=keyframe)
            rows.append(row)
            print(f"  уровень {level}, ключевой каждые {keyframe:>3}: "
                  f"разность {row['delta_ms']:5.1f} мс, сжатие {row['compress_ms']:6.1f}, "
                  f"файл {row['write_ms']:4.1f} → предел {row['max_fps']:5.1f} кадр/с, "
                  f"{row['kib_per_frame']:6.1f} КиБ/кадр")

    # Второй разрез: стоимость при **разной доле изменения экрана**. TASK-18, пункт 4 —
    # без него 6.5 кадр/с и 246 кадр/с с одной машины выглядят загадкой, хотя это просто
    # два разных экрана.
    print("\nстоимость при разной доле изменения экрана (уровень 6):")
    change = scan(width=W, height=H, frames=args.frames, level=6, keyframe=30, fps=30.0)
    for r in change:
        print(f"  изменилось {r['change_measured']:>5.0%}: "
              f"разность {r['delta_ms']:5.1f} мс, сжатие {r['compress_ms']:6.1f}, "
              f"файл {r['write_ms']:5.1f} → предел {r['max_fps']:6.1f} кадр/с, "
              f"{r['kib_per_frame']:7.1f} КиБ/кадр — {r['meaning']}")

    now = [r for r in rows if r["compress_level"] == 6 and r["keyframe_interval"] == 30]
    best = min(rows, key=lambda r: r["total_ms"])
    payload = {
        "unit": "кадр",
        "why_unit": "каждый кадр отдаётся сжатию отдельно, и время на нём — отдельное "
                    "наблюдение. Настроек несколько, и утверждение делается о кадрах "
                    "при каждой",
        "frame": {"width": W, "height": H, "format": "gray8",
                  "source": "синтетический рабочий стол (corpus.domains): TASK-09 дал по "
                            "нему 65.8 КиБ/кадр против 49.3 на живом экране"},
        "not_measured": "захват экрана: в контейнере его нет. Стадия «ждал кадр» "
                        "печатается harness record на машине оператора",
        "operator": {
            "record_fps": 6.5, "selftest_fps": 32.5,
            "record_gib_per_hour": 8.8, "selftest_gib_per_hour": 1.5,
            "note": "числа из отчёта оператора, обе записи на одной машине",
        },
        "delta_forms": forms,
        "change_scan": change,
        "operator_kib_per_frame": 1.37 * 1024,
        "frame_period_ms_at_30fps": budget_ms(30.0),
        "rows": rows,
        "current": now[0] if now else None,
        "cheapest": best,
        "frames_per_setting": args.frames,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    if now:
        cur = now[0]
        print(f"\nсейчас в схеме: уровень 6, ключевой каждые 30 — "
              f"{cur['total_ms']:.1f} мс на кадр, предел {cur['max_fps']:.1f} кадр/с")
        share = cur["compress_ms"] / cur["total_ms"] if cur["total_ms"] else 0
        print(f"доля сжатия в этом времени: {share:.0%}")
        print(f"дешевле всего: уровень {best['compress_level']}, ключевой каждые "
              f"{best['keyframe_interval']} — {best['total_ms']:.1f} мс, предел "
              f"{best['max_fps']:.1f} кадр/с, {best['kib_per_frame']:.1f} КиБ/кадр")
    print(f"записано: {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
