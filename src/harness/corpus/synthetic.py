"""Синтетический корпус: записанные сессии с известной истиной.

Зачем он существует. По правилам проекта «корпус записей важнее кода»: любой
модуль обязан тестироваться офлайн на записанных сессиях, без запуска игры. Но
пока нет ни одной записи, тестировать не на чем — а первую запись негде взять,
пока нет харнесса. Синтетика разрывает этот круг: она даёт корпус, у которого
истина известна точно, а не размечена руками.

Что она честно есть и чего честно не заменяет:

- **Есть:** экранный слой, прибитый к экрану, и мировой слой, который смещается
  при повороте камеры. Ровно то различие, которое должен находить 0.6. Истина
  (маска интерфейса, сдвиг камеры на каждом кадре, пеленг звука) пишется в
  отладочный поток, а не в журнал, — значит проверяемый код её не видит.
- **Не заменяет:** настоящий захват. Здесь нет ни дрожания частоты кадров, ни
  сжатия видео, ни разорванных кадров, ни рассинхрона звука. Модуль, работающий
  на синтетике, обязан быть перепроверен на первой живой записи; сессия помечена
  `synthetic: true` именно для того, чтобы это нельзя было забыть.

Генерация детерминирована по сиду: тот же сид — тот же корпус до бита, иначе
тесты на нём ничего не значат.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.action import Action, output_id
from ..core.clocks import Stamp
from ..core.journal import Actor, Kind as EntryKind
from ..core.profile import MILESTONE_0, Profile
from ..core.symbols import Symbolizer
from ..debug.channel import DebugChannel
from ..session import Recorder

# Надписи интерфейса. В журнал они попадают только символами; настоящий текст
# уходит в отладочный поток. Это и есть инвариант 5 в работе.
HUD_LABELS = ("Здоровье", "Голод", "Кислород", "Опыт: 27", "Инвентарь")


@dataclass(slots=True)
class HudRect:
    """Прямоугольник интерфейса в экранных координатах. Не двигается никогда."""

    left: int
    top: int
    width: int
    height: int
    label: str | None = None
    animated: bool = False   # содержимое меняется, положение — нет

    def as_dict(self) -> dict[str, Any]:
        return {"left": self.left, "top": self.top, "width": self.width,
                "height": self.height, "label": self.label, "animated": self.animated}


@dataclass(slots=True)
class SyntheticWorld:
    """Мир: большая текстура, по которой ездит окно камеры, плюс интерфейс сверху."""

    width: int = 320
    height: int = 180
    world_scale: int = 4            # во сколько раз мир больше кадра
    seed: int = 0
    hud: list[HudRect] = field(default_factory=list)
    sound_world_xy: tuple[float, float] = (0.0, 0.0)
    _texture: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.hud:
            self.hud = self.default_hud()
        rng = np.random.default_rng(self.seed)
        self._texture = self._make_texture(rng)
        if self.sound_world_xy == (0.0, 0.0):
            self.sound_world_xy = (self._texture.shape[1] * 0.75,
                                   self._texture.shape[0] * 0.25)

    def default_hud(self) -> list[HudRect]:
        """Интерфейс по краям, как он обычно и бывает. Ни одна координата этой
        раскладки не попадает ни в журнал, ни в проверяемый код: 0.6 обязан
        найти её сам."""
        w, h = self.width, self.height
        return [
            HudRect(4, h - 16, 88, 8, HUD_LABELS[0], animated=True),    # полоса снизу слева
            HudRect(4, h - 28, 88, 8, HUD_LABELS[1], animated=True),
            HudRect(w - 60, 4, 56, 10, HUD_LABELS[3], animated=True),   # счётчик сверху справа
            HudRect(w // 2 - 40, h - 12, 80, 10, HUD_LABELS[4]),        # панель по центру снизу
        ]

    def _make_texture(self, rng: np.random.Generator) -> np.ndarray:
        """Текстура мира: несколько масштабов шума, чтобы был и крупный рельеф,
        и мелкая деталь. Мелкая деталь нужна: без неё сдвиг не определить."""
        H = self.height * self.world_scale
        W = self.width * self.world_scale
        acc = np.zeros((H, W), dtype=np.float32)
        amp = 1.0
        for step in (32, 16, 8, 4, 2, 1):
            gh, gw = max(2, H // step), max(2, W // step)
            grid = rng.random((gh, gw), dtype=np.float32)
            # растягивание повтором: дёшево и достаточно, сглаживание ниже
            up = np.repeat(np.repeat(grid, step, axis=0), step, axis=1)[:H, :W]
            if up.shape != (H, W):
                pad_h, pad_w = H - up.shape[0], W - up.shape[1]
                up = np.pad(up, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode="edge")[:H, :W]
            acc += up * amp
            amp *= 0.55
        acc -= acc.min()
        acc /= max(1e-6, acc.max())
        return (acc * 235 + 10).astype(np.uint8)

    # --- кадр ---------------------------------------------------------------

    def world_view(self, cam_x: float, cam_y: float) -> np.ndarray:
        """Окно мира в позиции камеры. Позиция целочисленная: субпиксельного
        сдвига в вехе 0 нет, и притворяться, что он есть, нельзя."""
        tex = self._texture
        H, W = tex.shape
        x = int(round(cam_x)) % (W - self.width)
        y = int(round(cam_y)) % (H - self.height)
        return tex[y:y + self.height, x:x + self.width]

    def draw_hud(self, frame: np.ndarray, phase: float) -> None:
        """Нарисовать интерфейс поверх мира. Меняется содержимое, не положение."""
        for i, r in enumerate(self.hud):
            y0, y1 = r.top, min(frame.shape[0], r.top + r.height)
            x0, x1 = r.left, min(frame.shape[1], r.left + r.width)
            if y1 <= y0 or x1 <= x0:
                continue
            frame[y0:y1, x0:x1] = 20                                  # подложка
            frame[y0:y0 + 1, x0:x1] = 210                             # рамка
            frame[y1 - 1:y1, x0:x1] = 210
            frame[y0:y1, x0:x0 + 1] = 210
            frame[y0:y1, x1 - 1:x1] = 210
            if r.animated:
                # заполнение полосы гуляет по своему закону, не по камере
                fill = 0.5 + 0.45 * np.sin(phase * (0.7 + 0.13 * i) + i)
                fx = x0 + 1 + int((x1 - x0 - 2) * max(0.0, min(1.0, fill)))
                frame[y0 + 1:y1 - 1, x0 + 1:fx] = 170
            else:
                frame[y0 + 2:y1 - 2, x0 + 2:x1 - 2] = 90

    def hud_mask(self) -> np.ndarray:
        """Истинная маска интерфейса. Уходит в отладочный поток, больше никуда."""
        m = np.zeros((self.height, self.width), dtype=bool)
        for r in self.hud:
            m[r.top:r.top + r.height, r.left:r.left + r.width] = True
        return m

    # --- звук ---------------------------------------------------------------

    def bearing(self, cam_x: float, cam_y: float) -> float:
        """Пеленг источника звука относительно центра камеры, в градусах.

        0° — прямо перед камерой, положительные — вправо. Агент этого числа не
        получает: он получает два канала и должен вывести пеленг сам."""
        cx = cam_x + self.width / 2
        cy = cam_y + self.height / 2
        dx = self.sound_world_xy[0] - cx
        dy = self.sound_world_xy[1] - cy
        return float(np.degrees(np.arctan2(dx, max(1e-6, abs(dy)))))

    def audio_block(self, cam_x: float, cam_y: float, n: int, rate: int,
                    t0: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
        """Стерео-блок: разница амплитуд и задержка по пеленгу.

        Разница каналов здесь настоящая (панорама плюс межканальная задержка), и
        именно из неё будущий модуль обязан считать пеленг. Моно тут невозможно
        по построению — это структурный переключатель `audio_channels`.
        """
        bearing = self.bearing(cam_x, cam_y)
        t = (t0 + np.arange(n, dtype=np.float64) / rate)
        tone = 0.35 * np.sin(2 * np.pi * 440.0 * t)
        tone += 0.12 * np.sin(2 * np.pi * 1200.0 * t)
        noise = 0.02 * rng.standard_normal(n)
        base = tone + noise

        pan = np.clip(bearing / 90.0, -1.0, 1.0)          # −1 слева, +1 справа
        gain_l = float(np.sqrt(max(0.0, (1.0 - pan) / 2.0)))
        gain_r = float(np.sqrt(max(0.0, (1.0 + pan) / 2.0)))
        # межканальная задержка до ~0.6 мс, как у человеческой головы
        delay = int(round(abs(pan) * 0.0006 * rate))
        left = base.copy()
        right = base.copy()
        if delay:
            if pan > 0:
                left = np.concatenate([np.zeros(delay), base[:-delay] if delay else base])
            else:
                right = np.concatenate([np.zeros(delay), base[:-delay] if delay else base])
        stereo = np.stack([left * gain_l, right * gain_r], axis=1)
        return (np.clip(stereo, -1.0, 1.0) * 32000).astype(np.int16), bearing


@dataclass(frozen=True, slots=True)
class Segment:
    """Отрезок сессии: сколько кадров и куда едет камера.

    `acting=False` даёт то самое «фоновое изменение кадра в отсутствие действий»
    из 0.6, `acting=True` — превышение над фоном при действии.
    """

    frames: int
    vx: float
    vy: float
    acting: bool


DEFAULT_TIMELINE: tuple[Segment, ...] = (
    Segment(12, 0.0, 0.0, False),      # стоим: только интерфейс и шум
    Segment(18, 6.0, 0.0, True),       # панорама вправо
    Segment(8, 0.0, 0.0, False),
    Segment(16, -4.0, 2.0, True),      # влево и вниз
    Segment(10, 0.0, 0.0, False),
    Segment(14, 0.0, 5.0, True),       # вперёд
    Segment(8, 3.0, -3.0, True),       # по диагонали
    Segment(10, 0.0, 0.0, False),
)


def generate_session(root: str | Path, *, profile: Profile | None = None,
                     seed: int = 0, timeline: tuple[Segment, ...] = DEFAULT_TIMELINE,
                     width: int = 320, height: int = 180,
                     with_audio: bool = True, noise: int = 0,
                     note: str | None = None) -> Path:
    """Записать синтетическую сессию. Возвращает путь к каталогу сессии.

    Истина (маска интерфейса, сдвиг камеры покадрово, пеленг, таблица символов)
    идёт в `<сессия>/debug/`. Ни журнал, ни хранилища её не содержат.

    `noise` — амплитуда шума по мировому слою, в уровнях яркости. По умолчанию 0,
    потому что цифровой захват экрана шума не даёт. Ставьте больше нуля, когда
    хотите изобразить источник, сжатый с потерями.
    """
    root = Path(root)
    profile = profile or MILESTONE_0
    if int(profile.parameters.get("capture_width", width)) != width:
        profile = profile.with_parameters(capture_width=width, capture_height=height)

    rng = np.random.default_rng(seed)
    world = SyntheticWorld(width=width, height=height, seed=seed)
    sym = Symbolizer(str(profile.structural.get("symbol_salt_id", "s0")))

    rate = int(profile.parameters.get("audio_rate", 48000))
    fps = float(profile.parameters.get("capture_fps", 30.0))
    samples_per_frame = int(round(rate / fps))
    mouse_out = output_id("BTN", 0x110)   # какой-то выход мыши: имя неважно и неизвестно

    with Recorder(root, profile=profile, source=f"synthetic:seed={seed}",
                  synthetic=True, note=note) as rec, \
         DebugChannel(root / "debug", mode="a") as dbg:

        # Истина, не зависящая от кадра.
        dbg.write(Stamp(0, 0, None), "world_setup", {
            "seed": seed, "width": width, "height": height,
            "world_scale": world.world_scale,
            "hud": [r.as_dict() for r in world.hud],
            "sound_world_xy": list(world.sound_world_xy),
            "timeline": [{"frames": s.frames, "vx": s.vx, "vy": s.vy, "acting": s.acting}
                         for s in timeline],
        })
        for label in HUD_LABELS:
            dbg.write_symbol(sym.symbolize(label), label, salt_id=sym.salt_id)

        cam_x = float(world._texture.shape[1] // 3)
        cam_y = float(world._texture.shape[0] // 3)
        frame_no = 0
        audio_t = 0.0

        for seg in timeline:
            for _ in range(seg.frames):
                prev = (cam_x, cam_y)
                cam_x += seg.vx
                cam_y += seg.vy

                frame = world.world_view(cam_x, cam_y).copy()
                # Шум — только по мировому слою и только если его попросили.
                #
                # Захват экрана цифровой и без потерь: неподвижная часть кадра
                # побитово совпадает с предыдущим кадром, никакого сенсорного шума
                # там нет. Добавлять его «для реалистичности» значило бы врать про
                # источник и заодно испортить замер объёма записи. Небольшой шум
                # осмыслен только тогда, когда источник сам сжат с потерями —
                # чужой стрим, запись через видеокодек, — и тогда он задаётся явно.
                if noise:
                    frame = np.clip(frame.astype(np.int16)
                                    + rng.integers(-noise, noise + 1, size=frame.shape),
                                    0, 255).astype(np.uint8)
                world.draw_hud(frame, phase=frame_no * 0.21)

                audio = None
                bearing = None
                if with_audio:
                    audio, bearing = world.audio_block(cam_x, cam_y, samples_per_frame,
                                                       rate, audio_t, rng)
                    audio_t += samples_per_frame / rate

                entry = rec.record_frame(frame, audio=audio, audio_offset_ms=0.0)

                # Действие: движение камеры — следствие действия человека.
                if seg.acting:
                    dx = int(round(seg.vx))
                    dy = int(round(seg.vy))
                    if dx or dy:
                        act = Action.mouse(dx, dy, duration_ms=int(1000 / fps))
                    else:
                        act = Action.key(mouse_out, duration_ms=int(1000 / fps))
                    rec.journal.append(EntryKind.ACTION, rec.clocks.stamp(), Actor.HUMAN,
                                       action=act,
                                       event={"code": "delivered", "reason": None,
                                              "latency_ms": 0.0, "device": "synthetic"})

                dbg.write(entry.stamp, "frame_truth", {
                    "frame_index": frame_no,
                    "cam_x": cam_x, "cam_y": cam_y,
                    "cam_dx": cam_x - prev[0], "cam_dy": cam_y - prev[1],
                    "acting": seg.acting,
                    "bearing_deg": bearing,
                })
                frame_no += 1

        # Маска интерфейса — истина для проверки 0.6.
        mask = world.hud_mask()
        dbg.write(rec.clocks.stamp(), "hud_mask", {
            "height": int(mask.shape[0]), "width": int(mask.shape[1]),
            "rows": [int(v) for v in np.packbits(mask, axis=None)],
            "true_pixels": int(mask.sum()),
        })
        rec.record_note(f"синтетический корпус, сид {seed}, кадров {frame_no}")

    return root


def load_hud_mask(session_root: str | Path) -> np.ndarray:
    """Прочитать истинную маску интерфейса из отладочного потока.

    Функция живёт в `harness.corpus`, потому что нужна тестам и отчётам. Она
    импортирует `harness.debug` — и поэтому `harness.corpus` тоже относится к
    стороне исследователя, а не агента. Тест на изоляцию это учитывает.
    """
    dbg = DebugChannel(Path(session_root) / "debug", mode="r")
    for rec in dbg.read("hud_mask"):
        t = rec["truth"]
        bits = np.unpackbits(np.array(t["rows"], dtype=np.uint8))
        need = int(t["height"]) * int(t["width"])
        return bits[:need].reshape(int(t["height"]), int(t["width"])).astype(bool)
    raise KeyError(f"в отладочном потоке {session_root} нет истинной маски интерфейса")


def load_frame_truth(session_root: str | Path) -> list[dict[str, Any]]:
    dbg = DebugChannel(Path(session_root) / "debug", mode="r")
    return [rec["truth"] for rec in dbg.read("frame_truth")]
