"""Мир с настоящей глубиной: планы, движущиеся с разной скоростью.

Зачем отдельный мир. Глубину нельзя проверить там, где её нет. В основном
синтетическом мире одна плоская текстура: при движении камеры **все** мировые
пиксели смещаются одинаково, и «что близко, что далеко» неразличимо в принципе — не
потому что метод плохой, а потому что различать нечего. Проверять глубину на таком
мире значило бы мерить собственные фантазии.

Здесь глубина есть по построению, и она ровно та, которую человек читает без всяких
координат:

**Параллакс движения.** Планы движутся с разными коэффициентами: ближний уезжает
далеко, дальний почти стоит. Это самый надёжный признак глубины из тех, что даёт
одна камера, и он не требует ни стереопары, ни метрики — только «сместилось сильнее,
значит ближе».

**Наплыв (looming).** Отдельное пятно растёт в кадре, то есть приближается. По
скорости расширения считается **тау** — время до контакта. Дэвид Ли показал, что
животные при прыжке и посадке пользуются именно тау, а не расстоянием: олуша
складывает крылья при фиксированном тау, а не на фиксированной высоте. Тау не
требует знать ни размер объекта, ни скорость — только относительную скорость
расширения, а это видно прямо на картинке.

Истина здесь — номер плана у каждого пикселя и настоящее число кадров до контакта.
Она уходит в отладочный поток и проверяемому коду недоступна: `harness.corpus` — это
сторона исследователя.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.action import Action, Kind as ActionKind
from ..core.profile import Profile
from .world import Observation

# Номера слоёв истины. Ноль — экранное обрамление, дальше по возрастанию дальности.
TRUTH_CHROME = 0
TRUTH_NEAR = 1
TRUTH_MID = 2
TRUTH_FAR = 3
TRUTH_LOOMING = 4


@dataclass(slots=True)
class Plane:
    """Один план глубины: текстура и то, насколько она отзывается на движение.

    `parallax` — во сколько раз этот план смещается относительно движения камеры.
    Единица — «на расстоянии действия», меньше — дальше, больше — ближе. Никакой
    метрики: это отношение, а не метры.
    """

    name: str
    parallax: float
    texture: np.ndarray
    coverage: float                  # какая доля плана непрозрачна
    truth_layer: int


class DepthWorld:
    """Мир из нескольких планов плюс наплывающее пятно плюс обрамление.

    Интерфейс тот же, что у остальных доменов (`step`, `truth`, `screen_mask`,
    `animated_mask`), поэтому по нему проходит тот же кросс-доменный замер.
    """

    name = "depth"

    def __init__(self, profile: Profile, *, seed: int = 0, n_outputs: int = 16) -> None:
        p = profile.parameters
        self.profile = profile
        self.seed = int(seed)
        self.width = int(p["capture_width"])
        self.height = int(p["capture_height"])
        self.fps = float(p["capture_fps"])
        rng = np.random.default_rng(seed)
        self._rng = rng

        # Три плана с разным откликом на движение. Коэффициенты различаются в разы, а
        # не на проценты: различать надо то, что различимо глазом, иначе замер
        # проверяет не признак, а разрядность.
        self.planes: list[Plane] = [
            self._plane("ближний", 2.4, 0.30, TRUTH_NEAR, rng, blob=6),
            self._plane("средний", 1.0, 0.45, TRUTH_MID, rng, blob=4),
            self._plane("дальний", 0.15, 1.00, TRUTH_FAR, rng, blob=3),
        ]

        # Наплывающее пятно: растёт, то есть приближается. Контакт — когда его размер
        # достигает `looming_contact_px`.
        self.looming_size = 6.0
        self.looming_growth = 1.06          # во сколько раз за шаг
        # «Контакт» определяется так же, как его определяет агент: предмет занял
        # заданную долю кадра. Это не придирка — при разных определениях сверять тау
        # с истиной бессмысленно: одна сторона считает до одного момента, другая до
        # другого, и расхождение выглядит ошибкой метода. Замер с расхождением давал
        # 96 % относительной ошибки, из которых почти вся была этим.
        fraction = float(p["looming_contact_fraction"])
        self.looming_contact_px = 2.0 * float(
            np.sqrt(fraction * self.width * self.height / np.pi))
        self.looming_on = True

        self.cam_x = 0.0
        self.cam_y = 0.0
        self.t_world = 0
        self.t_content: float | None = None
        self._steps = 0
        self.last_action_changed: bool | None = None

        from ..core.action import output_id

        self.outputs = tuple(output_id("OUT", code) for code in range(2, 2 + n_outputs))
        moves = [self._left, self._right, self._up, self._down,
                 self._toggle_looming, self._reset_looming]
        order = rng.permutation(len(self.outputs))
        self._effects: dict[str, Any] = {}
        for i, out in enumerate(self.outputs):
            slot = int(order[i])
            self._effects[out] = moves[slot] if slot < len(moves) else None

    # --- построение ---------------------------------------------------------

    def _plane(self, name: str, parallax: float, coverage: float, layer: int,
               rng: np.random.Generator, *, blob: int) -> Plane:
        """План — крупные связные области с мелкой текстурой внутри.

        Крупные и связные — не для красоты. Первая версия делала разрежённую сыпь из
        пятен по 3–9 px, и глубина на ней не находилась в принципе: окно сопоставления
        у **каждого** пикселя накрывало сразу два плана — свой и просвечивающий
        сзади, — а такое окно не совпадает ни при каком одном смещении. Замер: 0–29 %
        верных вместо дальности. Это свойство мира, а не метода: у настоящей сцены
        передний план — это листва и стены, то есть большие связные куски, и смешение
        глубин в окне бывает только на их границах.

        Мелкая текстура внутри области нужна по другой причине: на однородной заливке
        сопоставлять нечего, и любое смещение одинаково хорошо.
        """
        h = int(self.height * 3)
        w = int(self.width * 3)
        # Крупные связные острова: сглаженное случайное поле с порогом по квантилю.
        coarse = rng.random((max(4, h // (blob * 6)), max(4, w // (blob * 6))))
        for _ in range(2):
            coarse = 0.25 * (np.roll(coarse, 1, 0) + np.roll(coarse, -1, 0)
                             + np.roll(coarse, 1, 1) + np.roll(coarse, -1, 1))
        field = np.repeat(np.repeat(coarse, blob * 6, axis=0), blob * 6, axis=1)[:h, :w]
        if field.shape != (h, w):
            pad = np.zeros((h, w))
            pad[:field.shape[0], :field.shape[1]] = field
            field = pad
        mask = field >= np.quantile(field, 1.0 - coverage)

        # Мелкая текстура внутри: тоже блоками, чтобы было что сопоставлять.
        # Верх яркости оставлен наплывающему предмету: он должен быть отличим по
        # яркости, иначе размер измерить нечем и тау неопределимо в принципе.
        fine = rng.integers(40, 200, (h // blob + 1, w // blob + 1), dtype=np.uint8)
        fine_big = np.repeat(np.repeat(fine, blob, axis=0), blob, axis=1)[:h, :w]
        texture = np.where(mask, fine_big, 0).astype(np.uint8)
        return Plane(name, parallax, texture, coverage, layer)

    def default_chrome(self) -> list[tuple[int, int, int, int, bool]]:
        w, h = self.width, self.height
        return [(0, 0, 10, w, True),                 # полоса сверху, анимированная
                (h - 12, 0, 12, w, False)]           # полоса снизу, статичная

    # --- действия ----------------------------------------------------------

    def _left(self) -> None:
        self.cam_x -= 6.0

    def _right(self) -> None:
        self.cam_x += 6.0

    def _up(self) -> None:
        self.cam_y -= 4.0

    def _down(self) -> None:
        self.cam_y += 4.0

    def _toggle_looming(self) -> None:
        self.looming_on = not self.looming_on

    def _reset_looming(self) -> None:
        self.looming_size = 6.0

    def apply(self, action: Action) -> None:
        if action.kind is ActionKind.MOUSE_MOVE:
            self.cam_x += action.dx
            self.cam_y += action.dy
            return
        scale = max(0.0, action.duration_ms / 200.0)
        for out in action.outputs_touched():
            effect = self._effects.get(out)
            if effect is None:
                continue
            for _ in range(max(1, int(round(scale)))):
                effect()

    def snapshot(self) -> dict[str, Any]:
        return {"cam": (round(self.cam_x, 2), round(self.cam_y, 2)),
                "looming": round(self.looming_size, 3),
                "looming_on": self.looming_on}

    # --- шаг ---------------------------------------------------------------

    def frames_to_contact(self) -> float | None:
        """Истинное число шагов до контакта. Только для отладочного потока.

        Считается из настоящей скорости роста, а не из картинки: это то, с чем
        сверяется тау, посчитанное по пикселям. Определение контакта — то же, что у
        агента: предмет занял `looming_contact_fraction` кадра.
        """
        if not self.looming_on or self.looming_size >= self.looming_contact_px:
            return None
        return float(np.log(self.looming_contact_px / self.looming_size)
                     / np.log(self.looming_growth))

    def advance(self) -> None:
        if self.looming_on:
            self.looming_size = min(self.looming_contact_px * 1.5,
                                    self.looming_size * self.looming_growth)

    def step(self, action: Action | None = None, *,
             with_audio: bool = True) -> Observation:
        before = self.snapshot()
        acted = action is not None and not action.masked
        if acted:
            self.apply(action)
        self.last_action_changed = (self.snapshot() != before) if acted else None
        self.advance()
        self.t_world += 1
        self._steps += 1
        frame = self.render()
        audio = None
        if with_audio:
            n = int(round(int(self.profile.parameters["audio_rate"]) / self.fps))
            audio = np.zeros((n, 2), dtype=np.int16)
        return Observation(frame, audio, self.t_world, None,
                           changed=self.snapshot() != before)

    # --- отрисовка ---------------------------------------------------------

    def _plane_view(self, plane: Plane) -> tuple[np.ndarray, np.ndarray]:
        """Вид плана при текущем положении камеры и маска его непрозрачности."""
        h, w = self.height, self.width
        th, tw = plane.texture.shape
        ox = int(round(self.cam_x * plane.parallax)) % max(1, tw - w)
        oy = int(round(self.cam_y * plane.parallax)) % max(1, th - h)
        view = plane.texture[oy:oy + h, ox:ox + w]
        return view, view > 0

    def _looming_mask(self) -> np.ndarray:
        """Круг наплывающего пятна. Центр не двигается: приближение, а не пролёт."""
        h, w = self.height, self.width
        cy, cx = h * 0.5, w * 0.5
        r = self.looming_size * 0.5
        ys = np.arange(h)[:, None] - cy
        xs = np.arange(w)[None, :] - cx
        return (ys * ys + xs * xs) <= r * r

    def render(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        # Дальний рисуется первым, ближний последним: перекрытие — тоже признак
        # глубины, и порядок отрисовки его задаёт.
        for plane in sorted(self.planes, key=lambda p: p.parallax):
            view, mask = self._plane_view(plane)
            frame[mask] = view[mask]
        if self.looming_on:
            frame[self._looming_mask()] = 255
        self._draw_chrome(frame)
        return frame

    def _draw_chrome(self, frame: np.ndarray) -> None:
        phase = self._steps * 0.19
        for top, left, height, width, animated in self.default_chrome():
            y1 = min(frame.shape[0], top + height)
            x1 = min(frame.shape[1], left + width)
            frame[top:y1, left:x1] = 24
            frame[top:top + 1, left:x1] = 200
            frame[y1 - 1:y1, left:x1] = 200
            if animated:
                fill = 0.5 + 0.45 * float(np.sin(phase))
                fx = left + 1 + int((x1 - left - 2) * max(0.0, min(1.0, fill)))
                frame[top + 1:y1 - 1, left + 1:fx] = 170
            else:
                frame[top + 2:y1 - 2, left + 2:x1 - 2] = 90

    # --- истина ------------------------------------------------------------

    def screen_mask(self) -> np.ndarray:
        m = np.zeros((self.height, self.width), dtype=bool)
        for top, left, height, width, _ in self.default_chrome():
            m[top:top + height, left:left + width] = True
        return m

    def animated_mask(self) -> np.ndarray:
        m = np.zeros((self.height, self.width), dtype=bool)
        for top, left, height, width, animated in self.default_chrome():
            if animated:
                m[top:top + height, left:left + width] = True
        return m

    def depth_truth(self) -> np.ndarray:
        """Номер слоя у каждого пикселя. Только для отладочного потока.

        Порядок тот же, что при отрисовке: кто нарисован позже, тот и виден.
        """
        truth = np.full((self.height, self.width), TRUTH_FAR, dtype=np.int8)
        for plane in sorted(self.planes, key=lambda p: p.parallax):
            _, mask = self._plane_view(plane)
            truth[mask] = plane.truth_layer
        if self.looming_on:
            truth[self._looming_mask()] = TRUTH_LOOMING
        truth[self.screen_mask()] = TRUTH_CHROME
        return truth

    def truth(self) -> dict[str, Any]:
        return {"domain": self.name, "seed": self.seed,
                "outputs": list(self.outputs),
                "planes": [{"name": p.name, "parallax": p.parallax,
                            "layer": p.truth_layer} for p in self.planes],
                "looming_size": round(self.looming_size, 3),
                "frames_to_contact": self.frames_to_contact(),
                "state": self.snapshot()}
