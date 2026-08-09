"""Интерактивный синтетический мир: действия здесь имеют последствия.

Зачем он нужен отдельно от записанного корпуса. Запись — это прошлое: по ней
можно проверить восприятие, но нельзя проверить ничего, что требует ответа мира
на действие. А это половина проекта: лепет, обратимость, ошибка предсказания,
граф мест, аллостаз. Поэтому здесь мир, который отвечает.

Что он умеет и чего нет:

- **Есть:** движение, поворот, переключаемые эффекты (свет), необратимые эффекты
  (сломать), молчащие выходы, звук с пеленгом относительно взгляда.
- **Нет:** правил, целей, счёта, врагов, наград. Мир просто реагирует. Всё, что
  агент про него узнает, он узнает нажатиями.

Соответствие «выход → эффект» выводится из сида и пишется **только** в отладочный
поток. Ни агент, ни проверяемый код его не видят: иначе лепет проверялся бы на
задаче, ответ к которой лежит рядом.

Молчащие выходы — не украшение. Именно на них проверяется, что заглушённая или
бесполезная попытка попадает в журнал: агент обязан иметь возможность выучить
«здесь ничего нет», а для этого попытка должна остаться в записи.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from ..core.action import Action, Kind as ActionKind, output_id
from ..core.profile import Profile
from .synthetic import HUD_LABELS, HudRect, SyntheticWorld


class Effect(StrEnum):
    """Что выход делает с миром. Агенту это неизвестно по построению."""

    FORWARD = "forward"      # движение вперёд по миру
    BACK = "back"
    LEFT = "left"            # шаг в сторону
    RIGHT = "right"
    TURN_LEFT = "turn_left"  # поворот взгляда: меняет и картинку, и пеленг звука
    TURN_RIGHT = "turn_right"
    TOGGLE_LIGHT = "toggle_light"   # обратимо: нажал — включилось, нажал — выключилось
    BREAK_PANEL = "break_panel"     # необратимо: элемент интерфейса больше не вернуть
    SILENT = "silent"        # ничего не делает


REVERSIBLE = {Effect.FORWARD, Effect.BACK, Effect.LEFT, Effect.RIGHT,
              Effect.TURN_LEFT, Effect.TURN_RIGHT, Effect.TOGGLE_LIGHT, Effect.SILENT}

# Обратные пары: чем откатывается эффект. Нужно для проверки обратимости —
# агент должен уметь *попробовать* откатить, а не узнать это из таблицы.
INVERSE: dict[Effect, Effect] = {
    Effect.FORWARD: Effect.BACK, Effect.BACK: Effect.FORWARD,
    Effect.LEFT: Effect.RIGHT, Effect.RIGHT: Effect.LEFT,
    Effect.TURN_LEFT: Effect.TURN_RIGHT, Effect.TURN_RIGHT: Effect.TURN_LEFT,
    Effect.TOGGLE_LIGHT: Effect.TOGGLE_LIGHT,
}


@dataclass(frozen=True, slots=True)
class Observation:
    """Что мир отдал в ответ на шаг."""

    frame: np.ndarray
    audio: np.ndarray | None
    t_world: int
    bearing_deg: float | None      # истина, только для отладочного потока
    changed: bool                  # изменилось ли состояние мира на этом шаге


@dataclass(slots=True)
class WorldState:
    cam_x: float
    cam_y: float
    yaw: float = 0.0               # градусы, положительные — вправо
    light: bool = True
    broken: set[int] = field(default_factory=set)   # индексы сломанных панелей

    def snapshot(self) -> dict[str, Any]:
        return {"cam_x": round(self.cam_x, 3), "cam_y": round(self.cam_y, 3),
                "yaw": round(self.yaw, 3), "light": self.light,
                "broken": sorted(self.broken)}


class InteractiveWorld:
    """Мир, отвечающий на действия. Вся истина доступна только через `truth()`."""

    # Насколько сильно один эффект двигает мир за 100 мс удержания.
    STEP_PX = 6.0
    TURN_DEG = 9.0
    YAW_TO_PX = 4.0        # во сколько пикселей смещения превращается градус поворота

    def __init__(self, profile: Profile, *, seed: int = 0, n_outputs: int = 24) -> None:
        self.profile = profile
        self.seed = int(seed)
        p = profile.parameters
        self.width = int(p["capture_width"])
        self.height = int(p["capture_height"])
        self.rate = int(p["audio_rate"])
        self.fps = float(p["capture_fps"])
        self._rng = np.random.default_rng(seed)

        self.scene = SyntheticWorld(width=self.width, height=self.height, seed=seed)
        self.state = WorldState(cam_x=float(self.scene._texture.shape[1] // 3),
                                cam_y=float(self.scene._texture.shape[0] // 3))
        self.outputs, self._effects = self._wire_outputs(n_outputs)
        self.t_world = 0
        self._audio_t = 0.0
        self._steps = 0
        self.last_action_changed: bool | None = None

        # --- два свойства из TASK-05, оба структурные ------------------------
        #
        # Переменная стоимость прохода. Без неё время прохода тождественно числу
        # шагов, `mu` ребра равна 1/30 у всех рёбер, `sigma` нулю, и всякая сверка
        # `mu` сравнивает число с самим собой (вырождение 13.1 в MEASUREMENT.md).
        # По умолчанию выключена намеренно: прежний мир остаётся контролем, на
        # котором замер обязан остаться вакуумным, иначе новый ничего не доказывает.
        self.variable_cost = bool(profile.structural["world_variable_cost"])
        self.cost_min = int(p["world_cost_min_ticks"])
        self.cost_max = max(self.cost_min, int(p["world_cost_max_ticks"]))
        self.cost_jitter = float(p["world_cost_jitter"])
        # Ограниченность области. Без неё фронтир не кончается: в 34.7 % решений
        # разведки место незнакомо вовсе, и замыкание петель почти не срабатывает
        # (вырождение 13.2).
        self.bounded = bool(profile.structural["world_bounded"])
        self.extent = float(p["world_extent_px"])
        self._home = (self.state.cam_x, self.state.cam_y)
        # Сколько тактов стоил последний переход. Истина исследователя: агенту она
        # недоступна, он видит только то, что кадры сменились не за один такт.
        self.last_cost_ticks = 1
        self.walls_hit = 0

    # --- проводка тела ------------------------------------------------------

    def _wire_outputs(self, n: int) -> tuple[tuple[str, ...], dict[str, Effect]]:
        """Раздать эффекты по выходам. Раскладка выводится из сида.

        Скан-коды берутся подряд, как на настоящей клавиатуре, но какой код что
        делает — решает сид. Поэтому выученное на одном сиде не переносится на
        другой, и «открытие» нельзя спутать с «воспоминанием»
        (`randomize_world` в профиле про то же).
        """
        if n < 12:
            raise ValueError("выходов должно быть хотя бы 12, иначе нечего открывать")
        outputs = tuple(output_id("OUT", code) for code in range(2, 2 + n))
        if len(set(outputs)) != len(outputs):
            raise ValueError("коллизия идентификаторов выходов: увеличьте длину хеша")

        живые = [Effect.FORWARD, Effect.BACK, Effect.LEFT, Effect.RIGHT,
                 Effect.TURN_LEFT, Effect.TURN_RIGHT,
                 Effect.TOGGLE_LIGHT, Effect.TOGGLE_LIGHT,
                 Effect.BREAK_PANEL]
        effects = list(живые) + [Effect.SILENT] * (n - len(живые))
        order = self._rng.permutation(n)
        return outputs, {outputs[i]: effects[order[i]] for i in range(n)}

    # --- шаг мира -----------------------------------------------------------

    def step(self, action: Action | None = None, *, with_audio: bool = True) -> Observation:
        """Один тик мира. Мир идёт и когда действия нет — это и есть фон.

        Длительность удержания влияет на величину эффекта: удержание 340 мс
        двигает дальше, чем 40 мс. Так и должно быть — иначе действие снова
        превратилось бы в дискретное событие (инвариант 8).
        """
        before = self.state.snapshot()
        acted = action is not None and not action.masked
        if acted:
            self._apply(action)
        # Стоимость прохода считается **после** применения действия: она зависит от
        # местности, в которую пришли, а не от ребра. Ребра агент ещё не знает, а
        # местность есть свойство мира и существует до всякого знания о ней.
        self.last_cost_ticks = self._cost_of(before)
        # Истина исследователя: подействовало ли именно действие. Здесь она
        # совпадает с общим изменением, потому что этот мир сам ничего не делает,
        # — но у миров, которые идут сами, не совпадёт, поэтому величина отдельная.
        self.last_action_changed = ((self.state.snapshot() != before) if acted
                                    else None)
        # Такты мира идут по стоимости прохода, а не по одному за шаг. Именно здесь
        # время перестаёт быть тождественным топологии: два перехода между теми же
        # местами могут стоить разное, и `mu` ребра начинает нести информацию.
        self.t_world += self.last_cost_ticks
        self._steps += 1
        frame = self._render()
        audio, bearing = (self._audio() if with_audio else (None, None))
        return Observation(frame, audio, self.t_world, bearing,
                           changed=self.state.snapshot() != before)

    def _cost_of(self, before: Any) -> int:
        """Сколько тактов занял этот переход. Один — если переменная цена выключена.

        Стоимость выводится из **местности**: из яркости текстуры в том месте, куда
        пришли. Так она оказывается свойством мира, устойчивым между проходами, а не
        случайным числом на ребре, — и агент, вернувшись сюда другой дорогой, застанет
        ту же местность и ту же примерную цену.

        Разброс от прохода к проходу добавляется поверх: помеха, обход, задержка. Он
        и есть то, что делает `sigma` величиной о мире. При `world_cost_jitter = 0`
        `sigma` снова станет нулём — это проверяется тестом, потому что иначе
        «переменная стоимость заработала» было бы нечем подтвердить.
        """
        if not self.variable_cost:
            return 1
        tex = self.scene._texture
        h, w = tex.shape[:2]
        y = int(self.state.cam_y) % h
        x = int(self.state.cam_x) % w
        # Яркость местности в [0,1] → цена в объявленном диапазоне.
        terrain = float(tex[y, x]) / 255.0
        base = self.cost_min + terrain * (self.cost_max - self.cost_min)
        if self.cost_jitter > 0.0:
            base *= 1.0 + self._rng.uniform(-self.cost_jitter, self.cost_jitter)
        return int(max(self.cost_min, min(self.cost_max, round(base))))

    def _clamp_to_extent(self) -> None:
        """Удержать камеру внутри области. Стены не рисуются: их узнают по последствию.

        Отдельного изображения стены нет намеренно. Агент не должен получать разметку
        мира ни одним каналом (инвариант 4); границу он узнаёт тем же способом, каким
        узнаёт всё остальное, — нажал, а мир не сдвинулся.
        """
        if not self.bounded:
            return
        hx, hy = self._home
        cx = min(hx + self.extent, max(hx - self.extent, self.state.cam_x))
        cy = min(hy + self.extent, max(hy - self.extent, self.state.cam_y))
        if cx != self.state.cam_x or cy != self.state.cam_y:
            self.walls_hit += 1
        self.state.cam_x, self.state.cam_y = cx, cy

    def _apply(self, action: Action) -> None:
        if action.kind is ActionKind.MOUSE_MOVE:
            # Мышь поворачивает взгляд: горизонталь — рыскание, вертикаль игнорируется,
            # потому что вертикального взгляда в этом мире нет.
            self.state.yaw += action.dx * 0.35
            return
        if action.kind is ActionKind.NOTHING or not action.outputs_touched():
            return

        scale = max(0.0, action.duration_ms / 100.0)
        for out in action.outputs_touched():
            effect = self._effects.get(out, Effect.SILENT)
            self._apply_effect(effect, scale)
        self._clamp_to_extent()

    def _apply_effect(self, effect: Effect, scale: float) -> None:
        s = self.state
        yaw_rad = np.radians(s.yaw)
        if effect is Effect.FORWARD or effect is Effect.BACK:
            sign = 1.0 if effect is Effect.FORWARD else -1.0
            s.cam_y += sign * self.STEP_PX * scale * float(np.cos(yaw_rad))
            s.cam_x += sign * self.STEP_PX * scale * float(np.sin(yaw_rad))
        elif effect is Effect.LEFT or effect is Effect.RIGHT:
            sign = -1.0 if effect is Effect.LEFT else 1.0
            s.cam_x += sign * self.STEP_PX * scale
        elif effect is Effect.TURN_LEFT:
            s.yaw -= self.TURN_DEG * scale
        elif effect is Effect.TURN_RIGHT:
            s.yaw += self.TURN_DEG * scale
        elif effect is Effect.TOGGLE_LIGHT:
            s.light = not s.light
        elif effect is Effect.BREAK_PANEL:
            # Необратимо: сломать можно только то, что ещё не сломано, и вернуть
            # это нельзя ничем. Именно на этом агент учится, что «я не умею
            # откатить» — про мир, а не про запрет.
            candidates = [i for i in range(len(self.scene.hud)) if i not in s.broken]
            if candidates:
                s.broken.add(candidates[0])

    # --- отрисовка ----------------------------------------------------------

    def _render(self) -> np.ndarray:
        s = self.state
        frame = self.scene.world_view(s.cam_x + s.yaw * self.YAW_TO_PX, s.cam_y).copy()
        if not s.light:
            # Свет выключен: мир темнеет, интерфейс — нет. Так и в игре: подсветка
            # интерфейса не зависит от освещения сцены, и это ещё один признак,
            # по которому слои различимы.
            frame = (frame.astype(np.uint16) * 45 // 100).astype(np.uint8)
        self._draw_hud(frame)
        return frame

    def _draw_hud(self, frame: np.ndarray) -> None:
        phase = self._steps * 0.21
        for i, r in enumerate(self.scene.hud):
            if i in self.state.broken:
                continue                       # сломанная панель просто исчезла
            self._draw_rect(frame, r, phase, i)

    @staticmethod
    def _draw_rect(frame: np.ndarray, r: HudRect, phase: float, i: int) -> None:
        y0, y1 = r.top, min(frame.shape[0], r.top + r.height)
        x0, x1 = r.left, min(frame.shape[1], r.left + r.width)
        if y1 <= y0 or x1 <= x0:
            return
        frame[y0:y1, x0:x1] = 20
        frame[y0:y0 + 1, x0:x1] = 210
        frame[y1 - 1:y1, x0:x1] = 210
        frame[y0:y1, x0:x0 + 1] = 210
        frame[y0:y1, x1 - 1:x1] = 210
        if r.animated:
            fill = 0.5 + 0.45 * float(np.sin(phase * (0.7 + 0.13 * i) + i))
            fx = x0 + 1 + int((x1 - x0 - 2) * max(0.0, min(1.0, fill)))
            frame[y0 + 1:y1 - 1, x0 + 1:fx] = 170
        else:
            frame[y0 + 2:y1 - 2, x0 + 2:x1 - 2] = 90

    def _audio(self) -> tuple[np.ndarray, float]:
        n = int(round(self.rate / self.fps))
        block, bearing = self.scene.audio_block(self.state.cam_x, self.state.cam_y,
                                               n, self.rate, self._audio_t, self._rng)
        self._audio_t += n / self.rate
        # Пеленг считается относительно взгляда: повернулся — источник «поехал».
        # Именно это связывает поворот со слухом и делает пеленг выучиваемым.
        return block, float(bearing - self.state.yaw)

    # --- истина (только в отладочный поток) ---------------------------------

    def truth(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "outputs": list(self.outputs),
            "wiring": {out: str(eff) for out, eff in self._effects.items()},
            "live_outputs": [o for o, e in self._effects.items() if e is not Effect.SILENT],
            "silent_outputs": [o for o, e in self._effects.items() if e is Effect.SILENT],
            "irreversible_outputs": [o for o, e in self._effects.items()
                                     if e not in REVERSIBLE],
            "hud": [r.as_dict() for r in self.scene.hud],
            "hud_labels": list(HUD_LABELS),
            "state": self.state.snapshot(),
        }

    def effect_of(self, output: str) -> Effect:
        """Только для отладочного потока и тестов. Агентскому коду недоступно:
        модуль лежит в `harness.corpus`, который относится к стороне исследователя."""
        return self._effects.get(output, Effect.SILENT)

    def hud_mask(self) -> np.ndarray:
        """Истинная маска интерфейса с учётом сломанного."""
        m = np.zeros((self.height, self.width), dtype=bool)
        for i, r in enumerate(self.scene.hud):
            if i in self.state.broken:
                continue
            m[r.top:r.top + r.height, r.left:r.left + r.width] = True
        return m

    def animated_mask(self) -> np.ndarray:
        """Часть интерфейса, которая меняется, не двигаясь: ползущие заполнения.

        Нужна замеру отдельно, потому что такая часть не отделяется ни по
        параллаксу (она не смещается вместе с миром, но и не совпадает с собой),
        ни по неподвижности (она меняется). Это известный пробел, и он должен быть
        виден в таблице отдельной графой, а не растворяться в общей полноте.
        """
        m = np.zeros((self.height, self.width), dtype=bool)
        for i, r in enumerate(self.scene.hud):
            if i in self.state.broken or not r.animated:
                continue
            m[r.top:r.top + r.height, r.left:r.left + r.width] = True
        return m
