"""Домены: игра, документ, рабочий стол, видео.

Правило из CLAUDE.md: «архитектура обязана быть доменно-независимой: если решение
работает только в Minecraft, оно неправильное». Заявить это легко, проверить —
только так: сделать несколько разных миров и прогнать по ним одни и те же модули.

Что общего у всех доменов и что различается.

**Общее:** кадр состоит из движущегося содержимого и неподвижного обрамления.
Это и есть то различие, которое ищет 0.6, и оно не про игры: в браузере
прокручивается страница, а панель вкладок стоит; в проигрывателе меняется картинка,
а полоса управления стоит.

**Различается механика движения**, и именно она проверяет модули на прочность:

| Домен | Как двигается содержимое | Чем это опасно для модулей |
|---|---|---|
| Игра | камера панорамирует в двух измерениях | ничем: на этом всё и строилось |
| Документ | прокрутка строго по вертикали, рывками | сдвиг по одной оси, а не по двум |
| Рабочий стол | двигаются отдельные окна, общего сдвига нет | параллакса нет вообще |
| Видео | содержимое меняется само, без действий | движение есть, а действия нет |

Рабочий стол — самый жёсткий случай, и он здесь нужен именно поэтому. Глобального
сдвига в нём не бывает, значит 0.6 обязан честно сказать «не знаю», а не выдать
уверенную ерунду. Видео — второй жёсткий: мир меняется сам, и агент не должен
приписывать эти изменения себе.

У всех доменов одинаковый интерфейс: `outputs`, `step(action)`, `truth()`,
`screen_mask()`. Поэтому любой модуль, который работает с одним, работает со всеми
без единой правки — или не работает, и это видно на замере.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..core.action import Action, Kind as ActionKind, output_id
from ..core.profile import Profile
from .world import InteractiveWorld, Observation


def _upscale_to(small: np.ndarray, factor: int, height: int, width: int) -> np.ndarray:
    """Растянуть маленькую картинку и обрезать ровно по кадру.

    Ровно по кадру — не придирка. Если содержимое окажется хоть на пиксель меньше,
    по краю останется неподвижная чёрная полоса, и разделение слоёв честно объявит
    её экранным слоем: она ведь действительно не двигается. Результат замера
    вырастет, а причина будет в генераторе домена, а не в методе. Такую ошибку
    почти невозможно заметить по числам, поэтому размер проверяется здесь.
    """
    big = np.repeat(np.repeat(small, factor, axis=0), factor, axis=1)
    if big.shape[0] < height or big.shape[1] < width:
        raise ValueError(
            f"содержимое {big.shape} меньше кадра {(height, width)}: по краю "
            "останется неподвижная полоса, и она попадёт в экранный слой")
    return np.ascontiguousarray(big[:height, :width])


@runtime_checkable
class Domain(Protocol):
    """Мир, в котором можно жить. Ровно то, что нужно любому модулю."""

    name: str
    outputs: tuple[str, ...]

    def step(self, action: Action | None = None, *, with_audio: bool = True) -> Observation: ...
    def truth(self) -> dict[str, Any]: ...
    def screen_mask(self) -> np.ndarray: ...
    def animated_mask(self) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Общая часть: содержимое плюс обрамление
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Chrome:
    """Прямоугольник обрамления: панель, полоса, рамка. Не двигается никогда."""

    top: int
    left: int
    height: int
    width: int
    animated: bool = False
    label: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"top": self.top, "left": self.left, "height": self.height,
                "width": self.width, "animated": self.animated, "label": self.label}


class LayeredDomain:
    """Домен из движущегося содержимого и неподвижного обрамления.

    Содержимое рисуют наследники; обрамление рисуется здесь одинаково для всех,
    потому что оно одинаково по смыслу: неподвижная часть экрана.
    """

    name = "layered"

    def __init__(self, profile: Profile, *, seed: int = 0, n_outputs: int = 16) -> None:
        self.profile = profile
        self.seed = int(seed)
        p = profile.parameters
        self.width = int(p["capture_width"])
        self.height = int(p["capture_height"])
        self.fps = float(p["capture_fps"])
        self._rng = np.random.default_rng(seed)
        self.chrome: list[Chrome] = self.default_chrome()
        self.outputs = tuple(output_id("OUT", code) for code in range(2, 2 + n_outputs))
        self.t_world = 0
        self.t_content: float | None = None
        self._steps = 0
        self._broken: set[int] = set()
        # Подействовало ли последнее действие — в отличие от «изменился ли мир».
        # В домене, который идёт сам, это разные вещи, и путать их нельзя: в
        # проигрывателе содержимое меняется каждый кадр, и по общему изменению
        # живыми выглядят все выходы разом.
        self.last_action_changed: bool | None = None

    # --- обрамление ---------------------------------------------------------

    def default_chrome(self) -> list[Chrome]:
        raise NotImplementedError

    def screen_mask(self) -> np.ndarray:
        m = np.zeros((self.height, self.width), dtype=bool)
        for i, c in enumerate(self.chrome):
            if i in self._broken:
                continue
            m[c.top:c.top + c.height, c.left:c.left + c.width] = True
        return m

    def animated_mask(self) -> np.ndarray:
        """Часть обрамления, которая меняется, не двигаясь. См. `world.animated_mask`."""
        m = np.zeros((self.height, self.width), dtype=bool)
        for i, c in enumerate(self.chrome):
            if i in self._broken or not c.animated:
                continue
            m[c.top:c.top + c.height, c.left:c.left + c.width] = True
        return m

    def _draw_chrome(self, frame: np.ndarray) -> None:
        phase = self._steps * 0.21
        for i, c in enumerate(self.chrome):
            if i in self._broken:
                continue
            y0, y1 = c.top, min(frame.shape[0], c.top + c.height)
            x0, x1 = c.left, min(frame.shape[1], c.left + c.width)
            if y1 <= y0 or x1 <= x0:
                continue
            frame[y0:y1, x0:x1] = 22
            frame[y0:y0 + 1, x0:x1] = 205
            frame[y1 - 1:y1, x0:x1] = 205
            frame[y0:y1, x0:x0 + 1] = 205
            frame[y0:y1, x1 - 1:x1] = 205
            if c.animated:
                fill = 0.5 + 0.45 * float(np.sin(phase * (0.7 + 0.13 * i) + i))
                fx = x0 + 1 + int((x1 - x0 - 2) * max(0.0, min(1.0, fill)))
                frame[y0 + 1:y1 - 1, x0 + 1:fx] = 165
            else:
                frame[y0 + 2:y1 - 2, x0 + 2:x1 - 2] = 88

    # --- шаг ----------------------------------------------------------------

    def content(self) -> np.ndarray:
        raise NotImplementedError

    def apply(self, action: Action) -> None:
        raise NotImplementedError

    def state_snapshot(self) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, action: Action | None = None, *,
             with_audio: bool = True) -> Observation:
        before = self.state_snapshot()
        acted = action is not None and not action.masked
        if acted:
            self.apply(action)
        self.last_action_changed = (self.state_snapshot() != before) if acted else None
        self.advance()
        self.t_world += 1
        self._steps += 1
        frame = self.content()
        self._draw_chrome(frame)
        audio = None
        if with_audio:
            audio = np.zeros((int(round(int(self.profile.parameters["audio_rate"])
                                        / self.fps)), 2), dtype=np.int16)
        return Observation(frame, audio, self.t_world, None,
                           changed=self.state_snapshot() != before)

    def advance(self) -> None:
        """Что мир делает сам, без агента. По умолчанию — ничего."""

    # --- истина -------------------------------------------------------------

    def truth(self) -> dict[str, Any]:
        return {"domain": self.name, "seed": self.seed,
                "outputs": list(self.outputs),
                "chrome": [c.as_dict() for c in self.chrome],
                "state": self.state_snapshot()}


# ---------------------------------------------------------------------------
# Документ: прокрутка по одной оси
# ---------------------------------------------------------------------------


class DocumentDomain(LayeredDomain):
    """Страница текста в окне с панелью сверху и полосой прокрутки справа.

    Сдвиг только вертикальный и рывками: прокрутка идёт строками, а не плавно.
    Для 0.6 это другой случай, чем панорама камеры, и он проверяет, что метод не
    предполагает движения по двум осям.

    «Текст» — полосы разной длины, а не буквы: буквы потребовали бы шрифта, а
    проверяем мы не распознавание, а разделение слоёв. Строки различимы между
    собой, и этого достаточно, чтобы сдвиг был определим.
    """

    name = "document"

    def __init__(self, profile: Profile, *, seed: int = 0, n_outputs: int = 16) -> None:
        super().__init__(profile, seed=seed, n_outputs=n_outputs)
        self.page = self._make_page()
        self.scroll = 0
        self.line_height = 6
        live = [self._scroll_down, self._scroll_up, self._page_down, self._page_up,
                self._to_top]
        order = self._rng.permutation(len(self.outputs))
        self._effects: dict[str, Any] = {}
        for i, out in enumerate(self.outputs):
            slot = int(order[i])
            self._effects[out] = live[slot] if slot < len(live) else None

    def default_chrome(self) -> list[Chrome]:
        w, h = self.width, self.height
        return [
            Chrome(0, 0, 14, w, animated=True, label="панель вкладок"),
            Chrome(14, w - 6, h - 14, 6, animated=False, label="полоса прокрутки"),
            Chrome(h - 10, 0, 10, w, animated=True, label="строка состояния"),
        ]

    def _make_page(self) -> np.ndarray:
        """Страница: строки «слов» разной длины на светлом фоне."""
        height = self.height * 6
        page = np.full((height, self.width), 235, dtype=np.uint8)
        rng = np.random.default_rng(self.seed + 1)
        y = 4
        while y < height - 8:
            x = 8
            while x < self.width - 30:
                word = int(rng.integers(8, 46))
                if x + word > self.width - 12:
                    break
                page[y:y + 4, x:x + word] = int(rng.integers(20, 90))
                x += word + int(rng.integers(3, 7))
            y += 6 + int(rng.integers(0, 3))
        return page

    # --- действия -----------------------------------------------------------

    def _scroll_down(self) -> None:
        self.scroll = min(self.page.shape[0] - self.height, self.scroll + self.line_height * 3)

    def _scroll_up(self) -> None:
        self.scroll = max(0, self.scroll - self.line_height * 3)

    def _page_down(self) -> None:
        self.scroll = min(self.page.shape[0] - self.height, self.scroll + self.height - 30)

    def _page_up(self) -> None:
        self.scroll = max(0, self.scroll - (self.height - 30))

    def _to_top(self) -> None:
        # Необратимо в том смысле, в котором это важно агенту: положение, где он
        # был, ничем не вернуть — «назад» здесь нет.
        self.scroll = 0

    def apply(self, action: Action) -> None:
        if action.kind is ActionKind.MOUSE_MOVE:
            self.scroll = int(np.clip(self.scroll + action.dy * 2, 0,
                                      self.page.shape[0] - self.height))
            return
        for out in action.outputs_touched():
            effect = self._effects.get(out)
            if effect is not None:
                effect()

    def content(self) -> np.ndarray:
        top = int(np.clip(self.scroll, 0, self.page.shape[0] - self.height))
        return self.page[top:top + self.height].copy()

    def state_snapshot(self) -> dict[str, Any]:
        return {"scroll": int(self.scroll)}


# ---------------------------------------------------------------------------
# Рабочий стол: общего сдвига не бывает
# ---------------------------------------------------------------------------


class DesktopDomain(LayeredDomain):
    """Несколько окон на фоне. Двигается одно — то, что в фокусе.

    Самый жёсткий домен для параллакса: глобального сдвига здесь нет вообще, и
    по параллаксу метод обязан честно сказать «не знаю», а не объявить пол-экрана
    интерфейсом. Ответ здесь даёт второй признак — неподвижность: обои и панель
    задач не меняются, окна меняются.
    """

    name = "desktop"

    def __init__(self, profile: Profile, *, seed: int = 0, n_outputs: int = 16) -> None:
        super().__init__(profile, seed=seed, n_outputs=n_outputs)
        self.background = self._make_background()
        rng = np.random.default_rng(seed + 2)
        self.windows = [
            {"id": i,
             "top": int(rng.integers(20, self.height - 70)),
             "left": int(rng.integers(10, self.width - 110)),
             "height": int(rng.integers(40, 64)),
             "width": int(rng.integers(70, 110)),
             "shade": int(rng.integers(120, 200))}
            for i in range(3)
        ]
        self.focus = 0
        # Какие окна за прогон хоть раз двинулись. Нужно истине, а не механике:
        # см. `screen_mask`.
        self._moved: set[int] = set()
        moves = [self._left, self._right, self._up, self._down, self._next_window,
                 self._close]
        order = self._rng.permutation(len(self.outputs))
        self._effects: dict[str, Any] = {}
        for i, out in enumerate(self.outputs):
            slot = int(order[i])
            self._effects[out] = moves[slot] if slot < len(moves) else None

    def default_chrome(self) -> list[Chrome]:
        return [Chrome(self.height - 12, 0, 12, self.width, animated=True,
                       label="панель задач")]

    def screen_mask(self) -> np.ndarray:
        """Экранный слой рабочего стола — всё, что не двигавшееся окно.

        Здесь две поправки к тому, что было, и обе — в истине, а не в методе.

        Сначала истиной была одна панель задач. Это прямая ошибка: обои прибиты к
        экрану ровно так же, как панель, — они не двигаются никогда. Замер с
        прежней истиной считал огромную неподвижную область «миром» и наказывал
        метод за правильный ответ.

        Вторая поправка тоньше. «Окно» — не свойство пикселя, а свойство поведения:
        слой определён системой отсчёта, а система отсчёта видна только по
        движению. Окно, которое за всю запись не шевельнулось, было в этой записи
        прибито к экрану — и назвать его экранным слоем правильно, а не ошибочно.
        Как только оно двинулось, оно становится содержимым, истина по этим
        пикселям меняется, и замер выводит их из подсчёта своим обычным правилом
        (см. `benchmark.bench_domain`). Поэтому здесь вычитаются только окна из
        `_moved`.

        Ошибка в истине хуже ошибки в методе: метод она не портит, а числа портит
        незаметно.
        """
        m = np.ones((self.height, self.width), dtype=bool)
        for w in self.windows:
            if w["id"] not in self._moved:
                continue
            m[w["top"]:w["top"] + w["height"], w["left"]:w["left"] + w["width"]] = False
        return m | super().screen_mask()

    def _make_background(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 3)
        return _upscale_to(rng.integers(60, 90, (self.height // 8 + 1,
                                                 self.width // 8 + 1), dtype=np.uint8),
                           8, self.height, self.width)

    # --- действия -----------------------------------------------------------

    def _shift_focused(self, dy: int, dx: int) -> None:
        if not self.windows:
            return
        w = self.windows[self.focus % len(self.windows)]
        before = (w["top"], w["left"])
        w["top"] = int(np.clip(w["top"] + dy, 0, self.height - w["height"] - 12))
        w["left"] = int(np.clip(w["left"] + dx, 0, self.width - w["width"]))
        if (w["top"], w["left"]) != before:
            # Упёршееся в край окно не двинулось, значит и в истине не двинулось.
            self._moved.add(w["id"])

    def _left(self) -> None:
        self._shift_focused(0, -8)

    def _right(self) -> None:
        self._shift_focused(0, 8)

    def _up(self) -> None:
        self._shift_focused(-6, 0)

    def _down(self) -> None:
        self._shift_focused(6, 0)

    def _next_window(self) -> None:
        if self.windows:
            self.focus = (self.focus + 1) % len(self.windows)

    def _close(self) -> None:
        # Закрыть окно необратимо: вернуть его нечем.
        if self.windows:
            self.windows.pop(self.focus % len(self.windows))
            self.focus = 0

    def apply(self, action: Action) -> None:
        if action.kind is ActionKind.MOUSE_MOVE:
            self._shift_focused(action.dy, action.dx)
            return
        for out in action.outputs_touched():
            effect = self._effects.get(out)
            if effect is not None:
                effect()

    def content(self) -> np.ndarray:
        frame = self.background.copy()
        for i, w in enumerate(self.windows):
            y0, x0 = w["top"], w["left"]
            y1, x1 = y0 + w["height"], x0 + w["width"]
            frame[y0:y1, x0:x1] = w["shade"]
            edge = 235 if i == self.focus % max(1, len(self.windows)) else 150
            frame[y0:y0 + 1, x0:x1] = edge
            frame[y1 - 1:y1, x0:x1] = edge
            frame[y0:y1, x0:x0 + 1] = edge
            frame[y0:y1, x1 - 1:x1] = edge
            frame[y0 + 4:y1 - 4, x0 + 4:x1 - 4] = w["shade"] - 25
        return frame

    def state_snapshot(self) -> dict[str, Any]:
        return {"focus": self.focus,
                "windows": [(w["top"], w["left"]) for w in self.windows]}


# ---------------------------------------------------------------------------
# Видео: мир меняется сам
# ---------------------------------------------------------------------------


class VideoDomain(LayeredDomain):
    """Проигрыватель: содержимое идёт само, полоса управления стоит.

    Здесь впервые появляется третьи часы: `t_content` — время внутри
    просматриваемого содержимого. Оно не совпадает ни с циклами агента, ни с
    тиками мира: на паузе идёт мир, но не содержимое, а при перемотке содержимое
    прыгает назад, хотя мир идёт вперёд.
    """

    name = "video"

    def __init__(self, profile: Profile, *, seed: int = 0, n_outputs: int = 16) -> None:
        super().__init__(profile, seed=seed, n_outputs=n_outputs)
        self.reel = self._make_reel()
        self.position = 0.0
        self.playing = True
        self.t_content = 0.0
        controls = [self._toggle_play, self._seek_forward, self._seek_back,
                    self._restart]
        order = self._rng.permutation(len(self.outputs))
        self._effects: dict[str, Any] = {}
        for i, out in enumerate(self.outputs):
            slot = int(order[i])
            self._effects[out] = controls[slot] if slot < len(controls) else None

    def default_chrome(self) -> list[Chrome]:
        w, h = self.width, self.height
        return [
            Chrome(h - 16, 0, 16, w, animated=True, label="полоса управления"),
            Chrome(4, w - 40, 10, 36, animated=False, label="индикатор качества"),
        ]

    def _make_reel(self, frames: int = 96) -> np.ndarray:
        """Плёнка: кадры, отличающиеся друг от друга, но связные по времени."""
        rng = np.random.default_rng(self.seed + 4)
        small = rng.integers(30, 220, (frames, self.height // 6 + 1,
                                       self.width // 6 + 1), dtype=np.uint8)
        # Сглаживание по времени: соседние кадры похожи, как в настоящем видео.
        smooth = np.empty_like(small, dtype=np.float32)
        acc = small[0].astype(np.float32)
        for i in range(frames):
            acc = acc * 0.7 + small[i].astype(np.float32) * 0.3
            smooth[i] = acc
        return smooth.round().astype(np.uint8)

    # --- действия -----------------------------------------------------------

    def _toggle_play(self) -> None:
        self.playing = not self.playing

    def _seek_forward(self) -> None:
        self.position = min(len(self.reel) - 1.0, self.position + 6.0)

    def _seek_back(self) -> None:
        self.position = max(0.0, self.position - 6.0)

    def _restart(self) -> None:
        self.position = 0.0

    def apply(self, action: Action) -> None:
        for out in action.outputs_touched():
            effect = self._effects.get(out)
            if effect is not None:
                effect()

    def advance(self) -> None:
        """Содержимое идёт само — если не на паузе. Мир идёт всегда."""
        if self.playing:
            # Ровно кадр за шаг. Дробный шаг давал бы через раз то нулевое
            # изменение, то полное — фон получается рваным, разброс огромным, и
            # на его фоне не видно уже никакого последствия действия. На замере
            # из-за этого не находилось ни одного живого выхода из четырёх.
            self.position = (self.position + 1.0) % len(self.reel)
        self.t_content = float(self.position / max(1e-9, self.fps))

    def content(self) -> np.ndarray:
        return _upscale_to(self.reel[int(self.position) % len(self.reel)], 6,
                           self.height, self.width)

    def state_snapshot(self) -> dict[str, Any]:
        return {"position": round(float(self.position), 3), "playing": self.playing}


# ---------------------------------------------------------------------------
# Игра — тот же интерфейс поверх уже существующего мира
# ---------------------------------------------------------------------------


class GameDomain:
    """Обёртка над `InteractiveWorld`, чтобы у всех доменов был один интерфейс."""

    name = "game"

    def __init__(self, profile: Profile, *, seed: int = 0, n_outputs: int = 24) -> None:
        self.world = InteractiveWorld(profile, seed=seed, n_outputs=n_outputs)
        self.outputs = self.world.outputs
        self.profile = profile
        self.seed = seed

    def step(self, action: Action | None = None, *,
             with_audio: bool = True) -> Observation:
        return self.world.step(action, with_audio=with_audio)

    def truth(self) -> dict[str, Any]:
        return {"domain": self.name, **self.world.truth()}

    def screen_mask(self) -> np.ndarray:
        return self.world.hud_mask()

    def animated_mask(self) -> np.ndarray:
        return self.world.animated_mask()

    @property
    def t_content(self) -> float | None:
        return None

    @property
    def last_action_changed(self) -> bool | None:
        return self.world.last_action_changed


def _depth_domain(profile: Profile, *, seed: int = 0) -> Any:
    """Мир с настоящей глубиной. Импорт отложен: он тянет свой модуль."""
    from .depth import DepthWorld

    return DepthWorld(profile, seed=seed)


DOMAINS: dict[str, Any] = {
    "game": GameDomain,
    "document": DocumentDomain,
    "desktop": DesktopDomain,
    "video": VideoDomain,
    "depth": _depth_domain,
}


def make_domain(name: str, profile: Profile, *, seed: int = 0) -> Domain:
    try:
        factory = DOMAINS[name]
    except KeyError:
        raise KeyError(f"нет домена {name!r}; есть {sorted(DOMAINS)}") from None
    return factory(profile, seed=seed)
