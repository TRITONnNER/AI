"""Кросс-доменный замер: одни и те же модули по разным мирам.

«Если решение работает только в Minecraft, оно неправильное» — проверяется здесь и
только здесь. Каждый домен прогоняется одним и тем же кодом, и получается таблица,
по которой видно, что где работает, а что нет.

Важное про чтение таблицы. Пустая клетка — не всегда провал, и признак, которым
получен ответ, стоит рядом с самим ответом. Разделять слои есть чем двумя способами:
по параллаксу (что сместилось вместе со всем кадром) и по неподвижности (что не
меняется, когда меняется остальное). Второй слабее — он не различает «прибито к
экрану» и «стоит на месте», — поэтому в отчёте видно, кто ответил.

Судится при этом ответ, а не выбор признака, и это стоило одной отброшенной гипотезы.
Сначала здесь было записано, что на рабочем столе параллакс обязан молчать: общего
сдвига там нет. Замер по трём сидам показал другое: когда крупное окно едет
согласованно, фазовая корреляция находит сдвиг по этому окну — и ответ выходит
верным (точность 1.00, IoU 0.73–0.77), потому что едущее окно и есть содержимое, а
остальное и есть экранный слой. Требование молчать наказывало бы за правильный
ответ. Поэтому графа «ожидаемо» осталась пояснением, а не критерием.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .behaviour.babbling import Babbler, run_babbling
from .core.clocks import Clocks
from .core.profile import Profile, from_schema
from .corpus.domains import DOMAINS, make_domain
from .model.places import PlaceGraph, fingerprint
from .vision.predict import PredictionError
from .vision.selfworld import (COUPLING, MERGED, NO_SIGNAL, PARALLAX, SCREEN,
                               STILLNESS, STRENGTH, UNDECIDED, LayerArbiter,
                               frame_energy)

# Что мы считаем правильным поведением в каждом домене. Не «сколько получилось», а
# «чего мы вообще ждём» — иначе честный отказ выглядел бы провалом.
EXPECTATION = {
    "game": "все три признака: камера то панорамирует, то стоит — работает и "
            "параллакс, и связь изменений с движением",
    "document": "все три: прокрутка даёт сдвиг по одной оси и остановки между ними",
    "desktop": "только неподвижность: общего сдвига нет, связывать не с чем. "
               "Крупное едущее окно иногда даёт сдвиг по себе, и тогда добавляется "
               "параллакс",
    "video": "только неподвижность: камера не движется вовсе, содержимое идёт само",
}

# Домены, в которых глобального сдвига по построению нет. Это пояснение к таблице, а
# не критерий: см. отброшенную гипотезу в описании модуля.
NO_PARALLAX = frozenset({"desktop", "video"})

# Пороги зачёта. Точность важнее полноты: пропущенный кусок панели — это неполное
# знание, а лишний кусок мира, названный панелью, — это ложное знание, и оно
# отравляет всё, что построено сверху.
#
# Полнота по неподвижному обрамлению не может дойти до единицы, и это свойство
# картинки, а не метода: заливка внутри панели однородна, а на однородном участке
# обе гипотезы дают одно и то же — там нечего различать, и оба признака честно
# оставляют пиксель нерешённым. Замер по шести сидам: игра 0.49–0.55, документ
# 0.63–0.71, рабочий стол 0.71–0.81, видео 1.00. Порог 0.4 ловит настоящую поломку
# и не срабатывает от дрожания на границе — порог 0.5 стоял бы ровно посреди
# измеренного разброса игры и переключался бы от сида.
MIN_PRECISION = 0.9
MIN_RECALL_STATIC = 0.4


@dataclass(slots=True)
class DomainResult:
    domain: str
    frames: int
    # разделение слоёв: признак и то, что получилось по выбранному признаку
    signal: str
    signal_reason: str
    iou: float | None
    recall: float | None
    precision: float | None
    recall_static: float | None
    recall_animated: float | None
    undecided: float
    # каждый признак по отдельности — чтобы видеть, кто чего стоит
    parallax_iou: float | None
    parallax_decided: float
    parallax_frames: int
    stillness_iou: float | None
    stillness_decided: float
    stillness_frames: int
    coupling_iou: float | None
    coupling_decided: float
    coupling_moving_frames: int
    coupling_still_frames: int
    # Пиксели, где два применимых признака ответили по-разному. Расхождение значит,
    # что один из них врёт, и его надо видеть, а не усреднять.
    disagreements: int
    disagreement_fraction: float
    decided_by: dict[str, int]
    # ошибка предсказания
    error_mean: float
    error_spikes: int
    shift_vs_copy: float
    # граф мест
    places: int
    edges: int
    # тело
    mask_unstable: bool
    live_found: int
    live_true: int
    silent_found: int
    silent_true: int
    ambiguous: int
    faint: int
    wrong_body: int
    background_rate: float
    # содержимое
    t_content_moves: bool
    expected: str = ""
    verdict: str = ""

    def as_dict(self) -> dict[str, Any]:
        def r(x: float | None, n: int = 3) -> float | None:
            return None if x is None else round(x, n)

        return {
            "domain": self.domain, "frames": self.frames,
            "signal": self.signal, "signal_reason": self.signal_reason,
            "iou": r(self.iou), "recall": r(self.recall),
            "precision": r(self.precision),
            "recall_static": r(self.recall_static),
            "recall_animated": r(self.recall_animated),
            "undecided": r(self.undecided, 4),
            "parallax_iou": r(self.parallax_iou),
            "parallax_decided": r(self.parallax_decided, 4),
            "parallax_frames": self.parallax_frames,
            "stillness_iou": r(self.stillness_iou),
            "stillness_decided": r(self.stillness_decided, 4),
            "stillness_frames": self.stillness_frames,
            "coupling_iou": r(self.coupling_iou),
            "coupling_decided": r(self.coupling_decided, 4),
            "coupling_moving_frames": self.coupling_moving_frames,
            "coupling_still_frames": self.coupling_still_frames,
            "disagreements": self.disagreements,
            "disagreement_fraction": r(self.disagreement_fraction, 4),
            "decided_by": self.decided_by,
            "error_mean": r(self.error_mean, 6),
            "error_spikes": self.error_spikes,
            "shift_vs_copy": r(self.shift_vs_copy),
            "places": self.places, "edges": self.edges,
            "live": f"{self.live_found}/{self.live_true}",
            "silent": f"{self.silent_found}/{self.silent_true}",
            "ambiguous": self.ambiguous, "faint": self.faint,
            "wrong_body": self.wrong_body,
            "background_rate": r(self.background_rate, 4),
            "t_content_moves": self.t_content_moves,
            "mask_unstable": self.mask_unstable,
            "expected": self.expected, "verdict": self.verdict,
        }


def _score(found: np.ndarray, truth: np.ndarray) -> tuple[float | None, float | None,
                                                          float | None]:
    """IoU, полнота, точность. `None` там, где считать нечего, а не ноль.

    Разница существенная: ноль значит «метод ошибся», `None` — «сверять не с чем».
    Спутать их значит записать отсутствие данных в провал.
    """
    if not truth.any():
        return None, None, None
    inter = int((found & truth).sum())
    union = int((found | truth).sum())
    iou = inter / union if union else None
    recall = inter / int(truth.sum())
    precision = inter / int(found.sum()) if found.any() else None
    return iou, recall, precision


class _TruthWatcher:
    """Обёртка домена, которая для замера считает, какие выходы **проявились**.

    Зачем не брать истину из проводки. Проводка говорит, к чему выход подключён, а
    не что он сделал. Выход, подключённый к «закрыть окно», после того как окна
    кончились, объективно ничего не делает; выход «перемотать» в остановленном
    проигрывателе — тоже. Требовать от агента вывода «этот выход отвечает» там, где
    выход за всю запись ни разу ничего не изменил, значит требовать угадывания.

    Поэтому истина здесь такая: выход считается живым, если хотя бы раз при
    одиночной пробе изменилось **внутреннее состояние домена**. Внутреннее
    состояние — истина исследователя (`domain.last_action_changed`), а не пиксели, и
    агентскому коду она недоступна: обёртка живёт в замере и не отдаёт наружу
    ничего, кроме того, что отдаёт домен.

    Именно `last_action_changed`, а не `Observation.changed`: второе включает то, что
    мир сделал сам. В проигрывателе содержимое идёт каждый кадр, и по общему
    изменению живыми оказались бы все шестнадцать выходов разом — а вместе с ними
    оказался бы «правильным» агент, приписывающий себе чужое движение.

    Сочетания в счёт не идут: если нажаты два выхода и что-то изменилось, неизвестно,
    какой из них подействовал. Приписать заслугу обоим значило бы соврать в истине.

    Планка повторяемости у истины та же, что у агента (`bar`). Выход, подействовавший
    ровно один раз за сорок проб, попадает в третью категорию — `ambiguous`, — и в
    зачёт не идёт ни в ту, ни в другую сторону. Так получается потому, что в этих
    домах есть необратимые действия: выход, закрывающий окна, после третьего раза
    объективно ничего не делает, и «правильного» ответа для него на всю запись просто
    нет. Требовать одного ответа значило бы требовать угадывания, а считать его
    молчащим — наказывать за верное наблюдение.
    """

    def __init__(self, domain: Any, *, bar: int = 2, sigmas: float = 3.0,
                 floor: float = 0.001) -> None:
        self._domain = domain
        self.bar = int(bar)
        self.sigmas = float(sigmas)
        self.floor = float(floor)
        self.changes: dict[str, int] = {}
        self.probes: dict[str, int] = {}
        # Величина изменения кадра при пробах, которые действительно подействовали, и
        # на холостом ходу. Нужна третьей категории истины — см. `observed_faint`.
        self.energy: dict[str, list[float]] = {}
        self.idle_energy: list[float] = []
        self._prev_frame: Any = None

    def _threshold(self) -> float:
        """Тот же порог заметности, что у агента, но посчитанный по истине."""
        if len(self.idle_energy) < 3:
            return self.floor
        arr = self.idle_energy
        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / len(arr)
        return mean + max(self.floor, self.sigmas * var ** 0.5)

    @property
    def observed_faint(self) -> set[str]:
        """Подействовали, но слабее, чем мир шумит сам.

        Третья честная категория. Выход, у которого последствие есть, но его размер
        в пикселях ниже порога заметности, не отличим от молчащего **по тем данным,
        которые у агента есть**. Пример из замера: смена окна в фокусе перекрашивает
        рамку — 284 пикселя из 57 600, изменение кадра 0.0016 против порога 0.002.
        Требовать такого вывода значит требовать не восприятия, а доступа к истине.

        Категория определена по величине изменения кадра и по уровню собственного
        шума мира — и то и другое свойства мира, а не ответа метода. Иначе это была
        бы подгонка: «неправильно там, где метод не справился».
        """
        thr = self._threshold()
        out = set()
        for o, n in self.changes.items():
            if n < self.bar:
                continue
            e = sorted(self.energy.get(o, []))
            if not e:
                continue
            median = e[len(e) // 2]
            if median <= thr:
                out.add(o)
        return out

    @property
    def observed_live(self) -> set[str]:
        """Выходы, подействовавшие заметно и с запасом по числу проявлений.

        Запас — не придирка. Чтобы заключить «отвечает», агенту нужно `bar`
        **обнаружений**, а каждое обнаружение может не состояться: последствие бывает
        на границе шума. Если последствие проявилось ровно `bar` раз, у агента нет ни
        одного запаса: единственный пропуск делает верный вывод недостижимым, и
        правильным поведением остаётся воздержаться. Поэтому истина требует вдвое:
        `2 × bar` проявлений.

        Замер, из которого это взялось: на рабочем столе выход «следующее окно» и
        выход «вверх» проявились по два раза за тридцать восемь проб (окно почти
        всегда упёрто в край), с изменением кадра 0.004–0.0075 при пороге 0.0032.
        Агент увидел по одному из двух и отказался делать вывод — и это правильный
        отказ, а не ошибка.
        """
        return {o for o, n in self.changes.items()
                if n >= 2 * self.bar} - self.observed_faint

    @property
    def observed_silent(self) -> set[str]:
        """Выходы, пробованные в одиночку и не подействовавшие ни разу."""
        return {o for o, n in self.probes.items() if self.changes.get(o, 0) == 0}

    @property
    def observed_ambiguous(self) -> set[str]:
        """Проявились, но без запаса: правильного ответа на эту запись нет."""
        return {o for o, n in self.changes.items() if 0 < n < 2 * self.bar}

    @property
    def outputs(self) -> tuple[str, ...]:
        return self._domain.outputs

    @property
    def name(self) -> str:
        return self._domain.name

    def step(self, action: Any = None, *, with_audio: bool = True) -> Any:
        obs = self._domain.step(action, with_audio=with_audio)
        prev, self._prev_frame = self._prev_frame, obs.frame
        energy = None if prev is None else frame_energy(prev, obs.frame)
        acted = action is not None and not action.masked
        if not acted:
            if energy is not None:
                self.idle_energy.append(energy)
        else:
            touched = list(action.outputs_touched())
            if len(touched) == 1:
                out = touched[0]
                self.probes[out] = self.probes.get(out, 0) + 1
                if self._domain.last_action_changed:
                    self.changes[out] = self.changes.get(out, 0) + 1
                    if energy is not None:
                        self.energy.setdefault(out, []).append(energy)
        return obs

    def truth(self) -> dict[str, Any]:
        return {**self._domain.truth(),
                "observed_live": sorted(self.observed_live),
                "observed_silent": sorted(self.observed_silent),
                "observed_ambiguous": sorted(self.observed_ambiguous)}

    def screen_mask(self) -> np.ndarray:
        return self._domain.screen_mask()

    def animated_mask(self) -> np.ndarray:
        return self._domain.animated_mask()


def bench_domain(name: str, *, seed: int = 0, frames: int = 140,
                 babble_steps: int = 700, motion_px: int = 40,
                 profile: Profile | None = None) -> DomainResult:
    """Прогнать один домен через все модули. Никаких доменных поправок в коде.

    `motion_px` — размах движения мыши в пикселях за шаг. Он влияет на результат
    разделения слоёв сильно и понятно: чем меньше движение относительно размера
    деталей, тем меньше кадров различают гипотезы. Замер размаха — в
    ARCHITECTURE-AGENT.md; здесь взят размах, при котором сдвиг заметно больше
    самих элементов обрамления, потому что мелкое дрожание проверяло бы не метод,
    а порог `flow_min_global_shift`.
    """
    profile = profile or from_schema(f"БЕНЧ-{name}", capture_width=320,
                                     capture_height=180, babble_repeats=2)
    domain = make_domain(name, profile, seed=seed)

    # --- восприятие: слои, ошибка предсказания, места --------------------
    arbiter = LayerArbiter(profile)
    shift_error = PredictionError(profile)
    copy_error = PredictionError(from_schema(f"БЕНЧ-{name}-copy", capture_width=320,
                                            capture_height=180, predictor="copy"))
    graph = PlaceGraph()
    t_content_seen: set[float] = set()

    # Слои замеряются под движением мыши, а не случайными выходами. Причина
    # содержательная: случайные выходы ломают само обрамление — панель можно
    # сломать, окно закрыть, — и тогда сверять голоса не с чем. Мышь есть в любом
    # домене и ничего не разрушает; что она делает, зависит от домена, и это как
    # раз то, что мы проверяем. Знания о том, какие выходы разрушительны, здесь
    # нет — иначе замер подглядывал бы в истину.
    rng = np.random.default_rng(seed + 100)
    from .core.action import Action

    # Истина обрамления меняется за прогон: панель можно сломать, окно — закрыть
    # или сдвинуть. Голоса накоплены по всей записи, поэтому сверять их с маской
    # последнего кадра нельзя — это разные вещи.
    #
    # Поэтому копим три маски: где обрамление было всегда, где оно было хоть раз,
    # и где было меняющееся обрамление. Пиксель, у которого истина за прогон
    # менялась, из подсчёта выпадает целиком — и из истины, и из ответа. Иначе
    # выходит подлог в обе стороны: пиксель, бывший фоном большую часть прогона,
    # честно получает голос «неподвижный», а истина последнего кадра называет его
    # окном, и метод наказан за правильный ответ. На рабочем столе это одно
    # съедало точность с 0.9 до 0.49.
    always_screen: np.ndarray | None = None
    ever_screen: np.ndarray | None = None
    always_animated: np.ndarray | None = None
    mask_unstable = False

    for i in range(frames):
        act = None
        if i % 3 != 0:
            dx = int(rng.integers(-motion_px, motion_px + 1))
            dy = int(rng.integers(-motion_px * 2 // 3, motion_px * 2 // 3 + 1))
            if dx or dy:
                act = Action.mouse(dx, dy, duration_ms=33)
        obs = domain.step(act, with_audio=False)
        arbiter.feed(obs.frame)
        shift_error.feed(obs.frame)
        copy_error.feed(obs.frame)
        graph.observe(fingerprint(obs.frame), i, seconds_per_seq=1.0 / 30.0)
        mask_now = domain.screen_mask()
        anim_now = domain.animated_mask()
        if always_screen is None:
            always_screen = mask_now.copy()
            ever_screen = mask_now.copy()
            always_animated = anim_now.copy()
        else:
            assert ever_screen is not None and always_animated is not None
            if not np.array_equal(always_screen, mask_now):
                mask_unstable = True
            always_screen &= mask_now
            ever_screen |= mask_now
            always_animated &= anim_now
        content = getattr(domain, "t_content", None)
        if content is not None:
            t_content_seen.add(round(float(content), 4))

    verd = arbiter.result()
    assert always_screen is not None and ever_screen is not None
    assert always_animated is not None
    scoreable = ~(ever_screen & ~always_screen)
    truth_mask = always_screen & scoreable
    found = verd.pixel_mask(SCREEN) & scoreable
    iou, recall, precision = _score(found, truth_mask)
    par_iou, _, _ = _score(verd.parallax.pixel_mask(SCREEN) & scoreable, truth_mask)
    still_iou, _, _ = _score(verd.stillness.pixel_mask(SCREEN) & scoreable, truth_mask)
    coup_iou, _, _ = _score(verd.coupling.pixel_mask(SCREEN) & scoreable, truth_mask)

    # Полнота отдельно по неподвижной и по меняющейся части обрамления. Разделение
    # взято из истины домена (`Chrome.animated`, `HudRect.animated`), а не из
    # ответа метода, — иначе это было бы подгонкой критерия под результат.
    static_truth = truth_mask & ~always_animated
    anim_truth = truth_mask & always_animated
    _, recall_static, _ = _score(found, static_truth)
    _, recall_animated, _ = _score(found, anim_truth)

    # --- тело: лепет ------------------------------------------------------
    fresh = _TruthWatcher(
        make_domain(name, profile, seed=seed),
        bar=int(profile.parameters["body_live_min_responses"]),
        sigmas=float(profile.parameters["babble_response_sigmas"]),
        floor=float(profile.parameters["babble_response_floor"]))
    babbler = Babbler(profile, fresh.outputs, rng_seed=seed)
    babble = run_babbling(fresh, babbler, steps=babble_steps, clocks=Clocks())
    live_true = fresh.observed_live
    silent_true = fresh.observed_silent
    said_live = set(babbler.body.by_state("live"))
    said_silent = set(babbler.body.by_state("silent"))
    # Уверенно наоборот. Это единственная категория, которая портит всё дальнейшее:
    # неполное знание достраивается пробами, ложное — нет.
    wrong_body = len(said_live & silent_true) + len(said_silent & live_true)

    ratio = (copy_error.summary()["mean"] / max(1e-9, shift_error.summary()["mean"]))

    res = DomainResult(
        domain=name, frames=frames, mask_unstable=mask_unstable,
        signal=verd.signal, signal_reason=verd.reason,
        iou=iou, recall=recall, precision=precision,
        recall_static=recall_static, recall_animated=recall_animated,
        undecided=float((verd.labels == UNDECIDED).mean()),
        parallax_iou=par_iou,
        parallax_decided=verd.parallax.decided_fraction,
        parallax_frames=verd.parallax.voting_frames,
        stillness_iou=still_iou,
        stillness_decided=verd.stillness.decided_fraction,
        stillness_frames=verd.stillness.voting_frames,
        coupling_iou=coup_iou,
        coupling_decided=verd.coupling.decided_fraction,
        coupling_moving_frames=verd.coupling.moving_frames,
        coupling_still_frames=verd.coupling.still_frames,
        disagreements=verd.disagreements,
        disagreement_fraction=verd.disagreement_fraction,
        decided_by={s: int(verd.by_signal(s).sum()) for s in STRENGTH},
        error_mean=float(shift_error.summary()["mean"]),
        error_spikes=len(shift_error.spikes),
        shift_vs_copy=float(ratio),
        places=len(graph), edges=len(graph.edges),
        live_found=len(said_live & live_true), live_true=len(live_true),
        silent_found=len(said_silent & silent_true), silent_true=len(silent_true),
        ambiguous=len(fresh.observed_ambiguous), faint=len(fresh.observed_faint),
        wrong_body=wrong_body,
        background_rate=float(babble["background_rate"]),
        t_content_moves=len(t_content_seen) > 1,
        expected=EXPECTATION.get(name, ""),
    )
    res.verdict = verdict(res)
    return res


def _layers_ok(res: DomainResult) -> tuple[bool, str]:
    """Правильно ли повёл себя разделитель слоёв: и ответ, и выбор признака.

    Критерий из двух частей, и обе части сформулированы через то, что признак
    в принципе способен увидеть:

    1. **Точность не ниже `MIN_PRECISION`.** Уверенная ерунда хуже отказа, и это
       главное требование: то, что метод назвал обрамлением, обрамлением и должно
       быть.
    2. **Полнота по неподвижной части обрамления не ниже `MIN_RECALL_STATIC`.**
       Меняющаяся часть — ползущие заполнения полос — не отделяется ни одним из
       двух признаков: она не смещается вместе с миром и при этом не совпадает с
       собой. Требовать её нахождения значило бы требовать третьего признака,
       которого нет. Поэтому полнота по ней остаётся в таблице отдельной графой
       как известный пробел, а не растворяется в общем числе.

    Общий IoU при этом тоже считается и печатается: он ниже как раз на величину
    этого пробела, и по нему видно, сколько пробел стоит.
    """
    if res.iou is None:
        return True, "обрамление исчезло за прогон, сверять не с чем"
    if res.signal == NO_SIGNAL:
        return False, "оба признака промолчали"
    signal_name = {PARALLAX: "параллакс", STILLNESS: "неподвижность",
                   COUPLING: "связь с движением",
                   MERGED: "три признака вместе"}[res.signal]
    precision = res.precision if res.precision is not None else 0.0
    if precision < MIN_PRECISION:
        return False, (f"{signal_name}: уверенная ерунда, точность "
                       f"{precision:.2f} < {MIN_PRECISION}")
    static = res.recall_static
    if static is None:
        return True, (f"{signal_name}: неподвижного обрамления нет, "
                      f"точность {precision:.2f}")
    if static < MIN_RECALL_STATIC:
        return False, (f"{signal_name}: неподвижное обрамление найдено на "
                       f"{static:.0%} < {MIN_RECALL_STATIC:.0%}")
    gap = "" if res.recall_animated is None else (
        f", меняющееся на {res.recall_animated:.0%}")
    return True, (f"{signal_name}: неподвижное обрамление на {static:.0%}, "
                  f"точность {precision:.2f}{gap}")


def _body_ok(res: DomainResult) -> tuple[bool, str]:
    """Правильно ли открыто тело.

    Зачёт по уверенным ошибкам, а не по полноте. Причина в том, что полнота здесь
    ограничена самим миром: в домене есть необратимые действия, и выход, который
    подействовал один раз из сорока, честно неопределим — про него нет правильного
    ответа на всю запись. Такие выходы считаются отдельно (`ambiguous`) и в зачёт не
    идут.

    Уверенная ошибка — другое дело. «Отвечает» про выход, который не сделал ничего,
    или «молчит» про выход, который повторяемо действовал, — это ложное знание, и
    оно не достраивается пробами, а отравляет всё, что построено сверху.
    """
    if res.wrong_body:
        return False, (f"тело: {res.wrong_body} уверенно наоборот "
                       f"(живых {res.live_found}/{res.live_true}, "
                       f"молчащих {res.silent_found}/{res.silent_true})")
    tails = []
    if res.ambiguous:
        tails.append(f"неопределимых по записи {res.ambiguous}")
    if res.faint:
        tails.append(f"незаметных в пикселях {res.faint}")
    tail = ("; " + ", ".join(tails)) if tails else ""
    if res.live_found == res.live_true and res.silent_found == res.silent_true:
        return True, f"тело открыто полностью{tail}"
    return True, (f"тело открыто без ошибок: живых {res.live_found}/{res.live_true}, "
                  f"молчащих {res.silent_found}/{res.silent_true}{tail}")


def verdict(res: DomainResult) -> str:
    """Правильно ли повёл себя харнесс в этом домене.

    Для домена без параллакса правильным считается молчание параллакса и ответ
    второго признака. Иначе таблица наказывала бы за честность и премировала бы
    за уверенную ерунду.
    """
    _, layers = _layers_ok(res)
    _, body = _body_ok(res)
    return f"{layers}; {body}"


@dataclass(slots=True)
class Report:
    results: list[DomainResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"domains": [r.as_dict() for r in self.results],
                "ok": all(self.passed(r) for r in self.results)}

    @staticmethod
    def passed(res: DomainResult) -> bool:
        body_ok, _ = _body_ok(res)
        layers_ok, _ = _layers_ok(res)
        return body_ok and layers_ok

    def table(self) -> str:
        rows = [("домен", "признак", "IoU", "точн.", "полн. неподв.",
                 "полн. аним.", "не реш.", "расх.", "паралл.", "неподв.", "связь",
                 "ошибка", "всплеск", "сдвиг/копия", "места", "живые", "молчащие",
                 "t_content")]

        def num(x: float | None) -> str:
            return "—" if x is None else f"{x:.2f}"

        for r in self.results:
            rows.append((
                r.domain,
                {PARALLAX: "параллакс", STILLNESS: "неподвижн.",
                 COUPLING: "связь", MERGED: "три вместе",
                 NO_SIGNAL: "нет"}[r.signal],
                num(r.iou), num(r.precision),
                num(r.recall_static), num(r.recall_animated),
                f"{r.undecided * 100:.0f}%",
                f"{r.disagreement_fraction * 100:.1f}%",
                num(r.parallax_iou), num(r.stillness_iou), num(r.coupling_iou),
                f"{r.error_mean:.4f}",
                str(r.error_spikes),
                f"{r.shift_vs_copy:.2f}",
                str(r.places),
                f"{r.live_found}/{r.live_true}",
                f"{r.silent_found}/{r.silent_true}",
                "идёт" if r.t_content_moves else "—",
            ))
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        lines = []
        for n, row in enumerate(rows):
            lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
            if n == 0:
                lines.append("  ".join("-" * w for w in widths))
        return "\n".join(lines)


def run_all(*, seed: int = 0, frames: int = 140, babble_steps: int = 700,
            motion_px: int = 40, domains: list[str] | None = None) -> Report:
    names = domains or list(DOMAINS)
    return Report([bench_domain(n, seed=seed, frames=frames, motion_px=motion_px,
                                babble_steps=babble_steps) for n in names])
