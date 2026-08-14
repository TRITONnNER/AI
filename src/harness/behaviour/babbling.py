"""Лепет: агент открывает своё тело нажатиями.

Он не знает ни одного названия клавиши, ни одной функции, ни того, сколько
выходов вообще что-то делают. Он знает только, что выходы адресуемы. Всё
остальное добывается пробами — и это единственный способ, которым карта тела
может возникнуть, не будучи подсказкой.

Три вещи, которые делают лепет осмысленным, а не случайным перебором:

1. **Длительность — часть пробы.** Один и тот же выход, зажатый на 40 мс и на
   400 мс, может дать разное. Поэтому проба — это `(выход, удержание)`, и
   удержания перебираются тоже (инвариант 8).
2. **Обратимость проверяется попыткой откатить, а не берётся из таблицы.**
   После пробы, что-то изменившей, агент пытается вернуть как было — тем же
   выходом или другим. Получилось — обратимость растёт, не получилось — падает.
   Осторожность выводится отсюда и только отсюда (инварианты 9 и 10).
3. **Пауза после необратимого.** Если откатить не удалось, лепет замирает на
   `pause_after_irreversible_ms`: последствие надо увидеть, а не затоптать
   следующей пробой.

Порядок проб не случаен: сначала непробованное (там больше всего неизвестного),
потом то, где мало повторов, и в конце сочетания. Осторожные выходы отодвигаются
в хвост — но не исключаются: исключить значило бы завести список запретов.

## Цена ошибки, а не только полнота знания

Порядок проб долго определялся **одной** величиной — полнотой знания (сколько проб
уже потрачено, известна ли обратимость). Цену ошибки он не читал вовсе, и это
установил замер, а не чтение кода: счётчики нажатий необратимого выхода совпали до
единицы при работающем пороге и при снятом на пяти сидах из пяти. Диагностика
показала, откуда берутся нажатия — из ветвей «ни разу не пробован» (одно, и его не
избежать: заранее про выход не известно ничего), «недобранный повтор» (два) и
«обратимость неизвестна» (до четырёх). Ни одна из этих трёх ветвей порога не
спрашивала. Порог читала только последняя ветвь, «уточнение», и она необратимый
выход и так не выбирала.

Поэтому здесь появляется вторая величина — `error_cost`, цена ошибки. Она отделена
от `Reversibility` нарочно, и это разделение содержательное, а не оформительское:

- `reversibility` — **про мир**. Неудачная догадка об откате её не портит: она
  говорит о догадке, а не о том, обратимо ли последствие.
- `error_cost` — **про риск**. Она считает и неудачные догадки тоже, потому что
  инвариант 9 определяет осторожность как «я не умею это откатить», а неудачная
  догадка — ровно этот случай.

Цена входит в порядок ключом (дорогое — в хвост ветви, но не вне её) и отдельно
снимает те повторы, которые **ничего не решают**: если состояние выхода уже выведено,
повтор не изменит вывода, а цену заплатит. Это и есть «полнота знания и цена ошибки» в
одном сравнении, а не список запретов.

И есть граница, за которую цена не заходит. **Ветвь «обратимость неизвестна» не
пропускается никогда**, даже когда дорого в ней всё: это единственная ветвь, ради
которой цену вообще платят, — в ней находятся обратные пары. Промежуточная редакция
пропускала её, и обратных пар стало находиться 2 вместо 8 при неизменном риске. Цена
решает, **что раньше**, но не отменяет узнавание: осторожность, запрещающая узнавать,
— это слепота, а не осторожность.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

from ..core.action import Action, Reversibility, too_risky
from ..core.branches import Arbitration
from ..core.clocks import Stamp
from ..core.journal import (Actor, ActorLayer, Journal, Kind as EntryKind,
                            StateSnapshot)
from ..core.profile import Profile
from ..model.rebuild import BodyMap

# Потолок темпа исследования. Не «на всякий случай»: без него коэффициент усиления
# при скуке мог бы дать сотню проб за такт, и один такт съел бы весь бюджет прогона.
PACE_MAX = 4.0

# Ветви арбитража лепета. Имя ветви — начало `Probe.why`, потому что по журналу ветвь
# читается именно оттуда: два разных набора имён для одного и того же разъехались бы
# при первой правке, и сверить счётчики с журналом стало бы нельзя.
UNTRIED = "ни разу не пробован"
THIN = "недобранный повтор"
UNKNOWN_UNDO = "обратимость неизвестна"
COMBO = "сочетание двух живых выходов"
REFINE = "уточнение"

#: Ветви в порядке приоритета, с объяснением зачем каждая. Инвариант 26: счётчик по
#: необъявленному списку врёт умолчанием, поэтому список объявлен здесь, а не
#: вычисляется из кода выбора.
BABBLE_BRANCHES = (
    (UNTRIED, "выход, которого ни разу не касались: там больше всего неизвестного"),
    (THIN, "повторов не хватает, чтобы решить, отвечает выход или молчит"),
    (UNKNOWN_UNDO, "живой выход, обратимость которого ещё не выяснена"),
    (COMBO, "два живых выхода вместе: одиночные уже разобраны"),
    (REFINE, "всё разобрано, уточняем там, где разброс шире"),
)
BRANCHES = tuple(name for name, _ in BABBLE_BRANCHES)


def babble_arbitration() -> Arbitration:
    """Счётчики ветвей лепета. Инвариант 26.

    Отдельная функция, а не поле по умолчанию: одни и те же счётчики нужны и внутри
    одного лепета, и общими на серию прогонов (`tools/measure_branches.py`).
    """
    return Arbitration.of("лепет", BABBLE_BRANCHES)


@dataclass(frozen=True, slots=True)
class Probe:
    """Одна проба: что нажать и как долго."""

    output: str
    duration_ms: int
    modifiers: tuple[str, ...] = ()
    why: str = ""              # почему выбрана именно эта проба; для журнала

    def to_action(self, reversibility: Reversibility) -> Action:
        return Action.key(self.output, self.duration_ms, self.modifiers,
                          reversibility=reversibility)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Что вышло. `changed` и `undone` — объективные наблюдения, не мнения."""

    probe: Probe
    delivered: bool
    changed: bool
    undone: bool | None        # None — откат не пробовали
    error_value: float | None = None
    undo_output: str | None = None      # чем пробовали откатить
    undo_was_known: bool = False        # это был уже найденный способ или догадка

    def as_event(self) -> dict[str, Any]:
        d: dict[str, Any] = {"code": "delivered" if self.delivered else "failed",
                             "responded": self.changed, "why": self.probe.why}
        if self.undone is not None:
            d["undone"] = self.undone
            d["undo_output"] = self.undo_output
            d["undo_was_known"] = self.undo_was_known
        if self.error_value is not None:
            d["prediction_error"] = round(self.error_value, 6)
        return d


class Babbler:
    """Выбирает пробы и обновляет карту тела по результатам.

    Ничего не исполняет сам: исполнение передаётся снаружи функцией. Так лепет
    остаётся проверяемым офлайн — по интерактивному миру, без устройства и без игры.
    """

    def __init__(self, profile: Profile, outputs: Sequence[str], *,
                 body: BodyMap | None = None, journal: Journal | None = None,
                 rng_seed: int = 0,
                 state_extra: Callable[[], dict[str, Any]] | None = None,
                 arb: Arbitration | None = None) -> None:
        import random

        p = profile.parameters
        self.profile = profile
        self.outputs = tuple(dict.fromkeys(outputs))
        if not self.outputs:
            raise ValueError("нечего пробовать: тело без выходов")
        self.body = body or BodyMap(
            live_min_responses=int(p["body_live_min_responses"]),
            # Порог «молчит» — это и есть `babble_repeats`: столько раз лепет
            # пробует выход, прежде чем считать его молчащим. Отдельная ручка на то
            # же самое разошлась бы с ней при первой же настройке, и вывод «молчит»
            # стал бы недостижим — на замере так и было: пробовалось два раза,
            # требовалось три, и пятнадцать молчащих выходов из двадцати четырёх
            # навсегда остались неясными.
            silent_min_deliveries=int(p["babble_repeats"]),
            response_sigmas=float(p["body_excess_sigmas"]))
        self.journal = journal
        # Чем дополнить срез состояния: драйвы, настроение, активная цель. Лепет их
        # не знает — их знает тот, кто его вызывает, и отдаёт замыканием.
        self._state_extra = state_extra
        self.hold_min = int(p["babble_hold_min_ms"])
        self.hold_max = int(p["babble_hold_max_ms"])
        if self.hold_min > self.hold_max:
            raise ValueError("babble_hold_min_ms больше babble_hold_max_ms")
        self.repeats = int(p["babble_repeats"])
        self.combo_after = int(p["babble_combo_after"])
        self.stale_streak = int(p["babble_stale_streak"])
        self.caution_threshold = float(p["irreversibility_threshold"])
        self.rate = float(p["babble_rate"])
        self._pace: float | None = None   # первый такт не пропускается
        self.paced_out = 0          # тактов пропущено темпом: видно в отчёте
        self.pause_ms = int(p["pause_after_irreversible_ms"])
        self._rng = random.Random(rng_seed)
        self._pause_left_ms = 0
        self.probes_done = 0
        self.irreversible_hits = 0
        # Что уже пробовали в качестве отката для каждого выхода и что сработало.
        # Это тоже добытое знание, а не таблица: «X откатывается Y» выясняется
        # только тем, что Y однажды вернул мир как было.
        self._undo_tried: dict[str, set[str]] = {}
        self.inverse_found: dict[str, str] = {}
        # Выходы, для которых догадки кончились: все живые выходы уже пробованы
        # как откат и ни один не вернул мир. Осторожность у них остаётся
        # максимальной — и это верный итог, а не недоработка.
        self._undo_exhausted: set[str] = set()
        # Инвариант 26: у каждой ветви арбитража счётчик срабатываний. Ветвь с нулём
        # за полный прогон докладывается как дефект — иначе недостижимая ветвь
        # выглядит как реализованное поведение. Здесь это не формальность: именно
        # отсутствие таких счётчиков позволило три ветви подряд не читать цену ошибки
        # и не быть замеченными.
        self.arb = arb if arb is not None else babble_arbitration()
        # Сколько раз цена ошибки что-то изменила: отложила выбор в хвост ветви и
        # сняла бесполезный повтор. Без этих двух чисел «цена читается» — утверждение
        # о коде, а не о поведении (инвариант 25).
        self.deferred_by_cost = 0
        self.repeats_dropped_by_cost = 0
        # Сколько раз проба не состоялась из-за паузы после необратимого. Отдельно от
        # `paced_out`: темп и пауза — разные причины пропуска, и складывать их значит
        # потерять ту, которая объясняет поведение.
        self.paused_out = 0
        # Сколько раз пробовать было уже нечего: все кандидаты ничему не научат.
        self.probed_without_benefit = 0
        # Сколько раз ветвь пропущена целиком, потому что в ней всё дорого. Так может
        # случиться только с недобранными повторами: ветвь обратимости не пропускается
        # никогда — в ней цена платится за знание, которого иначе не будет.
        self.branch_skipped_by_cost = 0

    # --- цена ошибки --------------------------------------------------------

    def _cost_of(self, f: Any) -> float:
        """Цена ошибки по свидетельствам об этом выходе, без обращения к фону тела.

        Три случая, и ровно они выражают инвариант 9 («осторожность — это я не умею это
        откатить»):

        1. **Способ откатить найден и проверен** (`reversibility.is_known`) — цена и есть
           осторожность по этому способу. Прошлые неудачные догадки больше не считаются:
           вопрос «умею ли» уже отвечен «да», и помнить, сколько ключей не подошло
           раньше, значит отвечать на него «нет» навсегда.
        2. **Способ не найден, а попытки были** — цена максимальна. Не «наверное
           опасно», а буквально: последствие видели, вернуть не смогли.
        3. **Попыток не было** — цену берёт фон тела: про этот выход не известно ничего,
           и объявлять его дорогим значило бы завести список запретов, только
           вычисляемый (инвариант 17).

        Пункт 1 добавлен по замеру, и без него всё остальное врало. Промежуточная
        редакция складывала неудачные догадки с наблюдениями в одну оценку — и выход, для
        которого способ **уже нашёлся** после трёх неудачных догадок, оставался дорогим
        навсегда. Проверка «предсказывает ли цена неудачу следующей попытки отката» дала
        тогда **98 % ложных срабатываний** на 17 530 попытках: помеченное дорогим
        откатывалось почти всегда. Метка была не просто шумной, а перевёрнутой.

        Была и вторая промежуточная редакция — цена как доля израсходованных догадок
        (`undo_searches / живых`). Она мерила надежду, а не цену: порог не успевал
        сработать за прогон, и порядок проб снова не зависел от цены.
        """
        r: Reversibility = f.reversibility
        if r.is_known:
            return r.caution
        if int(f.undo_searches) == 0:
            return -1.0                     # свидетельств нет: цену берёт фон тела
        return Reversibility().observe(False).caution

    def base_cost(self) -> float:
        """Сколько обычно стоит ошибка в этом теле: выученный фон, а не константа.

        Ноль в начале — и это не «наверное безопасно», а «ни одного необратимого
        последствия ещё не видел». Обратное умолчание (незнание стоит максимум, как в
        `Reversibility.caution`) отправило бы в хвост сразу всё тело, и разведка
        встала бы совсем: незнание — это то, ради чего проба и делается.

        Знаменатель — выходы, у которых вопрос об откате вообще возникал. Считать по
        всем выходам значило бы делить на число, которое растёт от одного лишь
        появления новых непробованных выходов.
        """
        asked = [f for f in self.body.outputs.values()
                 if f.reversibility.is_known or f.undo_searches > 0]
        if not asked:
            return 0.0
        hard = sum(1 for f in asked if self._cost_of(f) >= 1.0)
        return hard / len(asked)

    def error_cost(self, output: str) -> float:
        """Чего будет стоить ошибка, если последствие этой пробы не удастся откатить.

        Для непробованного выхода это фон тела: про него самого не известно ничего, и
        подставлять максимум значило бы объявить опасным всё неизвестное — то есть
        завести список запретов, только вычисляемый (инвариант 17).

        Для выхода, у которого догадки об откате кончились, цена максимальна: это уже
        не «не знаю», а «перепробовал всё живое и не вернул».

        ## Всякое свидетельство «не умею» обязано быть отзываемым

        Три ошибки подряд в этой функции имели одну форму: состояние, означающее «я не
        сумел откатить», не отменялось после того, как агент способ **находил**. Все три
        нашла одна проверка — «предсказывает ли цена неудачу следующей попытки отката», —
        и каждая давала около 98 % ложных срабатываний:

        1. фон тела применялся как **пол** над своими свидетельствами (при фоне 0.8
           дорогим был и выход с работающим откатом);
        2. неудачные догадки складывались с наблюдениями в одну оценку и не забывались
           после находки способа;
        3. флаг «догадки кончились» (`_undo_exhausted`) держал цену на единице навсегда
           — даже когда `mu` того же выхода была 1.0 при десятке успешных откатов.

        Поэтому здесь порядок такой: **свои свидетельства сильнее любых умолчаний и
        флагов**, а флаг «догадки кончились» читается только пока обратимость неизвестна.
        """
        f = self.body.outputs.get(output)
        if f is None:
            return self.base_cost()
        own = self._cost_of(f)
        if own < 0.0:
            # Своих свидетельств нет. Либо фон тела, либо максимум — если догадки об
            # откате кончились и способ так и не нашёлся.
            return 1.0 if output in self._undo_exhausted else self.base_cost()
        if not f.reversibility.is_known and output in self._undo_exhausted:
            return 1.0
        return own

    def costly(self, output: str, threshold: float) -> bool:
        """Дорого ли ошибиться на этом выходе при этом пороге.

        Сравнение — общее для всего проекта (`core.action.too_risky`), и оно строгое:
        порог 1.0 означает «рискованным не считается ничего», а незнание стоит ровно
        единицу.
        """
        return too_risky(self.error_cost(output), threshold)

    # --- выбор пробы --------------------------------------------------------

    def _holds(self) -> tuple[int, ...]:
        """Три удержания: короткое, среднее, длинное. Больше не нужно — разница
        между 300 и 320 мс не выучивается, а перебор растёт линейно."""
        mid = (self.hold_min + self.hold_max) // 2
        return (self.hold_min, mid, self.hold_max)

    def pace(self, explore_rate: float | None = None) -> int:
        """Сколько проб на этом такте. Темп исследования — ось модуляции.

        Число, а не «да/нет», потому что ось двусторонняя: страх тормозит
        исследование, скука его ускоряет. Доля тактов выразить второе не может —
        быстрее, чем каждый такт, доля не бывает, и «скука ускоряет лепет» при
        базовом темпе 1.0 оказалось бы невыполнимым требованием. Поэтому 0.5 — проба
        через такт, 2.0 — две пробы за такт.

        Отдельный метод, а не значение `None` из `next_probe`: `None` там уже значит
        «идёт пауза после необратимого», и совмещение двух смыслов в одном ответе
        сразу же дало ложное «при высокой осторожности лепет встал совсем» — вызов из
        одной точки не мог отличить «темп пропустил такт» от «лепет остановился».

        Счётчик, а не случайность: прогон обязан воспроизводиться от сида, и «страх
        притормозил исследование» должно читаться в числе проб, а не в шуме. Первый
        такт не пропускается никогда — пропущенный первый такт неотличим от
        неработающего лепета.

        До этого ручка `babble_rate` не читалась ни одной строкой кода: объявленный и
        никем не спрошенный темп.
        """
        rate = max(0.0, min(PACE_MAX, self.rate if explore_rate is None
                            else float(explore_rate)))
        if self._pace is None:
            # Первый такт не пропускается никогда, и остаток подобран так, чтобы
            # дальше темп шёл ровно `rate`: иначе фиксированный старт завышал бы
            # долю проб на малых темпах (при 0.5 выходило 0.75).
            self._pace = 1.0 - rate
        self._pace += rate
        times = int(self._pace)
        self._pace -= times
        if times == 0:
            self.paced_out += 1
        return times

    def next_probe(self, *, caution_threshold: float | None = None) -> Probe | None:
        """Что пробовать дальше. `None`, если идёт пауза после необратимого."""
        if self._pause_left_ms > 0:
            self.paused_out += 1
            return None
        threshold = self.caution_threshold if caution_threshold is None else caution_threshold

        untried = [o for o in self.outputs if o not in self.body.outputs]
        if untried:
            # Цена ошибки здесь ничего не решает, и это надо сказать прямо: у всех
            # непробованных выходов она одна и та же — фон тела. Различать их по цене
            # нечем, потому что про каждый из них не известно ровно ничего. Первое
            # нажатие необратимого выхода поэтому неизбежно, и никакая осторожность
            # его не предотвратит — предотвратить можно только повторы.
            out = untried[0]
            return self._chose(UNTRIED, Probe(out, self._holds()[0], why=UNTRIED))

        # Недобранные повторы: выход, отвечавший не всегда, надо перепроверить.
        thin = [o for o in self.outputs
                if self.body.outputs[o].delivered < self.repeats]
        # Повтор, который ничего не решает, цены не стоит. Состояние выхода уже
        # выведено (`live` или `silent`), значит ещё одна проба вывода не изменит —
        # а если цена ошибки на этом выходе выше порога, платить за неё нечем.
        # Это не запрет: пока состояние `unclear`, повтор происходит при любой цене,
        # потому что тогда он решает.
        useful = [o for o in thin
                  if self.body.outputs[o].state == "unclear"
                  or not self.costly(o, threshold)]
        if len(useful) < len(thin):
            # Счётчик по решениям, а не по кандидатам: снятый повтор остаётся снятым и
            # всплывает в каждой следующей проверке ветви, поэтому подсчёт кандидатов
            # давал 9341 «снятие» на 1374 пробы — число, которое нельзя прочесть.
            self.repeats_dropped_by_cost += 1
        if not useful and thin:
            self.branch_skipped_by_cost += 1
        if useful:
            out = min(useful, key=lambda o: (self.costly(o, threshold),
                                             self.body.outputs[o].delivered, o))
            self._note_defer(useful, out, threshold)
            n = self.body.outputs[out].delivered
            return self._chose(THIN, Probe(out, self._holds()[n % 3],
                                           why=f"{THIN} {n + 1} из {self.repeats}"))

        # Живые выходы с неизвестной обратимостью — самое ценное непознанное.
        #
        # Из перебора выпадают два вида выходов, и по разным причинам:
        #
        # - те, для кого догадки об откате кончились (`_undo_exhausted`);
        # - те, кто перестал реагировать (`no_change_streak`). Ломать можно только
        #   то, что ещё не сломано: такой выход отвечал раньше, а теперь не даёт
        #   ничего, и попытки отката по нему не случаются вообще. На замере
        #   именно на этом застревал весь лепет — 1170 проб из 1266 уходили в
        #   один такой выход, потому что его счётчик поисков не мог вырасти.
        #
        # Порядок — по числу уже потраченных проб: оно растёт всегда, поэтому
        # голодания не бывает ни при какой форме мира.
        unknown = [o for o in self.outputs
                   if self.body.outputs[o].state == "live"
                   and not self.body.outputs[o].reversibility.is_known
                   and self._can_teach(o)]
        # Здесь цена меняет **только порядок**, и ветвь не пропускается никогда, даже
        # если дорого всё. Это единственная ветвь, ради которой цену вообще платят: в
        # ней и находятся обратные пары. Промежуточная редакция пропускала ветвь, когда
        # все кандидаты дороги, — и это стоило знания, а не риска: обратных пар
        # находилось 2 вместо 7, ветвь не срабатывала ни разу за 1500 шагов, а лепет
        # уходил в сочетания, которые про обратимость не говорят ничего. Осторожность,
        # запрещающая узнавать, — это не осторожность, а слепота.
        if unknown:
            out = min(unknown, key=lambda o: (self.costly(o, threshold),
                                              self.body.outputs[o].delivered,
                                              self.body.outputs[o].undo_searches, o))
            self._note_defer(unknown, out, threshold)
            return self._chose(UNKNOWN_UNDO, Probe(out, self._holds()[1],
                                                   why=UNKNOWN_UNDO))

        # Сочетания — только после того, как одиночные разобраны.
        live = sorted(o for o in self.outputs if self.body.outputs[o].state == "live")
        if len(self.body.outputs) >= self.combo_after and len(live) >= 2:
            safe = [o for o in live if not self.costly(o, threshold)]
            if len(safe) >= 2:
                a, b = self._rng.sample(safe, 2)
                return self._chose(COMBO, Probe(a, self._holds()[1], modifiers=(b,),
                                                why=COMBO))

        # Всё разобрано: продолжаем уточнять то, где разброс больше. Дорогие уходят в
        # хвост, но не исключаются — списка запретов здесь нет.
        #
        # Три ключа, и порядок между ними важен:
        #
        # 1. **Может ли проба чему-то научить.** Первым, потому что бесплатной пользы
        #    не бывает: у выхода, который перестал реагировать, попытки отката не
        #    случаются вообще, и σ не сдвинется, сколько ни жми.
        # 2. **Цена ошибки.** Ключ был `caution >= threshold`, и это ровно тот случай,
        #    ради которого цена отделена от осторожности: незнание стоит единицу,
        #    поэтому по осторожности дорогим оказывалось **всё** непробованное на
        #    откат, а контроль «порог снят» не менял ничего.
        # 3. **Ширина разброса и полнота знания** — то, ради чего ветвь и нужна.
        #
        # Первый ключ появился здесь по замеру, и это второе появление той же ошибки
        # на строку ниже: в ветви «обратимость неизвестна» перестающий реагировать
        # выход уже отсеивался, а здесь нет — и 41 проба из 41 по необратимому выходу
        # на сиде 5 ушла именно в него, потому что при неизвестной обратимости σ равна
        # единице, то есть «шире всего» навсегда.
        candidates = sorted(
            self.outputs,
            key=lambda o: (not self._can_teach(o),
                           self.costly(o, threshold),
                           self.body.outputs[o].reversibility.sigma * -1,
                           self.body.outputs[o].delivered, o))
        out = candidates[0]
        self._note_defer(self.outputs, out, threshold)
        if not self._can_teach(out):
            # Все кандидаты бесполезны — тоже число, и его надо видеть: значит лепет
            # доработал до конца и продолжает жать только потому, что его просят.
            self.probed_without_benefit += 1
        return self._chose(REFINE, Probe(out, self._rng.choice(self._holds()),
                                         why=REFINE))

    def _can_teach(self, output: str) -> bool:
        """Может ли проба этого выхода вообще что-нибудь сдвинуть.

        Два случая, когда не может, и оба выучены, а не объявлены:

        - выход **перестал реагировать** (`no_change_streak >= babble_stale_streak`).
          Ломать можно только то, что ещё не сломано: попытка отката происходит только
          после изменения кадра, поэтому по такому выходу она не случается вообще. На
          замере на этом застревал весь лепет — 1170 проб из 1266 уходили в один такой
          выход;
        - **догадки об откате кончились** при неизвестной обратимости. Ещё одна проба
          даст ещё одно последствие и ни одной новой попытки его вернуть.

        Это не запрет и не список: оба признака сняты с наблюдений и оба обратимы —
        стоит выходу снова ответить, как он вернётся в начало очереди.
        """
        f = self.body.outputs.get(output)
        if f is None:
            return True                     # непробованный учит по определению
        if f.no_change_streak >= self.stale_streak:
            return False
        return not (output in self._undo_exhausted and not f.reversibility.is_known)

    def _chose(self, branch: str, probe: Probe) -> Probe:
        self.arb.hit(branch)
        return probe

    def _note_defer(self, pool: Sequence[str], chosen: str, threshold: float) -> None:
        """Отложила ли цена чей-то выбор: в пуле было дорогое, а выбрано не оно."""
        if self.costly(chosen, threshold):
            return
        if any(o != chosen and self.costly(o, threshold) for o in pool):
            self.deferred_by_cost += 1

    def reversibility_of(self, output: str) -> Reversibility:
        f = self.body.outputs.get(output)
        return f.reversibility if f else Reversibility()

    def undo_candidate(self, output: str) -> str | None:
        """Чем попробовать откатить последствие выхода.

        Порядок догадок — от самой дешёвой к самой дорогой:

        1. Уже найденная обратная пара, если она есть.
        2. Тот же выход: для переключателя это верно, и проверить дешевле всего.
        3. Другой живой выход, ещё не пробованный как откат для этого.

        `None` значит «идей больше нет», и тогда выход остаётся с неизвестной или
        низкой обратимостью — то есть осторожным. Это честный итог: агент не
        нашёл способа откатить, а не «выход запрещён».
        """
        known = self.inverse_found.get(output)
        if known is not None:
            return known
        tried = self._undo_tried.setdefault(output, set())
        if output not in tried:
            return output

        # Догадка о симметрии: если Y однажды вернул то, что сделал `output`, то,
        # возможно, `output` вернёт то, что делает Y. Это именно догадка и она
        # проверяется пробой, а не принимается за факт, — но проверять её стоит
        # раньше случайного перебора, потому что она чаще верна.
        for other, inverse in self.inverse_found.items():
            if inverse == output and other not in tried:
                return other

        live = [o for o in self.outputs
                if o != output and self.body.outputs.get(o) is not None
                and self.body.outputs[o].state == "live" and o not in tried]
        # Сначала те, чью обратимость мы уже видели: они предсказуемее, и если
        # такой выход вернёт мир, пара найдена надёжнее.
        if not live:
            self._undo_exhausted.add(output)
            return None
        live.sort(key=lambda o: (not self.body.outputs[o].reversibility.is_known, o))
        return live[0]

    def note_undo_attempt(self, output: str, undo_output: str, worked: bool) -> None:
        """Запомнить, что пробовали и чем кончилось."""
        self._undo_tried.setdefault(output, set()).add(undo_output)
        if worked:
            self.inverse_found[output] = undo_output
            # Обратимость симметрична: если Y вернул последствие X, то X, скорее
            # всего, вернёт последствие Y. Это догадка, и она будет проверена
            # отдельной пробой, а не принята за факт.
            self._undo_tried.setdefault(undo_output, set())

    # --- результат ----------------------------------------------------------

    def absorb(self, result: ProbeResult, stamp: Stamp | None = None) -> None:
        """Записать результат пробы: в карту тела и, если есть, в журнал."""
        probe = result.probe
        self.probes_done += 1
        for out in (*probe.modifiers, probe.output):
            f = self.body.fact(out, stamp.t_self if stamp else 0)
            f.tries += 1
            if not result.delivered:
                continue
            f.delivered += 1
            f.durations_ms.append(probe.duration_ms)
            if result.changed:
                f.responded += 1
                f.no_change_streak = 0
            else:
                f.no_change_streak += 1
            if result.undone is None:
                continue
            if result.undone:
                # Способ найден: это и есть свидетельство обратимости.
                f.reversibility = f.reversibility.observe(True)
                f.undo_method = result.undo_output
            elif result.undo_was_known:
                # Известный способ перестал работать — вот это уже про мир.
                f.reversibility = f.reversibility.observe(False)
            else:
                # Догадка не сработала. Это про догадку, а не про мир: оценка
                # обратимости остаётся неизвестной, то есть осторожность —
                # максимальной. Смешивать одно с другим значило бы считать
                # необратимым всё, к чему не сразу подобрался ключ.
                f.undo_searches += 1

        if result.undone is False and result.undo_was_known:
            self.irreversible_hits += 1
            self._pause_left_ms = self.pause_ms
        elif result.undone is False:
            # Не нашли способ — тоже повод притормозить, но короче: последствие
            # уже случилось и его надо разглядеть.
            self._pause_left_ms = max(self._pause_left_ms, self.pause_ms // 3)

        if self.journal is not None and stamp is not None:
            action = probe.to_action(self.reversibility_of(probe.output))
            # Лепет — не рефлекс, не навык и не план: ни один из них ещё не
            # существует. Он происходит потому, что любопытство вне коридора и
            # больше ответить нечем, поэтому инициатор — контур драйвов. Если
            # когда-нибудь появится отдельный контур лепета, изменится набор
            # слоёв (структурное изменение), а не эта атрибуция.
            # Срез состояния, а не только событие. Ошибка предсказания лежала
            # **только** в событии (`ProbeResult.as_event`), и из-за этого
            # `vitals.prediction_error_mean` докладывал «ни одна запись не несёт
            # ошибки предсказания» по журналу, каждая запись которого её несла:
            # читатели построены на `StateSnapshot`, а число было положено рядом.
            # Величина в записи есть, а измерить её нельзя — худший вид расхождения,
            # потому что данные выглядят целыми.
            #
            # Драйвы, настроение и цель приходят от вызывающего (`state_extra`): лепет
            # про них не знает и знать не должен — иначе `behaviour/babbling` начнёт
            # импортировать мотивацию, и слой перестанет быть проверяемым отдельно.
            extra = dict(self._state_extra() if self._state_extra else {})
            state = StateSnapshot(prediction_error=result.error_value,
                                 drives=extra.get("drives", {}),
                                 mood=extra.get("mood"),
                                 goal_id=extra.get("goal_id"))
            self.journal.append(EntryKind.ACTION, stamp, Actor.AGENT,
                                ActorLayer.DRIVE, action=action, state=state,
                                event={**result.as_event(), "device": "babble",
                                       "latency_ms": float(probe.duration_ms)})

    def tick_pause(self, dt_ms: float) -> None:
        self._pause_left_ms = max(0, self._pause_left_ms - int(dt_ms))

    @property
    def paused(self) -> bool:
        return self._pause_left_ms > 0

    # --- сводка -------------------------------------------------------------

    def progress(self) -> dict[str, Any]:
        """«118 из 512 выходов проверено» — та же сводка, что в пульте."""
        st = self.body.stats()
        return {
            "outputs_total": len(self.outputs),
            "outputs_touched": st["known_outputs"],
            "live": st["live"], "silent": st["silent"],
            "unclear": st["unclear"],
            "untried": len(self.outputs) - st["known_outputs"],
            "background_rate": round(self.body.background_rate, 4),
            "probes_done": self.probes_done,
            "irreversible_hits": self.irreversible_hits,
            "unknown_reversibility": st["unknown_reversibility"],
            "not_undoable": self.body.dangerous(self.caution_threshold),
            "inverse_pairs": dict(sorted(self.inverse_found.items())),
            "undo_search_exhausted": sorted(self._undo_exhausted),
            "paused_ms": self._pause_left_ms,
            # Инвариант 26: срабатывания по ветвям. Ноль у ветви за полный прогон —
            # дефект, а не «не понадобилось».
            "branch_hits": {b.name: b.hits for b in self.arb},
            "branch_defects": self.arb.as_dict()["dead"],
            "paused_out": self.paused_out,
            "paced_out": self.paced_out,
            # Инвариант 25: сдвиг числа от того, что цена ошибки читается.
            "deferred_by_cost": self.deferred_by_cost,
            "repeats_dropped_by_cost": self.repeats_dropped_by_cost,
            "probed_without_benefit": self.probed_without_benefit,
            "branch_skipped_by_cost": self.branch_skipped_by_cost,
            "base_cost": round(self.base_cost(), 4),
            "error_cost": {o: round(self.error_cost(o), 4)
                           for o in sorted(self.body.outputs)},
        }


def run_babbling(world: Any, babbler: Babbler, *, steps: int,
                 clocks: Any, journal: Journal | None = None,
                 undo_with: Callable[[str], str | None] | None = None,
                 error: Any = None,
                 explore_rate: float | None = None,
                 caution_threshold: float | None = None) -> dict[str, Any]:
    """Прогон лепета по миру. Проверяемо офлайн, без игры и без устройства.

    `undo_with` — чем агент пытается откатить последствие. По умолчанию он
    пробует тот же выход ещё раз: это самая простая догадка «нажму опять, вдруг
    вернётся», и для переключателя она верна, а для необратимого — нет. Именно
    так обратимость и выясняется: попыткой, а не таблицей.

    ## Последствие определяется по пикселям, а не по состоянию мира

    Раньше здесь читалось `Observation.changed` — то есть изменилось ли внутреннее
    состояние мира. Это истина, которой у агента нет: он видит только пиксели.
    В игре ошибка не проявлялась, потому что там изменение состояния всегда видно
    в кадре, — ровно такие ошибки и живут годами.

    И сравнивать надо не с нулём, а с тем, как мир меняется **сам**. В видео
    картинка идёт без всяких действий; в браузере крутится анимация. Сравнение с
    нулём объявило бы живыми все выходы подряд — что на замере и произошло: в
    домене «видео» из двенадцати молчащих выходов не нашлось ни одного.

    Поэтому фон измеряется на шагах без действия, и последствием считается
    превышение над фоном на `babble_response_sigmas` сигм.
    """
    from ..core.action import Action as _Action
    from ..vision.selfworld import frame_energy

    sigmas = float(babbler.profile.parameters["babble_response_sigmas"])
    floor = float(babbler.profile.parameters["babble_response_floor"])
    idle: list[float] = []
    # Порог врёт и сам по себе: анимация интерфейса или идущее видео иногда дают
    # превышение над фоном без всякого действия. Насколько часто — измеримо тем же
    # тестом по паре кадров без действия, и без этой величины нельзя отличить
    # «ответил один раз» от «порог сработал один раз».
    idle_tests = 0
    false_alarms = 0

    def background() -> tuple[float, float]:
        if len(idle) < 3:
            return 0.0, 0.0
        arr = idle[-64:]
        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / len(arr)
        return mean, var ** 0.5

    def differs(change: float) -> bool:
        """Отличается ли изменение кадра от того, как мир меняется сам.

        Проверка двусторонняя, и это не формальность. В видео нажатие «пауза»
        *уменьшает* изменение кадра до нуля — последствие налицо, а односторонняя
        проверка «стало больше фона» его не увидит. На замере из-за этого в домене
        «видео» не находилось ни одного живого выхода из четырёх.
        """
        mean, sigma = background()
        # Пол нужен: при побитово неподвижном фоне сигма равна нулю, и любое
        # дрожание в один уровень яркости стало бы «последствием».
        return abs(change - mean) > max(floor, sigmas * max(sigma, 1e-4))

    def slots() -> Iterator[bool]:
        """Такты и пробы внутри такта. Темп может дать больше одной пробы за такт.

        Отдельный генератор, чтобы тело пробы осталось нетронутым: переписывать
        семьдесят строк ради двух осей модуляции — это заводить новые ошибки в
        механизме, который уже замерен.
        """
        for _ in range(steps):
            babbler.tick_pause(1000.0
                               / float(babbler.profile.parameters["capture_fps"]))
            times = babbler.pace(explore_rate)
            if times == 0:
                yield False
                continue
            for _ in range(times):
                yield True

    for probing in slots():
        # Две оси модуляции читаются здесь: темп исследования решает, сколько проб
        # на такте, порог необратимости — что именно выбрать. Без этих двух строк
        # обе оси считались бы реализованными и не влияли бы ни на что.
        probe = (babbler.next_probe(caution_threshold=caution_threshold)
                 if probing else None)
        if probe is None:
            world.step(None, with_audio=False)      # мир идёт и во время паузы
            clocks.tick_self()
            clocks.set_world(clocks.t_world + 1)
            continue

        idle_a = world.step(None, with_audio=False).frame
        clocks.tick_self()
        clocks.set_world(clocks.t_world + 1)
        before = world.step(None, with_audio=False).frame
        clocks.tick_self()
        clocks.set_world(clocks.t_world + 1)
        # Два шага без действия подряд дают замер фона: столько мир меняется сам.
        # Этот же замер проверяется тем же порогом — до того, как попадёт в фон,
        # иначе проба проверяла бы саму себя.
        sample_idle = frame_energy(idle_a, before)
        if len(idle) >= 3:
            idle_tests += 1
            if differs(sample_idle):
                false_alarms += 1
            babbler.body.set_background_rate(false_alarms / idle_tests)
        idle.append(sample_idle)

        action = probe.to_action(babbler.reversibility_of(probe.output))
        after = world.step(action, with_audio=False)
        clocks.tick_self()
        clocks.set_world(clocks.t_world + 1)
        changed = differs(frame_energy(before, after.frame))

        err = None
        if error is not None:
            sample = error.feed(after.frame)
            err = sample.value if sample else None

        undone: bool | None = None
        undo_output = None
        undo_known = False
        # Обратимость проверяется только у одиночной пробы. У сочетания мир
        # изменили два выхода, и откат одного из них мир не вернёт — а вина за
        # это досталась бы одному, хотя виноваты оба. На замере именно так
        # обратимость движения падала с 1.0 до 0.33 при верно найденной паре.
        # Обратимость сочетания — отдельный вопрос, и он требует отдельной пробы.
        if changed and not probe.modifiers:
            undo_known = probe.output in babbler.inverse_found
            undo_output = (babbler.undo_candidate(probe.output) if undo_with is None
                           else undo_with(probe.output))
            if undo_output is not None:
                world.step(_Action.key(undo_output, probe.duration_ms), with_audio=False)
                clocks.tick_self()
                clocks.set_world(clocks.t_world + 1)
                restored = world.step(None, with_audio=False).frame
                clocks.tick_self()
                clocks.set_world(clocks.t_world + 1)
                # Откат удался, если расхождение с тем, что было до пробы, не
                # больше того, как мир меняется сам. Побитового совпадения не
                # будет никогда: интерфейс анимирован, а в видео картинка идёт.
                undone = not differs(frame_energy(before, restored))
                babbler.note_undo_attempt(probe.output, undo_output, undone)

        babbler.absorb(ProbeResult(probe, True, changed, undone, err,
                                   undo_output=undo_output,
                                   undo_was_known=undo_known),
                       clocks.stamp())

    return babbler.progress()
