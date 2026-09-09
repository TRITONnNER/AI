"""Ограничитель ресурсов.

Предел, который никто не проверяет, — это не предел, а комментарий. Поэтому
здесь не только числа из профиля, но и то, что при упоре в них происходит.

Что ограничивается и что происходит при упоре:

| Ресурс | Мягкий порог | Упор |
|---|---|---|
| Память процесса | вытеснение рабочего контекста | отказ принимать кадры |
| Диск сессии | — | отказ принимать кадры |
| Число карточек | заявка на консолидацию | заявка на консолидацию |
| Расход на модель | — | отказ обращаться к модели |

Одного правила здесь нет ни при каких обстоятельствах: **журнал не обрезается**.
Инвариант 1 не знает исключений «кончилось место». Если места нет, запись
останавливается и это пишется в журнал последней записью — но уже записанное
остаётся целым. Обратный порядок (выкинуть старое, чтобы записать новое) сделал
бы журнал невоспроизводимым, а из него выводится всё остальное.

Измеритель подменяемый. Иначе тест на упор в четыре гигабайта потребовал бы
занять четыре гигабайта, то есть его никто не будет запускать.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from .clocks import Stamp
from .journal import Actor, ActorLayer, Journal, Kind as EntryKind
from .profile import Profile

# Что именно упёрлось. Набор закрыт: новый вид предела — это новая настройка в
# схеме и новая ветка реакции, а не свободная строка.
BREACH_RAM_WARN = "ram_warn"
BREACH_RAM_CAP = "ram_cap"
BREACH_DISK_CAP = "disk_cap"
BREACH_BELIEF_CAP = "belief_cap"
BREACH_SPEND_CAP = "spend_cap"
BREACH_MODEL_RATE = "model_rate"

# Операции, которые спрашивают разрешения.
OP_FRAME = "frame"          # записать кадр
OP_MODEL = "model_call"     # обратиться к большой модели
OP_BELIEF = "belief"        # завести карточку
OP_TRACE = "trace"          # дописать причинную запись — нулевой уровень
OP_SEGMENT = "segment"      # положить сегмент уровней 1–3

# Какой уровень журнала затрагивает операция. Нужно затем, чтобы отказ по месту
# был отказом **по уровням 1–3**, а нулевой уровень не отказывался никогда: он в
# отдельном резерве, и при его исчерпании система останавливает запись и сообщает,
# а не чистит (`STORAGE.md`, раздел 2).
OP_LEVEL = {OP_FRAME: 2, OP_SEGMENT: 1, OP_TRACE: 0}


class ResourceError(RuntimeError):
    pass


@runtime_checkable
class Measurer(Protocol):
    """Откуда берутся настоящие числа."""

    def rss_mb(self) -> float | None: ...
    def dir_mb(self, path: Path) -> float: ...


class ProcMeasurer:
    """Измерение на Linux: `/proc/self/statm`, иначе `getrusage`.

    `getrusage` даёт **пиковое** потребление, а не текущее, и по нему нельзя
    заметить, что вытеснение помогло. Поэтому он помечен как приближение, а не
    подсунут как точное значение: `is_peak_only` виден вызывающему.
    """

    def __init__(self) -> None:
        self.is_peak_only = not Path("/proc/self/statm").exists()

    def rss_mb(self) -> float | None:
        try:
            fields = Path("/proc/self/statm").read_text().split()
            pages = int(fields[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        except (OSError, IndexError, ValueError):
            pass
        try:
            import resource
            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return kb / 1024.0          # Linux отдаёт КиБ
        except Exception:
            return None                  # честное «не знаю», а не ноль

    def dir_mb(self, path: Path) -> float:
        total = 0
        for p in Path(path).rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
        return total / (1024 * 1024)


@dataclass(frozen=True, slots=True)
class Breach:
    code: str
    measured: float
    limit: float
    unit: str
    action: str          # что сделано в ответ

    def as_dict(self) -> dict[str, object]:
        return {"code": self.code, "measured": round(self.measured, 3),
                "limit": self.limit, "unit": self.unit, "action": self.action}


@dataclass(frozen=True, slots=True)
class ResourceState:
    rss_mb: float | None
    disk_mb: float
    beliefs: int
    usd_spent: float
    rss_known: bool

    def as_dict(self) -> dict[str, object]:
        return {"rss_mb": None if self.rss_mb is None else round(self.rss_mb, 1),
                "disk_mb": round(self.disk_mb, 1), "beliefs": self.beliefs,
                "usd_spent": round(self.usd_spent, 4), "rss_known": self.rss_known}


@dataclass(frozen=True, slots=True)
class Admission:
    """Ответ на «можно ли». Отказ всегда с причиной — молча не отказываем."""

    allowed: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.allowed


class ResourceGovernor:
    """Считает, сравнивает с профилем и делает то, что решено при упоре."""

    def __init__(self, profile: Profile, *, journal: Journal | None = None,
                 session_root: Path | str | None = None,
                 measurer: Measurer | None = None,
                 on_evict: Callable[[int], int] | None = None,
                 on_consolidate: Callable[[], None] | None = None,
                 now: Callable[[], float] | None = None) -> None:
        import time as _time

        self._now = now or _time.monotonic
        p = profile.parameters
        self.ram_cap_mb = float(p["ram_cap_mb"])
        self.ram_warn = float(p["ram_warn_fraction"])
        self.disk_cap_mb = float(p["session_disk_cap_mb"])
        self.belief_cap = int(p["belief_cap"])
        # Шестая ось модуляции: какая доля предела карточек отдана текущей задаче.
        # Единица — весь предел, то есть прежнее поведение; настроение сужает долю,
        # когда прогноз рушится. Это не «экономия памяти», а ставка: вкладывать весь
        # предел в одну задачу дороже, когда дела идут хуже ожидаемого.
        self.task_share = 1.0
        self.spend_cap_usd = float(p["spend_cap_usd"])
        # Предел частоты обращений к большой модели. Это не деньги, а темп: у
        # бесплатных тарифов ограничение именно на число запросов, и упереться в
        # него посреди прогона — рабочая ситуация, а не поломка.
        self.model_rate_cap = float(p["token_budget_per_min"])
        self._model_calls_at: list[float] = []
        self.working_frames = int(p["working_context_frames"])
        self.journal = journal
        self.session_root = Path(session_root) if session_root else None
        self.measurer = measurer or ProcMeasurer()
        self._on_evict = on_evict
        self._on_consolidate = on_consolidate

        self.usd_spent = 0.0
        self.model_calls = 0
        self.beliefs = 0
        self._refuse_frames: str | None = None
        self._refuse_model: str | None = None
        self._seen: set[str] = set()          # чтобы не спамить журнал одним и тем же

    # --- измерение ----------------------------------------------------------

    def state(self) -> ResourceState:
        rss = self.measurer.rss_mb()
        disk = self.measurer.dir_mb(self.session_root) if self.session_root else 0.0
        return ResourceState(rss, disk, self.beliefs, self.usd_spent, rss is not None)

    # --- проверка и реакция -------------------------------------------------

    def check(self, stamp: Stamp) -> list[Breach]:
        """Посмотреть на все пределы и сделать то, что решено. Возвращает упоры."""
        st = self.state()
        out: list[Breach] = []

        if st.rss_mb is not None:
            if st.rss_mb >= self.ram_cap_mb:
                self._refuse_frames = (
                    f"память процесса {st.rss_mb:.0f} МиБ при пределе "
                    f"{self.ram_cap_mb:.0f} МиБ")
                out.append(Breach(BREACH_RAM_CAP, st.rss_mb, self.ram_cap_mb, "МиБ",
                                  "запись кадров остановлена, журнал не обрезан"))
            elif st.rss_mb >= self.ram_cap_mb * self.ram_warn:
                freed = self._evict()
                out.append(Breach(BREACH_RAM_WARN, st.rss_mb,
                                  self.ram_cap_mb * self.ram_warn, "МиБ",
                                  f"вытеснено кадров рабочего контекста: {freed}"))
            else:
                self._refuse_frames = None

        if self.disk_cap_mb > 0 and st.disk_mb >= self.disk_cap_mb:
            self._refuse_frames = (
                f"сессия занимает {st.disk_mb:.0f} МиБ при пределе "
                f"{self.disk_cap_mb:.0f} МиБ")
            out.append(Breach(BREACH_DISK_CAP, st.disk_mb, self.disk_cap_mb, "МиБ",
                              "запись кадров остановлена, журнал не обрезан"))

        if self.beliefs >= self.belief_cap:
            if self._on_consolidate is not None:
                self._on_consolidate()
            out.append(Breach(BREACH_BELIEF_CAP, float(self.beliefs),
                              float(self.belief_cap), "карточек",
                              "заявлена консолидация: забывание по ценности"))

        rate = self.model_call_rate()
        if self.model_rate_cap > 0 and rate > self.model_rate_cap:
            self._refuse_model = (
                f"частота обращений {rate:.2f} запр/с при пределе "
                f"{self.model_rate_cap:.2f}")
            out.append(Breach(BREACH_MODEL_RATE, rate, self.model_rate_cap, "запр/с",
                              "обращения к модели придётся отложить; мир идёт дальше"))

        if self.spend_cap_usd > 0 and self.usd_spent >= self.spend_cap_usd:
            self._refuse_model = (
                f"расход ${self.usd_spent:.2f} при пределе ${self.spend_cap_usd:.2f}")
            out.append(Breach(BREACH_SPEND_CAP, self.usd_spent, self.spend_cap_usd, "$",
                              "обращения к модели остановлены, мир идёт дальше"))

        for b in out:
            self._journal_once(b, stamp)
        return out

    def task_cap(self) -> int:
        """Предел карточек для текущей задачи после модуляции долей ресурсов.

        Не ниже одной карточки: доля, сжатая до нуля, означала бы «задача не имеет
        права ничего узнать», а это не сужение ставки, а остановка.
        """
        return max(1, int(self.belief_cap * max(0.0, min(1.0, self.task_share))))

    def set_task_share(self, share: float) -> None:
        """Принять ось модуляции. Отдельный метод: значение приходит от настроения."""
        self.task_share = max(0.05, min(1.0, float(share)))

    def _evict(self) -> int:
        """Вытеснить рабочий контекст. Он эфемерный и выводим из журнала."""
        if self._on_evict is None:
            return 0
        return int(self._on_evict(max(1, self.working_frames // 2)))

    def _journal_once(self, breach: Breach, stamp: Stamp) -> None:
        if self.journal is None or breach.code in self._seen:
            return
        self._seen.add(breach.code)
        self.journal.append(EntryKind.RESOURCE, stamp, Actor.NONE,
                            ActorLayer.INTERRUPT, event={"code": breach.code, **breach.as_dict()})

    def forget_breach(self, code: str) -> None:
        """Разрешить снова сообщить об этом упоре — после того, как отпустило."""
        self._seen.discard(code)

    # --- разрешения ---------------------------------------------------------

    def admit(self, op: str, *, cost_usd: float = 0.0) -> Admission:
        """Можно ли выполнить операцию прямо сейчас.

        `cost_usd` — во что обойдётся **этот** вызов модели, если он платный. С ним
        предел принуждается **до** траты (см. `would_breach_model`); без него
        остаётся только запоздалый отказ, и это объявленное ограничение, а не
        умолчание: тот, кто не назвал цену, платит первым превышением.

        Нулевой уровень пропускается **всегда**, и это не поблажка, а инвариант 14:
        нехватка места никогда не запускает удаление следа и никогда не мешает его
        дописать. Отказ по месту относится к уровням 1–3, то есть к сенсорной
        роскоши. Резерв нулевого уровня выделен отдельно и в потолок не входит;
        когда кончится он, система останавливает запись и сообщает — но не чистит,
        и делается это не здесь.
        """
        if OP_LEVEL.get(op) == 0:
            return Admission(True)
        if op == OP_SEGMENT and self._refuse_frames:
            return Admission(False, self._refuse_frames)
        if op == OP_FRAME and self._refuse_frames:
            return Admission(False, self._refuse_frames)
        if op == OP_MODEL:
            if self._refuse_model:
                return Admission(False, self._refuse_model)
            ahead = self.would_breach_model(cost_usd)
            if ahead is not None:
                return Admission(False, ahead)
        if op == OP_BELIEF:
            allowed = self.task_cap()
            if self.beliefs >= allowed:
                return Admission(
                    False, f"карточек {self.beliefs} при пределе {allowed}"
                    + (f" (доля задачи {self.task_share:.2f} от {self.belief_cap})"
                       if allowed != self.belief_cap else ""))
        return Admission(True)

    def require(self, op: str, *, cost_usd: float = 0.0) -> None:
        a = self.admit(op, cost_usd=cost_usd)
        if not a:
            raise ResourceError(f"{op} не разрешена: {a.reason}")

    def would_breach_model(self, cost_usd: float = 0.0) -> str | None:
        """Что нарушится, если сделать **ещё один** вызов модели. `None` — ничего.

        Здесь предел принуждается до превышения, а не после, и разница не
        косметическая. `check` отмечает упор, когда `usd_spent` уже перевалил за
        предел: деньги к этому моменту потрачены, и «предел» описывает прошлое.
        Отказ обязан случиться **перед** вызовом, иначе объявленный потолок расхода
        не потолок, а отметка о том, что его пробили.

        Частота считается так же: вопрос не «превышена ли она сейчас», а «превысит
        ли её этот вызов». Первая формулировка пропускает ровно один лишний вызов
        каждый раз, когда упирается, и на длинной серии это систематическая
        недостача, а не округление.
        """
        cost = max(0.0, float(cost_usd))
        if self.spend_cap_usd > 0:
            after = self.usd_spent + cost
            if after > self.spend_cap_usd:
                return (f"вызов довёл бы расход до ${after:.4f} при пределе "
                        f"${self.spend_cap_usd:.2f}")
        if self.model_rate_cap > 0:
            self.model_call_rate()          # заодно чистит окно от старых вызовов
            after_rate = (len(self._model_calls_at) + 1) / 60.0
            if after_rate > self.model_rate_cap:
                return (f"вызов довёл бы частоту до {after_rate:.2f} запр/с при "
                        f"пределе {self.model_rate_cap:.2f}")
        return None

    # --- расход -------------------------------------------------------------

    def spend(self, usd: float = 0.0, *, model_calls: int = 0,
              at: float | None = None) -> None:
        if usd < 0 or model_calls < 0:
            raise ResourceError("расход не бывает отрицательным")
        self.usd_spent += float(usd)
        self.model_calls += int(model_calls)
        now = self._now() if at is None else float(at)
        self._model_calls_at.extend([now] * int(model_calls))

    def model_call_rate(self, at: float | None = None) -> float:
        """Обращений к модели в секунду за последнюю минуту.

        Окно минутное, а мера — в секунду, потому что предел объявлен в запр/с:
        средняя за минуту не наказывает за короткую серию и при этом не даёт
        держать высокий темп долго.
        """
        now = self._now() if at is None else float(at)
        self._model_calls_at = [t for t in self._model_calls_at if now - t <= 60.0]
        return len(self._model_calls_at) / 60.0

    def note_beliefs(self, count: int) -> None:
        self.beliefs = max(0, int(count))

    def report(self) -> dict[str, object]:
        st = self.state()
        return {
            **st.as_dict(),
            "ram_cap_mb": self.ram_cap_mb,
            "disk_cap_mb": self.disk_cap_mb,
            "belief_cap": self.belief_cap,
            "spend_cap_usd": self.spend_cap_usd,
            "model_rate_cap": self.model_rate_cap,
            "model_call_rate": round(self.model_call_rate(), 4),
            "model_calls": self.model_calls,
            "frames_refused": self._refuse_frames,
            "model_refused": self._refuse_model,
        }
