"""`harness doctor`: что на этой машине не так и что именно ввести, чтобы стало так.

`TASK-07`, часть 2. Задача команды — не «проверить окружение», а довести человека,
впервые получившего репозиторий, до первой записи **без чтения кода и без
угадывания**. Поэтому здесь два правила, и оба нарушаются легко:

1. **Каждый недостающий пункт сопровождается строкой, которую можно скопировать.**
   Не «установите ffmpeg», а `sudo apt install -y ffmpeg`. Разница не в вежливости:
   «установите ffmpeg» отправляет оператора искать, как это делается в его системе,
   и это ровно то думание, которого задача не допускает.

2. **Блокирующее отделено от «понадобится позже».** Отсутствие `/dev/uinput` не
   мешает записывать: инъекция ввода нужна к М5. Если смешать, оператор будет чинить
   права на устройство ввода перед записью, которой они не нужны, и бросит.

## Чёрный кадр — это отказ

Главная ловушка на macOS: без разрешения «Запись экрана» захват возвращает кадр
нужного размера, заполненный нулями. Ни одна проверка «вернулся ли кадр» этого не
поймает — вернулся, правильной формы, нужного типа. Поэтому проверка **активная**:
берутся два кадра и считается разброс яркости. Нулевой разброс — отказ, а не успех.

То же самое, но по другой причине, случается под Wayland через XWayland. Там отказ
приходит раньше — на выборе механизма (`harness.machine`), — но проверка содержимого
всё равно нужна: она единственная не зависит от того, угадали ли мы причину.

## Чего эта команда не делает

Не ставит ничего сама. Установка пакетов и выдача разрешений — действия с
последствиями на машине человека, и решать их должен человек. Команда печатает, что
ввести; вводит оператор.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from .machine import (Machine, Session, detect, enable_uinput, find_loopback,
                      grant_screen_recording, install_capture, install_ffmpeg,
                      pip_install, switch_to_x11)

#: Минимальная версия Python. Стоит здесь, а не в голове: `StrEnum` появился в 3.11,
#: и на 3.10 проект падает импортом, а не понятным сообщением.
PYTHON_MIN = (3, 11)

#: Сколько байт занимает кадр 1080p в журнале. **Измерено**, а не оценено:
#: `tools/measure_storage_rate.py`, числа в `docs/measurements/storage_rate.json`.
#:
#: Прежнее значение — 90 КиБ — я взял из головы в TASK-07 и подписал как замер. На
#: машине оператора это дало «64.5 ГиБ ≈ 7.0 часов» там, где честный ответ около 170:
#: настоящий конвейер (кадр, дельта, zlib) пишет 3.6 КиБ на кадр экранного содержимого
#: и 2.3 КиБ на неподвижном экране.
#:
#: Берётся вариант «экран с движением»: он ближе всего к тому, что будет записывать
#: оператор по минимальному плану. Ошибка в эту сторону не бесплатна, поэтому вместе с
#: числом печатается предел для несжимаемого содержимого — видео во весь экран и игра
#: с частицами ближе к шуму, чем к рабочему столу.
BYTES_PER_FRAME_1080P = int(3.6 * 1024)

#: Тот же кадр, если содержимое несжимаемо. Не «на всякий случай»: разница между
#: 0.37 и 209 ГиБ/ч — это разница между «влезет неделя» и «влезет двадцать минут», и
#: скрывать её за одним средним числом нельзя.
WORST_BYTES_PER_FRAME_1080P = int(2026 * 1024)


class State(StrEnum):
    """Три исхода проверки. Третий обязателен: «не проверено» ≠ «нет»."""

    YES = "есть"
    NO = "нет"
    UNKNOWN = "не проверено"


@dataclass(frozen=True, slots=True)
class Check:
    """Одна проверка: что искали, что нашли, мешает ли это записи, что делать."""

    name: str
    state: State
    detail: str
    blocks: bool = False        # блокирует запись прямо сейчас
    fix: str = ""               # копируемая строка; пусто, если чинить нечего
    later: str = ""             # к какой вехе понадобится, если не блокирует

    def __post_init__(self) -> None:
        if self.state is not State.YES and not self.fix:
            raise ValueError(
                f"проверка {self.name!r} нашла нехватку и не сказала, что делать. "
                "Пункт без команды — это приглашение угадывать, а угадывание здесь "
                "и есть дефект (TASK-07, часть 2)")
        if self.blocks and self.later:
            raise ValueError(
                f"проверка {self.name!r} и блокирует, и «понадобится позже» — "
                "это разные вещи, и смешение их приводит к тому, что оператор "
                "чинит перед записью то, что записи не нужно")

    @property
    def mark(self) -> str:
        return {State.YES: "  есть  ", State.NO: "  НЕТ   ",
                State.UNKNOWN: " не пров"}[self.state]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "state": str(self.state), "detail": self.detail,
                "blocks": self.blocks, "fix": self.fix, "later": self.later}


@dataclass(slots=True)
class Report:
    machine: Machine
    checks: list[Check] = field(default_factory=list)

    @property
    def blocking(self) -> list[Check]:
        return [c for c in self.checks if c.blocks and c.state is not State.YES]

    @property
    def later(self) -> list[Check]:
        return [c for c in self.checks if c.later and c.state is not State.YES]

    @property
    def can_record(self) -> bool:
        return not self.blocking

    def verdict(self) -> str:
        """Одна строка в конце. Именно одна: её читают, остальное просматривают."""
        if self.can_record:
            return ("ЗАПИСЫВАТЬ МОЖНО. Дальше: harness selftest, "
                    "затем harness record --plan")
        first = self.blocking[0]
        return (f"ЗАПИСЫВАТЬ НЕЛЬЗЯ. Чинить первым: {first.name} — {first.fix}")

    def as_dict(self) -> dict[str, Any]:
        return {"machine": self.machine.as_dict(),
                "checks": [c.as_dict() for c in self.checks],
                "can_record": self.can_record,
                "verdict": self.verdict()}

    def render_text(self) -> str:
        m = self.machine
        out = [f"Машина: {m.os_name} {m.release}, оконная система {m.session} "
               f"({m.session_source})",
               f"Python: {'.'.join(str(x) for x in m.python)}",
               ""]
        out.append("Нужно для записи:")
        out += _block([c for c in self.checks if c.blocks or not c.later])
        later = [c for c in self.checks if c.later]
        if later:
            out += ["", "Понадобится позже (записи не мешает):"]
            out += _block(later)
        out += ["", self.verdict()]
        return "\n".join(out)


def _block(checks: list[Check]) -> list[str]:
    rows: list[str] = []
    for c in checks:
        rows.append(f"  [{c.mark}] {c.name:<28} {c.detail}")
        if c.state is not State.YES and c.fix:
            head = "чинить" if c.blocks else f"понадобится к {c.later}" if c.later \
                else "чинить"
            rows.append(f"             {head}: {c.fix}")
    return rows


# ---------------------------------------------------------------------------
# Сами проверки
# ---------------------------------------------------------------------------


def check_python(m: Machine) -> Check:
    ok = m.python >= PYTHON_MIN
    want = ".".join(str(x) for x in PYTHON_MIN)
    return Check(
        "версия Python", State.YES if ok else State.NO,
        f"{'.'.join(str(x) for x in m.python)} (нужно {want}+)",
        blocks=not ok,
        fix="" if ok else (
            f"поставьте Python {want} или новее: "
            + ("winget install --id Python.Python.3.12 -e" if m.system == "Windows"
               else "brew install python@3.12" if m.system == "Darwin"
               else "sudo apt install -y python3.12 python3.12-venv")))


def check_deps(m: Machine) -> Check:
    """Обязательные пакеты. Необязательные — отдельными проверками, не здесь."""
    import importlib.util

    from .capture.select import CANDIDATES

    need = {"numpy": "numpy>=1.26"}
    # Пакет захвата спрашивается у кандидатов этой оконной системы, а не у одного
    # зашитого имени: на Windows их два, и раньше проверялся только запасной.
    for b in CANDIDATES[m.session]:
        need[b.module] = b.module
    missing = [spec for mod, spec in need.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return Check("зависимости", State.YES,
                     "все на месте: " + ", ".join(need))
    return Check("зависимости", State.NO, "нет: " + ", ".join(missing),
                 blocks=True, fix=pip_install(" ".join(missing), m))


def check_session(m: Machine) -> Check:
    """Оконная система и есть ли под неё механизм захвата.

    Печатает **выбранный механизм, альтернативы и причину выбора**. Строка «есть» без
    объяснения, почему не взят установленный и более подходящий механизм, — дефект
    вывода: именно так на Windows брался mss при работающем dxcam, и отчёт при этом
    был зелёным.
    """
    from .capture.select import plan

    if m.session is Session.NONE:
        return Check("графическая сессия", State.NO,
                     f"нет ({m.session_source})", blocks=True,
                     fix="запустите на машине с экраном; в контейнере и по ssh "
                         "без проброса X захватывать нечего")
    choice = plan(m)
    if choice.chosen is None:
        return Check("механизм захвата", State.NO, choice.why_text(), blocks=True,
                     fix=switch_to_x11() if m.session is Session.WAYLAND
                     else install_capture(m))
    return Check("механизм захвата", State.YES, choice.why_text())


def check_fullscreen(m: Machine) -> Check | None:
    """Умеет ли выбранный механизм захватывать полноэкранные приложения.

    Отдельным пунктом, а не примечанием: через mss на Windows игра **не попадёт в
    кадр**, и молча — кадры будут, содержимое будет не то. Записи из минимального
    набора это не мешает (браузер и рабочий стол видны), поэтому пункт не
    блокирующий; полному набору с игрой — мешает, и там он назван.

    `None` — механизма нет вовсе, и говорить об играх нечего: об этом уже сказал
    предыдущий пункт, а два «НЕТ» об одной причине заставляют чинить дважды.
    """
    from .capture.select import plan

    choice = plan(m)
    if choice.chosen is None:
        return None
    if not choice.caveat:
        return Check("полноэкранные приложения", State.YES,
                     f"{choice.chosen.name} их видит")
    return Check("полноэкранные приложения", State.NO,
                 f"активен {choice.chosen.name}: {choice.caveat}",
                 later="записям игры (полный набор `--set full`)",
                 fix=install_capture(m) + "   # затем повторите harness doctor: "
                     "он выберет dxcam сам, если тот запустится")


def probe_screen(m: Machine, *, frames: int = 2, pause_s: float = 0.25
                 ) -> tuple[State, str, tuple[int, int] | None, bool]:
    """Взять кадры и посмотреть **содержимое**. Возвращает (состояние, текст, размер, чёрный).

    Единственная проверка, которая ловит отсутствие разрешения на macOS: кадр
    приходит нужной формы и заполнен нулями. «Вернулся ли кадр» на это отвечает «да».
    """
    import time

    from .capture.base import UNCHANGED
    from .capture.select import open_screen

    choice = open_screen(m, gray=True)
    if choice.source is None:
        return State.NO, choice.why_text(), None, False
    cap = choice.source
    try:
        shots = []
        unchanged = 0
        for i in range(max(2, frames)):
            f = cap.read()
            if f is UNCHANGED:
                # Экран не менялся — это ответ, а не пустота. Судить о содержимом
                # будем по тем кадрам, которые пришли.
                unchanged += 1
                if i == 0:
                    time.sleep(pause_s)
                continue
            if f is None:
                if shots:
                    break
                return State.NO, "захват вернул пустоту вместо кадра", None, False
            shots.append(f.image)
            if i == 0:
                time.sleep(pause_s)
        if not shots:
            return (State.NO,
                    f"механизм {choice.chosen.name} ни разу не отдал кадр "
                    f"(«без изменений» {unchanged} раз): первый кадр обязан прийти "
                    "всегда, даже на неподвижном экране", None, False)
    except Exception as e:                      # backend'ы бросают своё
        return State.NO, f"захват сорвался: {e}", None, False
    finally:
        cap.stop()

    h, w = shots[0].shape[:2]
    size = (w, h)
    spread = max(float(s.std()) for s in shots)
    peak = max(int(s.max()) for s in shots)
    if peak == 0:
        return (State.NO, f"кадр {w}×{h} целиком чёрный: захват вернул пустышку",
                size, True)
    if spread < 1.0:
        return (State.NO,
                f"кадр {w}×{h} однороден (разброс яркости {spread:.2f}): это не экран",
                size, True)
    detail = f"{w}×{h}, разброс яркости {spread:.1f}"
    if unchanged:
        detail += f", «без изменений» {unchanged} из {max(2, frames)}"
    return State.YES, detail, size, False


def check_display(m: Machine, probe: tuple[State, str, tuple[int, int] | None, bool]
                  ) -> Check:
    state, detail, _size, black = probe
    if state is State.YES:
        return Check("экран и его разрешение", State.YES, detail)
    fix = grant_screen_recording(m) if black else install_capture(m)
    return Check("экран и его разрешение", State.NO, detail, blocks=True, fix=fix)


def check_permissions(m: Machine,
                      probe: tuple[State, str, tuple[int, int] | None, bool]) -> Check:
    """Разрешения. Проверяются активно — по содержимому кадра, а не по файлу.

    Отдельным пунктом, а не примечанием к захвату: на macOS это самая частая причина
    пустой записи, и оператор должен увидеть её под своим именем, иначе будет искать
    ошибку в коде.
    """
    state, _detail, _size, black = probe
    if m.system == "Darwin":
        if state is State.YES:
            return Check("разрешение на запись экрана", State.YES,
                         "выдано: кадр непустой")
        return Check("разрешение на запись экрана", State.NO,
                     "не выдано или отозвано: кадр пустой" if black
                     else "проверить не удалось — захват не запустился",
                     blocks=True, fix=grant_screen_recording(m))
    if state is State.YES:
        return Check("разрешение на запись экрана", State.YES,
                     "в этой системе не спрашивается")
    return Check("разрешение на запись экрана", State.UNKNOWN,
                 "проверять нечем: захват не запустился", blocks=True,
                 fix=grant_screen_recording(m))


def check_ffmpeg(m: Machine) -> Check:
    """Уровень 3 журнала. Записи не мешает: без него кадры лежат кадрами."""
    path = shutil.which("ffmpeg")
    if path:
        return Check("ffmpeg", State.YES, path)
    return Check("ffmpeg", State.NO, "не найден в PATH",
                 later="третьему уровню журнала (сжатие кадров в видео)",
                 fix=install_ffmpeg(m))


def check_audio(m: Machine) -> Check:
    """Стереозахват. Моно не годится: пеленг считается из разницы каналов."""
    try:
        import sounddevice as sd            # noqa: PLC0415
    except ImportError:
        return Check("стереозахват звука", State.NO, "нет пакета sounddevice",
                     later="слуху агента (пеленг из разницы каналов)",
                     fix=pip_install("sounddevice", m))
    try:
        devices = list(sd.query_devices())
    except Exception as e:
        return Check("стереозахват звука", State.UNKNOWN,
                     f"звуковая подсистема не отвечает: {e}",
                     later="слуху агента",
                     fix=find_loopback(m))
    stereo_in = [d for d in devices if int(d.get("max_input_channels", 0)) >= 2]
    loop = [d for d in stereo_in
            if any(w in str(d.get("name", "")).lower()
                   for w in ("monitor", "loopback", "stereo mix", "стерео"))]
    if loop:
        return Check("стереозахват звука", State.YES,
                     f"петлевой вход: {loop[0]['name']}")
    if stereo_in:
        return Check("стереозахват звука", State.NO,
                     f"стереовходов {len(stereo_in)}, но петлевого среди них нет "
                     f"(микрофон не годится: он пишет комнату, а не то, что слышит "
                     f"агент)",
                     later="слуху агента",
                     fix=find_loopback(m))
    return Check("стереозахват звука", State.NO, "стереовходов нет вовсе",
                 later="слуху агента", fix=find_loopback(m))


def check_disk(m: Machine, *, path: Path, fps: float, size: tuple[int, int] | None
               ) -> Check:
    """Сколько часов записи влезет. Место — не «сколько гигабайт», а «сколько часов».

    Гигабайты оператору ничего не говорят: чтобы понять, хватит ли их, надо знать
    размер кадра и частоту, то есть читать код. Часы говорят сразу.
    """
    target = path if path.exists() else path.parent
    try:
        usage = shutil.disk_usage(target)
    except OSError as e:
        return Check("свободное место", State.UNKNOWN, f"{target}: {e}",
                     blocks=True, fix=f"создайте каталог: mkdir -p {path}")
    free_gb = usage.free / (1 << 30)
    per_frame = BYTES_PER_FRAME_1080P
    worst = WORST_BYTES_PER_FRAME_1080P
    if size:
        # Пропорция по площади кадра от 1080p: 320×180 в тридцать шесть раз меньше.
        area = (size[0] * size[1]) / (1920 * 1080)
        per_frame = max(512, int(BYTES_PER_FRAME_1080P * area))
        worst = max(2048, int(WORST_BYTES_PER_FRAME_1080P * area))
    hours = usage.free / max(1.0, per_frame * fps * 3600.0)
    worst_hours = usage.free / max(1.0, worst * fps * 3600.0)
    detail = (f"{free_gb:.1f} ГиБ свободно — это около {hours:.0f} ч записи "
              f"экрана при {fps:g} кадр/с, и всего {worst_hours:.1f} ч, если "
              f"писать видео во весь экран или игру с частицами")
    # Порог — по **реалистичной** оценке, а предел по шуму печатается рядом как
    # предупреждение. Блокировать по шуму нельзя: минимальный план — это браузер и
    # рабочий стол, там расход экранный, и отказ при тридцати свободных гигабайтах
    # остановил бы оператора на ровном месте. Полчаса — длина минимального набора:
    # запись, оборванная по месту, оборвётся честно, но работа пропадёт.
    if hours < 0.5:
        return Check("свободное место", State.NO, detail, blocks=True,
                     fix=f"освободите место или укажите другой диск: "
                         f"harness record ДРУГОЙ_ПУТЬ")
    return Check("свободное место", State.YES, detail)


def check_uinput(m: Machine) -> Check:
    """Права на инъекцию ввода. Для записи не нужны — сказать, но не блокировать."""
    if m.system == "Windows":
        return Check("права на инъекцию ввода", State.YES,
                     "SendInput доступен без настройки")
    if m.system == "Darwin":
        return Check("права на инъекцию ввода", State.NO,
                     "на macOS инъекция не реализована",
                     later="М5 (агент нажимает сам)", fix=enable_uinput(m))
    dev = Path("/dev/uinput")
    if not dev.exists():
        return Check("права на инъекцию ввода", State.NO, "/dev/uinput нет",
                     later="М5 (агент нажимает сам)", fix=enable_uinput(m))
    if os.access(dev, os.W_OK):
        return Check("права на инъекцию ввода", State.YES, "/dev/uinput доступен на запись")
    return Check("права на инъекцию ввода", State.NO,
                 "/dev/uinput есть, но записи в него нет",
                 later="М5 (агент нажимает сам)", fix=enable_uinput(m))


def run(*, path: Path | None = None, fps: float | None = None,
        probe: Callable[..., tuple[State, str, tuple[int, int] | None, bool]] | None = None
        ) -> Report:
    """Полный осмотр. `probe` подменяется в тестах: без экрана его не выполнить."""
    from .core.profile import MILESTONE_0

    m = detect()
    where = Path(path) if path else Path.home() / "harness-live"
    rate = float(fps if fps is not None else MILESTONE_0.parameters["capture_fps"])

    rep = Report(machine=m)
    rep.checks.append(check_python(m))
    rep.checks.append(check_deps(m))
    session = check_session(m)
    rep.checks.append(session)
    fullscreen = check_fullscreen(m)
    if fullscreen is not None:
        rep.checks.append(fullscreen)

    # Кадр берётся один раз, а выводов из него три: экран, разрешения, размер для
    # оценки места. Три отдельных захвата дали бы три разных ответа на одной и той
    # же машине — например, если разрешение выдали между проверками.
    if session.state is State.YES:
        shot = (probe or probe_screen)(m)
    else:
        shot = (State.UNKNOWN, "механизма захвата нет — пробовать нечем", None, False)
    if shot[0] is State.UNKNOWN:
        rep.checks.append(Check("экран и его разрешение", State.UNKNOWN, shot[1],
                                blocks=True, fix=session.fix or switch_to_x11()))
        rep.checks.append(Check("разрешение на запись экрана", State.UNKNOWN,
                                "проверять нечем: механизма захвата нет",
                                blocks=True, fix=session.fix or switch_to_x11()))
    else:
        rep.checks.append(check_display(m, shot))
        rep.checks.append(check_permissions(m, shot))

    rep.checks.append(check_disk(m, path=where, fps=rate, size=shot[2]))
    rep.checks.append(check_ffmpeg(m))
    rep.checks.append(check_audio(m))
    rep.checks.append(check_uinput(m))
    return rep
