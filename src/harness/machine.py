"""Что это за машина: ОС, оконная система, чем здесь захватывать экран.

Отдельный модуль, а не пара строк в `doctor`, по одной причине: **выбор механизма
захвата и рассказ о том, чего не хватает, обязаны опираться на одно и то же знание.**
Если backend решает по `DISPLAY`, а `doctor` — по `XDG_SESSION_TYPE`, то отчёт и
поведение расходятся, и оператор читает «захват есть» перед чёрным кадром.

## Почему X11 и Wayland различаются здесь, а не в комментарии

Это разные механизмы захвата, а не разные настройки одного. `mss` работает через
X11-протокол; под Wayland её либо не пускают вовсе, либо она видит только окна
XWayland, либо отдаёт чёрный кадр — и последнее хуже всего, потому что выглядит как
успех. Wayland требует PipeWire с портальным разрешением, и этого backend'а в проекте
нет. Поэтому на Wayland захват **отказывается**, а не пробует.

Молча выбрать не тот механизм нельзя: запись, сделанная под Wayland через XWayland,
неотличима по формату от настоящей, а по содержанию — мусор.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Session(StrEnum):
    """Оконная система. Не «ОС»: на Linux их две, и захват у них разный."""

    X11 = "x11"
    WAYLAND = "wayland"
    WINDOWS = "windows"
    MACOS = "macos"
    NONE = "none"          # графической сессии нет: сервер, контейнер, ssh


# Таблицы «есть ли захват для этой системы» здесь больше нет. Она была третьей копией
# одного знания (реестр кандидатов, эта таблица, проверка внутри `ScreenCapture.start`),
# и копии расходились. Единственный источник — `capture.select.CANDIDATES`; здесь у него
# только спрашивают.


@dataclass(frozen=True, slots=True)
class Machine:
    """Снимок машины. Всё, из чего дальше выводятся и проверки, и подсказки."""

    system: str                  # Linux / Windows / Darwin
    session: Session
    session_source: str          # по чему определили: важно для отчёта
    python: tuple[int, int, int]
    release: str

    @property
    def has_capture(self) -> bool:
        """Есть ли механизм в принципе. Какой именно — решает `capture.select`.

        Импорт внутри метода, а не наверху файла: `capture.select` импортирует `Session`
        отсюда, и импорт наверху был бы круговым. Это цена единственного источника
        знания, и она дешевле двух таблиц, которые расходятся.
        """
        from .capture.select import supported

        return supported(self.session)

    @property
    def capture_backend(self) -> str | None:
        """Что бы выбрали на этой машине. `None` — механизма нет.

        Смотрит на установленные пакеты через `capture.select.plan`, а не на таблицу:
        ответ зависит от машины, а не только от оконной системы.
        """
        from .capture.select import plan

        choice = plan(self)
        return choice.chosen.name if choice.chosen else None

    @property
    def os_name(self) -> str:
        return {"Linux": "Linux", "Windows": "Windows",
                "Darwin": "macOS"}.get(self.system, self.system)

    def as_dict(self) -> dict[str, Any]:
        return {"system": self.system, "os": self.os_name,
                "session": str(self.session), "session_source": self.session_source,
                "python": ".".join(str(x) for x in self.python),
                "release": self.release,
                "capture_backend": self.capture_backend}


def state_dir() -> "Path":
    """Куда класть то, что переживает запуск: последний замер расхода.

    Домашняя папка, а не каталог проекта: доктор и selftest могут запускаться из
    разных мест, а замер относится к машине, а не к копии репозитория.
    """
    from pathlib import Path

    return Path.home() / ".harness"


def detect(env: dict[str, str] | None = None) -> Machine:
    """Определить машину. `env` подменяется в тестах — иначе их не написать."""
    e = os.environ if env is None else env
    system = platform.system()
    if system == "Windows":
        session, source = Session.WINDOWS, "platform.system()"
    elif system == "Darwin":
        session, source = Session.MACOS, "platform.system()"
    else:
        session, source = _linux_session(e)
    return Machine(system=system, session=session, session_source=source,
                   python=(
                       __import__("sys").version_info[0],
                       __import__("sys").version_info[1],
                       __import__("sys").version_info[2]),
                   release=platform.release())


def _linux_session(e: dict[str, str]) -> tuple[Session, str]:
    """X11, Wayland или ничего. Порядок проверок — от надёжного к косвенному.

    `XDG_SESSION_TYPE` спрашивается первым, потому что он единственный говорит прямо.
    `WAYLAND_DISPLAY` перед `DISPLAY` — потому что под Wayland `DISPLAY` тоже обычно
    выставлен (XWayland), и решение по нему дало бы «x11» на Wayland-машине: ровно та
    ошибка, из-за которой захват вернул бы чёрный кадр вместо отказа.
    """
    kind = (e.get("XDG_SESSION_TYPE") or "").strip().lower()
    if kind == "wayland":
        return Session.WAYLAND, "XDG_SESSION_TYPE=wayland"
    if kind == "x11":
        return Session.X11, "XDG_SESSION_TYPE=x11"
    if e.get("WAYLAND_DISPLAY"):
        return Session.WAYLAND, f"WAYLAND_DISPLAY={e['WAYLAND_DISPLAY']}"
    if e.get("DISPLAY"):
        return Session.X11, f"DISPLAY={e['DISPLAY']}"
    return Session.NONE, "ни XDG_SESSION_TYPE, ни WAYLAND_DISPLAY, ни DISPLAY"


# ---------------------------------------------------------------------------
# Подсказки: что именно ввести на этой машине
# ---------------------------------------------------------------------------
#
# Строки здесь — не «установите ffmpeg», а то, что можно скопировать в терминал.
# Разница не в вежливости: «установите ffmpeg» заставляет оператора искать, как это
# делается в его системе, а это и есть то думание, которое задача запрещает.

def pip_install(packages: str, machine: Machine) -> str:
    """Команда установки пакетов для этой машины."""
    exe = "py -m pip" if machine.system == "Windows" else "python3 -m pip"
    return f"{exe} install {packages}"


def install_ffmpeg(machine: Machine) -> str:
    if machine.system == "Windows":
        return ("winget install --id Gyan.FFmpeg -e   "
                "# затем закройте и откройте терминал, чтобы обновился PATH")
    if machine.system == "Darwin":
        return "brew install ffmpeg"
    if shutil.which("dnf"):
        return "sudo dnf install -y ffmpeg"
    if shutil.which("pacman"):
        return "sudo pacman -S --needed ffmpeg"
    return "sudo apt update && sudo apt install -y ffmpeg"


def install_capture(machine: Machine) -> str:
    """Чем ставить зависимости захвата именно здесь.

    На Windows первым называется dxcam: он предпочтительный механизм, и предлагать
    вместо него mss значило бы советовать тот путь, который не видит полноэкранных
    игр и не держит тридцати кадров.
    """
    if machine.session is Session.WAYLAND:
        return switch_to_x11()
    if machine.system == "Windows":
        return pip_install("dxcam mss", machine)
    return pip_install("mss", machine)


def switch_to_x11() -> str:
    """Wayland: единственное, что оператор может сделать сам за минуту."""
    return ("выйдите из сеанса (Выйти / Log out), на экране входа нажмите шестерёнку "
            "у поля пароля и выберите сеанс «X11» или «Xorg», затем войдите снова. "
            "Проверить: echo $XDG_SESSION_TYPE — должно напечатать x11")


def grant_screen_recording(machine: Machine) -> str:
    if machine.system == "Darwin":
        return ("Системные настройки → Конфиденциальность и безопасность → Запись "
                "экрана → включить галочку у Терминала (или у приложения, из "
                "которого запускаете), затем **закрыть и открыть терминал заново** "
                "— без перезапуска разрешение не подхватится")
    if machine.session is Session.WAYLAND:
        return switch_to_x11()
    return "разрешение на захват экрана в этой системе не спрашивается"


def find_loopback(machine: Machine) -> str:
    """Как найти петлевое устройство: не микрофон, а то, что слышно из колонок."""
    if machine.system == "Linux":
        return ("pactl list short sources | grep monitor   "
                "# возьмите имя со словом monitor и передайте: "
                "harness record ... --audio-device ИМЯ")
    if machine.system == "Darwin":
        return ("brew install blackhole-2ch   # затем в «Настройка Audio-MIDI» "
                "создайте устройство с несколькими выходами; петлевого входа в "
                "macOS нет из коробки")
    return ("в «Звук → Запись» включите «Стерео микшер» (Stereo Mix). Если его нет, "
            "поставьте VB-CABLE: winget install --id VBAudio.VBCable -e")


def enable_uinput(machine: Machine) -> str:
    if machine.system == "Linux":
        return ("sudo modprobe uinput && sudo usermod -aG input $USER   "
                "# затем выйдите из системы и войдите снова")
    if machine.system == "Windows":
        return "не требуется: инъекция идёт через SendInput"
    return "на macOS инъекция ввода не реализована; для записи она не нужна"
