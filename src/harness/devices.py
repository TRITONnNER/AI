"""Устройства ввода и вывода.

Из дизайна пульта, экран «Устройства»: «видеть и управлять — разные права».
Это не удобство интерфейса, а содержательное различие. Источник, который агент
видит, но не управляет, «остаётся для него погодой»: он меняется сам, его надо
предсказывать, а влиять нельзя. Источник, которым можно управлять, но не видно, —
это действие с невидимым последствием, и такое надо уметь задавать отдельно,
потому что именно на нём ломается связь действия и результата.

Что здесь есть:

- **Реестр источников** с раздельными правами, частотой и задержкой.
- **Склейка**: несколько видимых источников собираются в один кадр, и агент
  получает их как одну картинку. Переключение между ними становится его
  собственным действием (`device_switch_is_action`), а не настройкой снаружи.
- **Бюджет задержки**: у каждого звена своя, и они складываются. Задержка —
  диагностический признак, поэтому она считается и показывается, а не
  замазывается.
- **Области действия ввода**: `device` шире `app`, `app` шире `window`. Та же
  вложенность, что у маски ввода.

Чего здесь нет: настоящих имён окон и приложений в том, что уходит агенту.
Идентификатор устройства для агента — непрозрачный (`SRC_1A`), а человеческое
имя живёт рядом и в перечислении для агента не участвует (инвариант 4).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Sequence

import numpy as np

from .core.clocks import Stamp
from .core.journal import Actor, ActorLayer, Journal, Kind as EntryKind

SCOPES = ("device", "app", "profile", "window")


class DeviceError(ValueError):
    pass


class Kind(StrEnum):
    DISPLAY = "display"        # монитор целиком
    WINDOW = "window"          # окно приложения
    REGION = "region"          # прямоугольник, заданный рамкой
    AUDIO_IN = "audio_in"      # петлевой захват звука
    AUDIO_OUT = "audio_out"    # агент издаёт звук
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    GAMEPAD = "gamepad"


VISUAL = (Kind.DISPLAY, Kind.WINDOW, Kind.REGION)


def source_id(human_name: str) -> str:
    """Непрозрачный идентификатор источника из человеческого имени.

    Имя окна — это текст с экрана, и агенту он не достаётся (инварианты 4 и 5).
    Идентификатор стабилен между запусками: иначе агент не смог бы связать опыт
    с источником после перезапуска.
    """
    h = hashlib.blake2b(human_name.encode("utf-8"), digest_size=2).hexdigest().upper()
    return f"SRC_{h}"


@dataclass(slots=True)
class Device:
    """Одно звено ввода-вывода."""

    name: str                      # человеческое имя; агенту не показывается
    kind: Kind
    see: bool = False              # входит ли в кадр агента
    control: bool = False          # можно ли на него влиять
    hz: float = 0.0                # частота, 0 — неизвестна
    latency_ms: float = 0.0        # задержка этого звена
    region: tuple[int, int, int, int] | None = None   # left, top, width, height
    scope: str = "window"          # область действия ввода
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise DeviceError("у устройства должно быть имя — для исследователя")
        if self.scope not in SCOPES:
            raise DeviceError(f"нет такой области действия: {self.scope!r}")
        if self.hz < 0 or self.latency_ms < 0:
            raise DeviceError("частота и задержка не бывают отрицательными")
        if self.kind in VISUAL and self.region is not None:
            _, _, w, h = self.region
            if w <= 0 or h <= 0:
                raise DeviceError(f"пустая область: {self.region}")
        if self.kind not in VISUAL and self.region is not None:
            raise DeviceError(f"{self.kind}: область имеет смысл только у видимого")
        if self.control and self.kind is Kind.AUDIO_IN:
            raise DeviceError("петлевым захватом звука управлять нельзя: это вход")

    @property
    def id(self) -> str:
        return source_id(self.name)

    @property
    def is_weather(self) -> bool:
        """Видно или слышно, но не управляемо: для агента это погода.

        Меняется само, предсказывать надо, влиять нельзя. Относится к тому, что
        вообще может входить в восприятие: к картинке и к звуку на входе.
        """
        return self.see and not self.control and self.kind in (*VISUAL, Kind.AUDIO_IN)

    @property
    def is_blind_hand(self) -> bool:
        """Управляемо, но не видно: действие с невидимым последствием.

        Только про то, что *могло бы* быть видно, — то есть про источники
        картинки. Клавиатура невидима по своей природе, и записывать её в
        «невидимые руки» значило бы засорять диагностику шумом.
        """
        return self.control and not self.see and self.kind in VISUAL

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": str(self.kind), "see": self.see,
                "control": self.control, "hz": self.hz,
                "latency_ms": self.latency_ms, "scope": self.scope,
                "region": list(self.region) if self.region else None}

    def for_agent(self) -> dict[str, Any]:
        """То же, но без человеческого имени и без области в экранных координатах.

        Координаты области — это система отсчёта, которой у агента быть не должно
        (инвариант 4). Размер он видит из самого кадра, положение ему незачем.
        """
        return {"id": self.id, "see": self.see, "control": self.control}


@dataclass(slots=True)
class Registry:
    """Все устройства сессии и то, как из них собирается кадр."""

    devices: list[Device] = field(default_factory=list)
    max_sources: int = 4
    switch_is_action: bool = True
    active: str | None = None          # какой источник в фокусе, если склейки нет
    journal: Journal | None = None

    # --- состав -------------------------------------------------------------

    def add(self, device: Device) -> Device:
        if any(d.id == device.id for d in self.devices):
            raise DeviceError(f"устройство {device.name!r} уже есть в реестре")
        self.devices.append(device)
        if device.see and self.active is None:
            self.active = device.id
        return device

    def get(self, device_id: str) -> Device:
        for d in self.devices:
            if d.id == device_id:
                return d
        raise DeviceError(f"нет устройства {device_id}")

    def visible(self) -> list[Device]:
        return [d for d in self.devices if d.see and d.kind in VISUAL]

    def controllable(self) -> list[Device]:
        return [d for d in self.devices if d.control]

    def by_kind(self, kind: Kind) -> list[Device]:
        return [d for d in self.devices if d.kind is kind]

    # --- права --------------------------------------------------------------

    def set_rights(self, device_id: str, *, see: bool | None = None,
                   control: bool | None = None, stamp: Stamp | None = None,
                   reason: str = "") -> Device:
        """Изменить права. Пишется в журнал: смена прав меняет мир агента."""
        d = self.get(device_id)
        before = (d.see, d.control)
        if see is not None:
            d.see = see
        if control is not None:
            d.control = control
        if d.control and d.kind is Kind.AUDIO_IN:
            d.control = before[1]
            raise DeviceError("петлевым захватом звука управлять нельзя")
        if len(self.visible()) > self.max_sources:
            d.see, d.control = before
            raise DeviceError(
                f"видимых источников стало больше предела {self.max_sources}; "
                "поднимите device_max_sources в профиле")
        if self.journal is not None and stamp is not None and before != (d.see, d.control):
            self.journal.append(EntryKind.DEVICE, stamp, Actor.HUMAN, ActorLayer.HUMAN,
                                event={"code": "rights", "device": d.id,
                                       "see": d.see, "control": d.control,
                                       "was_see": before[0], "was_control": before[1],
                                       "reason": reason})
        if self.active == d.id and not d.see:
            visible = self.visible()
            self.active = visible[0].id if visible else None
        return d

    def switch(self, device_id: str, stamp: Stamp | None = None, *,
               actor: Actor = Actor.AGENT,
               actor_layer: ActorLayer = ActorLayer.NONE) -> Device:
        """Перевести фокус на источник.

        При `switch_is_action` это действие агента и пишется от его имени: он сам
        решил посмотреть в другое место, и в журнале это должно быть видно как
        его выбор, а не как настройка снаружи.
        """
        d = self.get(device_id)
        if not d.see:
            raise DeviceError(f"{device_id} не виден: переключаться некуда")
        was, self.active = self.active, d.id
        if self.journal is not None and stamp is not None and was != d.id:
            # Когда переключение — действие агента, слой приходит от того, кто
            # переключил: посмотреть в другое место может решить и планировщик, и
            # драйв. Когда это настройка снаружи, инициатор — человек.
            # Слой в теле события — **тот же**, что в самой записи. Раньше здесь стояло
            # `str(actor_layer)` всегда: при выключенном `switch_is_action` запись
            # заявляла инициатором человека, а её тело — слой агента. Расхождение
            # заявленного и настоящего инициатора внутри одной записи — это ровно то, что
            # метрика конфабуляции считает у планировщика, и заводить его в самом журнале
            # значит портить измерительный прибор.
            real_layer = actor_layer if self.switch_is_action else ActorLayer.HUMAN
            self.journal.append(
                EntryKind.DEVICE, stamp,
                actor if self.switch_is_action else Actor.HUMAN,
                real_layer,
                event={"code": "switch", "device": d.id, "was": was,
                       "as_action": self.switch_is_action,
                       "actor_layer": str(real_layer),
                       # Что просил вызывающий — тоже в записи: если он просил один слой,
                       # а настройка превратила его в другой, это видно, а не потеряно.
                       "asked_layer": str(actor_layer)})
        return d

    # --- кадр ---------------------------------------------------------------

    def displays(self) -> list[Device]:
        """Видимые мониторы. Их число решает, склейка или внимание (TASK-21, часть 5)."""
        return [d for d in self.devices if d.see and d.kind is Kind.DISPLAY]

    def attention_frame(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        """Кадр того источника, **на который агент смотрит сейчас**.

        TASK-21, часть 5. Человек с двумя мониторами имеет одно поле зрения и переводит
        взгляд; он не видит оба монитора одновременно как склеенную картинку двойной
        ширины. Поэтому несколько мониторов — это **направление внимания**, а не два
        параллельных канала: в кадр идёт активный источник, а перевод взгляда есть
        отдельное действие (`switch`, и оно пишется в журнал со своим `actor_layer`).

        Почему это важнее удобства: склейка двух мониторов даёт агенту то, чего нет ни у
        одного человека, — одновременное наблюдение двух мест. С такой картинкой он
        никогда не научится тому, что взгляд надо переводить, что за пределами взгляда
        мир продолжает меняться и что об этом надо помнить. Ровно эти три вещи и есть
        содержание задачи о внимании.
        """
        if self.active is None:
            raise DeviceError("нет активного источника: смотреть некуда")
        if self.active not in frames:
            raise DeviceError(f"нет кадра от активного источника {self.active}")
        return frames[self.active]

    def compose(self, frames: dict[str, np.ndarray]) -> np.ndarray:
        """Склеить кадры видимых источников в один.

        Склейка по горизонтали в порядке добавления в реестр — порядок стабилен,
        иначе агент видел бы «прыгающий» мир при перезапуске. Источники разной
        высоты дополняются снизу нулями: додумывать содержимое нельзя, а чёрное
        поле агент увидит и сам разберётся, что это край.

        **Два монитора не склеиваются** (TASK-21, часть 5): для них есть
        `attention_frame`. Отказ здесь, а не соглашение в документации, потому что
        соглашение отваливается на первом же вызывающем, который о нём не знал.
        """
        visible = self.visible()
        if not visible:
            raise DeviceError("нет ни одного видимого источника")
        screens = self.displays()
        if len(screens) > 1:
            raise DeviceError(
                f"видимых мониторов {len(screens)}, и склеивать их в один кадр нельзя: "
                "человек с двумя мониторами имеет одно поле зрения и переводит взгляд, а "
                "не смотрит в оба одновременно. Кадр берётся у активного источника "
                "(attention_frame), перевод взгляда — действие (switch)")
        missing = [d.id for d in visible if d.id not in frames]
        if missing:
            raise DeviceError(f"нет кадров от источников: {missing}")

        parts = [frames[d.id] for d in visible]
        if len(parts) == 1:
            return parts[0]
        if len({p.ndim for p in parts}) != 1:
            raise DeviceError("склеиваются кадры разной размерности: серый и цветной")
        height = max(p.shape[0] for p in parts)
        padded = []
        for p in parts:
            if p.shape[0] == height:
                padded.append(p)
                continue
            pad = [(0, height - p.shape[0]), (0, 0)] + ([(0, 0)] if p.ndim == 3 else [])
            padded.append(np.pad(p, pad, mode="constant"))
        return np.concatenate(padded, axis=1)

    # --- задержка -----------------------------------------------------------

    def latency_budget(self) -> dict[str, Any]:
        """Из чего складывается задержка до действия.

        Считается по звеньям, а не замеряется целиком: целиком видно только на
        живой машине, а по звеньям видно, какое из них виновато.
        """
        rows = [{"device": d.id, "kind": str(d.kind), "latency_ms": d.latency_ms}
                for d in self.devices if d.latency_ms > 0]
        see = sum(d.latency_ms for d in self.devices if d.see)
        control = sum(d.latency_ms for d in self.devices if d.control)
        return {"links": rows, "see_ms": round(see, 3), "control_ms": round(control, 3),
                "round_trip_ms": round(see + control, 3)}

    def frame_period_ms(self, capture_fps: float) -> float:
        return 1000.0 / max(1e-6, capture_fps)

    def latency_in_frames(self, capture_fps: float) -> float:
        return self.latency_budget()["round_trip_ms"] / self.frame_period_ms(capture_fps)

    # --- сводки -------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "devices": len(self.devices),
            "visible": [d.id for d in self.visible()],
            "controllable": [d.id for d in self.controllable()],
            "weather": [d.id for d in self.devices if d.is_weather],
            "blind_hands": [d.id for d in self.devices if d.is_blind_hand],
            "active": self.active,
            "switch_is_action": self.switch_is_action,
            "latency": self.latency_budget(),
        }

    def for_agent(self) -> list[dict[str, Any]]:
        """Что об устройствах знает агент: сколько их и что с каждым можно."""
        return [d.for_agent() for d in self.devices if d.see or d.control]

    # --- сборка из профиля --------------------------------------------------

    @classmethod
    def from_profile(cls, profile, *, journal: Journal | None = None,
                     devices: Iterable[Device] | Sequence[Device] = ()) -> Registry:
        p = profile.parameters
        s = profile.structural
        reg = cls(max_sources=int(p["device_max_sources"]),
                  switch_is_action=bool(s["device_switch_is_action"]),
                  journal=journal)
        for d in devices:
            reg.add(d)
        if bool(s["audio_output_enabled"]) and not reg.by_kind(Kind.AUDIO_OUT):
            reg.add(Device("вывод звука", Kind.AUDIO_OUT, control=True,
                           note="агент может издавать звук: вывод — тоже канал действия"))
        return reg


def default_registry(profile, *, journal: Journal | None = None) -> Registry:
    """Набор по умолчанию: один видимый и управляемый экран, клавиатура, мышь, звук.

    Задержки — не выдуманные «типичные», а нули: неизвестная задержка обязана
    выглядеть как неизвестная. Заполняются они замером на живой машине.
    """
    p = profile.parameters
    reg = Registry.from_profile(profile, journal=journal)
    reg.add(Device("основной экран", Kind.DISPLAY, see=True, control=True,
                   hz=float(p["capture_fps"]), scope=str(p["input_scope"])))
    reg.add(Device("клавиатура", Kind.KEYBOARD, control=True,
                   scope=str(p["input_scope"])))
    reg.add(Device("мышь", Kind.MOUSE, control=True, scope=str(p["input_scope"]),
                   note=f"режим указателя: {profile.structural['pointer_mode']}"))
    reg.add(Device("петлевой звук", Kind.AUDIO_IN, see=True,
                   hz=1000.0 / float(p["audio_block_ms"])))
    return reg
