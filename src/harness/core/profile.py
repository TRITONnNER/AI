"""Профиль настроек: единственное место, где живут константы поведения.

Инвариант 11: структурные переключатели форкают журнал. Параметры меняются на
ходу. Смешивать в одной структуре нельзя — поэтому здесь две отдельные группы,
а не одна с флажками.

## Два хеша, а не один

В CLAUDE.md сказано «хеш профиля штампуется в каждую запись», а в 0.5 —
«по хешу можно отфильтровать журнал и получить только совместимые записи».
Эти два требования несовместимы с одним хешем: параметры разрешено крутить на
ходу, и если хеш считается по ним тоже, то каждый поворот ручки делает
предыдущие записи «несовместимыми», хотя ветка журнала та же.

Поэтому хеша два, и штампуются оба:

- `structure_hash` — только по `structural`. Определяет ветку журнала и
  совместимость записей. Меняется → форк, старый опыт с новым не сравнивается.
- `profile_hash` — по `structural` и `parameters` вместе. Точная настройка
  конкретной записи: по нему видно, при каких ручках получен этот кусок опыта.

Записи сравнимы, когда совпал `structure_hash`. Различие `profile_hash` внутри
одной ветки — это не помеха сравнению, а условие эксперимента, и его нужно
видеть.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

# Значения, которые могут лежать в профиле. Ничего сложнее: хеш обязан быть
# стабильным между запусками и между машинами, а для этого сериализация должна
# быть однозначной.
Scalar = bool | int | float | str | None


class ProfileError(ValueError):
    """Профиль собран так, что хеш перестал что-либо значить."""


def _check_group(name: str, group: Mapping[str, Any]) -> dict[str, Scalar]:
    out: dict[str, Scalar] = {}
    for k, v in group.items():
        if not isinstance(k, str) or not k:
            raise ProfileError(f"{name}: ключ не строка: {k!r}")
        if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
            out[k] = v
        elif isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")):
                raise ProfileError(f"{name}.{k}: nan/inf не даёт стабильного хеша")
            out[k] = v
        else:
            raise ProfileError(
                f"{name}.{k}: тип {type(v).__name__} не сериализуется однозначно; "
                "в профиле допустимы только bool, int, float, str, None"
            )
    return out


def _canonical(obj: Mapping[str, Scalar]) -> bytes:
    """Однозначная сериализация: сортированные ключи, без пробелов, без ASCII-эскейпов.

    Отдельно фиксируем представление float: repr() в Python стабилен между
    версиями для double, а json.dumps использует его же.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _digest(*parts: bytes) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(len(p).to_bytes(8, "big"))
        h.update(p)
    return h.hexdigest()


def short(digest: str) -> str:
    """Как хеш показывается человеку: #4F2A9C. Для глаза, не для сравнения."""
    return "#" + digest[:6].upper()


@dataclass(frozen=True, slots=True)
class Profile:
    """Все настройки агента. Захардкоженных констант поведения нигде больше нет.

    Неизменяем: «поворот ручки» — это новый объект профиля и новая запись в
    журнале о том, что ручку повернули. Мутировать настройки молча нельзя,
    иначе журнал перестанет объяснять поведение.
    """

    name: str
    parameters: dict[str, Scalar] = field(default_factory=dict)
    structural: dict[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ProfileError("у профиля должно быть имя: оно попадёт в журнал")
        both = set(self.parameters) & set(self.structural)
        if both:
            raise ProfileError(
                f"ключи и в parameters, и в structural: {sorted(both)}. "
                "Настройка либо крутится на ходу, либо форкает журнал, третьего нет"
            )
        object.__setattr__(self, "parameters", _check_group("parameters", self.parameters))
        object.__setattr__(self, "structural", _check_group("structural", self.structural))

    # --- хеши ---------------------------------------------------------------

    @property
    def structure_hash(self) -> str:
        """Ветка журнала. Совместимость записей определяется этим и только этим."""
        return _digest(b"structural", _canonical(self.structural))

    @property
    def profile_hash(self) -> str:
        """Точная настройка записи: структура плюс параметры."""
        return _digest(
            b"structural", _canonical(self.structural),
            b"parameters", _canonical(self.parameters),
        )

    # --- изменения ----------------------------------------------------------

    def with_parameters(self, **changes: Scalar) -> Profile:
        """Повернуть ручки. Ветка журнала не меняется, `structure_hash` тот же."""
        unknown = set(changes) - set(self.parameters)
        if unknown:
            raise ProfileError(
                f"нет таких параметров: {sorted(unknown)}. "
                "Новую настройку сначала объявите в профиле по умолчанию"
            )
        return Profile(self.name, {**self.parameters, **changes}, dict(self.structural))

    def with_structural(self, **changes: Scalar) -> Profile:
        """Переключить структуру. Вызывающий обязан форкнуть журнал: продолжать
        писать в старую ветку с новым `structure_hash` журнал не даст."""
        unknown = set(changes) - set(self.structural)
        if unknown:
            raise ProfileError(f"нет таких структурных переключателей: {sorted(unknown)}")
        return Profile(self.name, dict(self.parameters), {**self.structural, **changes})

    def forks_journal(self, other: Profile) -> bool:
        """Нужен ли форк при переходе от self к other."""
        return self.structure_hash != other.structure_hash

    def diff(self, other: Profile) -> list[dict[str, Any]]:
        """Различия для человека: что было, что стало, структурное или нет."""
        out: list[dict[str, Any]] = []
        for group, hard in (("parameters", False), ("structural", True)):
            a: dict[str, Scalar] = getattr(self, group)
            b: dict[str, Scalar] = getattr(other, group)
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    out.append({"key": k, "group": group, "structural": hard,
                                "from": a.get(k), "to": b.get(k)})
        return out

    def is_clean_ablation(self, other: Profile) -> bool:
        """Ровно одно различие. Иначе сравнение прогонов ничего не доказывает."""
        return len(self.diff(other)) == 1

    # --- сохранение ---------------------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": dict(self.parameters),
            "structural": dict(self.structural),
            "profile_hash": self.profile_hash,
            "structure_hash": self.structure_hash,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Profile:
        p = cls(str(d["name"]), dict(d.get("parameters", {})), dict(d.get("structural", {})))
        for key in ("profile_hash", "structure_hash"):
            claimed = d.get(key)
            if claimed is not None and claimed != getattr(p, key):
                raise ProfileError(
                    f"{key} в файле не совпадает с пересчитанным: {claimed} != {getattr(p, key)}. "
                    "Либо файл правили руками, либо изменилась схема хеширования"
                )
        return p

    def save(self, path) -> None:
        from pathlib import Path
        Path(path).write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> Profile:
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Профиль вехи 0. Всё, что здесь есть, — настройки харнесса; поведения агента в
# этой вехе нет вообще, поэтому и настроек поведения нет.
# ---------------------------------------------------------------------------

MILESTONE_0 = Profile(
    name="ХАРНЕСС-0",
    parameters={
        # захват
        "capture_fps": 30.0,
        "capture_width": 1280,
        "capture_height": 720,
        # звук
        "audio_rate": 48000,
        "audio_block_ms": 20.0,
        "audio_sync_tolerance_ms": 50.0,   # из 0.1: рассинхрон больше — брак
        # инъекция ввода
        "watchdog_still_seconds": 20.0,    # экран не менялся столько — стоп
        "watchdog_still_threshold": 0.002,  # доля изменившихся пикселей
        "stop_max_latency_frames": 1.0,    # из 0.2: СТОП за один кадр
        # разделение себя и мира (0.6)
        # Окно поддержки для попиксельной классификации, в пикселях, нечётное.
        # Должно быть меньше самого мелкого элемента интерфейса: окно крупнее
        # элемента смешивает слои внутри себя (замер — в ARCHITECTURE-HARNESS.md).
        "flow_window": 7,
        "flow_min_global_shift": 1.5,      # ниже этого сдвига кадр неинформативен
        "screen_layer_tolerance": 0.5,     # насколько пиксель может отличаться от нуля
        "world_layer_tolerance": 0.5,      # ...и от глобального сдвига
        "selfworld_min_votes": 3,          # пиксель судится минимум по столько кадров
        # хранение
        "frame_shard_bytes": 64 * 1024 * 1024,
        "frame_compress_level": 6,
        # Через сколько кадров ставить опорный. Он же ограничивает цену промотки:
        # произвольный кадр собирается не больше чем из этого числа блоков.
        "frame_keyframe_interval": 30,
    },
    structural={
        # Меняешь — форкаешь журнал. Всё, что здесь, меняет форму данных или
        # смысл опыта, а не его количество.
        "frame_format": "gray8",       # gray8 | rgb8 — меняет форму кадра
        "audio_channels": 2,           # стерео обязательно: из разницы каналов пеленг
        "symbol_salt_id": "s0",        # смена соли меняет все символы разом
        "text_symbolized": True,       # False дало бы агенту читаемый текст
        "debug_channel_enabled": True,  # поток исследователя, агенту недоступен
    },
)
