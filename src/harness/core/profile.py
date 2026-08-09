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

    def for_frame(self, image: Any) -> Profile:
        """Профиль, у которого заявленный размер кадра совпадает с настоящим.

        Настройки `capture_width` и `capture_height` **не читает ни одна строка на
        пути захвата экрана**: и mss, и dxcam отдают монитор целиком, а уменьшать
        никто не просил. Читают их только синтетические миры — они по этим числам
        кадр рисуют.

        Из-за этого первая живая сессия оператора получила профиль с
        `capture_width: 320, capture_height: 180` при кадрах 1920×1080. Вреда для
        данных нет — кадры целы, — но запись **врёт о себе**: любой, кто посчитает по
        `session.json` расход или размер, получит числа в тридцать шесть раз меньше.
        Именно на этом сломался разбор расхода места: 49.3 КиБ на запись поделили на
        56 КиБ сырого кадра 320×180 и получили «сжатие не работает», тогда как делить
        надо было на 2025 КиБ, и сжатие работает в сорок один раз.

        Объявленная и никем не читаемая настройка — это ложь о возможностях. Здесь она
        попадала прямо в запись, поэтому живой захват обязан строить профиль по
        первому кадру: размер кадра решает экран, а не пожелание в профиле.
        """
        h, w = int(image.shape[0]), int(image.shape[1])
        if (int(self.parameters["capture_width"]) == w
                and int(self.parameters["capture_height"]) == h):
            return self
        return self.with_parameters(capture_width=w, capture_height=h)

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
        """Различия для человека: что было, что стало, структурное или нет.

        У каждой строки есть `live`: могло ли это различие что-то изменить в живой
        записи. Настройки синтетического мира лежат и в живом профиле — так решено
        сознательно (`settings.SYNTHETIC_ONLY`), — и без этой подписи они давали бы
        ложные различия при сверке двух живых сессий.
        """
        from .settings import applies_to_live

        out: list[dict[str, Any]] = []
        for group, hard in (("parameters", False), ("structural", True)):
            a: dict[str, Scalar] = getattr(self, group)
            b: dict[str, Scalar] = getattr(other, group)
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    out.append({"key": k, "group": group, "structural": hard,
                                "from": a.get(k), "to": b.get(k),
                                "live": applies_to_live(k)})
        return out

    def live_diff(self, other: Profile) -> list[dict[str, Any]]:
        """Только те различия, которые могли повлиять на живую запись."""
        return [d for d in self.diff(other) if d["live"]]

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
# Профили собираются из схемы (`harness.core.settings`), а не набираются руками:
# так ни одна ручка не потеряется и ни одна не окажется не в своей группе.
# ---------------------------------------------------------------------------


def from_schema(name: str, **overrides: Scalar) -> Profile:
    """Профиль со значениями по умолчанию и точечными изменениями.

    Куда попадёт изменённая ручка — в `parameters` или в `structural`, — решает
    схема, а не вызывающий. Ошибиться в эту сторону слишком дорого: настройка,
    меняющая форму опыта, обязана форкать журнал.
    """
    from . import settings as sch

    parameters = sch.defaults(structural=False)
    structural = sch.defaults(structural=True)
    for key, value in overrides.items():
        setting = sch.BY_KEY.get(key)
        if setting is None:
            raise ProfileError(f"неизвестная настройка {key!r}: её нет в схеме")
        checked = setting.check(value)
        (structural if setting.structural else parameters)[key] = checked
    sch.validate(parameters, structural)
    return Profile(name, parameters, structural)


def validate_against_schema(profile: Profile) -> None:
    """Проверить готовый профиль. Отдельно от конструктора: загруженный из файла
    профиль может быть от другой версии схемы, и об этом надо сказать прямо."""
    from . import settings as sch

    sch.validate(profile.parameters, profile.structural)


# Полный профиль по умолчанию: все настройки схемы со значениями по умолчанию.
DEFAULT = from_schema("ПОЛНЫЙ-0")

# Профиль вехи 0: то же, но с кадром 320×180 — размер синтетического корпуса и
# рабочий размер для офлайн-прогонов. Частота захвата и формат кадра прежние.
MILESTONE_0 = from_schema("ХАРНЕСС-0", capture_width=320, capture_height=180)

# Профиль лепета: агент открывает своё тело. Отличается от вехи 0 двумя ручками,
# то есть чистой абляцией не является — и `is_clean_ablation` это скажет. Для
# сравнения прогонов нужна пара, различающаяся ровно одной ручкой.
# `babble_rate` здесь больше не задаётся: пока ручку никто не читал, значение 0.9
# было незамеченным, а с появлением потребителя оно молча замедлило бы лепет на
# десятую часть во всех прогонах на этом профиле — и сдвиг чисел пришлось бы
# объяснять новой осью модуляции, тогда как причина была бы в старой описке.
BABBLE = from_schema("ЛЕПЕТ-К7", capture_width=320, capture_height=180,
                     babble_repeats=4)
