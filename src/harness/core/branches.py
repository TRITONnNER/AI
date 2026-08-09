"""Счётчики срабатываний на ветках арбитража. Инвариант 26.

Основание — конкретный провал, а не осторожность. Первая версия замыкания петли в
разведке была **мёртвым кодом**: правило «идти в недозамкнутое место» стояло под
условием «пробовать непробованное здесь», и это условие срабатывало **1500 раз из
1500**. Непробованное не кончается никогда — каждый шаг открывает новое место и
пополняет очередь быстрее, чем она расходуется, — поэтому до нижнего правила дело не
доходило ни разу. Числа не сдвинулись ни на десятую, а код читался верно.

Установил это **замер**, а не чтение. Отсюда правило: в следующий раз это должно
обнаруживаться механически.

## Что здесь есть

`Arbitration` — набор именованных веток с порядком приоритета. Каждая ветка при
срабатывании отмечается, и по итогам прогона видно:

- **ветки с нулём срабатываний** — докладываются как дефект. Ветка либо мертва, либо
  её условие никогда не выполняется, и оба случая требуют объяснения;
- **ветки, чей приоритетный предшественник срабатывает более чем в 95 % случаев** —
  признак ненасыщаемого условия наверху, при котором всё нижележащее недостижимо.
  Именно этот случай и произошёл: 100 % у второго пункта из шести.

## Чего здесь нет

Никакой автоматической реакции. Счётчик ничего не меняет в поведении и не может: он
только считает и докладывает. Если бы он влиял на выбор, замер перестал бы измерять
выбор — он измерял бы себя.

Оверхед — один инкремент словаря на решение. На 20 Гц это ничто, и отключать его
незачем: инвариант 25 требует предъявлять сдвиг числа на всякую правку поведения, а
предъявлять его нечем, если не считать, что вообще происходило.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

#: Доля срабатываний предшественника, выше которой нижние ветки считаются
#: заблокированными. Порог из инварианта 26, и он тут константа не по недосмотру:
#: это не настройка поведения агента, а порог отчётности для исследователя. В
#: `profile_hash` он не попадает, потому что на опыт агента не влияет никак.
STARVATION_BAR = 0.95


class BranchError(RuntimeError):
    pass


@dataclass(slots=True)
class Branch:
    """Одна ветка арбитража: имя, зачем она и сколько раз сработала."""

    name: str
    why: str
    hits: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "why": self.why, "hits": self.hits}


@dataclass(slots=True)
class Arbitration:
    """Набор веток в порядке приоритета плюс счётчики.

    Порядок задаётся при создании и не меняется: он и есть предмет проверки. Ветка,
    отмеченная не по списку, — ошибка, а не тихое добавление: набор ветвей должен
    совпадать с тем, что объявлено, иначе отчёт о нулевых срабатываниях врёт
    умолчанием.
    """

    name: str
    branches: list[Branch] = field(default_factory=list)
    decisions: int = 0

    @classmethod
    def of(cls, name: str, spec: Sequence[tuple[str, str]]) -> Arbitration:
        return cls(name, [Branch(n, w) for n, w in spec])

    def __iter__(self) -> Iterator[Branch]:
        return iter(self.branches)

    def _find(self, name: str) -> Branch:
        for b in self.branches:
            if b.name == name:
                return b
        raise BranchError(
            f"{self.name}: нет ветки {name!r}. Объявлены: "
            f"{[b.name for b in self.branches]}. Ветка, которой нет в списке, не "
            "попадёт в отчёт о нулевых срабатываниях, и мёртвый код снова окажется "
            "невидимым")

    def hit(self, name: str) -> str:
        """Отметить срабатывание. Возвращает имя — удобно писать `return arb.hit(...)`."""
        self._find(name).hits += 1
        self.decisions += 1
        return name

    # --- отчёт --------------------------------------------------------------

    @property
    def dead(self) -> list[Branch]:
        """Ветки, не сработавшие ни разу. Докладываются как дефект."""
        return [b for b in self.branches if b.hits == 0]

    @property
    def dominant(self) -> Branch | None:
        """Ветка, срабатывающая чаще порога. Под ней всё может быть недостижимо."""
        if not self.decisions:
            return None
        for b in self.branches:
            if b.hits / self.decisions > STARVATION_BAR:
                return b
        return None

    def starved(self) -> list[Branch]:
        """Ветки ниже доминирующей: они не просто мертвы, а заблокированы сверху.

        Различение существенное. Мёртвая ветка может быть мертва потому, что её
        случай в этом мире не встречается, — это данные о мире. Заблокированная
        мертва потому, что до неё не доходит управление, — это дефект кода, и
        лечится он порядком, а не миром.
        """
        top = self.dominant
        if top is None:
            return []
        idx = self.branches.index(top)
        return [b for b in self.branches[idx + 1:] if b.hits == 0]

    def share(self, name: str) -> float | None:
        if not self.decisions:
            return None
        return self._find(name).hits / self.decisions

    def as_dict(self) -> dict[str, Any]:
        top = self.dominant
        return {
            "name": self.name,
            "decisions": self.decisions,
            "branches": [b.as_dict() for b in self.branches],
            "shares": {b.name: (None if not self.decisions
                                else round(b.hits / self.decisions, 4))
                       for b in self.branches},
            "dead": [b.name for b in self.dead],
            "dominant": None if top is None else top.name,
            "starved": [b.name for b in self.starved()],
        }

    def report(self) -> str:
        """Строки для отчёта о прогоне. Нулевые срабатывания называются вслух."""
        if not self.decisions:
            return (f"{self.name}: ни одного решения за прогон. Арбитраж не "
                    "вызывался — это тоже дефект, если он должен был вызываться")
        rows = [f"{self.name}: решений {self.decisions}"]
        for b in self.branches:
            mark = "  " if b.hits else "! "
            rows.append(f"  {mark}{b.name:<28} {b.hits:>6} "
                        f"{b.hits / self.decisions:>6.1%}")
        top = self.dominant
        if top is not None:
            rows.append(f"  ! ветка «{top.name}» срабатывает в "
                        f"{top.hits / self.decisions:.1%} случаев — выше порога "
                        f"{STARVATION_BAR:.0%}. Ненасыщаемое условие наверху делает "
                        "нижние недостижимыми")
        starved = self.starved()
        if starved:
            rows.append("  ! заблокированы сверху (дефект кода, не свойство мира): "
                        + ", ".join(b.name for b in starved))
        dead = [b for b in self.dead if b not in starved]
        if dead:
            rows.append("  ! ни разу не сработали (объяснить в отчёте): "
                        + ", ".join(b.name for b in dead))
        for b in dead:
            rows.append(f"      {b.name}: {b.why}")
        return "\n".join(rows)

    @property
    def has_defects(self) -> bool:
        return bool(self.dead or self.starved() or self.dominant is not None)


@dataclass(slots=True)
class Ledger:
    """Все арбитражи прогона вместе. То, что печатается в отчёте о прогоне."""

    arbitrations: dict[str, Arbitration] = field(default_factory=dict)

    def add(self, arb: Arbitration) -> Arbitration:
        self.arbitrations[arb.name] = arb
        return arb

    def get(self, name: str) -> Arbitration:
        try:
            return self.arbitrations[name]
        except KeyError:
            raise BranchError(f"нет арбитража {name!r}") from None

    def as_dict(self) -> dict[str, Any]:
        return {"arbitrations": [a.as_dict()
                                 for a in self.arbitrations.values()],
                "defects": self.defects()}

    def defects(self) -> list[str]:
        out: list[str] = []
        for a in self.arbitrations.values():
            for b in a.starved():
                out.append(f"{a.name}/{b.name}: заблокирована сверху")
            for b in a.dead:
                if b not in a.starved():
                    out.append(f"{a.name}/{b.name}: ни одного срабатывания")
            top = a.dominant
            if top is not None:
                out.append(f"{a.name}/{top.name}: срабатывает в "
                           f"{a.share(top.name):.0%} случаев")
        return out

    def report(self) -> str:
        rows = [a.report() for a in self.arbitrations.values()]
        found = self.defects()
        rows.append("")
        if found:
            rows.append(f"дефектов арбитража: {len(found)}")
            rows += [f"  • {x}" for x in found]
        else:
            rows.append("дефектов арбитража нет: все ветки срабатывали, "
                        "ни одна не доминирует")
        return "\n".join(rows)
