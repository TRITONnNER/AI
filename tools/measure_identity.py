"""Замер по TASK-14: тождество карточек на двух источниках отпечатков.

Проверяется не схема карточки, а **тождество**, и ошибиться в нём можно в обе стороны.
Поэтому мир строится с известной истиной: в нём ровно `n_true` сущностей, каждая
наблюдается многократно, а отпечаток может их путать. Истина известна, значит измеримы
обе ошибки, а не только та, которую видно.

## Два источника отпечатков, и это условие задачи

Отпечаток строится восприятием, чьё качество на живом экране ещё не измерено. Поэтому
механика тождества прогоняется дважды: огрубляющим отпечатком (`Quantized`) и точным
(`Exact`). Если механика где-то опирается на свойства огрубления, это вылезет числом —
одна из двух колонок окажется вырожденной.

**Единица независимости.** Для чисел о карточках — карточка. Для утверждения «механика не
зависит от отпечатка» — **источник отпечатка**, и их ровно два: внутри одного источника
все карточки опознаны одной функцией, и независимых наблюдений о методе там одно.

Запуск: `python3 tools/measure_identity.py`. Результат — `docs/measurements/identity.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.core.profile import from_schema                       # noqa: E402
from harness.model.beliefs import (BeliefStore, EntityKind, Origin,  # noqa: E402
                                   Provenance, Relation, entity_id)
from harness.model.identity import (Exact, Quantized, revise)      # noqa: E402

BRANCH = "замер-тождества"


def build(store: BeliefStore, fp: Any, *, n_true: int, sightings: int,
          confuse: bool, drift: bool = False) -> dict[str, str]:
    """Мир с известной истиной: `n_true` сущностей по `sightings` встреч каждая.

    `confuse=True` — отпечаток различает сущности хуже, чем они различны: две из них
    получают близкие значения и попадают в одну корзину огрубления. Это и есть слипание,
    и его нужно уметь поймать расщеплением.

    Возвращает истину: карточка → настоящая сущность. Истина остаётся у исследователя и
    в механику не попадает (инвариант 12, тот же принцип).
    """
    truth: dict[str, str] = {}
    seq = 1
    for t in range(n_true):
        # Значение сущности. При `confuse` две первые сущности почти совпадают, и
        # огрубляющий отпечаток их не различит.
        base = 0.05 if (confuse and t < 2) else 0.3 + t * 0.7
        # Исход действия — свойство настоящей сущности, а не отпечатка. Именно на этом
        # расщепление и держится: слипшаяся карточка получит половину «да», половину
        # «нет», и по эпизодам их можно развести обратно.
        outcome = t % 2 == 0
        for i in range(sightings):
            # `drift` — значение одной сущности уползает больше чем на корзину: та же
            # вещь при другом освещении. Огрубляющий отпечаток даёт ей соседние корзины,
            # то есть **дробит** одну сущность на несколько карточек. Это вторая ошибка
            # тождества, и лечится она близостью отпечатков, а не расщеплением.
            shift = 0.0
            if drift and t == 0:
                shift = 0.11 * i
            elif confuse and t < 2:
                shift = 0.001 * i
            print_ = fp([base + shift])
            ent = entity_id(f"{fp.name}|{print_}")
            store.touch(ent, str(EntityKind.THING), seq, fingerprint=print_,
                        key="нажать", outcome=outcome, source=fp.name)
            store.learn_affordance(ent, "нажать", outcome,
                                   Provenance(Origin.EXPERIENCE, BRANCH, seq),
                                   kind=str(EntityKind.THING))
            truth.setdefault(ent, f"сущность-{t}")
            if truth[ent] != f"сущность-{t}":
                truth[ent] = "слипшаяся"
            seq += 1
    return truth


def one(fp: Any, *, n_true: int, sightings: int, confuse: bool,
        drift: bool = False) -> dict[str, Any]:
    profile = from_schema("тождество", split_min_observations=6, split_middle_band=0.12,
                          identity_min_shared_keys=1, identity_min_confidence=0.6,
                          identity_max_passes=6)
    store = BeliefStore(BRANCH)
    truth = build(store, fp, n_true=n_true, sightings=sightings, confuse=confuse,
                  drift=drift)
    glued_before = sum(1 for v in truth.values() if v == "слипшаяся")

    rep = revise(store, profile=profile, source=fp.name, fp=fp)
    d = rep.as_dict()
    d["glued_before"] = glued_before
    d["true_entities"] = n_true
    # Насколько число карточек после сна ближе к истине, чем до. Это и есть польза
    # пересмотра, выраженная числом, а не «слияния случились».
    d["error_before"] = abs(rep.before - n_true)
    d["error_after"] = abs(rep.after - n_true)
    d["split_confidence"] = [round(e.identity_confidence, 3)
                             for e in store.entities.values()
                             if e.identity_note.startswith("расщепление")][:4]
    return d


def main() -> int:
    cases = []
    # Три случая, и каждый — про свою ошибку тождества. Источников два, и вся разница
    # между ними должна быть в том, умеет ли источник отвечать на «это близко».
    # Случаи названы по тому, что делает **мир**, а не по тому, что делает отпечаток:
    # один и тот же мир под огрублением слипается, а под точным хешем дробится, и
    # направление ошибки — свойство источника, а не мира.
    настройки = (("сущности различимы", False, False),
                 ("две сущности почти одинаковы", True, False),
                 ("одна сущность меняет вид", False, True))
    for fp in (Quantized(bucket=0.1), Exact()):
        for name, confuse, drift in настройки:
            got = one(fp, n_true=6, sightings=8, confuse=confuse, drift=drift)
            got["case"] = name
            cases.append(got)

    data = {"cases": cases,
            "unit": "карточка",
            "unit_for_method": "источник отпечатка",
            "n_sources": 2,
            "why_two_sources": "механика тождества не должна зависеть от качества "
                               "отпечатка: если зависит, одна из колонок вырождается"}
    out = ROOT / "docs" / "measurements" / "identity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    for c in cases:
        print(f"{c['source']:<12} {c['case']:<32} карточек {c['before']:>2} → "
              f"{c['after']:>2} (истина {c['true_entities']}), "
              f"слияний {c['merges']} за {c['merge_passes']} прох., "
              f"расщеплений {c['splits']}, "
              f"ошибка {c['error_before']} → {c['error_after']}")
    print(f"\nзаписано: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
