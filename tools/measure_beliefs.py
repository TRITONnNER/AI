"""Замер по TASK-12: конвейер убеждений, переоткрытие отложенного, дрейф памяти.

Четыре утверждения, каждое числом.

**1. Свидетельство не становится опытом от подтверждения.** До правки любое собственное
подтверждение повышало происхождение до `experience` — и догадку, и пересказ. Считается,
сколько убеждений корпуса меняли происхождение по прежнему правилу и сколько по новому.

**2. Отложенное отличается от «не проверено».** Прежнее условие `not h.checked` считало
вопросы очередью дел. Считается, сколько гипотез попадало в очередь по старому счёту и
сколько по новому.

**3. Переоткрытие работает и промахивается.** Замкнутая петля из `MIND.md`, раздел 5:
одинокий артефакт → аномалия → `deferred` → встреча с деятелями → разрешение задним
числом. Считаются доля разрешённых, среднее ожидание, доля ложных предложений и покрытие
(инвариант 31).

**4. Переходы конвейера все живые.** Инвариант 26: переход без ни одного срабатывания за
полный прогон — дефект.

Запуск: `python3 tools/measure_beliefs.py`. Результат — `docs/measurements/beliefs.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harness.core.branches import Ledger                        # noqa: E402
from harness.model.beliefs import (Belief, BeliefStore, Origin,  # noqa: E402
                                   Provenance, Stage, Testimony, merge_testimony,
                                   pipeline_arbitration)
from harness.model.questions import (Missing, OpenQuestions, token)  # noqa: E402

BRANCH = "замер-убеждений"


def rumours_do_not_become_experience(n: int = 40) -> dict[str, object]:
    """Сколько пересказов сменило бы происхождение по прежнему правилу.

    Считается на одном и том же наборе двумя правилами, а не «до и после правки»: старое
    правило воспроизводится здесь явно, потому что иначе сравнивать было бы нечего —
    старого кода уже нет.
    """
    store = BeliefStore(BRANCH)
    testimonies = [Testimony(f"ENT_{i:04X}|dyn|что-то", source="сосед",
                             trust=0.6, branch=BRANCH, seq=i) for i in range(n)]
    merge_testimony(store, testimonies, requires_recheck=True)

    flipped_old = flipped_new = 0
    for i, b in enumerate(list(store.beliefs())):
        own = Provenance(Origin.EXPERIENCE, BRANCH, 1000 + i)
        after = b.observe(True, own)
        # Прежнее правило: любое своё подтверждение повышало происхождение.
        if b.provenance.origin is not Origin.EXPERIENCE:
            flipped_old += 1
        if after.provenance.origin is not b.provenance.origin:
            flipped_new += 1
    return {
        "пересказов": n,
        "сменили_происхождение_по_прежнему_правилу": flipped_old,
        "сменили_по_новому": flipped_new,
        "единица": "утверждение",
    }


def hunches_still_become_experience() -> dict[str, object]:
    """Контроль: догадка обязана повышаться, иначе правка запретила лишнее."""
    prov = Provenance(Origin.HUNCH, BRANCH, 1)
    b = Belief("ENT_0001|dyn|догадка", 0.5, 0.5, 1, prov)
    after = b.observe(True, Provenance(Origin.EXPERIENCE, BRANCH, 2))
    return {"догадка_стала_опытом": after.provenance.origin is Origin.EXPERIENCE,
            "проверок_записано": len(after.checked_at)}


def queue_is_not_the_unexplained(n_questions: int = 12,
                                 n_in_work: int = 5) -> dict[str, object]:
    """Сколько гипотез попадало в «ожидает проверки» по старому счёту и по новому."""
    from harness.model.beliefs import Hypothesis

    store = BeliefStore(BRANCH)
    prov = Provenance(Origin.HUNCH, BRANCH, 1)
    for i in range(n_questions):
        store.add_hypothesis(Hypothesis.question(
            f"загадка-{i}", 0.5, prov, because="проверить нечем",
            reopens_on=(token(Missing.PLACE),)))
    for i in range(n_in_work):
        store.add_hypothesis(Hypothesis(f"дело-{i}", "нажать и посмотреть", 0.5, prov))

    old_count = sum(1 for h in store.hypotheses.values() if not h.checked)
    return {
        "вопросов": n_questions,
        "в_работе": n_in_work,
        "ожидает_проверки_по_прежнему_условию": old_count,
        "ожидает_проверки_теперь": len(store.unchecked_hypotheses()),
        "отложено_теперь": len(store.deferred()),
        "единица": "вопрос",
    }


def reopening_loop() -> dict[str, object]:
    """Замкнутая петля археологии, с долей ложных предложений и покрытием.

    Три вида вопросов намеренно: объявившие нехватку места (переоткроются), объявившие
    нехватку чтения (возможность появится, но тест не построится — ложное предложение) и
    слепые (не объявили ничего, не переоткроются никогда).
    """
    from harness.model.beliefs import Hypothesis

    reg = OpenQuestions()
    prov = Provenance(Origin.HUNCH, BRANCH, 1)
    seq = 10
    for i in range(6):                     # переоткроются: нужно место
        reg.ask(Hypothesis.question(
            f"кто оставил артефакт {i}", 0.4, prov,
            because="деятеля не наблюдал ни разу",
            reopens_on=(token(Missing.PLACE), token(Missing.ENTITY_KIND))), seq + i)
    for i in range(3):                     # ложные предложения: нужно чтение
        reg.ask(Hypothesis.question(
            f"что написано на табличке {i}", 0.3, prov,
            because="символы вижу, разобрать не умею",
            reopens_on=(token(Missing.READING),)), seq + 10 + i)
    for i in range(4):                     # слепые: не объявили нехватку
        reg.ask(Hypothesis.question(
            f"почему мир такой {i}", 0.2, prov,
            because="не знаю даже того, чего мне не хватает"), seq + 20 + i)

    def build(q, by: str) -> str | None:
        """Тест строится только там, где возможность и правда годится."""
        if by.startswith(str(Missing.READING)):
            return None                    # умение читать появилось, но не помогло
        return f"дойти до {by} и посмотреть"

    ledger = Ledger()
    arb = ledger.add(pipeline_arbitration())
    reopened = reg.offer(
        [token(Missing.PLACE, "PL_7F3A"), token(Missing.READING, "SYM_00A1")],
        seq=200, build_test=build, arb=arb)
    # Часть загадок разрешается **отрицательно**, и это не для полноты счётчиков.
    # `MIND.md`, раздел 5: «для артефактов прошлой линии вывод будет неверным —
    # обоснованно, с хорошей поддержкой». Переоткрытый вопрос, чья проверка опровергла
    # догадку, — ожидаемый исход, а не сбой, и переход `hypothesis→refuted` обязан
    # срабатывать. Без него ветка мертва (инвариант 26), что и показал первый прогон.
    for i, r in enumerate(reopened):
        reg.resolve(r.claim, outcome=i % 3 != 2, seq=260 + i * 7, arb=arb)

    s = reg.stats()
    return {**{k: v for k, v in s.items()},
            "переоткрыто": len(reopened),
            "переходы": arb.as_dict(),
            "мёртвые_переходы": [b.name for b in arb.dead],
            "единица": "вопрос"}


def main() -> int:
    data = {
        "свидетельство_не_опыт": rumours_do_not_become_experience(),
        "догадка_повышается": hunches_still_become_experience(),
        "очередь_не_перечень": queue_is_not_the_unexplained(),
        "переоткрытие": reopening_loop(),
    }
    out = ROOT / "docs" / "measurements" / "beliefs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    r = data["свидетельство_не_опыт"]
    print(f"пересказов {r['пересказов']}: по прежнему правилу происхождение сменили "
          f"{r['сменили_происхождение_по_прежнему_правилу']}, по новому "
          f"{r['сменили_по_новому']}")
    q = data["очередь_не_перечень"]
    print(f"гипотез {q['вопросов'] + q['в_работе']}: «ожидает проверки» было "
          f"{q['ожидает_проверки_по_прежнему_условию']}, стало "
          f"{q['ожидает_проверки_теперь']}, отложено {q['отложено_теперь']}")
    p = data["переоткрытие"]
    print(f"вопросов задано {p['asked']}, переоткрыто {p['переоткрыто']}, "
          f"разрешено {p['resolved']}, среднее ожидание {p['mean_wait_entries']:.0f} "
          f"записей")
    print(f"  покрытие {p['coverage']:.0%}, ложных предложений "
          f"{p['false_offer_share']:.0%} ({p['offers_refused']} из {p['offers']})")
    dead = p["мёртвые_переходы"]
    print(f"  мёртвых переходов конвейера: {len(dead)}"
          + (f" — {dead}" if dead else ""))
    print(f"записано: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
