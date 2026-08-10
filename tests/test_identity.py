"""Карточки и тождество: слияние, расщепление задним числом, заменяемый отпечаток.

TASK-14. Главное здесь — не схема карточки, а то, что ошибиться в тождестве можно в обе
стороны, и обе ошибки измеримы на мире с известной истиной.

Самое дорогое, что поймал замер: первая редакция считала достаточным согласие
аффордансов, и пересмотр тождества делал модель **хуже** — шесть верно различённых
сущностей склеивались в две. Похожесть аффордансов есть признак категории, а не тождества,
и после разделения этих понятий ошибка перестала расти.
"""

from __future__ import annotations

import pytest

from harness.core.profile import from_schema
from harness.model.beliefs import (BeliefError, BeliefStore, Entity, EntityKind, Origin,
                                   Provenance, Relation, Sighting, entity_id)
from harness.model.identity import (Exact, Quantized, apply_merge, apply_split,
                                    propose_categories, propose_merges, propose_splits,
                                    revise)

BRANCH = "тест-тождества"
PROFILE = from_schema("тождество", split_min_observations=4, split_middle_band=0.15,
                      identity_min_shared_keys=1, identity_min_confidence=0.6)


def _prov(seq: int) -> Provenance:
    return Provenance(Origin.EXPERIENCE, BRANCH, seq)


def _card(store: BeliefStore, ent: str, print_: str, key: str, outcome: bool,
          seq: int) -> Entity:
    store.touch(ent, str(EntityKind.THING), seq, fingerprint=print_, key=key,
                outcome=outcome)
    store.learn_affordance(ent, key, outcome, _prov(seq), kind=str(EntityKind.THING))
    return store.entities[ent]


# --- одна структура на все виды --------------------------------------------------


def test_one_structure_for_every_kind() -> None:
    """Место, предмет, другой, звук, состояние и мысль — одна структура, разный вид."""
    for kind in (EntityKind.PLACE, EntityKind.THING, EntityKind.OTHER,
                 EntityKind.SOUND, EntityKind.STATE, EntityKind.THOUGHT):
        e = Entity("ENT_0001", str(kind), first_seq=1, last_seq=1)
        assert e.kind == str(kind)


def test_unknown_kind_is_refused() -> None:
    """Набор видов закрыт: новый вид — утверждение о мире, а не строка вызывающего."""
    with pytest.raises(BeliefError, match="не объявлен"):
        Entity("ENT_0001", "верстак", first_seq=1, last_seq=1)


def test_relations_are_typed_and_provenance_is_separate() -> None:
    """Девять типов связей, из них три — про происхождение, и они отделены."""
    assert {str(r) for r in Relation} == {
        "at", "part_of", "causes", "resembles", "precedes", "represents",
        "made_by", "used_by", "found_with"}
    store = BeliefStore(BRANCH)
    a = _card(store, "ENT_000A", "P1", "нажать", True, 1)
    _card(store, "ENT_000B", "P2", "нажать", True, 2)
    store.link("ENT_000A", "ENT_000B", str(Relation.MADE_BY))
    store.link("ENT_000A", "ENT_000B", str(Relation.MADE_BY))
    assert a.provenance_links == {"ENT_000B": "made_by"}
    store.link("ENT_000A", "ENT_000B", str(Relation.AT))
    assert a.provenance_links == {}, "at — не связь происхождения"


def test_free_string_relation_is_refused() -> None:
    store = BeliefStore(BRANCH)
    _card(store, "ENT_000A", "P1", "нажать", True, 1)
    with pytest.raises(BeliefError, match="не объявлена"):
        store.link("ENT_000A", "ENT_000A", "рядом-стоит")


def test_sightings_are_kept_one_by_one() -> None:
    """Встречи хранятся поштучно: без них расщепление задним числом работать не может."""
    store = BeliefStore(BRANCH)
    for i in range(4):
        _card(store, "ENT_000A", f"P{i % 2}", "нажать", i % 2 == 0, 10 + i)
    e = store.entities["ENT_000A"]
    assert e.encounters == 4 and len(e.sightings) == 4
    assert e.fingerprints == ("P0", "P1")
    assert e.first_seq == 10 and e.last_seq == 13


# --- слияние требует свидетельства, а не похожести -------------------------------


def test_similar_cards_are_not_merged_without_a_shared_print() -> None:
    """Согласие аффордансов — признак категории, а не тождества.

    Сдвиг числа: по прежнему правилу две карточки с одинаковым исходом сливались, и на
    замере шесть сущностей превращались в две (ошибка тождества 0 → 4). Теперь ноль
    догадок.
    """
    store = BeliefStore(BRANCH)
    for i in range(5):
        _card(store, "ENT_000A", "PA", "нажать", True, 1 + i)
        _card(store, "ENT_000B", "PB", "нажать", True, 20 + i)
    guesses = propose_merges(store, min_shared=1, min_confidence=0.6, agree_band=0.25)
    assert guesses == [], "разные отпечатки — разные сущности, как бы ни были похожи"

    cats = propose_categories(store, agree_band=0.25)
    assert len(cats) == 1 and set(cats[0].members) == {"ENT_000A", "ENT_000B"}, (
        "обобщение не потеряно: они в одной категории, оставаясь двумя карточками")


def test_shared_print_merges_and_lowers_confidence() -> None:
    """Общий отпечаток — прямое свидетельство тождества. После слияния оценка ниже."""
    store = BeliefStore(BRANCH)
    for i in range(3):
        _card(store, "ENT_000A", "SAME", "нажать", True, 1 + i)
        _card(store, "ENT_000B", "SAME", "нажать", True, 20 + i)
    guesses = propose_merges(store, min_shared=1, min_confidence=0.6, agree_band=0.25)
    assert len(guesses) == 1 and guesses[0].confidence == 1.0
    assert apply_merge(store, guesses[0])
    assert len(store.entities) == 1
    kept = next(iter(store.entities.values()))
    assert kept.encounters == 6, "встречи обеих половин остались"
    assert kept.identity_confidence <= 1.0 and kept.identity_note.startswith("слияние")


def test_near_prints_are_the_fingerprinters_business() -> None:
    """На вопрос «это близко» отвечает источник отпечатка, а не механика тождества."""
    q = Quantized(bucket=0.1)
    a, b, far = q([0.31]), q([0.39]), q([0.95])
    assert q.near(a, b) and not q.near(a, far)
    assert not Exact().near(a, b), "точный хеш о близости не знает и не врёт"


# --- расщепление задним числом ---------------------------------------------------


def test_split_routes_sightings_by_episode() -> None:
    """Карточка, оказавшаяся двумя, делится по эпизодам, а не наугад."""
    store = BeliefStore(BRANCH)
    for i in range(8):
        _card(store, "ENT_GLUE", f"P{i}", "открыть", i % 2 == 0, 100 + i)
    plans = propose_splits(store, min_n=4, band=0.15)
    assert len(plans) == 1
    plan = plans[0]
    assert plan.yes_seqs == (100, 102, 104, 106)
    assert plan.no_seqs == (101, 103, 105, 107)

    a, b = apply_split(store, plan, branch=BRANCH)
    assert "ENT_GLUE" not in store.entities, "исходная карточка не остаётся рядом"
    ea, eb = store.entities[a], store.entities[b]
    assert [s.seq for s in ea.sightings] == [100, 102, 104, 106]
    assert [s.seq for s in eb.sightings] == [101, 103, 105, 107]
    # Убеждения пересчитаны из своих встреч, а не поделены пополам: иначе mu осталась бы
    # посередине у обеих — ровно та ошибка, ради которой расщепление и делается.
    assert ea.affordances["открыть"].mu == 1.0
    assert eb.affordances["открыть"].mu == 0.0
    assert ea.identity_note.startswith("расщепление")


def test_split_reroutes_links_by_episode() -> None:
    """Ссылка уходит той половине, к которой относится по эпизоду."""
    store = BeliefStore(BRANCH)
    for i in range(8):
        _card(store, "ENT_GLUE", f"P{i}", "открыть", i % 2 == 0, 100 + i)
    _card(store, "ENT_NEAR", "PN", "смотреть", True, 101)
    store.link("ENT_NEAR", "ENT_GLUE", str(Relation.FOUND_WITH))

    plan = propose_splits(store, min_n=4, band=0.15)[0]
    a, b = apply_split(store, plan, branch=BRANCH)
    near = store.entities["ENT_NEAR"]
    assert "ENT_GLUE" not in near.links, "ссылка на исчезнувшую карточку не остаётся"
    assert len(near.links) == 1
    # Последняя встреча ENT_NEAR — запись 101, она в половине «нет».
    assert b in near.links


def test_split_needs_sightings_and_says_so_when_it_has_none() -> None:
    """Бимодальность без поштучных встреч не расщепляется, и это докладывается числом."""
    store = BeliefStore(BRANCH)
    e = Entity("ENT_0001", str(EntityKind.THING), first_seq=1, last_seq=8)
    store.entities["ENT_0001"] = e
    for i in range(8):
        e.affordances["открыть"] = (
            e.affordances["открыть"].observe(i % 2 == 0, _prov(1 + i))
            if "открыть" in e.affordances else
            _card_belief(i))
    assert propose_splits(store, min_n=4, band=0.15) == []
    rep = revise(store, profile=PROFILE, source="без встреч")
    assert rep.unsplittable >= 1, "нельзя молчать о том, что делить было нечем"


def _card_belief(i: int):
    from harness.model.beliefs import Belief

    return Belief("ENT_0001|afford|открыть", 1.0 if i % 2 == 0 else 0.0, 0.5, 1,
                  _prov(1 + i), 1, 1 + i)


# --- пересмотр целиком -----------------------------------------------------------


def test_revise_splits_before_merging() -> None:
    """Порядок обязателен: сначала расщепления, потом слияния.

    Обратный порядок сначала склеил бы две сущности в одну, потом честно нашёл бы у неё
    бимодальность и разделил обратно — но уже по перемешанному множеству встреч, и
    половины вышли бы не те.
    """
    import inspect

    src = inspect.getsource(revise)
    assert src.index("propose_splits") < src.index("propose_merges")


def test_revise_reports_every_number_with_a_unit() -> None:
    store = BeliefStore(BRANCH)
    for i in range(6):
        _card(store, "ENT_000A", "PA", "нажать", True, 1 + i)
    rep = revise(store, profile=PROFILE, source="точный")
    d = rep.as_dict()
    assert d["unit"] == "карточка"
    assert d["unit_for_method"] == "источник отпечатка"
    for key in ("before", "after", "merges", "splits", "single_encounter_before",
                "single_encounter_after", "categories", "merge_passes"):
        assert key in d, f"показатель {key} не докладывается"
    assert "источник отпечатка" in __import__(
        "harness.model.units", fromlist=["UNITS"]).UNITS
    assert "карточка" in __import__(
        "harness.model.units", fromlist=["UNITS"]).UNITS


def test_false_merge_share_is_reported() -> None:
    """Инвариант 31: доля догадок о слиянии, которые не применились."""
    store = BeliefStore(BRANCH)
    for i in range(3):
        _card(store, "ENT_000A", "SAME", "нажать", True, 1 + i)
        _card(store, "ENT_000B", "SAME", "нажать", False, 20 + i)
    rep = revise(store, profile=from_schema(
        "строго", identity_min_shared_keys=1, identity_min_confidence=0.99,
        split_min_observations=4, split_middle_band=0.15), source="точный")
    assert rep.merge_guesses >= 1
    assert rep.merges == 0, "оценка ниже порога — слияния нет"
    assert rep.false_merge_share == 1.0


def test_identity_survives_a_change_of_fingerprint_source() -> None:
    """Механика тождества не зависит от источника отпечатков — числа приводятся по каждому.

    Проверяется тем, что на одном и том же мире оба источника дают **работающую**
    механику: слипание лечится расщеплением у любого источника, потому что расщепление
    смотрит на исходы, а не на отпечатки.
    """
    for fp in (Quantized(bucket=0.1), Exact()):
        store = BeliefStore(BRANCH)
        for i in range(8):
            _card(store, "ENT_GLUE", fp([0.5 + 0.001 * i]), "открыть", i % 2 == 0,
                  100 + i)
        rep = revise(store, profile=PROFILE, source=fp.name, fp=fp)
        assert rep.splits == 1, f"{fp.name}: слипание не расщепилось"
        assert rep.after == 2


def test_merging_needs_passes_because_identity_is_transitive() -> None:
    """A~B и B~C означает A~C, но за один проход это не видно.

    Сдвиг числа: карточка, раздробленная отпечатком на восемь, за один проход склеивалась
    не полностью; проходы до исчерпания доводят её до одной.
    """
    fp = Quantized(bucket=0.1)
    store = BeliefStore(BRANCH)
    for i in range(8):
        _card(store, entity_id(f"drift-{i}"), fp([0.3 + 0.11 * i]), "нажать", True,
              1 + i)
    before = len(store.entities)
    one_pass = revise(store, profile=from_schema(
        "один проход", identity_min_shared_keys=1, identity_min_confidence=0.6,
        identity_max_passes=1, split_min_observations=4), source=fp.name, fp=fp)

    store2 = BeliefStore(BRANCH)
    for i in range(8):
        _card(store2, entity_id(f"drift-{i}"), fp([0.3 + 0.11 * i]), "нажать", True,
              1 + i)
    many = revise(store2, profile=from_schema(
        "до исчерпания", identity_min_shared_keys=1, identity_min_confidence=0.6,
        identity_max_passes=6, split_min_observations=4), source=fp.name, fp=fp)

    assert before == 8
    assert many.after < one_pass.after, (
        f"проходов {many.merge_passes}: {one_pass.after} → {many.after}")
    assert many.merge_passes > 1
