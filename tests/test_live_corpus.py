"""Приём живых записей с чужой машины. `TASK-03`, часть 2.

Саму запись делает оператор — в контейнере нет ни дисплея, ни игры. Здесь
проверяется то, что можно проверить без них: что чужая запись подхватывается **без
правок кода**, что негодная отвергается до попадания в корпус, и что IoU без
разметки не считается вовсе, а не считается по маске самого детектора.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.core.journal import Actor, ActorLayer, FormatError
from harness.core.profile import MILESTONE_0
from harness.corpus.live import (KINDS, METHODS, LiveError, Region, bench_live,
                                 compare_to_synthetic, ingest, plan_text,
                                 read_regions, truth_mask, write_regions)
from harness.session import Recorder


def _record(root: Path, *, synthetic: bool, frames: int = 30,
            human: bool = True, chrome: bool = True) -> Path:
    """Запись, похожая на демонстрацию оператором: обрамление стоит, мир едет."""
    rng = np.random.default_rng(3)
    world = rng.integers(40, 200, (180, 320 + frames * 3), dtype=np.uint8)
    panel = rng.integers(0, 255, (180, 80), dtype=np.uint8)
    actor = Actor.HUMAN if human else Actor.NONE
    layer = ActorLayer.HUMAN if human else ActorLayer.NONE
    with Recorder(root, profile=MILESTONE_0, source="screen_mss",
                  synthetic=synthetic, note="демонстрация") as rec:
        for i in range(frames):
            frame = world[:, i * 3:i * 3 + 320].copy()
            if chrome:
                frame[:, :80] = panel
            rec.record_frame(frame, actor=actor, actor_layer=layer)
    return root


def test_synthetic_recording_is_not_accepted_as_live(tmp_path: Path) -> None:
    """Смысл живого корпуса ровно в том, что он не порождён допущениями детектора."""
    src = _record(tmp_path / "fake", synthetic=True)
    with pytest.raises(LiveError, match="помечена синтетической"):
        ingest(src, tmp_path / "corpus", kind="play")
    assert not (tmp_path / "corpus").exists(), (
        "негодная запись попала в корпус: проверки обязаны идти до копирования")


def test_live_recording_is_accepted_as_is(tmp_path: Path) -> None:
    src = _record(tmp_path / "demo", synthetic=False)
    got = ingest(src, tmp_path / "corpus", kind="camera_only")
    assert got.frames == 30 and got.format == "v2"
    assert got.actor_layers.get("human") == 30
    assert got.path.exists() and (got.path / "session.json").exists()
    # «Как есть»: журнал скопирован побитово, ничего не пересчитано.
    orig = (src / "journal").rglob("entries.jsonl")
    copy = (got.path / "journal").rglob("entries.jsonl")
    assert (next(orig).read_bytes() == next(copy).read_bytes())


def test_unknown_kind_is_refused_with_the_closed_list(tmp_path: Path) -> None:
    src = _record(tmp_path / "demo", synthetic=False)
    with pytest.raises(LiveError, match="Набор закрыт"):
        ingest(src, tmp_path / "corpus", kind="что-нибудь")
    # Набор закрыт, но не заморожен: TASK-07 добавил три вида, которым игра не
    # нужна, — прокрутку, видео и перетаскивание окон. Проверяется закрытость, а не
    # конкретный список: **каждый** объявленный вид обязан входить хотя бы в один
    # набор записи, иначе он объявлен и никем не записывается.
    from harness.corpus.live import SETS

    assert set(KINDS) >= {"stillness", "camera_only", "play", "menu", "death",
                          "unfamiliar"}
    in_sets = {k for spec in SETS.values() for k in spec["kinds"]}
    assert in_sets == set(KINDS), (
        f"виды вне наборов: {set(KINDS) - in_sets}. Вид, которого нет ни в одном "
        "наборе, никто не запишет")


def test_recording_without_human_layer_is_warned_not_refused(tmp_path: Path) -> None:
    """Странно, но не смертельно: запись принимается с пометкой."""
    src = _record(tmp_path / "demo", synthetic=False, human=False)
    got = ingest(src, tmp_path / "corpus", kind="play")
    assert any("слоем human" in w for w in got.warnings)


def test_tampered_recording_never_enters_the_corpus(tmp_path: Path) -> None:
    src = _record(tmp_path / "demo", synthetic=False)
    entries = next((src / "journal").rglob("entries.jsonl"))
    lines = entries.read_text(encoding="utf-8").splitlines()
    # Правка, которая оставляет строку разбираемой: ловится хешем, а не парсером.
    lines[2] = lines[2].replace('"t_world":2', '"t_world":3')
    entries.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LiveError, match="цепочка хешей"):
        ingest(src, tmp_path / "corpus", kind="play")
    assert not (tmp_path / "corpus").exists()


def test_unparseable_line_gives_a_clear_refusal(tmp_path: Path) -> None:
    """Битая строка — внятный отказ, а не сырой ValueError из перечисления."""
    src = _record(tmp_path / "demo", synthetic=False)
    entries = next((src / "journal").rglob("entries.jsonl"))
    lines = entries.read_text(encoding="utf-8").splitlines()
    lines[2] = lines[2].replace('"kind":"frame"', '"kind":"framez"')
    entries.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(LiveError, match="доехала испорченной"):
        ingest(src, tmp_path / "corpus", kind="play")
    assert not (tmp_path / "corpus").exists()


def test_iou_is_refused_without_markup(tmp_path: Path) -> None:
    """На живой записи истины нет, и подставить её нечем.

    Считать IoU против маски, полученной тем же детектором, значило бы сравнить
    метод с самим собой и получить единицу при любом качестве.
    """
    src = _record(tmp_path / "demo", synthetic=False)
    got = ingest(src, tmp_path / "corpus", kind="camera_only")
    score = bench_live(got.path)
    assert score.iou is None and score.recall is None and score.precision is None
    assert "истины не существует" in score.absent_reason
    # А то, что считается без истины, посчитано: это признаки жизни, не качество.
    assert score.decided > 0.0
    assert "не считается" in score.text()


def test_markup_makes_iou_computable(tmp_path: Path) -> None:
    src = _record(tmp_path / "demo", synthetic=False)
    got = ingest(src, tmp_path / "corpus", kind="camera_only")
    write_regions(got.path, [Region(0, 0, 180, 80, "screen")], note="панель слева")
    score = bench_live(got.path)
    assert score.iou is not None and score.truth_px == 180 * 80
    assert score.iou > 0.5, f"обрамление стоит, а найдено плохо: IoU {score.iou}"


def test_markup_lives_in_the_debug_channel(tmp_path: Path) -> None:
    """Координаты — истина исследователя, и агентскому коду они недоступны."""
    src = _record(tmp_path / "demo", synthetic=False)
    got = ingest(src, tmp_path / "corpus", kind="play")
    path = write_regions(got.path, [Region(1, 2, 3, 4)])
    assert path.parent.name == "debug"
    assert read_regions(got.path)[0] == Region(1, 2, 3, 4, "screen")
    # В журнал координаты не попали ни одним способом.
    text = next((got.path / "journal").rglob("entries.jsonl")).read_text(
        encoding="utf-8")
    assert "regions" not in text and '"top"' not in text


def test_both_existing_paths_run_on_the_same_recording(tmp_path: Path) -> None:
    """Синтетические числа сняты разными путями, поэтому нужны оба."""
    src = _record(tmp_path / "demo", synthetic=False)
    got = ingest(src, tmp_path / "corpus", kind="camera_only")
    write_regions(got.path, [Region(0, 0, 180, 80, "screen")])
    scores = {m: bench_live(got.path, method=m) for m in METHODS}
    assert set(scores) == {"parallax", "arbiter"}
    for m, s in scores.items():
        assert s.method == m and s.iou is not None


def test_verdict_says_grew_when_it_grew() -> None:
    """Сводка описывает случившееся, а не ожидание.

    Падение метрики на живом ожидалось, но если она выросла, писать «падение
    ожидалось» нельзя: ожидание важнее результата ровно один раз — когда оно не
    подтвердилось.
    """
    from harness.corpus.live import LiveScore

    high = [LiveScore(path=Path("x"), kind="play", frames=10, iou=0.99),
            LiveScore(path=Path("y"), kind="menu", frames=10, iou=0.98),
            LiveScore(path=Path("z"), kind="death", frames=10, iou=0.97)]
    grew = compare_to_synthetic(high, synthetic_median=0.667)
    assert "выросла" in grew["verdict"] and "ожидалось падение" in grew["verdict"]

    low = [LiveScore(path=Path("x"), kind="play", frames=10, iou=0.3),
           LiveScore(path=Path("y"), kind="menu", frames=10, iou=0.35),
           LiveScore(path=Path("z"), kind="death", frames=10, iou=0.4)]
    fell = compare_to_synthetic(low, synthetic_median=0.667)
    assert "Падение — ожидавшийся исход" in fell["verdict"]


def test_unmarked_recordings_do_not_improve_the_average() -> None:
    """Записи без разметки в сравнение не входят и перечисляются отдельно."""
    from harness.corpus.live import LiveScore

    mixed = [LiveScore(path=Path("x"), kind="play", frames=10, iou=0.4),
             LiveScore(path=Path("y"), kind="menu", frames=10)]
    cmp = compare_to_synthetic(mixed, synthetic_median=0.667)
    assert cmp["comparable"] == 1 and cmp["unscored"] == ["menu"]


def test_nothing_to_compare_says_so(tmp_path: Path) -> None:
    from harness.corpus.live import LiveScore

    cmp = compare_to_synthetic(
        [LiveScore(path=Path("x"), kind="play", frames=10)], synthetic_median=0.667)
    assert cmp["comparable"] == 0
    assert "единственный честный ответ" in cmp["verdict"]


def test_plan_names_every_recording_of_its_set() -> None:
    """План печатает свой набор целиком. Наборов два, и по умолчанию минимальный.

    Прежняя редакция теста требовала все шесть видов в одном выводе. Это перестало
    быть верным по замыслу: план по умолчанию печатает минимальный набор, потому что
    план, первой строкой требующий установить игру, откладывается на выходные.
    """
    from harness.corpus.live import SETS

    for set_name, spec in SETS.items():
        text = plan_text(set_name)
        for name in spec["kinds"]:
            k = KINDS[name]
            assert name in text and k["title"] in text and k["duration"] in text
        assert "harness mark" in text, (
            "без разметки IoU не посчитается — это надо сказать в каждом наборе")


def test_truth_mask_is_built_from_rectangles() -> None:
    mask = truth_mask([Region(2, 3, 4, 5), Region(0, 0, 1, 1, "world")], (10, 10))
    assert mask.sum() == 20, "область world в маску экрана не входит"
    assert mask[2, 3] and not mask[0, 0]
