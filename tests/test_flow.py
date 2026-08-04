"""Оптический поток и дальность: что заработало, что нет и почему именно.

Главный тест здесь — отрицательный: дальность **не находится** ни одним из трёх
методов, и это закреплено числами. Когда он упадёт, кто-то сделал настоящий шаг, и
тогда надо обновить таблицу в описании `harness.vision.flow`, а не порог в тесте.

Второй по важности — про мир, а не про метод: на текстуре из блоков одного размера
большое смещение не находится в принципе, потому что пирамиде масштабов нечего ловить
на грубом уровне. Этот тест держит поправку в мире осмысленной: без него легко
вернуться к вырожденной текстуре и снова «измерять метод».
"""

from __future__ import annotations

import numpy as np
import pytest

from harness.core.action import Action
from harness.core.profile import from_schema
from harness.corpus import depth as dep
from harness.vision import flow as fl
from harness.vision.layers import FAR, MID, NEAR, UNKNOWN, DepthFromParallax

pytestmark = pytest.mark.skipif(not fl.available(),
                                reason="нет OpenCV: плотный поток посчитать нечем")


def _profile(**kw):
    base = dict(capture_width=320, capture_height=180)
    base.update(kw)
    return from_schema("ТЕСТ-поток", **base)


def _octaves(rng: np.random.Generator, h: int, w: int,
             octaves: tuple[tuple[int, float], ...]) -> np.ndarray:
    """Текстура из октав с весами. Вес — доля энергии на этом масштабе."""
    acc = np.zeros((h, w), dtype=np.float64)
    total = 0.0
    for octave, weight in octaves:
        cells = rng.integers(40, 200, (h // octave + 2, w // octave + 2), dtype=np.uint8)
        big = np.repeat(np.repeat(cells, octave, axis=0), octave, axis=1)[:h, :w]
        acc += weight * big.astype(np.float64)
        total += weight
    return np.clip(acc / total, 0, 255).astype(np.uint8)


# --- почему мир пришлось поправить ------------------------------------------


def test_large_shift_needs_more_than_one_scale() -> None:
    """Смещение 34 px находится на текстуре из нескольких октав и не находится на одной.

    Это про мир, а не про метод. Пирамида начинает с грубого уровня, чтобы поймать
    смещение целиком; блоки одного размера при уменьшении усредняются в ровное поле, и
    ловить там нечего. Замер, который стоил поправки в `corpus/depth.py`:

    | Текстура | Найденное смещение (ждём 34) |
    |---|---|
    | один масштаб, блоки 6 px | −2.4 |
    | четыре октавы, равные веса | +34.0 |
    | четыре октавы, веса как в мире | +34.0 |

    Веса октав, как видно, роли не играют: важно само наличие масштабов. В мире они
    всё равно неравные — так у естественных изображений, — но обосновывать их этим
    замером нельзя, он про другое. Первая версия этого теста утверждала обратное:
    у неё генератор случайных чисел был общий на все случаи, поэтому «равные веса»
    получили другую текстуру, а не другие веса.
    """
    h, w, shift = 180, 400, 34
    cases = {
        "один масштаб": ((6, 1.0),),
        "равные веса": ((24, 1.0), (12, 1.0), (6, 1.0), (3, 1.0)),
        "веса как в мире": ((24, 0.45), (12, 0.25), (6, 0.18), (3, 0.12)),
    }
    found = {}
    for name, octaves in cases.items():
        tex = _octaves(np.random.default_rng(0), h, w, octaves)
        a = tex[:, 40:40 + 320].copy()
        b = tex[:, 40 - shift:40 - shift + 320].copy()
        found[name] = float(np.median(fl.dense_flow(a, b)[..., 0]))

    for name in ("равные веса", "веса как в мире"):
        assert abs(found[name] - shift) < 3.0, found
    assert abs(found["один масштаб"] - shift) > 15.0, (
        f"на одном масштабе смещение внезапно нашлось: {found}. Тогда поправка в мире "
        "не нужна — перезамерьте и обновите таблицу в harness.vision.flow")


# --- отказ, а не тихий откат ------------------------------------------------


def test_missing_opencv_is_a_loud_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Нет OpenCV — отказ. Тихо посчитать «как-нибудь иначе» нельзя.

    Тихий откат здесь страшнее отсутствия дальности: замер показал бы числа, а по
    числам было бы не видно, что считал не тот метод.
    """
    monkeypatch.setattr(fl, "_CV2", None)
    monkeypatch.setattr(fl, "_CV2_TRIED", True)
    assert not fl.available()
    with pytest.raises(fl.FlowUnavailable, match="нет OpenCV"):
        fl.cv2_module()
    with pytest.raises(fl.FlowUnavailable):
        fl.DepthFromFlow(_profile())


def test_unknown_flow_preset_and_method_fail_loudly() -> None:
    a = np.zeros((32, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="неизвестный режим потока"):
        fl.dense_flow(a, a, preset="самый лучший")
    with pytest.raises(ValueError, match="неизвестный метод дальности"):
        fl.depth_from_profile(type("P", (), {"structural": {"depth_method": "ага"}})())


def test_depth_method_comes_from_the_profile() -> None:
    for method, cls in (("effort", fl.DepthFromEffort), ("flow", fl.DepthFromFlow),
                        ("parallax", DepthFromParallax)):
        assert isinstance(fl.depth_from_profile(_profile(depth_method=method)), cls)


def test_effort_refuses_without_effort() -> None:
    """Не двигался — параллакса нет. Это отсутствие данных, а не «всё далеко»."""
    p = _profile()
    d = fl.DepthFromEffort(p)
    w = dep.DepthWorld(p, seed=3)
    w.looming_on = False
    for _ in range(6):
        d.feed(w.step(None, with_audio=False).frame, effort=0.0)
    r = d.result()
    assert r.voting_frames == 0 and r.skipped_frames >= 4
    assert (r.bands == UNKNOWN).all()
    assert not r.trustworthy


def test_flow_refuses_when_the_camera_stands_still() -> None:
    p = _profile()
    d = fl.DepthFromFlow(p)
    w = dep.DepthWorld(p, seed=3)
    w.looming_on = False
    for _ in range(6):
        d.feed(w.step(None, with_audio=False).frame)
    r = d.result()
    assert r.voting_frames == 0
    assert (r.bands == UNKNOWN).all()


# --- главный отрицательный результат ----------------------------------------


def _accuracy(kind: str, seed: int, frames: int = 14) -> dict[str, float]:
    p = _profile(flow_window=11, depth_method=kind)
    w = dep.DepthWorld(p, seed=seed)
    w.looming_on = False
    d = fl.depth_from_profile(p)
    rng = np.random.default_rng(1)
    for _ in range(frames):
        dx = int(rng.integers(-14, 15))
        act = Action.mouse(dx, 0, duration_ms=33) if dx else None
        frame = w.step(act, with_audio=False).frame
        if kind == "effort":
            d.feed(frame, effort=float(dx))
        else:
            d.feed(frame)
    r = d.result()
    truth = w.depth_truth()
    out = {}
    for t, band, name in ((dep.TRUTH_NEAR, NEAR, "ближний"),
                          (dep.TRUTH_MID, MID, "средний"),
                          (dep.TRUTH_FAR, FAR, "дальний")):
        m = (truth == t) & (r.bands != UNKNOWN)
        out[name] = 0.0 if not m.any() else float((r.bands[m] == band).mean())
    return out


@pytest.mark.parametrize("kind", ["parallax", "flow", "effort"])
def test_depth_is_partly_found_but_not_reliably(kind: str) -> None:
    """Дальность находится частично, и разброс больше разницы между методами.

    Замер, который стоит за этим тестом: 25 прогонов на метод (5 сидов × 5 длин), мир
    с глубиной, три плана, случайное угадывание 0.33.

    | Метод | Среднее | Мин | Макс | σ |
    |---|---|---|---|---|
    | parallax | 0.49 | 0.25 | 0.83 | 0.16 |
    | flow | 0.50 | 0.30 | 0.73 | 0.14 |
    | effort | 0.54 | 0.27 | 0.82 | 0.20 |

    Здесь проверяется не точность, а именно это: результат выше случайного и заведомо
    ниже надёжного. Порог сверху щедрый (0.80) — при таком разбросе одиночный прогон
    может дать и 0.83, и тест не должен падать от везения. Он упадёт, когда среднее по
    нескольким прогонам действительно уедет вверх, и тогда таблицу в описании
    `harness.vision.flow` надо переписать, а не порог здесь.
    """
    runs = [_accuracy(kind, seed=seed, frames=frames)
            for seed, frames in ((3, 16), (5, 22), (7, 28))]
    balances = [sum(r.values()) / 3.0 for r in runs]
    mean = sum(balances) / len(balances)
    assert mean > 0.20, (
        f"{kind}: баланс {mean:.2f} — ниже случайного 0.33 с запасом. Что-то "
        f"сломалось: {runs}")
    assert mean < 0.80, (
        f"{kind}: баланс {mean:.2f} при замеренных 0.49–0.54. Дальность заработала — "
        "перезамерьте 25 прогонов и обновите таблицу в harness.vision.flow")


def test_single_run_says_nothing_about_depth_accuracy() -> None:
    """Разброс между прогонами больше разницы между методами. Это тоже измерено.

    Тест существует затем, чтобы этот вывод нельзя было потерять. Три формулировки
    итога в истории этого модуля были неверны подряд, и все три — потому, что делались
    по одному прогону.

    Замеренный разброс на этой выборке (5 сидов × длины 15 и 30) — 0.14. Он и близко
    не весь: по всем 25 прогонам у потока от 0.30 до 0.73, причём длины 20 и 25 дают
    заметно хуже, чем 15 и 30. Почему точность так зависит от длины прогона —
    неизвестно; медиана по кадрам должна была бы сглаживать, а не так скакать.
    """
    balances = [sum(_accuracy("flow", seed=seed, frames=frames).values()) / 3.0
                for seed in (3, 5, 7, 11, 13) for frames in (15, 30)]
    assert max(balances) - min(balances) > 0.10, (
        f"разброс между прогонами стал маленьким: {balances}. Если это правда, "
        "одиночному прогону снова можно верить — но это надо перезамерить")


def test_depth_results_are_marked_untrustworthy() -> None:
    """Пока баланс на уровне случайного, ответ обязан быть помечен ненадёжным."""
    p = _profile()
    for kind in ("effort", "flow"):
        d = fl.depth_from_profile(_profile(depth_method=kind))
        assert not d.result().trustworthy
        assert not d.result().summary()["trustworthy"]
    del p


def test_effort_reference_is_the_agents_own_action() -> None:
    """Опора — своё усилие, а не движение, найденное в картинке.

    Проверяется по поведению: при вдвое большем усилии найденное отношение не
    меняется, потому что делится на само усилие. Если бы опорой было движение из
    картинки, отношение поехало бы вместе с ним.
    """
    p = _profile(flow_window=11, depth_search_px=24)
    ratios = []
    for scale in (1, 2):
        w = dep.DepthWorld(p, seed=3)
        w.looming_on = False
        d = fl.DepthFromEffort(p)
        for _ in range(8):
            dx = 6 * scale
            d.feed(w.step(Action.mouse(dx, 0, duration_ms=33),
                          with_audio=False).frame, effort=float(dx))
        r = d.result()
        ratios.append(float(np.median(r.ratio[np.isfinite(r.ratio)])))
    assert abs(ratios[0] - ratios[1]) < 0.35, (
        f"отношение поехало вместе с усилием: {ratios}. Значит опорой стало не своё "
        "действие, а что-то, найденное в картинке")
