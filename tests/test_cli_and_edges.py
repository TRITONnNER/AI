"""Командная строка и краевые случаи, на которых обычно и рассыпается харнесс."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness.core.blobstore import BlobStore, FrameStore, StoreError
from harness.core.profile import MILESTONE_0, Profile, ProfileError
from harness.core.symbols import SymbolError, Symbolizer, normalize
from harness.session import Recorder, Session, SessionError


# --- CLI -------------------------------------------------------------------


@pytest.mark.environment("подменяются и сессия, и наличие пакета mss")
def test_cli_backends_reports_honestly(capsys: pytest.CaptureFixture[str],
                                      as_session, with_package,
                                      without_package) -> None:
    """Доклад о доступности соответствует **объявленной** машине, а не текущей.

    Прежняя редакция сравнивала строку доклада с наличием `DISPLAY` и падала на машине
    оператора: на Windows этой переменной нет вовсе, графическая сессия определяется
    через `platform.system()`, и тест сравнивал доклад с признаком, которого на этой
    платформе не существует. Проверялось окружение, а не поведение.

    Доступность — **конъюнкция**: нужна графическая сессия и нужен пакет. Объявляются
    оба условия, и проверяются три случая, потому что двух мало: «есть сессия, нет
    пакета» и «нет сессии, есть пакет» — разные причины недоступности, и доклад,
    путающий их, отправит оператора чинить не то.
    """
    from harness.cli import main
    from harness.machine import Session

    cases = ((Session.X11, True, True),      # сессия и пакет — доступен
             (Session.X11, False, False),    # пакета нет
             (Session.NONE, True, False))    # сессии нет
    for session, has_mss, expect in cases:
        as_session(session)
        (with_package("mss") if has_mss else without_package("mss"))
        assert main(["backends"]) == 0
        out = capsys.readouterr().out
        assert "synthetic" in out and "replay" in out
        lines = [ln for ln in out.splitlines() if "screen_mss" in ln]
        assert lines, "механизм захвата экрана вообще не упомянут в докладе"
        for line in lines:
            assert ("[есть]" in line) == expect, (
                f"объявлено: сессия {session}, mss {'есть' if has_mss else 'нет'} — "
                f"а доклад говорит иначе: {line!r}")


def test_cli_profile_shows_both_hashes(capsys: pytest.CaptureFixture[str]) -> None:
    from harness.cli import main

    assert main(["profile"]) == 0
    out = capsys.readouterr().out
    assert "profile_hash" in out and "structure_hash" in out
    assert "Блок А" in out and "Блок Б" in out


def test_cli_gen_verify_selfworld(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from harness.cli import main

    root = tmp_path / "s"
    assert main(["gen-corpus", str(root), "--seed", "5"]) == 0
    capsys.readouterr()
    assert main(["verify", str(root)]) == 0
    assert '"ok": true' in capsys.readouterr().out
    assert main(["journal", str(root)]) == 0
    capsys.readouterr()
    assert main(["selfworld", str(root)]) == 0
    out = capsys.readouterr().out
    assert "IoU по пикселям" in out
    assert main(["replay", str(root), "--at", "3"]) == 0


def test_cli_verify_fails_on_broken_session(tmp_path: Path,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    from harness.cli import main

    root = tmp_path / "s"
    with Recorder(root, profile=MILESTONE_0, source="test", synthetic=True) as rec:
        a = np.zeros((16, 16), dtype=np.uint8)
        rec.record_frame(a, t_world=0)
        rec.record_frame(a, t_world=5)      # разрыв
    assert main(["verify", str(root)]) == 1
    assert '"ok": false' in capsys.readouterr().out


@pytest.mark.environment("сессия подменяется на «ничего»")
def test_cli_record_without_display_fails_loudly(tmp_path: Path, as_session) -> None:
    """Без графической сессии запись отказывает, а не пишет чёрные кадры.

    Сессия **подменяется**, а не предполагается отсутствующей. Прежняя редакция удаляла
    `DISPLAY` и ждала кода 2; на машине оператора дисплей есть, удаление переменных на
    Windows не значит ничего, запись прошла и вернула 0. Тест падал, хотя код был прав.
    """
    from harness.cli import main
    from harness.machine import Session

    as_session(Session.NONE)
    assert main(["record", str(tmp_path / "s"), "--frames", "1"]) == 2
    assert not (tmp_path / "s" / "frames").exists()


@pytest.mark.environment("сессия подменяется на Wayland")
def test_cli_record_under_wayland_fails_loudly(tmp_path: Path, as_session) -> None:
    """Под Wayland запись тоже отказывает: mss отдала бы чёрный кадр или XWayland.

    Это второй случай отказа, и он важнее первого: сессия есть, захват формально
    возможен, и молчаливая запись была бы неотличима по формату от настоящей.
    """
    from harness.cli import main
    from harness.machine import Session

    as_session(Session.WAYLAND)
    assert main(["record", str(tmp_path / "s"), "--frames", "1"]) == 2
    assert not (tmp_path / "s" / "frames").exists()


# --- хранилище -------------------------------------------------------------


def test_shard_rollover(tmp_path: Path) -> None:
    """Шарды переключаются по размеру, индекс остаётся сплошным."""
    # Блок случайного шума 40×40 не сжимается, то есть даёт около 1.6 КиБ на диске.
    # Порог шарда 5 КиБ — значит переключений должно быть несколько.
    rng = np.random.default_rng(0)
    with BlobStore(tmp_path / "b", mode="a", shard_bytes=5_000) as st:
        for _ in range(12):
            st.append(rng.integers(0, 255, (40, 40), dtype=np.uint8))
    shards = sorted((tmp_path / "b").glob("shard-*.bin"))
    assert len(shards) >= 3, f"шарды не переключились: {len(shards)}"
    with BlobStore(tmp_path / "b", mode="r") as st:
        assert len(st) == 12
        assert {r.shard for r in (st.ref(i) for i in range(12))} == {
            int(p.stem.split("-")[1]) for p in shards}


def test_reopen_appends_not_overwrites(tmp_path: Path) -> None:
    with BlobStore(tmp_path / "b", mode="a") as st:
        st.append(np.zeros((4, 4), dtype=np.uint8))
    with BlobStore(tmp_path / "b", mode="a") as st:
        st.append(np.ones((4, 4), dtype=np.uint8))
    with BlobStore(tmp_path / "b", mode="r") as st:
        assert len(st) == 2
        assert st.read(0).sum() == 0 and st.read(1).sum() == 16


def test_read_only_store_refuses_write(tmp_path: Path) -> None:
    with BlobStore(tmp_path / "b", mode="a") as st:
        st.append(np.zeros((4, 4), dtype=np.uint8))
    with BlobStore(tmp_path / "b", mode="r") as st:
        with pytest.raises(StoreError, match="только на чтение"):
            st.append(np.zeros((4, 4), dtype=np.uint8))


def test_broken_index_is_detected(tmp_path: Path) -> None:
    with BlobStore(tmp_path / "b", mode="a") as st:
        for i in range(3):
            st.append(np.full((4, 4), i, dtype=np.uint8))
    index = tmp_path / "b" / "index.jsonl"
    lines = index.read_text(encoding="utf-8").splitlines()
    del lines[1]
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(StoreError, match="blob_id"):
        BlobStore(tmp_path / "b", mode="r")


def test_delta_block_cannot_be_read_raw(tmp_path: Path) -> None:
    """Разность нельзя прочитать как данные: иначе вернутся разности молча."""
    with FrameStore(tmp_path / "fs", mode="a", keyframe_interval=10) as fs:
        fs.append_frame(np.zeros((8, 8), dtype=np.uint8))
        fs.append_frame(np.ones((8, 8), dtype=np.uint8))
        ref = fs.ref(1)
    assert ref.enc == "delta"
    with FrameStore(tmp_path / "fs", mode="r") as fs:
        assert np.array_equal(fs.read(1), np.ones((8, 8), dtype=np.uint8))
        with pytest.raises(StoreError, match="delta"):
            BlobStore.read(fs, 1)


def test_audio_refuses_mono(tmp_path: Path) -> None:
    from harness.core.blobstore import AudioStore

    with AudioStore(tmp_path / "a", mode="a") as st:
        with pytest.raises(StoreError, match="Моно"):
            st.append_block(np.zeros((100, 1), dtype=np.int16))
        st.append_block(np.zeros((100, 2), dtype=np.int16))


def test_frame_shape_change_forces_keyframe(tmp_path: Path) -> None:
    """Смена формы кадра не должна дать разность от кадра другой формы."""
    with FrameStore(tmp_path / "fs", mode="a", keyframe_interval=100) as fs:
        fs.append_frame(np.zeros((8, 8), dtype=np.uint8))
        ref = fs.append_frame(np.zeros((16, 16), dtype=np.uint8))
    assert ref.enc == "raw"


# --- сессия ----------------------------------------------------------------


def test_session_refuses_non_session_dir(tmp_path: Path) -> None:
    (tmp_path / "junk").mkdir()
    with pytest.raises(SessionError, match="не похоже на сессию"):
        Session.open(tmp_path / "junk")


def test_recorder_refuses_nonempty_dir(tmp_path: Path) -> None:
    root = tmp_path / "s"
    root.mkdir()
    (root / "чужое.txt").write_text("данные", encoding="utf-8")
    with pytest.raises(SessionError, match="не пуст"):
        Recorder(root, profile=MILESTONE_0, source="test", synthetic=True)


def test_seek_out_of_range(session: Session) -> None:
    with pytest.raises(SessionError, match="за пределы"):
        session.seek(len(session) + 10)


# --- профиль и символы -----------------------------------------------------


def test_profile_rejects_unhashable_values() -> None:
    with pytest.raises(ProfileError, match="не сериализуется"):
        Profile("плохой", parameters={"x": [1, 2, 3]})
    with pytest.raises(ProfileError, match="nan/inf"):
        Profile("плохой", parameters={"x": float("nan")})


def test_profile_rejects_unknown_key_change() -> None:
    with pytest.raises(ProfileError, match="нет таких параметров"):
        MILESTONE_0.with_parameters(неизвестная_ручка=1)


def test_profile_load_detects_edited_hash(tmp_path: Path) -> None:
    import json

    path = tmp_path / "p.json"
    MILESTONE_0.save(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    d["parameters"]["capture_fps"] = 99.0        # значение правили, хеш прежний
    path.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ProfileError, match="не совпадает с пересчитанным"):
        Profile.load(path)


def test_profile_diff_marks_structural() -> None:
    other = MILESTONE_0.with_structural(frame_format="rgb8")
    diff = MILESTONE_0.diff(other)
    assert len(diff) == 1 and diff[0]["structural"] is True
    assert MILESTONE_0.forks_journal(other)


def test_symbol_normalization_rules() -> None:
    assert normalize("  Здоровье   ") == "Здоровье"
    assert normalize("Здоровье\n\t20") == "Здоровье 20"
    sym = Symbolizer("s0")
    # регистр значим: «ОПЫТ» и «Опыт» на экране — разные надписи
    assert sym.symbolize("Опыт") != sym.symbolize("ОПЫТ")
    with pytest.raises(SymbolError, match="пустая надпись"):
        sym.symbolize("   ")


def test_symbol_collision_is_reported(tmp_path: Path) -> None:
    """Коллизия символов не проглатывается: агент увидел бы две надписи как одну."""
    from harness.debug.channel import DebugChannel, DebugChannelError

    with DebugChannel(tmp_path / "d", mode="a") as dbg:
        dbg.write_symbol("SYM_AAAA", "Здоровье", salt_id="s0")
        dbg.write_symbol("SYM_AAAA", "Голод", salt_id="s0")
    with pytest.raises(DebugChannelError, match="коллизии символов"):
        DebugChannel(tmp_path / "d", mode="r").symbol_table()


@pytest.mark.environment("подменяются сессия и отсутствие пакетов dxcam, mss")
def test_unavailable_backends_fail_loudly(as_session, without_package) -> None:
    """Недоступный backend падает с внятным текстом, а не отдаёт чёрные кадры.

    Это главное правило проекта в действии: функция, возвращающая правдоподобное
    значение вместо реального, ломает эксперимент незаметно.

    **Недоступность объявляется, а не берётся из окружения.** Прежняя редакция ждала
    отказа от `WindowsScreenCapture`, полагаясь на то, что `dxcam` не установлен; на
    машине оператора он установлен и работает (TASK-08), и тест падал, проверив
    окружение вместо поведения. Теперь отсутствие пакета объявлено фикстурой, и
    проверяется именно текст отказа — что он называет настоящую причину.
    """
    from harness.capture.base import BackendUnavailable
    from harness.capture.screen import (LoopbackAudio, MacScreenCapture, ScreenCapture,
                                        WindowsScreenCapture)
    from harness.inject.base import InjectionUnavailable
    from harness.inject.stop import HotkeyListener, StopSwitch
    from harness.inject.uinput_device import WindowsSendInput
    from harness.machine import Session

    as_session(Session.NONE)
    without_package("dxcam", "mss")

    # ScreenCaptureKit по-прежнему не написан и говорит это прямо.
    with pytest.raises(BackendUnavailable, match="не реализован"):
        MacScreenCapture().start()

    # А dxcam с TASK-08 **написан**: проверка на машине оператора показала, что
    # `dxcam.create()` и `grab()` работают с первой попытки. Здесь он отказывает по
    # другой причине — нет пакета, — и текст обязан называть именно её: «не
    # реализован» отправило бы оператора ждать того, что уже готово.
    with pytest.raises(BackendUnavailable, match="dxcam"):
        WindowsScreenCapture().start()

    cap = ScreenCapture()
    with pytest.raises(BackendUnavailable):
        cap.start()
    with pytest.raises(BackendUnavailable, match="не запущен"):
        cap.read()

    with pytest.raises(ValueError, match="моно ломает пеленг"):
        LoopbackAudio(channels=1)

    with pytest.raises(BackendUnavailable, match="не запущен"):
        LoopbackAudio().read()

    dev = WindowsSendInput()
    with pytest.raises(InjectionUnavailable, match="не реализована"):
        dev.key_down("OUT_0A11")

    with pytest.raises(NotImplementedError, match="не реализован"):
        HotkeyListener(StopSwitch()).start()


def test_uinput_without_device_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    from harness.inject.base import InjectionUnavailable
    from harness.inject.uinput_device import UinputDevice

    monkeypatch.setattr("os.path.exists", lambda p: False if p == "/dev/uinput" else True)
    dev = UinputDevice(resolve=lambda o: 17)
    with pytest.raises(InjectionUnavailable, match="modprobe uinput"):
        dev.open()
    with pytest.raises(InjectionUnavailable, match="не открыто"):
        dev.key_down("OUT_0A11")


def test_keymap_resolver_is_one_way() -> None:
    """Резолвер, который получает устройство, не даёт узнать имя клавиши."""
    from harness.debug.keymap import Keymap

    km = Keymap.linux_evdev()
    out = km.output_for_key("W")
    resolve = km.resolver()
    assert resolve(out) == 17
    for attr in ("keys", "table", "name_for_output", "items"):
        assert not hasattr(resolve, attr)
    with pytest.raises(KeyError, match="не отображается"):
        resolve("OUT_FFFF")
