"""Холодный архив следа и органы машины. TASK-32, направление B.

Три утверждения, и каждое — про то, чего в проекте быть не должно:

1. **Удаления следа не существует ни в какой ветке** (инвариант 14). Вывоз освобождает
   место, ничего не теряя; понижение выбрасывает подробности и к нулевому уровню
   неприменимо; удаления нет, и попытка упирается в объяснение, а не в `AttributeError`.
2. **Вывоз обратим, и это проверяется байтами**, а не наличием файла: привезённый сегмент
   сверяется с адресом по содержимому.
3. **Орган, который нечем прочитать, не возвращает ноль.** «Видеопамяти свободно 0» и
   «видеопамять читать нечем» ведут к противоположным решениям.

Единица независимости — **утверждение об устройстве механизма**. Окружения не требует:
видеопамять читается через подставной запускатель, а не через настоящий nvidia-smi.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core.archive import Archive, ArchiveError, oldest_first
from harness.core.levels import Level, Placement, content_id
from harness.core.organs import Organ, disk, ram, read_all, vram


def _placement(payload: bytes, seg: str = "seg-0001", level: Level = Level.TRACE
               ) -> Placement:
    return Placement(seg, level, content_id(payload), len(payload), level)


# --- удаления не существует ---------------------------------------------------


def test_deletion_of_the_trace_does_not_exist(tmp_path: Path) -> None:
    """Инвариант 14: попытка удалить упирается в объяснение, а не в отсутствие имени."""
    arc = Archive(tmp_path / "cold")
    with pytest.raises(ArchiveError, match="удаления следа не существует"):
        arc.forget("seg-0001")
    # И ни одного другого способа выбросить содержимое в объекте нет.
    forbidden = [name for name in dir(arc)
                 if any(word in name for word in ("delete", "remove", "purge", "drop"))]
    assert forbidden == [], f"нашлись подозрительные операции: {forbidden}"


def test_archiving_needs_a_reason(tmp_path: Path) -> None:
    """Вывоз без причины — потерянный сегмент: никто не привезёт его обратно."""
    arc = Archive(tmp_path / "cold")
    payload = b"trace-payload"
    with pytest.raises(ArchiveError, match="без причины"):
        arc.put(_placement(payload), payload, reason="")


# --- вывоз обратим, и это проверяется байтами --------------------------------


def test_put_and_get_return_the_same_bytes(tmp_path: Path) -> None:
    """Вывоз ничего не теряет — тем и отличается от понижения."""
    arc = Archive(tmp_path / "cold")
    payload = b"a" * 4096
    ref = arc.put(_placement(payload), payload, reason="старый след, место кончается")
    assert ref.nbytes == len(payload)
    assert arc.has("seg-0001") and arc.bytes_cold() == len(payload)
    assert arc.get("seg-0001") == payload


def test_content_is_checked_on_the_way_in_and_out(tmp_path: Path) -> None:
    """Хранилище, тихо накопившее не то, обнаружится через месяцы. Значит — не тихо."""
    arc = Archive(tmp_path / "cold")
    payload = b"payload"
    wrong = Placement("seg-0002", Level.TRACE, content_id("другое".encode("utf-8")), 7, Level.TRACE)
    with pytest.raises(ArchiveError, match="не сходится с адресом"):
        arc.put(wrong, payload, reason="проверка")

    arc.put(_placement(payload), payload, reason="проверка")
    # Порча в холодном хранилище: файл подменили. Возврат обязан отказать, а не отдать.
    path = arc.root / arc.refs["seg-0001"].where
    path.write_bytes("подмена".encode("utf-8"))
    with pytest.raises(ArchiveError, match="не тот"):
        arc.get("seg-0001")


def test_missing_cold_storage_is_not_a_lost_segment(tmp_path: Path) -> None:
    """Диск вынули — сегмент существует и не потерян, и отказ говорит именно это."""
    cold = tmp_path / "cold"
    arc = Archive(cold)
    payload = b"x" * 128
    arc.put(_placement(payload), payload, reason="вывоз")
    # Индекс остаётся в памяти горячей стороны: она обязана помнить, что сегмент есть.
    for child in sorted(cold.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    cold.rmdir()
    assert arc.has("seg-0001"), "горячая сторона обязана помнить про вывезенное"
    with pytest.raises(ArchiveError, match="недоступно"):
        arc.get("seg-0001")


def test_report_says_how_much_is_cold_and_why(tmp_path: Path) -> None:
    """Отчёт о вывезенном — числами и причинами, а не «архив в порядке»."""
    arc = Archive(tmp_path / "cold")
    for i, reason in enumerate(("место кончается", "старый след")):
        payload = bytes([i]) * 256
        arc.put(_placement(payload, seg=f"seg-{i:04d}"), payload, reason=reason)
    got = arc.report()
    assert got["segments"] == 2 and got["bytes"] == 512
    assert set(got["reasons"]) == {"место кончается", "старый след"}
    assert got["unit"] == "сегмент"


def test_oldest_first_keeps_the_freshest_hot() -> None:
    """Свежий след читают чаще всего: вывозить его — платить временем за место."""
    ps = [_placement(bytes([i]), seg=f"seg-{i:04d}") for i in range(5)]
    assert [p.segment_id for p in oldest_first(ps, keep_hot=2)] == \
        ["seg-0000", "seg-0001", "seg-0002"]
    assert list(oldest_first(ps, keep_hot=99)) == []


# --- органы: ноль запрещён ----------------------------------------------------


def test_an_unreadable_organ_never_returns_zero() -> None:
    """«Свободно 0» и «читать нечем» ведут к противоположным решениям."""
    with pytest.raises(ValueError, match="причина не названа"):
        Organ("выдуманный", "МиБ", None, None, "источник")

    unknown = Organ("выдуманный", "МиБ", None, None, "источник", why="нечем прочитать")
    assert unknown.known is False and unknown.free_share is None
    assert unknown.as_dict()["free"] is None, "ноль вместо None — молчаливая заглушка"


def test_vram_says_which_reason_it_is() -> None:
    """Нет видеокарты и нет утилиты — разные причины, и обе не ноль."""
    def missing(*a, **kw):
        raise FileNotFoundError("nvidia-smi")

    got = vram(run=missing)
    assert got.known is False and got.free is None
    assert "nvidia-smi" in got.why

    class Done:
        returncode = 0
        stdout = "8192, 6144\n"

    ok = vram(run=lambda *a, **kw: Done())
    assert ok.known and ok.total == 8192 and ok.free == 6144


def test_ram_and_disk_are_concrete_numbers_with_units(tmp_path: Path) -> None:
    """Не «загрузка 60 %», а сколько именно и в чём."""
    r = ram()
    d = disk(tmp_path)
    for organ in (r, d):
        assert organ.unit == "МиБ"
        if organ.known:
            assert organ.total and organ.total > 0 and organ.free is not None
            assert 0.0 <= (organ.free_share or 0.0) <= 1.0
        else:
            assert organ.why, "непрочитанный орган обязан назвать причину"
    assert r.source.endswith("meminfo") or r.why


def test_read_all_counts_the_unreadable_separately(tmp_path: Path) -> None:
    """Отчёт с тремя `None` не должен выглядеть как отчёт с тремя нулями."""
    got = read_all(journal_path=tmp_path, run=lambda *a, **kw: (_ for _ in ()).throw(
        FileNotFoundError("nvidia-smi")))
    assert got["known"] + got["unknown"] == len(got["organs"])
    assert got["unknown"] >= 1 and got["unknown_why"], (
        "непрочитанные органы обязаны быть названы вместе с причиной")
    assert got["unit"] == "орган"
