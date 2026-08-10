"""Сервер панели: запись из браузера. `TASK-19`.

Проверяется без дисплея — тем же поддельным экраном, которым проверяется цикл записи.
Иначе единственное, что можно было бы сказать про сервер, — «у меня открылось».

Четыре ограничения из докстринга `panelserver` проверяются здесь по одному, потому что это
не оформление, а то, чем эта программа отличается от веб-сервиса:

1. слушает только `127.0.0.1`;
2. произвольных путей не принимает — только виды из плана;
3. запуск записи требует ключа страницы;
4. «Прервать» закрывает сессию тем же путём, что `Ctrl+C`.

Пятое — то, ради чего задача заведена: после записи фикстура пересобирается **сама**.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from harness.capture.base import UNCHANGED, Frame
from harness.core.journal import Kind
from harness.panelserver import (HOST, PanelServer, Runner, State, _usable, plan_cards)
from harness.recording import RecordRefused
from harness.session import Session


class FakeScreen:
    """Экран без дисплея. Тот же приём, что в `test_record_progress`."""

    name = "тест-экран"

    def __init__(self, *, w: int = 64, h: int = 48, still_ratio: int = 0,
                 delay_s: float = 0.0) -> None:
        self.w, self.h = w, h
        self.still_ratio = still_ratio
        self.delay_s = delay_s
        self.reads = 0
        self.stopped = False

    def read(self):
        self.reads += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.still_ratio and self.reads % (self.still_ratio + 1) != 0:
            return UNCHANGED
        return Frame(np.full((self.h, self.w), (self.reads * 13) % 251, dtype=np.uint8),
                     self.reads, 0)

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture()
def fake_screen(monkeypatch):
    """Подменить **выбор механизма** — ту же точку, через которую идёт доктор.

    Ниже подменять нечего: дисплея нет, и без подмены сервер записи проверить невозможно, а
    проверять надо именно его.
    """
    def force(screen: FakeScreen) -> FakeScreen:
        import harness.capture.select as select_mod

        monkeypatch.setattr(
            select_mod, "open_screen",
            lambda m, **kw: select_mod.Choice(chosen=select_mod.MSS,
                                             considered=(select_mod.MSS,), source=screen))
        return screen

    return force


@pytest.fixture()
def server(tmp_path: Path):
    """Поднятый сервер на свободном порту. Закрывается по выходу из теста.

    Каталог панели — **копия** во временном месте, а не тот, что в репозитории. Первая
    редакция этой обвязки отдала серверу настоящий `panel/`, и проверка пересборки
    перезаписала фикстуру, лежащую в репозитории: проверка испортила артефакт, который
    проверяла. Полный прогон это и поймал.
    """
    import shutil

    real = Path(__file__).resolve().parent.parent / "panel"
    panel = tmp_path / "panel"
    panel.mkdir()
    for name in ("index.html", "fixture.json"):
        if (real / name).exists():
            shutil.copy2(real / name, panel / name)
    srv = PanelServer(port=0, live_root=tmp_path / "live", panel_dir=panel,
                      repo=Path(__file__).resolve().parent.parent)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def _get(srv: PanelServer, path: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(srv.url.rstrip("/") + path, timeout=10) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _post(srv: PanelServer, path: str, body: dict, *, token: str | None = "real"
          ) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(srv.url.rstrip("/") + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token is not None:
        req.add_header("X-Harness-Token", srv.token if token == "real" else token)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _wait(srv: PanelServer, *, running: bool | None = None, limit: float = 30.0,
          stage: str | None = None) -> dict:
    """Дождаться нужного состояния. Без сна в основном потоке проверки.

    `stage` нужен потому, что «запись кончилась» и «всё кончилось» — разные моменты: между
    ними идёт пересборка фикстуры, и первая редакция проверки успевала посмотреть на
    состояние до неё.
    """
    deadline = time.monotonic() + limit
    st: dict = {}
    while time.monotonic() < deadline:
        st = srv.state.snapshot()
        if stage is not None:
            if st["stage"] == stage:
                return st
        elif st["running"] is running:
            return st
        time.sleep(0.02)
    raise AssertionError(
        f"состояние не наступило: running={running}, stage={stage}, было {st}")


# --- 1. адрес и что отдаётся ------------------------------------------------


def test_listens_only_on_loopback(server: PanelServer) -> None:
    """Слушает `127.0.0.1`, и адрес не параметр.

    «Дай послушать наружу» — это другая программа с другими требованиями: запуск записи
    экрана по сети означал бы, что экран оператора пишет кто угодно из сети.
    """
    assert HOST == "127.0.0.1"
    assert server.server_address[0] == "127.0.0.1"
    import ast

    from harness import panelserver

    src = Path(panelserver.__file__).read_text(encoding="utf-8")
    # Докстринг из проверки вычитается: он **объясняет**, почему адреса всех интерфейсов
    # здесь нет, и первая редакция проверки нашла эту фразу и объявила дефектом собственное
    # объяснение.
    doc = ast.get_docstring(ast.parse(src)) or ""
    code = src.replace(doc, "")
    assert "0.0.0.0" not in code, "в сервере появился адрес всех интерфейсов"
    # Адрес берётся из константы, а не из аргумента: параметра «host» нет вовсе.
    assert "def serve(" in code
    signature = code.split("def serve(")[1].split(")")[0]
    assert "host" not in signature, f"адрес стал параметром: {signature}"
    assert "(HOST, int(port))" in code, "адрес прослушивания не из константы"


def test_serves_only_the_listed_files(server: PanelServer) -> None:
    """Отдаётся перечисленное, а не каталог. Иначе раздавались бы и записи."""
    code, _ = _get(server, "/api/plan")
    assert code == 200
    for bad in ("/../src/harness/cli.py", "/fixture.json.bak", "/panel.mjs",
                "/../../etc/passwd"):
        code, _ = _get(server, bad)
        assert code == 404, f"{bad} отдался"


def test_the_page_carries_the_token_and_the_api_requires_it(server: PanelServer) -> None:
    """Ключ вставлен в страницу и обязателен для запуска записи.

    Защита не от сети — сервер в сеть не смотрит, — а от **другой вкладки** того же
    браузера: без ключа любая открытая страница смогла бы начать запись экрана.
    """
    with urllib.request.urlopen(server.url, timeout=10) as r:
        html = r.read().decode("utf-8")
    assert f'content="{server.token}"' in html
    assert 'name="harness-token"' in html

    code, got = _post(server, "/api/record", {"kind": "stillness"}, token=None)
    assert code == 403 and got["code"] == "no_token"
    # Ключ из букв ASCII: заголовки HTTP кодируются latin-1, и кириллица в них не уедет
    # даже до сервера — первая редакция проверки падала на клиенте, а не на отказе.
    code, got = _post(server, "/api/record", {"kind": "stillness"}, token="wrong-token")
    assert code == 403 and got["code"] == "no_token"


# --- 2. только виды из плана ------------------------------------------------


def test_the_protocol_has_no_path_at_all() -> None:
    """Путь записи вычисляет сервер по виду. В протоколе пути нет.

    Это сильнее проверки пути: подделать нельзя то, чего не передают. Отсюда и проверка —
    по коду сервера, а не по поведению на одном хитром значении.
    """
    from harness import panelserver

    src = Path(panelserver.__file__).read_text(encoding="utf-8")
    api = src[src.index("if path == \"/api/record\""):src.index("if path == \"/api/stop\"")]
    assert "body.get(\"kind\"" in api
    assert "body.get(\"path\"" not in api and "body[\"path\"]" not in api


def test_unknown_kind_is_refused(server: PanelServer) -> None:
    """Вид, которого нет в плане, — отказ с перечислением того, что есть."""
    code, got = _post(server, "/api/record", {"kind": "чужое"})
    assert code == 409 and got["code"] == "unknown_kind"
    assert "stillness" in got["hint"]
    assert not server.state.snapshot()["running"]


def test_plan_cards_come_from_the_plan(tmp_path: Path) -> None:
    """Карточки — из `corpus.live`, а не из второго списка в панели."""
    from harness.corpus.live import KINDS, SETS

    cards = plan_cards(tmp_path)
    assert [c["kind"] for c in cards] == list(SETS["minimal"]["kinds"])
    for c in cards:
        k = KINDS[c["kind"]]
        assert c["title"] == k["title"] and c["how"] == k["how"]
        assert c["duration"] == k["duration"] and c["seconds"] > 0
        assert c["done"] == {"recorded": False}, "непрочитанная запись объявлена сделанной"
    still = [c for c in cards if c["kind"] == "stillness"][0]
    assert still["progress_to_file"], "на неподвижности ход обязан идти в файл"


# --- 3. запись, ход, прерывание ----------------------------------------------


def test_recording_from_the_panel_writes_a_real_session(server: PanelServer,
                                                        fake_screen) -> None:
    """Кнопка «Записать» делает настоящую запись, и путь подставляется сам."""
    fake_screen(FakeScreen(still_ratio=2))
    code, got = _post(server, "/api/record", {"kind": "scroll", "seconds": 0.6})
    assert code == 200 and got["started"]
    st = _wait(server, running=False)

    result = st["result"]
    assert result, f"итога нет: {st}"
    assert result["written"] > 0 and result["unchanged"] > 0
    path = Path(result["path"])
    with Session.open(path) as s:
        assert s.verify()["ok"]
        assert s.meta.synthetic is False
        assert "из панели" in (s.meta.note or "")
    # Разбивка и приговор по бюджету — те же числа, что печатает команда.
    assert "capture_ms" in result["stages"] and "fits" in result["verdict"]


def test_progress_is_the_same_numbers_as_the_terminal(server: PanelServer,
                                                      fake_screen) -> None:
    """Ход в панели — те же наблюдения, что печатаются в терминал, а не второй счётчик.

    Второй счётчик разошёлся бы с первым, и «в панели одно, в консоли другое» стало бы
    вопросом без ответа.
    """
    fake_screen(FakeScreen(still_ratio=1, delay_s=0.004))
    _post(server, "/api/record", {"kind": "scroll", "seconds": 2.2})
    seen = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        code, st = _get(server, "/api/progress")
        assert code == 200
        if st["sample"]:
            seen.append(st["sample"])
        if not st["running"] and st["result"]:
            break
        time.sleep(0.05)
    _wait(server, running=False)
    assert seen, "ход не появился ни разу"
    last = seen[-1]
    # Те же величины, что в строке терминала, включая долю изменившихся.
    for key in ("elapsed_s", "total_s", "share", "written", "unchanged", "mib",
                "change_share", "line"):
        assert key in last, key
    assert "из" in last["line"] and "кадров" in last["line"]
    assert last["turns"] == last["written"] + last["unchanged"]


def test_stop_closes_the_session_the_same_way_as_ctrl_c(server: PanelServer,
                                                        fake_screen) -> None:
    """«Прервать» ведёт в ту же отметку, что `Ctrl+C`, и запись остаётся годной.

    Два места нажатия — одно событие: человек решил остановиться. Разделять их значило бы
    завести два класса одного события и потом сравнивать записи, «сделанные по-разному».
    """
    fake_screen(FakeScreen(delay_s=0.01))
    _post(server, "/api/record", {"kind": "scroll", "seconds": 60})
    _wait(server, running=True)
    # Дождаться первых кадров, иначе прерывать нечего.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if (server.state.snapshot()["sample"] or {}).get("written", 0) > 0:
            break
        time.sleep(0.02)
    code, got = _post(server, "/api/stop", {})
    assert code == 200 and got["stopped"]
    st = _wait(server, running=False)

    result = st["result"]
    assert result["interrupted"] and result["stop_reason"] == "stopped"
    with Session.open(Path(result["path"])) as s:
        mark = s.interrupted
        assert mark is not None, "отметки о прерывании нет"
        assert "Прервать" in mark["reason"], mark
        assert s.verify()["ok"], "прерванная из панели запись обязана быть целой"
        kinds = {e.kind for e in s.journal
                 if e.event.get("code") == "record_interrupted"}
        assert kinds == {Kind.INTERVENTION}


def test_two_recordings_at_once_are_refused(server: PanelServer, fake_screen) -> None:
    """Одна запись за раз: два потока кадров с одного экрана — это не две записи."""
    fake_screen(FakeScreen(delay_s=0.01))
    _post(server, "/api/record", {"kind": "scroll", "seconds": 30})
    _wait(server, running=True)
    code, got = _post(server, "/api/record", {"kind": "video", "seconds": 1})
    assert code == 409 and got["code"] == "busy_running"
    assert "Прервать" in got["hint"]
    _post(server, "/api/stop", {})
    _wait(server, running=False)


def test_stop_without_recording_says_so(server: PanelServer) -> None:
    """«Прервать» на пустом месте не притворяется, что что-то остановил."""
    code, got = _post(server, "/api/stop", {})
    assert code == 200 and got["stopped"] is False and "не идёт" in got["why"]


# --- 4. годность записи, а не только целостность -----------------------------


def test_stillness_without_static_marks_is_declared_unusable() -> None:
    """Запись неподвижности без отметок «без изменений» — НЕ годна, и сказано почему.

    Она цела: `verify` её пропускает. И она бесполезна — мерила мигающий курсор, а не фон
    экрана. Первый замер оператора вышел ровно таким, и панель обязана сказать это красным.
    """
    bad = _usable("stillness", written=300, unchanged=0)
    assert not bad["ok"] and "мигало" in bad["why"]

    good = _usable("stillness", written=14, unchanged=286)
    assert good["ok"] and "фон экрана записан" in good["why"]

    # На записях про изменения правило обратное, и это разные вопросы к разным записям.
    assert not _usable("scroll", written=0, unchanged=300)["ok"]
    assert _usable("scroll", written=250, unchanged=50)["ok"]


def test_done_state_is_read_from_the_recording(server: PanelServer, fake_screen,
                                               tmp_path: Path) -> None:
    """Список сделанного читается из записей, а не помнится в отдельном файле.

    Второй источник истины о том, что записано, разошёлся бы с журналом — и разошёлся бы
    молча.
    """
    fake_screen(FakeScreen(still_ratio=3))
    before = plan_cards(tmp_path / "live")
    assert all(not c["done"]["recorded"] for c in before)

    _post(server, "/api/record", {"kind": "video", "seconds": 0.5})
    _wait(server, running=False)

    after = plan_cards(tmp_path / "live")
    video = [c for c in after if c["kind"] == "video"][0]
    assert video["done"]["recorded"] and video["done"]["turns"] > 0
    assert "usable" in video["done"]
    # Путь для следующей записи того же вида предлагается свободный.
    assert video["free_path"] != video["path"]


# --- 5. то, ради чего всё: фикстура пересобирается сама ----------------------


def test_the_fixture_is_rebuilt_after_recording(server: PanelServer,
                                                fake_screen) -> None:
    """После записи фикстура пересобирается сама, и об этом сказано.

    Между записью и панелью стоял ручной запуск `tools/make_panel_fixture.py`, о котором
    оператор не обязан помнить. Пока он о нём не помнил, на первом экране был синтетический
    шум вместо его рабочего стола.
    """
    fixture = server.panel_dir / "fixture.json"
    was = json.loads(fixture.read_text(encoding="utf-8"))
    fake_screen(FakeScreen(w=320, h=240, still_ratio=2))
    _post(server, "/api/record", {"kind": "windows", "seconds": 0.8})
    st = _wait(server, stage="done", limit=300)

    fx = st["fixture"]
    assert fx, "о пересборке фикстуры не сказано ничего"
    assert fx["rebuilt"], fx["why"]
    now = json.loads(fixture.read_text(encoding="utf-8"))
    # Зеркало и путь сессии теперь от **этой** записи, а не от синтетической.
    assert now["about"]["session"] != was["about"]["session"]
    assert str(Path(st["result"]["path"])) in now["about"]["session"]


def test_a_failed_rebuild_is_not_silent(tmp_path: Path) -> None:
    """Пересборка не молчит ни в одном исходе.

    Молча оставить прежнюю фикстуру значило бы показывать оператору чужую запись под видом
    его собственной — то есть ровно ту ложь, ради устранения которой пересборка и заведена.
    """
    state = State()
    runner = Runner(state, root=tmp_path, repo=tmp_path / "нет-такого-репозитория")
    state.path = tmp_path / "нет-записи"
    runner._rebuild_fixture()
    assert state.fixture["rebuilt"] is False
    assert "пересобирать нечего" in state.fixture["why"]

    # Запись есть, а сборщика нет: сказано и это, вместе с последствием.
    (tmp_path / "запись").mkdir()
    (tmp_path / "запись" / "session.json").write_text("{}", encoding="utf-8")
    state.path = tmp_path / "запись"
    runner._rebuild_fixture()
    assert state.fixture["rebuilt"] is False
    assert "не ваша запись" in state.fixture["why"]


def test_refusal_reaches_the_panel(server: PanelServer, monkeypatch) -> None:
    """Отказ записи виден в панели, а не только в терминале сервера."""
    import harness.capture.select as select_mod

    monkeypatch.setattr(
        select_mod, "open_screen",
        lambda m, **kw: select_mod.Choice(chosen=None, considered=(), source=None))
    _post(server, "/api/record", {"kind": "scroll", "seconds": 1})
    st = _wait(server, running=False)
    assert st["refusal"]["code"] == "no_capture"
    assert "harness doctor" in st["refusal"]["hint"]


def test_the_panel_says_so_when_it_cannot_record() -> None:
    """Панель, открытая файлом, говорит, что записывать нечем, а не рисует кнопку.

    Кнопка, которая ничего не делает, — обещание поведения, которого нет, и это та же
    молчаливая заглушка, что запрещена в коде.
    """
    html = (Path(__file__).resolve().parent.parent / "panel" / "index.html").read_text(
        encoding="utf-8")
    assert "function noServer" in html
    assert "записывать отсюда нечем" in html
    assert "harness panel" in html


def test_the_command_exists_and_needs_no_arguments() -> None:
    """`harness panel` — одна команда на всё: сервер и браузер."""
    from harness.cli import build_parser

    args = build_parser().parse_args(["panel"])
    assert args.fn.__name__ == "cmd_panel"
    assert args.port == 8765 and args.no_browser is False


def test_the_rebuild_writes_where_the_server_serves(tmp_path: Path, server: PanelServer,
                                                    fake_screen) -> None:
    """Фикстура пересобирается **в тот каталог, который отдаёт этот сервер**.

    Без явного `--out` сборщик пишет в панель своего репозитория, и сервер с другим
    каталогом панели молча правил бы чужой файл. Ровно так проверка пересборки перезаписала
    фикстуру, лежащую в репозитории: испортила артефакт, который проверяла, и поймал это
    полный прогон, а не чтение.
    """
    repo_fixture = Path(__file__).resolve().parent.parent / "panel" / "fixture.json"
    before = repo_fixture.read_bytes()

    assert server.panel_dir != repo_fixture.parent, "сервер отдаёт панель репозитория"
    fake_screen(FakeScreen(w=320, h=240, still_ratio=2))
    _post(server, "/api/record", {"kind": "scroll", "seconds": 0.5})
    st = _wait(server, stage="done", limit=300)

    assert st["fixture"]["rebuilt"], st["fixture"]["why"]
    assert (server.panel_dir / "fixture.json").exists()
    assert repo_fixture.read_bytes() == before, (
        "пересборка тронула фикстуру репозитория, а её просили в свой каталог")
