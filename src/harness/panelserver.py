"""Локальный сервер панели: `harness panel`. `TASK-19`.

Оператор до сих пор записывает через `cmd.exe`, и вопрос «почему» справедлив: панель уже
есть и журнал читает, не хватало обратного направления. Панель — статический HTML от
фикстуры и процессов запускать не может, поэтому здесь минимальный сервер: он отдаёт
панель, принимает запуск записи и показывает ход.

**Это не веб-сервис, а замена терминалу на одной машине.** Отсюда четыре ограничения, и
каждое проверяется тестом:

1. слушает только `127.0.0.1`. Не `0.0.0.0`, не имя машины — адрес задан константой и
   параметром не является;
2. **произвольных путей не принимает**. Куда писать, решает сервер по виду записи
   (`corpus.live.KINDS` плюс `paths.live_path`); вид, которого нет в наборе, — отказ. Путь,
   пришедший из браузера, не открывался бы никогда, но его здесь просто нет в протоколе;
3. запуск записи требует **ключа**, который сервер печатает в терминал и вставляет в
   страницу. Без него любая открытая в том же браузере страница могла бы послать
   `POST` на `127.0.0.1` и начать запись экрана — а это ровно то, чего быть не должно;
4. запись идёт **в этом же процессе**, потоком, а не подпроцессом. Иначе «Прервать» на
   Windows пришлось бы делать сигналом, а `CTRL_C_EVENT` там доставляется всей группе
   процессов и работает не всегда. Поток останавливается флагом, и флаг ведёт в тот же
   `record_interrupted`, что и `Ctrl+C`.

**Ход отдаётся опросом, а не потоком событий.** Обновление идёт раз в секунду
(`progress_min_interval_s`), и на такой частоте `Server-Sent Events` дают только вторую
точку отказа — долгоживущее соединение, которое рвётся при засыпании машины и требует
переподключения. Выбор объявлен здесь, чтобы не выглядеть недоделкой.

**И то, ради чего всё:** после записи фикстура пересобирается сама. Между записью и
панелью стоял ручной запуск `tools/make_panel_fixture.py`, о котором оператор не обязан
помнить, — и пока он о нём не помнил, на первом экране был синтетический шум вместо его
рабочего стола.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .paths import live_path, show
from .session import next_free_path
from .progress import Progress
from .recording import Outcome, Plan, RecordRefused, record

#: Адрес прослушивания. Константа, а не параметр: «дай послушать наружу» — это другая
#: программа с другими требованиями, и превращать одну в другую ключом нельзя.
HOST = "127.0.0.1"

#: Что сервер отдаёт как файлы. Набор закрыт и перечислен: раздавать каталог целиком значит
#: раздавать всё, что в него положат, включая записи и отладочный поток.
STATIC: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/fixture.json": ("fixture.json", "application/json; charset=utf-8"),
}


@dataclass
class State:
    """Что сейчас происходит. Один экземпляр на сервер, под замком."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    kind: str | None = None
    path: Path | None = None
    #: На какой стадии дело. «Запись кончилась» и «всё кончилось» — разные моменты: между
    #: ними идёт пересборка фикстуры, и она занимает секунды. Панель, увидевшая только
    #: `running = false`, показала бы итог без ответа на вопрос «а моя ли это запись
    #: теперь на первом экране».
    stage: str = ""            # "" | recording | rebuilding | done | refused
    running: bool = False
    stop_asked: bool = False
    started_unix: float | None = None
    sample: dict[str, Any] = field(default_factory=dict)
    ready: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    refusal: dict[str, Any] = field(default_factory=dict)
    fixture: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running, "stage": self.stage, "kind": self.kind,
                "path": None if self.path is None else show(self.path),
                "stop_asked": self.stop_asked,
                "started_unix": self.started_unix,
                "sample": dict(self.sample), "ready": dict(self.ready),
                "result": dict(self.result), "refusal": dict(self.refusal),
                "fixture": dict(self.fixture),
            }


def plan_cards(root: Path | None = None) -> list[dict[str, Any]]:
    """Карточки записей для панели: вид, длительность, зачем, как, путь и состояние.

    Берутся из `corpus.live` — того же места, откуда их печатает `--plan`. Второй список
    здесь разошёлся бы с первым молча.
    """
    from .corpus.live import KINDS, SETS, _seconds_of
    from .progress import FILE_KINDS

    out = []
    for name in SETS["minimal"]["kinds"]:
        k = KINDS[name]
        path = live_path(name, root)
        out.append({
            "kind": name, "title": k["title"], "duration": k["duration"],
            "why": k["why"], "how": k["how"],
            "seconds": _seconds_of(k["duration"]),
            "path": show(path), "free_path": show(next_free_path(path)),
            "done": _done_state(path),
            "progress_to_file": name in FILE_KINDS,
        })
    return out


def _done_state(path: Path) -> dict[str, Any]:
    """Сделана ли эта запись и годна ли она. Читается из самой записи, а не помнится.

    Помнить состояние в отдельном файле значило бы завести второй источник истины о том,
    что записано; журнал уже знает.
    """
    from .session import Session

    if not (path / "session.json").exists():
        return {"recorded": False}
    try:
        with Session.open(path) as s:
            turns = len(s._cursors)
            marks = sum(1 for e in s.journal.frames()
                        if e.event.get("code") == "unchanged")
            stop = s.interrupted
            return {
                "recorded": True, "turns": turns, "unchanged": marks,
                "written": turns - marks,
                "interrupted": stop is not None,
                "interrupted_note": (stop or {}).get("note", ""),
                "verify_ok": bool(s.verify()["ok"]),
                "usable": _usable(path.name, turns - marks, marks),
            }
    except Exception as e:
        return {"recorded": True, "broken": f"{type(e).__name__}: {e}"}


def _usable(kind: str, written: int, unchanged: int) -> dict[str, Any]:
    """Годна ли запись **для того, ради чего она делалась**.

    Это не про целостность — целостность проверяет `verify`. Запись неподвижности с нулём
    отметок «без изменений» цела и при этом бесполезна: она мерила мигающий курсор, а не
    фон экрана. Первый замер оператора вышел ровно таким, и панель обязана сказать это
    красным, а не показать зелёную галочку.
    """
    from .progress import FILE_KINDS

    if kind in FILE_KINDS:
        if unchanged == 0:
            return {"ok": False,
                    "why": "ни одной отметки «без изменений»: в кадре что-то мигало — "
                           "курсор, часы или само окно записи. Опорного уровня не "
                           "получилось, запись надо переснять"}
        return {"ok": True,
                "why": f"отметок «без изменений» {unchanged} из {written + unchanged} — "
                       "фон экрана записан"}
    if written == 0:
        return {"ok": False,
                "why": "ни одного изменившегося кадра: экран не менялся вовсе, а эта "
                       "запись про изменения"}
    return {"ok": True, "why": f"изменившихся кадров {written}"}


class Runner:
    """Запись в отдельном потоке. Один экземпляр на сервер: запись идёт одна за раз.

    Одна за раз не по лени: две записи с одного экрана — это два потока кадров с одного
    источника и две линии в одном каталоге. Отказ понятнее, чем очередь.
    """

    def __init__(self, state: State, *, root: Path | None = None,
                 repo: Path | None = None, panel_dir: Path | None = None) -> None:
        self.state = state
        self.root = root
        self.repo = repo
        self.panel_dir = panel_dir or ((repo / "panel") if repo else Path("panel"))
        self._thread: threading.Thread | None = None

    def start(self, kind: str, *, seconds: float | None = None) -> dict[str, Any]:
        from .corpus.live import KINDS, _seconds_of

        if kind not in KINDS:
            raise RecordRefused(
                f"неизвестный вид записи {kind!r}", code="unknown_kind",
                hint=f"виды берутся из плана: {', '.join(sorted(KINDS))}")
        with self.state.lock:
            if self.state.running:
                raise RecordRefused(
                    f"уже идёт запись «{self.state.kind}»", code="busy_running",
                    hint="дождитесь конца или нажмите «Прервать»")
            self.state.running = True
            self.state.stage = "recording"
            self.state.stop_asked = False
            self.state.kind = kind
            self.state.sample = {}
            self.state.ready = {}
            self.state.result = {}
            self.state.refusal = {}
            self.state.fixture = {}
            self.state.started_unix = time.time()

        # Путь **вычисляется здесь**, из вида записи. Из браузера путь не приходит вовсе:
        # его нет в протоколе, а значит его нельзя ни подделать, ни ошибиться в нём.
        path = next_free_path(live_path(kind, self.root))
        with self.state.lock:
            self.state.path = path

        plan = Plan(path=path, kind=kind,
                    seconds=float(seconds if seconds is not None
                                  else _seconds_of(KINDS[kind]["duration"])),
                    actor_human=True, with_audio=True,
                    note=f"из панели, вид {kind}",
                    # Ход в панели показывается опросом, поэтому в терминал сервера его
                    # печатать не надо: он там никому не виден и, для записей с файловым
                    # каналом, ушёл бы в файл рядом с сессией без всякой нужды.
                    progress_channel="none")
        self._thread = threading.Thread(target=self._run, args=(plan,),
                                        name=f"record-{kind}", daemon=True)
        self._thread.start()
        return {"started": True, "kind": kind, "path": show(path),
                "seconds": plan.seconds}

    def stop(self) -> dict[str, Any]:
        with self.state.lock:
            if not self.state.running:
                return {"stopped": False, "why": "запись не идёт"}
            self.state.stop_asked = True
        return {"stopped": True}

    def _should_stop(self) -> bool:
        with self.state.lock:
            return self.state.stop_asked

    def _on_sample(self, s: Any) -> None:
        from .progress import render_line

        with self.state.lock:
            self.state.sample = {
                "elapsed_s": round(s.elapsed_s, 2), "total_s": round(s.total_s, 2),
                "share": round(s.share, 4), "written": s.written,
                "unchanged": s.unchanged, "turns": s.turns,
                "mib": round(s.disk_bytes / (1 << 20), 2),
                "gib_per_hour": (None if s.gib_per_hour is None
                                 else round(s.gib_per_hour, 2)),
                "left_s": None if s.left_s is None else round(s.left_s, 1),
                "change_share": (round(s.written / s.turns, 4) if s.turns else None),
                "line": render_line(s),
            }

    def _run(self, plan: Plan) -> None:
        from .core.profile import MILESTONE_0
        from .cost import verdict

        try:
            prog = Progress.from_profile(
                MILESTONE_0, total_turns=plan.turns, path=plan.path,
                seconds=plan.seconds, kind=None, channel="none")
            out: Outcome = record(
                plan, progress=prog, on_progress=self._on_sample,
                should_stop=self._should_stop,
                on_ready=lambda o: self._publish_ready(o))
            got = out.as_dict()
            got["verdict"] = verdict(
                out.turns.stages(),
                fps=float(MILESTONE_0.parameters["capture_fps"])).as_dict()
            got["usable"] = _usable(plan.kind or "", out.turns.written,
                                    out.turns.unchanged)
            with self.state.lock:
                self.state.result = got
        except RecordRefused as e:
            with self.state.lock:
                self.state.refusal = e.as_dict()
        except Exception as e:                     # захват может сломаться по-своему
            with self.state.lock:
                self.state.refusal = {"refused": True, "code": "failed",
                                      "why": f"{type(e).__name__}: {e}",
                                      "hint": "подробности в терминале, где запущен "
                                              "harness panel"}
            raise
        finally:
            # Запись кончилась — сразу же снимаем «идёт», но стадия остаётся: пересборка
            # фикстуры это часть дела, и молча показать итог раньше неё значило бы оставить
            # оператора с вопросом, его ли запись теперь на первом экране.
            with self.state.lock:
                self.state.running = False
                self.state.stop_asked = False
                self.state.stage = "rebuilding"
            self._rebuild_fixture()
            with self.state.lock:
                self.state.stage = "refused" if self.state.refusal else "done"

    def _publish_ready(self, out: Outcome) -> None:
        with self.state.lock:
            self.state.ready = {
                "mechanism": out.mechanism, "caveat": out.mechanism_caveat,
                "frame": {"width": out.frame[0], "height": out.frame[1]},
                "audio": out.audio_note, "lineage_id": out.lineage_id,
                "profile_hash": out.profile_hash, "path": show(out.path),
            }

    def _rebuild_fixture(self) -> None:
        """Пересобрать фикстуру по только что сделанной записи.

        То, ради чего задача и заведена: между записью и панелью стоял ручной запуск
        `tools/make_panel_fixture.py`, о котором оператор не обязан помнить. Пока он о нём
        не помнил, на первом экране был синтетический шум вместо его рабочего стола.

        Пересборка **не молчит ни в одном исходе**: получилось — сказано, не получилось —
        сказано почему. Молча оставить старую фикстуру значило бы показывать оператору
        чужую запись под видом его собственной.
        """
        path = self.state.path
        repo = self.repo or _repo_root()
        tool = None if repo is None else repo / "tools" / "make_panel_fixture.py"
        if path is None or not (path / "session.json").exists():
            self._set_fixture(False, "записи нет — пересобирать нечего")
            return
        if tool is None or not tool.exists():
            self._set_fixture(
                False, f"не найден сборщик фикстуры ({show(tool) if tool else 'tools/'}). "
                       "Панель показывает прежнюю фикстуру, и это не ваша запись")
            return
        # Куда писать фикстуру, сказано **явно**: в тот каталог панели, который этот сервер
        # и отдаёт. Без `--out` сборщик пишет в панель своего репозитория — то есть сервер
        # с другим каталогом панели молча правил бы чужой файл. Ровно так проверка этого
        # механизма перезаписала фикстуру, лежащую в репозитории.
        out = self.panel_dir / "fixture.json"
        try:
            proc = subprocess.run(
                [sys.executable, str(tool), "--session", str(path), "--out", str(out)],
                cwd=str(repo), capture_output=True, text=True, timeout=600)
        except Exception as e:
            self._set_fixture(False, f"сборщик фикстуры не запустился: {e}")
            return
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            self._set_fixture(False, "сборщик фикстуры отказался: " + " / ".join(tail))
            return
        self._set_fixture(True, f"фикстура пересобрана по записи {show(path)}")


    def _set_fixture(self, ok: bool, why: str) -> None:
        with self.state.lock:
            self.state.fixture = {"rebuilt": ok, "why": why}


def _repo_root() -> Path | None:
    """Корень репозитория, если панель запущена из него. Иначе `None`, и это сказано.

    Ищется по наличию `panel/index.html` и `tools/`: установленный пакет без репозитория
    фикстуру пересобрать не может, и придумывать вместо этого нечего.
    """
    here = Path(__file__).resolve()
    for base in [here.parent.parent.parent, *here.parents]:
        if (base / "panel" / "index.html").exists() and (base / "tools").is_dir():
            return base
    return None


class Handler(BaseHTTPRequestHandler):
    """Обработчик. Ничего, кроме перечисленного, не отдаёт и не делает."""

    server_version = "harness-panel"
    protocol_version = "HTTP/1.1"

    # --- служебное --------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        # Тишина по умолчанию: сервер живёт в том же терминале, где оператор читает
        # подсказки, и лог обращений браузера их бы затопил. Ошибки печатаются явно.
        if self.server.verbose:                      # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # Панель и сервер — одна страница на одной машине; кэш здесь только мешает,
        # показывая прошлую фикстуру после пересборки.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _text(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self) -> bool:
        """Ключ на месте. Без него — отказ.

        Защита не от сети (сервер в сеть не смотрит), а от **другой страницы в том же
        браузере**: любая открытая вкладка может послать запрос на `127.0.0.1`, и без ключа
        она смогла бы начать запись экрана. Ключ выдаётся только той странице, которую
        отдал сам сервер.
        """
        token = self.headers.get("X-Harness-Token", "")
        return secrets.compare_digest(token, self.server.token)  # type: ignore[attr-defined]

    # --- маршруты ---------------------------------------------------------

    def do_GET(self) -> None:                        # noqa: N802
        path = self.path.split("?", 1)[0]
        srv = self.server                            # type: ignore[assignment]
        if path in STATIC:
            name, ctype = STATIC[path]
            src = srv.panel_dir / name               # type: ignore[attr-defined]
            if not src.exists():
                return self._json({"error": f"нет {name}"}, 404)
            body = src.read_bytes()
            if name == "index.html":
                # Ключ вставляется в страницу, которую отдал сервер. Так он попадает
                # только туда, а не в чужую вкладку.
                body = body.replace(
                    b"<head>",
                    f'<head><meta name="harness-token" content="{srv.token}">'  # type: ignore[attr-defined]
                    .encode("utf-8"), 1)
            return self._text(body, ctype)
        if path == "/api/plan":
            return self._json({
                "cards": plan_cards(srv.live_root),   # type: ignore[attr-defined]
                "live_root": show(live_path("", srv.live_root).parent),  # type: ignore[attr-defined]
                "server": True,
            })
        if path == "/api/progress":
            return self._json(srv.state.snapshot())   # type: ignore[attr-defined]
        return self._json({"error": "нет такого адреса", "path": path}, 404)

    def do_POST(self) -> None:                       # noqa: N802
        path = self.path.split("?", 1)[0]
        srv = self.server                            # type: ignore[assignment]
        if not self._authorised():
            return self._json(
                {"refused": True, "code": "no_token",
                 "why": "запрос без ключа страницы",
                 "hint": "откройте панель командой harness panel, а не по прямому адресу"},
                403)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            return self._json({"refused": True, "code": "bad_json",
                               "why": "тело запроса не разобралось"}, 400)
        if path == "/api/record":
            try:
                # Из тела берутся **вид и длительность**, и ничего больше. Пути в протоколе
                # нет: его вычисляет сервер по виду.
                got = srv.runner.start(     # type: ignore[attr-defined]
                    str(body.get("kind", "")),
                    seconds=(float(body["seconds"]) if body.get("seconds") else None))
                return self._json(got)
            except RecordRefused as e:
                return self._json(e.as_dict(), 409)
        if path == "/api/stop":
            return self._json(srv.runner.stop())      # type: ignore[attr-defined]
        return self._json({"error": "нет такого адреса", "path": path}, 404)


class PanelServer(ThreadingHTTPServer):
    """Сервер панели. Живёт на `127.0.0.1` и знает про одну запись за раз."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, *, port: int = 8765, panel_dir: Path | None = None,
                 live_root: Path | None = None, repo: Path | None = None,
                 verbose: bool = False) -> None:
        repo = repo or _repo_root()
        self.panel_dir = panel_dir or ((repo / "panel") if repo else Path("panel"))
        self.live_root = live_root
        self.verbose = verbose
        self.token = secrets.token_urlsafe(24)
        self.state = State()
        self.runner = Runner(self.state, root=live_root, repo=repo,
                             panel_dir=self.panel_dir)
        super().__init__((HOST, int(port)), Handler)

    @property
    def url(self) -> str:
        return f"http://{HOST}:{self.server_address[1]}/"


def serve(*, port: int = 8765, open_browser: bool = True,
          panel_dir: Path | None = None, live_root: Path | None = None,
          verbose: bool = False) -> int:
    """Поднять панель и открыть браузер. Возврат — код выхода команды."""
    srv = PanelServer(port=port, panel_dir=panel_dir, live_root=live_root,
                      verbose=verbose)
    print(f"панель: {srv.url}")
    print(f"слушает только {HOST} — наружу ничего не открыто")
    print("запись запускается из панели, экран 8. Здесь ничего набирать больше не надо.")
    print("остановить сервер: Ctrl+C")
    if not (srv.panel_dir / "fixture.json").exists():
        print(f"\nВНИМАНИЕ: нет {show(srv.panel_dir / 'fixture.json')} — панель откроется "
              "пустой.\nСоберите её: python3 tools/make_panel_fixture.py")
    if open_browser:
        import webbrowser

        threading.Thread(target=lambda: webbrowser.open(srv.url), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nсервер остановлен")
    finally:
        srv.runner.stop()
        srv.server_close()
    return 0
