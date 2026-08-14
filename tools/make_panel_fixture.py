"""Фикстура для панели исследователя: числа из настоящего прогона, а не выдуманные.

Панель — `panel/index.html`, она читает ровно один файл, `panel/fixture.json`. Этот
скрипт его собирает **по записанной сессии**, а не по воображению:

    python3 tools/make_panel_fixture.py                 # свой синтетический прогон
    python3 tools/make_panel_fixture.py --session ПУТЬ  # по чужой записи

## Откуда какое число, и почему это написано в самой фикстуре

У каждой величины в фикстуре есть поле `source` — идентификатор в коде или в
`docs/measurements/`, откуда она взялась. Это не украшение отчёта: панель, показывающая
число без происхождения, ничем не отличается от панели с выдуманным числом, а отличать их
надо на взгляд.

Три вида происхождения, и они не смешиваются:

- **`живая запись оператора`** — снято на настоящей машине с настоящего экрана. Таких
  чисел мало: расход места на запись, размер кадра, механизм захвата;
- **`синтетический прогон`** — посчитано по журналу, который написал этот скрипт
  (`harness loop`): цели, драйвы, ошибка предсказания, граф мест, карта тела;
- **`нет источника`** — величина объявлена, а измерять её пока нечем. Тогда в фикстуре
  лежит `null` и причина словами. Панель обязана показать причину, а не ноль: ноль здесь
  читался бы как «измерили и получилось нуль».

Агента в проекте пока нет (`CLAUDE.md`: «ИИ пока не пишем вообще»), поэтому величины,
которые бывают только у агента — речь, конфабуляция на поведении, — честно приходят
третьим видом. Подставить туда правдоподобное было бы худшим из возможных решений: панель
для этого и делается, чтобы отличать измеренное от предполагаемого.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

#: Дополнять ли профиль старой записи значениями по умолчанию. Ставится флагом
#: `--fill-old-profile`: молча дополнять нельзя, это меняет `profile_hash`.
FILL_OLD = False

LIVE = "живая запись оператора"
SYNTH = "синтетический прогон"
ABSENT = "нет источника"


# ---------------------------------------------------------------------------
# Величина: значение, единица, происхождение, разряды
# ---------------------------------------------------------------------------


def val(label: str, value: Any, unit: str, source: str, origin: str, *,
        digits: int = 2, note: str = "", alarm: bool = False) -> dict[str, Any]:
    """Одна величина для панели.

    `digits` — значащие разряды **по смыслу величины**, а не по вкусу вывода: драйв это
    один знак, доля бюджета — целые проценты, задержка — миллисекунды целыми. Панель
    печатает столько, сколько сказано здесь, и не больше: `8.9103 %` вместо `8.9 %`
    обещает точность, которой в величине нет.

    `source` — где это число живёт в коде. Пустой не принимается.
    """
    if not label:
        raise ValueError("величина без подписи")
    if not source:
        raise ValueError(
            f"величина {label!r} без указания источника. Число без происхождения на "
            "панели неотличимо от выдуманного, а отличать их надо на взгляд")
    if value is None and not note:
        raise ValueError(
            f"величина {label!r} отсутствует без объяснения. Пустое место обязано "
            "называть причину: иначе непонятно, не измерили или измерили и вышло нуль")
    return {"label": label, "value": value, "unit": unit, "source": source,
            "origin": origin, "digits": digits, "note": note, "alarm": alarm}


# ---------------------------------------------------------------------------
# Прогон, по которому собирается фикстура
# ---------------------------------------------------------------------------


def make_session(where: Path, *, rounds: int, seed: int) -> Path:
    """Записать настоящую сессию тем же кодом, каким её пишет оператор.

    Не отдельный «генератор данных для панели», а `harness loop` — тот самый замкнутый
    круг: давление драйва → цель → пробы → тест. Иначе панель показывала бы числа,
    которых в проекте нет ни в одном прогоне.
    """
    where.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "harness.cli", "loop", str(where),
           "--rounds", str(rounds), "--seed", str(seed)]
    env = {"PYTHONPATH": str(ROOT / "src")}
    import os

    subprocess.run(cmd, check=True, capture_output=True,
                   env={**os.environ, **env})
    return where


def make_frames(where: Path, *, seed: int) -> Path:
    """Запись с кадрами — для зеркала первого экрана.

    Отдельная запись, а не подмешивание кадров в основную: зеркало из другой записи — не
    то, что агент видел в этот момент, и панель об этом **говорит** (`mirror_from`).
    Пишется тем же `gen-corpus`, которым записывает оператор.
    """
    where.parent.mkdir(parents=True, exist_ok=True)
    import os

    subprocess.run([sys.executable, "-m", "harness.cli", "gen-corpus", str(where),
                    "--seed", str(seed)],
                   check=True, capture_output=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    return where


def frame_png(image: Any) -> str:
    """Кадр записи как data-URI. Зеркало экрана агента — настоящий кадр, не заглушка."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    plt.imsave(buf, image, cmap="gray", vmin=0, vmax=255, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Сбор
# ---------------------------------------------------------------------------


def check_profile(where: Path, profile: Any) -> Any:
    """Проверить, что профиль записи сходится со схемой, и сказать внятно, если нет.

    Запись, сделанная до того как настройку завели, законна: она не обязана знать о
    настройках, появившихся позже. Незаконно другое — падать на ней `KeyError` из
    словаря, как было на машине оператора: из `KeyError: 'confab_min_episodes'` не
    следует ни причина, ни что делать.

    Дополнение значениями по умолчанию меняет `profile_hash`, поэтому делается только по
    явному флагу и **говорит об этом**: дополненный профиль годится, чтобы прочитать
    старую запись, и не годится, чтобы сравнивать прогоны.
    """
    missing = profile.missing_settings()
    if not missing:
        return profile
    if not FILL_OLD:
        raise SystemExit(
            f"запись {where}: {profile.era()}.\n"
            "Фикстуру по ней собрать нельзя, не решив, что делать с недостающим.\n"
            "  • по свежей записи:  python3 tools/make_panel_fixture.py\n"
            f"  • или по этой, дополнив значениями по умолчанию (profile_hash изменится, "
            f"сравнивать её с другими прогонами после этого нельзя):\n"
            f"      python3 tools/make_panel_fixture.py --session {where} "
            "--fill-old-profile")
    filled = profile.filled_from_schema()
    print(f"профиль записи дополнен значениями по умолчанию: {', '.join(missing)}")
    print(f"  profile_hash изменился: {profile.profile_hash[:8]} → "
          f"{filled.profile_hash[:8]}. Для сравнения прогонов эта фикстура не годится")
    return filled


def collect(session_path: Path, frames_path: Path | None = None) -> dict[str, Any]:
    from harness.core.levels import Ceiling
    from harness.doctor import BYTES_PER_FRAME_1080P
    from harness.model import confabulation, vitals
    from harness.model.questions import from_journal as questions_from_journal
    from harness.model.rebuild import rebuild_from_journal
    from harness.session import Session

    out: dict[str, Any] = {}
    with Session.open(session_path) as s:
        profile = check_profile(session_path, s.profile)
        v = vitals.from_journal(s.journal, profile=profile)
        conf = confabulation.measure(
            s.journal, min_episodes=int(profile.parameters["confab_min_episodes"]))
        questions = questions_from_journal(s.journal).stats()
        rebuilt = rebuild_from_journal(s.journal)
        frames = list(s.journal.frames())
        # Часы берутся с последней записи **любого** вида. Первая редакция брала их с
        # последнего кадра, а в записи `harness loop` кадров нет вовсе — и полоса
        # состояния показывала «0 · 0 · —» по журналу на 381 запись.
        last = None
        for e in s.journal:
            last = e
        clocks = last.stamp if last is not None else None
        mirror = frame_png(s.image(len(s) - 1)) if len(s) else ""
        mirror_from = "эта же запись" if len(s) else ""
        mirror_size = (int(profile.parameters["capture_width"]),
                       int(profile.parameters["capture_height"]))
        # Всё, что считается по журналу **основной** записи, считается здесь, внутри её
        # `with`. До TASK-15 эти восемь строк стояли внутри ветки «зеркало взять из другой
        # записи», и без `--frames-from` сборка падала `UnboundLocalError: goals` — то есть
        # сборка по умолчанию не работала вовсе, а замечено это было только тогда, когда
        # её впервые запустили без аргументов.
        errors = [e.state.prediction_error for e in s.journal
                  if e.state.prediction_error is not None]
        moods = [tuple(e.state.mood) for e in s.journal if e.state.mood is not None]
        drives_last = next((e.state.drives for e in reversed(list(s.journal))
                            if e.state.drives), {})
        goals = [e.event for e in s.journal if str(e.kind) == "goal"]
        layers = confabulation.layer_histogram(s.journal)
        body = rebuilt.body.stats()
        entries = len(s.journal)
    if not mirror and frames_path is not None:
        # Кадров в записи нет: `harness loop` их не пишет — цикл лепета кадры видит,
        # но в журнал не кладёт. Зеркало берётся из отдельной записи с кадрами, и это
        # **сказано на экране**, а не подставлено молча: зеркало из другой записи — не
        # то, что агент видел в этот момент, и путать одно с другим нельзя.
        with Session.open(frames_path) as f:
            mirror = frame_png(f.image(len(f) - 1))
            mirror_from = f"другая запись: {frames_path.name}"
            mirror_size = (int(f.profile.parameters["capture_width"]),
                           int(f.profile.parameters["capture_height"]))

    # --- верхняя полоса ---------------------------------------------------
    # Часы — три, и третьи гаснут, когда содержимого нет: `t_content` в этом прогоне
    # всегда null, потому что агент ничего не просматривал.
    out["bar"] = {
        "driver": "НАБЛЮДЕНИЕ",
        "driver_why": "агента в проекте пока нет: журнал писал `harness loop`, "
                      "инъекции ввода в этом прогоне не было",
        "t_self": clocks.t_self if clocks else 0,
        "t_world": clocks.t_world if clocks else 0,
        "t_content": clocks.t_content if clocks else None,
        "profile_name": profile.name,
        "profile_hash": profile.profile_hash[:8],
        "structure_hash": profile.structure_hash[:8],
        "instances": {"live": 1, "total": 1,
                      "source": "instances.Colony"},
    }

    # Осталось часов записи, а не проценты. Расход — с машины оператора, потолок — из
    # профиля этого прогона: обе величины настоящие, и обе подписаны.
    ceiling = Ceiling.from_profile(profile)
    mb_per_hour = BYTES_PER_FRAME_1080P * float(profile.parameters["capture_fps"]) \
        * 3600 / (1024 * 1024)
    hours = ceiling.hours_left(0.0, mb_per_hour)
    out["bar"]["hours_left"] = val(
        "осталось записи", None if hours is None else round(hours, 1), "ч",
        "core.levels.Ceiling.hours_left", LIVE, digits=1,
        note="" if hours is not None else "потолок не задан")
    out["bar"]["hours_basis"] = (
        f"{BYTES_PER_FRAME_1080P / 1024:.1f} КиБ на запись при кадре 1920×1080, "
        f"механизм screen_dxcam — замер на машине оператора; потолок "
        f"{ceiling.cap_mb / 1024:.0f} ГиБ из профиля")

    # --- экран 1: ровно шесть величин ------------------------------------
    err_mean = v.value("prediction_error_mean")
    last_goal = goals[-1] if goals else {}
    goal_budget = last_goal.get("budget_ticks")
    goal_spent = last_goal.get("spent_ticks")
    share = (None if not goal_budget or goal_spent is None
             else min(1.0, goal_spent / goal_budget))
    # Драйв в срезе — не число, а значение с прогнозом (аллостаз): берётся текущее
    # значение, а прогноз показывается рядом на экране «Состояние».
    lead = (max(((k, float(v_["value"])) for k, v_ in drives_last.items()),
                key=lambda kv: kv[1]) if drives_last else None)

    out["live"] = {
        "mirror": mirror,
        "mirror_note": (f"{mirror_size[0]}×{mirror_size[1]}, синтетический мир, "
                        f"источник кадра — {mirror_from or 'кадров нет'}"),
        "mirror_from": mirror_from,
        "ribbon": [round(x, 4) for x in errors[-160:]],
        "ribbon_label": "ошибка предсказания по записям",
        "values": [
            val("точность предсказания",
                None if err_mean is None else round(1.0 - float(err_mean), 3),
                "0..1", "model.vitals.prediction_error_mean", SYNTH, digits=3,
                note="" if err_mean is not None
                     else "ни одна запись не несёт ошибки предсказания"),
            val("бюджет активной цели", None if share is None else round(share, 2),
                "доля", "behaviour.goals.GoalStack", SYNTH, digits=2,
                note="" if share is not None else "активной цели в конце прогона нет",
                alarm=bool(share is not None and share > 0.9)),
            val("настроение", list(moods[-1]) if moods else None, "валентность×возбуждение",
                "core.journal.StateSnapshot.mood", SYNTH, digits=2,
                note="" if moods else "настроение не снималось ни в одной записи"),
            val("ведущий драйв", None if lead is None else round(lead[1], 1), "0..1",
                "model.drives.Motivation", SYNTH, digits=1,
                note="" if lead is not None else "драйвы в записях не сняты"),
            val("конфабуляция", conf.rate, "доля",
                "model.confabulation.measure", ABSENT if conf.rate is None else SYNTH,
                digits=2,
                note="" if conf.rate is not None else
                     (f"эпизодов с объяснениями {conf.episodes_explained} при пороге "
                      f"{conf.min_episodes}: доля по объяснениям была бы "
                      f"{(conf.share_by_explanation or 0):.0%}, но независимыми "
                      "наблюдениями они не являются")),
            val("задержка планировщика", None, "мс",
                "behaviour.planner.Planner", ABSENT,
                note="контура планировщика в этом прогоне не было: цикл вёл "
                     "`harness loop`, и мерить задержку не у чего"),
        ],
        "lead_drive_name": None if lead is None else lead[0],
        "overlays": [
            {"key": "g", "label": "границы слоёв", "on": False,
             "source": "vision.selfworld"},
            {"key": "w", "label": "окна внимания", "on": False,
             "source": "perception.layers"},
            {"key": "e", "label": "отмеченные сущности", "on": False,
             "source": "model.beliefs.Entity"},
            {"key": "s", "label": "пеленг звука", "on": False,
             "source": "capture.screen.LoopbackAudio"},
            {"key": "f", "label": "контур предсказания", "on": False,
             "source": "model.forward.ForwardModel"},
        ],
    }

    # --- экран 2: состояние ------------------------------------------------
    out["state"] = {
        # Драйв показывается со своим прогнозом: аллостаз — это предсказание нужды, и
        # без прогноза полоса драйва отвечает на другой вопрос.
        "drives": [{"name": k, "value": round(float(v_["value"]), 3),
                    "forecast": round(float(v_.get("forecast", v_["value"])), 3),
                    "digits": 1}
                   for k, v_ in sorted(drives_last.items())],
        "drives_source": "core.journal.StateSnapshot.drives",
        "mood_trail": [[round(a, 3), round(b, 3)] for a, b in moods[-120:]],
        "modulator": None,
        "modulator_note": "собственного символа состояния у агента нет: символы "
                          "заводит восприятие, а агента ещё нет",
        "body": [
            val("выходов живых", body["live"], "штук", "model.rebuild.BodyMap", SYNTH,
                digits=0),
            val("выходов молчащих", body["silent"], "штук", "model.rebuild.BodyMap",
                SYNTH, digits=0),
            val("обратимость неизвестна", body["unknown_reversibility"], "штук",
                "model.rebuild.BodyMap", SYNTH, digits=0),
            val("средняя σ графа мест", v.value("place_sigma_mean"), "с",
                "model.vitals.place_sigma_mean",
                SYNTH if v.value("place_sigma_mean") is not None else ABSENT,
                digits=2,
                note="" if v.value("place_sigma_mean") is not None
                     else "граф мест в журнал не пишется отдельным видом записи"),
        ],
    }

    # --- экран 3: цели и речь ---------------------------------------------
    out["goals"] = {
        # Поля берутся те, что в записи и правда есть: `kind` и `target` — вид цели и
        # её предмет, `budget_ticks`/`spent_ticks` — бюджет. Колонки «тест» здесь нет,
        # потому что в записи цели её нет: подписать пустой столбец словом «тест»
        # значило бы выдумать подпись.
        "stack": [{"code": g.get("code", "?"), "id": g.get("id", ""),
                   "kind": g.get("kind", ""), "target": g.get("target", ""),
                   "drive": g.get("drive", ""),
                   "budget": g.get("budget_ticks"),
                   "spent": g.get("spent_ticks")} for g in goals[-14:]],
        "stack_source": "behaviour.goals.GoalStack",
        "questions": {
            "open": questions["open"], "asked": questions["asked"],
            "coverage": questions["coverage"], "blind": questions["blind"],
            "source": "model.questions.OpenQuestions",
            "note": "" if questions["asked"] else
                    "в этом прогоне ни одна гипотеза не оказалась без теста",
        },
        "speech": {
            "grounded": None, "fluent": None,
            "note": "оба канала пусты: речи в проекте пока нет. Разделение каналов "
                    "объявлено (инвариант 20), наполнять его нечем",
            "source": "INTERFACE.md, каналы общения",
        },
    }

    # --- экран 4: память ---------------------------------------------------
    ents = sorted(rebuilt.beliefs.entities.values(), key=lambda e: -e.encounters)
    out["memory"] = {
        "cards": [{"id": e.id, "kind": e.kind, "encounters": e.encounters,
                   "uses": e.uses, "recalls": e.recalls,
                   "value": round(e.value(), 3), "rank": e.rank(),
                   "affordances": [{"key": k, "mu": round(b.mu, 3),
                                    "sigma": round(b.sigma, 3), "n": b.n,
                                    "origin": str(b.provenance.origin),
                                    "own": b.n_experience}
                                   for k, b in sorted(e.affordances.items())][:8],
                   "links": dict(sorted(e.links.items()))}
                  for e in ents[:24]],
        "cards_source": "model.beliefs.Entity",
        "pipeline": rebuilt.beliefs.stats()["by_stage"],
        "pipeline_source": "model.beliefs.Stage",
        "single_encounter": sum(1 for e in ents if e.encounters == 1),
    }

    # --- экран 5: журнал ---------------------------------------------------
    marks = []
    for e in list(s.journal) if False else []:      # журнал уже закрыт; метки ниже
        pass
    out["journal"] = {
        "entries": entries,
        "by_layer": layers,
        "by_layer_source": "core.journal.Entry.actor_layer",
        "errors": [round(x, 4) for x in errors[-320:]],
        "spikes": [i for i, x in enumerate(errors[-320:])
                   if errors and x > (sum(errors) / len(errors)) * 2],
    }

    # --- экран 6: тело -----------------------------------------------------
    out["body"] = {
        "outputs": [{"id": k, "state": f["state"],
                     "responses": f.get("responses", 0),
                     "reversibility": f.get("reversibility", {})}
                    for k, f in sorted(rebuilt.body.as_dict()["outputs"].items())],
        "source": "model.rebuild.BodyMap",
    }

    # --- показатели целиком (для экрана 7) --------------------------------
    out["vitals"] = v.as_dict()
    return out


def experiments() -> dict[str, Any]:
    """Экран 7: настоящие замеры проекта из `docs/measurements/`.

    Ничего не пересчитывается: панель показывает то, что лежит в файлах замеров, вместе
    с единицей независимости и `n`. Пересчёт здесь означал бы, что на панели числа одни,
    а в `MEASUREMENT.md` другие.
    """
    base = ROOT / "docs" / "measurements"
    runs: list[dict[str, Any]] = []
    files: dict[str, Any] = {}
    for path in sorted(base.glob("*.json")):
        files[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    b = files.get("beliefs", {})
    if b:
        r = b["свидетельство_не_опыт"]
        q = b["очередь_не_перечень"]
        p = b["переоткрытие"]
        runs += [
            {"name": "свидетельство не становится опытом",
             "value": f"{r['сменили_по_новому']} из {r['пересказов']}",
             "was": f"{r['сменили_происхождение_по_прежнему_правилу']} из "
                    f"{r['пересказов']}",
             "unit": "утверждение", "n": r["пересказов"], "task": "TASK-12"},
            {"name": "очередь дел против перечня непонятого",
             "value": f"{q['ожидает_проверки_теперь']} в работе, "
                      f"{q['отложено_теперь']} отложено",
             "was": f"{q['ожидает_проверки_по_прежнему_условию']} одним числом",
             "unit": "вопрос", "n": q["вопросов"] + q["в_работе"], "task": "TASK-12"},
            {"name": "переоткрытие отложенного",
             "value": f"{p['resolved']} разрешено, ожидание "
                      f"{p['mean_wait_entries']:.0f} записей",
             "was": "механизма не было", "unit": "вопрос", "n": p["asked"],
             "task": "TASK-12"},
            {"name": "покрытие переоткрытия",
             "value": f"{p['coverage']:.0%}", "was": "не измерялось",
             "unit": "вопрос", "n": p["asked"], "task": "TASK-12, инвариант 31"},
            {"name": "ложные предложения переоткрытия",
             "value": f"{p['false_offer_share']:.0%} "
                      f"({p['offers_refused']} из {p['offers']})",
             "was": "не измерялось", "unit": "вопрос", "n": p["offers"],
             "task": "TASK-12, инвариант 31"},
        ]

    pa = files.get("params_audit", {})
    if pa:
        reds = pa["редакции"]
        runs.append({"name": "мёртвые и однобокие параметры",
                     "value": f"{pa['дефектов']} из {pa['всего_настроек']}",
                     "was": f"{reds[0]['объявлено']} объявлено дефектными первой "
                            f"редакцией детектора, настоящих {reds[0]['настоящих']}",
                     "unit": "настройка", "n": pa["всего_настроек"], "task": "TASK-10"})
        runs.append({"name": "покрытие прогона со счётчиками",
                     "value": f"{pa['прогон']['прочитано_ключей']} из "
                              f"{pa['всего_настроек']}",
                     "was": "не измерялось", "unit": "настройка",
                     "n": pa["всего_настроек"], "task": "TASK-10, инвариант 31"})

    idn = files.get("identity", {})
    if idn:
        by = {(c["source"], c["case"]): c for c in idn["cases"]}
        for (src, case), c in sorted(by.items()):
            runs.append({
                "name": f"тождество: {case} ({src})",
                "value": f"карточек {c['before']} → {c['after']} при истине "
                         f"{c['true_entities']}, ошибка {c['error_before']} → "
                         f"{c['error_after']}",
                "was": "прежнее правило склеивало похожие: 6 сущностей в 2, "
                       "ошибка 0 → 4",
                "unit": idn["unit"], "n": c["before"], "task": "TASK-14"})

    sr = files.get("storage_rate", [])
    if sr:
        worst = max(sr, key=lambda x: x["gib_per_hour"])
        real = next((x for x in sr if x["kind"] == "рабочий стол"), worst)
        runs.append({"name": "расход места на запись",
                     "value": f"{real['gib_per_hour']:.2f} ГиБ/ч на «{real['kind']}»",
                     "was": f"оценка по шуму давала {worst['gib_per_hour']:.0f} ГиБ/ч",
                     "unit": "сессия", "n": real["frames"], "task": "TASK-08"})

    tb = files.get("text_boundary", {})
    if isinstance(tb, dict) and tb:
        # Ключи берутся точные. Первая редакция искала `captions` и `predicted`, которых
        # в файле нет, и получала `n = 0` из значения по умолчанию — то есть печатала на
        # панели выдуманный нуль рядом с настоящим числом столкновений. Нуль в колонке
        # `n` хуже отсутствия строки: он утверждает, что наблюдений не было.
        runs.append({"name": "столкновения символов на границе текста",
                     "value": f"{tb['collisions']} при ширине {tb['digits']} разрядов",
                     "was": f"при 4 разрядах ожидалось "
                            f"{tb['collisions_expected_narrow']:.0f}",
                     "unit": tb["unit"], "n": tb["n_distinct_captions"],
                     "task": "TASK-06"})
        runs.append({"name": "утечки читаемого текста к агенту",
                     "value": f"{tb['leaks_through_boundary']} через границу",
                     "was": f"{tb['leaks_naive_path']} наивным путём",
                     "unit": tb["unit"], "n": tb["n_distinct_captions"],
                     "task": "TASK-06, инвариант 5"})

    bc = files.get("balance_curve_8seeds", {})
    if bc:
        summary = bc.get("summary", {})
        nonzero = sum(1 for k in summary if summary[k].get("both_nonzero"))
        runs.append({"name": "кривая баланса разведки",
                     "value": f"не разрешилась, p = 0.19",
                     "was": f"ненулевой результат в {nonzero} классах из "
                            f"{len(summary)}",
                     "unit": "прогон", "n": 8, "task": "TASK-06"})

    # --- TASK-24: четыре направления, по строке на утверждение ------------------
    #
    # Строки заводятся вместе с замерами нарочно. Файл, который панель читает и не
    # показывает ни одной строкой, выглядит на экране так же, как файл, которого нет, — и
    # «панель читает все замеры» становится правдой про чтение и ложью про показ.

    lc = files.get("live_cycle", {})
    if lc and lc.get("by_ratio"):
        worst = max(abs(v["confab_median"] - v["reflex_share_median"])
                    for v in lc["by_ratio"].values())
        runs.append({"name": "конфабуляция против доли рефлекса",
                     "value": f"разрыв до {worst * 100:.1f} п.п.",
                     "was": "0.12 п.п. — метрика была переименованием доли рефлекса",
                     "unit": "прогон", "n": len(lc.get("rows", [])),
                     "task": "TASK-24 A"})

    dr = files.get("drives", {})
    if dr and dr.get("part_one"):
        one = dr["part_one"]
        rows_ = one.get("rows", [])
        worst_false = max((r["false"] for r in rows_), default=0)
        runs.append({"name": "связь области с классом событий",
                     "value": f"обе найдены с {one['episodes_to_find_all']} эпизодов",
                     "was": "корреляционной машинерии не было (М6)",
                     "unit": "связь", "n": len(rows_),
                     "task": "TASK-24 B"})
        runs.append({"name": "ложные связи на молчащем классе",
                     "value": f"0 по пикселям, худший прогон механики {worst_false}",
                     "was": "не измерялось", "unit": "связь",
                     "n": len(rows_), "task": "TASK-24 B, инвариант 31"})

    at = files.get("attention", {})
    if at and at.get("verdicts"):
        v1 = at["verdicts"].get("окон 1", {})
        v4 = at["verdicts"].get("окон 4", {})
        fired = sum(x["fired"] for x in at["false_positives"].values())
        checks = sum(x["checks"] for x in at["false_positives"].values())
        runs.append({"name": "арбитраж внимания против раздачи по порядку",
                     "value": (f"{v1.get('median_ratio', 0):.2f}× при одном окне, "
                               f"{v4.get('median_ratio', 0):.2f}× при четырёх"),
                     "was": "окна никто не распределял: ось была мёртвой",
                     "unit": "прогон", "n": len(at.get("rows", [])),
                     "task": "TASK-24 C"})
        runs.append({"name": "ложные срабатывания вывода о внимании",
                     "value": f"{fired} из {checks} проверок",
                     "was": "40–90 % у первого правила вывода",
                     "unit": "прогон", "n": checks,
                     "task": "TASK-24 C, инвариант 31"})

    hr = files.get("hour", {})
    if hr and hr.get("series"):
        rate = hr["series"]["loops_per_s"]
        runs.append({"name": "час непрерывной работы: частота цикла",
                     "value": f"падение {rate[0] / rate[-1]:.1f}× ({rate[0]:.0f} → "
                              f"{rate[-1]:.0f} об/с)",
                     "was": "ожидалось «не растёт»; час не проверялся вовсе",
                     "unit": "отрезок", "n": hr.get("n", 0),
                     "task": "TASK-24 D"})
        runs.append({"name": "час непрерывной работы: память",
                     "value": f"{hr['series']['rss_mb'][0]:.0f} → "
                              f"{hr['series']['rss_mb'][-1]:.0f} МиБ, полка",
                     "was": "ожидалась полка — совпало", "unit": "отрезок",
                     "n": hr.get("n", 0), "task": "TASK-24 D"})

    hc = files.get("hour_classifier", {})
    if hc:
        runs.append({"name": "ложные срабатывания классификатора роста",
                     "value": f"{hc['false_share']:.0%} (все на классе «не растёт»)",
                     "was": "не измерялось", "unit": "серия",
                     "n": len(hc.get("rows", [])), "task": "TASK-24 D, инвариант 31"})

    sd = files.get("slowdown", {})
    if sd and sd.get("rows"):
        by = {r["case"]: r for r in sd["rows"]}
        as_is = by.get("как есть", {}).get("drop") or 0.0
        without = by.get("без графа мест", {}).get("drop") or 0.0
        runs.append({"name": "причина замедления цикла",
                     "value": f"без графа мест {without:.2f}× против {as_is:.2f}×",
                     "was": "причина не была установлена", "unit": "прогон",
                     "n": len(sd["rows"]), "task": "TASK-24 D"})

    rv = files.get("reversibility", {})
    if rv and rv.get("verdict"):
        v = rv["verdict"]
        by = v.get("counts_by_seed", {})
        diverged = sum(1 for x in by.values() if x["порог"] != x["снят"])
        runs.append({"name": "цена ошибки в порядке проб разведки",
                     "value": (f"счётчики разошлись на {diverged} сидах из {len(by)}: "
                               + ", ".join(f"{x['порог']}/{x['снят']}"
                                           for x in by.values())),
                     "was": "совпадали до единицы на пяти сидах из пяти",
                     "unit": "прогон", "n": len(rv.get("rows", [])),
                     "task": "TASK-33 A"})
        pr = v.get("predicted", {}).get("без отката", {})
        if pr:
            runs.append({"name": "метка «дорого» как предсказание отката",
                         "value": (f"{pr['false_alarm_share']:.0%} ложных срабатываний, "
                                   f"{pr['false_confirmation_share']:.0%} ложных "
                                   f"подтверждений"),
                         "was": "98 % и 100 %: метка была перевёрнутой",
                         "unit": "прогон", "n": pr.get("n", 0),
                         "task": "TASK-33 A, инварианты 31 и 32"})

    rg = files.get("regime", {})
    if rg and rg.get("baseline_check"):
        bc = rg["baseline_check"]
        stab = rg.get("reversible_stability", {})
        thin = sorted(d for d, x in stab.items() if x["determined"] < x["runs"])
        runs.append({"name": "признак без опорного уровня молчит",
                     "value": (f"{bc['answered_without_baseline']} из "
                               f"{bc['cells_blind']} клеток ответило без фона"),
                     "was": "фона не было вовсе: сравнивали с нулём",
                     "unit": "признак × прогон", "n": bc["cells_blind"],
                     "task": "TASK-33 B"})
        if thin:
            runs.append({"name": "на чём стояли прежние «ноль ошибок»",
                         "value": ("ответ по обратимости неполон на доменах: "
                                   + ", ".join(thin)),
                         "was": "0 ошибок из 5 — за счёт неизмеренных клеток",
                         "unit": "домен", "n": len(stab),
                         "task": "TASK-33 B, инвариант 31"})

    jd = files.get("judge", {})
    if jd and jd.get("cases"):
        st = jd["cases"].get("уверенный", {})
        runs.append({"name": "калибровка модели себя в чужих глазах",
                     "value": (f"промах {st.get('miss_mean', 0):.3f} при пределе "
                               f"{st.get('best_possible', 0):.3f}"),
                     "was": "оценок ноль: судьи не было ни у одного домена",
                     "unit": "оценка", "n": st.get("judged", 0), "task": "TASK-33 C"})
        runs.append({"name": "целей, закрытых без оценки судьи",
                     "value": f"{jd['verdict']['closed_without_judgement']} из "
                              f"{jd['verdict']['pending_seen']}",
                     "was": "проверять было нечем", "unit": "оценка",
                     "n": jd["verdict"]["pending_seen"],
                     "task": "TASK-33 C, инвариант 10"})

    return {"runs": runs, "files": sorted(files), "source": "docs/measurements/"}


def profile_blocks() -> dict[str, Any]:
    """Конфигуратор: блок А и блок Б, разделённые физически.

    Разделение — не оформление. Поворот ручки из блока Б форкает журнал (инвариант 11),
    и предупреждение об этом обязано стоять на границе, а не в подсказке.
    """
    from harness.core.settings import SCHEMA, applies_of

    rows = [{"key": s.key, "group": s.group, "unit": s.unit, "default": s.default,
             "block": "Б" if s.structural else "А", "note": s.note[:120],
             "applies": [str(p) for p in applies_of(s)],
             "planned": s.planned} for s in SCHEMA]
    return {
        "rows": rows,
        "warning": "ручка блока Б меняет форму опыта: журнал форкается, и прежние "
                   "записи в новой ветке несравнимы (инвариант 11)",
        "source": "core.settings.SCHEMA",
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", type=Path, default=None,
                    help="готовая запись; по умолчанию пишется своя через harness loop")
    ap.add_argument("--frames-from", type=Path, default=None,
                    help="запись с кадрами для зеркала, если в основной их нет")
    ap.add_argument("--fill-old-profile", action="store_true",
                    help="дополнить профиль старой записи значениями по умолчанию; "
                         "profile_hash при этом изменится")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", type=Path, default=ROOT / "panel" / "fixture.json")
    args = ap.parse_args(argv[1:])

    global FILL_OLD
    FILL_OLD = bool(args.fill_old_profile)

    tmp: tempfile.TemporaryDirectory | None = None
    frames_from = args.frames_from
    if args.session is None:
        tmp = tempfile.TemporaryDirectory(prefix="panel-session-")
        session = make_session(Path(tmp.name) / "session", rounds=args.rounds,
                               seed=args.seed)
        print(f"записана сессия: {session}")
        if frames_from is None:
            # Кадров `harness loop` не пишет, а главный объект первого экрана — зеркало.
            # Поэтому запись с кадрами делается **здесь**, а не требуется флагом: до
            # TASK-15 команда из `panel/README.md` без `--frames-from` собирала фикстуру
            # без зеркала, то есть документированный путь давал не тот артефакт, который
            # лежит в репозитории.
            frames_from = make_frames(Path(tmp.name) / "panel-frames", seed=args.seed)
            print(f"записана запись с кадрами для зеркала: {frames_from}")
    else:
        session = args.session

    data = collect(session, frames_from)
    data["experiments"] = experiments()
    data["profile"] = profile_blocks()
    data["about"] = {
        "session": str(session),
        "origins": {LIVE: "снято на машине оператора",
                    SYNTH: "посчитано по журналу этого прогона",
                    ABSENT: "измерять пока нечем; причина названа при величине"},
        "how_to_swap": "панель читает единственный источник — константа SOURCE в "
                       "panel/index.html. Чтобы смотреть другую запись, соберите по ней "
                       "фикстуру: python3 tools/make_panel_fixture.py --session ПУТЬ",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    if tmp is not None:
        tmp.cleanup()

    n_live = len(data["live"]["values"])
    absent = [x["label"] for x in data["live"]["values"] if x["value"] is None]
    print(f"величин на первом экране: {n_live}")
    print(f"из них без числа: {len(absent)} — {', '.join(absent) or 'нет'}")
    print(f"прогонов на экране «Опыты»: {len(data['experiments']['runs'])}")
    # Путь печатается относительным, если он внутри репозитория, и абсолютным иначе:
    # `relative_to` на чужом пути бросает ValueError, и падение на **последней** строке
    # успешной работы обесценивает всё, что она сделала.
    shown = (args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out)
    print(f"записано: {shown} "
          f"({args.out.stat().st_size / 1024:.0f} КиБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
