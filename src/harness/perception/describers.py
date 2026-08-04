"""Описатели: бесплатные сервисы и локальные модели за файрволом.

Описатель отвечает на единственный вопрос — «что я вижу». Он подставляется в
`PerceptionFirewall` и никогда не вызывается напрямую: всё, что он скажет,
проходит проверку на императивы и оценки, а сырой ответ уходит в отладочный поток.

## Почему провайдер не важен

Любой из них — сменная деталь. Поэтому здесь один класс на семейство протоколов,
а не один на сервис:

- `OpenAICompatible` — Groq, OpenRouter, Together, Mistral, локальный
  llama.cpp/vLLM, LM Studio. Все говорят на `/chat/completions`.
- `GeminiAIStudio` — у Google своя форма запроса.
- `OllamaLocal` — локальный Ollama, `/api/chat`.

Свой сервис добавляется строкой в `PROVIDERS`, а не новым кодом.

## Что делает бесплатный тариф пригодным

Бесплатные тарифы ограничены не деньгами, а числом запросов: у Google AI Studio
это порядка 1500 запросов в сутки, у Groq и OpenRouter — десятки в минуту. При
30 кадрах в секунду час игры даёт 108 000 кадров. Отправлять их подряд бессмысленно
при любой квоте, поэтому здесь четыре механизма экономии, и они важнее выбора
провайдера:

1. **Спрашивать только когда удивило.** `AskGate` пускает запрос, если ошибка
   предсказания выше порога новизны. Единая валюта решает и это — отдельной меры
   «интересности» не заводится. На спокойной сцене запросов не будет вообще.
2. **Кэш по отпечатку кадра.** Побитово одинаковый кадр даёт тот же ответ
   бесплатно. Неподвижный экран — обычное дело, и это самая большая экономия.
3. **Уменьшение и обрезка.** Кадр уменьшается до `model_max_side_px` и, если
   задано окно внимания, обрезается. Токенов у картинки тем меньше, чем она меньше.
4. **Цепочка провайдеров.** Кончилась квота у одного — спрашиваем следующего.
   Кто ответил, пишется в отладочный поток: иначе непонятно, чья это была модель.

## Чего здесь нет

Ни одного ключа в коде и ни одного значения по умолчанию для ключа. Нет ключа —
`BackendUnavailable` с указанием, какую переменную окружения выставить. Молча
подменять живую модель локальной заглушкой нельзя: сравнение прогонов сразу
перестанет что-либо значить.

Числа лимитов в `PROVIDERS` — справочные, по состоянию на август 2026, и их надо
перепроверять: тарифы меняются чаще, чем код. `probe()` проверяет не лимиты, а
наличие ключа и доступность узла.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from ..capture.base import BackendUnavailable
from .firewall import QUESTION, ZONES, SIZES
from .imagecodec import base64_png, crop, data_uri, downscale, frame_fingerprint

# Формат ответа задаётся здесь и требуется от любой модели. Это не украшение:
# свободный текст пришлось бы разбирать догадками, а разбор догадками — источник
# тихих ошибок. Всё, что не разобралось, отбрасывается и считается.
FORMAT_RULES = (
    "Ответь строками вида `МЕТКА | зона | размер`, по одной на каждый различимый "
    "объект, и ничего кроме них.\n"
    f"зона — одно из: {', '.join(ZONES)}.\n"
    f"размер — одно из: {', '.join(SIZES)}.\n"
    "МЕТКА — короткий ярлык объекта латиницей, без пробелов.\n"
    "Не советуй, не оценивай важность, не предлагай действий: только то, что видно."
)


@dataclass(frozen=True, slots=True)
class Provider:
    """Справка о сервисе. Лимиты — по состоянию на август 2026, перепроверяйте."""

    name: str
    kind: str                  # "openai" | "gemini" | "ollama"
    base_url: str
    model: str
    key_env: str | None
    rpm: int | None            # запросов в минуту на бесплатном тарифе
    rpd: int | None            # запросов в сутки
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "model": self.model,
                "key_env": self.key_env, "rpm": self.rpm, "rpd": self.rpd,
                "note": self.note}


# Порядок — предлагаемый порядок обхода: сначала то, что не требует сети и квоты.
PROVIDERS: tuple[Provider, ...] = (
    Provider("ollama", "ollama", "http://127.0.0.1:11434", "moondream", None,
             None, None,
             "локально, без сети и без квоты; моделью может быть moondream, "
             "smolvlm, qwen-vl или llava — что установлено"),
    Provider("llamacpp", "openai", "http://127.0.0.1:8080/v1", "local", None,
             None, None,
             "локальный llama.cpp или vLLM с OpenAI-совместимым API"),
    Provider("gemini", "gemini",
             "https://generativelanguage.googleapis.com/v1beta",
             "gemini-2.5-flash", "GEMINI_API_KEY", 60, 1500,
             "Google AI Studio: самая большая суточная квота из бесплатных, "
             "картинки на входе поддержаны"),
    Provider("groq", "openai", "https://api.groq.com/openai/v1",
             "meta-llama/llama-4-scout-17b-16e-instruct", "GROQ_API_KEY", 30, None,
             "очень быстрый; проверьте, что выбранная модель принимает картинки"),
    Provider("openrouter", "openai", "https://openrouter.ai/api/v1",
             "google/gemini-2.0-flash-exp:free", "OPENROUTER_API_KEY", 20, None,
             "один ключ на много бесплатных моделей; лимит зависит от модели"),
    Provider("mistral", "openai", "https://api.mistral.ai/v1",
             "pixtral-12b-2409", "MISTRAL_API_KEY", None, None,
             "бесплатный тариф с картинками, лимиты уточняйте в консоли"),
)

BY_NAME = {p.name: p for p in PROVIDERS}


# ---------------------------------------------------------------------------
# Экономия квоты
# ---------------------------------------------------------------------------


class RateLimiter:
    """Ведро с жетонами на минуту и на сутки.

    Отказ не бросает исключение и не ждёт: он возвращает «нельзя», и решать, что
    делать, — дело вызывающего. Ждать внутри было бы блокировкой основного цикла,
    а мир не ставится на паузу (инвариант 3).
    """

    def __init__(self, rpm: int | None, rpd: int | None, *,
                 now: Callable[[], float] = time.monotonic) -> None:
        self.rpm = rpm
        self.rpd = rpd
        self._now = now
        self._minute: list[float] = []
        self._day: list[float] = []
        self.refused = 0

    def allow(self) -> bool:
        t = self._now()
        self._minute = [x for x in self._minute if t - x < 60.0]
        self._day = [x for x in self._day if t - x < 86400.0]
        if self.rpm is not None and len(self._minute) >= self.rpm:
            self.refused += 1
            return False
        if self.rpd is not None and len(self._day) >= self.rpd:
            self.refused += 1
            return False
        self._minute.append(t)
        self._day.append(t)
        return True

    def state(self) -> dict[str, Any]:
        return {"used_last_minute": len(self._minute), "used_last_day": len(self._day),
                "rpm": self.rpm, "rpd": self.rpd, "refused": self.refused}


class AnswerCache:
    """Кэш ответов по побитовому отпечатку кадра.

    Тождество, а не похожесть: приблизительный кэш отдавал бы ответ про другой
    кадр, и найти это было бы почти невозможно.
    """

    def __init__(self, size: int = 4096) -> None:
        self.size = max(1, int(size))
        self._store: dict[str, str] = {}
        self._order: list[str] = []
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> str | None:
        value = self._store.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: str, value: str) -> None:
        if key in self._store:
            return
        self._store[key] = value
        self._order.append(key)
        while len(self._order) > self.size:
            self._store.pop(self._order.pop(0), None)

    def state(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {"size": len(self._store), "cap": self.size, "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else 0.0}


class AskGate:
    """Спрашивать модель только тогда, когда мир удивил.

    Порог берётся из того же `PredictionError`, что и всё остальное: заводить
    отдельную меру «интересности» значило бы иметь две валюты вместо одной.
    """

    def __init__(self, min_novelty: float, *, min_gap_cycles: int = 0) -> None:
        self.min_novelty = float(min_novelty)
        self.min_gap = max(0, int(min_gap_cycles))
        self._last_asked_at: int | None = None
        self.asked = 0
        self.skipped_quiet = 0
        self.skipped_gap = 0

    def should_ask(self, error_value: float | None, t_self: int) -> bool:
        if error_value is None:
            # Ошибку ещё не посчитали — на первом кадре спросить надо, иначе
            # агент не увидит вообще ничего.
            if self._last_asked_at is None:
                self._last_asked_at = t_self
                self.asked += 1
                return True
            return False
        if self._last_asked_at is not None and t_self - self._last_asked_at < self.min_gap:
            self.skipped_gap += 1
            return False
        if error_value < self.min_novelty:
            self.skipped_quiet += 1
            return False
        self._last_asked_at = t_self
        self.asked += 1
        return True

    def state(self) -> dict[str, Any]:
        total = self.asked + self.skipped_quiet + self.skipped_gap
        return {"asked": self.asked, "skipped_quiet": self.skipped_quiet,
                "skipped_gap": self.skipped_gap,
                "ask_rate": round(self.asked / total, 4) if total else 0.0,
                "min_novelty": self.min_novelty, "min_gap_cycles": self.min_gap}


# ---------------------------------------------------------------------------
# Сами описатели
# ---------------------------------------------------------------------------


def _http_post(url: str, payload: dict[str, Any], headers: dict[str, str],
               timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise BackendUnavailable(
            f"{url} ответил {e.code}: {detail}. Для бесплатных тарифов 429 значит "
            "«квота кончилась» — это нормально, попробуйте другого провайдера") from e
    except urllib.error.URLError as e:
        raise BackendUnavailable(
            f"{url} недоступен: {e.reason}. Если это локальный узел — запустите его "
            "(например `ollama serve`); если сетевой — проверьте доступ") from e


class RemoteDescriber:
    """Общая часть: единственный вопрос, уменьшение кадра, кэш, лимит."""

    def __init__(self, provider: Provider, *, max_side: int = 512,
                 timeout_s: float = 30.0, cache: AnswerCache | None = None,
                 limiter: RateLimiter | None = None,
                 attention_box: tuple[int, int, int, int] | None = None,
                 model: str | None = None, internet: bool = True) -> None:
        self.provider = provider
        self.model = model or provider.model
        self.max_side = int(max_side)
        self.timeout_s = float(timeout_s)
        self.cache = cache or AnswerCache()
        self.limiter = limiter or RateLimiter(provider.rpm, provider.rpd)
        self.attention_box = attention_box
        # Разрешён ли этому прогону интернет. Ручка структурная: прогон без сети и
        # прогон с сетью — разный опыт, и смешивать их нельзя.
        self.internet = bool(internet)
        self.calls = 0
        self.name = f"{provider.name}:{self.model}"

    # --- доступность --------------------------------------------------------

    def api_key(self) -> str | None:
        if self.provider.key_env is None:
            return None
        return os.environ.get(self.provider.key_env)

    def probe(self) -> tuple[bool, str]:
        if not self.internet and self.provider.key_env is not None:
            return False, ("интернет этому прогону не разрешён "
                           "(`internet_access=False`)")
        return self._probe()

    def _probe(self) -> tuple[bool, str]:
        """Можно ли пользоваться. Возвращает (да/нет, почему).

        Для сетевого сервиса проверяется только ключ: тратить запрос из суточной
        квоты на проверку — расточительство. Для локального узла проверяется
        именно доступность: «ключ не нужен» ещё не значит «оно запущено», а
        сказать «настроено» про неработающий узел было бы обещанием сверх знания.
        """
        if self.provider.key_env is not None:
            if not self.api_key():
                return False, (f"нет ключа: выставьте переменную окружения "
                               f"{self.provider.key_env}")
            return True, "ключ есть; квота проверится первым запросом"

        host, port = self._host_port()
        if host is None:
            return True, "локальный узел, адрес не разобран — проверится запросом"
        import socket
        try:
            with socket.create_connection((host, port), timeout=0.35):
                return True, f"локальный узел отвечает на {host}:{port}"
        except OSError as e:
            hint = ("запустите `ollama serve` и поставьте модель "
                    "(`ollama pull moondream`)" if self.provider.kind == "ollama"
                    else "запустите локальный сервер с OpenAI-совместимым API")
            return False, f"{host}:{port} не отвечает ({e.strerror or e}); {hint}"

    def _host_port(self) -> tuple[str | None, int]:
        from urllib.parse import urlparse
        parsed = urlparse(self.provider.base_url)
        if not parsed.hostname:
            return None, 0
        return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)

    def require(self) -> None:
        if not self.internet and self.provider.key_env is not None:
            raise BackendUnavailable(
                f"{self.provider.name}: этому прогону интернет не разрешён "
                "(`internet_access=False`). Это заявленное ограничение прогона, а "
                "не поломка: локальный узел спрашивать можно, сетевой сервис — нет")
        ok, why = self.probe()
        if not ok:
            raise BackendUnavailable(f"{self.provider.name}: {why}")

    # --- подготовка кадра ---------------------------------------------------

    def prepare(self, frame: np.ndarray) -> np.ndarray:
        image = frame
        if self.attention_box is not None:
            image = crop(image, self.attention_box)
        return downscale(image, self.max_side)

    def prompt(self, question: str) -> str:
        if question != QUESTION:
            raise ValueError(
                "описателю задан не тот вопрос. Через файрвол проходит один вопрос — "
                "«что я вижу», и он константа")
        return f"{question}\n\n{FORMAT_RULES}"

    # --- главный вызов ------------------------------------------------------

    def describe(self, frame: Any, question: str) -> str:
        text = self.prompt(question)
        image = self.prepare(np.asarray(frame))
        key = frame_fingerprint(image) + "|" + self.name
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if not self.limiter.allow():
            raise BackendUnavailable(
                f"{self.provider.name}: квота исчерпана "
                f"({self.limiter.state()}). Это не поломка: у бесплатных тарифов "
                "так и задумано. Возьмите следующего провайдера в цепочке")
        self.require()
        answer = self._ask(image, text)
        self.calls += 1
        self.cache.put(key, answer)
        return answer

    def _ask(self, image: np.ndarray, text: str) -> str:
        raise NotImplementedError

    def state(self) -> dict[str, Any]:
        return {"name": self.name, "calls": self.calls,
                "cache": self.cache.state(), "limiter": self.limiter.state()}


class OpenAICompatible(RemoteDescriber):
    """`/chat/completions` с картинкой в `image_url`. Умеет большинство сервисов."""

    def _ask(self, image: np.ndarray, text: str) -> str:
        headers = {}
        key = self.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": self.model,
            "temperature": 0.0,          # описание не должно меняться от прогона к прогону
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_uri(image)}},
                ],
            }],
        }
        data = _http_post(f"{self.provider.base_url}/chat/completions", payload,
                          headers, self.timeout_s)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise BackendUnavailable(
                f"{self.provider.name}: ответ не той формы: {str(data)[:300]}") from e


class GeminiAIStudio(RemoteDescriber):
    """Google AI Studio. Самая большая суточная квота из бесплатных."""

    def _ask(self, image: np.ndarray, text: str) -> str:
        key = self.api_key()
        url = (f"{self.provider.base_url}/models/{self.model}:generateContent"
               f"?key={key}")
        payload = {
            "contents": [{"parts": [
                {"text": text},
                {"inline_data": {"mime_type": "image/png", "data": base64_png(image)}},
            ]}],
            "generationConfig": {"temperature": 0.0},
        }
        data = _http_post(url, payload, {}, self.timeout_s)
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as e:
            raise BackendUnavailable(
                f"gemini: ответ не той формы: {str(data)[:300]}") from e


class OllamaLocal(RemoteDescriber):
    """Локальный Ollama. Ни сети, ни квоты, ни расхода — только своё железо."""

    def _ask(self, image: np.ndarray, text: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.0},
            "messages": [{"role": "user", "content": text,
                          "images": [base64_png(image)]}],
        }
        data = _http_post(f"{self.provider.base_url}/api/chat", payload, {},
                          self.timeout_s)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as e:
            raise BackendUnavailable(
                f"ollama: ответ не той формы: {str(data)[:300]}") from e


KINDS: dict[str, type[RemoteDescriber]] = {
    "openai": OpenAICompatible,
    "gemini": GeminiAIStudio,
    "ollama": OllamaLocal,
}


def make(name: str, **kwargs: Any) -> RemoteDescriber:
    """Собрать описателя по имени провайдера."""
    provider = BY_NAME.get(name)
    if provider is None:
        raise KeyError(f"нет провайдера {name!r}; есть {sorted(BY_NAME)}")
    return KINDS[provider.kind](provider, **kwargs)


def from_profile(name: str | None, profile: Any, **kwargs: Any) -> RemoteDescriber:
    """То же, но параметры берутся из профиля, а не задаются на месте.

    `name=None` — взять того, кто назван в профиле (`model_provider`). Иначе
    настройка «какой сервис спрашивать» существовала бы только на бумаге.
    """
    p = profile.parameters
    chosen = name or str(profile.structural["model_provider"])
    return make(chosen, max_side=int(p["model_max_side_px"]),
                timeout_s=float(p["model_timeout_s"]),
                internet=bool(profile.structural["internet_access"]),
                cache=AnswerCache(int(p["model_cache_size"])), **kwargs)


def gate_from_profile(profile: Any) -> AskGate:
    """Порог «когда вообще спрашивать» — из профиля, а не из вызова."""
    p = profile.parameters
    return AskGate(float(p["model_min_novelty"]),
                   min_gap_cycles=int(p["model_min_gap_cycles"]))


def chain_from_profile(profile: Any, **kwargs: Any) -> Chain:
    """Цепочка по профилю: сначала названный сервис, потом запасные.

    `model_fallback=False` означает «спрашивать только названного»: без этого
    настройка бессмысленна, а сравнение прогонов ломается — ответы разных моделей
    это разные данные.
    """
    p = profile.parameters
    first = str(profile.structural["model_provider"])
    if not bool(p["model_fallback"]):
        names: list[str] = [first]
    else:
        names = [first] + [q.name for q in PROVIDERS if q.name != first]
    return available(names, max_side=int(p["model_max_side_px"]),
                     timeout_s=float(p["model_timeout_s"]),
                     internet=bool(profile.structural["internet_access"]),
                     **kwargs)


@dataclass(slots=True)
class Chain:
    """Цепочка описателей: кончилась квота у одного — спрашиваем следующего.

    Кто именно ответил, видно в `answered_by` и уходит в отладочный поток. Без
    этого сравнение прогонов теряет смысл: ответы разных моделей — разные данные,
    и путать их нельзя.
    """

    describers: list[RemoteDescriber]
    name: str = "chain"
    answered_by: list[str] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)

    def describe(self, frame: Any, question: str) -> str:
        if not self.describers:
            raise BackendUnavailable(
                "в цепочке нет ни одного описателя. Настройте хотя бы один: "
                "локальный Ollama или ключ бесплатного сервиса")
        problems: list[str] = []
        for d in self.describers:
            try:
                answer = d.describe(frame, question)
            except BackendUnavailable as e:
                problems.append(f"{d.name}: {e}")
                self.failures.append((d.name, str(e)))
                continue
            self.answered_by.append(d.name)
            return answer
        raise BackendUnavailable(
            "ни один описатель не ответил:\n  " + "\n  ".join(problems))

    def state(self) -> dict[str, Any]:
        from collections import Counter
        return {"describers": [d.name for d in self.describers],
                "answered_by": dict(Counter(self.answered_by)),
                "failures": len(self.failures),
                "per_describer": [d.state() for d in self.describers]}


def probe_all(**kwargs: Any) -> list[dict[str, Any]]:
    """Что настроено на этой машине. Для CLI и для отчёта в журнал сессии."""
    out: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        describer = KINDS[provider.kind](provider, **kwargs)
        ok, why = describer.probe()
        out.append({**provider.as_dict(), "configured": ok, "why": why})
    return out


def available(names: Iterable[str] | None = None, **kwargs: Any) -> Chain:
    """Цепочка из того, что настроено. Порядок — как в `PROVIDERS`.

    Локальное идёт первым: у него нет ни квоты, ни расхода, ни зависимости от
    сети, поэтому при прочих равных спрашивать надо его.
    """
    wanted = list(names) if names is not None else [p.name for p in PROVIDERS]
    chain: list[RemoteDescriber] = []
    for name in wanted:
        provider = BY_NAME.get(name)
        if provider is None:
            continue
        describer = KINDS[provider.kind](provider, **kwargs)
        ok, _ = describer.probe()
        if ok:
            chain.append(describer)
    return Chain(chain)
