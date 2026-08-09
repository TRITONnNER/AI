# ARCHITECTURE.md — модули и интерфейсы

Проектирование интерфейсов до содержимого — единственное, что делает параллельную работу возможной и позволяет переписывать любой модуль, не трогая соседей.

---

## Золотое правило

**Каждый модуль — функция от среза журнала.**

```
module(journal_slice, state) -> (output, new_state)
```

Отсюда следует всё остальное: любой модуль тестируется офлайн на записанном корпусе без запуска игры, две версии сравниваются на одних данных, а воспроизводимость получается бесплатно.

Модуль, который нельзя прогнать по записи, спроектирован неправильно.

---

## Карта модулей

```
harness/     захват, инъекция, часы, сторож          не знает про агента
journal/     TraceRecord, уровни, адресация, replay  не знает про агента
profile/     настройки, хеш, параметры/структурные   не знает про агента
─────────────────────────────────────────────────────────────────
perception/  слои, отпечатки, символы, звук          М2
body/        лепет, карта действий, обратимость      М1
world/       граф мест, прямая модель, ошибка        М3
beliefs/     конвейер, происхождение, deferred       М4
sleep/       консолидация, вытеснение, сновидение    М4, М8
drives/      гомеостаз, аллостаз, настроение         М6
goals/       структура, жизненный цикл, бюджет       М6
control/     рефлекс, навыки, планировщик, арбитраж  М5, М7
social/      модель других, ReflectedSelf, речь      М9
─────────────────────────────────────────────────────────────────
gate/        классификация внешних действий          НЕ часть агента
debug/       наземная истина                          НЕДОСТУПЕН агенту
```

Три нижних слоя не знают про агента вообще. `gate/` и `debug/` — намеренно вне агента: первый должен быть тупым и неподкупным, второй недостижимым.

---

## Контракты данных

Заморозить до написания содержимого модулей.

### Запись журнала

```
TraceRecord {
  t_self: int              # цикл восприятия агента
  t_world: int             # тик мира
  t_content: int | null    # время внутри просматриваемого содержимого
  wall_clock: float        # якорь реального времени

  profile_hash: str
  lineage_id: str
  branch_id: str

  actor_layer: enum        # reflex | skill | planner | drive | interrupt | human | none
  action: Action | null
  stated_reason_id: str | null

  prediction_error: float
  drives: dict             # значение и прогноз по каждому
  mood: (valence, arousal)
  goal_id: str | null
  goal_transition: enum | null

  segment_ref: SegmentRef | null   # может указывать на понижённый сегмент
}
```

### Действие

```
Action {
  keys: list[ScanCode]
  duration_ms: int
  modifiers: set[ScanCode]
  mouse_delta: (dx, dy) | null
}
```

Никогда не дискретное событие. Длительность — ось, а не свойство.

### Восприятие

```
Percept {
  t_self: int
  screen_mask: Mask              # что прибито к экрану
  world_mask: Mask
  entities: list[EntityObservation]   # отпечаток, положение в кадре, слой
  symbols: list[SymbolObservation]    # SYM_*, область, пространство имён
  sound_bearings: list[(bearing, signature)]
  flow: FlowField                # оптический поток, для параллакса и tau
  focus_windows: list[Rect]      # куда смотрело внимание и почём
}
```

Из `Percept` **не должно быть возможности** восстановить координаты, имена или разметку. Это проверяется тестом инварианта 4.

### Карта действий

```
ActionMap {
  live: dict[ScanCode, EffectClass]   # axis|locomotion|mode|impulse|modifier|contextual
  silent: set[ScanCode]
  combos: list[(modifier, key, EffectClass, confidence)]
  reversibility: dict[ActionSignature, (mu, sigma, n)]
  context_conditioned: bool           # эффект зависит от класса состояния
}
```

### Карточка и убеждение

```
Entity {
  id, kind, fingerprint, agent_label: SymbolRef | null
  encounters: list[EpisodeRef]
  affordances: list[Affordance]     # действие, результат, mu, sigma, n, reversibility
  dynamics: list[Observation]
  relations: list[(RelationType, EntityId, confidence)]
  rank, last_access
}

RelationType = at | part_of | causes | resembles | precedes | represents

Belief {
  claim, provenance, source_ref
  mu, sigma, n
  episode_refs: list[EpisodeRef]
  state: verified | refuted | deferred
  test: TestSpec | null             # null => это вопрос, не гипотеза
}
```

`test == null` — формальный признак вопроса. Не отдельный тип, а отсутствие конструируемой проверки.

### Цель

```
Goal {
  statement, source: drive | curiosity | testimony
  satisfaction_test: TestSpec
  budget, spent, remaining_estimate, sigma
  expires_at
  state: proposed | active | suspended | abandoned | satisfied
  saved_context                     # для возврата после прерывания
}
```

Цель без `satisfaction_test` не создаётся. Это отсекает дрейф на длинных задачах в момент постановки.

---

## Поток данных

```
harness.capture ──Frame──> perception ──Percept──┐
                                                  ├──> world ──WorldState──> control
body.action_map ──────────────────────────────────┘                            │
                                                                               │
drives + goals ────────────────────────────────────────────────────────────────┤
                                                                               ↓
                                             gate ←──ActionRequest── control.arbitrate
                                              │
                                              ↓
                                       harness.inject

           всё вышеперечисленное ──────> journal (дозапись)
```

Ключевое: **`control` не вызывает `inject` напрямую.** Между ними `gate`. Даже когда шлюз пропускает всё, путь один — иначе однажды кто-то добавит обход.

---

## Правила модулей

**`perception` не знает, что такое игра.** Никаких имён, никаких специальных случаев. Если код содержит слово `minecraft`, он в неправильном месте.

**`body` не знает названий клавиш.** Только скан-коды и наблюдаемые эффекты.

**`beliefs` не создаёт утверждений сам.** Только принимает с происхождением.

**`control` не форматирует объяснений.** Оно отдаёт `stated_reason_id`, ссылку на реплику. Сопоставление объяснения с `actor_layer` — работа метрик, а не контроллера.

**`gate` не использует модель.** Только механические признаки: метод запроса, тип элемента, домен. Тупой и неподкупный.

**`debug` пишется, но не читается агентским кодом.** Тест: собрать агента без `debug/` в путях импорта — он должен работать.

---

## Раскладка

```
project/
  harness/  journal/  profile/
  perception/  body/  world/  beliefs/  sleep/
  drives/  goals/  control/  social/
  gate/
  debug/            # отдельно, без импорта из агентского кода
  evals/            # 20-30 воспроизводимых задач с автопроверкой
  corpus/           # записанные сессии для офлайн-тестов
  panel/            # интерфейс исследователя
  tests/invariants/ # по тесту на инвариант
```

---

## Тесты инвариантов

Заводятся вместе с соответствующим модулем.

| Инвариант | Тест |
|---|---|
| 1 | Журнал открыт только на дозапись; попытка перезаписи падает |
| 2 | Ни одной записи без трёх часов и хеша профиля |
| 3 | Планировщик прерывается за N мс и отдаёт частичный результат |
| 4 | Фаззинг `Percept`: ни одно поле не восстанавливает координаты или имена |
| 5 | Ни один нехешированный текст не доходит до планировщика |
| 7 | `Belief` без происхождения не создаётся |
| 8 | `Action` без `duration_ms` не конструируется |
| 10 | Реплики агента не входят ни в один расчёт последствий |
| 12 | **Сборка без `debug/` в путях импорта работает** |
| 13 | Ни одной записи без `actor_layer` — `tests/test_invariants.py::test_invariant_13_no_entry_without_actor_layer`, плюс статическая проверка всех мест вызова `append` в `src/` |
| 14 | Заполнение диска не удаляет нулевой уровень; запись останавливается — `test_invariant_14_disk_pressure_never_deletes_the_trace`, и отдельно `test_invariant_14_eviction_has_no_reference_to_the_reserve`: у механизма вытеснения нет ссылки на резерв |
| 16 | `Hypothesis` с `test = null` создаётся и попадает в `deferred` — `tests/test_journal_v2.py::test_criterion_7_*` |
| 15 | Пропажа сигнала супервизора останавливает инъекцию за N мс |
| 19 | Ни одного вызова `inject` в обход `gate` (проверка статическим анализом) |
| 21 | Нет пути, по которому речь оператора становится действием без прохода через генератор целей |

Инварианты 12 и 19 проверяются статически, а не только тестом: путь обхода не должен существовать в коде.
