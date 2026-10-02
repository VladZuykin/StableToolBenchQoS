# StableToolBenchQoS

Экспериментальный форк [StableToolBench](https://github.com/THUNLP-MT/StableToolBench). Ответы виртуальных инструментов генерирует LLM, а вероятность успеха, задержка и стоимость вызова моделируются локально.

## О чем проект

- ответы инструментов **синтетические**: успешный вызов всегда уходит в LLM, а не в реальный API;
- **cost_units** — условная стоимость вызова, но токены выбранной модели при успешном вызове тратятся по-настоящему;
- серверу нужен **SIMULATOR_API_KEY**, провайдер должен быть совместим с OpenAI API, иначе будет HTTP 500;
- server/tools, официальный кеш в Git не лежат, их нужно скачать;
- при **QOS_ENABLED=true** сервер отвечает только для API, у которых есть QoS-профиль, иначе возвращает QoS profile not found и LLM не вызывает;
- одинаковый seed повторяет QoS-симуляцию, но не гарантирует одинаковые ответы LLM;
- **toolbench_key** не проверяется, аутентификации нет, сервер только для локальных экспериментов.

От StableToolBench сохранены формат запросов, FastAPI-сервер, документация инструментов, официальный кеш и агентный pipeline. Мы заменили способ выполнения виртуального API: теперь он не вызывает реальные инструменты, а генерирует ответы через LLM и моделирует QoS локально. Дополнительно есть скрипты и Streamlit сервис для разметки эквивалетных API, свой дополнительный кеш для этих инструментов и перефразировки заданий на эти инструменты.

## Быстрый запуск

Проверено на Windows, Git Bash, Python 3.11.13. Нужны Git, curl, Python и ключ провайдера, совместимого с [OpenAI Python SDK](https://github.com/openai/openai-python).

### 1. Зависимости и данные

```bash
git clone https://github.com/VladZuykin/StableToolBenchQoS.git
cd StableToolBenchQoS

python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_win.txt
python -m pip check

bash scripts/download_stabletoolbench_cache.sh
```

Скрипт скачивает описания инструментов и официальный кеш, создаёт server/tools/ и server/tool_response_cache/. Если обе папки уже есть, ничего не скачивается.

В PowerShell окружение активируется через .\.venv\Scripts\Activate.ps1, но shell-скрипты всё равно запускайте через Git Bash.

### 2. Сервер

В первом терминале:

```bash
source .venv/Scripts/activate

read -s -p "Simulator API key: " SIMULATOR_API_KEY
echo
export SIMULATOR_API_KEY

export SIMULATOR_API_BASE="https://api.deepseek.com"
export SIMULATOR_MODEL="deepseek-chat"
export SIMULATOR_SEED="42"

bash scripts/run_virtual_server.sh
```

Когда появится Uvicorn running on http://0.0.0.0:8080, сервер слушает http://127.0.0.1:8080/virtual.

### 3. Запрос

Во втором Git Bash:

```bash
curl -sS -X POST "http://127.0.0.1:8080/virtual" \
  -H "Content-Type: application/json" \
  --data-binary @- <<'JSON'
{
  "category": "Artificial Intelligence/Machine Learning",
  "tool_name": "ai_content_detector_v2",
  "api_name": "chat_gpt_detector_for_ai_content_detector_v2",
  "tool_input": {
    "text": "Christmas is a time of joy, love, and giving."
  },
  "strip": "",
  "toolbench_key": ""
}
JSON
```

Ответ без QoS:

```json
{
  "error": "",
  "response": {
    "all_tokens": 12,
    "used_tokens": 12,
    "real_probability": 0.31,
    "fake_probability": 0.69
  }
}
```

Содержимое response генерирует модель, поэтому у вас оно может отличаться. Остановить сервер: Ctrl+C.

## Как обрабатывается вызов

```text
POST /virtual
  -> нормализация api_id и аргументов
  -> если QoS включён, решаем, успешен ли вызов
     -> неуспешный: ждём задержку, возвращаем ошибку, LLM не трогаем
     -> успешный: до 5 примеров из кеша уходят в LLM
  -> генерация JSON-ответа
  -> ждём оставшуюся часть задержки
  -> ответ клиенту и запись в server/server.log
```

Кеши (официальный и сгенерированный) нужны только как примеры для LLM. Если в кеше уже есть ответ на текущий **tool_input**, он из примеров убирается и напрямую не возвращается.

## HTTP API

### POST /virtual

Генерирует ответ одного API-метода виртуального инструмента.

| Поле | Тип | Обязательное | Описание |
|---|---|:---:|---|
| **category** | string | да | Категория StableToolBench, пробелы, запятые и / заменяются на _ |
| **tool_name** | string | да | Имя инструмента из запроса ToolBench |
| **api_name** | string | да | Имя API, суффикс _for_{tool_name} можно оставить, сервер его уберёт |
| **tool_input** | object/string | да | Аргументы JSON-объектом или строкой, пустая строка = {} |
| **strip** | string | да | Для совместимости с ToolBench, не используется |
| **toolbench_key** | string | да | Для совместимости, не проверяется, может быть пустым |

Если нарушена схема запроса, FastAPI вернёт 422. Если строку **tool_input** не удалось разобрать как JSON, описание ошибки будет в поле error.

Успешный ответ:

```json
{
  "error": "",
  "response": {},
  "qos": {
    "api_id": "Category/tool/api",
    "profile": "normal",
    "profile_found": true,
    "succeeded": true,
    "success_rate": 0.94,
    "latency_ms": 812.42,
    "expected_latency_ms": 850.0,
    "latency_distribution": "lognormal",
    "latency_log_sigma": 0.12,
    "cost_units": 0.0005,
    "call_index": 0
  }
}
```

qos есть только при включённой симуляции. Тип response зависит от API: объект, массив, строка, число, bool или null.

#### Поля qos

| Поле | Смысл |
|---|---|
| **api_id** | Ключ {category}/{tool}/{api} |
| **profile** | normal, degraded или outage |
| **profile_found** | Нашлась ли запись API в файле профилей |
| **succeeded** | Успешен ли этот вызов |
| **success_rate** | Вероятность успеха (ошибки = 1 - success_rate) |
| **latency_ms** | Задержка этого вызова |
| **expected_latency_ms** | Средняя задержка API |
| **latency_distribution** | Пока только lognormal |
| **latency_log_sigma** | Разброс задержки, чем больше, тем сильнее вызовы отличаются от среднего |
| **cost_units** | Условная стоимость попытки, неуспешной тоже |
| **call_index** | Номер вызова этого API с запуска сервера, с нуля |

#### Ошибки

Смоделированные ошибки приходят с HTTP 200, чтобы не ломать контракт ToolBench:

```json
{
  "error": "API not working error...",
  "response": "",
  "qos": {
    "profile_found": true,
    "succeeded": false,
    "success_rate": 0.01
  }
}
```

| Условие | error | LLM вызывается |
|---|---|:---:|
| Вызов неуспешен по success_rate | API not working error... | нет |
| Нет профиля API | QoS profile not found for {api_id} | нет |
| tool_input не парсится | Tool input parse error... | нет |
| LLM три раза вернула невалидный JSON | Failed to generate fake response | да |
| api_name = chat_with_user | статический Chat with user. | нет |

Проблемы с авторизацией, сетью или самим провайдером обычно дают HTTP 500, подробности смотрите в терминале сервера.

Swagger UI: <http://127.0.0.1:8080/docs>, OpenAPI JSON: <http://127.0.0.1:8080/openapi.json>

## Настройка сервера

Переменные окружения важнее [server/config.yml](server/config.yml). Секреты в YAML лучше не хранить.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| **SIMULATOR_API_KEY** | пусто | Ключ LLM |
| **SIMULATOR_API_BASE** | из config или https://api.openai.com/v1 | Адрес OpenAI-совместимого API |
| **SIMULATOR_MODEL** | из config или gpt-4-turbo | Модель |
| **SIMULATOR_SEED** | 42 | Seed для модели |
| **SIMULATOR_TEMPERATURE** | 0 | Температура |
| **SERVER_LOG_FILE** | server/server.log | Журнал запросов и ответов |
| **SERVER_PORT** | 8080 | Порт |
| **QOS_ENABLED** | false | Включить QoS |
| **QOS_PROFILE** | normal | normal, degraded или outage |
| **QOS_SEED** | 42 | Seed QoS-симуляции |
| **QOS_SLEEP_ENABLED** | true | Реально ждать задержку, при false она только пишется в ответ |
| **QOS_PROFILES_FILE** | data/qos/v5/api_qos_profiles.jsonl | Файл профилей |

Пути к папкам инструментов и кеша задаются в server/config.yml. Переменной **SERVER_MODE** нет, успешные ответы всегда создаёт LLM.

## QoS-профили

Профиль — это вероятность успеха, средняя задержка, разброс задержки и условная стоимость API. Для каждого API есть три варианта: normal, degraded, outage. Какой использовать, выбирается через **QOS_PROFILE**.

### 1. Каталог API и исходные метрики

```bash
python scripts/build_tool_catalog.py
```

На выходе:

```text
data/catalog/tools.jsonl
data/catalog/statistics.json
```

В tools.jsonl лежат API из server/tools с метриками StableToolBench (avgLatency, avgServiceLevel, avgSuccessRate), в statistics.json краткая статистика по каталогу и покрытию кеша.

### 2. Профили для тестового набора

```bash
python scripts/build_qos_profiles.py \
  --catalog data/catalog/tools.jsonl \
  --generation-config configs/qos_generation_v1.json \
  --output-dir data/qos/v5 \
  --seed 42
```

Создаются два файла:

- api_qos_profiles.jsonl — normal, degraded и outage для каждого API из **solvable_queries** и **solvable_queries_example** (его и грузит сервер);
- metadata.json — настройки запуска и статистика.

Исходные показатели берутся из поля score в server/tools/{category}/{tool}.json, а не из кеша ответов. Для каждого целевого API генератор берёт показатели *другого* инструмента (выбор зависит от seed и **api_id**, так что результат повторяется). Так API не получает собственную историю, но общее распределение **success_rate** и задержки остаётся близким к исходному. Оба значения берутся у одного инструмента, чтобы не терять связь между ними.

Базовый профиль:

```text
success_rate = (avgServiceLevel / 100) * (avgSuccessRate / 100)
expected_latency_ms = avgLatency
```

normal оставляет это как есть. В degraded вероятность успеха умножается на 0.6375, задержка на 2. В outage множители 0.01 и 4.

Задержка отдельного вызова логнормальная, где m — средняя задержка из профиля:

```text
mu = ln(m) - sigma^2 / 2
latency_ms ~ LogNormal(mu, sigma)
```

**latency_log_sigma** выбирается для каждого API из диапазона 0.05..0.25 (см. configs/qos_generation_v1.json).

### Стоимость

В StableToolBench нет нормальных данных о реальных ценах, поэтому тариф конкретного инструмента не используется. Стоимость берётся из распределения в [configs/qos_generation_v1.json](configs/qos_generation_v1.json):

```text
cost_units ~ LogNormal(median=0.001, log_sigma=1.0)
```

Для каждого API она выбирается один раз и одинакова во всех сценариях, значение ограничено от 0.00005 до 0.05.

### Включить QoS

```bash
export QOS_ENABLED="true"
export QOS_PROFILE="normal"
export QOS_SEED="42"
export QOS_SLEEP_ENABLED="true"
bash scripts/run_virtual_server.sh
```

Успешность вызова определяется до обращения к LLM. Если **succeeded=false**, модель не вызывается и токены не тратятся. При успехе время генерации входит в общую задержку, сервер ждёт только остаток.

При одинаковом **QOS_SEED** последовательность результатов для каждого API повторяется. Если запросы к одному API идут параллельно, порядок **call_index** может отличаться.

## Расширение кеша

Официальный кеш генератор не трогает. Новые примеры лежат отдельно в data/generated_cache/solvable_v1/responses, сервер использует оба источника.

Цель по умолчанию:

- три уникальных входа для API с параметрами;
- один вход {} для API без параметров;
- один LLM-вызов генерирует все недостающие примеры API;
- повторный запуск продолжает с того места, где остановился.

Без --execute скрипт только считает покрытие и недостающие примеры, LLM не вызывает:

```bash
python scripts/build_solvable_response_cache.py
```

Пробный запуск на пять LLM-вызовов:

```bash
python scripts/build_solvable_response_cache.py \
  --execute \
  --max-llm-calls 5
```

Сгенерировать всё недостающее:

```bash
python scripts/build_solvable_response_cache.py --execute
```

Используются те же SIMULATOR_* переменные, их можно переопределить флагами --api-key, --api-base, --model, --seed, --temperature.

| Аргумент | По умолчанию | Назначение |
|---|---|---|
| **--query-root PATH** | solvable_queries и solvable_queries_example | Папка с заданиями, флаг можно повторять |
| **--tools-root PATH** | server/tools | Документация инструментов |
| **--official-cache-root PATH** | server/tool_response_cache | Официальный кеш |
| **--output-dir PATH** | data/generated_cache/solvable_v1 | Куда писать новые примеры |
| **--min-examples N** | 3 | Цель для API с параметрами |
| **--max-llm-calls N** | без лимита | Бюджет запуска |
| **--limit N** | без лимита | Сколько API обработать |
| **--execute** | выключен | Разрешить платные LLM-вызовы |

В output-папке ещё лежат metadata.json, checkpoint.json и errors.jsonl, ключ API туда не пишется. Если модель вернула битый JSON, ошибка попадает в журнал, а при следующем запуске скрипт попробует снова.

## ToolBench-агент

Агент и симулятор независимы, модели и провайдеры у них могут быть разные. Виртуальный сервер должен работать, во втором терминале:

```bash
source .venv/Scripts/activate

read -s -p "Agent API key: " AGENT_API_KEY
echo
export AGENT_API_KEY
export AGENT_API_BASE="https://api.deepseek.com"
export AGENT_MODEL="deepseek-chat"

bash inference_openai_compatible_pipeline_virtual.sh
```

Скрипт гоняет один пример через CoT@1, максимум 5 шагов, один поток. Результат по умолчанию:

```text
data/answer/agent_smoke/llm_virtual/900001_CoT@1.json
```

| Переменная | По умолчанию |
|---|---|
| **SERVICE_URL** | http://localhost:8080/virtual |
| **TOOL_ROOT_DIR** | server/tools |
| **INPUT_QUERY_FILE** | solvable_queries_example/smoke/cache_hit.json |
| **OUTPUT_DIR** | data/answer/agent_smoke/llm_virtual |
| **TOOLBENCH_KEY** | dummy |

Отдельно проверить только вызов инструментов агентом: python scripts/test_agent_tool_call.py (после установки AGENT_*).

## Тесты

Внешняя LLM не нужна:

```bash
python -m unittest discover -s tests -v
```

Проверяются воспроизводимость QoS, распределение задержки, ошибки до вызова LLM, объединение кешей и продолжение прерванной генерации.

Быстрая ручная проверка без ожидания:

```bash
export QOS_ENABLED="true"
export QOS_PROFILE="outage"
export QOS_SLEEP_ENABLED="false"
bash scripts/run_virtual_server.sh
```

Дальше повторите curl из быстрого запуска, часть запросов должна вернуть API not working error....

## Воспроизводимость

Вместе с результатом стоит сохранять:

- Git commit репозитория;
- версии Python 3.11.13 и пакетов (python -m pip freeze);
- провайдера, точный ID модели и все *_SEED;
- metadata.json от генераторов кеша и QoS;
- файлы заданий;
- сценарий QoS и порядок вызовов каждого API.

Облачная модель может меняться под тем же именем, так что точное воспроизведение через время не гарантировано.

## Структура репозитория

| Путь | Что там |
|---|---|
| server/main.py | Обработчик /virtual и генерация через LLM |
| server/qos_simulator.py | Успешность, задержка и стоимость вызова |
| server/config.yml | Несекретные значения по умолчанию |
| scripts/build_qos_profiles.py | Построение QoS-профилей |
| scripts/build_solvable_response_cache.py | Дополнительные примеры для кеша |
| configs/qos_generation_v1.json | Настройки генерации success rate, задержки и стоимости |
| solvable_queries/ | Основной набор заданий |
| solvable_queries_example/ | Маленькие примеры для проверки запуска |
| tests/ | Тесты |
| data/ | Локальные артефакты, в Git не лежат |

# Граф API и кластерный benchmark

Этот документ описывает экспериментальную часть StableToolBenchQoS, которая пока не вошла в основной README: поиск похожих API, построение графа функциональных отношений, ручную проверку разметки и подготовку заданий с альтернативными инструментами.

## Что уже подготовлено

В репозитории зафиксированы итоговый граф после ручной проверки и два набора заданий:

| Артефакт | Содержимое |
|---|---|
| `data/relation_graph/v16_human/clusters.jsonl` | Кластеры функционально связанных API |
| `data/relation_graph/v16_human/api_to_cluster.jsonl` | Соответствие API кластеру |
| `data/relation_graph/v16_human/graph_summary.json` | Статистика итогового графа |
| `data/relation_graph/v16_human/human_override_metadata.json` | Сводка применения ручной разметки |
| `data/benchmark/cluster_queries_v1/queries.json` | Исходные задания с добавленными альтернативными API |
| `data/benchmark/cluster_queries_paraphrased_v1/queries.json` | Исходные задания и их перефразировки |
| `data/qos/v5/api_qos_profiles.jsonl` | QoS-профили для solvable-набора и API из графа |

Актуальная версия графа содержит 6 470 API и 6 027 кластеров. В 274 кластерах находится больше одного API, максимальный размер кластера — 19 API. Кластерный набор содержит 193 исходных задания из 119 кластеров и 991 добавленную альтернативу. После генерации двадцати перефразировок на задание итоговый набор содержит 4 053 запроса.

## Как устроен граф

Сначала для каждого API находятся кандидаты двумя способами:

- dense retrieval с моделью `Qwen/Qwen3-Embedding-0.6B` и косинусным сходством;
- BM25 по тексту документации.

Затем LLM классифицирует пары API. Используются три отношения:

| Отношение | Смысл |
|---|---|
| `interchangeable` | API выполняют одну функцию и могут заменять друг друга после простой локальной адаптации параметров |
| `contains` | Возможности одного API включают возможности другого; направление хранится отдельно |
| `different_capability` | API решают разные задачи |

При экспорте графа `interchangeable` объединяет API в один кластер. Отношение `contains` сохраняется как направленное ребро, но само по себе не объединяет кластеры.

LLM-разметка не считается эталоном. Интерфейс Streamlit позволяет проверять спорные пары, а последние человеческие решения применяются поверх автоматических.

## Зависимости

Основное окружение создаётся по инструкции из `README.md`. Для построения retrieval-кандидатов дополнительно установите зависимости:

```bash
python -m pip install -r requirements_retrieval.txt
```

Для интерфейса ручной проверки лучше использовать отдельное окружение, потому что современный Streamlit может обновить версии `starlette`, `uvicorn` и `anyio`, несовместимые со старой версией FastAPI в StableToolBench:

```bash
python -m venv .venv-review
source .venv-review/Scripts/activate
python -m pip install -r requirements_review.txt
```

## Построение графа с нуля

Полный запуск требует много времени, памяти и платных LLM-вызовов. Готовые итоговые файлы уже находятся в репозитории; повторять весь процесс нужно только для изменения методики или каталога.

### 1. Каталог API

```bash
python scripts/build_tool_catalog.py
```

Каталог `data/catalog/tools.jsonl` содержит 49 937 API endpoints, принадлежащих 10 648 инструментам из 50 категорий. Один инструмент может предоставлять несколько endpoints.

### 2. Retrieval-кандидаты

Пилот на небольшой части каталога:

```bash
python scripts/build_retrieval_pool.py --limit 1000 --device cpu
```

Полный запуск:

```bash
python scripts/build_retrieval_pool.py \
  --model Qwen/Qwen3-Embedding-0.6B \
  --top-k 30 \
  --output-dir data/retrieval
```

Скрипт сравнивает описания API и создаёт список пар потенциально похожих инструментов. Кандидаты находятся двумя способами: по сходству эмбеддингов Qwen и с помощью текстового поиска BM25. Полученные пары объединяются и сохраняются для последующей классификации отношений между API. Эмбеддинги кешируются, поэтому при повторном запуске их не нужно вычислять заново. Для ускорения обработки можно использовать GPU, передав --device cuda.

### 3. LLM-разметка отношений

Используется любой провайдер с OpenAI-совместимым API:

```bash
export ANNOTATOR_API_KEY="..."
export ANNOTATOR_API_BASE="https://api.deepseek.com"
export ANNOTATOR_MODEL="deepseek-chat"
```

expand_relation_graph.py берёт пары потенциально похожих API из data/retrieval/pooled_pairs.jsonl от Qwen и BM25 и передаёт ещё не обработанные пары в LLM. Модель получает документацию обоих API и определяет отношение между ними: взаимозаменяемость, включение возможностей или различие функций.
Перед обращением к LLM скрипт проверяет уже сохранённые данные:
- предыдущие ответы модели в журнале решений;
- ручные решения (про них написано ниже) из data/annotations/human_pair_reviews.jsonl;
- checkpoint завершённых запросов.
Если для пары уже есть решение, повторный LLM-вызов не выполняется. Новые ответы записываются в журнал, ошибки сохраняются отдельно, поэтому остановленный процесс можно продолжить без повторной оплаты уже обработанных пар.

```bash
python scripts/expand_relation_graph.py \
  --pairs data/retrieval/pooled_pairs.jsonl \
  --catalog data/retrieval/catalog.jsonl \
  --max-llm-calls 1000
```

Создает 3 файла:

- data/relation_graph/pair_decisions.jsonl — решения по парам API. Для каждой пары записывается отношение (interchangeable, contains, different_capability и т. д.), направление, уверенность и объяснение модели. Сюда также могут попадать решения, выведенные из уже известных связей без нового обращения к LLM.
- data/relation_graph/errors.jsonl — пары, которые не удалось разметить из-за ошибки LLM, таймаута или некорректного ответа.
- data/relation_graph/checkpoint.json — сводка запуска: сколько пар обработано, сколько сделано LLM-вызовов, какие отношения получены и где остановилась обработка.

## Ручная проверка

### Подготовить очередь для разметки

Пример очереди из API, которые модель назвала взаимозаменяемыми:

```bash
python scripts/build_graph_review_queue.py \
  --decisions data/relation_graph/v16_migrated/pair_decisions.jsonl \
  --relation interchangeable \
  --limit 100 \
  --output data/annotations/graph_review_queue.jsonl
```

Скрипт исключает уже проверенные пары, если передать один или несколько `--existing-reviews`.

Команда сохраняет очередь на ручную проверку в файл:
data/annotations/graph_review_queue.jsonl

### Запустить Streamlit


```bash
export REVIEW_CATALOG_PATH="data/retrieval/catalog.jsonl"
export REVIEW_ANNOTATIONS_PATH="data/annotations/graph_review_queue.jsonl"
export REVIEW_PILOT_PAIRS_PATH="data/annotations/graph_review_queue.jsonl"
export REVIEW_AUDIT_PAIRS_PATH="data/annotations/graph_review_queue.jsonl"
export REVIEW_REVIEWS_PATH="data/annotations/human_graph_reviews.jsonl"

streamlit run apps/review_annotations.py
```

Интерфейс откроется по адресу <http://localhost:8501>. В нём доступны:

- фильтрация по статусу, классу и версии prompt;
- нижний и верхний пороги confidence;
- показ только записей `needs_human_review` (это пишет сама модель, когда составляет пары);
- поиск по названию и `api_id`;
- изменение отношения, направления и уверенности;
- комментарий проверяющего.

Решения дописываются в JSONL-журнал. Повторная проверка не удаляет старую запись: актуальным считается последнее решение для `pair_id`, поэтому сохраняется история изменений.

### Применить ручные решения

```bash
python scripts/apply_human_graph_reviews.py \
  --decisions data/relation_graph/v16_migrated/pair_decisions.jsonl \
  --reviews data/annotations/human_pair_reviews.jsonl \
  --reviews data/annotations/human_graph_reviews.jsonl \
  --output-dir data/relation_graph/v16_human
```

Вручную размечено 302 пары API. Из них 242 присутствовали в используемой версии графа: для 229 пар автоматическая разметка была подтверждена, а для 13 — исправлена. Остальные 60 пар не входили в исходный журнал решений этой версии графа

### Экспортировать кластеры

```bash
python scripts/export_relation_graph.py \
  --run-dir data/relation_graph/v16_human \
  --catalog data/catalog/tools.jsonl
```

После применения ручных исправлений команда заново строит граф и создаёт три файла:
- clusters.jsonl — группы взаимозаменяемых API. Каждая строка содержит один кластер: его идентификатор, список входящих API и решения, на основании которых они были объединены.
- api_to_cluster.jsonl — удобное отображение «API → кластер». Для каждого API указано, к какой группе взаимозаменяемых инструментов он относится. Этот файл используется при построении QoS-профилей и заданий для бенчмарка.
- graph_summary.json — общая статистика графа: количество API, кластеров и связей, число одиночных и многокомпонентных кластеров, размер крупнейшего кластера и распределение типов отношений.

## Анализ разметки

```bash
python scripts/analyze_graph_reviews.py \
  --decisions data/relation_graph/v16_migrated/pair_decisions.jsonl \
  --queue data/annotations/graph_review_queue.jsonl \
  --retrieval-pairs data/retrieval/pooled_pairs.jsonl \
  --reviews data/annotations/human_graph_reviews.jsonl \
  --output-dir data/review_analysis/graph_reviews_v1
```

Скрипт рассчитывает следующую статистику.
Для всех решений модели:
- общее количество размеченных пар;
- сколько получено отношений interchangeable, contains и different_capability;
- распределение этих отношений по диапазонам embedding score;
- распределение по квартилям BM25 score;
- количество пар, для которых embedding или BM25 score отсутствует.
Для пар, проверенных человеком:
- количество событий разметки и уникальных проверенных пар;
- количество решений каждого типа после ручной проверки;
- сколько решений модели подтверждено и сколько оказалось ошибочными;
- долю ошибок модели;
- число и долю ошибок в каждом диапазоне embedding score;
- число и долю ошибок в каждом диапазоне BM25 score;
- число и долю ошибок для каждого значения confidence LLM;
- накопленную долю ошибок для порогов confidence, например ≤ 0.70, ≤ 0.80 и ≤ 0.90.
Создаются два файла:
data/review_analysis/graph_reviews_v1/
├── analysis.json  # полные числовые результаты
└── report.md      # таблицы и краткое описание

Команда:

```bash
python scripts/analyze_graph_reviews.py
```

## Задания с альтернативными API

### Исходный набор

```bash
python scripts/build_cluster_queries.py \
  --clusters data/relation_graph/v16_human/clusters.jsonl \
  --catalog data/catalog/tools.jsonl \
  --output-dir data/benchmark/cluster_queries_v1
```

Этот скрипт создаёт задания для проверки выбора между взаимозаменяемыми API.
Он работает так:
1. Берёт готовые пользовательские задания из StableToolBench.
2. Находит API, который использовался в каждом задании.
3. Проверяет, есть ли у этого API взаимозаменяемые варианты в построенном графе.
4. Если такие варианты есть, добавляет их в список доступных инструментов задания.
5. Сохраняет полученные задания в:
data/benchmark/cluster_queries_v1/queries.json

### Перефразировки

Dry run показывает объём работы:

```bash
python scripts/expand_cluster_queries.py
```

Пилот на пять запросов к модели:

```bash
python scripts/expand_cluster_queries.py --execute --max-llm-calls 5
```

Полная генерация:

```bash
python scripts/expand_cluster_queries.py --execute
```

По умолчанию создаётся двадцать перефразировок каждого задания. Используются `PARAPHRASE_API_KEY`, `PARAPHRASE_API_BASE`, `PARAPHRASE_MODEL`; если они не заданы, скрипт берёт соответствующие `SIMULATOR_*` переменные. Промежуточные результаты сохраняются по одному файлу на исходное задание, поэтому повторный запуск продолжает генерацию, а не начинает её заново.

Перефразирование меняет только текст запроса. Список API, параметры, исходный `query_id` и связь с кластером сохраняются в метаданных записи.

## QoS для графа и solvable-набора

QoS генерируется независимо от кластеров. Файл `api_to_cluster.jsonl` расширяет целевой список API и добавляет `cluster_id`, но не влияет на значения success rate, latency или стоимости.

```bash
python scripts/build_qos_profiles.py \
  --api-to-cluster data/relation_graph/v16_human/api_to_cluster.jsonl \
  --output-dir data/qos/v5 \
  --seed 42
```
Аргументы:
--api-to-cluster сообщает скрипту, какие API входят в построенный граф;
--output-dir задаёт папку для результата;
--seed 42 позволяет при повторном запуске получить те же значения.

Текущий профиль охватывает 7 546 API: объединение 2 491 solvable endpoints и 6 470 endpoints графа; пересечение составляет 1 415 API.

## Кеш ответов для кластерного benchmark

Проверить покрытие без обращения к LLM:

```bash
python scripts/build_solvable_response_cache.py \
  --query-root data/benchmark/cluster_queries_v1
```

В кластерных заданиях используется много API. Чтобы сервер мог имитировать их ответы, для каждого API желательно заранее сохранить несколько примеров. Эти примеры используются как контекст при генерации ответа.

Сгенерировать недостающие примеры:

```bash
python scripts/build_solvable_response_cache.py \
  --query-root data/benchmark/cluster_queries_v1 \
  --execute
```

С флагом --execute скрипт вызывает LLM и сохраняет созданные примеры в:
data/generated_cache/solvable_v1/responses

Неполное покрытие кеша не блокирует успешный вызов: сервер может сгенерировать ответ без полного набора примеров. Однако качество и стабильность формата такого ответа могут быть ниже.

## Что хранится в Git


```text
data/
├── relation_graph/v16_human/
│   ├── clusters.jsonl                 # итоговые кластеры API
│   ├── api_to_cluster.jsonl           # соответствие API → кластер
│   ├── graph_summary.json             # статистика графа
│   ├── human_override_metadata.json   # сведения о ручных исправлениях
│   └── pair_decisions.jsonl           # решения после ручных исправлений
│
├── relation_graph/v16_migrated/
│   └── pair_decisions.jsonl           # исходные решения модели
│
├── benchmark/
│   ├── cluster_queries_v1/
│   │   ├── queries.json               # 193 исходных задания для кластеров
│   │   └── metadata.json
│   └── cluster_queries_paraphrased_v1/
│       ├── queries.json               # 3 955 расширенных заданий c переформулировками
│       └── metadata.json
│
├── generated_cache/solvable_v1/
│   ├── responses/                     # 2 870 сгенерированных примеров ответов для 1 310 API
│   └── metadata.json
│
└── qos/v5/
    ├── api_qos_profiles.jsonl          # QoS-профили 7 546 API
    └── metadata.json
```

## Ссылки

- [StableToolBench (ACL Anthology)](https://aclanthology.org/2024.findings-acl.664/)
- [ToolBench](https://github.com/OpenBMB/ToolBench) и [статья](https://arxiv.org/abs/2307.16789)

## Лицензия

Код репозитория под [Apache License 2.0](LICENSE).
