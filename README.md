# StableToolBenchQoS

Экспериментальный форк [StableToolBench](https://github.com/THUNLP-MT/StableToolBench). Ответы виртуальных инструментов генерирует LLM, а вероятность успеха, задержка и стоимость вызова моделируются локально.

## О чем проект

- ответы инструментов **синтетические**: успешный вызов всегда уходит в LLM, а не в реальный API;
- **cost_units** — условная стоимость вызова, но токены выбранной модели при успешном вызове тратятся по-настоящему;
- серверу нужен **SIMULATOR_API_KEY**, провайдер должен быть совместим с OpenAI API, иначе будет HTTP 500;
- server/tools, официальный кеш и всё содержимое data/ в Git не лежат, их нужно скачать или сгенерировать;
- при **QOS_ENABLED=true** сервер отвечает только для API, у которых есть QoS-профиль, иначе возвращает QoS profile not found и LLM не вызывает;
- одинаковый seed повторяет QoS-симуляцию, но не гарантирует одинаковые ответы LLM;
- **toolbench_key** не проверяется, аутентификации нет, сервер только для локальных экспериментов.

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

## Ссылки

- [StableToolBench (ACL Anthology)](https://aclanthology.org/2024.findings-acl.664/)
- [ToolBench](https://github.com/OpenBMB/ToolBench) и [статья](https://arxiv.org/abs/2307.16789)

## Лицензия

Код репозитория под [Apache License 2.0](LICENSE).
