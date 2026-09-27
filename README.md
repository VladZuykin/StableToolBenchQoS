# StableToolBenchQoS

StableToolBenchQoS — экспериментальный форк [StableToolBench](https://github.com/THUNLP-MT/StableToolBench), в котором ответы виртуальных инструментов генерирует LLM, а вероятность успеха, задержка и стоимость моделируются локально.

## Ограничения

- Ответы инструментов **синтетические**: успешный вызов всегда отправляется внешней LLM, а не реальному API. Не используйте ответы для медицинских, юридических, финансовых или производственных решений.
- `cost_units` — синтетическая денежная стоимость вызова. Каждый успешный виртуальный вызов при этом действительно расходует токены выбранной модели.
- Серверу нужен `SIMULATOR_API_KEY`. Если ключ не задан или провайдер несовместим с OpenAI API, запросы к модели будут завершаться ошибкой HTTP 500.
- Данные `server/tools`, официальный кеш и все результаты в `data/` не хранятся в Git. Их нужно скачать или создать локально.
- При `QOS_ENABLED=true` сервер обслуживает только API, для которых создан QoS-профиль. В противном случае он возвращает `QoS profile not found` и не вызывает LLM.
- Одинаковый seed позволяет повторить локальную симуляцию QoS, но не гарантирует дословно одинаковые ответы внешней LLM.
- Сервер предназначен для локальных экспериментов: `toolbench_key` не проверяется, а аутентификация клиентов не реализована.

## Быстрый запуск

Ниже — минимальный проверенный сценарий для Windows, Git Bash и Python 3.11.13. Нужны Git, `curl`, Python и ключ провайдера с API, совместимым с [OpenAI Python SDK](https://github.com/openai/openai-python).

### 1. Установить зависимости и данные

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

Скрипт скачивает описания инструментов и официальный кеш StableToolBench, а затем создаёт `server/tools/` и `server/tool_response_cache/`. Если обе папки уже существуют, повторно ничего не скачивается.

В PowerShell окружение активируется командой `.\.venv\Scripts\Activate.ps1`, но shell-скрипты проекта всё равно следует запускать через Git Bash или `bash`.

### 2. Запустить сервер

В первом терминале:

```bash
source .venv/Scripts/activate

read -s -p "Simulator API key: " SIMULATOR_API_KEY
echo
export SIMULATOR_API_KEY

# Рабочий пример; можно указать другого OpenAI-совместимого провайдера.
export SIMULATOR_API_BASE="https://api.deepseek.com"
export SIMULATOR_MODEL="deepseek-chat"
export SIMULATOR_SEED="42"

bash scripts/run_virtual_server.sh
```

После сообщения `Uvicorn running on http://0.0.0.0:8080` сервер принимает запросы по адресу `http://127.0.0.1:8080/virtual`.

### 3. Отправить запрос

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

Без QoS ответ имеет вид:

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

Содержимое `response` генерируется моделью и может отличаться. Остановить сервер можно через `Ctrl+C`.

## Как устроен вызов

При запросе к `/virtual` сервер выполняет следующие действия:

```text
POST /virtual
  -> нормализация идентификатора API и входных аргументов
  -> проверка успешности вызова, если QoS включён
     -> неуспешный вызов: задержка и ответ с ошибкой, без обращения к LLM
     -> успешный вызов: до 5 примеров из кеша передаются LLM
  -> генерация нового JSON-ответа
  -> ожидание оставшейся части смоделированной задержки
  -> ответ клиенту и запись в server/server.log
```

Официальный и дополнительно сгенерированный кеши используются только как примеры для LLM. Если в кеше есть ответ для текущего `tool_input`, он исключается из примеров и не возвращается напрямую.

## HTTP API

### `POST /virtual`

Генерирует ответ одного API-метода виртуального инструмента.

#### Тело запроса

| Поле | Тип | Обязательное | Значение |
|---|---:|:---:|---|
| `category` | string | да | Категория StableToolBench. Пробелы, запятые и `/` нормализуются в `_`. |
| `tool_name` | string | да | Имя инструмента из запроса ToolBench. |
| `api_name` | string | да | Имя API. Суффикс `_for_<tool_name>` допустим и удаляется сервером. |
| `tool_input` | object или string | да | JSON-объект аргументов либо строка, содержащая такой объект. Пустая строка означает `{}`. |
| `strip` | string | да | Поле совместимости с ToolBench; виртуальный сервер его не использует. |
| `toolbench_key` | string | да | Поле совместимости; не проверяется и может быть пустым. |

FastAPI проверяет наличие и тип полей запроса. При нарушении схемы возвращается HTTP 422. Если строку `tool_input` невозможно разобрать как JSON, описание ошибки возвращается в поле `error`.

#### Успешный ответ

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

Поле `qos` присутствует только при включённой симуляции. Тип `response` зависит от API: это может быть объект, массив, строка, число, логическое значение или `null`.

#### Поля `qos`

| Поле | Смысл |
|---|---|
| `api_id` | Нормализованный ключ `<category>/<tool>/<api>`. |
| `profile` | Активный сценарий: `normal`, `degraded` или `outage`. |
| `profile_found` | Найдена ли запись API в загруженном JSONL-файле профилей. |
| `succeeded` | Успешен ли конкретный вызов. |
| `success_rate` | Заданная вероятность успеха. Вероятность ошибки равна `1 - success_rate`. |
| `latency_ms` | Задержка конкретного вызова в миллисекундах. |
| `expected_latency_ms` | Средняя задержка для этого API. |
| `latency_distribution` | Сейчас поддерживается только `lognormal`. |
| `latency_log_sigma` | Разброс задержки: чем больше значение, тем сильнее отдельные вызовы отличаются от среднего. |
| `cost_units` | Условная стоимость одной попытки, включая неуспешную. |
| `call_index` | Порядковый номер вызова этого API с момента запуска сервера, начиная с нуля. |

#### Ошибки виртуального инструмента

Смоделированные ошибки возвращаются с HTTP 200, чтобы сохранить контракт ToolBench:

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

| Условие | `error` | Вызывается LLM |
|---|---|:---:|
| Вызов оказался неуспешным согласно `success_rate` | `API not working error...` | нет |
| Профиль API отсутствует | `QoS profile not found for <api_id>` | нет |
| `tool_input` невозможно разобрать | `Tool input parse error...` | нет |
| LLM трижды вернула невалидный JSON | `Failed to generate fake response` | да |
| `api_name` нормализуется в `chat_with_user` | статический `Chat with user.` | нет |

Ошибки авторизации, сети и самого LLM-провайдера обычно приводят к HTTP 500. Подробности выводятся в терминал, где запущен сервер.

### Интерактивная схема

FastAPI автоматически публикует OpenAPI-интерфейс после запуска:

- Swagger UI: `http://127.0.0.1:8080/docs`
- OpenAPI JSON: `http://127.0.0.1:8080/openapi.json`

## Настройка виртуального сервера

Переменные окружения имеют приоритет над [server/config.yml](server/config.yml). Секреты в YAML сохранять не следует.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `SIMULATOR_API_KEY` | пусто | Ключ доступа к LLM. |
| `SIMULATOR_API_BASE` | значение из config или `https://api.openai.com/v1` | Адрес OpenAI-совместимого API. |
| `SIMULATOR_MODEL` | значение из config или `gpt-4-turbo` | Название модели. |
| `SIMULATOR_SEED` | `42` | Seed, передаваемый модели. |
| `SIMULATOR_TEMPERATURE` | `0` | Температура генерации. |
| `SERVER_LOG_FILE` | `server/server.log` | Путь к текстовому журналу запросов и ответов. |
| `SERVER_PORT` | `8080` | TCP-порт виртуального сервера. |
| `QOS_ENABLED` | `false` | Включить симуляцию QoS. |
| `QOS_PROFILE` | `normal` | Сценарий `normal`, `degraded` или `outage`. |
| `QOS_SEED` | `42` | Seed локальной симуляции QoS. |
| `QOS_SLEEP_ENABLED` | `true` | Реально ждать выбранную задержку. При `false` она только возвращается в ответе. |
| `QOS_PROFILES_FILE` | `data/qos/v5/api_qos_profiles.jsonl` | Альтернативный файл профилей. |

Порт можно изменить через `SERVER_PORT`. Пути к папкам инструментов и кеша задаются в `server/config.yml`. Переменной `SERVER_MODE` нет: успешные ответы всегда создаёт LLM.

## Подготовка QoS-профилей

QoS-профиль — это набор характеристик виртуального API: вероятность успешного вызова, средняя задержка, разброс задержки и условная стоимость. Для каждого API создаются три варианта профиля — `normal`, `degraded` и `outage`. Сервер выбирает нужный вариант через `QOS_PROFILE` и использует его при каждом вызове инструмента.

### 1. Создать каталог API и исходных метрик

```bash
python scripts/build_tool_catalog.py
```

Результаты:

```text
data/catalog/tools.jsonl
data/catalog/statistics.json
```

В `tools.jsonl` записывается список API из `server/tools` вместе с исходными метриками StableToolBench: `avgLatency`, `avgServiceLevel` и `avgSuccessRate`. В `statistics.json` сохраняется краткая статистика по каталогу и покрытию кеша. Эти файлы использует следующий шаг.

### 2. Создать QoS-профили для тестового набора

```bash
python scripts/build_qos_profiles.py \
  --catalog data/catalog/tools.jsonl \
  --generation-config configs/qos_generation_v1.json \
  --output-dir data/qos/v5 \
  --seed 42
```

Скрипт создаёт два файла:

- `api_qos_profiles.jsonl` — параметры `normal`, `degraded` и `outage` для каждого API, который используется в заданиях из `solvable_queries` и `solvable_queries_example`;
- `metadata.json` — настройки запуска и статистика результата.

Именно `api_qos_profiles.jsonl` загружает виртуальный сервер при включённом QoS.

Исходные показатели хранятся в поле `score` файлов `server/tools/<category>/<tool>.json`, а не в кеше ответов. Генератор берёт по одной полной записи на инструмент, а затем для каждого целевого API выбирает показатели другого инструмента. Выбор зависит от `seed` и `api_id`, поэтому повторный запуск с теми же параметрами даёт тот же результат.

Так целевой API не получает собственные исторические показатели, но итоговое распределение `success_rate` и средней задержки остаётся близким к исходным данным StableToolBench. Оба значения берутся у одного инструмента, чтобы не потерять связь между ними.

Для базового профиля используются следующие значения:

```text
success_rate = (avgServiceLevel / 100) * (avgSuccessRate / 100)
expected_latency_ms = avgLatency
```

Сценарий `normal` оставляет эти значения без изменений. В `degraded` вероятность успеха умножается на `0.6375`, а средняя задержка — на `2`. В `outage` используются множители `0.01` и `4` соответственно.

Задержка отдельных вызовов меняется по логнормальному распределению вокруг `expected_latency_ms`. В формуле ниже `m` — это средняя задержка из профиля:

```text
mu = ln(m) - sigma^2 / 2
latency_ms ~ LogNormal(mu, sigma)
```

Для каждого API значение `latency_log_sigma` выбирается из диапазона `0.05..0.25`, заданного в `configs/qos_generation_v1.json`. При одинаковых `seed` и `api_id` результат будет тем же.

### Откуда берётся стоимость

[configs/qos_generation_v1.json](configs/qos_generation_v1.json) задаёт распределение условной стоимости. В StableToolBench нет достаточных данных о реальных ценах, поэтому тариф конкретного инструмента не используется:

```text
cost_units ~ LogNormal(median=0.001, log_sigma=1.0)
```

Для каждого API стоимость выбирается один раз и остаётся одинаковой во всех сценариях. Значение ограничено диапазоном от `0.00005` до `0.05`. Это синтетическая денежная стоимость вызова.

### Включить QoS

```bash
export QOS_ENABLED="true"
export QOS_PROFILE="normal"
export QOS_SEED="42"
export QOS_SLEEP_ENABLED="true"
bash scripts/run_virtual_server.sh
```

Сервер определяет результат вызова до обращения к LLM. Если `succeeded=false`, модель не вызывается и её токены не расходуются. При успехе время генерации входит в общую задержку, поэтому сервер ждёт только оставшуюся часть.

При одинаковом `QOS_SEED` последовательность результатов для каждого API повторяется. Если несколько запросов к одному API выполняются одновременно, порядок их `call_index` может отличаться.

## Расширение кеша для тестового набора

Генератор не меняет официальный кеш. Новые примеры сохраняются отдельно в `data/generated_cache/solvable_v1/responses`, а сервер использует оба источника вместе.

Цель по умолчанию:

- три уникальных входа для API с параметрами;
- один вход `{}` для API без параметров;
- один LLM-вызов генерирует все недостающие примеры API;
- повторный запуск продолжает работу по фактическому покрытию.

Чтобы узнать текущее покрытие кеша и количество недостающих примеров, запустите скрипт без `--execute`. В этом режиме он ничего не генерирует и не обращается к LLM:

```bash
python scripts/build_solvable_response_cache.py
```

Для проверки можно разрешить только пять обращений к LLM:

```bash
python scripts/build_solvable_response_cache.py \
  --execute \
  --max-llm-calls 5
```

Чтобы сгенерировать все недостающие примеры:

```bash
python scripts/build_solvable_response_cache.py --execute
```

Скрипт использует те же `SIMULATOR_API_KEY`, `SIMULATOR_API_BASE`, `SIMULATOR_MODEL`, `SIMULATOR_SEED` и `SIMULATOR_TEMPERATURE`. Их можно переопределить флагами `--api-key`, `--api-base`, `--model`, `--seed` и `--temperature`.

Основные аргументы:

| Аргумент | По умолчанию | Назначение |
|---|---|---|
| `--query-root PATH` | `solvable_queries` и `solvable_queries_example` | Папка с заданиями; флаг можно повторять. |
| `--tools-root PATH` | `server/tools` | Документация инструментов. |
| `--official-cache-root PATH` | `server/tool_response_cache` | Папка официального кеша. |
| `--output-dir PATH` | `data/generated_cache/solvable_v1` | Папка для новых примеров и служебных файлов. |
| `--min-examples N` | `3` | Цель для API с параметрами. |
| `--max-llm-calls N` | без лимита | Бюджет текущего запуска. |
| `--limit N` | без лимита | Ограничить число обрабатываемых API. |
| `--execute` | выключен | Разрешить платные LLM-вызовы. Без флага выполняется только планирование. |

В той же папке сохраняются `metadata.json`, `checkpoint.json` и `errors.jsonl`. Ключ API в них не записывается. Если модель вернула некорректный или оборванный JSON, ошибка сохраняется в журнале, а при следующем запуске скрипт снова попробует получить недостающие примеры.

## Запуск ToolBench-агента

Агент и симулятор — независимые роли и могут использовать разные модели и провайдеров. Сначала оставьте виртуальный сервер работающим, затем во втором терминале выполните:

```bash
source .venv/Scripts/activate

read -s -p "Agent API key: " AGENT_API_KEY
echo
export AGENT_API_KEY
export AGENT_API_BASE="https://api.deepseek.com"
export AGENT_MODEL="deepseek-chat"

bash inference_openai_compatible_pipeline_virtual.sh
```

Скрипт запускает один пример через `CoT@1`, максимум пять шагов и один поток. Результат по умолчанию записывается в:

```text
data/answer/agent_smoke/llm_virtual/900001_CoT@1.json
```

Настройки запуска:

| Переменная | По умолчанию |
|---|---|
| `SERVICE_URL` | `http://localhost:8080/virtual` |
| `TOOL_ROOT_DIR` | `server/tools` |
| `INPUT_QUERY_FILE` | `solvable_queries_example/smoke/cache_hit.json` |
| `OUTPUT_DIR` | `data/answer/agent_smoke/llm_virtual` |
| `TOOLBENCH_KEY` | `dummy` |

Отдельно проверить только вызов инструментов агентом можно командой `python scripts/test_agent_tool_call.py` после задания переменных `AGENT_*`.

## Проверка работы

Локальные тесты не обращаются к внешней LLM:

```bash
python -m unittest discover -s tests -v
```

Тесты проверяют воспроизводимость QoS, распределение задержки, обработку ошибок до вызова LLM, объединение кешей и продолжение прерванной генерации примеров.

Для быстрой ручной проверки без реального ожидания:

```bash
export QOS_ENABLED="true"
export QOS_PROFILE="outage"
export QOS_SLEEP_ENABLED="false"
bash scripts/run_virtual_server.sh
```

Затем повторите запрос `curl` из раздела быстрого запуска. При низком `success_rate` часть запросов должна возвращать `API not working error...`.

## Воспроизводимость

Чтобы повторить эксперимент, фиксируйте вместе с результатом:

- Git commit этого репозитория;
- Python 3.11.13 и установленные версии (`python -m pip freeze`);
- провайдера, точный ID/версию модели и все `*_SEED`;
- файлы `metadata.json`, созданные генераторами кеша и QoS;
- использованные файлы заданий;
- сценарий QoS и порядок вызовов каждого API.

Если результаты должны точно воспроизводиться позднее, сохраните вывод `python -m pip freeze` и точное название использованной модели. Содержимое облачной модели может со временем измениться, даже если её имя осталось прежним.

## Структура репозитория

| Путь | Назначение |
|---|---|
| `server/main.py` | Обработчик `/virtual` и генерация ответов через LLM. |
| `server/qos_simulator.py` | Симуляция успешности, задержки и стоимости вызова. |
| `server/config.yml` | Несекретные значения сервера по умолчанию. |
| `scripts/build_qos_profiles.py` | Воспроизводимое построение QoS-профилей. |
| `scripts/build_solvable_response_cache.py` | Генерация дополнительных примеров для кеша. |
| `configs/qos_generation_v1.json` | Настройки генерации вероятности успеха, задержки и стоимости. |
| `solvable_queries/` | Основной набор тестовых заданий. |
| `solvable_queries_example/` | Небольшие примеры для проверки запуска. |
| `tests/` | Изолированные тесты текущей функциональности. |
| `data/` | Локальные производные артефакты; игнорируются Git. |

## Фоновая информация

- [StableToolBench, ACL Anthology](https://aclanthology.org/2024.findings-acl.664/) — опубликованная версия статьи про Stable бенчмарк.
- [ToolBench](https://github.com/OpenBMB/ToolBench) и [статья](https://arxiv.org/abs/2307.16789) — исходный бенчмарк.

## Лицензия

Код этого репозитория распространяется по [Apache License 2.0](LICENSE). Данные StableToolBench, модели и внешние API остаются под лицензиями и условиями их владельцев; Apache 2.0 этого репозитория на них автоматически не распространяется.
