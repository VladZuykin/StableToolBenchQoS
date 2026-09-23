# StableToolBenchQoS

Экспериментальный fork [StableToolBench](SOURCE_README.md) для НИР по теме «Разработка алгоритма подбора инструментов для агентных систем среди семантически схожих кандидатов».

## Quickstart

Ниже приведён минимальный воспроизводимый запуск одного задания через OpenAI-совместимого LLM-агента и локальный StableToolBench в режиме `cache_only`. Команды выполняются из Git Bash в корне репозитория.

### 1. Подготовить окружение и данные

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_win.txt
python -m pip check
bash scripts/download_stabletoolbench_cache.sh
```

### 2. Запустить виртуальный сервер

В первом терминале:

```bash
source .venv/Scripts/activate
bash scripts/run_virtual_server.sh
```

Сервер будет доступен на `http://localhost:8080/virtual`. По умолчанию используются только сохранённые ответы: внешние инструменты и LLM-симулятор не вызываются.

### 3. Запустить агента

Во втором терминале:

```bash
source .venv/Scripts/activate
read -s -p "Agent API key: " AGENT_API_KEY
echo
export AGENT_API_KEY

# Пример для DeepSeek; можно указать другой OpenAI-совместимый API.
export AGENT_API_BASE="https://api.deepseek.com"
export AGENT_MODEL="deepseek-flash"

bash inference_openai_compatible_pipeline_virtual.sh
```

Smoke-тест содержит одно задание и точный cache hit. Агент должен вызвать `chat_gpt_detector_for_ai_content_detector_v2`, получить вероятности и завершить задачу через `Finish`. Trace сохраняется в:

```text
data/answer/agent_smoke/cache_hit/900001_CoT@1.json
```

Запуск использует API агента и обычно требует двух обращений к модели. Сервер останавливается сочетанием `Ctrl+C` в первом терминале.

## Задача проекта

В исходном ToolBench агент выбирает инструмент главным образом по описанию и семантическому соответствию запросу. В этом проекте исследуется выбор среди нескольких похожих инструментов с учётом дополнительных характеристик качества обслуживания (QoS):

- доступности инструмента;
- задержки ответа;
- стоимости вызова;
- успешности предыдущих вызовов;
- стабильности результата.

Цель — проверить, позволяет ли учёт истории использования и QoS выбирать более подходящий инструмент, чем один только семантический поиск.

## Предполагаемая схема

```text
Запрос пользователя
        ↓
Формирование семантически похожих кандидатов
        ↓
Ранжирование кандидатов с учётом QoS и статистики
        ↓
Выбор и вызов инструмента
        ↓
Обновление статистики по результату вызова
```

StableToolBench используется как воспроизводимая среда с описаниями инструментов, кэшем их ответов и виртуальным API-сервером. В качестве агента можно использовать любую модель с OpenAI-совместимым API.

## Текущее состояние

- создано виртуальное окружение Python 3.11.13;
- добавлен набор зависимостей для Windows — `requirements_win.txt`;
- подключение агента обобщено для OpenAI-совместимых провайдеров;
- tool calling проверен на DeepSeek `deepseek-flash`;
- добавлен воспроизводимый скрипт загрузки StableToolBench Cache;
- выполнен сквозной cache-hit запуск: агент выбрал инструмент, получил ответ виртуального сервера и сохранил полный trace.

## Установка на Windows

Команды ниже выполняются из Git Bash в корне репозитория.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_win.txt
python -m pip check
```

Используемая версия Python записана в `.python-version`.

## Настройка и проверка LLM-агента

Ключ передаётся только через переменную окружения и не сохраняется в репозитории:

```bash
source .venv/Scripts/activate
read -s -p "Agent API key: " AGENT_API_KEY
echo
export AGENT_API_KEY
export AGENT_API_BASE="https://api.deepseek.com"
export AGENT_MODEL="deepseek-flash"

python scripts/test_agent_tool_call.py
```

Здесь DeepSeek приведён только как пример. Можно указать другой OpenAI-совместимый endpoint и модель. При успешной проверке агент должен выбрать тестовую функцию `get_weather` и сформировать её аргументы. Тест обращается к модели через штатный адаптер `ChatGPTFunction` из ToolBench.

## Настройка LLM-симулятора API

```bash
read -s -p "Simulator API key: " SIMULATOR_API_KEY
echo
export SIMULATOR_API_KEY
export SIMULATOR_API_BASE="https://another-provider.example/v1"
export SIMULATOR_MODEL="another-model"
```

Агент и симулятор — независимые роли: они могут использовать разные модели и даже разных провайдеров. Значения `SIMULATOR_*` имеют приоритет над устаревшими полями `api_key`, `api_base` и `model` в `server/config.yml`, поэтому секретный ключ не нужно сохранять в YAML.

## Загрузка StableToolBench Cache

Для запуска виртуального API-сервера нужны описания инструментов и кэш ответов. Они не хранятся в Git из-за размера.

```bash
bash scripts/download_stabletoolbench_cache.sh
```

Скрипт скачивает официальный архив StableToolBench с зафиксированной ревизии Hugging Face, безопасно распаковывает его и создаёт:

```text
server/tools/
server/tool_response_cache/
```

При наличии обеих папок повторный запуск ничего не скачивает. Другую ревизию можно указать через `STABLETOOLBENCH_CACHE_REVISION`.

## Запуск виртуального API-сервера

Для первого воспроизводимого эксперимента сервер по умолчанию работает в режиме `cache_only`: возвращает сохранённый ответ при точном совпадении входа и `Cache miss` в остальных случаях. В этом режиме он не обращается к реальному ToolBench и не вызывает LLM-симулятор.

```bash
bash scripts/run_virtual_server.sh
```

Сервер будет доступен по адресу `http://localhost:8080/virtual`. Полный режим с обращением к реальному API и последующей симуляцией включается явно:

```bash
export SERVER_MODE="full"
bash scripts/run_virtual_server.sh
```

Для основного сравнения рекомендуется отдельно фиксировать режим сервера: смешивание кэшированных, реальных и сгенерированных ответов меняет условия эксперимента.

## Ближайшие этапы

1. Сформировать контролируемый набор семантически похожих инструментов.
2. Зафиксировать baseline без QoS-ранжирования.
3. Добавить моделирование доступности, задержки и стоимости инструментов.
4. Реализовать QoS-aware ранжирование семантически похожих кандидатов.
5. Сравнить методы по качеству выбора, успешности решения, стоимости и задержке.

## Исходная документация

Оригинальный README проекта StableToolBench сохранён без удаления: [SOURCE_README.md](SOURCE_README.md).
