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

## Аудит инструментов и QoS

После загрузки StableToolBench Cache единый каталог API и агрегированная статистика строятся командой:

```bash
python scripts/build_tool_catalog.py
```

Результаты сохраняются в игнорируемый Git каталог:

```text
data/catalog/tools.jsonl
data/catalog/statistics.json
```

`tools.jsonl` содержит одну нормализованную запись на API: описание инструмента и метода, параметры, тариф, исходные QoS-поля и количество доступных кэшированных входов. `statistics.json` содержит покрытие описаний, QoS и кэша, их пересечение для формирования экспериментальных кандидатов, распределения категорий, HTTP-методов и тарифов, числовые сводки QoS и найденные ошибки данных.

## Семантически похожие кандидаты

После аудита построить группы ближайших API можно командой:

```bash
python scripts/build_candidate_groups.py
```

По умолчанию используется `sentence-transformers/all-MiniLM-L6-v2`. Текст имеет версию `api_first_v2`: название и описание конкретного API и его параметры располагаются перед общим описанием инструмента и категорией, чтобы наиболее важные данные сохранялись при truncation. QoS и тариф не включаются в embedding-текст, чтобы семантический retrieval не получал информацию, предназначенную для последующего ранжирования. Дубли нормализованных API удаляются, а API того же инструмента не используются как его кандидаты; для диагностического запуска их можно вернуть флагом `--allow-same-tool`. Скрипт строит соседей как по всему каталогу, так и внутри категории и сохраняет:

```text
data/candidates/embeddings.npz
data/candidates/neighbors.jsonl
data/candidates/statistics.json
data/candidates/manual_review.csv
```

`manual_review.csv` содержит по пять соседей в каждом режиме для 50 воспроизводимо выбранных целевых API. Для обеих сторон пары сохраняются описания, параметры, точное число токенов embedding-текста и число токенов, обрезанных лимитом модели. Поле `manual_relevance_0_1_2` предназначено для ручной оценки: `0` — нерелевантный, `1` — тематически связанный, `2` — функционально взаимозаменяемый.

## Общий retrieval-пул для автоматической разметки

Современные embedding-модели устанавливаются в отдельное окружение, чтобы не обновлять зафиксированные зависимости StableToolBench:

```bash
python -m venv .venv-retrieval
source .venv-retrieval/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_retrieval.txt
```

Для NVIDIA GPU после основной установки следует заменить CPU-сборку PyTorch на зафиксированную CUDA-сборку:

```bash
python -m pip install --force-reinstall -r requirements_retrieval_cuda.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

На проверенной конфигурации используется PyTorch `2.14.0+cu130`; драйвер NVIDIA может поддерживать более новую версию CUDA, поскольку wheel содержит собственный совместимый runtime.

Общий пул кандидатов Qwen3 + BM25 строится командой:

```bash
python scripts/build_retrieval_pool.py
```

Для GPU с 4 ГБ памяти следует начать с небольшого batch:

```bash
python scripts/build_retrieval_pool.py --batch-size 2 --max-seq-length 2048
```

По умолчанию dense retriever использует `Qwen/Qwen3-Embedding-0.6B`, специальную инструкцию для поиска эквивалентных, включающих и связанных возможностей и `top-30`. Каждый API кодируется Qwen один раз; полученные нормализованные векторы используются для API-to-API поиска. BM25 независимо извлекает ещё `top-30` по названиям, описаниям и параметрам. Обратные и повторяющиеся пары объединяются, но rank, score, направление и метод-источник сохраняются.

Результаты:

```text
data/retrieval/model_indexes/   # кэш instruction-aware API embeddings
data/retrieval/catalog.jsonl   # документация API без повторения в каждой паре
data/retrieval/neighbors.jsonl # top-K каждого метода для каждого API
data/retrieval/pooled_pairs.jsonl
data/retrieval/statistics.json
```

### Анализ пула и пилотная выборка

После полного retrieval-запуска можно получить небольшую воспроизводимую выборку для проверки схемы разметки:

```bash
python scripts/analyze_retrieval_pool.py --sample-size 500 --seed 42
```

Скрипт не запускает Qwen повторно и не обращается к LLM. Он читает готовый `pooled_pairs.jsonl`, вычисляет признаки пары и поровну выбирает примеры из восьми диапазонов: взаимные и односторонние соседи Qwen с рангами `1–3`, `4–10` и `11–30`, а также пары только BM25 с рангами `1–10` и `11–30`. Такое распределение нужно для пилотной проверки разных по сложности случаев, а не для оценки естественной доли классов во всём каталоге. Пересечение Qwen и BM25 сохраняется в данных, но не является обязательным условием отбора.

Результаты сохраняются в:

```text
data/retrieval_analysis/retrieval_analysis.json # размеры слоёв, квоты и квантили score
data/retrieval_analysis/pilot_pairs.jsonl       # полные записи для последующей LLM-разметки
data/retrieval_analysis/pilot_review.csv        # плоская таблица для просмотра и ручной проверки
```

При одинаковых входных файлах и `--seed` будут выбраны те же пары. Поля `manual_relation`, `manual_direction` и `comment` в CSV намеренно оставлены пустыми.

### Пилотная LLM-разметка пар

Разметчик использует любой OpenAI-совместимый API. Настройки называются нейтрально и не привязаны к DeepSeek:

Разметчик запускается в основном окружении StableToolBench, где `openai` уже установлен. Если сейчас активно `.venv-retrieval`, сначала переключитесь:

```bash
deactivate
source .venv/Scripts/activate
```

```bash
read -s -p "Annotator API key: " ANNOTATOR_API_KEY
echo
export ANNOTATOR_API_KEY
export ANNOTATOR_API_BASE="https://api.deepseek.com"
export ANNOTATOR_MODEL="deepseek-chat"
```

Перед платным запуском можно вывести документацию первой пары и итоговый prompt без обращения к API:

```bash
python scripts/annotate_candidate_pairs.py --dry-run --limit 1
```

Первый пилотный запуск ограничен 20 парами:

```bash
python scripts/annotate_candidate_pairs.py --limit 20
```

Перед применением `--limit` пары чередуются по retrieval-слоям, поэтому небольшой пилот содержит сильные и слабые, взаимные и односторонние Qwen-соседства, а также оба диапазона BM25. Скрипт печатает фактический состав выбранных слоёв перед обращением к модели.

Для каждой пары модель выбирает одно из трёх отношений: `interchangeable`, `contains` или `different_capability`. Точные дубликаты входят в `interchangeable`. `different_capability` объединяет все случаи, где между API не нужно создавать ребро взаимозаменяемости или включения: семантически близкие, относящиеся к одной области, workflow-связанные и полностью несвязанные API. Неопределённость не является отдельным отношением: при недостаточной документации модель выбирает наиболее вероятный класс, снижает `confidence` и устанавливает `needs_human_review=true`. Степень текстовой близости уже хранится в retrieval rank/score. Дополнительно сохраняются направление `contains`, объяснение и признак необходимости ручной проверки.

`interchangeable` требует двусторонней заменяемости: аргументы обеих схем должны получаться из запроса пользователя и друг из друга детерминированным локальным преобразованием. Внешний lookup, geocoding, реестр активов, преобразование provider-specific имени во внутренний ID или доступ к состоянию другой системы запрещают такое объединение. Одинаково документированные login/order/inventory/account endpoints разных stateful-систем также не считаются взаимозаменяемыми. Совпадающий boilerplate, Swagger/OpenAPI sample, схема, путь и примеры не доказывают общий backend: для stateful API требуется явное указание на общий deployment или namespace. `contains` означает функционально более широкий API в любом документированном смысле при общей базовой задаче или ресурсе: больше данных, сценариев, период, детализация, дополнительные поля, режимы входа либо операции. Точное воспроизведение узкого ответа и наличие тех же фильтров не требуются; направление всегда указывает от более широкой возможности к более узкой.

Parameter mappings не создаются и не проверяются вручную. После выбора API исполняющая LLM получает его нативную документацию и самостоятельно формирует аргументы вызова. Workflow-граф и передача идентификаторов также исключены из текущего этапа: необходимые API позже подаются агенту явно.

Успешные ответы немедленно дописываются в `data/annotations/pilot_pair_annotations.jsonl`, а ошибки — в `data/annotations/pilot_pair_annotation_errors.jsonl`. Повтор той же команды пропускает уже успешно обработанные `pair_id`, поэтому прерванный запуск можно безопасно продолжить. Retrieval-score модели не передаются: решение принимается только по документации API.

После общего пилота можно построить отдельную диагностическую выборку сильных и пограничных отношений:

```bash
python scripts/build_relation_audit_sample.py
python scripts/annotate_candidate_pairs.py \
  --input data/retrieval_analysis/relation_audit_pairs.jsonl \
  --limit 20
```

Отбор создаёт по пять ещё не размеченных кандидатов для аудита дубликатов, взаимозаменяемости, включения и сложных границ классов. `audit_sampling.target` является названием эвристического набора, а не истинной меткой. Эти подсказки не передаются LLM: модель получает только документацию двух API.

### Последовательное построение графа с ограниченным бюджетом

После ручной проверки пилота граф расширяется в порядке убывания Qwen-score. Ограничение задаётся на фактические HTTP-запросы к разметчику, а не на количество рассмотренных пар:

```bash
python scripts/expand_relation_graph.py --max-llm-calls 1000 --dry-run
python scripts/expand_relation_graph.py --max-llm-calls 1000
```

Скрипт использует последние решения из `human_pair_reviews.jsonl` как начальные рёбра. `interchangeable` объединяет API через Union-Find; известные отношения внутри компоненты, `contains` между компонентами и `different_capability` распространяются без нового обращения к LLM. Поэтому при бюджете 1000 запросов фактически обработанных пар может быть больше. Для каждого прямого и выведенного решения сохраняются Qwen-score, источник решения и идентификаторы опорных рёбер. Повторный запуск восстанавливает граф по прямым решениям, пропускает уже обработанные пары и учитывает запросы из успешных ответов и журнала ошибок, поэтому общий лимит сохраняется между запусками.

Progress bar показывает общий расход HTTP-запросов, число рассмотренных и выведенных пар, текущее количество Qwen-пар, покрытых графом без LLM (`covered`), уже реально сэкономленные вызовы (`saved`) и прирост покрытия после последнего прямого решения (`delta`). Эти же значения сохраняются в `candidate_coverage_impact` каждого нового LLM-решения и в `graph_candidate_coverage` checkpoint.

Результаты:

```text
data/relation_graph/pair_decisions.jsonl
data/relation_graph/errors.jsonl
data/relation_graph/checkpoint.json
```

API и пары, до которых обработка не дошла в пределах бюджета, остаются отложенными: им не присваивается `different_capability`.

Промежуточный snapshot эксперимента `v16_migrated` (автоматическая разметка, не gold standard): 6000 запросов к LLM-разметчику дали 7003 прямых и выведенных решения. Для 6470 API построено 6142 компоненты взаимозаменяемости, из них 209 содержат более одного API; крупнейшая компонента содержит 18 API. Зафиксировано 633 прямых отношения `contains`, а граф позволил покрыть без дополнительного LLM-вызова 1721 пару из текущего retrieval-пула. Сырые результаты и кэши находятся в игнорируемом каталоге `data/`; числа приведены как ориентир и могут меняться при продолжении запуска или изменении prompt.

Для чистого повторного эксперимента без ручных seed-решений и без смешивания с предыдущим prompt результаты записываются в отдельную папку:

```bash
python scripts/expand_relation_graph.py \
  --no-human-seeds \
  --run-dir data/relation_graph/v13_no_human \
  --max-llm-calls 200 \
  --dry-run

python scripts/expand_relation_graph.py \
  --no-human-seeds \
  --run-dir data/relation_graph/v13_no_human \
  --max-llm-calls 200
```

Существующие результаты других запусков при этом не читаются и не перезаписываются.

После изменения схемы точный набор ранее проверявшихся 80 пар можно полностью переразметить одной командой:

```bash
python scripts/annotate_candidate_pairs.py \
  --input data/retrieval_analysis/reannotation_80_pairs.jsonl \
  --limit 0
```

Предыдущие результаты v1–v7 сохранены в `data/annotations/archive/pair-relations-v1-v7_2026-09-24/`, а частичный запуск v8 — в `data/annotations/archive/pair-relations-v8_partial_2026-09-24/`. Они не участвуют в новом запуске.

Таблица для ручного аудита всех накопленных аннотаций строится без вызовов LLM:

```bash
python scripts/build_annotation_review.py
```

Результат `data/annotations/pilot_annotation_review.csv` объединяет документацию обоих API, retrieval-слой, версию prompt и решение LLM. Диагностические пары располагаются первыми. Человек заполняет `human_relation`, `human_direction`, автоматически вычисляемое `human_relation_correct` и `human_comment`. Скрипт сохраняет уже заполненные ручные поля при пересборке CSV. Исходный JSONL остаётся неизменяемым источником автоматической разметки.

Для удобной проверки через браузер используется отдельное окружение, чтобы современные зависимости Streamlit не конфликтовали со старым стеком StableToolBench. Первоначальная установка:

```bash
python -m venv .venv-review
source .venv-review/Scripts/activate
python -m pip install -r requirements_review.txt
python -m streamlit run apps/review_annotations.py
```

В этом репозитории `.venv-review` уже создано. Для последующих запусков достаточно:

```bash
source .venv-review/Scripts/activate
python -m streamlit run apps/review_annotations.py
```

Интерфейс показывает документацию LEFT и RIGHT рядом, параметры, решение и объяснение модели. Доступны фильтры по статусу, версии prompt, классу и диагностической выборке. Кнопка сохранения добавляет событие в `data/annotations/human_pair_reviews.jsonl` и переводит к следующей паре. Файл append-only хранит историю исправлений; актуальным считается последнее решение для `pair_id`. При следующей сборке CSV эти решения подставляются автоматически.

`pooled_pairs.jsonl` является входом следующего этапа — пакетной LLM-классификации отношений между API. QoS не включается в тексты и не показывается аннотатору, чтобы доступность, популярность или задержка не влияли на решение о функциональной связи.

Быстрая проверка без скачивания embedding-модели:

```bash
python scripts/build_retrieval_pool.py --skip-dense --limit 100 --top-k 5
```

На Windows по умолчанию используется один поток поиска. После проверки окружения параллелизм можно включить явно, например `--n-jobs 4`.

## Канонические функциональные кластеры

Семантически близкие API планируется объединять в функциональные кластеры, не удаляя конкретные реализации. Кластер описывает общую возможность, а его участники сохраняют собственные параметры, QoS и инфраструктурные ограничения. Это позволяет сначала найти требуемую функцию, а затем выбрать внутри группы конкретный API с учётом доступности, успешности, задержки и стоимости.

Планируемая структура кластера:

```json
{
  "cluster_id": "get_current_weather",
  "canonical_name": "Get current weather",
  "canonical_description": "Returns current weather conditions for a location.",
  "use_when": [
    "The user asks about weather right now",
    "The user asks for the current temperature or conditions"
  ],
  "do_not_use_when": [
    "The user asks for a future weather forecast",
    "The user asks for historical weather",
    "The user asks for long-term climate statistics"
  ],
  "required_information": ["location"],
  "optional_information": ["units", "language"],
  "canonical_parameters": [
    {
      "name": "location",
      "type": "string",
      "required": true,
      "description": "City name or geographic location"
    }
  ],
  "returns": {
    "description": "Current weather conditions for the requested location",
    "fields": []
  },
  "confusable_with": [
    {
      "cluster_id": "weather_forecast",
      "difference": "Use weather_forecast only for future conditions."
    },
    {
      "cluster_id": "historical_weather",
      "difference": "Use historical_weather only for past dates or periods."
    }
  ],
  "members": [
    {
      "api_id": "provider_current_weather",
      "provider": "provider_name"
    }
  ],
  "annotation": {
    "source": "llm",
    "model": "model-name",
    "prompt_version": "cluster-v1",
    "confidence": 0.0,
    "human_status": "unreviewed"
  }
}
```

`use_when`, `do_not_use_when` и `confusable_with` образуют контрастивную документацию: она не только описывает функцию, но и отделяет её от семантически близких возможностей. Эквивалентные API входят в один функциональный кластер. После выбора конкретного участника исполняющая LLM получает его исходное описание и нативную схему параметров из каталога и самостоятельно формирует вызов.

## План разработки

### 1. Зафиксировать входной каталог и протокол эксперимента

- Версионировать схему нормализованного API, правила фильтрации и fingerprint каталога.
- Сформировать воспроизводимую выборку целевых API и отдельный human-verified test set.
- Не использовать тестовую разметку для настройки порогов, prompt и правил кластеризации.

### 2. Построить общий пул кандидатов

- Получить top-30 соседей с помощью `Qwen3-Embedding-0.6B`, `gte-large-en-v1.5` и `all-MiniLM-L6-v2`.
- Добавить BM25 и точные совпадения имён/сигнатур.
- Объединить результаты моделей в уникальные неориентированные пары и сохранить rank/score каждого источника.
- Вынести современные retrieval-зависимости в отдельное окружение, чтобы не конфликтовать с зависимостями StableToolBench.

### 3. Выполнить LLM-разметку пар

- Классифицировать пары как `different_capability`, `contains` или `interchangeable`; точные дубликаты считать частным случаем `interchangeable`.
- Сохранять структурированный ответ, confidence, объяснение, модель и версию prompt.
- Поддержать нейтральную OpenAI-compatible конфигурацию LLM через переменные окружения.

### 4. Построить и проверить функциональные кластеры

- Построить граф взаимозаменяемости по отношению `interchangeable` и отдельно учитывать направленные рёбра `contains`.
- Получить первоначальные компоненты связности.
- Выявлять нетранзитивные тройки, bridge-рёбра, слишком крупные группы и несовместимые сигнатуры.
- Выполнять LLM-аудит каждого подозрительного кластера и при необходимости разделять его на подкластеры.
- Не удалять конкретные API: кластер используется как функциональная группа для последующего выбора реализации.

### 5. Сгенерировать каноническую документацию

- Сформировать `canonical_name`, `canonical_description`, `use_when`, `do_not_use_when` и `confusable_with`.
- Сохранить нативные схемы параметров участников: исполняющая LLM сама формирует аргументы выбранного API.
- Индексировать отдельно канонический документ кластера и технические данные его участников.

### 6. Сделать интерфейс ручной проверки

- Показывать кластер целиком, его участников, соседние группы и причины решений LLM.
- Поддержать принятие, разделение, перенос или исключение API и исправление канонической документации.
- В первую очередь показывать человеку низкую уверенность, разногласия моделей, bridge-рёбра, нетранзитивность и инфраструктурные конфликты.
- Дополнительно проверять случайную долю уверенно принятых решений для оценки ошибок автоматической разметки.
- Хранить provenance всех автоматических и ручных изменений.

### 7. Реализовать retrieval функциональных групп

- Сравнить поиск по исходной документации, каноническому описанию и полной контрастивной документации.
- Реализовать BM25, dense retrieval и их гибрид.
- Добавить cross-encoder reranking кандидатов.
- Для первого эксперимента ограничиться single-tool запросами; multi-tool decomposition добавить отдельным этапом.

### 8. Реализовать выбор API внутри группы

- Зафиксировать baseline: случайный выбор, cosine top-1 и выбор по средней исторической успешности.
- Реализовать статический QoS-reranker по релевантности, доступности, успешности, задержке и стоимости.
- Реализовать contextual bandit, обновляющий оценки после наблюдения результата вызова.
- Добавить воспроизводимый симулятор стационарных, изменяющихся и контекстно-зависимых QoS.

#### Подготовка QoS-профилей

После экспорта актуального графа QoS-профили конкретных API строятся отдельно от функциональных кластеров:

```bash
python scripts/export_relation_graph.py \
  --run-dir data/relation_graph/v16_migrated

python scripts/build_qos_profiles.py \
  --api-to-cluster data/relation_graph/v16_migrated/api_to_cluster.jsonl \
  --output-dir data/qos/v2 \
  --monetization-config configs/qos_monetization_v1.json \
  --seed 42
```

`data/qos/v2/api_qos_profiles.jsonl` содержит исходные показатели StableToolBench, нормализованные вероятности доступности и успеха, ожидаемую задержку, условную стоимость и профили `normal`, `degraded`, `outage`. Исходные `avgLatency`, `avgServiceLevel`, `avgSuccessRate` и `popularityScore` заданы в StableToolBench на уровне инструмента и поэтому повторяются для его API. Сценарии деградации, latency jitter и монетизация являются параметрами воспроизводимого эксперимента, а не наблюдаемыми денежными ценами.

Монетизация задаётся отдельно в `configs/qos_monetization_v1.json`. Для каждого тарифа детерминированно выбирается один шаблон: бесплатная квота, free-then-pay-as-you-go, подписка с включёнными вызовами и overage либо предоплаченный пакет. Политика хранит `billing_period_steps`, `included_calls`, `upfront_cost_units`, `overage_cost_per_call_units`, `hard_call_limit`, `requests_per_minute`, `concurrent_requests`, срок сгорания пакета и spending cap. Поэтому реальная стоимость следующего вызова является состоянием симулятора, а `reference_cost_per_call_units` используется только для статического сравнения. Конфигурация, формулы и seed сохраняются в `data/qos/v2/metadata.json`.

### 9. Провести сравнение

- Для candidate generation измерить Precision@K, Recall@K, MRR и nDCG@K.
- Для кластеризации измерить pairwise precision/recall/F1 и B-cubed F1.
- Для выбора API измерить Success@1, среднюю награду, latency, стоимость, error rate и cumulative regret.
- Сравнить embedding-модели, варианты документации, retriever/reranker и два алгоритма выбора при одинаковых данных и seed.

## Исходная документация

Оригинальный README проекта StableToolBench сохранён без удаления: [SOURCE_README.md](SOURCE_README.md).
