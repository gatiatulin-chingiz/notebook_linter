# Jupyter Notebook Linter

Этот проект представляет собой инструмент для автоматического улучшения и структурирования Jupyter Notebook файлов с использованием локальной LLM модели.

## 🎯 Возможности

- **Загрузка Jupyter Notebook**: Чтение `.ipynb` файлов в формате JSON
- **Автоматическое разбиение**: Разделение больших тетрадок на логические части для обработки моделью
- **Структурирование**: Добавление понятных заголовков и разделов
- **Комментирование кода**: Обрамление кода подробными комментариями
- **Анализ outputs**: Генерация выводов на основе результатов выполнения ячеек
- **Улучшение читаемости**: Создание более понятной и структурированной версии тетрадки

## 📁 Структура проекта

```
notebook_linter/
├── notebook_linter_module.py    # 🎯 ГЛАВНЫЙ МОДУЛЬ - основная библиотека
├── main.py                      # 🔧 Простой CLI интерфейс
├── requirements.txt             # 📦 Зависимости проекта
├── README.md                    # 📚 Основная документация
├── example_notebook.ipynb       # 📄 Пример входного файла
└── example_usage.ipynb          # 🎓 Пример использования в Jupyter
```

## 🚀 Установка

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Подготовка модели

Убедитесь, что у вас есть локальная LLM модель (например, Qwen3-0.6B) в папке `./Текстовые/Qwen3-0.6B/`

## 📖 Использование

### Способ 1: Из Jupyter Notebook (рекомендуется)

#### Быстрый старт:
```python
# Импорт модуля
from notebook_linter_module import *

# Инициализация линтера
init_linter()

# Обработка notebook
improved_content = lint_notebook("your_notebook.ipynb")
```

#### Подробное использование:

```python
# 1. Импорт
from notebook_linter_module import *

# 2. Инициализация с настройками
linter = init_linter(
    model_name="./Текстовые/Qwen3-0.6B",
    max_tokens=4000,
    max_new_tokens=8192,
    temperature=0.7
)

# 3. Анализ структуры notebook
display_notebook_info("example_notebook.ipynb")

# 4. Обработка с отображением результата
improved_content = lint_notebook("example_notebook.ipynb")

# 5. Быстрая обработка (автоматическая инициализация)
quick_result = quick_lint("example_notebook.ipynb")
```

#### Дополнительные функции:

```python
# Анализ структуры
info = analyze_notebook_structure("file.ipynb")
print(f"Ячеек: {info['total_cells']}")

# Обработка без отображения результата
result = lint_notebook("file.ipynb", display_result=False)

# Указание выходного файла
result = lint_notebook("file.ipynb", "custom_output.md")

# Отображение результата как Markdown
display_result_as_markdown(result)

# Отображение результата как HTML
display_result_as_html(result)

# Сохранение в файл
save_result_to_file(result, "custom_output.md")
```

#### Пакетная обработка:
```python
import os

# Обработка всех notebook файлов в папке
for file in os.listdir('.'):
    if file.endswith('.ipynb'):
        print(f"Обрабатываю {file}...")
        lint_notebook(file, f"{file}_improved.md")
```

### Способ 2: Через командную строку

#### Базовый запуск:
```bash
python main.py your_notebook.ipynb
```

#### Интерактивный режим:
```bash
python main.py
# Введите путь к файлу при запросе
```

### Способ 3: Программное использование

#### Из Python скрипта:
```python
from notebook_linter_module import NotebookLinter

# Инициализация линтера
linter = NotebookLinter(model_name="./Текстовые/Qwen3-0.6B")

# Обработка тетрадки
improved_content = linter.process_notebook("path/to/your/notebook.ipynb")

# Сохранение результата
linter.save_improved_notebook(improved_content, "improved_notebook.md")
```

## ⚙️ Настройки

### Параметры модели

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `model_name` | `"./Текстовые/Qwen3-0.6B"` | Путь к локальной модели |
| `max_tokens` | `4000` | Максимум токенов для одной части |
| `max_new_tokens` | `8192` | Максимум новых токенов для генерации |
| `temperature` | `0.7` | Температура для генерации |

### Параметры функций

| Функция | Параметры | Описание |
|---------|-----------|----------|
| `lint_notebook()` | `file_path, output_path, display_result` | Обработка notebook файла |
| `quick_lint()` | `file_path, model_name` | Быстрая обработка |
| `display_notebook_info()` | `file_path` | Анализ структуры |
| `lint_current_notebook()` | `output_path` | Обработка текущей тетрадки |

## 🎯 Основные функции

### `init_linter(model_name, max_tokens, max_new_tokens, temperature)`
Инициализация линтера с указанными параметрами.

### `lint_notebook(file_path, output_path, display_result)`
Обработка notebook файла с возможностью сохранения и отображения результата.

### `quick_lint(file_path, model_name)`
Быстрая обработка с автоматической инициализацией линтера.

### `display_notebook_info(file_path)`
Анализ и отображение структуры notebook в виде таблицы.

### `lint_current_notebook(output_path)`
Попытка обработать текущую тетрадку (если возможно определить путь).

### `analyze_notebook_structure(file_path)`
Получение информации о структуре notebook как словарь.

### `display_result_as_markdown(content)`
Отображение результата как Markdown.

### `display_result_as_html(content)`
Отображение результата как HTML.

### `save_result_to_file(content, file_path)`
Сохранение результата в файл.

## 💡 Советы по использованию

### Для больших файлов:
```python
init_linter(max_tokens=6000, max_new_tokens=12000)
```

### Для более креативных результатов:
```python
init_linter(temperature=0.8)
```

### Для более длинных ответов:
```python
init_linter(max_new_tokens=16000)
```

### Для экономии памяти:
```python
init_linter(max_tokens=2000)
```

## 🔧 Обработка ошибок

Модуль автоматически обрабатывает:
- Ошибки загрузки файлов
- Проблемы с кодировкой
- Ошибки в outputs ячеек
- Проблемы с токенизацией

При возникновении ошибок проверьте:
- Существование файла
- Правильность пути к модели
- Достаточность памяти для модели

## 📊 Примеры улучшений

### До обработки:
```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
```

### После обработки:
```python
# Импорт библиотеки pandas для работы с данными
import pandas as pd

# Загрузка данных из CSV файла
df = pd.read_csv('data.csv')

# Вывод первых 5 строк для предварительного просмотра данных
print(df.head())

# Вывод: Первые 5 строк загруженного датасета показывают структуру данных
# и позволяют оценить качество и формат загруженной информации
```

## 🎓 Интеграция с Jupyter

Модуль специально разработан для использования в Jupyter Notebook:
- Автоматическое отображение результатов
- Интеграция с IPython.display
- Поддержка Markdown и HTML вывода
- Красивые таблицы для анализа структуры
- Глобальное состояние линтера

## 📋 Требования

- Python 3.7+
- transformers
- torch
- pandas
- matplotlib
- ipython
- jupyter
- notebook
- requests
- Локальная LLM модель (например, Qwen3-0.6B)

## 📄 Поддерживаемые форматы

- **Входные файлы**: `.ipynb` (Jupyter Notebook JSON)
- **Выходные файлы**: `.md` (Markdown)

## 🤔 Зачем нужен main.py?

`main.py` выполняет роль **простого CLI интерфейса**:

### ✅ Преимущества:
1. **Простота использования**: `python main.py file.ipynb`
2. **Интерактивный режим**: `python main.py`
3. **Минимальные зависимости**: Не требует знания Python
4. **Обратная совместимость**: Сохраняет возможность использования как раньше
5. **Быстрое тестирование**: Удобно для быстрой проверки работы линтера

### 🔄 Что изменилось:
- **Раньше**: `main.py` содержал весь код линтера (221 строка)
- **Теперь**: `main.py` - простой CLI, который использует `notebook_linter_module.py` (80 строк)

## 📈 Обработка больших файлов

Если ваш Jupyter Notebook слишком большой для обработки моделью целиком, система автоматически:

1. Разбивает тетрадку на логические части
2. Обрабатывает каждую часть отдельно
3. Объединяет результаты в финальный документ

Логическое разбиение учитывает:
- Количество токенов в ячейках
- Тип ячеек (markdown/code)
- Связность содержимого

## 🎯 Формат выходного файла

Результат сохраняется в формате Markdown (`.md`) с:
- Структурированными заголовками
- Комментированным кодом
- Анализом outputs
- Логическими разделами

## 📝 Лицензия

Этот проект предназначен для личного использования и обучения. 