# Jupyter Notebook Linter

Простой инструмент для автоматического улучшения и структурирования Jupyter Notebook файлов с использованием локальной LLM модели.

## 🎯 Что делает

- Загружает `.ipynb` файлы и разбивает их на логические части
- Добавляет понятные заголовки и структурирует содержимое
- Обрамляет код подробными комментариями
- Анализирует outputs и генерирует выводы
- Возвращает готовую обработанную тетрадку в формате Markdown

## 📁 Структура

```
notebook_linter/
├── notebook_linter_module.py    # 🎯 Одна функция process_notebook
├── main.py                      # 🔧 CLI интерфейс
├── requirements.txt             # 📦 Зависимости
├── README.md                    # 📚 Документация
├── example_notebook.ipynb       # 📄 Пример входного файла
└── example_usage.ipynb          # 🎓 Пример использования
```

## 🚀 Быстрый старт

### 1. Установка
```bash
pip install -r requirements.txt
```

### 2. Использование

#### Из Python/Jupyter:
```python
from notebook_linter_module import process_notebook
from transformers import AutoTokenizer, AutoModelForCausalLM

# Загрузка модели
model_name = "./Текстовые/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# Обработка
improved_content = process_notebook(model, tokenizer, "notebook.ipynb")

# Сохранение
with open("improved_notebook.md", "w", encoding="utf-8") as f:
    f.write(improved_content)
```

#### Из командной строки:
```bash
python main.py notebook.ipynb
```

## ⚙️ Функция `process_notebook`

```python
process_notebook(
    model,                    # Загруженная модель (AutoModelForCausalLM)
    tokenizer,               # Загруженный токенизатор (AutoTokenizer)
    file_path,               # Путь к .ipynb файлу
    max_tokens=4000,         # Максимум токенов для одной части
    max_new_tokens=8192,     # Максимум новых токенов для генерации
    temperature=0.7          # Температура для генерации
) -> str                    # Возвращает обработанную тетрадку
```

## 💡 Примеры использования

### Базовая обработка:
```python
improved_content = process_notebook(model, tokenizer, "notebook.ipynb")
```

### Для больших файлов:
```python
improved_content = process_notebook(
    model, tokenizer, "large_notebook.ipynb",
    max_tokens=6000, max_new_tokens=12000
)
```

### Для более креативных результатов:
```python
improved_content = process_notebook(
    model, tokenizer, "notebook.ipynb",
    temperature=0.8
)
```

## 📊 Пример улучшения

### До:
```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
```

### После:
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

## 🔧 Требования

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

## 📄 Форматы

- **Вход**: `.ipynb` (Jupyter Notebook JSON)
- **Выход**: `.md` (Markdown)

## 🎯 Особенности

- **Автоматическое разбиение** больших файлов на логические части
- **Умная обработка** outputs и генерация выводов
- **Структурирование** с понятными заголовками
- **Комментирование** кода
- **Простота использования** - одна функция

## 📝 Лицензия

Для личного использования и обучения. 