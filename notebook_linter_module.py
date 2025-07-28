"""
Jupyter Notebook Linter Module
Модуль для использования линтера прямо из Jupyter Notebook
"""

import json
import os
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from IPython.display import display, HTML, Markdown
import pandas as pd

class NotebookLinter:
    def __init__(self, model_name: str = "./Текстовые/Qwen3-0.6B", max_tokens: int = 4000, 
                 max_new_tokens: int = 8192, temperature: float = 0.7):
        """Инициализация модели для обработки Jupyter Notebook"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
    def load_notebook(self, file_path: str) -> Dict[str, Any]:
        """Загрузка Jupyter Notebook из JSON файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            return notebook
        except Exception as e:
            raise Exception(f"Ошибка загрузки файла {file_path}: {str(e)}")
    
    def extract_cell_content(self, cell: Dict[str, Any]) -> str:
        """Извлечение содержимого ячейки"""
        cell_type = cell.get('cell_type', '')
        source = cell.get('source', [])
        
        if isinstance(source, list):
            content = ''.join(source)
        else:
            content = str(source)
            
        if cell_type == 'code':
            outputs = cell.get('outputs', [])
            output_text = self._extract_outputs(outputs)
            return f"```python\n{content}\n```\n\nOutputs:\n{output_text}"
        else:
            return content
    
    def _extract_outputs(self, outputs: List[Dict[str, Any]]) -> str:
        """Извлечение outputs из ячейки кода"""
        output_texts = []
        for output in outputs:
            output_type = output.get('output_type', '')
            
            if output_type == 'stream':
                text = ''.join(output.get('text', []))
                output_texts.append(f"Stream: {text}")
            elif output_type == 'execute_result':
                data = output.get('data', {})
                if 'text/plain' in data:
                    text = ''.join(data['text/plain'])
                    output_texts.append(f"Result: {text}")
            elif output_type == 'error':
                error_name = output.get('ename', '')
                error_value = output.get('evalue', '')
                output_texts.append(f"Error: {error_name}: {error_value}")
        
        return '\n'.join(output_texts) if output_texts else "No outputs"
    
    def split_notebook_into_chunks(self, notebook: Dict[str, Any]) -> List[str]:
        """Разбиение тетрадки на логические части"""
        cells = notebook.get('cells', [])
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, cell in enumerate(cells):
            cell_content = self.extract_cell_content(cell)
            cell_tokens = len(self.tokenizer.encode(cell_content))
            
            if current_tokens + cell_tokens > self.max_tokens and current_chunk:
                chunk_text = self._format_chunk(current_chunk, i - len(current_chunk))
                chunks.append(chunk_text)
                
                current_chunk = [cell]
                current_tokens = cell_tokens
            else:
                current_chunk.append(cell)
                current_tokens += cell_tokens
        
        if current_chunk:
            chunk_text = self._format_chunk(current_chunk, len(cells) - len(current_chunk))
            chunks.append(chunk_text)
        
        return chunks
    
    def _format_chunk(self, cells: List[Dict[str, Any]], start_index: int) -> str:
        """Форматирование части тетрадки"""
        chunk_text = f"# Часть тетрадки (ячейки {start_index + 1}-{start_index + len(cells)})\n\n"
        
        for i, cell in enumerate(cells, start_index + 1):
            cell_type = cell.get('cell_type', '')
            content = self.extract_cell_content(cell)
            
            if cell_type == 'markdown':
                chunk_text += f"## Ячейка {i} (Markdown)\n{content}\n\n"
            else:
                chunk_text += f"## Ячейка {i} (Code)\n{content}\n\n"
        
        return chunk_text
    
    def generate_improved_notebook(self, chunk: str) -> str:
        """Генерация улучшенной версии части тетрадки"""
        prompt = f"""Ты - эксперт по структурированию Jupyter Notebook. Проанализируй данную часть тетрадки и улучши её:

1. Добавь понятные заголовки для разделов
2. Обрами код подробными комментариями
3. На основе outputs напиши выводы и объяснения
4. Структурируй содержимое логически
5. Сделай тетрадку более читаемой и понятной

Часть тетрадки:
{chunk}

Улучшенная версия:"""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content
    
    def process_notebook(self, file_path: str) -> str:
        """Основной метод обработки тетрадки"""
        print(f"Загружаю тетрадку: {file_path}")
        notebook = self.load_notebook(file_path)
        
        print("Разбиваю тетрадку на части...")
        chunks = self.split_notebook_into_chunks(notebook)
        
        print(f"Тетрадка разбита на {len(chunks)} частей")
        
        improved_parts = []
        for i, chunk in enumerate(chunks, 1):
            print(f"Обрабатываю часть {i}/{len(chunks)}...")
            improved_chunk = self.generate_improved_notebook(chunk)
            improved_parts.append(improved_chunk)
        
        final_notebook = "\n\n" + "="*80 + "\n\n".join(improved_parts)
        return final_notebook
    
    def save_improved_notebook(self, improved_content: str, output_path: str):
        """Сохранение улучшенной тетрадки"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(improved_content)
        print(f"Улучшенная тетрадка сохранена в: {output_path}")

# Глобальная переменная для хранения экземпляра линтера
_linter_instance = None

def init_linter(model_name: str = "./Текстовые/Qwen3-0.6B", 
                max_tokens: int = 4000, 
                max_new_tokens: int = 8192, 
                temperature: float = 0.7) -> NotebookLinter:
    """
    Инициализация линтера
    
    Args:
        model_name: Путь к модели
        max_tokens: Максимальное количество токенов для одной части
        max_new_tokens: Максимальное количество новых токенов для генерации
        temperature: Температура для генерации
    
    Returns:
        Экземпляр NotebookLinter
    """
    global _linter_instance
    
    print("🔄 Инициализация Jupyter Notebook Linter...")
    print(f"📁 Модель: {model_name}")
    print(f"⚙️  Максимум токенов на часть: {max_tokens}")
    print(f"⚙️  Максимум новых токенов: {max_new_tokens}")
    print(f"⚙️  Температура: {temperature}")
    
    _linter_instance = NotebookLinter(
        model_name=model_name,
        max_tokens=max_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )
    
    print("✅ Линтер инициализирован успешно!")
    return _linter_instance

def get_linter() -> Optional[NotebookLinter]:
    """Получение текущего экземпляра линтера"""
    global _linter_instance
    if _linter_instance is None:
        print("⚠️  Линтер не инициализирован. Используйте init_linter()")
        return None
    return _linter_instance

def lint_notebook(file_path: str, output_path: Optional[str] = None, 
                  display_result: bool = True) -> str:
    """
    Обработка Jupyter Notebook файла
    
    Args:
        file_path: Путь к входному файлу .ipynb
        output_path: Путь для сохранения результата (опционально)
        display_result: Показывать ли результат в ячейке
    
    Returns:
        Улучшенное содержимое тетрадки
    """
    linter = get_linter()
    if linter is None:
        return ""
    
    try:
        # Обработка тетрадки
        improved_content = linter.process_notebook(file_path)
        
        # Сохранение результата
        if output_path is None:
            base_name = os.path.splitext(file_path)[0]
            output_path = f"{base_name}_improved.md"
        
        linter.save_improved_notebook(improved_content, output_path)
        
        # Отображение результата
        if display_result:
            print("\n" + "="*60)
            print("📄 РЕЗУЛЬТАТ ОБРАБОТКИ")
            print("="*60)
            display(Markdown(improved_content))
        
        return improved_content
        
    except Exception as e:
        print(f"❌ Ошибка при обработке: {str(e)}")
        return ""

def analyze_notebook_structure(file_path: str) -> Dict[str, Any]:
    """
    Анализ структуры Jupyter Notebook
    
    Args:
        file_path: Путь к файлу .ipynb
    
    Returns:
        Словарь с информацией о структуре
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        cells = notebook.get('cells', [])
        
        # Статистика по типам ячеек
        cell_types = {}
        total_code_cells = 0
        total_markdown_cells = 0
        
        for cell in cells:
            cell_type = cell.get('cell_type', 'unknown')
            cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
            
            if cell_type == 'code':
                total_code_cells += 1
                outputs = cell.get('outputs', [])
                if outputs:
                    total_markdown_cells += 1
        
        # Анализ размеров
        total_size = os.path.getsize(file_path)
        
        structure_info = {
            'total_cells': len(cells),
            'cell_types': cell_types,
            'code_cells': total_code_cells,
            'markdown_cells': total_markdown_cells,
            'cells_with_outputs': total_markdown_cells,
            'file_size_kb': total_size / 1024,
            'estimated_tokens': len(json.dumps(notebook)) // 4  # Примерная оценка
        }
        
        return structure_info
        
    except Exception as e:
        print(f"❌ Ошибка при анализе структуры: {str(e)}")
        return {}

def display_notebook_info(file_path: str):
    """
    Отображение информации о Jupyter Notebook
    
    Args:
        file_path: Путь к файлу .ipynb
    """
    info = analyze_notebook_structure(file_path)
    
    if not info:
        return
    
    print("📊 АНАЛИЗ СТРУКТУРЫ NOTEBOOK")
    print("="*40)
    print(f"📁 Файл: {file_path}")
    print(f"📄 Общее количество ячеек: {info['total_cells']}")
    print(f"💻 Ячейки кода: {info['code_cells']}")
    print(f"📝 Markdown ячейки: {info['markdown_cells']}")
    print(f"📊 Ячейки с выводами: {info['cells_with_outputs']}")
    print(f"📏 Размер файла: {info['file_size_kb']:.1f} KB")
    print(f"🔤 Примерное количество токенов: {info['estimated_tokens']}")
    
    # Создаем DataFrame для красивого отображения
    df_info = pd.DataFrame([
        ['Общее количество ячеек', info['total_cells']],
        ['Ячейки кода', info['code_cells']],
        ['Markdown ячейки', info['markdown_cells']],
        ['Ячейки с выводами', info['cells_with_outputs']],
        ['Размер файла (KB)', f"{info['file_size_kb']:.1f}"],
        ['Примерное количество токенов', info['estimated_tokens']]
    ], columns=['Параметр', 'Значение'])
    
    display(df_info)

def quick_lint(file_path: str, model_name: str = "./Текстовые/Qwen3-0.6B") -> str:
    """
    Быстрая обработка тетрадки с автоматической инициализацией
    
    Args:
        file_path: Путь к файлу .ipynb
        model_name: Путь к модели
    
    Returns:
        Улучшенное содержимое тетрадки
    """
    print("🚀 Быстрая обработка тетрадки...")
    
    # Инициализация линтера
    init_linter(model_name=model_name)
    
    # Обработка
    return lint_notebook(file_path)

# Функции для работы с текущей тетрадкой
def get_current_notebook_path() -> Optional[str]:
    """Получение пути к текущей тетрадке (если возможно)"""
    try:
        # Попытка получить путь из переменных окружения Jupyter
        import ipykernel
        import requests
        from notebook.notebookapp import list_running_servers
        
        servers = list_running_servers()
        if servers:
            server = servers[0]
            url = f"{server['url']}api/sessions"
            response = requests.get(url, params={'token': server.get('token', '')})
            if response.status_code == 200:
                sessions = response.json()
                for session in sessions:
                    if session['kernel']['id'] == ipykernel.get_connection_file():
                        return session['notebook']['path']
    except:
        pass
    
    return None

def lint_current_notebook(output_path: Optional[str] = None) -> str:
    """
    Обработка текущей тетрадки (если возможно определить путь)
    
    Args:
        output_path: Путь для сохранения результата
    
    Returns:
        Улучшенное содержимое тетрадки
    """
    current_path = get_current_notebook_path()
    
    if current_path is None:
        print("❌ Не удалось определить путь к текущей тетрадке")
        print("Используйте lint_notebook() с явным указанием пути")
        return ""
    
    print(f"📁 Обрабатываю текущую тетрадку: {current_path}")
    return lint_notebook(current_path, output_path)

# Утилиты для работы с результатами
def save_result_to_file(content: str, file_path: str):
    """Сохранение результата в файл"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Результат сохранен в: {file_path}")

def display_result_as_markdown(content: str):
    """Отображение результата как Markdown"""
    display(Markdown(content))

def display_result_as_html(content: str):
    """Отображение результата как HTML"""
    html_content = f"""
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6;">
        <h3 style="color: #495057; margin-top: 0;">Результат обработки</h3>
        <div style="background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #ced4da;">
            {content.replace(chr(10), '<br>')}
        </div>
    </div>
    """
    display(HTML(html_content)) 