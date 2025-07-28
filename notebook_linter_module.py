"""
Jupyter Notebook Linter Module
Упрощенный модуль с одной основной функцией для обработки Jupyter Notebook
"""

import json
import os
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM

def process_notebook(model, tokenizer, file_path: str, max_tokens: int = 4000, 
                    max_new_tokens: int = 8192, temperature: float = 0.7) -> Dict[str, Any]:
    """
    Обработка Jupyter Notebook файла
    
    Args:
        model: Загруженная модель (например, AutoModelForCausalLM)
        tokenizer: Загруженный токенизатор (например, AutoTokenizer)
        file_path: Путь к .ipynb файлу
        max_tokens: Максимальное количество токенов для одной части
        max_new_tokens: Максимальное количество новых токенов для генерации
        temperature: Температура для генерации
    
    Returns:
        Dict[str, Any]: Обработанная тетрадка в формате .ipynb (JSON структура)
    """
    
    def load_notebook_from_file(file_path: str) -> Dict[str, Any]:
        """Загрузка Jupyter Notebook из .ipynb файла"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            return notebook
        except Exception as e:
            raise Exception(f"Ошибка загрузки файла {file_path}: {str(e)}")
    
    def extract_cell_content(cell: Dict[str, Any]) -> str:
        """Извлечение содержимого ячейки"""
        cell_type = cell.get('cell_type', '')
        source = cell.get('source', [])
        
        if isinstance(source, list):
            content = ''.join(source)
        else:
            content = str(source)
            
        if cell_type == 'code':
            outputs = cell.get('outputs', [])
            output_text = extract_outputs(outputs)
            return f"```python\n{content}\n```\n\nOutputs:\n{output_text}"
        else:
            return content
    
    def extract_outputs(outputs: List[Dict[str, Any]]) -> str:
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
    
    def split_notebook_into_chunks(notebook: Dict[str, Any]) -> List[str]:
        """Разбиение тетрадки на логические части"""
        cells = notebook.get('cells', [])
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, cell in enumerate(cells):
            cell_content = extract_cell_content(cell)
            cell_tokens = len(tokenizer.encode(cell_content))
            
            if current_tokens + cell_tokens > max_tokens and current_chunk:
                chunk_text = format_chunk(current_chunk, i - len(current_chunk))
                chunks.append(chunk_text)
                
                current_chunk = [cell]
                current_tokens = cell_tokens
            else:
                current_chunk.append(cell)
                current_tokens += cell_tokens
        
        if current_chunk:
            chunk_text = format_chunk(current_chunk, len(cells) - len(current_chunk))
            chunks.append(chunk_text)
        
        return chunks
    
    def format_chunk(cells: List[Dict[str, Any]], start_index: int) -> str:
        """Форматирование части тетрадки"""
        chunk_text = f"# Часть тетрадки (ячейки {start_index + 1}-{start_index + len(cells)})\n\n"
        
        for i, cell in enumerate(cells, start_index + 1):
            cell_type = cell.get('cell_type', '')
            content = extract_cell_content(cell)
            
            if cell_type == 'markdown':
                chunk_text += f"## Ячейка {i} (Markdown)\n{content}\n\n"
            else:
                chunk_text += f"## Ячейка {i} (Code)\n{content}\n\n"
        
        return chunk_text
    
    def generate_improved_notebook(chunk: str) -> str:
        """Генерация улучшенной версии части тетрадки"""
        prompt = f"""Ты - эксперт по структурированию Jupyter Notebook. Проанализируй данную часть тетрадки и улучши её:

1. Структурируй содержимое тетрадки логически (создай разделы)
2. Обрами код подробными комментариями
3. Опиши действия, которые выполняются по ходу тетрадки
4. Сделай тетрадку более читаемой и понятной
5. Не меняй код, только структуру
6. Ответ верни в формате JSON

Часть тетрадки:
{chunk}

Улучшенная версия:"""

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return content
    
    def create_improved_notebook_structure(original_notebook: Dict[str, Any], improved_content: str) -> Dict[str, Any]:
        """Создание структуры улучшенной тетрадки в формате .ipynb"""
        # Создаем новую структуру тетрадки
        improved_notebook = {
            "cells": [],
            "metadata": original_notebook.get("metadata", {}),
            "nbformat": original_notebook.get("nbformat", 4),
            "nbformat_minor": original_notebook.get("nbformat_minor", 4)
        }
        
        # Разбиваем улучшенный контент на ячейки
        sections = improved_content.split("\n\n" + "="*80 + "\n\n")
        
        for section in sections:
            if section.strip():
                # Создаем markdown ячейку для каждого раздела
                cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [section.strip()]
                }
                improved_notebook["cells"].append(cell)
        
        return improved_notebook
    
    # Основная логика обработки
    print(f"Загружаю тетрадку: {file_path}")
    original_notebook = load_notebook_from_file(file_path)
    
    print("Разбиваю тетрадку на части...")
    chunks = split_notebook_into_chunks(original_notebook)
    
    print(f"Тетрадка разбита на {len(chunks)} частей")
    
    improved_parts = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Обрабатываю часть {i}/{len(chunks)}...")
        improved_chunk = generate_improved_notebook(chunk)
        improved_parts.append(improved_chunk)
    
    # Объединяем все улучшенные части
    final_content = "\n\n" + "="*80 + "\n\n".join(improved_parts)
    
    # Создаем структуру улучшенной тетрадки в формате .ipynb
    improved_notebook = create_improved_notebook_structure(original_notebook, final_content)
    
    return improved_notebook 