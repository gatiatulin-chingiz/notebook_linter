#!/usr/bin/env python3
"""
Jupyter Notebook Linter - CLI интерфейс
Простой командный интерфейс для использования линтера
"""

import sys
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from notebook_linter_module import process_notebook

def main():
    """Основная функция CLI интерфейса"""
    print("Jupyter Notebook Linter - CLI")
    print("=" * 40)
    print("Для использования в Jupyter Notebook импортируйте:")
    print("from notebook_linter_module import process_notebook")
    print("\nИли используйте этот CLI интерфейс:")
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
        if not os.path.exists(input_file):
            print(f"❌ Файл '{input_file}' не найден!")
            return
        
        if not input_file.endswith('.ipynb'):
            print("⚠️  Предупреждение: Файл не имеет расширения .ipynb")
        
        try:
            print(f"📁 Обрабатываю файл: {input_file}")
            
            # Загрузка модели и токенизатора
            print("🔄 Загружаю модель и токенизатор...")
            model_name = "./Текстовые/Qwen3-0.6B"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Обработка
            print("🔄 Обработка notebook...")
            improved_notebook = process_notebook(model, tokenizer, input_file)
            
            # Сохранение результата
            output_file = input_file.replace('.ipynb', '_improved.ipynb')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(improved_notebook, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Результат сохранен в: {output_file}")
            
            # Статистика
            input_size = os.path.getsize(input_file)
            output_size = os.path.getsize(output_file)
            print(f"📊 Размер входного файла: {input_size / 1024:.1f} KB")
            print(f"📊 Размер выходного файла: {output_size / 1024:.1f} KB")
            print(f"📄 Количество ячеек в улучшенной тетрадке: {len(improved_notebook['cells'])}")
            
        except Exception as e:
            print(f"❌ Ошибка: {str(e)}")
            return
    
    else:
        # Интерактивный режим
        print("\n📝 Интерактивный режим:")
        
        # Запрос файла
        input_file = input("Введите путь к .ipynb файлу (или Enter для выхода): ").strip()
        
        if not input_file:
            print("👋 До свидания!")
            return
        
        if not os.path.exists(input_file):
            print(f"❌ Файл '{input_file}' не найден!")
            return
        
        try:
            print(f"\n📁 Обрабатываю файл: {input_file}")
            
            # Загрузка модели и токенизатора
            print("🔄 Загружаю модель и токенизатор...")
            model_name = "./Текстовые/Qwen3-0.6B"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Обработка
            print("🔄 Обработка notebook...")
            improved_notebook = process_notebook(model, tokenizer, input_file)
            
            # Сохранение результата
            output_file = input_file.replace('.ipynb', '_improved.ipynb')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(improved_notebook, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Результат сохранен в: {output_file}")
            print(f"📄 Количество ячеек в улучшенной тетрадке: {len(improved_notebook['cells'])}")
            
        except Exception as e:
            print(f"❌ Ошибка: {str(e)}")

if __name__ == "__main__":
    main()