#!/usr/bin/env python3
"""
Jupyter Notebook Linter - CLI интерфейс
Простой командный интерфейс для использования линтера
"""

import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from notebook_linter_module import init_linter, lint_notebook, display_notebook_info

def main():
    """Основная функция CLI интерфейса"""
    print("Jupyter Notebook Linter - CLI")
    print("=" * 40)
    print("Для использования в Jupyter Notebook импортируйте:")
    print("from notebook_linter_module import *")
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
            
            # Инициализация линтера
            print("🔄 Инициализация линтера...")
            init_linter()
            
            # Анализ структуры
            print("\n📊 Анализ структуры:")
            display_notebook_info(input_file)
            
            # Обработка
            print("\n🔄 Обработка notebook...")
            improved_content = lint_notebook(input_file, display_result=False)
            
            # Сохранение результата
            output_file = input_file.replace('.ipynb', '_improved.md')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(improved_content)
            
            print(f"\n✅ Результат сохранен в: {output_file}")
            
            # Статистика
            input_size = os.path.getsize(input_file)
            output_size = os.path.getsize(output_file)
            print(f"📊 Размер входного файла: {input_size / 1024:.1f} KB")
            print(f"📊 Размер выходного файла: {output_size / 1024:.1f} KB")
            
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
            
            # Инициализация линтера
            print("🔄 Инициализация линтера...")
            init_linter()
            
            # Обработка
            print("🔄 Обработка notebook...")
            improved_content = lint_notebook(input_file, display_result=False)
            
            # Сохранение результата
            output_file = input_file.replace('.ipynb', '_improved.md')
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(improved_content)
            
            print(f"\n✅ Результат сохранен в: {output_file}")
            
        except Exception as e:
            print(f"❌ Ошибка: {str(e)}")

if __name__ == "__main__":
    main()