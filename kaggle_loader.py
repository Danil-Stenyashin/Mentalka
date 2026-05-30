"""
Скрипт для загрузки датасета "Suicide and Depression Detection" с Kaggle.

Требования:
    pip install kagglehub pandas

Использование:
    python kaggle_loader.py
    
Или вручную:
    1. Скачайте с https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
    2. Положите Suicide_Detection.csv в папку проекта
    3. Запустите: python run_all.py
"""

import os
import sys

print("=" * 60)
print("ЗАГРУЗКА ДАТАСЕТА С KAGGLE")
print("=" * 60)

try:
    import kagglehub
    print("\n[1/3] Подключение к KaggleHub...")
    
    # Скачиваем датасет
    path = kagglehub.dataset_download("nikhileswarkomati/suicide-watch")
    print(f"       Датасет загружен: {path}")
    
    # Ищем CSV
    import glob
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    if csv_files:
        csv_path = csv_files[0]
        print(f"\n[2/3] Найден CSV: {os.path.basename(csv_path)}")
        
        # Копируем в проект
        import shutil
        target = "Suicide_Detection.csv"
        shutil.copy2(csv_path, target)
        print(f"       Скопировано: {target}")
        
        # Проверяем
        import pandas as pd
        df = pd.read_csv(target)
        print(f"\n[3/3] Проверка датасета:")
        print(f"       Строк: {len(df)}")
        print(f"       Колонки: {list(df.columns)}")
        print(f"       Распределение:\n{df.iloc[:, -1].value_counts()}")
        
        print(f"\n{'='*60}")
        print(f"ГОТОВО! Теперь запустите: python run_all.py")
        print(f"{'='*60}")
        
    else:
        print("       CSV не найден в загруженном архиве")
        print(f"       Файлы: {os.listdir(path)}")
        
except ImportError:
    print("\n? kagglehub не установлен.")
    print("Установите: pip install kagglehub")
    print("\nИли скачайте вручную:")
    print("  https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch")
    print("  Положите Suicide_Detection.csv в эту папку")
    sys.exit(1)
    
except Exception as e:
    print(f"\n? Ошибка: {e}")
    print("\nСкачайте вручную:")
    print("  https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch")
    sys.exit(1)
