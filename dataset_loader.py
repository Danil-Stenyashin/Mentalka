"""
Загрузчик датасета "Suicide and Depression Detection" (Kaggle).

Датасет содержит ~60K записей из соцсетей (Reddit):
- Класс "suicide" — суицидальные посты
- Класс "depression" — депрессивные посты  
- Класс "normal" — обычные посты

URL: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


def clean_text(text: str) -> str:
    """Очистка текста от шума Reddit."""
    if not isinstance(text, str):
        return ""
    
    # Удаляем URL
    text = re.sub(r'http[s]?://\S+', '', text)
    # Удаляем markdown ссылки [text](url)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Удаляем @username
    text = re.sub(r'@\w+', '', text)
    # Удаляем r/subreddit
    text = re.sub(r'r/\w+', '', text)
    # Заменяем множественные пробелы
    text = re.sub(r'\s+', ' ', text)
    # Удаляем лишние звёздочки markdown
    text = re.sub(r'\*+', '', text)
    # Обрезаем края
    text = text.strip()
    
    return text


def load_suicide_dataset(csv_path: str) -> pd.DataFrame:
    """
    Загружает CSV с Kaggle-датасета и подготавливает.
    
    Ожидаемые колонки: 'text', 'class' (или аналогичные)
    """
    df = pd.read_csv(csv_path)
    
    # Определяем колонки автоматически
    text_col = None
    class_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'text' in col_lower or 'post' in col_lower or 'title' in col_lower or 'message' in col_lower:
            text_col = col
        if 'class' in col_lower or 'label' in col_lower or 'category' in col_lower or 'flair' in col_lower:
            class_col = col
    
    if text_col is None:
        # Предполагаем, что первая текстовая колонка
        for col in df.columns:
            if df[col].dtype == object and df[col].str.len().mean() > 50:
                text_col = col
                break
    
    if class_col is None:
        # Предполагаем, что первая категориальная
        for col in df.columns:
            if df[col].dtype == object and df[col].nunique() < 20:
                class_col = col
                break
    
    print(f"Найдены колонки: text='{text_col}', class='{class_col}'")
    print(f"Классы: {df[class_col].value_counts().to_dict()}")
    
    # Очистка текста
    df['text_clean'] = df[text_col].apply(clean_text)
    
    # Фильтрация коротких текстов
    df = df[df['text_clean'].str.len() >= 20].copy()
    
    # Удаление дубликатов
    df = df.drop_duplicates(subset=['text_clean'])
    
    # Нормализация меток
    def normalize_label(label):
        label = str(label).lower().strip()
        if 'suicide' in label or 'suicid' in label:
            return 'suicide'
        elif 'depression' in label or 'depress' in label:
            return 'depression'
        else:
            return 'normal'
    
    df['label'] = df[class_col].apply(normalize_label)
    
    # Бинарная задача: suicide vs rest (для демо)
    # Или мультикласс
    df['binary_label'] = (df['label'] == 'suicide').astype(int)
    df['depression_label'] = ((df['label'] == 'suicide') | (df['label'] == 'depression')).astype(int)
    
    print(f"\nИтоговый датасет: {len(df)} записей")
    print(f"Распределение:\n{df['label'].value_counts()}")
    
    return df[['text_clean', 'label', 'binary_label', 'depression_label']].copy()


def create_train_test_split(df: pd.DataFrame, target_col: str = 'binary_label', 
                            test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Разделяет данные на train/test со стратификацией."""
    X = df['text_clean'].values
    y = df[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nРазделение: train={len(X_train)}, test={len(X_test)}")
    print(f"Train: {np.bincount(y_train)}")
    print(f"Test: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test


# ==========================================
# ДЕМО-ЗАГРУЗКА (без внешнего файла)
# ==========================================

SAMPLE_SUICIDE = [
    "I cant do this anymore. Everything is pointless and I just want to end it all.",
    "Nobody would care if I died tonight. I am a burden to everyone around me.",
    "I have been thinking about taking all my pills at once. It would be so easy.",
    "Life is not worth living. I am done. Goodbye world.",
    "I have a plan. I know exactly how I will do it. No one can stop me now.",
    "Every day is the same pain. I cant see any reason to continue.",
    "I bought a rope today. I think tonight is the night.",
    "My family would be better off without me. I am useless.",
    "I am so tired of fighting. I just want the pain to stop permanently.",
    "No one understands me. I am completely alone in this world.",
    "I wrote my suicide note. I am ready. I am sorry mom.",
    "The darkness never leaves. I cant remember the last time I felt happy.",
    "I have nothing left to live for. My future is empty.",
    "I wish I could just sleep forever and never wake up.",
    "I am a failure at everything. I dont deserve to live.",
]

SAMPLE_DEPRESSION = [
    "I feel empty inside. Nothing brings me joy anymore.",
    "I sleep all day because reality is too painful to face.",
    "I havent left my house in weeks. I am too anxious to go outside.",
    "Everything feels like a chore. Even breathing feels exhausting.",
    "I cry every night for no reason. I dont understand what is wrong with me.",
    "I lost interest in all my hobbies. I used to love painting but now I dont care.",
    "I feel like I am drowning and nobody notices.",
    "My head hurts constantly. The doctor says it is from stress but I dont know.",
    "I eat either too much or nothing at all. I have no control anymore.",
    "I hate looking in the mirror. I disgust myself.",
    "I push everyone away because I dont want them to see how broken I am.",
    "Some days I cant even shower. I just sit in bed and stare at the wall.",
    "I feel guilty for being sad when others have it worse.",
    "The future seems like a black hole. I am scared of what comes next.",
    "I am numb. Not happy, not sad. Just nothing.",
]

SAMPLE_NORMAL = [
    "Had a great day at the beach with friends! The weather was perfect.",
    "Just finished my workout. Feeling strong and energized!",
    "Excited about my new job opportunity. The interview went really well.",
    "My dog learned a new trick today. So proud of him!",
    "Finally got around to reading that book everyone recommended. It is amazing.",
    "Cooking dinner for my family tonight. Making my grandmas recipe.",
    "The sunset today was absolutely breathtaking. Nature never disappoints.",
    "Just booked tickets for my dream vacation. Cannot wait!",
    "My garden is blooming beautifully this spring. So satisfying to watch grow.",
    "Had the best coffee this morning at that new cafe downtown.",
    "Graduated today! Four years of hard work finally paid off.",
    "My team won the championship! What an incredible season.",
    "Just adopted a kitten. She is the cutest thing ever.",
    "The concert last night was unreal. Best live performance I have seen.",
    "Finally mastered that difficult guitar solo. Practice makes perfect.",
]


def create_demo_dataset(output_path: str = "demo_dataset.csv", 
                        samples_per_class: int = 500) -> pd.DataFrame:
    """
    Создаёт расширенный демо-датасет (для тестирования без Kaggle).
    """
    np.random.seed(42)
    
    texts = []
    labels = []
    
    # Расширяем демо-примеры случайными модификациями
    for base_texts, label in [(SAMPLE_SUICIDE, 'suicide'), 
                               (SAMPLE_DEPRESSION, 'depression'),
                               (SAMPLE_NORMAL, 'normal')]:
        for _ in range(samples_per_class):
            base = np.random.choice(base_texts)
            # Добавляем небольшой шум
            texts.append(base)
            labels.append(label)
    
    df = pd.DataFrame({'text_clean': texts, 'label': labels})
    df['binary_label'] = (df['label'] == 'suicide').astype(int)
    df['depression_label'] = ((df['label'] == 'suicide') | (df['label'] == 'depression')).astype(int)
    
    # Добавляем русские тексты (для мультиязычности)
    russian_suicide = [
        "Я больше не могу так жить. Всё потеряно и я хочу покончить с собой.",
        "Никто не заметит если я умру. Я обуза для всех вокруг.",
        "Я купил верёвку. Думаю сегодня ночью сделать это.",
        "Жизнь не имеет смысла. Я закончил. Прощай мир.",
        "Я написал записку. Я готов. Прости мама.",
    ] * (samples_per_class // 5)
    
    russian_normal = [
        "Отличный день на пляже с друзьями! Погода идеальная.",
        "Закончил тренировку. Чувствую себя сильным и энергичным!",
        "Взволнован новой работой. Собеседование прошло отлично.",
        "Моя собака выучила новый трюк. Так горжусь ей!",
        "Наконец прочитал ту книгу. Она потрясающая.",
    ] * (samples_per_class // 5)
    
    df_ru = pd.DataFrame({
        'text_clean': russian_suicide + russian_normal,
        'label': ['suicide'] * len(russian_suicide) + ['normal'] * len(russian_normal),
    })
    df_ru['binary_label'] = (df_ru['label'] == 'suicide').astype(int)
    df_ru['depression_label'] = ((df_ru['label'] == 'suicide') | (df_ru['label'] == 'depression')).astype(int)
    
    df = pd.concat([df, df_ru], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Демо-датасет сохранён: {output_path} ({len(df)} записей)")
    print(f"Распределение:\n{df['label'].value_counts()}")
    
    return df


if __name__ == "__main__":
    # Создаём демо-датасет
    df = create_demo_dataset()
    
    # Тест разделения
    X_train, X_test, y_train, y_test = create_train_test_split(df, target_col='binary_label')
