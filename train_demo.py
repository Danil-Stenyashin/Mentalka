"""
Демо-скрипт обучения гибридной модели (CatBoost).

Включает:
1. Генерацию синтетических данных (для демонстрации)
2. Создание эмбеддингов через sentence-transformers
3. Извлечение лингвистических признаков
4. Обучение CatBoost с подбором порога
5. Сохранение модели и анализ важности признаков
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score
)
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier, Pool

from feature_extractor import extract_features, FEATURE_NAMES

# ==========================================
# 1. ДЕМО-ДАННЫЕ (замените на реальные)
# ==========================================

depressive_texts = [
    "Мне кажется, всё потеряно. Я не вижу смысла продолжать. Жизнь бессмысленна.",
    "Каждый день одно и то же. Я устал бороться. Не хватает сил даже встать с кровати.",
    "Никому не нужен. Все отвернулись. Лучше бы я не родился.",
    "Я бесполезен. Всё что я делаю — провал. Никогда ничего не получится.",
    "Внутри пустота. Ничего не чувствую. Просто лежу и смотрю в стену часами.",
    "Не могу спать. Не могу есть. Таблетки не помогают. Всё бессмысленно.",
    "Зачем я живу? Обуза для семьи. Без меня всем было бы лучше.",
    "Думаю о конце. Вечный покой. Не будет больше боли.",
    "Меня никто не понимает. Я совсем один. В четырёх стенах.",
    "Всё моя вина. Я всё порчу. Я никчёмный. Позор семьи.",
    "Нет сил больше. Всё бесполезно. Пустота внутри. Мёртв внутри.",
    "Боюсь закрывать глаза. Кошмары. Просыпаюсь в холодном поту.",
    "Прощайте. Не будет меня. Забудьте. Я обуза. Всем лучше без меня.",
    "Хочу умереть. Покончить с собой. Выпрыгнуть. Всё кончено.",
    "Я истощён. Выгорел. Нет энергии. Апатия. Не хочу ничего.",
    "Одиночество давит. Никого нет. Не с кем поговорить. Заперся дома.",
    "Тошнит от себя. Ненавижу себя. Отвращение к себе. Презираю.",
    "Болит голова. Боли в теле. Нет сил встать. Лежу весь день.",
    "Всегда плохо. Никогда не везёт. Все против меня. Всё пропало.",
    "Мысли о суициде. Петля. Таблетки. Прощай мир. Я не справлюсь.",
]

normal_texts = [
    "Сегодня отличный день! Встретился с друзьями, погуляли в парке.",
    "Рад новому проекту на работе. Интересные задачи, хорошая команда.",
    "Учусь программировать. Пока сложно, но интересно. Есть прогресс.",
    "Встал рано, позанимался спортом. Чувствую прилив энергии.",
    "Планирую отпуск. Хочу посетить новые места, отдохнуть.",
    "Вчера был день рождения друга. Весело отметили, много смеха.",
    "Сдал экзамен! Готовился долго, но результат стоил усилий.",
    "Нашёл хорошую книгу. Читаю каждый вечер с удовольствием.",
    "Весна наступила. Солнечно, тепло. Прекрасное настроение.",
    "Семья в сборе. Ужинали вместе, разговаривали. Тепло и уютно.",
    "Начал новое хобби — фотография. Интересно учиться новому.",
    "Получил повышение на работе. Ценят мой труд. Горжусь собой.",
    "Спортзал даёт результаты. Силы прибавляется. Мотивация растёт.",
    "Устроили пикник с друзьями. Природа, еда, смех. Хороший день.",
    "Записался на курсы. Хочу развиваться, учиться новому.",
    "Провёл выходные с семьёй. Играли в настольные игры, смеялись.",
    "Встал пораньше, сделал много дел. Продуктивный день.",
    "Нашёл новый рецепт. Приготовил ужин. Вкусно получилось.",
    "Погода супер. Гулял по городу, фотографировал. Красиво.",
    "Решил старую задачу. Победа! Чувствую уверенность в себе.",
]

all_texts = depressive_texts + normal_texts
all_labels = [1] * len(depressive_texts) + [0] * len(normal_texts)

print(f"Датасет: {len(all_texts)} текстов (депрессия: {sum(all_labels)}, норма: {len(all_labels)-sum(all_labels)})")

# ==========================================
# 2. СОЗДАНИЕ ЭМБЕДДИНГОВ
# ==========================================

print("\n[1/6] Создаём эмбеддинги через sentence-transformers...")
print("Используем модель: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
embeddings = encoder.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)

print(f"Размерность эмбеддингов: {embeddings.shape}")

# ==========================================
# 3. ИЗВЛЕЧЕНИЕ ЛИНГВИСТИЧЕСКИХ ПРИЗНАКОВ
# ==========================================

print("\n[2/6] Извлекаем лингвистические признаки...")

features_list = []
for i, text in enumerate(all_texts):
    features = extract_features(text)
    features_list.append(features)
    if i < 3:
        print(f"  Пример '{text[:40]}...':")
        print(f"    депрессивный индекс={features['depression_index']:.2f}, "
              f"суицидальный риск={features['suicide_risk_index']:.2f}")

features_df = pd.DataFrame(features_list)
print(f"Лингвистических признаков: {len(FEATURE_NAMES)}")
print(f"Топ-5 признаков по среднему значению (депрессия):\n"
      f"{features_df.iloc[:len(depressive_texts)].mean().nlargest(5)}")

# ==========================================
# 4. ОБЪЕДИНЕНИЕ ПРИЗНАКОВ
# ==========================================

print("\n[3/6] Объединяем эмбеддинги + лингвистические признаки...")

meta_features = features_df.values
X = np.hstack((embeddings, meta_features))
y = np.array(all_labels)

print(f"Итоговая размерность: {X.shape} (эмбеддинги + лингв. признаки)")

# ==========================================
# 5. РАЗДЕЛЕНИЕ И ОБУЧЕНИЕ
# ==========================================

print("\n[4/6] Разделяем данные и обучаем CatBoost...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# CatBoost с параметрами для малого датасета
cb_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=50,
    loss_function='Logloss',
    eval_metric='AUC',
    early_stopping_rounds=50,
    class_weights=[1.0, 2.0],  # баланс классов
)

print("Обучаем модель...")
cb_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    verbose=50,
)

# ==========================================
# 6. ОЦЕНКА КАЧЕСТВА
# ==========================================

print("\n[5/6] Оцениваем качество модели...")

y_pred_proba = cb_model.predict_proba(X_test)[:, 1]
y_pred = cb_model.predict(X_test)

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.3f}")

print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['Норма', 'Депрессия'])}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

# Подбор оптимального порога по F1
print("\n--- Подбор порога ---")
thresholds = np.linspace(0.1, 0.9, 50)
best_f1 = 0
best_thresh = 0.5
for t in thresholds:
    y_pred_t = (y_pred_proba >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Лучший порог по F1: {best_thresh:.3f} (F1={best_f1:.3f})")

# ==========================================
# 7. КРОСС-ВАЛИДАЦИЯ
# ==========================================

print("\n--- Кросс-валидация (5-fold) ---")
cv_scores = cross_val_score(
    CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4, verbose=0),
    X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc'
)
print(f"ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ==========================================
# 8. ВАЖНОСТЬ ПРИЗНАКОВ
# ==========================================

print("\n[6/6] Анализируем важность признаков...")

feature_importances = cb_model.get_feature_importance()

# Эмбеддинги (индексы 0-383)
embedding_importance = np.sum(feature_importances[:384])

# Лингвистические признаки (индексы 384+)
meta_importances = feature_importances[384:]

importance_dict = {"Эмбеддинги (нейросеть)": embedding_importance}
for name, imp in zip(FEATURE_NAMES, meta_importances):
    importance_dict[name] = imp

# График важности
plt.figure(figsize=(12, 10))
imp_series = pd.Series(importance_dict).sort_values(ascending=True)
imp_series.plot(kind='barh', color='teal')
plt.title("Вклад признаков в выявление депрессии (Feature Importance)")
plt.xlabel("Важность")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("График сохранён > feature_importance.png")

# Топ-15 лингвистических признаков
meta_imp_series = pd.Series(dict(zip(FEATURE_NAMES, meta_importances))).sort_values(ascending=False)
print(f"\nТоп-10 лингвистических признаков:")
print(meta_imp_series.head(10).to_string())

# ==========================================
# 9. СОХРАНЕНИЕ МОДЕЛИ
# ==========================================

print("\nСохраняем модель и конфигурацию...")

cb_model.save_model("cb_suicide_model.cbm")

# Сохраняем конфиг
config = {
    "threshold": float(best_thresh),
    "embedding_dim": int(embeddings.shape[1]),
    "meta_feature_count": len(FEATURE_NAMES),
    "feature_names": FEATURE_NAMES,
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "total_features": int(X.shape[1]),
    "metrics": {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
    }
}

with open("model_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print("Модель сохранена > cb_suicide_model.cbm")
print("Конфигурация сохранена > model_config.json")
print(f"\nРекомендуемый порог для бота: {best_thresh:.3f}")

# ROC-кривая
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
print("ROC-кривая сохранена > roc_curve.png")

print("\n? Обучение завершено!")
