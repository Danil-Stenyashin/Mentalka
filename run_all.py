"""
Мастер-скрипт: обучение всех моделей + бенчмарк + финальная модель.

Этапы:
1. Генерация/загрузка датасета
2. Обучение CatBoost (гибридная модель)
3. Обучение RuBERT Fine-tuned
4. Сравнительный бенчмарк
5. Создание ансамбля
6. Сохранение финальной модели для бота

Запуск: python run_all.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ГИБРИДНАЯ МОДЕЛЬ ВЫЯВЛЕНИЯ ДЕПРЕССИИ И СУИЦИДАЛЬНЫХ ТЕКСТОВ")
print("Мастер-скрипт обучения")
print("=" * 70)

# ==========================================
# ШАГ 1: ДАТАСЕТ
# ==========================================

print("\n[ШАГ 1/6] Загрузка датасета...")

from dataset_loader import create_demo_dataset, create_train_test_split

df = create_demo_dataset(output_path="dataset.csv", samples_per_class=300)
X_train, X_test, y_train, y_test = create_train_test_split(
    df, target_col='binary_label', test_size=0.2
)

# ==========================================
# ШАГ 2: CATBOOST (гибридная модель)
# ==========================================

print("\n[ШАГ 2/6] Обучение CatBoost (гибридная модель)...")

from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from feature_extractor import extract_features, FEATURE_NAMES

print("  Загрузка sentence-transformers...")
encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

print("  Создание эмбеддингов train...")
X_train_emb = encoder.encode(X_train.tolist(), convert_to_numpy=True, show_progress_bar=True)
X_test_emb = encoder.encode(X_test.tolist(), convert_to_numpy=True, show_progress_bar=True)

print("  Извлечение лингвистических признаков train...")
train_meta = []
for text in X_train:
    features = extract_features(text)
    train_meta.append([features[name] for name in FEATURE_NAMES])
train_meta = np.array(train_meta)

print("  Извлечение лингвистических признаков test...")
test_meta = []
for text in X_test:
    features = extract_features(text)
    test_meta.append([features[name] for name in FEATURE_NAMES])
test_meta = np.array(test_meta)

# Объединяем
X_train_cb = np.hstack((X_train_emb, train_meta))
X_test_cb = np.hstack((X_test_emb, test_meta))

print(f"  Размерность CatBoost: {X_train_cb.shape}")

# Обучение
print("  Обучение CatBoost...")
cb_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=4,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=50,
    loss_function='Logloss',
    eval_metric='AUC',
    early_stopping_rounds=30,
    class_weights=[1.0, 2.0],
)

cb_model.fit(X_train_cb, y_train, eval_set=(X_test_cb, y_test), verbose=50)

# Метрики
y_pred_cb_proba = cb_model.predict_proba(X_test_cb)[:, 1]
y_pred_cb = (y_pred_cb_proba >= 0.5).astype(int)

cb_metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred_cb)),
    'f1': float(f1_score(y_test, y_pred_cb)),
    'roc_auc': float(roc_auc_score(y_test, y_pred_cb_proba)),
}
print(f"\n  CatBoost метрики: {cb_metrics}")

# Сохранение
cb_model.save_model("cb_suicide_model.cbm")
print("  Сохранено: cb_suicide_model.cbm")

# ==========================================
# ШАГ 3: RuBERT (опционально)
# ==========================================

print("\n[ШАГ 3/6] Обучение RuBERT (опционально)...")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.linear_model import LogisticRegression
    
    print("  Загрузка RuBERT...")
    tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
    model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
    
    # Для CPU на маленьком датасете — быстрый baseline
    print("  Извлечение [CLS]-эмбеддингов (это займёт время)...")
    
    def get_cls_embeddings(texts, batch_size=8):
        model.eval()
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size].tolist()
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**enc)
            cls = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.extend(cls)
        return np.array(embeddings)
    
    X_train_ru = get_cls_embeddings(X_train)
    X_test_ru = get_cls_embeddings(X_test)
    
    print("  Обучение LogReg на RuBERT [CLS]...")
    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr_model.fit(X_train_ru, y_train)
    
    y_pred_lr_proba = lr_model.predict_proba(X_test_ru)[:, 1]
    y_pred_lr = lr_model.predict(X_test_ru)
    
    lr_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred_lr)),
        'f1': float(f1_score(y_test, y_pred_lr)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_lr_proba)),
    }
    print(f"\n  RuBERT+LogReg метрики: {lr_metrics}")
    
    # Сохраняем
    with open("rubert_logreg_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    print("  Сохранено: rubert_logreg_model.pkl")
    
    rubert_available = True
    
except ImportError:
    print("  ? transformers/torch не установлены. Пропускаем RuBERT.")
    rubert_available = False
    y_pred_lr_proba = None
    lr_metrics = None

# ==========================================
# ШАГ 4: БЕНЧМАРК
# ==========================================

print("\n[ШАГ 4/6] Сравнительный анализ...")

predictions = {
    'CatBoost (гибридная)': {
        'proba': y_pred_cb_proba,
        'pred': y_pred_cb,
    }
}

if rubert_available and y_pred_lr_proba is not None:
    predictions['RuBERT + LogReg'] = {
        'proba': y_pred_lr_proba,
        'pred': y_pred_lr,
    }
    
    # Ансамбль
    ensemble_proba = 0.6 * y_pred_cb_proba + 0.4 * y_pred_lr_proba
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    predictions['Ансамбль (CatBoost+RuBERT)'] = {
        'proba': ensemble_proba,
        'pred': ensemble_pred,
    }
    
    ensemble_metrics = {
        'accuracy': float(accuracy_score(y_test, ensemble_pred)),
        'f1': float(f1_score(y_test, ensemble_pred)),
        'roc_auc': float(roc_auc_score(y_test, ensemble_proba)),
    }
    print(f"  Ансамбль метрики: {ensemble_metrics}")

# Создаём таблицу
results = []
for name, preds in predictions.items():
    results.append({
        'Модель': name,
        'Accuracy': round(accuracy_score(y_test, preds['pred']), 3),
        'F1': round(f1_score(y_test, preds['pred']), 3),
        'ROC-AUC': round(roc_auc_score(y_test, preds['proba']), 3),
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1', ascending=False)

print("\n" + "=" * 50)
print(results_df.to_string(index=False))
print("=" * 50)

results_df.to_csv("benchmark_results.csv", index=False, encoding='utf-8-sig')

# ==========================================
# ШАГ 5: ВАЖНОСТЬ ПРИЗНАКОВ
# ==========================================

print("\n[ШАГ 5/6] Анализ важности признаков...")

import matplotlib.pyplot as plt

feature_importances = cb_model.get_feature_importance()
embedding_imp = np.sum(feature_importances[:384])
meta_imp = feature_importances[384:]

importance_dict = {"Эмбеддинги (нейросеть)": embedding_imp}
for name, imp in zip(FEATURE_NAMES, meta_imp):
    importance_dict[name] = imp

plt.figure(figsize=(12, 10))
imp_series = pd.Series(importance_dict).sort_values(ascending=True)
imp_series.plot(kind='barh', color='teal')
plt.title("Вклад признаков в выявление депрессии (Feature Importance)")
plt.xlabel("Важность")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("  Сохранено: feature_importance.png")

# Топ-10 лингвистических
meta_imp_series = pd.Series(dict(zip(FEATURE_NAMES, meta_imp))).sort_values(ascending=False)
print(f"\n  Топ-10 лингвистических признаков:")
for name, val in meta_imp_series.head(10).items():
    print(f"    {name}: {val:.2f}")

# ==========================================
# ШАГ 6: СОХРАНЕНИЕ КОНФИГА
# ==========================================

print("\n[ШАГ 6/6] Сохранение конфигурации...")

# Подбор порога
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_cb_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

config = {
    "threshold": best_threshold,
    "embedding_dim": 384,
    "meta_feature_count": len(FEATURE_NAMES),
    "feature_names": FEATURE_NAMES,
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "rubert_model": "DeepPavlov/rubert-base-cased" if rubert_available else None,
    "total_features": int(X_train_cb.shape[1]),
    "models_trained": {
        "catboost": cb_metrics,
        "rubert_logreg": lr_metrics,
        "ensemble": ensemble_metrics if rubert_available else None,
    },
    "benchmark": results,
    "best_model": "catboost",  # Или ensemble
}

with open("model_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(f"  Сохранено: model_config.json")
print(f"  Оптимальный порог: {best_threshold:.3f}")

# ==========================================
# ФИНАЛ
# ==========================================

print("\n" + "=" * 70)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("=" * 70)
print(f"\nФайлы проекта:")
print(f"  • cb_suicide_model.cbm       — CatBoost модель")
if rubert_available:
    print(f"  • rubert_logreg_model.pkl    — RuBERT + LogReg")
print(f"  • model_config.json          — конфигурация")
print(f"  • feature_importance.png     — график важности")
print(f"  • benchmark_results.csv      — сравнительная таблица")
print(f"  • dataset.csv                — датасет")
print(f"\nСледующий шаг: python bot.py")
print("=" * 70)
