"""
SHAP-визуализация для интерпретации предсказаний модели.

Показывает, какие признаки повлияли на конкретное предсказание.
Идеально для курсовой работы — демонстрирует "почему модель так решила".

Требования:
    pip install shap
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
from feature_extractor import extract_features, FEATURE_NAMES


def explain_prediction(text: str, model_path="cb_suicide_model.cbm", 
                       config_path="model_config.json",
                       output_path="shap_explanation.png"):
    """
    Создаёт SHAP-визуализацию для одного текста.
    
    Returns:
        shap_values, prediction_proba
    """
    print(f"\n{'='*60}")
    print(f"SHAP-анализ текста: '{text[:50]}...'")
    print(f"{'='*60}")
    
    # 1. Загрузка модели
    print("[1/4] Загрузка модели...")
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # 2. Создание признаков
    print("[2/4] Извлечение признаков...")
    encoder = SentenceTransformer(config["embedding_model"])
    
    vec = encoder.encode([text], convert_to_numpy=True)
    features = extract_features(text)
    meta = np.array([features[name] for name in FEATURE_NAMES]).reshape(1, -1)
    X = np.hstack((vec, meta))
    
    # 3. Предсказание
    proba = float(model.predict_proba(X)[0][1])
    print(f"[3/4] Предсказание: {proba:.1%}")
    
    # 4. SHAP
    print("[4/4] Расчёт SHAP-значений...")
    
    # CatBoost SHAP
    shap_values = model.get_feature_importance(
        type=shap.Explanation if hasattr(shap, 'Explanation') else 'ShapValues',
        data=X
    )
    
    # Построение вручную (для CatBoost используем встроенный get_feature_importance)
    # Создаём DataFrame для визуализации
    all_names = [f"emb_{i}" for i in range(config["embedding_dim"])] + FEATURE_NAMES
    
    # Получаем важность признаков для этого конкретного текста
    feature_importance = model.get_feature_importance(
        data=shap.Pool(X),
        type="ShapValues"
    )[0]
    
    # Берём только лингвистические (эмбеддинги неинтерпретируемы)
    meta_shap = feature_importance[config["embedding_dim"]:]
    
    # Создаём DataFrame для сортировки
    shap_df = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'shap_value': meta_shap,
        'value': meta[0]
    })
    shap_df['abs_shap'] = shap_df['shap_value'].abs()
    shap_df = shap_df.sort_values('abs_shap', ascending=False).head(15)
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # График 1: Водопад (waterfall-like)
    ax1 = axes[0]
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_df['shap_value']]
    bars = ax1.barh(range(len(shap_df)), shap_df['shap_value'], color=colors)
    ax1.set_yticks(range(len(shap_df)))
    ax1.set_yticklabels(shap_df['feature'], fontsize=9)
    ax1.axvline(x=0, color='black', linewidth=0.8)
    ax1.set_xlabel('SHAP value (влияние на предсказание)', fontsize=11)
    ax1.set_title(f'Топ-15 лингвистических признаков\nПредсказание: {proba:.1%}', 
                  fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # Добавляем значения
    for i, (bar, val) in enumerate(zip(bars, shap_df['shap_value'])):
        ax1.text(val + (0.001 if val > 0 else -0.001), i, 
                f'{val:+.3f}', va='center', 
                ha='left' if val > 0 else 'right', fontsize=8)
    
    # График 2: Распределение категорий
    ax2 = axes[1]
    
    # Группируем по категориям
    categories = {
        'Безысходность': ['hopelessness_freq', 'depression_index'],
        'Суицидальные': ['suicidal_freq', 'suicide_risk_index'],
        'Когнитивные': ['cognitive_distortions_freq'],
        'Соц. изоляция': ['social_isolation_freq'],
        'Физические': ['physical_symptoms_freq'],
        'Позитивные': ['positive_markers_freq'],
        'Интенсификация': ['intensification_index'],
        'Самообращение': ['i_pronoun_freq'],
        'Эмоц. баланс': ['emotional_balance', 'negative_emotion_index'],
        'Структурные': ['text_length', 'word_count', 'lexical_diversity'],
    }
    
    cat_shap = {}
    for cat, feats in categories.items():
        cat_sum = sum(shap_df[shap_df['feature'].isin(feats)]['shap_value'].values)
        cat_shap[cat] = cat_sum
    
    cat_df = pd.Series(cat_shap).sort_values(ascending=False)
    colors2 = ['#e74c3c' if v > 0 else '#2ecc71' for v in cat_df.values]
    
    ax2.barh(range(len(cat_df)), cat_df.values, color=colors2)
    ax2.set_yticks(range(len(cat_df)))
    ax2.set_yticklabels(cat_df.index, fontsize=10)
    ax2.axvline(x=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Суммарное SHAP-влияние', fontsize=11)
    ax2.set_title('Влияние по категориям', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nСохранено: {output_path}")
    
    return shap_df, proba


def batch_shap_analysis(texts: list, labels: list, model_path="cb_suicide_model.cbm",
                        output_path="shap_summary.png"):
    """
    Создаёт summary plot для набора текстов (для курсовой).
    """
    print(f"\n{'='*60}")
    print(f"SHAP Summary для {len(texts)} текстов")
    print(f"{'='*60}")
    
    # Загрузка
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    with open("model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    
    encoder = SentenceTransformer(config["embedding_model"])
    
    # Создаём признаки
    X_list = []
    for text in texts:
        vec = encoder.encode([text], convert_to_numpy=True)
        features = extract_features(text)
        meta = np.array([features[name] for name in FEATURE_NAMES]).reshape(1, -1)
        X_list.append(np.hstack((vec, meta))[0])
    
    X_full = np.array(X_list)
    
    # Получаем SHAP
    shap_values_full = model.get_feature_importance(
        data=shap.Pool(X_full),
        type="ShapValues"
    )
    
    # Берём только лингвистические
    meta_shap_all = shap_values_full[:, config["embedding_dim"]:, 1]  # класс 1 (риск)
    X_meta = X_full[:, config["embedding_dim"]:]
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Создаём summary вручную (более контролируемо)
    shap_summary = pd.DataFrame(meta_shap_all, columns=FEATURE_NAMES)
    mean_abs_shap = shap_summary.abs().mean().sort_values(ascending=True).tail(20)
    
    colors = []
    for feat in mean_abs_shap.index:
        corr = np.corrcoef(X_meta[:, FEATURE_NAMES.index(feat)], meta_shap_all[:, FEATURE_NAMES.index(feat)])[0,1]
        colors.append('#e74c3c' if corr > 0 else '#2ecc71')
    
    bars = ax.barh(range(len(mean_abs_shap)), mean_abs_shap.values, color=colors)
    ax.set_yticks(range(len(mean_abs_shap)))
    ax.set_yticklabels(mean_abs_shap.index, fontsize=10)
    ax.set_xlabel('Mean |SHAP value| (среднее абс. влияние)', fontsize=12)
    ax.set_title('Важность признаков (SHAP summary)\nTop-20 лингвистических маркеров', 
                 fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, mean_abs_shap.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
               f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Сохранено: {output_path}")
    
    return mean_abs_shap


# ==========================================
# DEMO
# ==========================================

if __name__ == "__main__":
    # Пример: объяснение одного текста
    test_text = "Мне кажется, всё потеряно. Я не вижу смысла продолжать. Жизнь бессмысленна."
    
    try:
        explain_prediction(test_text)
    except FileNotFoundError:
        print("\nМодель не найдена. Сначала обучите: python run_all.py")
        print("\nДемо без модели:")
        
        # Показываем лингвистические признаки
        features = extract_features(test_text)
        print(f"\nЛингвистические признаки для текста:")
        print(f"  '{test_text}'")
        print(f"\n{'='*50}")
        
        # Топ-10 признаков
        top_features = sorted(
            [(name, features[name]) for name in FEATURE_NAMES],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]
        
        for name, val in top_features:
            direction = "^ повышает риск" if val > 0 else "v снижает риск"
            print(f"  {name:30s}: {val:8.2f}  {direction}")
