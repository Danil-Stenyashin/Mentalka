"""
Скрипт для сравнительного бенчмаркинга моделей.

Сравнивает:
1. CatBoost (гибридная модель: эмбеддинги + лингв. признаки)
2. RuBERT Fine-tuned (глубокое NLP)
3. Logistic Regression на RuBERT [CLS] (baseline)
4. Ансамбль (CatBoost + RuBERT)

Результаты сохраняются в results.json и results.png
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, 
    recall_score, classification_report, confusion_matrix
)


def run_benchmark(y_true, predictions_dict):
    """
    Сравнивает модели и строит таблицу результатов.
    
    Args:
        y_true: истинные метки
        predictions_dict: {model_name: {'proba': [...], 'pred': [...]}}
    
    Returns:
        DataFrame с метриками
    """
    results = []
    
    for model_name, preds in predictions_dict.items():
        proba = np.array(preds['proba'])
        pred = np.array(preds['pred'])
        
        results.append({
            'Модель': model_name,
            'Accuracy': round(accuracy_score(y_true, pred), 3),
            'Precision': round(precision_score(y_true, pred), 3),
            'Recall': round(recall_score(y_true, pred), 3),
            'F1-score': round(f1_score(y_true, pred), 3),
            'ROC-AUC': round(roc_auc_score(y_true, proba), 3),
        })
    
    df = pd.DataFrame(results)
    
    # Сортируем по F1
    df = df.sort_values('F1-score', ascending=False)
    
    print("\n" + "=" * 70)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МОДЕЛЕЙ")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    
    # Сохраняем
    df.to_csv("benchmark_results.csv", index=False, encoding='utf-8-sig')
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Сохранено: benchmark_results.csv, benchmark_results.json")
    
    return df


def plot_comparison(results_df, output_path="benchmark_chart.png"):
    """Строит сравнительный график метрик."""
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        bars = ax.barh(results_df['Модель'], results_df[metric], color=colors[:len(results_df)])
        ax.set_xlim(0, 1.05)
        ax.set_title(metric, fontweight='bold')
        ax.set_xlabel('Score')
        
        # Добавляем значения на бары
        for bar, val in zip(bars, results_df[metric]):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
    
    plt.suptitle('Сравнение моделей выявления суицидальных текстов', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"График сохранён: {output_path}")
    
    return fig


def create_ensemble_predictions(y_true, cb_proba, rubert_proba, weights=[0.5, 0.5]):
    """
    Создаёт ансамбль из CatBoost и RuBERT.
    
    Args:
        cb_proba: вероятности CatBoost
        rubert_proba: вероятности RuBERT
        weights: веса [catboost_weight, rubert_weight]
    
    Returns:
        (proba, pred) ансамбля
    """
    cb_proba = np.array(cb_proba)
    rubert_proba = np.array(rubert_proba)
    
    # Weighted average
    ensemble_proba = weights[0] * cb_proba + weights[1] * rubert_proba
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    
    return ensemble_proba, ensemble_pred


def find_best_ensemble_weights(y_true, cb_proba, rubert_proba):
    """Подбирает оптимальные веса ансамбля по F1."""
    
    best_f1 = 0
    best_weights = (0.5, 0.5)
    
    for w_cb in np.linspace(0, 1, 21):
        w_rubert = 1 - w_cb
        proba, pred = create_ensemble_predictions(y_true, cb_proba, rubert_proba, 
                                                   [w_cb, w_rubert])
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = (w_cb, w_rubert)
    
    print(f"\nОптимальные веса ансамбля: CatBoost={best_weights[0]:.2f}, RuBERT={best_weights[1]:.2f}")
    print(f"Ансамбль F1: {best_f1:.3f}")
    
    return best_weights


# ==========================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (для демо)
# ==========================================

if __name__ == "__main__":
    print("Запустите этот скрипт после обучения всех моделей.")
    print("\nПример использования:")
    print("""
    from benchmark import run_benchmark, plot_comparison, create_ensemble_predictions
    
    predictions = {
        'CatBoost (гибридная)': {
            'proba': cb_proba,
            'pred': (cb_proba >= 0.5).astype(int)
        },
        'RuBERT Fine-tuned': {
            'proba': rubert_proba,
            'pred': (rubert_proba >= 0.5).astype(int)
        },
        'LogReg на RuBERT': {
            'proba': logreg_proba,
            'pred': (logreg_proba >= 0.5).astype(int)
        },
        'Ансамбль (CatBoost + RuBERT)': {
            'proba': ensemble_proba,
            'pred': ensemble_pred
        },
    }
    
    results_df = run_benchmark(y_test, predictions)
    plot_comparison(results_df)
    """)
