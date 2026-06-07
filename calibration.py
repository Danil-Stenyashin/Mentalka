"""
calibration.py  —  Улучшение #5: калибровка вероятностей (Platt Scaling).

CatBoost хорошо откалиброван, но дополнительная калибровка на валидационном сете
может уточнить пороги, особенно для дисбалансированных задач.

Используется:
- sklearn.calibration.CalibratedClassifierCV (изотоническая регрессия)
- Визуализация кривой надёжности до/после

Запуск (добавить в run_all.ipynb или запустить отдельно):
    python calibration.py
"""

import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score


# ---------------------------------------------------------------------------
# 1. Обёртка для CatBoost, совместимая со sklearn CalibrationCV
# ---------------------------------------------------------------------------

class CatBoostSklearnWrapper:
    """
    Тонкая обёртка CatBoostClassifier -> sklearn API.
    Нужна для CalibratedClassifierCV.
    """
    def __init__(self, cb_model):
        self.cb_model = cb_model
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        # CatBoost уже обучен — просто возвращаем себя
        return self

    def predict_proba(self, X):
        return self.cb_model.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


# ---------------------------------------------------------------------------
# 2. Основная функция калибровки
# ---------------------------------------------------------------------------

def calibrate_model(
    X_val: np.ndarray,
    y_val: np.ndarray,
    cb_model,
    method: str = 'isotonic',
    output_path: str = 'calibrated_model.pkl',
    config_path: str = 'model_config.json',
    plot: bool = True,
) -> object:
    """
    Выполняет калибровку CatBoost-модели на валидационном сете.

    Args:
        X_val:       Матрица признаков валидационного набора (np.ndarray).
        y_val:       Метки классов валидационного набора (0/1).
        cb_model:    Обученный CatBoostClassifier.
        method:      'isotonic' или 'sigmoid' (Platt Scaling).
        output_path: Путь для сохранения откалиброванной модели (pickle).
        config_path: Путь к model_config.json для обновления порога.
        plot:        Нарисовать ли кривую надёжности.

    Returns:
        Откалиброванная обёртка модели с методами predict_proba/predict.
    """
    print(f'[Калибровка] Метод: {method}')
    print(f'[Калибровка] Размер валидационного набора: {len(y_val)}')

    # Предсказание до калибровки
    raw_proba = cb_model.predict_proba(X_val)[:, 1]
    brier_before = brier_score_loss(y_val, raw_proba)
    auc_before   = roc_auc_score(y_val, raw_proba)
    print(f'[До]  Brier: {brier_before:.4f} | ROC-AUC: {auc_before:.4f}')

    # Калибровка через CalibratedClassifierCV с cv='prefit'
    wrapper    = CatBoostSklearnWrapper(cb_model)
    calibrated = CalibratedClassifierCV(wrapper, method=method, cv='prefit')
    calibrated.fit(X_val, y_val)

    # Предсказание после калибровки
    cal_proba    = calibrated.predict_proba(X_val)[:, 1]
    brier_after  = brier_score_loss(y_val, cal_proba)
    auc_after    = roc_auc_score(y_val, cal_proba)
    print(f'[После] Brier: {brier_after:.4f} | ROC-AUC: {auc_after:.4f}')
    print(f'[Улучшение] Brier: {brier_before - brier_after:+.4f} | '
          f'ROC-AUC: {auc_after - auc_before:+.4f}')

    # Оптимальный порог по F1 на валидации
    best_thr, best_f1 = _find_best_threshold(y_val, cal_proba)
    print(f'[Порог] Оптимальный порог: {best_thr:.4f} (F1={best_f1:.4f})')

    # Обновляем config
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        cfg['threshold']          = best_thr
        cfg['calibration_method'] = method
        cfg['brier_score_before'] = round(brier_before, 5)
        cfg['brier_score_after']  = round(brier_after,  5)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f'[Config] model_config.json обновлён (порог={best_thr:.4f})')

    # Сохраняем откалиброванную модель
    with open(output_path, 'wb') as f:
        pickle.dump(calibrated, f)
    print(f'[Сохранено] {output_path}')

    # График
    if plot:
        _plot_calibration(y_val, raw_proba, cal_proba, method)

    return calibrated


def _find_best_threshold(y_true, proba, steps=200):
    """Ищет порог, максимизирующий F1."""
    from sklearn.metrics import f1_score
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.3, 0.7, steps):
        preds = (proba >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def _plot_calibration(y_true, raw_proba, cal_proba, method):
    """Рисует кривую надёжности до/после калибровки."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, proba, label, color in [
        (axes[0], raw_proba, 'До калибровки',   '#e74c3c'),
        (axes[1], cal_proba, f'После ({method})', '#2ecc71'),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10)
        ax.plot(mean_pred, frac_pos, 's-', color=color, label=label, lw=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Идеальная калибровка')
        ax.set_xlabel('Средняя предсказанная вероятность')
        ax.set_ylabel('Доля положительных')
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('Кривая надёжности (Reliability Curve)', fontsize=13)
    plt.tight_layout()
    plt.savefig('calibration_curve.png', dpi=120)
    plt.show()
    print('[График] Сохранён: calibration_curve.png')


# ---------------------------------------------------------------------------
# 3. Загрузка откалиброванной модели в боте
# ---------------------------------------------------------------------------

def load_calibrated_model(path: str = 'calibrated_model.pkl'):
    """
    Загружает ранее сохранённую откалиброванную модель.
    Если файл не найден — возвращает None (бот использует исходный CatBoost).
    """
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f'[Калибровка] Загружена откалиброванная модель из {path}')
    return model


# ---------------------------------------------------------------------------
# 4. Пример запуска (если вызывается напрямую)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Этот скрипт запускается из run_all.ipynb или импортируется в бот.')
    print('Пример использования:')
    print('''
# В ноутбуке, после обучения:


# Использование в боте — автоматически через bot_v2.py
''')
