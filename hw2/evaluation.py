import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance
from config import *

def evaluate_model(model, X_test, y_test, model_name):
  """Оценка модели и визуализация результатов"""
  # Предсказание
  y_pred = model.predict(X_test)

  # Обратное преобразование логарифма
  y_test_exp = np.expm1(y_test)
  y_pred_exp = np.expm1(y_pred)

  # Вычисление метрик
  mse = np.mean((y_test_exp - y_pred_exp) ** 2)
  rmse = np.sqrt(mse)
  r2 = model.score(X_test, y_test)
  print(f"\nОценка модели {model_name}:")
  print(f"RMSE: {rmse:.2f}")
  print(f"R2: {r2:.4f}")

  # Графики
  plt.figure(figsize=(15, 10))

  # 1. Факт vs Прогноз
  plt.subplot(221)
  plt.scatter(y_test_exp, y_pred_exp, alpha=0.6)
  plt.plot([min(y_test_exp), max(y_test_exp)], [min(y_test_exp), max(y_test_exp)], 'r--')
  plt.title('Факт vs Прогноз')
  plt.xlabel('Фактические значения')
  plt.ylabel('Прогнозируемые значения')
  plt.grid(True)

  # 2. Остатки
  plt.subplot(222)
  residuals = y_test_exp - y_pred_exp
  plt.scatter(y_pred_exp, residuals, alpha=0.6)
  plt.hlines(0, min(y_pred_exp), max(y_pred_exp), colors='red')
  plt.title('Диаграмма остатков')
  plt.xlabel('Прогнозируемые значения')
  plt.ylabel('Остатки')
  plt.grid(True)

  # 3. Распределение ошибок
  plt.subplot(223)
  sns.histplot(residuals, kde=True)
  plt.title('Распределение ошибок')
  plt.xlabel('Ошибка')
  plt.grid(True)

  # 4. Важность признаков
  plt.subplot(224)
  if 'Linear' in model_name or 'Ridge' in model_name:
    # Для линейных моделей показываем коэффициенты
    if hasattr(model.named_steps['regressor'], 'coef_'):
        coefs = model.named_steps['regressor'].coef_
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        # Отбираем топ-10 самых значимых коэффициентов
        sorted_idx = np.argsort(np.abs(coefs))
        top_features = [feature_names[i] for i in sorted_idx[-10:]]
        top_coefs = coefs[sorted_idx[-10:]]
        plt.barh(top_features, top_coefs)
        plt.title('Топ-10 коэффициентов линейной модели')
    elif hasattr(model.named_steps['regressor'], 'feature_importances_'):
        # Для ансамблевых моделей показываем важность признаков
        feature_importances = model.named_steps['regressor'].feature_importances_
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        sorted_idx = np.argsort(feature_importances)
        plt.barh(
        [feature_names[i] for i in sorted_idx[-10:]],
        feature_importances[sorted_idx[-10:]]
        )
        plt.title('Топ-10 важных признаков')
        plt.tight_layout()
        plt.savefig(f'{model_name}_evaluation.png', dpi=300)
        plt.show()
    return r2