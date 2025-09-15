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

  # 4. Важность признаков (универсальный метод)
  plt.subplot(224)
  try:
    # Попытка получить важность признаков для модели
    if hasattr(model, 'feature_importances_'):
      # Для отдельных моделей (RandomForest, GradientBoosting)
      importances = model.feature_importances_
      feature_names = X_test.columns
    elif hasattr(model.named_steps['regressor'], 'feature_importances_'):
      # Для моделей в пайплайне
      importances = model.named_steps['regressor'].feature_importances_
      feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    else:
      # Для моделей без явной важности признаков
      # Используем permutation importance
      result = permutation_importance(
      model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE
      )
      importances = result.importances_mean
      feature_names = X_test.columns
      # Для ансамблей - дополнительная обработка
      if 'Voting' in model_name or 'Stacking' in model_name:
        # Вычисляем среднюю важность по базовым моделям
        base_importances = []
        for estimator in model.estimators_:
          if hasattr(estimator, 'feature_importances_'):
            base_importances.append(estimator.feature_importances_)
        if base_importances:
          importances = np.mean(base_importances, axis=0)
    # Сортируем признаки по важности
    sorted_idx = np.argsort(importances)
    top_features = [feature_names[i] for i in sorted_idx[-10:]]
    top_importances = importances[sorted_idx[-10:]]
    # Построение графика
    plt.barh(top_features, top_importances)
    plt.title('Топ-10 важных признаков' if len(top_features) == 10 else
    f'Топ-{len(top_features)} важных признаков')
    plt.xlabel('Важность')
  except Exception as e:
    print(f"Не удалось построить важность признаков: {str(e)}")
    plt.title('Важность признаков недоступна')
    plt.tight_layout()
    plt.savefig(f'{model_name}_evaluation.png', dpi=300)
    plt.show()
  return r2