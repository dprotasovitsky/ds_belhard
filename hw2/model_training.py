import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from config import *

def train_gradient_boosting(preprocessor):
    """Обучение Gradient Boosting с поиском по сетке"""
    pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=RANDOM_STATE))
    ])
    param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__learning_rate': [0.05, 0.1],
    'regressor__max_depth': [3, 5],
    'regressor__min_samples_split': [2, 5]
    }
    model = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
    )
    return model, 'Gradient Boosting'

def train_random_forest(preprocessor):
    """Пайплайн для Random Forest"""
    return Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
    ))
    ]), 'Random Forest'

def train_ridge_regression(preprocessor):
    """Пайплайн для Ridge регрессии"""
    return Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=0.5))
    ]), 'Ridge Regression'

def train_linear_regression(preprocessor):
    """Пайплайн для Linear Regression"""
    return Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
    ]), 'Linear Regression'

def train_models(X_train, y_train, preprocessor):
    """Обучение нескольких моделей"""
    models = []
    # Создаем словарь функций для обучения
    training_functions = {
    'LinearRegression': train_linear_regression,
    'Ridge': train_ridge_regression,
    'RandomForest': train_random_forest,
    'GradientBoosting': train_gradient_boosting
    }
    # Обучаем выбранные модели
    for model_name in MODELS:
      if model_name in training_functions:
        if model_name == 'GradientBoosting':
            # Для GradientBoosting используем GridSearch
            model, name = training_functions[model_name](preprocessor)
            model.fit(X_train, y_train)
            models.append((model.best_estimator_, f"{name} (Best)"))
        else:
            # Для остальных моделей обычное обучение
            model, name = training_functions[model_name](preprocessor)
            model.fit(X_train, y_train)
            models.append((model, name))
    return models

def save_best_model(models, X_test, y_test):
    """Сохранение лучшей модели по R2 score"""
    best_model = None
    best_score = -np.inf
    best_name = ""
    for model, name in models:
        score = model.score(X_test, y_test)
        print(f"{name} R2: {score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
    print(f"\nЛучшая модель: {best_name} с R2: {best_score:.4f}")
    joblib.dump(best_model, MODEL_SAVE_PATH)
    print(f"Модель сохранена как {MODEL_SAVE_PATH}")
    return best_model