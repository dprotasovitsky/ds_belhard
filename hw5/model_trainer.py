import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import joblib
import time
from tqdm import tqdm
from config import Config
import warnings

warnings.filterwarnings("ignore")


class AdvancedModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_models = {}
        self.results = {}
        self.feature_importance = {}
        self.training_times = {}

    def initialize_models(self):
        """Инициализация расширенного набора моделей"""
        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=Config.RANDOM_STATE),
            "Lasso Regression": Lasso(random_state=Config.RANDOM_STATE),
            "ElasticNet": ElasticNet(random_state=Config.RANDOM_STATE),
            "Random Forest": RandomForestRegressor(
                random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=Config.RANDOM_STATE
            ),
            "Support Vector Regression": SVR(),
            "K-Neighbors": KNeighborsRegressor(n_jobs=Config.N_JOBS),
            "XGBoost": xgb.XGBRegressor(
                random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS
            ),
            "LightGBM": lgb.LGBMRegressor(
                random_state=Config.RANDOM_STATE,
                n_jobs=Config.N_JOBS,
                force_col_wise=True,
            ),
        }

    def get_advanced_hyperparameters(self):
        """Расширенные гиперпараметры для настройки"""
        param_grids = {
            "Ridge Regression": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "solver": ["auto", "svd", "cholesky", "lsqr"],
            },
            "Lasso Regression": {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                "selection": ["cyclic", "random"],
                "max_iter": [1000, 2000, 5000],
            },
            "ElasticNet": {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "max_iter": [1000, 2000],
            },
            "Random Forest": {
                "n_estimators": [100, 200],
                "max_depth": [10, 15, None],
                "min_samples_split": [2, 5],
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4, 5, 6],
                "min_samples_split": [2, 5, 10],
                "subsample": [0.8, 0.9, 1.0],
            },
            "Support Vector Regression": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto", 0.1, 1],
                "epsilon": [0.01, 0.1, 0.2],
            },
            "K-Neighbors": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 4, 5, 6, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
            },
            "LightGBM": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.05, 0.1, 0.15],
                "num_leaves": [31, 50, 100],
            },
        }
        return param_grids

    def calculate_additional_metrics(self, y_true, y_pred):
        """Расчет дополнительных метрик"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)

        # Mean Absolute Percentage Error (MAPE)
        mape = (
            np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        )

        # Explained Variance Score
        explained_variance = 1 - np.var(y_true - y_pred) / np.var(y_true)
        return {
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "mae": mae,
            "mape": mape,
            "explained_variance": explained_variance,
        }

    def evaluate_model(
        self, model, X_train, X_test, y_train, y_test, model_name, feature_names=None
    ):
        """Расширенная оценка модели"""
        start_time = time.time()

        # Обучение модели
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        self.training_times[model_name] = training_time

        # Предсказания
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Метрики
        train_metrics = self.calculate_additional_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_additional_metrics(y_test, y_test_pred)

        # Кросс-валидация
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=Config.CV_FOLDS,
            scoring="neg_mean_squared_error",
            n_jobs=Config.N_JOBS,
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_std = np.sqrt(cv_scores.std())

        # Важность признаков
        if feature_names and hasattr(model, "feature_importances_"):
            self.feature_importance[model_name] = model.feature_importances_
        elif feature_names:
            # Для моделей без встроенной важности признаков используем permutation importance
            try:
                perm_importance = permutation_importance(
                    model,
                    X_test,
                    y_test,
                    n_repeats=10,
                    random_state=Config.RANDOM_STATE,
                )
                self.feature_importance[model_name] = perm_importance.importances_mean
            except:
                self.feature_importance[model_name] = np.zeros(X_train.shape[1])

        results = {
            "model": model,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "cv_rmse": cv_rmse,
            "cv_std": cv_std,
            "training_time": training_time,
            "y_test_pred": y_test_pred,
            "y_train_pred": y_train_pred,
        }
        return results

    def tune_hyperparameters(self, model, param_grid, X_train, y_train, method="grid"):
        """Настройка гиперпараметров с выбором метода"""
        if method == "random":
            search = RandomizedSearchCV(
                model,
                param_grid,
                cv=Config.CV_FOLDS,
                scoring="neg_mean_squared_error",
                n_iter=20,
                n_jobs=Config.N_JOBS,
                random_state=Config.RANDOM_STATE,
                verbose=0,
            )
        else:
            search = GridSearchCV(
                model,
                param_grid,
                cv=Config.CV_FOLDS,
                scoring="neg_mean_squared_error",
                n_jobs=Config.N_JOBS,
                verbose=0,
            )

        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_score_

    def train_all_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names=None,
        tune=True,
        tuning_method="grid",
    ):
        """Обучение всех моделей с улучшенной логикой"""
        print("=" * 60)
        print("РАСШИРЕННОЕ ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("=" * 60)

        self.initialize_models()
        results = {}

        for name, model in tqdm(self.models.items(), desc="Обучение моделей"):
            print(f"\n\U0001f3af Обучение {name}...")

            best_score = None
            if tune and name in self.get_advanced_hyperparameters():
                print(f" \U00002699 Настройка гиперпараметров для {name}...")
                param_grid = self.get_advanced_hyperparameters()[name]
                model, best_score = self.tune_hyperparameters(
                    model, param_grid, X_train, y_train, tuning_method
                )
                if best_score:
                    print(
                        f" \U00002705 Лучший score после настройки: {-best_score:.4f}"
                    )

            model_results = self.evaluate_model(
                model, X_train, X_test, y_train, y_test, name, feature_names
            )
            results[name] = model_results

            test_metrics = model_results["test_metrics"]
            print(
                f" \U0001f4ca {name} - Test RMSE: {test_metrics['rmse']:.4f}, "
                f"R²: {test_metrics['r2']:.4f}, "
                f"Время: {model_results['training_time']:.2f}с"
            )

        self.results = results
        return results

    def select_best_models(self, results, n_models=3, metric="rmse"):
        """Выбор лучших моделей по указанной метрике"""
        sorted_models = sorted(
            results.items(), key=lambda x: x[1]["test_metrics"][metric]
        )

        best_models = {}
        print(f"\n\U0001f3c6 ТОП-{n_models} ЛУЧШИХ МОДЕЛЕЙ (по {metric.upper()}):")
        for i, (name, result) in enumerate(sorted_models[:n_models]):
            best_models[name] = result
            test_metrics = result["test_metrics"]
            print(
                f"{i + 1}. {name}: "
                f"RMSE = {test_metrics['rmse']:.4f}, "
                f"R² = {test_metrics['r2']:.4f}, "
                f"MAE = {test_metrics['mae']:.4f}"
            )

        return best_models

    def save_best_model(self, results, filename="best_model.pkl"):
        """Сохранение лучшей модели с дополнительной информацией"""
        best_model_name = min(
            results.items(), key=lambda x: x[1]["test_metrics"]["rmse"]
        )[0]

        best_model = results[best_model_name]["model"]

        # Создаем папку для сохранения если не существует
        import os

        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

        model_path = f"{Config.MODEL_SAVE_PATH}{filename}"
        joblib.dump(best_model, model_path)

        # Сохраняем метаданные
        metadata = {
            "model_name": best_model_name,
            "training_date": pd.Timestamp.now(),
            "metrics": results[best_model_name]["test_metrics"],
            "feature_importance": self.feature_importance.get(best_model_name, None),
        }

        metadata_path = f"{Config.MODEL_SAVE_PATH}{filename}_metadata.pkl"
        joblib.dump(metadata, metadata_path)

        print(
            f"\n\U0001f4be Лучшая модель '{best_model_name}' сохранена в {model_path}"
        )
        print(f"\U0001f4cb Метаданные сохранены в {metadata_path}")

        return best_model_name, best_model

    def get_results_dataframe(self):
        """Создание DataFrame с результатами всех моделей"""
        results_data = []

        for name, result in self.results.items():
            train_metrics = result["train_metrics"]
            test_metrics = result["test_metrics"]

            results_data.append(
                {
                    "Model": name,
                    "Train_RMSE": f"{train_metrics['rmse']:.4f}",
                    "Test_RMSE": f"{test_metrics['rmse']:.4f}",
                    "Train_R2": f"{train_metrics['r2']:.4f}",
                    "Test_R2": f"{test_metrics['r2']:.4f}",
                    "Test_MAE": f"{test_metrics['mae']:.4f}",
                    "Test_MAPE": f"{test_metrics['mape']:.2f}%",
                    "CV_RMSE": f"{result['cv_rmse']:.4f}",
                    "CV_Std": f"{result['cv_std']:.4f}",
                    "Training_Time": f"{result['training_time']:.2f}s",
                }
            )

        return pd.DataFrame(results_data)
