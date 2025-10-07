import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from config import Config


class AdvancedEnsembleModel:
    def __init__(self):
        self.ensemble = None
        self.results = {}
        self.ensemble_types = {}

    def create_ensembles(self, best_models):
        """Создание нескольких типов ансамблей"""
        estimators = [
            (name.replace(" ", "_"), result["model"])
            for name, result in best_models.items()
        ]

        # 1. Voting Regressor с равными весами
        voting_equal = VotingRegressor(
            estimators=estimators, weights=[1.0] * len(estimators)
        )

        # 2. Voting Regressor с весами на основе производительности
        weights = [
            1 / result["test_metrics"]["rmse"] for _, result in best_models.items()
        ]
        normalized_weights = [w / sum(weights) for w in weights]

        voting_weighted = VotingRegressor(
            estimators=estimators, weights=normalized_weights
        )

        # 3. Stacking Regressor
        final_estimator = list(best_models.values())[0][
            "model"
        ]  # Используем лучшую модель как финальный estimator
        stacking = StackingRegressor(
            estimators=estimators, final_estimator=final_estimator, cv=Config.CV_FOLDS
        )

        self.ensemble_types = {
            "Voting_Equal": voting_equal,
            "Voting_Weighted": voting_weighted,
            "Stacking": stacking,
        }

        return self.ensemble_types

    def evaluate_ensembles(self, ensembles, X_train, X_test, y_train, y_test):
        """Оценка всех типов ансамблей"""
        ensemble_results = {}

        for name, ensemble in ensembles.items():
            print(f"\n\U0001f50d Оценка ансамбля: {name}")

            # Обучение
            ensemble.fit(X_train, y_train)

            # Предсказания
            y_train_pred = ensemble.predict(X_train)
            y_test_pred = ensemble.predict(X_test)

            # Метрики
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            results = {
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "y_test_pred": y_test_pred,
                "ensemble": ensemble,
            }

            ensemble_results[name] = results
            print(f" \U0001f4ca {name} - Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

        self.results = ensemble_results
        return ensemble_results

    def select_best_ensemble(self, ensemble_results):
        """Выбор лучшего ансамбля"""
        best_ensemble_name = min(
            ensemble_results.items(), key=lambda x: x[1]["test_rmse"]
        )[0]
        best_ensemble = ensemble_results[best_ensemble_name]["ensemble"]

        print(f"\n\U0001f3c6 Лучший ансамбль: {best_ensemble_name}")
        print(
            f"\U0001f4ca Test RMSE: {ensemble_results[best_ensemble_name]['test_rmse']:.4f}"
        )
        print(
            f"\U0001f4ca Test R²: {ensemble_results[best_ensemble_name]['test_r2']:.4f}"
        )

        self.ensemble = best_ensemble
        return best_ensemble_name, best_ensemble

    def compare_with_single_models(self, ensemble_results, single_model_results):
        """Расширенное сравнение с одиночными моделями"""
        print("\n" + "=" * 70)
        print("ДЕТАЛЬНОЕ СРАВНЕНИЕ АНСАМБЛЕЙ С ОДИНОЧНЫМИ МОДЕЛЯМИ")
        print("=" * 70)

        # Лучшая одиночная модель
        best_single_name, best_single_rmse, best_single_r2 = (
            self._get_best_single_model(single_model_results)
        )

        # Лучший ансамбль
        best_ensemble_name, best_ensemble_rmse, best_ensemble_r2 = (
            self._get_best_ensemble(ensemble_results)
        )

        # Сравнение
        improvement_rmse = best_single_rmse - best_ensemble_rmse
        improvement_r2 = best_ensemble_r2 - best_single_r2
        improvement_percent = (improvement_rmse / best_single_rmse) * 100

        print(f"Лучшая одиночная модель: {best_single_name}")
        print(f"Лучший ансамбль: {best_ensemble_name}")
        print("\n\U0001f4c8 УЛУЧШЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ:")
        print(f"RMSE: {best_single_rmse:.4f} → {best_ensemble_rmse:.4f}")
        print(f"Улучшение RMSE: {improvement_rmse:.4f} ({improvement_percent:.2f}%)")
        print(f"R²: {best_single_r2:.4f} → {best_ensemble_r2:.4f}")
        print(f"Улучшение R²: {improvement_r2:.4f}")

        if improvement_rmse > 0:
            print("\U00002705 Ансамбль показал лучший результат!")
        else:
            print("\U0000274c Одиночная модель показала лучший результат")

        # Сравнение всех ансамблей с лучшей одиночной моделью
        print(f"\n{'Ансамбль':<20} {'RMSE':<10} {'R²':<10} {'Улучшение RMSE':<15}")
        print("-" * 60)

        for name, result in ensemble_results.items():
            rmse_improvement = best_single_rmse - result["test_rmse"]
            r2_improvement = result["test_r2"] - best_single_r2
            print(
                f"{name:<20} {result['test_rmse']:<10.4f} {result['test_r2']:<10.4f} {rmse_improvement:>10.4f}"
            )

    def _get_best_single_model(self, single_model_results):
        """Получение информации о лучшей одиночной модели"""
        best_single = min(
            single_model_results.items(), key=lambda x: x[1]["test_metrics"]["rmse"]
        )
        best_single_name = best_single[0]
        best_single_rmse = best_single[1]["test_metrics"]["rmse"]
        best_single_r2 = best_single[1]["test_metrics"]["r2"]

        return best_single_name, best_single_rmse, best_single_r2

    def _get_best_ensemble(self, ensemble_results):
        """Получение информации о лучшем ансамбле"""
        best_ensemble = min(ensemble_results.items(), key=lambda x: x[1]["test_rmse"])
        best_ensemble_name = best_ensemble[0]
        best_ensemble_rmse = best_ensemble[1]["test_rmse"]
        best_ensemble_r2 = best_ensemble[1]["test_r2"]

        return best_ensemble_name, best_ensemble_rmse, best_ensemble_r2

    def save_best_ensemble(self, ensemble_results, filename="best_ensemble.pkl"):
        """Сохранение лучшего ансамбля"""
        best_ensemble_name, best_ensemble = self.select_best_ensemble(ensemble_results)

        # Создаем папку для сохранения если не существует
        import os

        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

        ensemble_path = f"{Config.MODEL_SAVE_PATH}{filename}"
        joblib.dump(best_ensemble, ensemble_path)

        # Сохраняем метаданные
        metadata = {
            "ensemble_name": best_ensemble_name,
            "training_date": pd.Timestamp.now(),
            "metrics": {
                "test_rmse": ensemble_results[best_ensemble_name]["test_rmse"],
                "test_r2": ensemble_results[best_ensemble_name]["test_r2"],
            },
            "ensemble_type": type(best_ensemble).__name__,
        }

        metadata_path = f"{Config.MODEL_SAVE_PATH}{filename}_metadata.pkl"
        joblib.dump(metadata, metadata_path)

        print(
            f"\n\U0001f4be Лучший ансамбль '{best_ensemble_name}' сохранен в {ensemble_path}"
        )
        print(f"\U0001f4cb Метаданные сохранены в {metadata_path}")
        return best_ensemble_name, best_ensemble
