import time
from typing import Any, Dict

import numpy as np
import pandas as pd
from classifier import ClassifierFactory
from hyperparameter_tuner import HyperparameterTuner
from settings import config
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


class AdvancedModelTrainer:
    """Класс для обучения и оценки моделей с подбором параметров"""

    def __init__(self):
        self.config = config
        self.classifier_factory = ClassifierFactory()
        self.tuner = HyperparameterTuner()
        self.trained_models = {}
        self.tuning_results = {}

    def train_models(self, X_train, y_train, X_test, y_test, tune_hyperparameters=True):
        """Обучение моделей с подбором гиперпараметров"""
        classifiers = self.classifier_factory.create_all_classifiers()

        if tune_hyperparameters:
            print("\U0001f3af Начинаем подбор гиперпараметров...")
            self.tuning_results = self.tuner.tune_all_models(
                classifiers, X_train, y_train
            )
            classifiers = {
                name: result["best_model"]
                for name, result in self.tuning_results.items()
            }

        results = {}

        for name, classifier in classifiers.items():
            print(f"\U0001f3cb Обучение {name}...")
            start_time = time.time()
            try:
                classifier.fit(X_train, y_train)
                training_time = time.time() - start_time

                y_pred = classifier.predict(X_test)
                y_pred_proba = (
                    classifier.predict_proba(X_test)[:, 1]
                    if hasattr(classifier, "predict_proba")
                    else None
                )

                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                cv_scores = cross_val_score(
                    classifier,
                    X_train,
                    y_train,
                    cv=self.config.hyperparameters.CV_FOLDS,
                    scoring="f1",
                )

                results[name] = {
                    "model": classifier,
                    "metrics": metrics,
                    "predictions": {"y_pred": y_pred, "y_pred_proba": y_pred_proba},
                    "training_time": training_time,
                    "cv_scores": cv_scores,
                    "is_tuned": tune_hyperparameters,
                }

                print(
                    f"\U00002705 {name}: F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}"
                )

            except Exception as e:
                print(f"\U0000274c Ошибка при обучении {name}: {e}")
                continue
        self.trained_models = results
        return results

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Расчет метрик качества"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "roc_auc": (
                roc_auc_score(y_true, y_pred_proba)
                if y_pred_proba is not None
                else None
            ),
        }

    def get_best_model(self, metric="f1_score"):
        """Получение лучшей модели по метрике"""
        if not self.trained_models:
            raise ValueError("Модели не обучены.")

        valid_models = {
            name: result
            for name, result in self.trained_models.items()
            if result["metrics"][metric] is not None
        }

        best_model_name = max(
            valid_models.items(), key=lambda x: x[1]["metrics"][metric]
        )[0]
        return best_model_name, self.trained_models[best_model_name]
