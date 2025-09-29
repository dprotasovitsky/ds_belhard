from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve


class ModelEvaluator:
    """Класс для оценки моделей"""

    def __init__(self, y_true, class_names=None):
        self.y_true = y_true
        self.class_names = class_names or ["Negative", "Positive"]

    def generate_detailed_report(self, results: Dict[str, Any]):
        """Генерация детального отчета для всех моделей"""
        detailed_reports = {}

        for model_name, result in results.items():
            y_pred = result["predictions"]["y_pred"]
            report = classification_report(
                self.y_true, y_pred, target_names=self.class_names, output_dict=True
            )
            cm = confusion_matrix(self.y_true, y_pred)
            detailed_reports[model_name] = {
                "classification_report": report,
                "confusion_matrix": cm,
                "metrics": result["metrics"],
            }
        return detailed_reports

    def plot_roc_curves(self, results: Dict[str, Any]):
        """Построение ROC-кривых"""
        plt.figure(figsize=(10, 8))

        for model_name, result in results.items():
            if result["predictions"]["y_pred_proba"] is not None:
                fpr, tpr, _ = roc_curve(
                    self.y_true, result["predictions"]["y_pred_proba"]
                )
                roc_auc = auc(fpr, tpr)

                plt.plot(
                    fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2
                )
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        plt.xlabel("Показатель ложных срабатываний")
        plt.ylabel("Показатель положительных срабатываний")
        plt.title("Сравнение ROC-кривых")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
