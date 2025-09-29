from typing import Any, Dict


class MetricsAnalyzer:
    """Анализатор оптимальных метрик для задачи"""

    @staticmethod
    def recommend_best_metric(business_context="balanced"):
        """Рекомендация лучшей метрики на основе бизнес-контекста"""
        recommendations = {
            "balanced": {
                "best_metric": "f1_score",
                "reason": "Баланс между Precision и Recall, оптимален для сбалансированных данных",
                "priority": ["f1_score", "accuracy", "roc_auc"],
            },
            "avoid_false_positives": {
                "best_metric": "precision",
                "reason": "Минимизация ложноположительных срабатываний",
                "priority": ["precision", "f1_score", "accuracy"],
            },
            "avoid_false_negatives": {
                "best_metric": "recall",
                "reason": "Минимизация пропущенных негативных твитов",
                "priority": ["recall", "f1_score", "roc_auc"],
            },
            "high_confidence": {
                "best_metric": "roc_auc",
                "reason": "Оценка общей разделяющей способности модели",
                "priority": ["roc_auc", "f1_score", "accuracy"],
            },
        }
        return recommendations.get(business_context, recommendations["balanced"])

    @staticmethod
    def analyze_class_balance(y):
        """Анализ баланса классов"""
        from collections import Counter

        counts = Counter(y)
        imbalance_ratio = max(counts.values()) / min(counts.values())

        return {
            "class_distribution": dict(counts),
            "imbalance_ratio": imbalance_ratio,
            "is_balanced": imbalance_ratio < 1.5,
            "recommendation": "F1-score" if imbalance_ratio < 1.5 else "ROC-AUC",
        }
