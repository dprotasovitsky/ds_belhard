from typing import Tuple

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from settings import config


class DataBalancer:
    """Класс для балансировки данных"""

    def __init__(self, config=config.data):
        self.config = config
        self.strategies = {
            "smote": SMOTE(random_state=config.RANDOM_STATE),
            "oversample": RandomOverSampler(random_state=config.RANDOM_STATE),
            "undersample": RandomUnderSampler(random_state=config.RANDOM_STATE),
            "none": None,
        }

    def balance_data(self, X, y, strategy: str = None) -> Tuple:
        """Балансировка данных"""
        if strategy is None:
            strategy = self.config.BALANCE_STRATEGY

        if strategy != "none" and strategy in self.strategies:
            print(f"\U00002696 Балансировка данных методом: {strategy}")
            return self.strategies[strategy].fit_resample(X, y)

        print("\U00002696 Балансировка не применяется")
        return X, y

    def analyze_class_balance(self, y) -> dict:
        """Анализ баланса классов"""
        from collections import Counter

        counts = Counter(y)
        total = len(y)
        balance_ratio = max(counts.values()) / min(counts.values())

        # Определение рекомендации
        if balance_ratio < 1.5:
            recommendation = "F1-score (сбалансированные классы)"
        else:
            recommendation = "ROC-AUC (несбалансированные классы)"

        return {
            "class_distribution": dict(counts),
            "total_samples": total,
            "imbalance_ratio": balance_ratio,
            "is_balanced": balance_ratio < 1.5,
            "recommendation": recommendation,
        }
