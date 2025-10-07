import numpy as np


class Config:
    """Конфигурация проекта"""

    # Настройки данных
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    # Настройки моделей
    CV_FOLDS = 5
    N_JOBS = -1
    # Настройки синтетических данных
    SYNTHETIC_SAMPLES = 1000
    SYNTHETIC_FEATURES = 20
    SYNTHETIC_NOISE = 15
    SYNTHETIC_INFORMATIVE = 10
    # Пути для сохранения
    MODEL_SAVE_PATH = "saved_models/"
    RESULTS_SAVE_PATH = "results/"
    # Метрики для оценки
    METRICS = ["rmse", "r2", "mae", "mape"]

    @classmethod
    def set_random_seed(cls):
        """Установка случайного seed для воспроизводимости"""
        np.random.seed(cls.RANDOM_STATE)
