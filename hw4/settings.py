# Конфигурационные параметры проекта
from dataclasses import dataclass
from typing import Any, Dict, List


# @dataclass
class DataConfig:
    """Конфигурация данных"""

    SAMPLE_SIZE: int = 50000
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    DATA_URL: str = (
        "https://raw.githubusercontent.com/marcoguerini/COMPUTO/master/Sentiment140.csv"
    )
    LOCAL_DATA_PATH: str = "training.1600000.processed.noemoticon.csv"
    BALANCE_STRATEGY: str = "smote"  # smote, oversample, undersample, none


@dataclass
class PreprocessingConfig:
    """Конфигурация предобработки"""

    STOPWORDS_LANGUAGE: str = "english"
    MIN_WORD_LENGTH: int = 2
    TEXT_CLEANING_PATTERNS: Dict[str, str] = None

    def __post_init__(self):
        if self.TEXT_CLEANING_PATTERNS is None:
            self.TEXT_CLEANING_PATTERNS = {
                "user_mentions": r"@[A-Za-z0-9_]+",
                "urls": r"https?://[A-Za-z0-9./]+",
                "html_tags": r"<[^>]+>",
                "numbers": r"\d+",
                "special_chars": r"[^a-zA-Z\s]",
            }


# @dataclass
class FeatureConfig:
    """Конфигурация извлечения признаков"""

    VECTORIZER_TYPE: str = "tfidf"
    MAX_FEATURES: int = 5000
    NGRAM_RANGE: tuple = (1, 2)
    USE_IDF: bool = True


@dataclass
class HyperparameterConfig:
    """Конфигурация гиперпараметров"""

    CV_FOLDS: int = 3
    N_ITER: int = 15
    SCORING: str = "f1"
    RANDOM_STATE: int = 42

    PARAM_GRIDS: Dict[str, List[Dict[str, Any]]] = None

    def __post_init__(self):
        if self.PARAM_GRIDS is None:
            self.PARAM_GRIDS = {
                "XGBoost": [
                    {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 6],
                        "subsample": [0.8, 1.0],
                    }
                ],
                "LightGBM": [
                    {
                        "n_estimators": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "max_depth": [3, 6],
                        "num_leaves": [31, 63],
                    }
                ],
                "CatBoost": [
                    {
                        "iterations": [100, 200],
                        "learning_rate": [0.01, 0.1],
                        "depth": [4, 6],
                    }
                ],
                "Linear SVM": [{"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}],
                "Random Forest": [
                    {
                        "n_estimators": [100, 200],
                        "max_depth": [10, 15, None],
                        "min_samples_split": [2, 5],
                    }
                ],
                "Extra Trees": [
                    {"n_estimators": [100, 200], "max_depth": [10, 15, None]}
                ],
            }


@dataclass
class ModelConfig:
    """Конфигурация моделей"""

    CLASSIFIERS: Dict[str, Dict] = None

    def __post_init__(self):
        if self.CLASSIFIERS is None:
            self.CLASSIFIERS = {
                "XGBoost": {
                    "class": "XGBClassifier",
                    "params": {
                        "random_state": 42,
                        "eval_metric": "logloss",
                        "n_jobs": -1,
                    },
                },
                "LightGBM": {
                    "class": "LGBMClassifier",
                    "params": {"random_state": 42, "verbose": -1, "n_jobs": -1},
                },
                "CatBoost": {
                    "class": "CatBoostClassifier",
                    "params": {
                        "random_state": 42,
                        "verbose": False,
                        "thread_count": -1,
                    },
                },
                "Linear SVM": {
                    "class": "SVC",
                    "params": {
                        "kernel": "linear",
                        "random_state": 42,
                        "probability": True,
                    },
                },
                "Random Forest": {
                    "class": "RandomForestClassifier",
                    "params": {"random_state": 42, "n_jobs": -1},
                },
                "Extra Trees": {
                    "class": "ExtraTreesClassifier",
                    "params": {"random_state": 42, "n_jobs": -1},
                },
            }


# @dataclass
class ProjectConfig:
    """Основная конфигурация проекта"""

    data: DataConfig = DataConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    features: FeatureConfig = FeatureConfig()
    models: ModelConfig = ModelConfig()
    hyperparameters: HyperparameterConfig = HyperparameterConfig()
    MODEL_SAVE_PATH: str = "best_sentiment_model.pkl"
    EXPERIMENT_LOG_PATH: str = "experiments.json"


config = ProjectConfig()
