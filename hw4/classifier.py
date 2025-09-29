from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from settings import config
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier


class ClassifierFactory:
    """Фабрика для создания классификаторов"""

    def __init__(self, config=config.models):
        self.config = config
        self.classifier_map = {
            "LogisticRegression": LogisticRegression,
            "MultinomialNB": MultinomialNB,
            "SVC": SVC,
            "RandomForestClassifier": RandomForestClassifier,
            "ExtraTreesClassifier": ExtraTreesClassifier,
            "XGBClassifier": XGBClassifier,
            "LGBMClassifier": LGBMClassifier,
            "CatBoostClassifier": CatBoostClassifier,
        }

    def create_classifier(self, classifier_name: str):
        """Создание классификатора по имени"""
        if classifier_name not in self.config.CLASSIFIERS:
            raise ValueError(f"Классификатор {classifier_name} не найден")

        classifier_info = self.config.CLASSIFIERS[classifier_name]
        class_name = classifier_info["class"]
        params = classifier_info["params"]

        if class_name not in self.classifier_map:
            raise ValueError(f"Класс {class_name} не поддерживается")

        return self.classifier_map[class_name](**params)

    def create_all_classifiers(self):
        """Создание всех классификаторов из конфигурации"""
        return {
            name: self.create_classifier(name)
            for name in self.config.CLASSIFIERS.keys()
        }

    def get_model_categories(self):
        """Получение категорий моделей"""
        return {
            "Gradient Boosting": ["XGBoost", "LightGBM", "CatBoost"],
            "Ensemble Methods": ["Random Forest", "Extra Trees"],
            "Linear Models": ["Linear SVM"],
        }
