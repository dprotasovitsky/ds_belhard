import time
import warnings
from typing import Any, Dict

import pandas as pd
from settings import config
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV


class HyperparameterTuner:
    """Класс для автоматического подбора гиперпараметров"""

    def __init__(self, config=config.hyperparameters):
        self.config = config
        self.scoring = make_scorer(f1_score)

    def tune_hyperparameters(self, model, param_grid, X_train, y_train, n_jobs=-1):
        """Подбор гиперпараметров для модели"""
        print(f"\U0001f50d Подбор параметров для {model.__class__.__name__}...")
        start_time = time.time()
        try:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=self.config.N_ITER,
                cv=self.config.CV_FOLDS,
                scoring=self.scoring,
                n_jobs=n_jobs,
                random_state=self.config.RANDOM_STATE,
                verbose=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                search.fit(X_train, y_train)
            tuning_time = time.time() - start_time
            return {
                "best_model": search.best_estimator_,
                "best_params": search.best_params_,
                "best_score": search.best_score_,
                "search_cv": search,
                "tuning_time": tuning_time,
                "success": True,
            }
        except Exception as e:
            return {
                "best_model": model,
                "best_params": model.get_params(),
                "best_score": 0,
                "search_cv": None,
                "tuning_time": time.time() - start_time,
                "success": False,
                "error": str(e),
            }

    def tune_all_models(self, classifiers: Dict[str, Any], X_train, y_train):
        """Подбор параметров для всех моделей"""
        tuned_models = {}

        for name, model in classifiers.items():
            if name in self.config.PARAM_GRIDS:
                result = self.tune_hyperparameters(
                    model, self.config.PARAM_GRIDS[name], X_train, y_train
                )
                if result["success"]:
                    print(
                        f"\U00002705 {name}: {result['best_score']:.4f} за {result['tuning_time']:.1f}с"
                    )
                else:
                    print(
                        f"\U0000274c {name}: ошибка - используется модель по умолчанию"
                    )
                tuned_models[name] = result
            else:
                tuned_models[name] = {
                    "best_model": model,
                    "best_params": model.get_params(),
                    "best_score": 0,
                    "success": False,
                }
        return tuned_models
