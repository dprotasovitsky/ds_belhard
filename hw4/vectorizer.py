from settings import config
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline


class TextVectorizer:
    """Класс для векторизации текста"""

    def __init__(self, config=config.features):
        self.config = config
        self.vectorizer = self._create_vectorizer()
        self.is_fitted = False

    def _create_vectorizer(self):
        """Создание векторизатора"""
        vectorizer_params = {
            "max_features": self.config.MAX_FEATURES,
            "ngram_range": self.config.NGRAM_RANGE,
            "stop_words": "english",
        }

        if self.config.VECTORIZER_TYPE == "tfidf":
            vectorizer_params["use_idf"] = self.config.USE_IDF
            return TfidfVectorizer(**vectorizer_params)
        elif self.config.VECTORIZER_TYPE == "count":
            return CountVectorizer(**vectorizer_params)
        else:
            raise ValueError(
                f"Неизвестный тип векторизатора:{self.config.VECTORIZER_TYPE}"
            )

    def fit(self, texts):
        """Обучение векторизатора"""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts):
        """Преобразование текстов в векторы"""
        if not self.is_fitted:
            raise ValueError("Векторизатор не обучен. Сначала вызовите fit().")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        """Обучение и преобразование"""
        X_transformed = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return X_transformed

    def get_feature_names(self):
        """Получение названий признаков"""
        if not self.is_fitted:
            raise ValueError("Векторизатор не обучен.")
        return self.vectorizer.get_feature_names_out()

    def create_pipeline(self, classifier=None):
        """Создание пайплайна для векторизации и классификации"""
        steps = [("vectorizer", self.vectorizer)]
        if classifier:
            steps.append(("classifier", classifier))
        return Pipeline(steps)
