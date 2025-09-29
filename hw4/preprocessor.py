import re
from typing import Callable

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from settings import config

nltk.download("stopwords")
nltk.download("punkt")


class TextPreprocessor:
    """Класс для предобработки текста твитов"""

    def __init__(self, config=config.preprocessing):
        self.config = config
        self._download_stopwords()
        self.stop_words = set(stopwords.words(self.config.STOPWORDS_LANGUAGE))
        self.stemmer = PorterStemmer()

    def _download_stopwords(self):
        """Загрузка стоп-слов при необходимости"""
        try:
            stopwords.words(self.config.STOPWORDS_LANGUAGE)
        except LookupError:
            nltk.download("stopwords")

    def clean_text(self, text: str) -> str:
        """Очистка и предобработка текста"""
        if not isinstance(text, str):
            return ""

        # Применение паттернов очистки
        for pattern in self.config.TEXT_CLEANING_PATTERNS.values():
            text = re.sub(pattern, "", text)
        text = text.lower().strip()

        # Токенизация и обработка слов
        words = text.split()
        processed_words = [
            self.stemmer.stem(word)
            for word in words
            if len(word) > self.config.MIN_WORD_LENGTH and word not in self.stop_words
        ]
        return " ".join(processed_words)

    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Предобработка текста в DataFrame"""
        df_clean = df.copy()
        df_clean["cleaned_text"] = df_clean[text_column].apply(self.clean_text)
        df_clean = df_clean[df_clean["cleaned_text"].str.len() > 0]
        return df_clean

    def create_preprocessing_function(self) -> Callable:
        """Создание функции предобработки"""
        return lambda text: self.clean_text(text)
