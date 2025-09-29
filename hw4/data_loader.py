import os
from io import StringIO
from typing import Optional, Tuple

import pandas as pd
import requests
from settings import config


class DataLoader:
    """Класс для загрузки и подготовки данных Sentiment140"""

    def __init__(self, config=config.data):
        self.config = config

    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Загрузка данных Sentiment140"""
        if sample_size is None:
            sample_size = self.config.SAMPLE_SIZE
        try:
            df = self._load_local_data()
        except FileNotFoundError:
            df = self._download_data()
        df = self._preprocess_data(df, sample_size)
        return df

    def _load_local_data(self) -> pd.DataFrame:
        """Загрузка данных из локального файла"""
        return pd.read_csv(self.config.LOCAL_DATA_PATH, encoding="latin-1", header=None)

    def _download_data(self) -> pd.DataFrame:
        """Загрузка данных из интернета"""
        print("\U0001f4e5 Скачивание данных Sentiment140...")
        response = requests.get(self.config.DATA_URL)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), encoding="latin-1")

    def _preprocess_data(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Предварительная обработка данных"""
        df.columns = ["target", "id", "date", "flag", "user", "text"]
        df = df[df["target"].isin([0, 4])]
        df["target"] = df["target"].apply(lambda x: 1 if x == 4 else 0)
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=self.config.RANDOM_STATE)
        return df

    def get_train_test_split(
        self, df: pd.DataFrame, text_column: str = "text", target_column: str = "target"
    ) -> Tuple:
        """Разделение данных на обучающую и тестовую выборки"""
        from sklearn.model_selection import train_test_split

        return train_test_split(
            df[text_column],
            df[target_column],
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=df[target_column],
        )
