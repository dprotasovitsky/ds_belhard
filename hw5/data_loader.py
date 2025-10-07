import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from scipy import stats
import os
from config import Config


class AdvancedDataLoader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.is_fitted = False

    def load_dataset(self, dataset_type="synthetic"):
        """Загрузка различных типов датасетов"""
        if dataset_type == "california":
            return self._load_california_housing()
        elif dataset_type == "synthetic_complex":
            return self._load_complex_synthetic()
        else:
            return self._load_synthetic()

    def _load_california_housing(self):
        """Загрузка California Housing dataset"""
        print("Загрузка California Housing dataset...")
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["PRICE"] = data.target * 100000  # Преобразуем в реальные цены

        print(f"Загружен датасет California Housing: {df.shape}")
        return df, data.feature_names

    def _load_synthetic(self):
        """Базовая генерация синтетических данных"""
        print("Генерация базовых синтетических данных...")
        X, y = make_regression(
            n_samples=Config.SYNTHETIC_SAMPLES,
            n_features=Config.SYNTHETIC_FEATURES,
            noise=Config.SYNTHETIC_NOISE,
            random_state=Config.RANDOM_STATE,
            n_informative=Config.SYNTHETIC_INFORMATIVE,
        )

        feature_names = [f"feature_{i:02d}" for i in range(Config.SYNTHETIC_FEATURES)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y + abs(y.min()) + 100  # Делаем значения положительными

        return df, feature_names

    def _load_complex_synthetic(self):
        """Генерация сложных синтетических данных с разными распределениями"""
        print("Генерация сложных синтетических данных...")
        n_samples = Config.SYNTHETIC_SAMPLES

        # Генерируем признаки с разными распределениями
        np.random.seed(Config.RANDOM_STATE)

        # Нормальное распределение
        X_normal = np.random.normal(0, 1, (n_samples, 5))

        # Равномерное распределение
        X_uniform = np.random.uniform(-1, 1, (n_samples, 5))

        # Экспоненциальное распределение
        X_exp = np.random.exponential(1, (n_samples, 5))

        # Категориальные признаки (преобразованные в числовые)
        X_cat = np.random.randint(0, 3, (n_samples, 5))

        # Объединяем все признаки
        X = np.hstack([X_normal, X_uniform, X_exp, X_cat])

        # Создаем нелинейную целевую переменную
        y = (
            X[:, 0] ** 2
            + np.sin(X[:, 1] * np.pi)
            + np.log(np.abs(X[:, 2]) + 1)
            + X[:, 3] * X[:, 4]
            + np.random.normal(0, 10, n_samples)
        )

        feature_names = (
            [f"normal_{i}" for i in range(5)]
            + [f"uniform_{i}" for i in range(5)]
            + [f"exp_{i}" for i in range(5)]
            + [f"cat_{i}" for i in range(5)]
        )

        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y

        # Добавляем немного пропущенных значений и выбросов
        mask = np.random.random(df.shape) < 0.01
        df[mask] = np.nan

        # Добавляем выбросы
        outlier_mask = np.random.random(len(df)) < 0.02
        df.loc[outlier_mask, "target"] *= 3

        return df, feature_names

    def analyze_dataset(self, df, target_column):
        """Расширенный анализ датасета"""
        print("=" * 60)
        print("РАСШИРЕННЫЙ АНАЛИЗ ДАТАСЕТА")
        print("=" * 60)

        print(f"Размер датасета: {df.shape}")
        print(f"Количество признаков: {df.shape[1] - 1}")
        print(f"Количество samples: {df.shape[0]}")

        # Анализ пропущенных значений
        self._analyze_missing_values(df)

        # Анализ выбросов
        self._analyze_outliers(df, target_column)

        # Статистический анализ
        self._statistical_analysis(df, target_column)

        # Анализ корреляций
        self._correlation_analysis(df, target_column)

    def _analyze_missing_values(self, df):
        """Анализ пропущенных значений"""
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100

        print("\nАНАЛИЗ ПРОПУЩЕННЫХ ЗНАЧЕНИЙ:")
        missing_info = pd.DataFrame(
            {"Пропущено": missing_values, "Процент": missing_percent}
        )
        missing_info = missing_info[missing_info["Пропущено"] > 0]

        if len(missing_info) > 0:
            print(missing_info)
        else:
            print("Пропущенных значений нет")

    def _analyze_outliers(self, df, target_column):
        """Анализ выбросов с использованием IQR"""
        print("\nАНАЛИЗ ВЫБРОСОВ:")
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns[:5]:  # Анализируем первые 5 числовых колонок
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_percent = (len(outliers) / len(df)) * 100

            print(f"{col}: {len(outliers)} выбросов ({outlier_percent:.2f}%)")

    def _statistical_analysis(self, df, target_column):
        """Статистический анализ"""
        print(f"\nСТАТИСТИЧЕСКИЙ АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ '{target_column}':")
        target = df[target_column]

        stats_dict = {
            "Минимум": target.min(),
            "Максимум": target.max(),
            "Среднее": target.mean(),
            "Медиана": target.median(),
            "Стандартное отклонение": target.std(),
            "Асимметрия": target.skew(),
            "Эксцесс": target.kurtosis(),
        }

        for stat, value in stats_dict.items():
            print(f"{stat}: {value:.4f}")

        # Проверка нормальности
        stat, p_value = stats.normaltest(target)
        print(f"Тест на нормальность: p-value = {p_value:.4f}")
        if p_value > 0.05:
            print("Распределение близко к нормальному")
        else:
            print("Распределение отличается от нормального")

    def _correlation_analysis(self, df, target_column):
        """Анализ корреляций"""
        print(f"\nАНАЛИЗ КОРРЕЛЯЦИЙ С '{target_column}':")
        numeric_df = df.select_dtypes(include=[np.number])

        if target_column in numeric_df.columns:
            correlations = numeric_df.corr()[target_column].sort_values(
                key=abs, ascending=False
            )
            # Топ-5 самых коррелирующих признаков
            top_correlations = correlations.drop(target_column).head(5)
            for feature, corr in top_correlations.items():
                print(f"{feature}: {corr:.4f}")

    def prepare_data(self, df, target_column, test_size=Config.TEST_SIZE):
        """Улучшенная подготовка данных"""
        # Создаем копию данных
        df_processed = df.copy()

        # Заполнение пропущенных значений
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = self.imputer.fit_transform(
            df_processed[numeric_columns]
        )

        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=Config.RANDOM_STATE,
            stratify=pd.qcut(y, 5, labels=False) if len(y) > 100 else None,
        )

        # Масштабирование признаков
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.is_fitted = True

        return (
            X_train_scaled,
            X_test_scaled,
            y_train.values,
            y_test.values,
            X.columns.tolist(),
        )

    def get_feature_importance_data(self, feature_names):
        """Получение информации о признаках для анализа важности"""
        if self.is_fitted:
            return {
                "feature_names": feature_names,
                "scaler": self.scaler,
                "imputer": self.imputer,
            }
        return None
