import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from config import (
    DATA_PATH,
    TARGET_COL,
    RANDOM_STATE,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TEST_SIZE,
)


def load_data():
    """Загрузка и первичный анализ данных"""
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Данные загружены. Размер: {df.shape}")
        print("Пропущенные значения:\n", df.isnull().sum())
        print(f"Типы данных:\n{df.dtypes}")
        return df
    except Exception as e:
        print(f"Ошибка загрузки CSV {DATA_PATH}: {e}")
        raise


def remove_outliers(df, column):
    """Удаление выбросов по межквартильному размаху"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    print(f"Границы выбросов для {column}: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(
        f"Обнаружено выбросов: {((df[column] < lower_bound) | (df[column] > upper_bound)).sum()}"
    )
    return df.loc[~((df[column] < lower_bound) | (df[column] > upper_bound))]


def create_features(df):
    """Создание новых признаков с использованием правильного синтаксиса loc"""
    # Используем loc для безопасного создания новых столбцов
    df = df.copy()
    # Объем рыбы
    df.loc[:, "Volume"] = df["Length1"] * df["Height"] * df["Width"]
    # Защита от деления на ноль
    df.loc[:, "Length_Ratio"] = np.where(
        df["Length1"] > 0, df["Length2"] / df["Length1"], 0
    )
    df.loc[:, "Height_Width_Ratio"] = np.where(
        df["Width"] > 0, df["Height"] / df["Width"], 0
    )
    # Дополнительные признаки
    df.loc[:, "Size_Index"] = (df["Length1"] + df["Height"] + df["Width"]) / 3
    df.loc[:, "Density"] = np.where(df["Volume"] > 0, df["Weight"] / df["Volume"], 0)
    print(
        "Созданы новые признаки: Volume, Length_Ratio, Height_Width_Ratio, Size_Index, Density"
    )
    return df


def preprocess_data(df):
    """Предобработка данных с улучшенной обработкой ошибок"""
    try:
        # Удаление выбросов
        print("\nУдаление выбросов...")
        original_size = df.shape[0]
        df = remove_outliers(df, TARGET_COL)
        new_size = df.shape[0]
        print(
            f"Удалено {original_size - new_size} выбросов ({((original_size - new_size)/original_size*100):.1f}%)"
        )
        # Создание новых признаков
        print("\nСоздание новых признаков...")
        df = create_features(df)
        # Логарифмирование целевой переменной
        print("\nЛогарифмирование целевой переменной...")
        y = np.log1p(df[TARGET_COL])
        # Подготовка фичей
        X = df.drop(columns=[TARGET_COL])
        # Список всех числовых признаков
        all_numerical = NUMERICAL_COLS + [
            "Volume",
            "Length_Ratio",
            "Height_Width_Ratio",
            "Size_Index",
            "Density",
        ]
        print(f"Используемые числовые признаки: {all_numerical}")
        # Преобразование категориальных признаков
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    CATEGORICAL_COLS,
                ),
                ("num", StandardScaler(), all_numerical),
            ],
            remainder="drop",
        )
        # Разделение данных
        print("\nРазделение данных на train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"Размер train: {X_train.shape[0]}, test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test, preprocessor
    except Exception as e:
        print(f"Ошибка при предобработке данных: {str(e)}")
        raise
