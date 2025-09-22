import sqlite3
import pandas as pd
import csv
from typing import Optional, List, Dict, Any
import logging
import os
from pathlib import Path
from settings import DATABASE_PATH, TABLE_NAME


class DatabaseManager:
    """Класс для управления операциями с базой данных SQLite"""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.conn = None
        self.logger = logging.getLogger(__name__)

    def check_database_exists(self) -> bool:
        """Проверка существования файла базы данных"""
        return os.path.exists(self.db_path)

    def get_database_info(self) -> Dict[str, Any]:
        """Получение информации о базе данных"""
        return {
            "path": self.db_path,
            "exists": self.check_database_exists(),
            "size": (
                os.path.getsize(self.db_path) if self.check_database_exists() else 0
            ),
            "directory_writable": (
                os.access(os.path.dirname(self.db_path), os.W_OK)
                if os.path.dirname(self.db_path)
                else False
            ),
        }

    def connect(self) -> None:
        """Установка соединения с базой данных с улучшенной обработкой ошибок"""
        try:
            # Создаем директорию для базы данных если её нет
            db_dir = os.path.dirname(self.db_path)
            if db_dir:  # Если путь содержит директорию
                os.makedirs(db_dir, exist_ok=True)
                self.logger.info(f"Создана/проверена директория: {db_dir}")

            self.conn = sqlite3.connect(self.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.logger.info(f"Успешное подключение к базе данных: {self.db_path}")

            # Проверяем доступность базы
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1")
            self.logger.debug("Тестовый запрос выполнен успешно")

        except sqlite3.Error as e:
            error_msg = f"Ошибка подключения к базе данных: {e}"
            self.logger.error(error_msg)

            # Проверяем права доступа
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.access(db_dir, os.W_OK):
                self.logger.error(f"Нет прав на запись в директорию: {db_dir}")

            raise ConnectionError(error_msg) from e

    def disconnect(self) -> None:
        """Закрытие соединения с базой данных"""
        if self.conn:
            self.conn.close()
            self.logger.info("Соединение с базой данных закрыто")

    def __enter__(self):
        """Поддержка контекстного менеджера"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Поддержка контекстного менеджера"""
        self.disconnect()

    def create_table(self, table_name: str = TABLE_NAME) -> None:
        """Создание таблицы в базе данных"""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY,
            gender TEXT,
            customer_type TEXT,
            age INTEGER,
            type_of_travel TEXT,
            class TEXT,
            flight_distance INTEGER,
            inflight_wifi_service INTEGER,
            departure_arrival_time_convenient INTEGER,
            ease_of_online_booking INTEGER,
            gate_location INTEGER,
            food_and_drink INTEGER,
            online_boarding INTEGER,
            seat_comfort INTEGER,
            inflight_entertainment INTEGER,
            on_board_service INTEGER,
            leg_room_service INTEGER,
            baggage_handling INTEGER,
            checkin_service INTEGER,
            inflight_service INTEGER,
            cleanliness INTEGER,
            departure_delay_in_minutes INTEGER,
            arrival_delay_in_minutes INTEGER,
            satisfaction TEXT
        );
        """

        try:
            cursor = self.conn.cursor()
            cursor.execute(create_table_query)
            self.conn.commit()
            self.logger.info(f"Таблица '{table_name}' создана успешно")
        except sqlite3.Error as e:
            self.logger.error(f"Ошибка создания таблицы: {e}")
            raise

    def table_exists(self, table_name: str = TABLE_NAME) -> bool:
        """Проверка существования таблицы"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                f"""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='{table_name}'
            """
            )
            return cursor.fetchone() is not None
        except sqlite3.Error as e:
            self.logger.error(f"Ошибка проверки существования таблицы: {e}")
            return False

    def get_table_row_count(self, table_name: str = TABLE_NAME) -> int:
        """Получение количества записей в таблице"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
        except sqlite3.Error as e:
            self.logger.error(f"Ошибка получения количества записей: {e}")
            return 0

    def clear_table(self, table_name: str = TABLE_NAME) -> int:
        """Очистка таблицы с возвратом количества удаленных записей"""
        try:
            cursor = self.conn.cursor()

            # Получаем количество записей до очистки
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count_before = cursor.fetchone()[0]

            if row_count_before > 0:
                cursor.execute(f"DELETE FROM {table_name}")
                self.conn.commit()
                self.logger.info(
                    f"Таблица '{table_name}' очищена. Удалено записей:{row_count_before}"
                )
                return row_count_before
            else:
                self.logger.info(f"Таблица '{table_name}' уже пуста")
                return 0
        except sqlite3.Error as e:
            self.logger.error(f"Ошибка очистки таблицы: {e}")
            self.conn.rollback()
        raise

    def import_data_from_csv(self, csv_path: str, table_name: str = TABLE_NAME) -> int:
        """
        Импорт данных из CSV файла в таблицу с использованием чистых SQL-операций.
        """
        try:
            # Проверяем существование CSV файла
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV файл не найден: {csv_path}")

            self.logger.info(f"Начинаем импорт данных из файла: {csv_path}")
            df = pd.read_csv(
                csv_path,
            )
            return self.import_data_from_dataframe(df, table_name)

        except Exception as e:
            self.logger.error(f"Ошибка импорта данных: {e}")
            raise

    def _insert_chunk(self, chunk: List[tuple], insert_query: str) -> None:
        """Вставка порции данных в таблицу"""
        try:
            cursor = self.conn.cursor()
            cursor.executemany(insert_query, chunk)
            self.conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"Ошибка вставки порции данных: {e}")
            self.conn.rollback()
            raise

    def import_data_from_dataframe(
        self, df: pd.DataFrame, table_name: str = TABLE_NAME
    ) -> int:
        """
        Импорт данных из DataFrame в таблицу с использованием чистых SQL-операций.
        Если таблица существует и содержит данные, она будет очищена перед
        импортом.
        """
        try:
            self.logger.info(
                f"Начинаем импорт данных из DataFrame в таблицу'{table_name}'"
            )

            # Проверяем существование таблицы
            if not self.table_exists(table_name):
                self.logger.warning(f"Таблица '{table_name}' не существует. Создаем...")
                self.create_table(table_name)

            # Проверяем и очищаем таблицу если в ней есть данные
            current_row_count = self.get_table_row_count(table_name)
            if current_row_count > 0:
                self.logger.info(
                    f"Таблица '{table_name}' содержит {current_row_count} записей. Очищаем..."
                )
                deleted_rows = self.clear_table(table_name)
                self.logger.info(f"Удалено {deleted_rows} существующих записей")

            # Мапинг столбцов DataFrame и БД
            column_mapping = {
                "id": "id",
                "Gender": "gender",
                "Customer Type": "customer_type",
                "Age": "age",
                "Type of Travel": "type_of_travel",
                "Class": "class",
                "Flight Distance": "flight_distance",
                "Inflight wifi service": "inflight_wifi_service",
                "Departure/Arrival time convenient": "departure_arrival_time_convenient",
                "Ease of Online booking": "ease_of_online_booking",
                "Gate location": "gate_location",
                "Food and drink": "food_and_drink",
                "Online boarding": "online_boarding",
                "Seat comfort": "seat_comfort",
                "Inflight entertainment": "inflight_entertainment",
                "On-board service": "on_board_service",
                "Leg room service": "leg_room_service",
                "Baggage handling": "baggage_handling",
                "Checkin service": "checkin_service",
                "Inflight service": "inflight_service",
                "Cleanliness": "cleanliness",
                "Departure Delay in Minutes": "departure_delay_in_minutes",
                "Arrival Delay in Minutes": "arrival_delay_in_minutes",
                "satisfaction": "satisfaction",
            }

            df = df.rename(columns=column_mapping)

            # Подготавливаем данные для вставки
            df_clean = self._clean_data(df)

            # Подготавливаем SQL запрос для вставки
            columns = [
                "id",
                "gender",
                "customer_type",
                "age",
                "type_of_travel",
                "class",
                "flight_distance",
                "inflight_wifi_service",
                "departure_arrival_time_convenient",
                "ease_of_online_booking",
                "gate_location",
                "food_and_drink",
                "online_boarding",
                "seat_comfort",
                "inflight_entertainment",
                "on_board_service",
                "leg_room_service",
                "baggage_handling",
                "checkin_service",
                "inflight_service",
                "cleanliness",
                "departure_delay_in_minutes",
                "arrival_delay_in_minutes",
                "satisfaction",
            ]

            placeholders = ", ".join(["?" for _ in columns])

            insert_query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({placeholders})
            """

            # Конвертируем DataFrame в список кортежей
            data_tuples = []
            for _, row in df_clean.iterrows():
                tuple_data = tuple(
                    row[col] if pd.notna(row[col]) else None for col in columns
                )
                data_tuples.append(tuple_data)

            # Вставляем данные порциями для эффективности
            chunk_size = 1000
            total_inserted = 0

            for i in range(0, len(data_tuples), chunk_size):
                chunk = data_tuples[i : i + chunk_size]
                self._insert_chunk(chunk, insert_query)
                total_inserted += len(chunk)
                self.logger.info(
                    f"Импортировано записей: {total_inserted}/{len(data_tuples)}"
                )
            self.logger.info(
                f"Импорт из DataFrame завершен. Всего импортировано записей: {total_inserted}"
            )
            return total_inserted

        except Exception as e:
            self.logger.error(f"Ошибка импорта данных из DataFrame: {e}")
            self.conn.rollback()
            raise

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка и предобработка данных"""
        self.logger.info("Начало очистки данных")

        df_clean = df.copy()

        # Удаление дубликатов
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_count - len(df_clean)
        if duplicates_removed > 0:
            self.logger.info(f"Удалено дубликатов: {duplicates_removed}")

        # Обработка пропущенных значений
        missing_values = df_clean.isnull().sum()
        if missing_values.any():
            self.logger.warning(
                f"Обнаружены пропущенные значения:\n{missing_values[missing_values > 0]}"
            )

        # Заполнение числовых значений медианой
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val)
                self.logger.info(
                    f"Заполнены пропуски в столбце {col} медианой: {median_val}"
                )

        # Заполнение текстовых значений модой
        text_cols = df_clean.select_dtypes(include=["object"]).columns
        for col in text_cols:
            if df_clean[col].isnull().any():
                mode_val = (
                    df_clean[col].mode()[0]
                    if not df_clean[col].mode().empty
                    else "Unknown"
                )
                df_clean[col].fillna(mode_val, inplace=True)
                self.logger.info(
                    f"Заполнены пропуски в столбце {col} модой: {mode_val}"
                )

            # Приведение типов данных
            type_mapping = {
                "age": "int32",
                "flight_distance": "int32",
                "departure_delay_in_minutes": "int32",
                "arrival_delay_in_minutes": "float",
            }
        for col, dtype in type_mapping.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(dtype)
            self.logger.info("Очистка данных завершена")
        return df_clean

    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Выполнение SQL запроса и возврат результата как DataFrame"""
        try:
            result = pd.read_sql_query(query, self.conn, params=params)
            self.logger.debug(f"Выполнен запрос: {query}")
            return result
        except Exception as e:
            self.logger.error(f"Ошибка выполнения запроса: {e}\nЗапрос: {query}")
            raise

    def get_table_info(self, table_name: str = TABLE_NAME) -> pd.DataFrame:
        """Получение информации о структуре таблицы"""
        query = f"PRAGMA table_info({table_name})"
        return self.execute_query(query)

    def vacuum_database(self) -> None:
        """Оптимизация базы данных"""
        try:
            self.conn.execute("VACUUM")
            self.logger.info("База данных оптимизирована (VACUUM)")
        except sqlite3.Error as e:
            self.logger.error(f"Ошибка оптимизации базы данных: {e}")
