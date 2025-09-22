import pandas as pd
from typing import Dict, List, Any
import logging
from database_manager import DatabaseManager


class DataAnalyzer:
    """Класс для анализа данных и извлечения insights"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)

    def get_basic_statistics(self) -> Dict[str, Any]:
        """Получение базовой статистики по данным"""
        self.logger.info("Расчет базовой статистики")

        query = """
        SELECT
            COUNT(*) as total_passengers,
            COUNT(DISTINCT customer_type) as customer_types,
            COUNT(DISTINCT type_of_travel) as travel_types,
            COUNT(DISTINCT class) as classes,
            AVG(age) as avg_age,
            AVG(flight_distance) as avg_flight_distance,
            SUM(CASE WHEN satisfaction = 'satisfied' THEN 1 ELSE 0 END) as
            satisfied_count,
            SUM(CASE WHEN satisfaction = 'neutral or dissatisfied' THEN 1 ELSE 0 END) as
            dissatisfied_count
        FROM passengers;
        """

        stats = self.db_manager.execute_query(query).iloc[0].to_dict()
        stats["satisfaction_rate"] = round(
            stats["satisfied_count"] / stats["total_passengers"] * 100, 2
        )

        self.logger.info(f"Базовая статистика рассчитана: {stats}")
        return stats

    def analyze_by_demographics(self) -> pd.DataFrame:
        """Анализ удовлетворенности по демографическим признакам"""
        self.logger.info("Анализ по демографическим признакам")

        query = """
        SELECT
            gender,
            customer_type,
            class,
            COUNT(*) as total,
            SUM(CASE WHEN satisfaction = 'satisfied' THEN 1 ELSE 0 END) as satisfied,
            ROUND(SUM(CASE WHEN satisfaction = 'satisfied' THEN 1 ELSE 0 END) * 100.0 /
            COUNT(*), 2) as satisfaction_rate,
            AVG(age) as avg_age,
            AVG(flight_distance) as avg_distance
        FROM passengers
        GROUP BY gender, customer_type, class
        ORDER BY satisfaction_rate DESC;
        """

        return self.db_manager.execute_query(query)

    def analyze_service_ratings(self) -> pd.DataFrame:
        """Анализ оценок сервисов"""
        self.logger.info("Анализ оценок сервисов")

        service_columns = [
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
        ]

        service_stats = []
        for service in service_columns:
            query = f"""
            SELECT
                '{service}' as service_name,
                AVG({service}) as avg_rating,
                MIN({service}) as min_rating,
                MAX({service}) as max_rating,
                COUNT(*) as rating_count
            FROM passengers
            WHERE {service} IS NOT NULL;
            """

            stats = self.db_manager.execute_query(query).iloc[0].to_dict()
            service_stats.append(stats)

        return pd.DataFrame(service_stats)

    def correlation_analysis(self) -> pd.DataFrame:
        """Анализ корреляций между признаками"""
        self.logger.info("Анализ корреляций")

        # Ограничение для производительности
        query = """
        SELECT
            age, flight_distance, departure_delay_in_minutes,
            arrival_delay_in_minutes, inflight_wifi_service,
            seat_comfort, cleanliness,
            CASE WHEN satisfaction = 'satisfied' THEN 1 ELSE 0 END as satisfaction_numeric
        FROM passengers
        LIMIT 5000;
        """

        data = self.db_manager.execute_query(query)
        correlation_matrix = data.corr()

        return correlation_matrix

    def delay_impact_analysis(self) -> pd.DataFrame:
        """Анализ влияния задержек на удовлетворенность"""
        self.logger.info("Анализ влияния задержек")

        query = """
        SELECT
            CASE
                WHEN departure_delay_in_minutes = 0 THEN 'No Delay'
                WHEN departure_delay_in_minutes <= 30 THEN 'Short Delay (<=30min)'
                WHEN departure_delay_in_minutes <= 120 THEN 'Medium Delay (30-120min)'
                ELSE 'Long Delay (>120min)'
            END as delay_category,
            COUNT(*) as total_passengers,
            SUM(CASE WHEN satisfaction = 'satisfied' THEN 1 ELSE 0 END) as satisfied,
            ROUND(SUM(CASE WHEN satisfaction = 'satisfied' THEN 1 ELSE 0 END) * 100.0 /
            COUNT(*), 2) as satisfaction_rate,
            AVG(departure_delay_in_minutes) as avg_delay
        FROM passengers
        GROUP BY delay_category
        ORDER BY avg_delay;
        """

        return self.db_manager.execute_query(query)
