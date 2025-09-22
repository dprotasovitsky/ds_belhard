import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging
from settings import PLOT_STYLE, COLOR_PALETTE


class DataVisualizer:
    """Класс для визуализации данных"""

    def __init__(self):
        plt.style.use(PLOT_STYLE)
        self.color_palette = COLOR_PALETTE
        self.logger = logging.getLogger(__name__)

    def plot_satisfaction_distribution(self, stats: Dict[str, Any]) -> None:
        """Визуализация распределения удовлетворенности"""
        self.logger.info("Создание графика распределения удовлетворенности")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Круговой график
        labels = ["Довольный", "Нейтральный/Недовольный"]
        sizes = [stats["satisfied_count"], stats["dissatisfied_count"]]
        colors = ["#66b3ff", "#ff9999"]

        ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Распределение удовлетворенности пассажиров")

        # Столбчатый график
        categories = ["Всего", "Довольный", "Недовольный"]
        values = [
            stats["total_passengers"],
            stats["satisfied_count"],
            stats["dissatisfied_count"],
        ]

        ax2.bar(categories, values, color=["lightblue", "lightgreen", "lightcoral"])
        ax2.set_title("Количество пассажиров по степени удовлетворенности")
        ax2.set_ylabel("Количество")

        plt.tight_layout()
        # plt.savefig("satisfaction_distribution.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_demographic_analysis(self, demo_data: pd.DataFrame) -> None:
        """Визуализация демографического анализа"""
        self.logger.info("Создание графиков демографического анализа")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Удовлетворенность по полу
        gender_data = demo_data.groupby("gender")["satisfaction_rate"].mean()
        axes[0, 0].bar(
            gender_data.index,
            gender_data.values,
            color=sns.color_palette(self.color_palette),
        )
        axes[0, 0].set_title("Уровень удовлетворенности в разбивке по полу")
        axes[0, 0].set_ylabel("Уровень удовлетворенности (%)")

        # Удовлетворенность по классу
        class_data = demo_data.groupby("class")["satisfaction_rate"].mean()
        axes[0, 1].bar(
            class_data.index,
            class_data.values,
            color=sns.color_palette(self.color_palette),
        )
        axes[0, 1].set_title("Уровень удовлетворенности в разбивке по классам")
        axes[0, 1].set_ylabel("Уровень удовлетворенности (%)")

        # Удовлетворенность по типу клиента
        customer_data = demo_data.groupby("customer_type")["satisfaction_rate"].mean()
        axes[1, 0].bar(
            customer_data.index,
            customer_data.values,
            color=sns.color_palette(self.color_palette),
        )
        axes[1, 0].set_title("Уровень удовлетворенности в разбивке по типам клиентов")
        axes[1, 0].set_ylabel("Уровень удовлетворенности (%)")

        plt.tight_layout()
        # plt.savefig("demographic_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_service_ratings(self, service_data: pd.DataFrame) -> None:
        """Визуализация оценок сервисов"""
        self.logger.info("Создание графиков оценок сервисов")

        plt.figure(figsize=(14, 8))

        # Сортировка сервисов по средней оценке
        service_data = service_data.sort_values("avg_rating", ascending=True)

        plt.barh(
            service_data["service_name"],
            service_data["avg_rating"],
            color=sns.color_palette(self.color_palette, len(service_data)),
        )

        plt.title("Средние оценки обслуживания")
        plt.xlabel("Средний рейтинг (1-5)")
        plt.xlim(0, 5)
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        # plt.savefig("service_ratings.png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> None:
        """Визуализация матрицы корреляций"""
        self.logger.info("Создание heatmap корреляций")

        plt.figure(figsize=(12, 10))

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
        )

        plt.title("Корреляционная матрица числовых признаков")
        plt.tight_layout()
        # plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
        plt.show()
