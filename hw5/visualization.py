import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class AdvancedVisualization:
    def __init__(self, style="seaborn"):
        self.style = style
        self.set_style()

    def set_style(self):
        """Установка стиля графиков"""
        if self.style == "seaborn":
            plt.style.use("seaborn-v0_8")
        elif self.style == "ggplot":
            plt.style.use("ggplot")
        else:
            plt.style.use("default")

        sns.set_palette("husl")

    def plot_dataset_analysis(self, df, target_column):
        """Комплексная визуализация датасета"""
        fig = plt.figure(figsize=(20, 15))

        # 1. Распределение целевой переменной
        plt.subplot(3, 3, 1)
        plt.hist(df[target_column], bins=30, alpha=0.7, edgecolor="black")
        plt.title(f"Распределение {target_column}")
        plt.xlabel(target_column)
        plt.ylabel("Частота")

        # 2. Boxplot целевой переменной
        plt.subplot(3, 3, 2)
        plt.boxplot(df[target_column])
        plt.title(f"Boxplot {target_column}")
        plt.ylabel(target_column)

        # 3. Корреляционная матрица (топ-10 признаков)
        plt.subplot(3, 3, 3)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        top_features = (
            df[numeric_cols]
            .corrwith(df[target_column])
            .abs()
            .sort_values(ascending=False)
            .head(10)
            .index
        )
        correlation_matrix = df[top_features].corr()
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
            linewidths=0.5,
        )
        plt.title("Корреляционная матрица (топ-10 признаков)")

        # 4. Scatter plots для самых коррелирующих признаков
        top_corr_features = (
            df[numeric_cols]
            .corrwith(df[target_column])
            .abs()
            .sort_values(ascending=False)
            .head(3)
            .index
        )

        for i, feature in enumerate(top_corr_features[:3], 4):
            if feature != target_column:
                plt.subplot(3, 3, i)
                plt.scatter(df[feature], df[target_column], alpha=0.5)
                plt.xlabel(feature)
                plt.ylabel(target_column)
                plt.title(f"{feature} vs {target_column}")

        plt.tight_layout()
        plt.show()

    def _convert_to_numeric(self, df, columns):
        """Конвертирует указанные колонки в числовой формат"""
        result = df.copy()
        for col in columns:
            if col in result.columns:
                # Убираем нечисловые символы и конвертируем
                result[col] = pd.to_numeric(
                    result[col].astype(str).str.replace("[^\d.-]", "", regex=True),
                    errors="coerce",
                )
        return result

    def plot_model_comparison(self, results_df):
        """Визуализация сравнения моделей с исправленной обработкой данных"""
        # Создаем копию DataFrame для обработки
        df_processed = results_df.copy()

        # Функция для безопасной конвертации в float
        def safe_float_convert(x):
            if isinstance(x, (int, float)):
                return float(x)
            elif isinstance(x, str):
                # Убираем все нечисловые символы кроме точки, минуса и цифр
                cleaned = "".join(c for c in x if c in "0123456789.-")
                try:
                    return float(cleaned) if cleaned else 0.0
                except (ValueError, TypeError):
                    return 0.0
            else:
                return 0.0

        # Конвертируем числовые колонки
        numeric_columns = [
            "Train_RMSE",
            "Test_RMSE",
            "Train_R2",
            "Test_R2",
            "Test_MAE",
            "CV_RMSE",
            "Training_Time",
        ]

        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].apply(safe_float_convert)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Сравнение Test_RMSE
        if "Test_RMSE" in df_processed.columns:
            # Убираем строки с NaN, нулевыми и бесконечными значениями
            rmse_data = (
                df_processed[["Model", "Test_RMSE"]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            rmse_data = rmse_data[
                rmse_data["Test_RMSE"] > 0
            ]  # Убираем нулевые значения

            if not rmse_data.empty:
                rmse_data = rmse_data.sort_values("Test_RMSE")
                bars = axes[0, 0].barh(rmse_data["Model"], rmse_data["Test_RMSE"])
                axes[0, 0].set_xlabel("RMSE")
                axes[0, 0].set_title("Сравнение моделей по RMSE (меньше - лучше)")

                # Добавляем значения на барчарты
                for bar in bars:
                    width = bar.get_width()
                    axes[0, 0].text(
                        width,
                        bar.get_y() + bar.get_height() / 2,
                        f"{width:.3f}",
                        ha="left",
                        va="center",
                        fontsize=8,
                    )

                # Поворачиваем подписи если нужно
                if len(rmse_data) > 5:
                    axes[0, 0].tick_params(axis="y", labelsize=8)
            else:
                axes[0, 0].text(
                    0.5,
                    0.5,
                    "Нет данных\nпо RMSE",
                    ha="center",
                    va="center",
                    transform=axes[0, 0].transAxes,
                )
                axes[0, 0].set_title("Сравнение моделей по RMSE")

        # 2. Сравнение Test_R2
        if "Test_R2" in df_processed.columns:
            r2_data = (
                df_processed[["Model", "Test_R2"]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            r2_data = r2_data[r2_data["Test_R2"] != 0]  # Убираем нулевые значения

            if not r2_data.empty:
                r2_data = r2_data.sort_values(
                    "Test_R2", ascending=False
                )  # Для R2 сортируем по убыванию
                bars = axes[0, 1].barh(r2_data["Model"], r2_data["Test_R2"])
                axes[0, 1].set_xlabel("R² Score")
                axes[0, 1].set_title("Сравнение моделей по R² (больше - лучше)")

                # Добавляем значения на барчарты
                for bar in bars:
                    width = bar.get_width()
                    axes[0, 1].text(
                        width,
                        bar.get_y() + bar.get_height() / 2,
                        f"{width:.3f}",
                        ha="left",
                        va="center",
                        fontsize=8,
                    )
                if len(r2_data) > 5:
                    axes[0, 1].tick_params(axis="y", labelsize=8)
            else:
                axes[0, 1].text(
                    0.5,
                    0.5,
                    "Нет данных\nпо R²",
                    ha="center",
                    va="center",
                    transform=axes[0, 1].transAxes,
                )
                axes[0, 1].set_title("Сравнение моделей по R²")

        # 3. Время обучения
        if "Training_Time" in df_processed.columns:
            time_data = (
                df_processed[["Model", "Training_Time"]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            time_data = time_data[
                time_data["Training_Time"] > 0
            ]  # Убираем нулевые значения

            if not time_data.empty:
                time_data = time_data.sort_values("Training_Time")
                bars = axes[1, 0].barh(time_data["Model"], time_data["Training_Time"])

                axes[1, 0].set_xlabel("Время (секунды)")
                axes[1, 0].set_title("Время обучения моделей")

                # Добавляем значения на барчарты
                for bar in bars:
                    width = bar.get_width()
                    axes[1, 0].text(
                        width,
                        bar.get_y() + bar.get_height() / 2,
                        f"{width:.1f}s",
                        ha="left",
                        va="center",
                        fontsize=8,
                    )

                if len(time_data) > 5:
                    axes[1, 0].tick_params(axis="y", labelsize=8)
            else:
                axes[1, 0].text(
                    0.5,
                    0.5,
                    "Нет данных\nпо времени",
                    ha="center",
                    va="center",
                    transform=axes[1, 0].transAxes,
                )
                axes[1, 0].set_title("Время обучения моделей")

        # 4. Сравнение всех метрик
        metrics_to_plot = ["Test_RMSE", "Test_R2", "CV_RMSE"]
        available_metrics = [m for m in metrics_to_plot if m in df_processed.columns]

        if len(available_metrics) >= 2:  # Нужно как минимум 2 метрики для сравнения
            comparison_data = (
                df_processed[["Model"] + available_metrics]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            # Убираем строки где все значения нулевые
            comparison_data = comparison_data[
                (comparison_data[available_metrics] != 0).any(axis=1)
            ]
            if not comparison_data.empty and len(comparison_data) > 1:
                # Устанавливаем Model как индекс для нормализации
                comparison_data = comparison_data.set_index("Model")
                # Нормализуем данные (0-1), где 0 - лучше, 1 - хуже
                comparison_data_normalized = comparison_data.copy()
                for col in comparison_data_normalized.columns:
                    col_data = comparison_data_normalized[col]
                    min_val = col_data.min()
                    max_val = col_data.max()
                    # Проверяем что диапазон не нулевой
                    if max_val - min_val > 1e-10:
                        if "R2" in col:
                            # Для R2 инвертируем: большее значение становится меньшим после нормализации
                            comparison_data_normalized[col] = 1 - (
                                (col_data - min_val) / (max_val - min_val)
                            )
                        else:
                            # Для RMSE и других: меньшее значение лучше
                            comparison_data_normalized[col] = (col_data - min_val) / (
                                max_val - min_val
                            )
                    else:
                        # Если все значения одинаковые, устанавливаем 0.5
                        comparison_data_normalized[col] = 0.5

                # Строим график
                comparison_data_normalized.plot(kind="bar", ax=axes[1, 1])
                axes[1, 1].set_title(
                    "Нормализованное сравнение метрик\n(меньше = лучше для всех метрик)"
                )
                axes[1, 1].tick_params(axis="x", rotation=45)
                axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                # Добавляем подписи значений
                for i, (idx, row) in enumerate(comparison_data_normalized.iterrows()):
                    for j, (col, val) in enumerate(row.items()):
                        axes[1, 1].text(
                            i,
                            val + 0.01,
                            f"{val:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=6,
                            rotation=90,
                        )
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    "Недостаточно данных\nдля сравнения",
                    ha="center",
                    va="center",
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title("Нормализованное сравнение метрик")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Недостаточно метрик\nдля сравнения",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].set_title("Нормализованное сравнение метрик")
        plt.tight_layout()
        plt.show()

        # Дополнительно: вывод таблицы с результатами
        print("\n" + "=" * 80)
        print("ТАБЛИЦА РЕЗУЛЬТАТОВ МОДЕЛЕЙ")
        print("=" * 80)

        # Выбираем только основные колонки для отображения
        display_columns = ["Model", "Test_RMSE", "Test_R2", "Test_MAE", "Training_Time"]
        available_display_columns = [
            col for col in display_columns if col in df_processed.columns
        ]

        if available_display_columns:
            display_df = df_processed[available_display_columns].copy()

            # Форматируем числовые колонки для красивого отображения
            for col in ["Test_RMSE", "Test_R2", "Test_MAE"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                    )
            if "Training_Time" in display_df.columns:
                display_df["Training_Time"] = display_df["Training_Time"].apply(
                    lambda x: f"{x:.2f}s" if pd.notna(x) and x > 0 else "N/A"
                )

            # Сортируем по Test_RMSE если есть
            if "Test_RMSE" in display_df.columns:
                # Создаем временную колонку для сортировки
                temp_sort = display_df["Test_RMSE"].replace("N/A", np.inf).astype(float)
                display_df = display_df.iloc[temp_sort.argsort()]

            print(display_df.to_string(index=False, max_colwidth=30))
        else:
            print("Нет данных для отображения")

    def plot_predictions_comparison(self, y_true, y_pred_dict, model_names):
        """Визуализация предсказаний разных моделей"""
        n_models = len(model_names)
        fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10))

        if n_models == 1:
            axes = np.array([axes]).T

        for i, (name, y_pred) in enumerate(y_pred_dict.items()):
            # Scatter plot предсказаний vs реальных значений
            axes[0, i].scatter(y_true, y_pred, alpha=0.6)
            axes[0, i].plot(
                [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
            )
            axes[0, i].set_xlabel("Реальные значения")
            axes[0, i].set_ylabel("Предсказанные значения")
            axes[0, i].set_title(f"{name}\nПредсказания vs Реальные значения")

            # Распределение ошибок
            errors = y_true - y_pred
            axes[1, i].hist(errors, bins=30, alpha=0.7, edgecolor="black")
            axes[1, i].axvline(x=0, color="r", linestyle="--")
            axes[1, i].set_xlabel("Ошибка")
            axes[1, i].set_ylabel("Частота")
            axes[1, i].set_title("Распределение ошибок")

        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, feature_importance_dict, feature_names, top_n=15):
        """Визуализация важности признаков"""
        n_models = len(feature_importance_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8))

        if n_models == 1:
            axes = [axes]

        for i, (model_name, importance) in enumerate(feature_importance_dict.items()):
            # Сортируем признаки по важности
            indices = np.argsort(importance)[::-1][:top_n]
            top_features = [feature_names[j] for j in indices]
            top_importance = importance[indices]

            axes[i].barh(range(len(top_features)), top_importance[::-1])
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features[::-1])
            axes[i].set_xlabel("Важность признака")
            axes[i].set_title(f"Важность признаков - {model_name}")

        plt.tight_layout()
        plt.show()

    def create_interactive_plot(self, y_true, y_pred, model_name):
        """Создание интерактивного графика с Plotly"""
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Предсказания vs Реальные значения",
                "Распределение ошибок",
            ),
        )

        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=y_true, y=y_pred, mode="markers", name="Предсказания", opacity=0.6
            ),
            row=1,
            col=1,
        )

        # Линия идеальных предсказаний
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                name="Идеальные предсказания",
                line=dict(color="red", dash="dash"),
            ),
            row=1,
            col=1,
        )

        # Гистограмма ошибок
        errors = y_true - y_pred
        fig.add_trace(go.Histogram(x=errors, name="Ошибки", nbinsx=30), row=1, col=2)

        fig.update_layout(title_text=f"Анализ модели: {model_name}", showlegend=True)

        fig.update_xaxes(title_text="Реальные значения", row=1, col=1)
        fig.update_yaxes(title_text="Предсказанные значения", row=1, col=1)
        fig.update_xaxes(title_text="Ошибка", row=1, col=2)
        fig.update_yaxes(title_text="Частота", row=1, col=2)

        fig.show()
