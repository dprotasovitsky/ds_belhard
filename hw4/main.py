"""
Главный скрипт для анализа тональности твитов
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from balancer import DataBalancer
from data_loader import DataLoader
from evaluator import ModelEvaluator
from helpers import predict_sentiment, save_model
from logger import ExperimentLogger
from metrics_analyzer import MetricsAnalyzer
from model_trainer import AdvancedModelTrainer
from preprocessor import TextPreprocessor
from report_generator import ReportGenerator
from settings import config
from vectorizer import TextVectorizer
from visualizer import ResultsVisualizer


def main():
    """Основная функция"""
    print("\U0001f680 ЗАПУСК АНАЛИЗА ТОНАЛЬНОСТИ ТВИТОВ")
    print("=" * 50)

    # Инициализация компонентов
    data_loader = DataLoader()
    preprocessor = TextPreprocessor()
    balancer = DataBalancer()
    vectorizer = TextVectorizer()
    trainer = AdvancedModelTrainer()
    logger = ExperimentLogger(config.EXPERIMENT_LOG_PATH)

    # 1. Загрузка и подготовка данных
    print("\n\U0001f4ca 1. Загрузка данных...")
    df = data_loader.load_data(sample_size=30000)
    print(f" Загружено твитов: {len(df)}")

    # 2. Предобработка текста
    print("\n\U0001f527 2. Предобработка текста...")
    df_clean = preprocessor.preprocess_dataframe(df, "text")
    print(f" После очистки: {len(df_clean)} твитов")

    # 3. Разделение данных
    X_train, X_test, y_train, y_test = data_loader.get_train_test_split(
        df_clean, "cleaned_text", "target"
    )

    # 4. Анализ баланса классов
    print("\n\U00002696 3. Анализ баланса классов...")
    balance_info = balancer.analyze_class_balance(y_train)
    print(f" Распределение: {balance_info['class_distribution']}")
    print(f" Коэффициент дисбаланса: {balance_info['imbalance_ratio']:.2f}")
    print(f" Рекомендация: {balance_info['recommendation']}")

    # 5. Векторизация
    print("\n\U0001f521 4. Векторизация текста...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f" Размерность признаков: {X_train_vec.shape}")

    # 6. Обучение моделей
    print("\n\U0001f916 5. Обучение моделей с подбором параметров...")
    results = trainer.train_models(
        X_train_vec, y_train, X_test_vec, y_test, tune_hyperparameters=True
    )

    # 7. Выбор лучшей модели
    print("\n\U0001f3c6 6. Выбор лучшей модели...")
    best_name, best_result = trainer.get_best_model("f1_score")
    print(f" Лучшая модель: {best_name}")
    print(f" F1-Score: {best_result['metrics']['f1_score']:.4f}")
    print(f" Accuracy: {best_result['metrics']['accuracy']:.4f}")

    # 8. Визуализация результатов
    print("\n\U0001f4c8 7. Визуализация результатов...")
    visualizer = ResultsVisualizer()
    visualizer.plot_model_comparison(results)
    visualizer.plot_confusion_matrices(results, y_test, top_n=3)

    # 9. ROC-кривые
    evaluator = ModelEvaluator(y_test)
    evaluator.plot_roc_curves(results)

    # 10. Детальный отчет
    print("\n\U0001f4cb 8. Генерация отчетов...")
    report_df = ReportGenerator.generate_performance_report(results)
    print(report_df.to_string(index=False))
    ReportGenerator.print_final_report(results, best_name)

    # 11. Сохранение модели
    print("\n\U0001f4be 9. Сохранение лучшей модели...")
    model_data = {
        "model": best_result["model"],
        "vectorizer": vectorizer.vectorizer,
        "preprocessor": preprocessor,
        "metrics": best_result["metrics"],
        "model_name": best_name,
        "feature_names": vectorizer.get_feature_names(),
        "config": {
            "sample_size": len(df),
            "vectorizer_type": config.features.VECTORIZER_TYPE,
            "balance_strategy": config.data.BALANCE_STRATEGY,
        },
    }
    save_model(model_data, config.MODEL_SAVE_PATH)

    # 12. Логирование эксперимента
    logger.log_experiment(
        {
            "best_model": best_name,
            "best_f1_score": best_result["metrics"]["f1_score"],
            "best_accuracy": best_result["metrics"]["accuracy"],
            "total_models_trained": len(results),
            "training_time_total": sum(r["training_time"] for r in results.values()),
        }
    )

    # 13. Демонстрация работы
    print("\n\U0001f50d 10. Тестирование на примерах...")
    test_tweets = [
        "I love this product! It's absolutely amazing!",
        "This is terrible and awful experience.",
        "The service was okay, nothing special.",
        "Highly recommended! Best purchase ever!",
        "Waste of money, completely disappointed.",
    ]
    for tweet in test_tweets:
        result = predict_sentiment(
            tweet, best_result["model"], vectorizer, preprocessor
        )
        icon = "\U0001f60a" if result["sentiment"] == "Positive" else "\U0001f61e"
        print(f"{icon} {result['sentiment']:8} ({result['confidence']:.1%}): {tweet}")
    print("\n\U00002705 АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")


if __name__ == "__main__":
    main()
