from typing import Any, Dict

import joblib


def save_model(model_data: Dict[str, Any], filepath: str):
    """Сохранение модели"""
    joblib.dump(model_data, filepath)
    print(f"\U0001f4be Модель сохранена: {filepath}")


def load_model(filepath: str) -> Dict[str, Any]:
    """Загрузка модели"""
    return joblib.load(filepath)


def predict_sentiment(text: str, model, vectorizer, preprocessor):
    """Предсказание тональности для нового текста"""
    cleaned_text = preprocessor.clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])

    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[1] if prediction == 1 else probability[0]

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "probability_negative": probability[0],
        "probability_positive": probability[1],
    }
