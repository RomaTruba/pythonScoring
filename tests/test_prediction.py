from src.app import CreditScoringApp
from sklearn.metrics import accuracy_score
import numpy as np


def test_prediction_accuracy():
    app = CreditScoringApp()
    app.generate_sample_data()
    app.train_models()
    app.evaluate_models()

    y1_pred = app.models['model1'].predict(app.X1_test_scaled)
    y1_pred_classes = np.argmax(y1_pred, axis=1)
    accuracy1 = accuracy_score(app.y1_test, y1_pred_classes)

    assert accuracy1 > 0.7, f"Точность модели 1 слишком низкая: {accuracy1}"
    print("Тест точности предсказаний пройден успешно")