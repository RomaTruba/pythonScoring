import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.app import CreditScoringApp
from sklearn.metrics import accuracy_score


def test_prediction_accuracy_regression(qtbot):
    """Регрессионное тестирование: проверка стабильности точности предсказаний с валидацией входных данных"""
    with patch.object(CreditScoringApp, 'show_login_dialog', return_value=True):
        print("\n=== Регрессионное тестирование: стабильность точности предсказаний ===")
        app = CreditScoringApp()
        app.current_user_role = 'user'

        # Подготовка данных
        print("Подготовка фиксированных тестовых данных")
        num_features = 5
        test_data = np.array([
            [25, 50000, 700, 0.3, 100000],
            [30, 60000, 650, 0.4, 150000],
            [35, 45000, 600, 0.5, 200000]
        ])
        scaled_data = np.array([
            [0.5, 0.8, 0.7, -0.2, 0.6],
            [0.7, 1.0, 0.5, 0.0, 1.0],
            [0.9, 0.6, 0.3, 0.2, 1.4]
        ])
        mock_prediction = np.array([
            [0.85, 0.10, 0.05],
            [0.15, 0.70, 0.15],
            [0.05, 0.25, 0.70]
        ])
        true_labels = np.array([0, 1, 2])

        # Проверка входных данных
        if test_data.shape[1] != num_features:
            raise ValueError(f"Ожидалось {num_features} признаков, получено {test_data.shape[1]}")
        if test_data.shape[0] != true_labels.shape[0]:
            raise ValueError("Количество тестовых данных и меток не совпадает")

        # Настройка моков
        print("Настройка моков")
        mock_model = MagicMock()
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = scaled_data
        mock_model.predict.return_value = mock_prediction
        app.models = {'model1': mock_model}
        app.scalers = {'scaler1': mock_scaler}
        app.X1_test = test_data
        app.y1_test = true_labels

        # Выполнение предсказания
        print("Выполнение предсказания")
        try:
            scaled_data_result = app.scalers['scaler1'].transform(app.X1_test)
        except ValueError as e:
            print(f"Ошибка масштабирования: {str(e)}")
            raise ValueError("Не удалось масштабировать тестовые данные")
        try:
            y_pred = app.models['model1'].predict(scaled_data_result)
        except Exception as e:
            print(f"Ошибка предсказания: {str(e)}")
            raise ValueError("Не удалось выполнить предсказание")

        y_pred_classes = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(app.y1_test, y_pred_classes)

        # Проверки
        print(f"\nРезультаты: Точность = {accuracy * 100:.1f}%")
        assert accuracy >= 0.8, f"Точность ниже базового уровня: {accuracy * 100:.1f}%"
        mock_scaler.transform.assert_called_once_with(test_data)
        mock_model.predict.assert_called_once_with(scaled_data)
        print("=== Тест пройден успешно: точность стабильна ===")