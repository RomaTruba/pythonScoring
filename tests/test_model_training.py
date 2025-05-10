import pytest
import numpy as np
import pandas as pd
import time
from unittest.mock import patch
from src.app import CreditScoringApp
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


def test_model_training_performance(qtbot):
    """Тестирование производительности: измерение времени обучения модели с проверкой входных данных"""
    with patch.object(CreditScoringApp, 'show_login_dialog', return_value=True):
        print("\n=== Тестирование производительности: обучение модели ===")
        app = CreditScoringApp()
        app.current_user_role = 'user'


        print("Шаг 1: Подготовка тестовых данных")
        np.random.seed(42)
        num_samples = 1000
        num_features = 5
        num_classes = 3
        try:
            X = np.random.rand(num_samples, num_features)
            y = np.random.randint(0, num_classes, num_samples)
            if X.size == 0 or y.size == 0:
                raise ValueError("Входные данные пусты")
            if X.shape[0] != y.shape[0]:
                raise ValueError("Размеры X и y не совпадают")
        except ValueError as e:
            print(f"Ошибка подготовки данных: {str(e)}")
            raise

        app.data1 = pd.DataFrame(X)
        app.data1['credit_class'] = y


        print("Шаг 2: Масштабирование данных")
        app.scalers = {'scaler1': StandardScaler()}
        try:
            app.X1_train_scaled = app.scalers['scaler1'].fit_transform(X)
        except ValueError as e:
            print(f"Ошибка масштабирования: {str(e)}")
            raise ValueError("Не удалось масштабировать данные")
        app.y1_train = y


        print("Шаг 3: Создание модели")
        model = Sequential([
            Input(shape=(num_features,)),
            Dense(8, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


        print("Шаг 4: Обучение модели с измерением времени")
        start_time = time.time()
        try:
            history = model.fit(app.X1_train_scaled, app.y1_train, epochs=5, batch_size=32, verbose=0)
        except Exception as e:
            print(f"Ошибка обучения: {str(e)}")
            raise ValueError("Не удалось обучить модель")
        duration = time.time() - start_time

        app.models = {'model1': model}


        print(f"\nРезультаты: Время обучения = {duration:.2f} секунд")
        assert duration < 10, f"Обучение заняло слишком долго: {duration:.2f} секунд"
        assert 'loss' in history.history, "История обучения не содержит loss"
        assert len(history.history['loss']) == 5, "Ожидалось 5 эпох обучения"
        print("=== Тест пройден успешно: обучение выполнено быстро ===")