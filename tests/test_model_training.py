# tests/test_model_training.py
import pytest
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import patch
from src.app import CreditScoringApp
from src.data_generator import DataProcessor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from PyQt5.QtWidgets import QApplication

@pytest.fixture(scope="session")
def qapp():
    app = QApplication([])
    yield app
    app.quit()

def test_model_training_performance(qapp):
    """Тестирование производительности: измерение времени обучения модели с использованием данных из Bank.csv"""
    with patch.object(CreditScoringApp, 'show_login_dialog', return_value=True):
        print("\n=== Тестирование производительности: обучение модели на данных из Bank.csv ===")
        app = CreditScoringApp()
        app.current_user_role = 'user'

        # Шаг 1: Загрузка данных из Bank.csv
        print("Шаг 1: Загрузка данных из Bank.csv")
        data_processor = DataProcessor()
        assert data_processor.load_bank_data(), "Не удалось загрузить данные из Bank.csv"
        assert data_processor.data1 is not None, "Данные data1 не загружены"

        # Подготовка данных для обучения
        app.data1 = data_processor.data1

        # Выбор признаков и целевой переменной
        # Предполагаем, что в data1 есть следующие столбцы, основанные на вашем коде data_generator.py
        feature_columns = ['age', 'credit_rating', 'requested_loans', 'issued_loans', 'overdue_loans']
        target_column = 'credit_class'

        # Проверяем наличие необходимых столбцов
        for col in feature_columns + [target_column]:
            assert col in app.data1.columns, f"Столбец {col} отсутствует в данных"

        # Преобразуем категориальную целевую переменную в числовую
        class_mapping = {'Хороший': 0, 'Средний': 1, 'Плохой': 2}
        app.data1[target_column] = app.data1[target_column].map(class_mapping)
        assert app.data1[target_column].notna().all(), "Целевая переменная содержит пропуски"

        X = app.data1[feature_columns].values
        y = app.data1[target_column].values

        # Проверка размеров данных
        num_samples, num_features = X.shape
        num_classes = len(class_mapping)
        print(f"Размер данных: {num_samples} образцов, {num_features} признаков, {num_classes} классов")
        if X.size == 0 or y.size == 0:
            raise ValueError("Входные данные пусты")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Размеры X и y не совпадают")

        # Шаг 2: Масштабирование данных
        print("Шаг 2: Масштабирование данных")
        app.scalers = {'scaler1': StandardScaler()}
        try:
            app.X1_train_scaled = app.scalers['scaler1'].fit_transform(X)
        except ValueError as e:
            print(f"Ошибка масштабирования: {str(e)}")
            raise ValueError("Не удалось масштабировать данные")
        app.y1_train = y

        # Шаг 3: Создание модели
        print("Шаг 3: Создание модели")
        model = Sequential([
            Input(shape=(num_features,)),
            Dense(8, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Шаг 4: Обучение модели с измерением времени
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