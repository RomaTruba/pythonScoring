
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app import CreditScoringApp
from src.data_generator import DataProcessor
from unittest.mock import patch
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication

@pytest.fixture(scope="session")
def qapp():
    app = QApplication([])
    yield app
    app.quit()

def test_data_generation_from_bank_csv(qapp):

    with patch.object(CreditScoringApp, 'show_login_dialog', return_value=True):
        print("\n=== Тестирование генерации данных из Bank.csv ===")
        app = CreditScoringApp()
        app.current_user_role = 'user'
        data_processor = DataProcessor()


        assert data_processor.load_bank_data(), "Не удалось загрузить Bank.csv"
        assert data_processor.data1 is not None


        ages = data_processor.data1['age']
        print(f"Диапазон возраста: {ages.min()} - {ages.max()}")
        assert (21 <= ages.min() <= ages.max() <= 60), "Возраст вне диапазона 21-60"

        print("=== Тест пройден успешно: данные из Bank.csv корректны ===")