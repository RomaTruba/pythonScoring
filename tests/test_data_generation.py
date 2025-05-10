import pytest
from src.app import CreditScoringApp
from src.data_generator import generate_credit_data
from unittest.mock import patch


def test_data_generation(qtbot):
    with patch.object(CreditScoringApp, 'show_login_dialog', return_value=True):
        print("Создание CreditScoringApp")
        app = CreditScoringApp()
        print("Установка роли пользователя")
        app.current_user_role = 'user'
        print("Генерация данных")
        # Убрали seed
        app.data1 = generate_credit_data(1500)
        app.data2 = generate_credit_data(1500)
        assert app.data1 is not None, "Data1 не сгенерированы"
        assert app.data2 is not None, "Data2 не сгенерированы"
        assert len(app.data1) == 1500, "Неверное количество записей в data1"
        assert len(app.data2) == 1500, "Неверное количество записей в data2"
        print("Тест генерации данных пройден успешно")