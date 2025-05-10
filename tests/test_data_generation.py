import pytest
from src.app import CreditScoringApp
from src.data_generator import generate_credit_data
from unittest.mock import patch


def test_data_generation_negative_input(qtbot):
    """Негативное тестирование: проверка обработки некорректного размера данных"""
    with patch.object(CreditScoringApp, 'show_login_dialog', return_value=True):
        print("\n=== Негативное тестирование: генерация данных с некорректным размером ===")
        app = CreditScoringApp()
        app.current_user_role = 'user'
        print("Попытка генерации данных с отрицательным размером")
        try:
            app.data1 = generate_credit_data(-1)
            pytest.fail("Ожидалась ошибка для отрицательного размера")
        except ValueError as e:
            print(f"Получена ожидаемая ошибка: {str(e)}")
            assert "negative dimensions are not allowed" in str(e).lower(), "Неверное сообщение об ошибке"
        print("=== Тест пройден успешно: ошибка обработана корректно ===")