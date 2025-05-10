import pytest
from src.app import CreditScoringApp
from unittest.mock import patch


def test_data_generation(qtbot):
    # Перехватываем вызов show_login_dialog, чтобы он не открывал GUI
    with patch.object(CreditScoringApp, 'show_login_dialog', return_value=True):
        app = CreditScoringApp()
        app.current_user_role = 'user'  # Устанавливаем роль вручную
        app.generate_sample_data()
        assert app.data1 is not None, "Data1 не сгенерированы"
        assert app.data2 is not None, "Data2 не сгенерированы"
        assert len(app.data1) == 1500, "Неверное количество записей в data1"
        assert len(app.data2) == 1500, "Неверное количество записей в data2"
        print("Тест генерации данных пройден успешно")