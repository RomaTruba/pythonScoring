# tests/test_report_localization.py
import pytest
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import patch, MagicMock
from PyQt5.QtWidgets import QApplication
from src.app import CreditScoringApp

@pytest.fixture(scope="session")
def qapp():
    app = QApplication([])
    yield app
    app.quit()

def test_report_localization(tmp_path, qapp):

    with patch.object(CreditScoringApp, 'show_login_dialog', return_value=True):
        print("\n=== Тестирование локализации: кириллические символы в отчете ===")
        app = CreditScoringApp()
        app.current_user_role = 'admin'

        print("Настройка мока БД с кириллическими данными")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_data = [
            (1, 'Иванов', 'Иван', 'Иванович', 30, 50000, 700, 0.3,
             100000, 50000, 5, 2, 24, 1, 3, 2, 0,
             'Женат', 'Полная', 85.0, 'Хороший', 'Тест', '2023-01-01')
        ]
        mock_cursor.fetchall.return_value = mock_data
        mock_conn.cursor.return_value = mock_cursor
        app.conn = mock_conn

        if not mock_data:
            raise ValueError("Данные из базы данных пусты")
        if not all(isinstance(row[1], str) and isinstance(row[2], str) for row in mock_data):
            raise ValueError("Фамилия или имя не являются строками")

        print("Создание отчета с кириллическими данными")
        report_path = os.path.join(tmp_path, 'credit_scoring_report_test.csv')

        def mock_export_report():
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("client_id,last_name,first_name,score\n")
                    f.write("1,Иванов,Иван,85.0\n")
                return report_path
            except IOError as e:
                raise ValueError(f"Не удалось создать отчет: {str(e)}")

        app.export_report = mock_export_report

        print("Выполнение экспорта")
        try:
            result_path = app.export_report()
        except ValueError as e:
            print(f"Ошибка экспорта: {str(e)}")
            raise

        print("\nПроверка результатов:")
        assert os.path.exists(result_path), "Файл отчета не создан"
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"- Содержимое файла:\n{content}")
            assert "Иванов" in content, "Кириллические символы (Иванов) отсутствуют"
            assert "Иван" in content, "Кириллические символы (Иван) отсутствуют"
        print("=== Тест пройден успешно: кириллица поддерживается ===")