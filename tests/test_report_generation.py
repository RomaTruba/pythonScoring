# tests/test_report_generation.py
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

def test_report_generation(tmp_path, qapp):
    """Тестирование генерации отчетов с поддержкой Qt"""
    with patch.object(CreditScoringApp, 'show_login_dialog', return_value=True):
        print("\n=== Тест генерации отчетов ===")

        print("Инициализация CreditScoringApp")
        app = CreditScoringApp()
        app.current_user_role = 'admin'

        print("Настройка мока БД")
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (1, 'Иванов', 'Иван', 'Иванович', 30, 50000, 700, 0.3,
             100000, 50000, 5, 2, 24, 1, 3, 2, 0,
             'Женат', 'Полная', 85.0, 'Хороший', 'Тест', '2023-01-01')
        ]
        mock_conn.cursor.return_value = mock_cursor
        app.conn = mock_conn

        print("Подготовка тестового экспорта")
        report_path = os.path.join(tmp_path, 'credit_scoring_report_test.csv')

        def mock_export_report():
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("client_id,last_name,first_name,score\n")
                f.write("1,Иванов,Иван,85.0\n")
            return report_path

        app.export_report = mock_export_report

        print("Выполнение экспорта")
        result_path = app.export_report()

        print("\nПроверка результатов:")
        print(f"- Путь к файлу: {result_path}")

        assert os.path.exists(result_path), "Файл отчета не создан"
        assert 'credit_scoring_report' in result_path, "Некорректное имя файла"

        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print("- Содержимое файла:")
            print(content)
            assert "Иванов" in content, "Фамилия отсутствует в отчете"
            assert "85.0" in content, "Скоринговый балл отсутствует"

        print("\n=== Тест пройден успешно ===")