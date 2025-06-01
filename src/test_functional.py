import pytest
import pandas as pd
import numpy as np
from src.data_generator import DataProcessor
from src.app import CreditScoringApp
from src.database import DatabaseManager
from PyQt5.QtWidgets import QApplication


@pytest.fixture
def app():
    app = QApplication([])
    window = CreditScoringApp()
    yield window
    window.close()


@pytest.fixture
def data_processor():
    return DataProcessor()


def test_load_bank_data(data_processor):
    # Подготовка тестового CSV
    test_data = pd.DataFrame({
        'age': np.random.normal(35, 5, 100).clip(21, 60).astype(int),
        'credit_rating': np.random.uniform(300, 999, 100).astype(int),
        'requested_loans': np.random.poisson(2, 100),
        'issued_loans': np.random.binomial(5, 0.7, 100),
        'overdue_loans': np.random.binomial(5, 0.2, 100)
    })
    test_data.to_csv('tests/bank.csv', index=False)

    assert data_processor.load_bank_data()
    assert data_processor.data1 is not None
    assert data_processor.data2 is not None
    assert (21 <= data_processor.data1['age'].min() <= data_processor.data1['age'].max() <= 60)
    assert (0 <= data_processor.data1['credit_rating'].min() <= data_processor.data1['credit_rating'].max() <= 999)
    # Проверка кредитной статистики
    good = data_processor.data1[data_processor.data1['risk_score'] < 0.3]
    bad = data_processor.data1[data_processor.data1['risk_score'] >= 0.7]
    assert good['issued_loans'].mean() > bad['issued_loans'].mean()
    assert bad['overdue_loans'].mean() > good['overdue_loans'].mean()


def test_calculate_score(app):
    # Настройка данных для теста
    app.client_age.setText('30')
    app.client_income.setText('100000')
    app.client_credit_rating.setText('800')
    app.client_debt_income.setText('0.2')
    app.client_loan_amount.setText('200000')
    app.client_savings.setText('50000')
    app.client_employment.setText('10')
    app.client_credit_cards.setText('2')
    app.client_num_children.setText('1')
    app.client_requested_loans.setText('5')
    app.client_issued_loans.setText('4')
    app.client_overdue_loans.setText('0')
    app.client_marital.setCurrentText('Женат/Замужем')
    app.client_employment_type.setCurrentText('Полная занятость')
    app.client_loan_term.setCurrentText('24')

    app.load_bank_data()  # Загружаем данные для масштабирования
    app.train_models()  # Обучаем модели
    app.calculate_score()

    # Проверка результата
    result_text = app.score_result.text()
    assert "Рекомендация" in app.score_comment.toPlainText()
    assert "Хороший" in app.score_comment.toPlainText() or "Средний" in app.score_comment.toPlainText()
    assert float(app.score_comment.toPlainText().split("Общий рейтинг клиента:")[1].split('/')[0]) > 70