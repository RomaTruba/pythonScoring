from src.app import CreditScoringApp
from src.scoring import save_client_data
import os


def test_report_generation():
    app = CreditScoringApp()
    app.generate_sample_data()

    client_data = {
        'age': 30,
        'income': 60000,
        'credit_rating': 700,
        'debt_to_income': 0.3,
        'loan_amount': 100000,
        'savings': 50000,
        'employment_years': 5,
        'num_credit_cards': 2,
        'loan_term': 24,
        'num_children': 1,
        'requested_loans': 3,
        'issued_loans': 2,
        'overdue_loans': 0,
        'marital_status': 'Женат/Замужем',
        'employment_type': 'Полная занятость'
    }
    save_client_data(app, client_data, 85.0, 'Хороший', 'Тестовый клиент')

    app.export_report()
    assert any('credit_scoring_report' in f for f in os.listdir('.')), "Отчет не создан"
    print("Тест генерации отчета пройден успешно")