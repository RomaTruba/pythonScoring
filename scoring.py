import numpy as np
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QVBoxLayout
from src.model_trainer import ensemble_predict


def calculate_score(app):
    try:
        if not app.models:
            QMessageBox.warning(app, "Предупреждение", "Модели не обучены!")
            return

        try:
            client_data = {
                'age': int(app.client_age.text()),
                'income': int(app.client_income.text()),
                'credit_rating': int(app.client_credit_rating.text()),
                'debt_to_income': float(app.client_debt_income.text()),
                'loan_amount': float(app.client_loan_amount.text()),
                'savings': int(app.client_savings.text()),
                'employment_years': int(app.client_employment.text()),
                'num_credit_cards': int(app.client_credit_cards.text()),
                'loan_term': int(app.client_loan_term.currentText()),
                'num_children': int(app.client_num_children.text()),
                'requested_loans': int(app.client_requested_loans.text()),
                'issued_loans': int(app.client_issued_loans.text()),
                'overdue_loans': int(app.client_overdue_loans.text()),
                'marital_status': app.client_marital.currentText(),
                'employment_type': app.client_employment_type.currentText()
            }

            if not (21 <= client_data['age'] <= 60):
                raise ValueError("Возраст должен быть от 21 до 60 лет")
            if not (0 <= client_data['credit_rating'] <= 999):
                raise ValueError("Кредитный рейтинг должен быть от 0 до 999")

        except ValueError as e:
            app.score_result.setText(f"Ошибка: {str(e)}")
            return