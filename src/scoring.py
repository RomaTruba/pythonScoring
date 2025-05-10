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

        input_data = {
            'age': client_data['age'],
            'income': client_data['income'],
            'credit_rating': client_data['credit_rating'],
            'debt_to_income': client_data['debt_to_income'],
            'loan_amount': client_data['loan_amount'],
            'savings': client_data['savings'],
            'employment_years': client_data['employment_years'],
            'num_credit_cards': client_data['num_credit_cards'],
            'loan_term': client_data['loan_term'],
            'num_children': client_data['num_children'],
            'requested_loans': client_data['requested_loans'],
            'issued_loans': client_data['issued_loans'],
            'overdue_loans': client_data['overdue_loans']
        }

        for status in ['Холост/Не замужем', 'Женат/Замужем', 'Разведен/Разведена', 'Вдовец/Вдова']:
            input_data[f'marital_status_{status}'] = 1 if client_data['marital_status'] == status else 0

        for emp_type in ['Полная занятость', 'Частичная занятость', 'Самозанятый', 'Безработный']:
            input_data[f'employment_type_{emp_type}'] = 1 if client_data['employment_type'] == emp_type else 0

        df = pd.DataFrame([input_data], columns=app.X1_train.columns)
        X_scaled = app.scalers['scaler1'].transform(df)

        ensemble_pred = ensemble_predict(
            [app.models['model1'], app.models['model2']],
            X_scaled,
            weights=[0.6, 0.4]
        )[0]

        pred_class = np.argmax(ensemble_pred)
        class_names = ['Хороший', 'Средний', 'Плохой']

        if pred_class == 0:
            color = "green"
            comment = "Низкий риск. Кредит может быть одобрен на выгодных условиях."
        elif pred_class == 1:
            color = "orange"
            comment = "Средний риск. Кредит может быть одобрен с ограничениями."
        else:
            color = "red"
            comment = "Высокий риск. Кредит не рекомендуется к выдаче."


        factors = [
            ('age', client_data['age'], 0.15,
             lambda x: min(max((x - 25) / (45 - 25) * 100, 0), 100) if x <= 45 else max(
                 100 - (x - 45) / (60 - 45) * 50, 50)),
            ('income', client_data['income'], 0.15, lambda x: min(x / 300000 * 100, 100)),
            ('credit_rating', client_data['credit_rating'], 0.25, lambda x: x / 999 * 100),
            ('debt_to_income', client_data['debt_to_income'], 0.15, lambda x: 100 - (x * 125)),
            ('savings', client_data['savings'], 0.1, lambda x: min(x / 150000 * 100, 100)),
            ('employment_years', client_data['employment_years'], 0.1, lambda x: min(x / 30 * 100, 100)),
            ('num_children', client_data['num_children'], 0.05, lambda x: 80 + (x * 5)),
            ('overdue_loans', client_data['overdue_loans'], 0.05, lambda x: 100 - (x * 25))
        ]
        client_score = sum(calc(value) * weight for _, value, weight, calc in factors)


        client_id = save_client_data(app, client_data, client_score, class_names[pred_class], comment)
        if not client_id:
            return

        result_text = f"""
        <div style="color:{color}; font-weight:bold; font-size:14px; margin-bottom:10px;">
            Рекомендация: {comment}
        </div>
        <div style="margin-bottom:10px;">
            <b>Вероятности:</b><br>
            - Хороший: {ensemble_pred[0]:.1%}<br>
            - Средний: {ensemble_pred[1]:.1%}<br>
            - Плохой: {ensemble_pred[2]:.1%}
        </div>
        <div style="margin-bottom:10px;">
            <b>Общий рейтинг клиента:</b> {client_score:.1f}/100<br>
            <b>Срок кредита:</b> {client_data['loan_term']} месяцев
        </div>
        """

        app.score_comment.setHtml(result_text)
        app.current_client_id = client_id


        app.right_panel.layout().removeWidget(app.analysis_group)
        app.analysis_group.deleteLater()
        app.analysis_group = display_scoring_breakdown(client_data, client_score, class_names[pred_class])
        app.right_panel.layout().addWidget(app.analysis_group)

    except Exception as e:
        QMessageBox.critical(app, "Ошибка", f"Ошибка расчета: {str(e)}")
        app.score_result.setText("Ошибка расчета")
        app.score_result.setStyleSheet("color: red; font-weight: bold;")


def save_client_data(app, client_data, score, risk_class, comment):
    try:
        client_id = app.client_id.text() or f"NEW_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        app.cursor.execute('''
            INSERT OR REPLACE INTO clients (
                client_id, last_name, first_name, middle_name, age, income, credit_rating,
                debt_to_income, loan_amount, savings, employment_years, num_credit_cards,
                loan_term, num_children, requested_loans, issued_loans, overdue_loans,
                marital_status, employment_type, score, risk_class, comment, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            client_id,
            app.client_last_name.text(),
            app.client_first_name.text(),
            app.client_middle_name.text(),
            client_data['age'],
            client_data['income'],
            client_data['credit_rating'],
            client_data['debt_to_income'],
            client_data['loan_amount'],
            client_data['savings'],
            client_data['employment_years'],
            client_data['num_credit_cards'],
            client_data['loan_term'],
            client_data['num_children'],
            client_data['requested_loans'],
            client_data['issued_loans'],
            client_data['overdue_loans'],
            client_data['marital_status'],
            client_data['employment_type'],
            score,
            risk_class,
            comment,
            datetime.now()
        ))
        app.conn.commit()
        return client_id
    except Exception as e:
        QMessageBox.critical(app, "Ошибка", f"Ошибка сохранения данных клиента: {str(e)}")
        return None


def export_report(app):
    try:
        app.cursor.execute('SELECT * FROM clients')
        data = app.cursor.fetchall()
        columns = [desc[0] for desc in app.cursor.description]
        df = pd.DataFrame(data, columns=columns)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'credit_scoring_report_{timestamp}.csv'
        df.to_csv(filename, index=False)
        QMessageBox.information(app, "Успех", f"Отчет сохранен как {filename}")
    except Exception as e:
        QMessageBox.critical(app, "Ошибка", f"Ошибка экспорта отчета: {str(e)}")


def display_scoring_breakdown(client_data, score, risk_class):
    analysis_group = QGroupBox("Детализация скоринга")
    analysis_layout = QVBoxLayout()

    table = QTableWidget()
    table.setRowCount(8)
    table.setColumnCount(3)
    table.setHorizontalHeaderLabels(['Фактор', 'Значение', 'Вес в скоре'])

    factors = [
        ('Возраст', client_data['age'], 0.15,
         lambda x: min(max((x - 25) / (45 - 25) * 100, 0), 100) if x <= 45 else max(
             100 - (x - 45) / (60 - 45) * 50, 50)),
        ('Доход', client_data['income'], 0.15, lambda x: min(x / 300000 * 100, 100)),
        ('Кредитный рейтинг', client_data['credit_rating'], 0.25, lambda x: x / 999 * 100),
        ('Долг/Доход', client_data['debt_to_income'], 0.15, lambda x: 100 - (x * 125)),
        ('Сбережения', client_data['savings'], 0.1, lambda x: min(x / 150000 * 100, 100)),
        ('Стаж работы', client_data['employment_years'], 0.1, lambda x: min(x / 30 * 100, 100)),
        ('Дети', client_data['num_children'], 0.05, lambda x: 80 + (x * 5)),
        ('Просрочки', client_data['overdue_loans'], 0.05, lambda x: 100 - (x * 25))
    ]

    for row, (name, value, weight, calc) in enumerate(factors):
        table.setItem(row, 0, QTableWidgetItem(name))
        table.setItem(row, 1, QTableWidgetItem(str(value)))
        score_contribution = calc(value) * weight
        table.setItem(row, 2, QTableWidgetItem(f"{score_contribution:.1f}"))

    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    analysis_layout.addWidget(table)
    analysis_group.setLayout(analysis_layout)

    return analysis_group


def clear_database(app):
    reply = QMessageBox.question(app, 'Подтверждение', 'Вы уверены, что хотите очистить базу данных?',
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
    if reply == QMessageBox.Yes:
        try:
            app.cursor.execute('DELETE FROM clients')
            app.conn.commit()
            QMessageBox.information(app, "Успех", "База данных очищена")
        except Exception as e:
            QMessageBox.critical(app, "Ошибка", f"Ошибка очистки базы данных: {str(e)}")