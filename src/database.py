import sqlite3
from datetime import datetime
from PyQt5.QtWidgets import QMessageBox
import pandas as pd


class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect('credit_scoring.db')
        self.cursor = self.conn.cursor()
        self.init_database()

    def init_database(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS clients (
                client_id TEXT PRIMARY KEY,
                last_name TEXT,
                first_name TEXT,
                middle_name TEXT,
                age INTEGER,
                income REAL,
                credit_rating INTEGER,
                debt_to_income REAL,
                loan_amount REAL,
                savings REAL,
                employment_years INTEGER,
                num_credit_cards INTEGER,
                loan_term INTEGER,
                num_children INTEGER,
                requested_loans INTEGER,
                issued_loans INTEGER,
                overdue_loans INTEGER,
                marital_status TEXT,
                employment_type TEXT,
                score REAL,
                risk_class TEXT,
                comment TEXT,
                created_at TIMESTAMP
            )
        ''')
        self.conn.commit()

    def save_client(self, app, client_data, score, risk_class, comment):
        try:
            client_id = app.client_id.text() or f"NEW_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.cursor.execute('''
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
            self.conn.commit()
            return client_id
        except Exception as e:
            QMessageBox.critical(app, "Ошибка", f"Ошибка сохранения данных клиента: {str(e)}")
            return None

    def export_report(self, app):
        try:
            self.cursor.execute('SELECT * FROM clients')
            data = self.cursor.fetchall()
            if not data:
                QMessageBox.warning(app, "Предупреждение", "База данных пуста. Нет данных для экспорта.")
                return
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(data, columns=columns)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'credit_scoring_report_{timestamp}.csv'
            df.to_csv(filename, index=False)
            QMessageBox.information(app, "Успех", f"Отчет сохранен как {filename}")
        except Exception as e:
            QMessageBox.critical(app, "Ошибка", f"Ошибка экспорта отчета: {str(e)}")

    def clear_database(self, app):
        reply = QMessageBox.question(app, 'Подтверждение', 'Вы уверены, что хотите очистить базу данных?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                self.cursor.execute('DELETE FROM clients')
                self.conn.commit()
                QMessageBox.information(app, "Успех", "База данных очищена")
            except Exception as e:
                QMessageBox.critical(app, "Ошибка", f"Ошибка очистки базы данных: {str(e)}")

    def close(self):
        self.conn.close()