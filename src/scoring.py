from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView)

class CreditScorer:
    def display_scoring_breakdown(self, client_data, score, risk_class):
        analysis_group = QGroupBox("Детализация скоринга")
        analysis_layout = QVBoxLayout()
        table = QTableWidget()
        table.setRowCount(8)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(['Фактор', 'Значение', 'Вес в скоре'])
        factors = [
            ('Возраст', client_data['age'], 0.15,
             lambda x: min(max((x - 25) / (45 - 25) * 100, 0), 100) if x <= 45 else max(100 - (x - 45) / (60 - 45) * 50, 50)),
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