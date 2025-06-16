from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTabWidget, QScrollArea, QMessageBox, QTextEdit,
                             QPushButton, QLabel, QLineEdit, QComboBox, QGroupBox, QFormLayout)
from src.login import LoginDialog
from src.database import DatabaseManager
from src.data_generator import DataProcessor
from src.model_trainer import MLModel
from src.visualization import Visualizer
from src.scoring import CreditScorer
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class CreditScoringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система кредитного скоринга")
        self.setGeometry(100, 100, 1400, 900)

        self.data1 = None
        self.data2 = None
        self.models = {}
        self.scalers = {}
        self.current_client_id = None
        self.current_user_role = 'admin'

        self.db_manager = DatabaseManager()
        self.data_processor = DataProcessor()
        self.ml_model = MLModel()
        self.visualizer = Visualizer()
        self.scorer = CreditScorer()

        if not self.show_login_dialog():
            self.close()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        self.create_menu()
        self.create_data_tab()
        self.create_model_tab()
        self.create_scoring_tab()
        self.create_analysis_tab()
        self.create_admin_tab()

        self.load_bank_data()

    def show_login_dialog(self):
        dialog = LoginDialog()
        if dialog.exec_():
            login = dialog.login_input.text()
            password = dialog.password_input.text()
            if login == 'admin' and password == 'admin123':
                return True
            else:
                QMessageBox.critical(self, "Ошибка", "Неверный логин или пароль. Доступ разрешен только администратору.")
                return False
        return False

    def load_bank_data(self):
        try:
            if not self.data_processor.load_bank_data():
                QMessageBox.critical(self, "Ошибка", "Не удалось загрузить данные из Bank.csv!")
                return
            self.data1 = self.data_processor.data1
            self.data2 = self.data_processor.data2
            self.data_preview.setText(f"Загружено {len(self.data1) + len(self.data2)} записей из Bank.csv\n\nПервые 5 записей набора 1:\n" +
                                      str(self.data1.head(5)))
            QMessageBox.information(self, "Успех", "Данные из Bank.csv успешно загружены!")
            if not self.data_processor.prepare_data(self):
                QMessageBox.critical(self, "Ошибка", "Ошибка подготовки данных! Обучение моделей не выполнено.")
                return
            self.X1_train = self.data_processor.X1_train
            self.X1_train_scaled = self.data_processor.X1_train_scaled
            self.X1_test_scaled = self.data_processor.X1_test_scaled
            self.y1_train = self.data_processor.y1_train
            self.y1_test = self.data_processor.y1_test
            self.y1_train_cat = self.data_processor.y1_train_cat
            self.y1_test_cat = self.data_processor.y1_test_cat
            self.X2_train = self.data_processor.X2_train
            self.X2_train_scaled = self.data_processor.X2_train_scaled
            self.X2_test_scaled = self.data_processor.X2_test_scaled
            self.y2_train = self.data_processor.y2_train
            self.y2_test = self.data_processor.y2_test
            self.y2_train_cat = self.data_processor.y2_train_cat
            self.y2_test_cat = self.data_processor.y2_test_cat
            self.train_models()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки данных: {str(e)}")

    def train_models(self):
        try:
            self.train_status.clear()
            self.ml_model.train_models(self)
            self.train_status.append("Модели успешно обучены!")
            QMessageBox.information(self, "Успех", "Модели успешно обучены!")
        except Exception as e:
            self.train_status.append(f"Ошибка обучения моделей: {str(e)}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка обучения моделей: {str(e)}")

    def evaluate_models(self):
        try:
            self.eval_results.clear()

            if 'model1' in self.ml_model.models and self.X1_test_scaled is not None and self.y1_test is not None:
                y1_pred = self.ml_model.models['model1'].predict(self.X1_test_scaled)
                y1_pred_classes = np.argmax(y1_pred, axis=1)
                report1 = classification_report(
                    self.y1_test, y1_pred_classes,
                    target_names=['Хороший', 'Средний', 'Плохой'])
                cm1 = confusion_matrix(self.y1_test, y1_pred_classes)
                self.eval_results.append("=== Модель 1 ===\n")
                self.eval_results.append(report1 + "\n")
                self.eval_results.append(f"Точность: {accuracy_score(self.y1_test, y1_pred_classes):.4f}\n\n")
                self.eval_results.append("Матрица ошибок:\n")
                self.eval_results.append(str(cm1) + "\n\n")

            if 'model2' in self.ml_model.models and self.X2_test_scaled is not None and self.y2_test is not None:
                y2_pred = self.ml_model.models['model2'].predict(self.X2_test_scaled)
                y2_pred_classes = np.argmax(y2_pred, axis=1)
                report2 = classification_report(
                    self.y2_test, y2_pred_classes,
                    target_names=['Хороший', 'Средний', 'Плохой'])
                cm2 = confusion_matrix(self.y2_test, y2_pred_classes)
                self.eval_results.append("=== Модель 2 ===\n")
                self.eval_results.append(report2 + "\n")
                self.eval_results.append(f"Точность: {accuracy_score(self.y2_test, y2_pred_classes):.4f}\n\n")
                self.eval_results.append("Матрица ошибок:\n")
                self.eval_results.append(str(cm2) + "\n\n")

            if self.ml_model.ensemble_model and self.X1_test_scaled is not None and self.y1_test is not None:
                ensemble_pred = self.ml_model.ensemble_model(self.X1_test_scaled)
                ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
                report_ensemble = classification_report(
                    self.y1_test, ensemble_pred_classes,
                    target_names=['Хороший', 'Средний', 'Плохой'])
                cm_ensemble = confusion_matrix(self.y1_test, ensemble_pred_classes)
                self.eval_results.append("=== Ансамблевая модель ===\n")
                self.eval_results.append(report_ensemble + "\n")
                self.eval_results.append(f"Точность: {accuracy_score(self.y1_test, ensemble_pred_classes):.4f}\n\n")
                self.eval_results.append("Матрица ошибок:\n")
                self.eval_results.append(str(cm_ensemble) + "\n\n")

            QMessageBox.information(self, "Успех", "Оценка моделей завершена!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка оценки моделей: {str(e)}")

    def create_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        export_action = file_menu.addAction("Экспорт отчета")
        export_action.triggered.connect(lambda: self.db_manager.export_report(self))
        exit_action = file_menu.addAction("Выход")
        exit_action.triggered.connect(self.close)
        admin_menu = menu_bar.addMenu("Администрирование")
        clear_db_action = admin_menu.addAction("Очистить базу данных")
        clear_db_action.triggered.connect(lambda: self.db_manager.clear_database(self))

    def create_data_tab(self):
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "Данные")
        layout = QVBoxLayout()
        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        layout.addWidget(self.data_preview)
        load_button = QPushButton("Загрузить данные из Bank.csv")
        load_button.clicked.connect(self.load_bank_data)
        layout.addWidget(load_button)
        self.data_tab.setLayout(layout)

    def create_model_tab(self):
        self.model_tab = QWidget()
        self.tabs.addTab(self.model_tab, "Модели")
        layout = QVBoxLayout()
        train_button = QPushButton("Обучить модели")
        train_button.clicked.connect(self.train_models)
        layout.addWidget(train_button)
        self.train_status = QTextEdit()
        self.train_status.setReadOnly(True)
        layout.addWidget(self.train_status)
        evaluate_button = QPushButton("Оценить модели")
        evaluate_button.clicked.connect(self.evaluate_models)
        layout.addWidget(evaluate_button)
        self.eval_results = QTextEdit()
        self.eval_results.setReadOnly(True)
        layout.addWidget(self.eval_results)
        self.model_tab.setLayout(layout)

    def create_scoring_tab(self):
        self.scoring_tab = QWidget()
        self.tabs.addTab(self.scoring_tab, "Скоринг")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll.setWidget(scroll_content)
        main_layout = QHBoxLayout(scroll_content)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        personal_group = QGroupBox("Персональные данные")
        personal_layout = QFormLayout()
        self.client_last_name = QLineEdit()
        personal_layout.addRow("Фамилия:", self.client_last_name)
        self.client_first_name = QLineEdit()
        personal_layout.addRow("Имя:", self.client_first_name)
        self.client_middle_name = QLineEdit()
        personal_layout.addRow("Отчество:", self.client_middle_name)
        self.client_id = QLineEdit()
        personal_layout.addRow("ID клиента:", self.client_id)
        personal_group.setLayout(personal_layout)
        left_layout.addWidget(personal_group)
        finance_group = QGroupBox("Финансовые данные")
        finance_layout = QFormLayout()
        self.client_age = QLineEdit()
        self.client_age.setPlaceholderText("21-60 лет")
        finance_layout.addRow("Возраст:", self.client_age)
        self.client_income = QLineEdit()
        self.client_income.setPlaceholderText("Пример: 60000")
        finance_layout.addRow("Доход (руб/мес):", self.client_income)
        self.client_credit_rating = QLineEdit()
        self.client_credit_rating.setPlaceholderText("0-999")
        finance_layout.addRow("Рейтинг НБКИ:", self.client_credit_rating)
        self.client_debt_income = QLineEdit()
        self.client_debt_income.setPlaceholderText("0.1-0.8")
        finance_layout.addRow("Долг/Доход:", self.client_debt_income)
        self.client_loan_amount = QLineEdit()
        self.client_loan_amount.setPlaceholderText("10000-500000")
        finance_layout.addRow("Сумма кредита (руб):", self.client_loan_amount)
        self.client_savings = QLineEdit()
        self.client_savings.setPlaceholderText("Пример: 50000")
        finance_layout.addRow("Сбережения (руб):", self.client_savings)
        self.client_employment = QLineEdit()
        self.client_employment.setPlaceholderText("0-40 лет")
        finance_layout.addRow("Стаж работы:", self.client_employment)
        self.client_credit_cards = QLineEdit()
        self.client_credit_cards.setPlaceholderText("0-5")
        finance_layout.addRow("Кол-во кредитных карт:", self.client_credit_cards)
        self.client_marital = QComboBox()
        self.client_marital.addItems(['Холост/Не замужем', 'Женат/Замужем', 'Разведен/Разведена', 'Вдовец/Вдова'])
        finance_layout.addRow("Семейное положение:", self.client_marital)
        self.client_employment_type = QComboBox()
        self.client_employment_type.addItems(['Полная занятость', 'Частичная занятость', 'Самозанятый', 'Безработный'])
        finance_layout.addRow("Тип занятости:", self.client_employment_type)
        self.client_loan_term = QComboBox()
        self.client_loan_term.addItems(['6', '12', '24', '36', '60'])
        finance_layout.addRow("Срок кредита (мес):", self.client_loan_term)
        self.client_num_children = QLineEdit()
        self.client_num_children.setPlaceholderText("0-3")
        finance_layout.addRow("Количество детей:", self.client_num_children)
        self.client_requested_loans = QLineEdit()
        self.client_requested_loans.setPlaceholderText("0-10")
        finance_layout.addRow("Запрошенные кредиты:", self.client_requested_loans)
        self.client_issued_loans = QLineEdit()
        self.client_issued_loans.setPlaceholderText("0-10")
        finance_layout.addRow("Выданные кредиты:", self.client_issued_loans)
        self.client_overdue_loans = QLineEdit()
        self.client_overdue_loans.setPlaceholderText("0-10")
        finance_layout.addRow("Просроченные кредиты:", self.client_overdue_loans)
        finance_group.setLayout(finance_layout)
        left_layout.addWidget(finance_group)
        calculate_button = QPushButton("Рассчитать кредитный скоринг")
        calculate_button.clicked.connect(self.calculate_score)
        left_layout.addWidget(calculate_button)
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        result_group = QGroupBox("Результаты скоринга")
        result_layout = QVBoxLayout()
        self.score_result = QLabel("Здесь будет результат")
        self.score_result.setStyleSheet("font-size: 14px;")
        result_layout.addWidget(self.score_result)
        self.score_comment = QTextEdit()
        self.score_comment.setReadOnly(True)
        result_layout.addWidget(self.score_comment)
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        self.analysis_group = QGroupBox("Детализация скоринга")
        right_layout.addWidget(self.analysis_group)
        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(self.right_panel, 3)
        self.scoring_tab.setLayout(QVBoxLayout())
        self.scoring_tab.layout().addWidget(scroll)

    def create_analysis_tab(self):
        self.analysis_tab = QWidget()
        self.tabs.addTab(self.analysis_tab, "Анализ")
        layout = QVBoxLayout()
        buttons_layout = QHBoxLayout()
        income_button = QPushButton("Доходы")
        income_button.clicked.connect(lambda: self.visualizer.plot_analysis(self, 'income'))
        buttons_layout.addWidget(income_button)
        age_button = QPushButton("Возраст")
        age_button.clicked.connect(lambda: self.visualizer.plot_analysis(self, 'age'))
        buttons_layout.addWidget(age_button)
        correlation_button = QPushButton("Корреляция")
        correlation_button.clicked.connect(lambda: self.visualizer.plot_analysis(self, 'correlation'))
        buttons_layout.addWidget(correlation_button)
        training_button = QPushButton("Обучение")
        training_button.clicked.connect(lambda: self.visualizer.plot_analysis(self, 'training'))
        buttons_layout.addWidget(training_button)
        loan_term_button = QPushButton("Сроки кредитов")
        loan_term_button.clicked.connect(lambda: self.visualizer.plot_analysis(self, 'loan_term'))
        buttons_layout.addWidget(loan_term_button)
        roc_button = QPushButton("ROC-AUC")
        roc_button.clicked.connect(lambda: self.visualizer.plot_analysis(self, 'roc_auc'))
        buttons_layout.addWidget(roc_button)
        credit_stats_button = QPushButton("Статистика кредитов")
        credit_stats_button.clicked.connect(lambda: self.visualizer.plot_analysis(self, 'credit_stats'))
        buttons_layout.addWidget(credit_stats_button)
        credit_rating_button = QPushButton("Рейтинг НБКИ")
        credit_rating_button.clicked.connect(lambda: self.visualizer.plot_analysis(self, 'credit_rating'))
        buttons_layout.addWidget(credit_rating_button)
        layout.addLayout(buttons_layout)
        layout.addWidget(self.visualizer.canvas)
        self.analysis_tab.setLayout(layout)

    def create_admin_tab(self):
        self.admin_tab = QWidget()
        self.tabs.addTab(self.admin_tab, "Администрирование")
        layout = QVBoxLayout()
        clear_db_button = QPushButton("Очистить базу данных")
        clear_db_button.clicked.connect(lambda: self.db_manager.clear_database(self))
        layout.addWidget(clear_db_button)
        export_report_button = QPushButton("Экспорт отчета")
        export_report_button.clicked.connect(lambda: self.db_manager.export_report(self))
        layout.addWidget(export_report_button)
        self.admin_tab.setLayout(layout)

    def calculate_score(self):
        try:
            if not self.ml_model.models:
                QMessageBox.warning(self, "Предупреждение", "Модели не обучены!")
                return
            try:
                client_data = {
                    'age': int(self.client_age.text()),
                    'income': int(self.client_income.text()),
                    'credit_rating': int(self.client_credit_rating.text()),
                    'debt_to_income': float(self.client_debt_income.text()),
                    'loan_amount': float(self.client_loan_amount.text()),
                    'savings': int(self.client_savings.text()),
                    'employment_years': int(self.client_employment.text()),
                    'num_credit_cards': int(self.client_credit_cards.text()),
                    'loan_term': int(self.client_loan_term.currentText()),
                    'num_children': int(self.client_num_children.text()),
                    'requested_loans': int(self.client_requested_loans.text()),
                    'issued_loans': int(self.client_issued_loans.text()),
                    'overdue_loans': int(self.client_overdue_loans.text()),
                    'marital_status': self.client_marital.currentText(),
                    'employment_type': self.client_employment_type.currentText()
                }
                if not (21 <= client_data['age'] <= 60):
                    raise ValueError("Возраст должен быть от 21 до 60 лет")
                if not (0 <= client_data['credit_rating'] <= 999):
                    raise ValueError("Кредитный рейтинг должен быть от 0 до 999")
            except ValueError as e:
                self.score_result.setText(f"Ошибка: {str(e)}")
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
            df = pd.DataFrame([input_data], columns=self.data_processor.X1_train.columns)
            X_scaled = self.scalers['scaler1'].transform(df)
            ensemble_pred = self.ml_model.ensemble_model(X_scaled)[0]
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
                 lambda x: min(max((x - 25) / (45 - 25) * 100, 0), 100) if x <= 45 else max(100 - (x - 45) / (60 - 45) * 50, 50)),
                ('income', client_data['income'], 0.15, lambda x: min(x / 300000 * 100, 100)),
                ('credit_rating', client_data['credit_rating'], 0.25, lambda x: x / 999 * 100),
                ('debt_to_income', client_data['debt_to_income'], 0.15, lambda x: 100 - (x * 125)),
                ('savings', client_data['savings'], 0.1, lambda x: min(x / 150000 * 100, 100)),
                ('employment_years', client_data['employment_years'], 0.1, lambda x: min(x / 30 * 100, 100)),
                ('num_children', client_data['num_children'], 0.05, lambda x: 80 + (x * 5)),
                ('overdue_loans', client_data['overdue_loans'], 0.05, lambda x: 100 - (x * 25))
            ]
            client_score = sum(calc(value) * weight for _, value, weight, calc in factors)
            client_id = self.db_manager.save_client(self, client_data, client_score, class_names[pred_class], comment)
            if not client_id:
                return
            resultRacoon = f"""
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
            self.score_comment.setHtml(resultRacoon)
            self.current_client_id = client_id
            self.right_panel.layout().removeWidget(self.analysis_group)
            self.analysis_group.deleteLater()
            self.analysis_group = self.scorer.display_scoring_breakdown(client_data, client_score, class_names[pred_class])
            self.right_panel.layout().addWidget(self.analysis_group)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка расчета: {str(e)}")
            self.score_result.setText("Ошибка расчета")
            self.score_result.setStyleSheet("color: red; font-weight: bold;")

    def closeEvent(self, event):
        self.db_manager.close()
        event.accept()