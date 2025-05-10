import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget, QTextEdit,
    QPushButton, QGroupBox, QFormLayout, QLineEdit, QComboBox, QLabel,
    QScrollArea, QTableWidget, QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from src.login import LoginDialog
from src.database import init_database
from src.data_generator import generate_sample_data
from src.model_trainer import train_models, evaluate_models
from src.scoring import calculate_score, save_client_data, export_report, display_scoring_breakdown, clear_database
from src.visualization import plot_analysis


class CreditScoringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система кредитного скоринга")
        self.setGeometry(100, 100, 1400, 900)

        # Инициализация данных и моделей
        self.data1 = None
        self.data2 = None
        self.models = {}
        self.scalers = {}
        self.current_client_id = None
        self.current_user_role = None

        # Создаем фигуры для графиков
        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)

        # Инициализация базы данных
        self.conn, self.cursor = init_database()

        # Показываем диалог авторизации
        if not self.show_login_dialog():
            sys.exit()

        # Создание главного виджета
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Основной макет
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Создание вкладок
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Создание интерфейса
        self.create_menu()
        self.create_data_tab()
        self.create_model_tab()
        self.create_scoring_tab()
        self.create_analysis_tab()
        if self.current_user_role == 'admin':
            self.create_admin_tab()

        # Генерация тестовых данных
        generate_sample_data(self)

    def show_login_dialog(self):
        dialog = LoginDialog()
        if dialog.exec_():
            login = dialog.login_input.text()
            password = dialog.password_input.text()
            if login == 'admin' and password == 'admin123':
                self.current_user_role = 'admin'
                return True
            elif login == 'user' and password == 'user123':
                self.current_user_role = 'user'
                return True
            else:
                QMessageBox.critical(self, "Ошибка", "Неверный логин или пароль")
                return False
        return False

    def create_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("Файл")
        if self.current_user_role == 'admin':
            export_action = file_menu.addAction("Экспорт отчета")
            export_action.triggered.connect(lambda: export_report(self))
        exit_action = file_menu.addAction("Выход")
        exit_action.triggered.connect(self.close)

        if self.current_user_role == 'admin':
            admin_menu = menu_bar.addMenu("Администрирование")
            clear_db_action = admin_menu.addAction("Очистить базу данных")
            clear_db_action.triggered.connect(lambda: clear_database(self))

    def create_data_tab(self):
        self.data_tab = QWidget()
        self.tabs.addTab(self.data_tab, "Данные")

        layout = QVBoxLayout()
        self.data_preview = QTextEdit()
        self.data_preview.setReadOnly(True)
        layout.addWidget(self.data_preview)

        generate_button = QPushButton("Сгенерировать данные")
        generate_button.clicked.connect(lambda: generate_sample_data(self))
        layout.addWidget(generate_button)

        self.data_tab.setLayout(layout)

    def create_model_tab(self):
        self.model_tab = QWidget()
        self.tabs.addTab(self.model_tab, "Модели")

        layout = QVBoxLayout()

        train_button = QPushButton("Обучить модели")
        train_button.clicked.connect(lambda: train_models(self))
        layout.addWidget(train_button)

        self.train_status = QTextEdit()
        self.train_status.setReadOnly(True)
        layout.addWidget(self.train_status)

        evaluate_button = QPushButton("Оценить модели")
        evaluate_button.clicked.connect(lambda: evaluate_models(self))
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
        self.client_marital.addItems(['Холост/Не замужем', 'Женат/Замужем',
                                      'Разведен/Разведена', 'Вдовец/Вдова'])
        finance_layout.addRow("Семейное положение:", self.client_marital)

        self.client_employment_type = QComboBox()
        self.client_employment_type.addItems(['Полная занятость', 'Частичная занятость',
                                              'Самозанятый', 'Безработный'])
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
        calculate_button.clicked.connect(lambda: calculate_score(self))
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
        income_button.clicked.connect(lambda: plot_analysis(self, 'income'))
        buttons_layout.addWidget(income_button)

        age_button = QPushButton("Возраст")
        age_button.clicked.connect(lambda: plot_analysis(self, 'age'))
        buttons_layout.addWidget(age_button)

        correlation_button = QPushButton("Корреляция")
        correlation_button.clicked.connect(lambda: plot_analysis(self, 'correlation'))
        buttons_layout.addWidget(correlation_button)

        training_button = QPushButton("Обучение")
        training_button.clicked.connect(lambda: plot_analysis(self, 'training'))
        buttons_layout.addWidget(training_button)

        loan_term_button = QPushButton("Сроки кредитов")
        loan_term_button.clicked.connect(lambda: plot_analysis(self, 'loan_term'))
        buttons_layout.addWidget(loan_term_button)

        roc_button = QPushButton("ROC-AUC")
        roc_button.clicked.connect(lambda: plot_analysis(self, 'roc_auc'))
        buttons_layout.addWidget(roc_button)

        credit_stats_button = QPushButton("Статистика кредитов")
        credit_stats_button.clicked.connect(lambda: plot_analysis(self, 'credit_stats'))
        buttons_layout.addWidget(credit_stats_button)

        credit_rating_button = QPushButton("Рейтинг НБКИ")
        credit_rating_button.clicked.connect(lambda: plot_analysis(self, 'credit_rating'))
        buttons_layout.addWidget(credit_rating_button)

        layout.addLayout(buttons_layout)

        layout.addWidget(self.canvas)

        self.analysis_tab.setLayout(layout)

    def create_admin_tab(self):
        self.admin_tab = QWidget()
        self.tabs.addTab(self.admin_tab, "Администрирование")

        layout = QVBoxLayout()

        clear_db_button = QPushButton("Очистить базу данных")
        clear_db_button.clicked.connect(lambda: clear_database(self))
        layout.addWidget(clear_db_button)

        export_report_button = QPushButton("Экспорт отчета")
        export_report_button.clicked.connect(lambda: export_report(self))
        layout.addWidget(export_report_button)

        self.admin_tab.setLayout(layout)

    def closeEvent(self, event):
        self.conn.close()
        event.accept()