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