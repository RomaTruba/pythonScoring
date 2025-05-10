pip install numpy>=1.21.0 pandas>=1.3.0 matplotlib>=3.4.0 seaborn>=0.11.0 scikit-learn>=0.24.0 tensorflow>=2.6.0 PyQt5>=5.15.0
pip install -r requirements.txt
Для запуска тестов выполните: python -m unittest discover tests
__init__.py: пустой файл, обозначающий, что папка src является Python-пакетом.
main.py: точка входа в приложение. Запускает приложение и отображает главное окно.
app.py: основной класс приложения CreditScoringApp, содержащий логику интерфейса и координацию между модулями.
login.py: Диалог авторизации (LoginDialog).
data_generator.py: Генерация тестовых данных и их подготовка.
model_trainer.py: Обучение и оценка моделей машинного обучения.
database.py: Работа с базой данных SQLite.
scoring.py: Логика расчета кредитного скоринга и экспорта данных.
visualization.py: Построение графиков и визуализация данных.
test_data_generation.py: Тесты для генерации данных.
test_model_training.py: Тесты для обучения моделей.
test_prediction.py: Тесты для точности предсказаний.
test_report.py: Тесты для экспорта отчетов.
requirements.txt
Авторизация
Админ: логин admin, пароль admin123
Пользователь: логин user, пароль user123
