СИСТЕМА АНАЛИЗА РЕЗУЛЬТАТОВ КРЕДИТНОГО СКОРИНГА

О проекте
Проект представляет собой приложение для кредитного скоринга, которое помогает автоматизировать процесс оценки клиентов. Оно загружает данные из файла Bank.csv, обучает ансамблевую нейронную сеть, рассчитывает кредитный рейтинг и предоставляет визуализацию результатов. Интерфейс разделён на вкладки для работы с данными, моделями, скорингом, анализом и администрированием.


pip install numpy>=1.21.0 pandas>=1.3.0 matplotlib>=3.4.0 seaborn>=0.11.0 scikit-learn>=0.24.0 tensorflow>=2.6.0 PyQt5>=5.15.0О проекте. Проект представляет собой приложение для кредитного скоринга, которое помогает автоматизировать процесс оценки клиентов. Оно загружает данные из файла Bank.csv, обучает ансамблевую нейронную сеть, рассчитывает кредитный рейтинг и предоставляет визуализацию результатов. Интерфейс разделён на вкладки для работы с данными, моделями, скорингом, анализом и администрированием.
pip install -r requirements.txt
__init__.py: пустой файл, обозначающий, что папка src является Python-пакетом.
main.py: Точка входа в приложение. Инициализирует приложение PyQt5 и отображает главное окно.
app.py: Основной класс CreditScoringApp, управляющий интерфейсом и координирующий работу модулей
login.py: Реализует класс LoginDialog для авторизации пользователей.
data_generator.py: отвечает за загрузку данных из bank.csv и предварительную обработку. 
model_trainer.py: Обучение и оценка моделей машинного обучения.
database.py: Работа с базой данных SQLite.
scoring.py: Логика расчета кредитного скоринга и экспорта данных.
visualization.py: Построение графиков и визуализация данных.
test_data_generation.py: Тесты для проверки корректности генерации и обработки данных из bank.csv.
test_model_training.py: Тесты для проверки производительности обучения моделей и метрик.
test_report_localization.py: Тесты для проверки экспорта отчетов с поддержкой кириллицы.
test_gui_scoring_tab.py: Тесты для проверки интерфейса вкладки «Скоринг», включая корректность работы полей ввода и кнопок.
Авторизация
Админ: логин admin, пароль 12345

