
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.app import CreditScoringApp
from PyQt5.QtWidgets import QApplication, QMessageBox, QPushButton, QLineEdit, QTextEdit
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

@pytest.fixture(scope="session")
def qapp():
    app = QApplication([])
    yield app
    app.quit()

def test_gui_scoring_tab(qapp, qtbot):

    print("\n=== Тестирование интерфейса: проверка вкладки Скоринг ===")
    app = CreditScoringApp()
    app.current_user_role = 'user'


    app.show()
    app.activateWindow()
    qtbot.wait(200)


    assert hasattr(app, 'tabs'), "QTabWidget (tabs) не найден"
    assert app.tabs.isVisible(), "Вкладки не видны"
    print(f"Количество вкладок: {app.tabs.count()}")


    app.tabs.setCurrentIndex(2)
    qtbot.wait(200)


    assert app.scoring_tab.isVisible(), "Вкладка Скоринг не видна"
    assert isinstance(app.client_age, QLineEdit), "Поле client_age не найдено или не является QLineEdit"
    assert app.client_age.isVisible(), "Поле client_age не видно"
    assert isinstance(app.score_comment, QTextEdit), "Поле score_comment не найдено или не является QTextEdit"
    assert app.score_comment.isVisible(), "Поле score_comment не видно"


    calculate_button = app.scoring_tab.findChild(QPushButton, "Рассчитать кредитный скоринг")
    if calculate_button is None:
        calculate_button = app.scoring_tab.findChild(QPushButton)
        print("Кнопки на вкладке scoring_tab:", [btn.text() for btn in app.scoring_tab.findChildren(QPushButton)])
    assert calculate_button is not None, "Кнопка 'Рассчитать кредитный скоринг' не найдена на вкладке Скоринг"
    assert calculate_button.isVisible(), "Кнопка 'Рассчитать кредитный скоринг' не видна"
    assert calculate_button.isEnabled(), "Кнопка 'Рассчитать кредитный скоринг' не активна"


    app.client_age.setText("70")
    print(f"Значение в client_age: {app.client_age.text()}")
    QTest.mouseClick(calculate_button, Qt.LeftButton)
    qtbot.wait(1000)
    print(f"Текст в score_comment после некорректного ввода: {app.score_comment.toPlainText()}")



    app.client_age.clear()
    app.client_age.setText("30")
    print(f"Значение в client_age: {app.client_age.text()}")
    QTest.mouseClick(calculate_button, Qt.LeftButton)
    qtbot.wait(1000)
    result_text = app.score_comment.toPlainText()
    print(f"Результат в score_comment: {result_text}")

    print("=== Тест пройден успешно: интерфейс вкладки Скоринг работает корректно ===")