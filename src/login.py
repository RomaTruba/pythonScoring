from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLineEdit, QPushButton, QMessageBox)


class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Авторизация")
        self.setFixedSize(300, 200)

        layout = QVBoxLayout()

        self.login_input = QLineEdit()
        self.login_input.setPlaceholderText("Логин")
        layout.addWidget(self.login_input)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Пароль")
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_input)

        login_button = QPushButton("Войти")
        login_button.clicked.connect(self.try_login)
        layout.addWidget(login_button)

        self.setLayout(layout)

    def try_login(self):
        login = self.login_input.text().strip()
        password = self.password_input.text().strip()


        if not login or not password:
            QMessageBox.warning(self, "Ошибка", "Логин и пароль не могут быть пустыми!")
            return


        if login == "admin" and password == "12345":
            print("Авторизация успешна!")
            self.accept()
        else:
            print("Неверный пароль! ")
            QMessageBox.critical(self, "Ошибка", "Неверный логин или пароль!")