import sys
from PyQt5.QtWidgets import QApplication
from src.app import CreditScoringApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = CreditScoringApp()
    window.show()
    sys.exit(app.exec_())