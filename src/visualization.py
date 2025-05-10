import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from PyQt5.QtWidgets import QMessageBox
from src.model_trainer import ensemble_predict


def plot_analysis(app, plot_type):
    try:
        app.figure.clear()

        if plot_type == 'income':
            ax = app.figure.add_subplot(111)
            sns.histplot(app.data1['income'], bins=30, kde=True, ax=ax)
            ax.set_title('Распределение доходов клиентов')
            ax.set_xlabel('Доход (руб)')
            ax.set_ylabel('Количество')

        elif plot_type == 'age':
            ax = app.figure.add_subplot(111)
            sns.histplot(app.data1['age'], bins=20, kde=True, ax=ax)
            ax.set_title('Распределение возраста клиентов')
            ax.set_xlabel('Возраст')
            ax.set_ylabel('Количество')

        elif plot_type == 'correlation':
            ax = app.figure.add_subplot(111)
            numeric_cols = app.data1.select_dtypes(include=np.number).columns
            corr_matrix = app.data1[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            ax.set_title('Корреляционная матрица')

        elif plot_type == 'training':
            if hasattr(app, 'history1'):
                app.figure.clear()
                ax1 = app.figure.add_subplot(121)
                ax2 = app.figure.add_subplot(122)

                ax1.plot(app.history1.history['accuracy'], label='Точность (обучение)')
                ax1.plot(app.history1.history['val_accuracy'], label='Точность (валидация)')
                ax1.set_title('График точности модели')
                ax1.set_xlabel('Эпоха')
                ax1.set_ylabel('Точность')
                ax1.legend()

                ax2.plot(app.history1.history['loss'], label='Потери (обучение)')
                ax2.plot(app.history1.history['val_loss'], label='Потери (валидация)')
                ax2.set_title('График потерь модели')
                ax2.set_xlabel('Эпоха')
                ax2.set_ylabel('Потери')
                ax2.legend()

                app.figure.tight_layout()
            else:
                QMessageBox.warning(app, "Предупреждение", "Нет данных об обучении модели")

        elif plot_type == 'loan_term':
            ax = app.figure.add_subplot(111)
            sns.countplot(x='loan_term', data=app.data1, ax=ax)
            ax.set_title('Распределение сроков кредита')
            ax.set_xlabel('Срок кредита (месяцы)')
            ax.set_ylabel('Количество')

        elif plot_type == 'roc_auc':
            if not app.models:
                QMessageBox.warning(app, "Предупреждение", "Модели не обучены!")
                return