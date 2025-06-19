import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QMessageBox
import pandas as pd
import numpy as np

class Visualizer:
    def __init__(self):
        self.figure = plt.figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)

    def plot_analysis(self, app, plot_type):
        try:
            self.figure.clear()

            if plot_type == 'income':
                if app.data1 is None or app.data1.empty:
                    raise ValueError("Данные не загружены или пусты")
                ax = self.figure.add_subplot(111)
                sns.histplot(app.data1['income'], bins=30, kde=True, ax=ax)
                ax.set_title('Распределение доходов клиентов')
                ax.set_xlabel('Доход (руб)')
                ax.set_ylabel('Количество')
                self.canvas.draw()

            elif plot_type == 'age':
                if app.data1 is None or app.data1.empty:
                    raise ValueError("Данные не загружены или пусты")
                ax = self.figure.add_subplot(111)
                sns.histplot(app.data1['age'], bins=20, kde=True, ax=ax)
                ax.set_title('Распределение возраста клиентов')
                ax.set_xlabel('Возраст')
                ax.set_ylabel('Количество')
                self.canvas.draw()

            elif plot_type == 'correlation':
                if app.data1 is None or app.data1.empty:
                    raise ValueError("Данные не загружены или пусты")
                ax = self.figure.add_subplot(111)
                numeric_cols = app.data1.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError("Нет числовых столбцов для построения корреляции")
                corr_matrix = app.data1[numeric_cols].corr()
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
                ax.set_title('Корреляционная матрица')
                self.canvas.draw()

            elif plot_type == 'training':
                if not hasattr(app.ml_model, 'history') or app.ml_model.history is None:
                    QMessageBox.warning(app, "Предупреждение", "Нет данных об обучении ансамблевой модели")
                    return
                ax1 = self.figure.add_subplot(121)
                ax2 = self.figure.add_subplot(122)
                ax1.plot(app.ml_model.history.history['accuracy'], label='Точность (обучение)')
                if 'val_accuracy' in app.ml_model.history.history:
                    ax1.plot(app.ml_model.history.history['val_accuracy'], label='Точность (валидация)')
                ax1.set_title('График точности ансамблевой модели')
                ax1.set_xlabel('Эпоха')
                ax1.set_ylabel('Точность')
                ax1.legend()
                ax2.plot(app.ml_model.history.history['loss'], label='Потери (обучение)')
                if 'val_loss' in app.ml_model.history.history:
                    ax2.plot(app.ml_model.history.history['val_loss'], label='Потери (валидация)')
                ax2.set_title('График потерь ансамблевой модели')
                ax2.set_xlabel('Эпоха')
                ax2.set_ylabel('Потери')
                ax2.legend()
                self.figure.tight_layout()
                self.canvas.draw()

            elif plot_type == 'loan_term':
                if app.data1 is None or app.data1.empty:
                    raise ValueError("Данные не загружены или пусты")
                ax = self.figure.add_subplot(111)
                sns.countplot(x='loan_term', data=app.data1, ax=ax)
                ax.set_title('Распределение сроков кредита')
                ax.set_xlabel('Срок кредита (месяцы)')
                ax.set_ylabel('Количество')
                self.canvas.draw()



            elif plot_type == 'roc_auc':
                if not app.ml_model.ensemble_model:
                    QMessageBox.warning(app, "Предупреждение", "Модели не обучены!")
                    return
                if not hasattr(app, 'y1_test_cat') or not hasattr(app, 'X1_test_scaled'):
                    QMessageBox.warning(app, "Предупреждение", "Тестовые данные для ROC-AUC не подготовлены!")
                    return
                ax = self.figure.add_subplot(111)
                from sklearn.metrics import roc_curve, auc
                y_test_cat = app.y1_test_cat
                X_test_scaled = app.X1_test_scaled
                if y_test_cat is None or X_test_scaled is None:
                    QMessageBox.warning(app, "Предупреждение", "Данные для ансамблевой модели не подготовлены!")
                    return
                ensemble_pred = app.ml_model.predict(X_test_scaled)
                for i in range(3):
                    fpr, tpr, _ = roc_curve(y_test_cat[:, i], ensemble_pred[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'Ансамбль Класс {i} (AUC = {roc_auc:.2f})')
                if hasattr(app.ml_model, 'model1_preds') and app.ml_model.model1_preds is not None:
                    for i in range(3):
                        fpr, tpr, _ = roc_curve(y_test_cat[:, i], app.ml_model.model1_preds[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, linestyle='--', label=f'Модель 1 Класс {i} (AUC = {roc_auc:.2f})')
                if hasattr(app.ml_model, 'model2_preds') and app.ml_model.model2_preds is not None:
                    for i in range(3):
                        fpr, tpr, _ = roc_curve(y_test_cat[:, i], app.ml_model.model2_preds[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, linestyle=':', label=f'Модель 2 Класс {i} (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('ROC кривые для ансамблевой модели и базовых моделей')
                ax.legend(loc="lower right")
                self.canvas.draw()

            elif plot_type == 'credit_stats':
                if app.data1 is None or app.data1.empty:
                    raise ValueError("Данные не загружены или пусты")
                ax = self.figure.add_subplot(111)
                grouped = app.data1.groupby('credit_class')[['requested_loans', 'issued_loans', 'overdue_loans']].mean()
                grouped.plot(kind='bar', ax=ax)
                ax.set_title('Средняя статистика по кредитам в зависимости от класса')
                ax.set_xlabel('Класс кредита')
                ax.set_ylabel('Количество')
                ax.set_xticklabels(['Хороший', 'Средний', 'Плохой'], rotation=0)
                ax.legend(['Запрошенные', 'Выданные', 'Просроченные'])
                self.canvas.draw()

            elif plot_type == 'credit_rating':
                if app.data1 is None or app.data1.empty:
                    raise ValueError("Данные не загружены или пусты")
                if 'credit_class' not in app.data1.columns or 'credit_rating' not in app.data1.columns:
                    raise ValueError("Столбцы 'credit_class' или 'credit_rating' отсутствуют в данных")
                ax = self.figure.add_subplot(111)
                sns.boxplot(x='credit_class', y='credit_rating', data=app.data1, ax=ax, order=['Хороший', 'Средний', 'Плохой'])
                ax.set_title('Распределение кредитного рейтинга по классам')
                ax.set_xlabel('Класс кредита')
                ax.set_ylabel('Рейтинг НБКИ')
                self.canvas.draw()

        except Exception as e:
            QMessageBox.critical(app, "Ошибка", f"Ошибка построения графика: {str(e)}")