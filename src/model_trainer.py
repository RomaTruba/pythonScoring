from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from PyQt5.QtWidgets import QMessageBox


class MLModel:
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.history1 = None
        self.history2 = None

    def train_models(self, app):
        try:

            required_attrs = ['X1_train_scaled', 'X2_train_scaled', 'y1_train', 'y2_train', 'y1_train_cat', 'y2_train_cat']
            missing_attrs = [attr for attr in required_attrs if not hasattr(app, attr)]
            if missing_attrs:
                app.train_status.append(f"Ошибка: Отсутствуют атрибуты: {', '.join(missing_attrs)}")
                QMessageBox.critical(app, "Ошибка", f"Отсутствуют необходимые данные: {', '.join(missing_attrs)}")
                return

            if len(np.unique(app.y1_train)) >= 3:
                model1 = models.Sequential([
                    layers.Dense(64, activation='relu', input_shape=(app.X1_train_scaled.shape[1],)),
                    layers.Dropout(0.3),
                    layers.Dense(32, activation='relu'),
                    layers.Dropout(0.2),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(3, activation='softmax')
                ])
                model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                self.history1 = model1.fit(
                    app.X1_train_scaled, app.y1_train_cat,
                    epochs=50, batch_size=32, validation_split=0.2, verbose=0
                )
                self.models['model1'] = model1
                app.train_status.append("Модель 1 обучена успешно!\n")
            else:
                app.train_status.append("Ошибка: В y1_train недостаточно классов!\n")

            if len(np.unique(app.y2_train)) >= 3:
                model2 = models.Sequential([
                    layers.Dense(32, activation='relu', input_shape=(app.X2_train_scaled.shape[1],)),
                    layers.Dropout(0.2),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(3, activation='softmax')
                ])
                model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                self.history2 = model2.fit(
                    app.X2_train_scaled, app.y2_train_cat,
                    epochs=50, batch_size=32, validation_split=0.2, verbose=0
                )
                self.models['model2'] = model2
                app.train_status.append("Модель 2 обучена успешно!\n")
            else:
                app.train_status.append("Ошибка: В y2_train недостаточно классов!\n")

            if not self.models:
                QMessageBox.warning(app, "Предупреждение", "Ни одна модель не была обучена!")
                return

            self.ensemble_model = lambda X: self.ensemble_predict(self.models['model1'], self.models['model2'], X, weights=[0.7, 0.3])
            app.train_status.append("Ансамблевая обучена успешно!\n")
        except Exception as e:
            app.train_status.append(f"Ошибка обучения моделей: {str(e)}")
            QMessageBox.critical(app, "Ошибка", f"Ошибка обучения моделей: {str(e)}")

    def ensemble_predict(self, model1, model2, X, weights=None):
        if weights is None:
            weights = [0.5, 0.5]
        preds = [model1.predict(X), model2.predict(X)]
        weighted_preds = np.zeros_like(preds[0])
        for pred, weight in zip(preds, weights):
            weighted_preds += pred * weight

        exp_preds = np.exp(weighted_preds)
        return exp_preds / np.sum(exp_preds, axis=1, keepdims=True)