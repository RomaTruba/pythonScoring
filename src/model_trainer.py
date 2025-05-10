import numpy as np
from tensorflow.keras import models, layers
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from PyQt5.QtWidgets import QMessageBox


def train_models(app):
    try:
        app.train_status.clear()
        app.models = {}

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
            history1 = model1.fit(
                app.X1_train_scaled, app.y1_train_cat,
                epochs=50, batch_size=32, validation_split=0.2, verbose=0
            )

            app.models['model1'] = model1
            app.history1 = history1
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
            history2 = model2.fit(
                app.X2_train_scaled, app.y2_train_cat,
                epochs=50, batch_size=32, validation_split=0.2, verbose=0
            )

            app.models['model2'] = model2
            app.history2 = history2
            app.train_status.append("Модель 2 обучена успешно!\n")
        else:
            app.train_status.append("Ошибка: В y2_train недостаточно классов!\n")

        if app.models:
            QMessageBox.information(app, "Успех", "Модели успешно обучены!")
        else:
            QMessageBox.warning(app, "Предупреждение", "Ни одна модель не была обучена!")

    except Exception as e:
        app.train_status.append(f"Ошибка обучения моделей: {str(e)}")
        QMessageBox.critical(app, "Ошибка", f"Ошибка обучения моделей: {str(e)}")