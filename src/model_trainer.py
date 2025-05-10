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

def evaluate_models(app):
    try:
        app.eval_results.clear()

        if 'model1' in app.models:
            y1_pred = app.models['model1'].predict(app.X1_test_scaled)
            y1_pred_classes = np.argmax(y1_pred, axis=1)

            report1 = classification_report(
                app.y1_test, y1_pred_classes,
                target_names=['Хороший', 'Средний', 'Плохой'])

            cm1 = confusion_matrix(app.y1_test, y1_pred_classes)

            app.eval_results.append("=== Модель 1 ===\n")
            app.eval_results.append(report1 + "\n")
            app.eval_results.append(f"Точность: {accuracy_score(app.y1_test, y1_pred_classes):.4f}\n\n")
            app.eval_results.append("Матрица ошибок:\n")
            app.eval_results.append(str(cm1) + "\n\n")

        if 'model2' in app.models:
            y2_pred = app.models['model2'].predict(app.X2_test_scaled)
            y2_pred_classes = np.argmax(y2_pred, axis=1)

            report2 = classification_report(
                app.y2_test, y2_pred_classes,
                target_names=['Хороший', 'Средний', 'Плохой'])

            cm2 = confusion_matrix(app.y2_test, y2_pred_classes)

            app.eval_results.append("=== Модель 2 ===\n")
            app.eval_results.append(report2 + "\n")
            app.eval_results.append(f"Точность: {accuracy_score(app.y2_test, y2_pred_classes):.4f}\n\n")
            app.eval_results.append("Матрица ошибок:\n")
            app.eval_results.append(str(cm2) + "\n\n")

        if len(app.models) >= 2:
            ensemble_pred = ensemble_predict([app.models['model1'], app.models['model2']],
                                            app.X1_test_scaled, weights=[0.6, 0.4])
            ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)

            report_ensemble = classification_report(
                app.y1_test, ensemble_pred_classes,
                target_names=['Хороший', 'Средний', 'Плохой'])

            cm_ensemble = confusion_matrix(app.y1_test, ensemble_pred_classes)

            app.eval_results.append("=== Ансамбль моделей ===\n")
            app.eval_results.append(report_ensemble + "\n")
            app.eval_results.append(f"Точность: {accuracy_score(app.y1_test, ensemble_pred_classes):.4f}\n")
            app.eval_results.append("Матрица ошибок:\n")
            app.eval_results.append(str(cm_ensemble) + "\n")

        QMessageBox.information(app, "Успех", "Оценка моделей завершена!")

    except Exception as e:
        QMessageBox.critical(app, "Ошибка", f"Ошибка оценки моделей: {str(e)}")


def ensemble_predict(models, X, weights=None):
    if weights is None:
        weights = [1 / len(models)] * len(models)

    preds = []
    for model in models:
        pred = model.predict(X)
        preds.append(pred)

    weighted_preds = np.zeros_like(preds[0])
    for pred, weight in zip(preds, weights):
        weighted_preds += pred * weight

    return weighted_preds