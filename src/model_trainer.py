import os
import numpy as np
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Average
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import resample

class MLModel:
    def __init__(self, input_dim):
        self.ensemble_model = None
        self.model1 = None
        self.model2 = None
        self.input_dim = input_dim
        self.ensemble_file = "ensemble_weights.h5"
        self.model1_preds = None
        self.model2_preds = None
        self.history = None

        if os.path.exists(self.ensemble_file):
            self.ensemble_model = load_model(self.ensemble_file)
        else:
            input_layer = Input(shape=(input_dim,))
            self.model1 = Sequential([
                Dense(64, activation='relu', input_shape=(input_dim,)),
                Dense(32, activation='relu'),
                Dropout(0.3),
                Dense(16, activation='relu'),
                Dropout(0.2),
                Dense(3, activation='softmax')
            ])
            self.model2 = Sequential([
                Dense(32, activation='relu', input_shape=(input_dim,)),
                Dense(16, activation='relu'),
                Dropout(0.2),
                Dense(3, activation='softmax')
            ])
            output1 = self.model1(input_layer)
            output2 = self.model2(input_layer)
            averaged_outputs = Average()([output1, output2])
            self.ensemble_model = Model(inputs=input_layer, outputs=averaged_outputs)
            self.ensemble_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X_train_scaled, y_train_cat, X_val_scaled=None, y_val_cat=None):
        if not os.path.exists(self.ensemble_file):
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            n_samples = X_train_scaled.shape[0]
            idx1 = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
            idx2 = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
            X_train_scaled1, y_train_cat1 = X_train_scaled[idx1], y_train_cat[idx1]
            X_train_scaled2, y_train_cat2 = X_train_scaled[idx2], y_train_cat[idx2]

            self.history = self.ensemble_model.fit(
                X_train_scaled, y_train_cat,
                epochs=100, batch_size=32, validation_split=0.2 if X_val_scaled is None else 0.0,
                validation_data=(X_val_scaled, y_val_cat) if X_val_scaled is not None else None,
                verbose=0, callbacks=[early_stopping]
            )
            if X_val_scaled is not None and y_val_cat is not None:
                self.model1_preds = self.model1.predict(X_val_scaled)
                self.model2_preds = self.model2.predict(X_val_scaled)
            self.ensemble_model.save(self.ensemble_file)
            print("Ансамблевая модель сохранена в ensemble_weights.h5")
        else:
            pass

    def train_models(self, app):
        if app.X1_train_scaled is not None and app.y1_train_cat is not None:
            self.fit(app.X1_train_scaled, app.y1_train_cat, app.X1_test_scaled, app.y1_test_cat)
        if app.X2_train_scaled is not None and app.y2_train_cat is not None:
            self.fit(app.X2_train_scaled, app.y2_train_cat, app.X2_test_scaled, app.y2_test_cat)

    def predict(self, X_scaled):
        return self.ensemble_model.predict(X_scaled)

    def ensemble_predict(self, X_scaled):
        return self.predict(X_scaled)