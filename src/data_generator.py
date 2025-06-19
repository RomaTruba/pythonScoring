import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from PyQt5.QtWidgets import QMessageBox

class DataProcessor:
    def __init__(self):
        self.data1 = None
        self.data2 = None
        self.X1_train = None
        self.X1_test = None
        self.X2_train = None
        self.X2_test = None
        self.y1_train = None
        self.y1_test = None
        self.y2_train = None
        self.y2_test = None
        self.X1_train_scaled = None
        self.X1_test_scaled = None
        self.X2_train_scaled = None
        self.X2_test_scaled = None
        self.y1_train_cat = None
        self.y2_train_cat = None
        self.y1_test_cat = None
        self.y2_test_cat = None
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()

        self.preserved_columns = ['age', 'savings']

    def load_bank_data(self):
        try:
            data = pd.read_csv('../src/bank.csv')
            data = self._preprocess_data(data)

            data.to_csv('../tests/bank.csv', index=False)
            self.data1, self.data2 = train_test_split(data, train_size=0.5, random_state=42)
            return True
        except Exception as e:
            QMessageBox.critical(None, "Ошибка", f"Ошибка загрузки данных: {str(e)}")
            return False

    def _preprocess_data(self, data):
        try:

            if 'balance' in data.columns and 'savings' not in data.columns:
                data = data.rename(columns={'balance': 'savings'})


            if 'savings' in data.columns:

                scale_factor = 15
                data['savings'] = data['savings'] * scale_factor

                data['savings'] = np.clip(data['savings'], 0, 500000)

            columns_to_drop = ['education', 'housing', 'contact', 'day', 'month',
                              'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
            data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

            for col in data.columns:
                if col.startswith('marital_status_') or col.startswith('employment_type_'):
                    data[col] = data[col].astype(int)

            if 'default' in data.columns:
                data['overdue_loans'] = data['default'].apply(lambda x: 1 if x == 'yes' else 0)
                data = data.drop(columns=['default'])

            np.random.seed(42)
            num_samples = len(data)

            if 'income' not in self.preserved_columns:
                income_groups = [(20000, 100000, 0.65), (100000, 300000, 0.25), (300000, 500000, 0.05)]
                income_samples = []
                for min_inc, max_inc, prob in income_groups:
                    count = int(num_samples * prob)
                    income_samples.append(np.random.uniform(min_inc, max_inc, size=count))
                remaining = num_samples - sum(len(arr) for arr in income_samples)
                if remaining > 0:
                    income_samples.append(np.random.uniform(20000, 150000, size=remaining))
                income = np.concatenate(income_samples)
                np.random.shuffle(income)
                data['income'] = income[:num_samples]


            if 'credit_rating' not in self.preserved_columns:
                rating_samples = []

                count_high = int(num_samples * 0.3)
                rating_samples.append(np.random.uniform(700, 999, size=count_high))

                count_mid = int(num_samples * 0.5)
                rating_samples.append(np.random.uniform(450, 700, size=count_mid))

                count_low = num_samples - count_high - count_mid
                rating_samples.append(np.random.uniform(0, 450, size=count_low))
                credit_rating = np.concatenate(rating_samples)
                np.random.shuffle(credit_rating)
                data['credit_rating'] = np.clip(credit_rating[:num_samples], 0, 999)

            if 'debt_to_income' not in self.preserved_columns:
                data['debt_to_income'] = np.random.uniform(0.1, 0.8, size=num_samples)

            if 'loan_amount' not in self.preserved_columns:
                if 'loan' in data.columns:
                    data['loan_amount'] = data['loan'].apply(
                        lambda x: np.random.uniform(10000, 500000) if x == 'yes' else np.random.uniform(10000, 200000))
                    data = data.drop(columns=['loan'])
                else:
                    data['loan_amount'] = np.random.uniform(10000, 500000, size=num_samples)

            if 'employment_years' not in self.preserved_columns:
                data['employment_years'] = np.random.randint(0, 40, size=num_samples)

            if 'num_credit_cards' not in self.preserved_columns:
                data['num_credit_cards'] = np.random.randint(0, 5, size=num_samples)

            if 'loan_term' not in self.preserved_columns:
                data['loan_term'] = np.random.choice([6, 12, 24, 36, 60], size=num_samples, p=[0.1, 0.3, 0.3, 0.2, 0.1])

            if 'num_children' not in self.preserved_columns:
                data['num_children'] = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.4, 0.3, 0.2, 0.1])

            if 'requested_loans' not in self.preserved_columns:
                data['requested_loans'] = np.random.poisson(lam=3, size=num_samples)
                data['issued_loans'] = np.random.binomial(data['requested_loans'], 0.5)
            else:
                data['issued_loans'] = np.random.binomial(data['requested_loans'], 0.5)

            data = pd.get_dummies(data, columns=['marital', 'job'], prefix=['marital_status', 'employment_type'])

            def age_risk_weight(age, age_max=data['age'].max()):
                if age < 25:
                    return (25 - age) / (25 - 18)
                elif age > 60:
                    return (age - 60) / (age_max - 60)
                else:
                    return 0

            risk_score = (
                0.05 * (1 - (data['age'] / data['age'].max())) +
                0.20 * (1 - (data['income'] / 500000)) +
                0.25 * (1 - (data['credit_rating'] / 999)) +
                0.15 * data['debt_to_income'] +
                0.10 * (data['loan_amount'] / 500000) -
                0.10 * (data['savings'] / 200000) -
                0.10 * (data['employment_years'] / 40) -
                0.05 * (data['num_credit_cards'] / 5) +
                0.05 * (data['loan_term'] / 60) -
                0.03 * (data['num_children'] / 3) +
                0.15 * (data['overdue_loans'] / (data['issued_loans'] + 1))
            )

            for status in ['married', 'single', 'divorced', 'unknown']:
                col_name = f'marital_status_{status}'
                if col_name in data.columns:
                    risk_score += (0 if status == 'single' else -0.1 if status == 'married' else 0.05 if status == 'divorced' else 0.02) * data[col_name]
            for job_type in ['unemployed', 'services', 'management', 'blue-collar', 'unknown']:
                col_name = f'employment_type_{job_type}'
                if col_name in data.columns:
                    risk_score += (-0.15 if job_type == 'management' else 0.05 if job_type == 'services' else 0.1 if job_type == 'blue-collar' else 0.2) * data[col_name]

            risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
            data['risk_score'] = risk_score

            data['credit_class'] = np.where(risk_score < 0.30, 'Хороший',
                                          np.where(risk_score < 0.55, 'Средний',
                                                   'Плохой'))

            return data
        except Exception as e:
            raise

    def prepare_data(self, app):
        try:
            if self.data1 is None or self.data2 is None:
                QMessageBox.critical(app, "Ошибка", "Данные не загружены!")
                return False

            if 'credit_class' not in self.data1.columns or 'credit_class' not in self.data2.columns:
                QMessageBox.critical(app, "Ошибка", f"Столбец 'credit_class' не найден! Столбцы data1: {self.data1.columns.tolist()}")
                return False

            numeric_columns = [col for col in self.data1.columns if col not in ['marital', 'job', 'risk_score', 'credit_class']]
            X1 = self.data1[numeric_columns]
            X2 = self.data2[numeric_columns]

            y1 = self.data1['credit_class'].map({'Хороший': 0, 'Средний': 1, 'Плохой': 2})
            y2 = self.data2['credit_class'].map({'Хороший': 0, 'Средний': 1, 'Плохой': 2})

            self.X1_train, self.X1_test, self.y1_train, self.y1_test = train_test_split(
                X1, y1, test_size=0.2, random_state=42)
            self.X2_train, self.X2_test, self.y2_train, self.y2_test = train_test_split(
                X2, y2, test_size=0.2, random_state=42)

            self.X1_train_scaled = self.scaler1.fit_transform(self.X1_train)
            self.X1_test_scaled = self.scaler1.transform(self.X1_test)
            self.X2_train_scaled = self.scaler2.fit_transform(self.X2_train)
            self.X2_test_scaled = self.scaler2.transform(self.X2_test)

            self.y1_train_cat = to_categorical(self.y1_train)
            self.y1_test_cat = to_categorical(self.y1_test)
            self.y2_train_cat = to_categorical(self.y2_train)
            self.y2_test_cat = to_categorical(self.y2_test)

            app.X1_train = self.X1_train
            app.X1_test_scaled = self.X1_test_scaled
            app.y1_train = self.y1_train
            app.y1_test = self.y1_test
            app.y1_train_cat = self.y1_train_cat
            app.X2_train = self.X2_train
            app.X2_test_scaled = self.X2_test_scaled
            app.y2_train = self.y2_train
            app.y2_test = self.y2_test
            app.y2_train_cat = self.y2_train_cat
            app.scalers['scaler1'] = self.scaler1
            app.scalers['scaler2'] = self.scaler2

            return True
        except Exception as e:
            QMessageBox.critical(app, "Ошибка", f"Ошибка подготовки данных: {str(e)}")
            return False