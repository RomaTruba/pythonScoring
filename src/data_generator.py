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

    def load_bank_data(self):
        try:
            data = pd.read_csv('../tests/bank.csv')
            print("Столбцы в Bank.csv:", data.columns.tolist())
            data = self._preprocess_data(data)
            self.data1, self.data2 = train_test_split(data, train_size=0.5, random_state=42)
            print("Столбцы в data1 после обработки:", self.data1.columns.tolist())
            print("Столбцы в data2 после обработки:", self.data2.columns.tolist())
            return True
        except Exception as e:
            QMessageBox.critical(None, "Ошибка", f"Ошибка загрузки данных: {str(e)}")
            return False

    def _preprocess_data(self, data):
        # Переименование столбцов
        data = data.rename(columns={
            'job': 'employment_type',
            'marital': 'marital_status',
            'balance': 'savings',
        })

        # Удаление ненужных столбцов
        columns_to_drop = ['education', 'housing', 'contact', 'day', 'month',
                           'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

        # Заполнение пропущенных значений
        data['marital_status'] = data['marital_status'].fillna('Холост/Не замужем')
        data['employment_type'] = data['employment_type'].fillna('Безработный')

        # Маппинг категорий
        employment_mapping = {
            'unemployed': 'Безработный',
            'services': 'Частичная занятость',
            'management': 'Полная занятость',
            'blue-collar': 'Полная занятость',
            'self-employed': 'Самозанятый',
            'technician': 'Полная занятость',
            'entrepreneur': 'Самозанятый',
            'admin.': 'Полная занятость',
            'student': 'Частичная занятость',
            'housemaid': 'Частичная занятость',
            'retired': 'Безработный',
            'unknown': 'Безработный'
        }
        data['employment_type'] = data['employment_type'].map(employment_mapping)

        marital_mapping = {
            'married': 'Женат/Замужем',
            'single': 'Холост/Не замужем',
            'divorced': 'Разведен/Разведена',
            'widowed': 'Вдовец/Вдова'
        }
        data['marital_status'] = data['marital_status'].map(marital_mapping)

        # One-hot кодирование
        data = pd.get_dummies(data, columns=['marital_status', 'employment_type'],
                              prefix=['marital_status', 'employment_type'])

        # Обработка просроченных кредитов
        data['overdue_loans'] = data['default'].apply(lambda x: 1 if x == 'yes' else 0)
        data = data.drop(columns=['default'])

        # Генерация дополнительных данных
        np.random.seed(42)
        num_samples = len(data)

        # Возраст
        age_groups = [(21, 25, 0.15), (25, 45, 0.7), (45, 60, 0.15)]
        age_samples = []
        for min_age, max_age, prob in age_groups:
            count = int(num_samples * prob)
            age_samples.append(np.random.uniform(min_age, max_age, size=count))
        remaining = num_samples - sum(len(arr) for arr in age_samples)
        if remaining > 0:
            age_samples.append(np.random.uniform(25, 45, size=remaining))
        age = np.concatenate(age_samples)
        np.random.shuffle(age)
        data['age'] = np.clip(age[:num_samples], 21, 60).astype(int)

        # Доход
        income_groups = [(20000, 150000, 0.8), (150000, 300000, 0.15), (300000, 500000, 0.05)]
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

        # Рейтинг
        rating_groups = [(500, 700, 0.6), (700, 850, 0.25), (300, 500, 0.1), (850, 999, 0.05)]
        rating_samples = []
        for min_rt, max_rt, prob in rating_groups:
            count = int(num_samples * prob)
            rating_samples.append(np.random.uniform(min_rt, max_rt, size=count))
        remaining = num_samples - sum(len(arr) for arr in rating_samples)
        if remaining > 0:
            rating_samples.append(np.random.uniform(500, 700, size=remaining))
        credit_rating = np.concatenate(rating_samples)
        np.random.shuffle(credit_rating)
        data['credit_rating'] = np.clip(credit_rating[:num_samples], 0, 999)

        # Долг/Доход
        data['debt_to_income'] = np.random.uniform(0.1, 0.8, size=num_samples)

        # Сумма кредита
        data['loan_amount'] = data['loan'].apply(
            lambda x: np.random.uniform(10000, 500000) if x == 'yes' else np.random.uniform(10000, 200000))
        data = data.drop(columns=['loan'])

        # Стаж работы
        data['employment_years'] = np.random.randint(0, 40, size=num_samples)

        # Количество кредитных карт
        data['num_credit_cards'] = np.random.randint(0, 5, size=num_samples)

        # Срок кредита
        data['loan_term'] = np.random.choice([6, 12, 24, 36, 60], size=num_samples, p=[0.1, 0.3, 0.3, 0.2, 0.1])

        # Количество детей
        data['num_children'] = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.4, 0.3, 0.2, 0.1])

        # Запрошенные и выданные кредиты
        data['requested_loans'] = np.random.poisson(lam=3, size=num_samples)
        data['issued_loans'] = np.random.binomial(data['requested_loans'], 0.5)
        data['overdue_loans'] = data['overdue_loans']

        # Влияние семейного положения и типа занятости
        marital_impact = {
            'Женат/Замужем': -0.1,
            'Холост/Не замужем': 0,
            'Разведен/Разведена': 0.05,
            'Вдовец/Вдова': 0.02
        }
        employment_impact = {
            'Полная занятость': -0.15,
            'Частичная занятость': 0.05,
            'Самозанятый': 0.1,
            'Безработный': 0.2
        }

        risk_score = (
                0.10 * (1 - (data['age'] / 60)) +  # Уменьшен вес возраста
                0.20 * (1 - (data['income'] / 500000)) +  # Увеличен вес дохода
                0.25 * (1 - (data['credit_rating'] / 999)) +  # Увеличен вес рейтинга
                0.15 * data['debt_to_income'] +  # Уменьшен вес долга
                0.10 * (data['loan_amount'] / 500000) -
                0.15 * (data['savings'] / 200000) -  # Увеличен вес сбережений
                0.05 * (data['employment_years'] / 40) -
                0.05 * (data['num_credit_cards'] / 5) +
                0.05 * (data['loan_term'] / 60) -
                0.03 * (data['num_children'] / 3) +
                0.15 * (data['overdue_loans'] / (data['issued_loans'] + 1))
        )

        for status in marital_impact:
            col_name = f'marital_status_{status}'
            if col_name in data.columns:
                risk_score += marital_impact[status] * data[col_name]
        for emp_type in employment_impact:
            col_name = f'employment_type_{emp_type}'
            if col_name in data.columns:
                risk_score += employment_impact[emp_type] * data[col_name]

        # Нормализация risk_score
        risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
        data['risk_score'] = risk_score

        # Назначение классов с новыми порогами
        data['credit_class'] = np.where(risk_score < 0.30, 'Хороший',  # Увеличен порог с 0.25 до 0.35
                                        np.where(risk_score < 0.55, 'Средний',
                                                 'Плохой'))  # Уменьшен порог с 0.75 до 0.65

        # Проверяем распределение
        print("Распределение credit_class:", data['credit_class'].value_counts().to_dict())

        return data

    def prepare_data(self, app):
        try:
            if self.data1 is None or self.data2 is None:
                QMessageBox.critical(app, "Ошибка", "Данные не загружены!")
                return False

            if 'credit_class' not in self.data1.columns or 'credit_class' not in self.data2.columns:
                QMessageBox.critical(app, "Ошибка", f"Столбец 'credit_class' не найден! Столбцы data1: {self.data1.columns.tolist()}")
                return False

            X1 = self.data1.drop(columns=['risk_score', 'credit_class'], errors='ignore')
            y1 = self.data1['credit_class'].map({'Хороший': 0, 'Средний': 1, 'Плохой': 2})

            X2 = self.data2.drop(columns=['risk_score', 'credit_class'], errors='ignore')
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

            print("Атрибуты установлены: y1_train:", hasattr(app, 'y1_train'), "y2_train:", hasattr(app, 'y2_train'))
            return True
        except Exception as e:
            QMessageBox.critical(app, "Ошибка", f"Ошибка подготовки данных: {str(e)}")
            return False