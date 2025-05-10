import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from PyQt5.QtWidgets import QMessageBox


def generate_credit_data(num_samples=1000, random_state=42):
    np.random.seed(random_state)

    age_groups = [
        (21, 25, 0.2),
        (25, 45, 0.6),
        (45, 60, 0.2)
    ]

    age_samples = []
    for min_age, max_age, prob in age_groups:
        count = int(num_samples * prob)
        age_samples.append(np.random.randint(min_age, max_age + 1, size=count))

    age = np.concatenate(age_samples)
    np.random.shuffle(age)

    income_groups = [
        (20000, 150000, 0.8),
        (150000, 300000, 0.15),
        (300000, 500000, 0.05)
    ]

    income_samples = []
    for min_inc, max_inc, prob in income_groups:
        count = int(num_samples * prob)
        income_samples.append(np.random.uniform(min_inc, max_inc, size=count))

    income = np.concatenate(income_samples)
    np.random.shuffle(income)

    rating_groups = [
        (500, 700, 0.6),
        (700, 850, 0.25),
        (300, 500, 0.1),
        (850, 999, 0.05)
    ]

    rating_samples = []
    for min_rt, max_rt, prob in rating_groups:
        count = int(num_samples * prob)
        rating_samples.append(np.random.uniform(min_rt, max_rt, size=count))

    credit_rating = np.concatenate(rating_samples)
    np.random.shuffle(credit_rating)
    credit_rating = np.clip(credit_rating, 0, 999)

    debt_to_income = np.random.uniform(0.1, 0.8, size=num_samples)
    loan_amount = np.random.uniform(10000, 500000, size=num_samples)
    savings = np.random.exponential(scale=50000, size=num_samples)
    employment_years = np.random.randint(0, 40, size=num_samples)
    num_credit_cards = np.random.randint(0, 5, size=num_samples)
    loan_term = np.random.choice([6, 12, 24, 36, 60], size=num_samples, p=[0.1, 0.3, 0.3, 0.2, 0.1])
    num_children = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.4, 0.3, 0.2, 0.1])

    requested_loans = np.random.poisson(lam=2, size=num_samples)
    issued_loans = np.random.binomial(requested_loans, 0.7)
    overdue_loans = np.random.binomial(issued_loans, 0.1)

    marital_status = np.random.choice(
        ['Холост/Не замужем', 'Женат/Замужем', 'Разведен/Разведена', 'Вдовец/Вдова'],
        size=num_samples, p=[0.3, 0.5, 0.15, 0.05]
    )

    employment_type = np.random.choice(
        ['Полная занятость', 'Частичная занятость', 'Самозанятый', 'Безработный'],
        size=num_samples, p=[0.7, 0.15, 0.1, 0.05]
    )

    risk_score = (
            0.15 * (1 - (age / 60)) +
            0.15 * (1 - (income / 500000)) +
            0.25 * (1 - (credit_rating / 999)) +
            0.2 * debt_to_income +
            0.1 * (loan_amount / 500000) -
            0.15 * (savings / 200000) -
            0.1 * (employment_years / 40) -
            0.05 * (num_credit_cards / 5) +
            0.05 * (loan_term / 60) -
            0.03 * (num_children / 3) +
            0.1 * (overdue_loans / (issued_loans + 1))
    )

    marital_impact = {
        'Холост/Не замужем': 0,
        'Женат/Замужем': -0.1,
        'Разведен/Разведена': 0.05,
        'Вдовец/Вдова': 0.02
    }

    employment_impact = {
        'Полная занятость': -0.15,
        'Частичная занятость': 0.05,
        'Самозанятый': 0.1,
        'Безработный': 0.2
    }

    for i in range(num_samples):
        risk_score[i] += marital_impact[marital_status[i]] + employment_impact[employment_type[i]]

    risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
    credit_class = np.where(risk_score < 0.3, 0, np.where(risk_score < 0.7, 1, 2))

    data = pd.DataFrame({
        'age': age,
        'income': income,
        'credit_rating': credit_rating,
        'debt_to_income': debt_to_income,
        'loan_amount': loan_amount,
        'savings': savings,
        'employment_years': employment_years,
        'num_credit_cards': num_credit_cards,
        'loan_term': loan_term,
        'num_children': num_children,
        'requested_loans': requested_loans,
        'issued_loans': issued_loans,
        'overdue_loans': overdue_loans,
        'marital_status': marital_status,
        'employment_type': employment_type,
        'risk_score': risk_score,
        'credit_class': credit_class
    })

    return data


def generate_sample_data(app):
    try:
        app.data1 = generate_credit_data(num_samples=1500, random_state=77)
        app.data2 = generate_credit_data(num_samples=1500, random_state=123)

        app.data_preview.setText(f"Сгенерировано 2 набора данных по 1500 записей\n\nПервые 5 записей набора 1:\n" +
                                 str(app.data1.head(5)))

        QMessageBox.information(app, "Успех", "Данные успешно сгенерированы!")

        if not prepare_data(app):
            return
        from src.model_trainer import train_models
        train_models(app)

    except Exception as e:
        QMessageBox.critical(app, "Ошибка", f"Ошибка генерации данных: {str(e)}")


def prepare_data(app):
    try:
        app.data1 = pd.get_dummies(app.data1, columns=['marital_status', 'employment_type'])
        app.data2 = pd.get_dummies(app.data2, columns=['marital_status', 'employment_type'])

        for col in app.data1.columns:
            if col not in app.data2.columns:
                app.data2[col] = 0
        for col in app.data2.columns:
            if col not in app.data1.columns:
                app.data1[col] = 0

        X1 = app.data1.drop(['risk_score', 'credit_class'], axis=1)
        y1 = app.data1['credit_class']
        app.X1_train, app.X1_test, app.y1_train, app.y1_test = train_test_split(
            X1, y1, test_size=0.2, random_state=42)

        X2 = app.data2.drop(['risk_score', 'credit_class'], axis=1)
        y2 = app.data2['credit_class']
        app.X2_train, app.X2_test, app.y2_train, app.y2_test = train_test_split(
            X2, y2, test_size=0.2, random_state=42)

        app.scalers['scaler1'] = StandardScaler()
        app.X1_train_scaled = app.scalers['scaler1'].fit_transform(app.X1_train)
        app.X1_test_scaled = app.scalers['scaler1'].transform(app.X1_test)

        app.scalers['scaler2'] = StandardScaler()
        app.X2_train_scaled = app.scalers['scaler2'].fit_transform(app.X2_train)
        app.X2_test_scaled = app.scalers['scaler2'].transform(app.X2_test)

        app.y1_train_cat = to_categorical(app.y1_train)
        app.y1_test_cat = to_categorical(app.y1_test)
        app.y2_train_cat = to_categorical(app.y2_train)
        app.y2_test_cat = to_categorical(app.y2_test)

        return True
    except Exception as e:
        QMessageBox.critical(app, "Ошибка", f"Ошибка подготовки данных: {str(e)}")
        return False