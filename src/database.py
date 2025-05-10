import sqlite3


def init_database():
    conn = sqlite3.connect('credit_scoring.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clients (
            client_id TEXT PRIMARY KEY,
            last_name TEXT,
            first_name TEXT,
            middle_name TEXT,
            age INTEGER,
            income REAL,
            credit_rating INTEGER,
            debt_to_income REAL,
            loan_amount REAL,
            savings REAL,
            employment_years INTEGER,
            num_credit_cards INTEGER,
            loan_term INTEGER,
            num_children INTEGER,
            requested_loans INTEGER,
            issued_loans INTEGER,
            overdue_loans INTEGER,
            marital_status TEXT,
            employment_type TEXT,
            score REAL,
            risk_class TEXT,
            comment TEXT,
            created_at TIMESTAMP
        )
    ''')
    conn.commit()
    return conn, cursor