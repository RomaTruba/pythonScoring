from src.app import CreditScoringApp


def test_data_generation():
    app = CreditScoringApp()
    app.generate_sample_data()
    assert app.data1 is not None, "Data1 не сгенерированы"
    assert app.data2 is not None, "Data2 не сгенерированы"
    assert len(app.data1) == 1500, "Неверное количество записей в data1"
    assert len(app.data2) == 1500, "Неверное количество записей в data2"
    print("Тест генерации данных пройден успешно")