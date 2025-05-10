from src.app import CreditScoringApp


def test_model_training():
    app = CreditScoringApp()
    app.generate_sample_data()
    app.train_models()
    assert 'model1' in app.models, "Модель 1 не обучена"
    assert 'model2' in app.models, "Модель 2 не обучена"
    assert hasattr(app, 'history1'), "История обучения модели 1 отсутствует"
    assert hasattr(app, 'history2'), "История обучения модели 2 отсутствует"
    print("Тест обучения моделей пройден успешно")