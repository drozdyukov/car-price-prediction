from pathlib import Path

from streamlit.testing.v1 import AppTest

from car_price import ui

ROOT = Path(__file__).resolve().parents[1]


def test_ui_can_predict_for_both_transmissions(monkeypatch, small_config):
    from car_price.train import train

    train(small_config)
    monkeypatch.setattr(ui, "load_config", lambda: small_config)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.success) == 1
    app.selectbox[2].select("Автоматическая")
    app.button[0].click().run()
    assert not app.exception
    assert len(app.success) == 1


def test_ui_reports_missing_model_in_russian(monkeypatch, small_config):
    monkeypatch.setattr(ui, "load_config", lambda: small_config)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert "Модель не найдена" in app.error[0].value
