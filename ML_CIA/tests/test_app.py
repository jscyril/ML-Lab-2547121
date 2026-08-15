from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_demo_presets_cover_negative_boundary_and_positive_cases():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()
    assert not app.exception
    assert len(app.selectbox[0].options) == 7

    app.selectbox[0].select("Borderline negative").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == "41.3%"
    assert "Screen negative" in app.success[0].value

    app.selectbox[0].select("High health burden").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.metric[0].value == "90.7%"
    assert "Screen positive" in app.error[0].value
