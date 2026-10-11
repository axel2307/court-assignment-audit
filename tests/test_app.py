"""Exercise the classroom state boundaries through Streamlit's actual widgets."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


def open_app():
    app = Path(__file__).resolve().parents[1] / "app.py"
    return AppTest.from_file(str(app), default_timeout=30).run()


def text(app):
    return " ".join(item.value for item in app.markdown)


def assert_fixed_case(app, original=None):
    assert not app.exception
    records = app.dataframe[0].value
    assert len(records) == 60
    assert records["Expediente"].nunique() == 60
    assert set(records["Juzgado"]) == set("ABCDEFGHIJKL")
    assert (records["Juzgado"] == "A").sum() == 10
    if original is not None:
        pd.testing.assert_frame_equal(records, original)
    return records.copy()


def test_initial_lesson_has_fixed_case_and_only_two_learning_controls():
    app = open_app()
    assert_fixed_case(app)
    assert app.title[0].value == "¿Diez veces en el mismo juzgado?"
    assert len(app.header) == 6
    assert app.header[0].value == "1. Empecemos por los expedientes"
    assert app.header[-1].value == "6. Ahora probá vos"
    assert len(app.tabs) == 0
    assert len(app.selectbox) == len(app.number_input) == 0
    assert len(app.button) == len(app.slider) == 1
    assert app.slider(key="practice_count").value == 10
    assert app.session_state["simulation_batch_number"] == 0
    assert len(app.get("download_button")) == 2
    assert len(app.get("file_uploader")) == 0
    assert "1,57071156 %" in text(app)
    assert "2,57542547 %" in text(app)
    assert "no es la probabilidad de fraude" in text(app)
    assert "muestra aleatoria" in " ".join(item.value for item in app.caption)


def test_practice_extremes_leave_the_original_case_and_conclusion_unchanged():
    app = open_app()
    original = assert_fixed_case(app)
    examples = [(0, "100,00 %"), (15, "0,0092 %"), (5, "56,72 %"), (10, "2,58 %")]
    for threshold, expected in examples:
        app.slider(key="practice_count").set_value(threshold).run()
        assert_fixed_case(app, original)
        assert f"{threshold} o más de 60 → {expected}" in text(app)
        assert "10 en lugar de 5 esperados en promedio" in text(app)
        assert "2,57542547 %" in text(app)
        assert app.session_state["simulation_batch_number"] == 0
        if threshold == 15:
            assert "sus barras casi no se ven" in " ".join(item.value for item in app.caption)


def test_simulation_survives_slider_reruns_and_repeated_clicks_get_distinct_batches():
    app = open_app()
    original = assert_fixed_case(app)
    app.button(key="simulate").click().run()
    assert_fixed_case(app, original)
    assert app.session_state["simulation_batch_number"] == 1
    assert "**26 de los 1.000 repartos**" in text(app)
    assert "Semilla 271828" in " ".join(item.value for item in app.caption)
    for threshold in (15, 0, 10):
        app.slider(key="practice_count").set_value(threshold).run()
        assert_fixed_case(app, original)
        assert app.session_state["simulation_batch_number"] == 1
        assert "**26 de los 1.000 repartos**" in text(app)
        assert "Semilla 271828" in " ".join(item.value for item in app.caption)
    app.button(key="simulate").click().run()
    assert_fixed_case(app, original)
    assert app.session_state["simulation_batch_number"] == 2
    assert "lote 2" in text(app)
    assert "Semilla 271829" in " ".join(item.value for item in app.caption)
    app.run()
    assert app.session_state["simulation_batch_number"] == 2
    assert_fixed_case(app, original)


def test_a_new_session_restarts_practice_and_simulation_without_changing_the_fixture():
    first = open_app()
    original = assert_fixed_case(first)
    first.button(key="simulate").click().run()
    first.slider(key="practice_count").set_value(15).run()
    fresh = open_app()
    assert_fixed_case(fresh, original)
    assert fresh.session_state["simulation_batch_number"] == 0
    assert fresh.slider(key="practice_count").value == 10
