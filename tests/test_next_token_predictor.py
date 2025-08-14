import pytest
from unittest.mock import patch, MagicMock

import next_token_predictor

class DummyResponse:
    class Choices:
        class LogProbs:
            top_logprobs = [{
                "day": -0.1,
                "world": -0.5,
                ".": -1.5,
            }]
        logprobs = LogProbs()
        def __init__(self):
            self.logprobs = [Choices()]
    choices = [Choices()]

    def __init__(self):
        self.choices = self.choices

@pytest.fixture
def app(qtbot):
    root = next_token_predictor.tk.Tk()
    app = next_token_predictor.TokenPredictorApp(root)
    yield app
    root.destroy()

def test_update_predictions_mocks_openai(monkeypatch):
    # Patch the openai client to use our dummy response
    dummy_response = DummyResponse()

    with patch.object(next_token_predictor.client.completions, 'create', return_value=dummy_response):
        root = next_token_predictor.tk.Tk()
        app = next_token_predictor.TokenPredictorApp(root)
        app.prompt_entry.insert(0, "It's a beautiful day, let's go to the")
        app.update_predictions()
        # After update_predictions, check that prediction_labels are filled
        texts = [lbl.cget('text') for lbl in app.prediction_labels if lbl.cget('text')]
        assert any("day" in t for t in texts)
        assert any("world" in t for t in texts)
        assert any("." in t for t in texts)
        root.destroy()

def test_temperature_entry(app):
    # Check initial temperature value
    assert app.temperature.get() == 0.2
    app.temp_entry.delete(0, 'end')
    app.temp_entry.insert(0, '0.8')
    app.temp_entry.event_generate('<Return>')
    assert float(app.temp_entry.get()) == 0.8
