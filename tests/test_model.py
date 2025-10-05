import pytest
import numpy as np
import model   # ✅ import the whole module, not functions , #loaded best model from mlflow

feature_vals = [222, 400, 2025] #Max power, Mileage and Year

@pytest.fixture
def mock_model(monkeypatch):
    class DummyModel:
        def predict(self, X):
            return np.array([1])  # always return 1 for testing
    monkeypatch.setattr(model, "load_model", lambda: DummyModel())

def test_model_input_shape(mock_model):
    X = model.get_X(*feature_vals)
    assert X.shape[1] == 35
    assert np.issubdtype(X.dtype, np.number)

def test_model_output_shape(mock_model):
    model_instance = model.load_model()   # patched to DummyModel
    X = model.get_X(*feature_vals)
    y_pred = model_instance.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (1,)