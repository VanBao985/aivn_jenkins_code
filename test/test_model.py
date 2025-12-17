import pytest
import pickle 
import os
import sys
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from train_model import train_and_save_model

class TestModelTraining:
    @pytest.fixture(scope='class', autouse=True)
    def setup_and_teardown(self):
        os.makedirs('models', exist_ok=True)

        yield

    def test_train_model_success(self):
        accuracy = train_and_save_model()
        assert accuracy is not None
        assert isinstance(accuracy, (float, np.floating))
        assert 0 < accuracy <= 1

    def test_model_accuracy_threshold(self):
        accuracy = train_and_save_model()
        assert accuracy >= 0.85, f'Model accuracy: {accuracy:.2f} is below threshold of 85%'

    def test_model_file_created(self):
        train_and_save_model()
        assert os.path.exists('models/iris_model.pkl'), 'Model file was not created'
        assert os.path.getsize('models/iris_model.pkl') > 0, 'Model file is empty'

    def test_model_can_loaded(self):
        train_and_save_model()
        with open ('models/iris_model.pkl', 'rb') as f:
            model = pickle.load(f)
        assert isinstance(model, RandomForestClassifier), 'Loaded model is not a RandomForestClassifier'
        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'predict')

    def test_model_can_predict(self):
        train_and_save_model()
        with open('models/iris_model.pkl', 'rb') as f:
            model = pickle.load(f)
        iris = load_iris()
        X_sample = iris.data[:5]

        predictions = model.predict(X_sample)
        probs = model.predict_proba(X_sample)

        assert len(predictions)==5
        assert all(pred in [0, 1, 2] for pred in predictions)
        assert probs.shape == (5, 3)
        assert all(np.isclose(prob.sum(), 1.0) for prob in probs)

    def test_model_parameters(self):
        train_and_save_model()
        with open('models/iris_model.pkl', 'rb') as f:
            model = pickle.load(f)
        assert model.n_estimators == 100
        assert model.random_state == 42
        assert model.n_classes_ == 3
        assert model.n_features_in_ == 4

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
        