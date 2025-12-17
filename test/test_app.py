import pytest
import os 
from fastapi.testclient import TestClient
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from app import app, load_model

load_model() 

client = TestClient(app)

def test_root():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json()['status'] == 'running'
    assert 'message' in response.json()

def test_health():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    assert response.json()['model_loaded'] is True

def test_predict():
    input = {
        'sepal_length': 5.1, 
        'sepal_width': 3.2,
        'petal_length': 1.2,
        'petal_width': 0.2
    }

    response = client.post('/predict', json=input)
    assert response.status_code == 200
    data = response.json()
    assert 'prediction' in data
    assert 'species' in data
    assert 'confidence' in data
    assert data['species'] in ['setosa', 'versicolor', 'virginica']
    assert 0.0 <= data['confidence'] <= 1.0

def test_predict_invalid_input():
    invalid_input = {
        'sepal_length': 5.1,
        'sepal_width': 3.0
    }

    response = client.post('/predict', json=invalid_input)
    assert response.status_code == 422  # validation error

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])