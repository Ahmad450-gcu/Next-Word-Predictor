from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_returns_requested_number_of_predictions():
    response = client.post("/predict", json={"text": "the game was released in", "top_k": 5})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 5
    for prediction in body["predictions"]:
        assert "word" in prediction
        assert 0.0 <= prediction["probability"] <= 1.0

def test_predict_rejects_empty_text():
    response = client.post("/predict", json={"text": "", "top_k": 5})
    assert response.status_code == 422

def test_predict_rejects_top_k_too_large():
    response = client.post("/predict", json={"text": "hello world", "top_k": 50})
    assert response.status_code == 422

def test_predict_default_top_k_is_five():
    response = client.post("/predict", json={"text": "the game was released in"})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 5