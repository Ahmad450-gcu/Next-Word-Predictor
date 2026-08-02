import json
from fastapi.testclient import TestClient
from app.core.config import settings
from app.main import app

client = TestClient(app)

def get_vocab_size():
    with open(settings.tokenizer_path) as f:
        word_index = json.load(f)
    return len(word_index)

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["vocab_size"] == get_vocab_size()