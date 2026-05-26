from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_predict_valid():
    response = client.post("/predict", json={"wt": 2.62, "hp": 110})
    assert response.status_code == 200
    body = response.json()
    assert "predicted_mpg" in body
    assert isinstance(body["predicted_mpg"], float)
    assert 5.0 < body["predicted_mpg"] < 50.0


def test_predict_missing_field():
    response = client.post("/predict", json={"wt": 2.62})
    assert response.status_code == 422


def test_predict_invalid_type():
    response = client.post("/predict", json={"wt": "heavy", "hp": 110})
    assert response.status_code == 422


def test_predict_negative_value():
    response = client.post("/predict", json={"wt": -1.0, "hp": 110})
    assert response.status_code == 422
