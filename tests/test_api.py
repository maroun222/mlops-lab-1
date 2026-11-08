from math import log
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.food11 import serve

BREAD = Path(__file__).parent / "fixtures" / "bread.jpg"


class FakeModel:
    def predict(self, batch):
        assert batch.shape == (1, 3, 128, 128)
        # The API applies softmax to logits: 90 / (90 + 10) = 0.90.
        return [[log(90.0)] + [0.0] * 10]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_json(client, monkeypatch):
    monkeypatch.setattr(serve, "model", FakeModel())
    with BREAD.open("rb") as image:
        response = client.post("/predict", files={"file": (BREAD.name, image, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "Bread"
    assert body["confidence"] == pytest.approx(0.90)


def test_predict_requires_model(client):
    with BREAD.open("rb") as image:
        response = client.post("/predict", files={"file": (BREAD.name, image, "image/jpeg")})
    assert response.status_code == 503
