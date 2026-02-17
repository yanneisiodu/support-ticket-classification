from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client with mocked model loading."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    label_mapping = {0: "hardware", 1: "network", 2: "software"}

    with patch("main.load_model", return_value=(mock_model, mock_tokenizer, label_mapping)):
        with patch("main.model_predict", return_value=np.array([1])):
            from main import app
            with TestClient(app) as c:
                yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_valid_ticket(client):
    response = client.post(
        "/ticket_support_classification",
        json={"message": "My printer is broken"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "ticket_category" in body
    assert "ticket_category_label" in body


def test_classify_empty_message(client):
    response = client.post(
        "/ticket_support_classification",
        json={"message": ""},
    )
    assert response.status_code == 422


def test_classify_missing_message(client):
    response = client.post(
        "/ticket_support_classification",
        json={},
    )
    assert response.status_code == 422


def test_classify_message_too_long(client):
    response = client.post(
        "/ticket_support_classification",
        json={"message": "x" * 5001},
    )
    assert response.status_code == 422
