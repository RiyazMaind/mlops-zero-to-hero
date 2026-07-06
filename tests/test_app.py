import json

from app import app

client = app.test_client()


def test_home():
    """Test the home endpoint."""

    response = client.get("/")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "running"


def test_health():
    """Test the health endpoint."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "healthy"


def test_predict():
    """Test prediction endpoint."""

    sample = {
        "features": [
            8.3252,
            41.0,
            6.9841,
            1.0238,
            322.0,
            2.5556,
            37.88,
            -122.23
        ]
    }

    response = client.post(
        "/predict",
        data=json.dumps(sample),
        content_type="application/json",
    )

    assert response.status_code == 200

    prediction = response.get_json()

    assert "prediction" in prediction

    assert isinstance(prediction["prediction"], float)

def test_predict_missing_features():
    """Test prediction with invalid JSON."""

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "error" in data

def test_predict_wrong_feature_length():
    """Prediction should fail with wrong number of features."""

    response = client.post(
        "/predict",
        json={
            "features": [1, 2, 3]
        },
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "error" in data