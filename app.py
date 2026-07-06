from pathlib import Path

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("models/model.pkl")

# Load the model once when the server starts
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return jsonify(
        {
            "message": "Welcome to the MLOps API!",
            "status": "running",
        }
    )


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON received"}), 400

    if "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400

    features = data["features"]

    try:
        prediction = model.predict([features])

        return jsonify(
            {
                "prediction": float(prediction[0])
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)