"""Flask application for serving a machine learning model."""

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")


@app.route("/")
def home():
    """Return a message indicating the API is running."""
    return "ML API Running"


@app.route("/predict", methods=["POST"])
def predict():
    """Predict the house price from input features."""
    data = request.get_json()
    prediction = model.predict([data["features"]])

    return jsonify({
        "prediction": prediction.tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)