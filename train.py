from pathlib import Path

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

print("Loading dataset...")

# Load data
X, y = fetch_california_housing(return_X_y=True)

print(f"Samples : {X.shape[0]}")
print(f"Features: {X.shape[1]}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Training model...")

# Create Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

print("Evaluating model...")

# Predict
predictions = model.predict(X_test)

# Metrics
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Create models folder if it doesn't exist
Path("models").mkdir(exist_ok=True)

# Save model
joblib.dump(model, "models/model.pkl")

print("Model saved to models/model.pkl")