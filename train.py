"""Train and save a Linear Regression model."""

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import joblib

X, y = fetch_california_housing(return_X_y=True)

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model saved.")