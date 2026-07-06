import joblib

model = joblib.load("models/model.pkl")

sample = [
    [
        8.3252,
        41.0,
        6.9841,
        1.0238,
        322.0,
        2.5556,
        37.88,
        -122.23,
    ]
]

prediction = model.predict(sample)

print(f"Predicted house value: {prediction[0]:.3f}")