import joblib
import numpy as np

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
def predict_risk(data):

    features = np.array([[
        data["age"],
        data["bmi"],
        data["bp"],
        data["glucose"],
        data["cholesterol"],
        data["smoking"],
        data["physical_activity"]
    ]])

    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)

    return encoder.inverse_transform(pred)[0]