import shap
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load model + scaler
model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

# ✅ CORRECT EXPLAINER FOR TREE MODELS
explainer = shap.TreeExplainer(model)

feature_names = [
    "age", "bmi", "bp", "glucose",
    "cholesterol", "smoking", "physical_activity"
]

def get_shap_explanation(data):

    values = np.array([[
        data["age"],
        data["bmi"],
        data["bp"],
        data["glucose"],
        data["cholesterol"],
        data["smoking"],
        data["physical_activity"]
    ]])

    values_scaled = scaler.transform(values)

    shap_values = explainer.shap_values(values_scaled)

    # ✅ Handle multi-class output appropriately
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # shap_values could be (1, n_features, n_classes)
    if len(np.shape(shap_values)) == 3:
        # Sum absolute impacts across all classes for simplicity
        contributions = np.abs(shap_values[0]).sum(axis=1)
    else:
        contributions = shap_values[0]

    feature_impact = list(zip(feature_names, contributions))

    # Sort by importance
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

    top_features = feature_impact[:3]

    explanation = "Top contributing factors:\n"

    for f, v in top_features:
        explanation += f"- {f} (impact: {round(float(v), 3)})\n"

    return explanation