import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load dataset
df = pd.read_csv(os.path.join(DATA_DIR, "health_data.csv"))

X = df.drop(["risk", "risk_level"], axis=1)
y = df["risk_level"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model
model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

# Hyperparameter tuning
params = {
    "n_estimators": [200, 300],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1]
}

grid = GridSearchCV(model, params, cv=3, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluation
y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(best_model, "model.pkl")
joblib.dump(le, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")