from fastapi import FastAPI
from pydantic import BaseModel

from services.risk_engine import predict_risk
from services.health_score import calculate_health_score
from services.recommendation_engine import generate_recommendations
from services.digital_twin import save_patient
from services.explainability import get_shap_explanation
from llm.gemma_engine import GemmaEngine

app = FastAPI()

llm = None  # 🔥 FIXED

class Patient(BaseModel):
    patient_id: str
    age: int
    bmi: float
    bp: int
    glucose: int
    cholesterol: int
    smoking: int
    physical_activity: int


@app.post("/analyze")
def analyze(patient: Patient):

    global llm

    data = patient.dict()

    # -------------------------------
    # 1. ML Prediction
    # -------------------------------
    risk = predict_risk(data)

    # -------------------------------
    # 2. Health Score
    # -------------------------------
    score = calculate_health_score(data)

    # -------------------------------
    # 3. Recommendations
    # -------------------------------
    recs = generate_recommendations(data)

    # -------------------------------
    # 4. SHAP Explainability
    # -------------------------------
    try:
        shap_exp = get_shap_explanation(data)
    except Exception:
        shap_exp = "Explainability not available"

    # -------------------------------
    # 🔥 5. LAZY LOAD LLM (GPU SAFE)
    # -------------------------------
    if llm is None:
        print("🔄 Loading LLM on GPU...")
        llm = GemmaEngine()
        print("✅ LLM Ready")

    # -------------------------------
    # 6. Generate Report
    # -------------------------------
    report = llm.generate_report(
        patient=data,
        risk=risk,
        recommendations=recs,
        shap_explanation=shap_exp
    )

    # -------------------------------
    # 7. Save Digital Twin
    # -------------------------------
    try:
        save_patient(data["patient_id"], data, risk, score)
    except Exception:
        pass

    # -------------------------------
    # FINAL RESPONSE
    # -------------------------------
    return {
        "risk": risk,
        "health_score": score,
        "recommendations": recs,
        "explainability": shap_exp,
        "ai_report": report
    }