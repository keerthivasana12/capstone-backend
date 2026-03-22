from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from services.risk_engine import predict_risk
from services.health_score import calculate_health_score
from services.recommendation_engine import generate_recommendations
from services.digital_twin import save_patient
from services.explainability import get_shap_explanation
from llm.gemma_engine import GemmaEngine

app = FastAPI()

# ✅ Enable CORS (VERY IMPORTANT for React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load LLM once (FAST - Gemini API)
llm = GemmaEngine()


class Patient(BaseModel):
    patient_id: str
    age: int
    bmi: float
    bp: int
    glucose: int
    cholesterol: int
    smoking: int
    physical_activity: int


@app.get("/")
def home():
    return {"message": "AI Health API Running 🚀"}


@app.post("/analyze")
def analyze(patient: Patient):

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
    # 5. LLM Report (Gemini)
    # -------------------------------
    try:
        report = llm.generate_report(
            patient=data,
            risk=risk,
            recommendations=recs,
            shap_explanation=shap_exp
        )
    except Exception as e:
        report = f"LLM Error: {str(e)}"

    # -------------------------------
    # 6. Save Digital Twin
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
