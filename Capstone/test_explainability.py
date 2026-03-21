import traceback
from llm.gemma_engine import GemmaEngine

llm = GemmaEngine()

data = {
    "age": 45, "bmi": 32, "bp": 145, 
    "glucose": 150, "cholesterol": 250
}
risk = "High"
recs = "Diet, Exercise"
shap_e = "BMI = 0.5 impact, BP = 0.3 impact"

print(llm.generate_report(data, risk, recs, shap_e))
