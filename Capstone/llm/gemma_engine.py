import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GemmaEngine:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found in environment")

        genai.configure(api_key=api_key)

        # 🔥 FAST + STABLE MODEL
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate_report(self, patient, risk, recommendations, shap_explanation):

        prompt = f"""
You are Dr. AI, a professional clinical decision support system.

STRICT RULES:
- Use ONLY given patient data
- DO NOT use placeholders
- DO NOT repeat sentences
- DO NOT hallucinate diseases
- Be medically accurate and professional

----------------------------
PATIENT DATA:
- Age: {patient['age']}
- BMI: {patient['bmi']}
- Blood Pressure: {patient['bp']}
- Glucose: {patient['glucose']}
- Cholesterol: {patient['cholesterol']}

RISK LEVEL: {risk}

KEY FACTORS (Explainable AI):
{shap_explanation}

RECOMMENDATIONS:
{recommendations}
----------------------------

Generate a structured medical report:

1. Clinical Interpretation
2. Risk Explanation
3. Key Contributing Factors
4. Preventive Plan
5. Lifestyle Advice

Minimum 8 lines.
"""

        try:
            response = self.model.generate_content(prompt)

            if not response or not response.text:
                return "LLM Error: Empty response from Gemini"

            text = response.text

        except Exception as e:
            return f"LLM Error: {str(e)}"

        # 🔥 CLEANUP (same logic as before)
        bad_patterns = ["[", "]", "Insert", "Name", "Value"]
        for p in bad_patterns:
            text = text.replace(p, "")

        # 🔥 Remove repetition
        sentences = list(dict.fromkeys(text.split(".")))
        text = ". ".join(sentences)

        return text.strip()
