import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GemmaEngine:

    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-0.5B-Instruct"

        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"

        self.headers = {
            "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
        }

    def generate_report(self, patient, risk, recommendations, shap_explanation):

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Dr. AI, a professional clinical assistant.\n"
                    "You MUST generate a factual medical report using ONLY the provided data.\n"
                    "DO NOT use placeholders like [Name], [Value], etc.\n"
                    "DO NOT repeat sentences.\n"
                    "DO NOT invent diseases.\n"
                    "Be concise, accurate, and clinical."
                )
            },
            {
                "role": "user",
                "content": f"""
PATIENT DATA:
- Age: {patient['age']}
- BMI: {patient['bmi']}
- Blood Pressure: {patient['bp']}
- Glucose: {patient['glucose']}
- Cholesterol: {patient['cholesterol']}

RISK LEVEL: {risk}

KEY FACTORS:
{shap_explanation}

RECOMMENDATIONS:
{recommendations}

Write a structured report with:

1. Clinical Interpretation
2. Risk Explanation
3. Key Contributing Factors
4. Preventive Plan
5. Lifestyle Advice

Rules:
- Use ONLY given values
- No placeholders
- No repetition
- Minimum 8 lines
"""
            }
        ]

        # Convert messages to text (simple join for API)
        prompt = ""
        for m in messages:
            prompt += f"{m['role'].upper()}:\n{m['content']}\n\n"

        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": prompt}
        )

        result = response.json()

        # Handle response safely
        if isinstance(result, list):
            text = result[0]["generated_text"]
        else:
            return "LLM API Error"

        # 🔥 HARD CLEANUP (same as your logic)
        bad_patterns = ["[", "]", "Insert", "Name", "Value"]
        for p in bad_patterns:
            text = text.replace(p, "")

        # 🔥 Remove repeated sentences
        sentences = list(dict.fromkeys(text.split(".")))
        text = ". ".join(sentences)

        return text.strip()