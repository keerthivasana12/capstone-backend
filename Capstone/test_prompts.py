from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

prompts = [
    "Write a friendly letter to a patient. Patient is 45 years old, high risk. Advise them to diet and exercise.",
    "As an AI doctor, write a friendly diagnosis message to a 45-year-old patient telling them they have High Risk health due to high BMI and Glucose, and they need to exercise."
]

for p in prompts:
    inputs = tokenizer(p, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150, min_new_tokens=30, do_sample=True, temperature=0.7)
    print("PROMPT:", p)
    print("OUTPUT:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("-" * 50)
