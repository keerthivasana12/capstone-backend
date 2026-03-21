from pymongo import MongoClient
from datetime import datetime

# Connect to MongoDB
client = MongoClient("mongodb+srv://Keerthi:keerthi22@capstone.8ggzsop.mongodb.net/?appName=capstone")
db = client["health_ai"]
collection = db["patients"]

def save_patient(patient_id, data, risk, score):
    """
    Save patient record into MongoDB.
    
    Parameters:
        patient_id (str/int): Unique identifier for the patient
        data (dict): Patient health data
        risk (str): Risk category (e.g., 'Low', 'Medium', 'High')
        score (int/float): Risk score
    """
    
    record = {
        "patient_id": patient_id,
        "data": data,
        "risk": risk,
        "score": score,
        "timestamp": datetime.utcnow()
    }

    collection.insert_one(record)
    print(f"Patient {patient_id} record saved successfully!")
