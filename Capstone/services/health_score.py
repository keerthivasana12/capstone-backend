def calculate_health_score(data):

    score = 100

    if data["bmi"] > 30: score -= 15
    if data["bp"] > 140: score -= 15
    if data["glucose"] > 140: score -= 15
    if data["cholesterol"] > 240: score -= 10
    if data["smoking"] == 1: score -= 20
    if data["physical_activity"] == 0: score -= 10

    return max(score, 0)