def generate_recommendations(data):

    recs = []

    if data["bmi"] > 30:
        recs.append("Adopt calorie deficit diet + exercise")

    if data["bp"] > 140:
        recs.append("Reduce sodium intake and monitor BP")

    if data["glucose"] > 140:
        recs.append("Control sugar intake")

    if data["smoking"] == 1:
        recs.append("Smoking cessation program")

    if data["physical_activity"] == 0:
        recs.append("At least 30 mins daily activity")

    return recs