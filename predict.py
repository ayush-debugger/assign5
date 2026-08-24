import pickle
import pandas as pd


MODEL_PATH = "model.pkl"


def predict_placement(
    cgpa,
    attendance,
    coding_score,
    projects,
    internship
):

    # Load trained model
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    # Convert input into DataFrame
    student = pd.DataFrame([
        {
            "CGPA": cgpa,
            "Attendance": attendance,
            "Coding_Score": coding_score,
            "Projects": projects,
            "Internship": internship,
        }
    ])

    # Make prediction
    prediction = model.predict(student)[0]

    if prediction == 1:
        return "Placed"
    else:
        return "Not Placed"


if __name__ == "__main__":

    result = predict_placement(
        cgpa=8.7,
        attendance=92,
        coding_score=88,
        projects=3,
        internship=1
    )

    print("Prediction:", result)