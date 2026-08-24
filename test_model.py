import pandas as pd

from train import train_model


def test_model_trains_and_predicts(tmp_path):

    model_path = tmp_path / "model.pkl"

    model, accuracy = train_model(
        model_path=model_path
    )

    # Model should be created
    assert model_path.exists()

    # Model must meet quality requirement
    assert accuracy >= 0.80

    # Test prediction
    sample = pd.DataFrame([
        {
            "CGPA": 8.7,
            "Attendance": 92,
            "Coding_Score": 88,
            "Projects": 3,
            "Internship": 1,
        }
    ])

    prediction = model.predict(sample)

    # Prediction must be 0 or 1
    assert prediction[0] in (0, 1)