import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = "data/placement.csv"
MODEL_PATH = "model.pkl"
MIN_ACCURACY = 0.80


def train_model(data_path=DATA_PATH, model_path=MODEL_PATH):

    # Load dataset
    data = pd.read_csv(data_path)

    # Required columns
    required_columns = [
        "Student_ID",
        "CGPA",
        "Attendance",
        "Coding_Score",
        "Projects",
        "Internship",
        "Placed",
    ]

    # Validate columns
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Check missing values
    if data[required_columns].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values."
        )

    # Convert categorical values
    features = data.drop(
        columns=["Student_ID", "Placed"]
    )

    features["Internship"] = features["Internship"].map({
        "No": 0,
        "Yes": 1
    })

    target = data["Placed"].map({
        "No": 0,
        "Yes": 1
    })

    # Check unsupported values
    if features.isnull().any().any():
        raise ValueError(
            "Dataset contains unsupported Internship values."
        )

    if target.isnull().any():
        raise ValueError(
            "Dataset contains unsupported Placed values."
        )

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train
    model.fit(x_train, y_train)

    # Evaluate
    predictions = model.predict(x_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"Test accuracy: {accuracy:.2%}")

    # CI/CD quality gate
    if accuracy < MIN_ACCURACY:
        raise ValueError(
            f"Model accuracy {accuracy:.2%} "
            f"is below required 80%."
        )

    # Save model
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Model saved to {model_path}")

    return model, accuracy


if __name__ == "__main__":
    train_model()