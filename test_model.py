import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA_PATH = "data/student_placement.csv"
MODEL_PATH = "student_placement_model.pkl"


def test_dataset_exists():
    assert os.path.exists(DATA_PATH)


def test_dataset_valid():
    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "CGPA",
        "Attendance",
        "CodingScore",
        "Projects",
        "Internship",
        "Placement"
    ]

    assert all(column in data.columns for column in required_columns)
    assert data.isnull().sum().sum() == 0


def test_model_exists():
    assert os.path.exists(MODEL_PATH)


def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_prediction():
    model = joblib.load(MODEL_PATH)

    student = pd.DataFrame([{
        "CGPA": 8.5,
        "Attendance": 90,
        "CodingScore": 85,
        "Projects": 3,
        "Internship": 1
    }])

    prediction = model.predict(student)[0]

    assert prediction in [0, 1]


def test_model_accuracy():
    data = pd.read_csv(DATA_PATH)

    X = data.drop("Placement", axis=1)
    y = data["Placement"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nModel Accuracy: {accuracy:.2f}")

    assert accuracy >= 0.80