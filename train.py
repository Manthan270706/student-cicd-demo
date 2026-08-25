import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("data/student_placement.csv")

# Validate required columns
required_columns = [
    "CGPA",
    "Attendance",
    "CodingScore",
    "Projects",
    "Internship",
    "Placement"
]

if not all(column in data.columns for column in required_columns):
    raise ValueError("Invalid dataset columns")

# Check missing values
if data.isnull().sum().sum() > 0:
    raise ValueError("Dataset contains missing values")

# Separate features and target
X = data.drop("Placement", axis=1)
y = data["Placement"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Define 3 models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

# Train and compare models
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"{name}: {accuracy:.2f}")


# Select best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_accuracy = results[best_model_name]

print("\nBest Model:", best_model_name)
print("Best Accuracy:", f"{best_accuracy:.2f}")

# Save best model
joblib.dump(best_model, "student_placement_model.pkl")

print("Model saved as student_placement_model.pkl")