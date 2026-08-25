import joblib
import pandas as pd


# Load trained model
model = joblib.load("student_placement_model.pkl")


# Get student details
cgpa = float(input("Enter CGPA: "))
attendance = float(input("Enter Attendance: "))
coding_score = float(input("Enter Coding Score: "))
projects = int(input("Enter Number of Projects: "))
internship = int(input("Enter Internship (1 = Yes, 0 = No): "))


# Create input data
student = pd.DataFrame([{
    "CGPA": cgpa,
    "Attendance": attendance,
    "CodingScore": coding_score,
    "Projects": projects,
    "Internship": internship
}])


# Make prediction
prediction = model.predict(student)[0]


# Display result
if prediction == 1:
    print("Predicted Placement: PLACED")
else:
    print("Predicted Placement: NOT PLACED")