import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings("ignore")

# Step 1: Dummy Dataset Creation
data = {
    'age': [25, 50, 40, 30, 70, 45, 35, 55, 65, 60],
    'blood_pressure': [120, 140, 130, 110, 150, 135, 125, 145, 155, 148],
    'heart_rate': [70, 80, 75, 65, 85, 78, 72, 82, 88, 85],
    'glucose_level': [90, 180, 150, 100, 200, 160, 110, 190, 210, 195],
    'bmi': [22, 30, 25, 20, 35, 28, 23, 32, 34, 31],
    'smoking_status': [0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    'exercise_level': [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
    'disease_risk': [0, 1, 1, 0, 1, 1, 0, 1, 1, 1],  # 1 = High Risk, 0 = Low Risk
}
df = pd.DataFrame(data)

# Step 2: Splitting the data
X = df[['age', 'blood_pressure', 'heart_rate', 'glucose_level', 'bmi', 'smoking_status', 'exercise_level']]
y = df['disease_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Training the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluating the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 5: Explaining the model using Permutation Importance
perm_importance = permutation_importance(model, X_test, y_test, scoring='accuracy')
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': perm_importance.importances_mean
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:\n", feature_importance)

# Step 6: Predicting for a new patient and generating recommendations
def predict_disease(patient_data):
    """Predict disease risk and provide recommendations."""
    prediction = model.predict([patient_data])[0]
    risk = "High Risk" if prediction == 1 else "Low Risk"
    explanation = feature_importance.to_dict('records')

    recommendations = []
    if prediction == 1:
        if patient_data[1] > 130:
            recommendations.append("Monitor and reduce blood pressure.")
        if patient_data[3] > 140:
            recommendations.append("Manage glucose levels with diet and exercise.")
        if patient_data[5] == 1:
            recommendations.append("Consider quitting smoking.")
        if patient_data[6] == 0:
            recommendations.append("Increase physical activity.")
    else:
        recommendations.append("Maintain current healthy habits.")

    return risk, explanation, recommendations

# Input: Example patient data [age, blood_pressure, heart_rate, glucose_level, bmi, smoking_status, exercise_level]
new_patient = [52, 145, 82, 175, 29, 1, 0]
risk, explanation, recommendations = predict_disease(new_patient)

# Final Output
print("\nNew Patient Prediction:")
print(f"Disease Risk: {risk}")
print("Explanation of Prediction:")
for feature in explanation:
    print(f"- {feature['Feature']}: {feature['Importance']:.4f}")
print("Recommendations:")
for rec in recommendations:
    print(f"- {rec}")