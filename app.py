from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

# Initialize Flask app with dynamic template folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Base directory of the project
TEMPLATE_DIR = os.path.join(BASE_DIR, "template")      # Template folder path
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")       # Model file path

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load the trained model
model = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input values from the form
    attendance_rate = float(request.form['attendanceRate'])
    study_hours = int(request.form['studyHours'])
    family_income = int(request.form['familyIncome'])

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'AttendanceRate': [attendance_rate],
        'StudyHoursPerWeek': [study_hours],
        'FamilyIncome': [family_income]
    })

    # Predict final grades
    prediction = model.predict(input_data)[0]
    prediction = round(prediction, 2)

    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
