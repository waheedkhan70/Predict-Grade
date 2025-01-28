import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data_path = "D:\\machine_learning_projects\\student_performance.csv"
df = pd.read_csv(data_path)

# Features and target variable
X = df[['AttendanceRate', 'StudyHoursPerWeek', 'FamilyIncome']]
y = df['FinalGrade']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Making predictions for new data
new_data = pd.DataFrame({
    'AttendanceRate': [0.88, 0.92],
    'StudyHoursPerWeek': [8, 10],
    'FamilyIncome': [45000, 60000]
})
predictions = model.predict(new_data)
print("Predicted final grades:", predictions)


import joblib
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")

