import pandas as pd
import pickle

# Load the data (must include the 'class' column)
df = pd.read_csv('Dataset1.csv')  # This includes the 'class' column

# Features used for prediction (exclude 'class')
#features = [
 #   "Age", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL",
  #  "BMI", "Glucose", "BloodPressure", "Insulin", "DiabetesPedigreeFunction"
#]
features = [
    "Gender","Age","Urea","Cr","HbA1c","Chol","TG","HDL","LDL","VLDL","BMI","class"
]

X = df[features]  # Input to model
y_actual = df['class']  # Actual labels

# Load the trained model
with open('models/trained_model1.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict
df['predicted_class'] = model.predict(X)

# Compare
comparison_df = df[['class', 'predicted_class']]
print(comparison_df.head(100))
comparison_df.to_csv('data.csv', index=False) 