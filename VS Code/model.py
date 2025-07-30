import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# === Load Training Data ===
time_df = pd.read_csv("C:/workspace-python/third/time_domain_features_train.csv")
nonlinear_df = pd.read_csv("C:/workspace-python/third/heart_rate_non_linear_features_train.csv")
frequency_df = pd.read_csv("C:/workspace-python/third/frequency_domain_features_train.csv")

# === Extract 'label' from 'condition' in nonlinear_df ===
label_map = {
    'no stress': 0,
    'interruption': 1,
    'time pressure': 2
}
if 'condition' not in nonlinear_df.columns:
    raise ValueError("Missing 'condition' column in heart_rate_non_linear_features_train.csv")

y = nonlinear_df['condition'].map(label_map).values

# === Drop non-feature columns from nonlinear_df ===
nonlinear_df = nonlinear_df.drop(columns=['condition', 'datasetId'])

# === Align Rows ===
if not (len(time_df) == len(nonlinear_df) == len(frequency_df)):
    raise ValueError("Mismatch in number of rows across training datasets!")

# === Combine All Features ===
X = pd.concat([time_df, nonlinear_df, frequency_df], axis=1)

# === Check for Non-Numeric Columns ===
non_numeric_columns = X.select_dtypes(exclude=[np.number]).columns
print("Non-numeric columns:", non_numeric_columns)

# === Drop the 'uuid' columns ===
X = X.drop(columns=non_numeric_columns)

# === Check for Data Size ===
print(f"Data Shape: {X.shape}")

# === Check for Missing Values ===
missing_data = X.isnull().sum()
print("Missing data in each column:", missing_data)

# Fill missing values if any
X = X.fillna(X.mean())

# === Scale Features ===
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling complete.")

# === Train RandomForest Model ===
print("Training the model...")
model = RandomForestClassifier(n_estimators=10, random_state=42)  # Reduced estimators
model.fit(X_scaled, y)
print("Model training complete.")

# === Save Model and Scaler ===
joblib.dump(model, "C:/workspace-python/third/stress_model.pkl")
joblib.dump(scaler, "C:/workspace-python/third/stress_scaler.pkl")

print("✅ stress_model.pkl and stress_scaler.pkl saved successfully!")
