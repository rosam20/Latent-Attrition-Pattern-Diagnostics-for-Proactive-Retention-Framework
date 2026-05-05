
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Example dummy dataset (age, gender, tenure, monthlycharge, churn)
data = {
    "age": [25, 45, 35, 50, 23, 40],
    "gender": [1, 0, 1, 0, 1, 0],  # 1=female, 0=male
    "tenure": [12, 60, 24, 100, 5, 80],
    "monthlycharge": [70, 120, 60, 100, 40, 110],
    "churn": [1, 0, 1, 0, 1, 0]  # 1=Yes, 0=No
}

df = pd.DataFrame(data)

X = df.drop("churn", axis=1)
y = df["churn"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train simple model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save both model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ model.pkl and scaler.pkl created successfully!")

import joblib
model = joblib.load("model.pkl")
print(type(model))

try:
    import shap
    print(shap.__version__)
except ImportError:
    print("shap is not installed; skipping shap import.")

