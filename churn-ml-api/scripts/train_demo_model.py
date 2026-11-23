"""Create a small demo model so the API runs out of the box.

REPLACE the output with your real Task 04 champion:
    models/champion_churn_model.joblib
(train it with the same scikit-learn version as in requirements.txt).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "tenure": rng.integers(1, 72, n),
    "MonthlyCharges": rng.uniform(20, 120, n).round(2),
    "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .2]),
})
logit = -0.04 * df.tenure + 0.025 * df.MonthlyCharges + (df.Contract == "Month-to-month") * 1.2 - 1.0
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), ["tenure", "MonthlyCharges"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Contract"]),
    ])),
    ("clf", LogisticRegression(max_iter=1000)),
]).fit(df, y)

joblib.dump(pipe, "models/champion_churn_model.joblib")
print("Saved models/champion_churn_model.joblib")
