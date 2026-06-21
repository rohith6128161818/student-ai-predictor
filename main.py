import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


print("Real Dataset ML Training Started 🚀")


# Load dataset

df = pd.read_csv(
    "data/students.csv"
)


print(df)



# Features

X = df[
    [
        "hours_studied",
        "attendance",
        "sleep_hours"
    ]
]


# Target

y = df["marks"]



# Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# Train

model = LinearRegression()

model.fit(
    X_train,
    y_train
)



# Accuracy

prediction = model.predict(X_test)

accuracy = r2_score(
    y_test,
    prediction
)


print(
    "Model Accuracy:",
    accuracy
)



# Save model

joblib.dump(
    model,
    "models/student_model.pkl"
)


print(
    "AI Model Saved ✅"
)