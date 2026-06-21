import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime


# Load model

model = joblib.load(
    "models/student_model.pkl"
)


st.set_page_config(
    page_title="Student AI Predictor",
    page_icon="🤖"
)


st.title("🤖 Student Performance Predictor")


# Inputs

hours = st.slider(
    "Study hours",
    1,
    10,
    5
)


attendance = st.slider(
    "Attendance %",
    50,
    100,
    80
)


sleep = st.slider(
    "Sleep hours",
    1,
    12,
    7
)



if st.button("Predict Marks 🚀"):


    data = pd.DataFrame(
        [[hours, attendance, sleep]],
        columns=[
            "hours_studied",
            "attendance",
            "sleep_hours"
        ]
    )


    prediction = model.predict(data)[0]


    prediction = min(prediction,100)



    st.success(
        f"Predicted Marks: {prediction:.2f}"
    )



    # Save history

    new_record = pd.DataFrame(
        {
            "time":[datetime.now()],
            "hours":[hours],
            "attendance":[attendance],
            "sleep":[sleep],
            "prediction":[prediction]
        }
    )


    file = "prediction_history.csv"


    if os.path.exists(file):

        old = pd.read_csv(file)

        new_record = pd.concat(
            [old,new_record],
            ignore_index=True
        )


    new_record.to_csv(
        file,
        index=False
    )


    st.info(
        "Prediction saved ✅"
    )



# Show history

if os.path.exists("prediction_history.csv"):

    st.subheader(
        "Prediction History"
    )


    history = pd.read_csv(
        "prediction_history.csv"
    )


    st.dataframe(history)

    # Analytics Dashboard

st.subheader("📊 AI Analytics Dashboard")


total_predictions = len(history)

average_marks = history["prediction"].mean()

best_prediction = history["prediction"].max()


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Total Predictions",
        total_predictions
    )


with col2:
    st.metric(
        "Average Marks",
        f"{average_marks:.2f}"
    )


with col3:
    st.metric(
        "Best Prediction",
        f"{best_prediction:.2f}"
    )



# Graph

st.subheader("Prediction Trend")


st.line_chart(
    history["prediction"]
)