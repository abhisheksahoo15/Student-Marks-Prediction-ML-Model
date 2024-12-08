import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Load the model
try:
    model = joblib.load("student_marks_predictor_model.pkl")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'student_marks_predictor_model.pkl' is available.")
    st.stop()

# Initialize a dataframe to store user inputs and predictions
data_store = pd.DataFrame()

# Streamlit app title
st.title("Student Marks Predictor")
st.write("Enter the number of study hours per day to predict your marks.")

# Input feature
study_hours = st.number_input("Study Hours", min_value=0, max_value=24, step=1, value=1)

# Predict button
if st.button("Predict"):
    # Validate input hours
    if study_hours < 0 or study_hours > 24:
        st.error("Please enter valid hours between 1 to 24 if you live on the Earth.")
    else:
        # Prepare input for prediction
        features_value = np.array([[study_hours]])
        try:
            output = model.predict(features_value)[0][0].round(2)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        # Save input and output in the dataframe
        new_entry = pd.DataFrame({'Study Hours': [study_hours], 'Predicted Output': [output]})
        data_store = pd.concat([data_store, new_entry], ignore_index=True)
        try:
            data_store.to_csv('smp_data_from_app.csv', index=False)
        except Exception as e:
            st.warning(f"Could not save data to CSV: {e}")

        # Display prediction
        if output > 100:
            st.success(f"You will get [100%] marks when you study [{study_hours}] hours per day.")
        else:
            st.success(f"You will get [{output}%] marks when you study [{study_hours}] hours per day.")

# Display stored data
if st.button("Show Prediction History"):
    if not data_store.empty:
        st.write("### Prediction History")
        st.dataframe(data_store)
    else:
        st.info("No predictions made yet.")
