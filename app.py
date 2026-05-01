import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model, label encoder, and symptoms list
try:
    model = joblib.load("model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    symptoms_list = joblib.load("symptoms_list.pkl")
except FileNotFoundError:
    st.error("Required files (model.pkl, label_encoder.pkl, symptoms_list.pkl) not found. Please ensure they are generated and in the correct directory.")
    st.stop()

# App Title and Description
st.title("Disease Prediction App")
st.write("Enter your symptoms to get a potential disease prediction.")

# Create input fields for symptoms
st.sidebar.header("Select Your Symptoms")
selected_symptoms = []
num_cols = 2 # For better layout of checkboxes
cols = st.sidebar.columns(num_cols)

for i, symptom in enumerate(symptoms_list):
    if cols[i % num_cols].checkbox(symptom.replace('_', ' ').title()):
        selected_symptoms.append(symptom)

# Prediction button
if st.sidebar.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Create input array for prediction
        input_data = np.zeros(len(symptoms_list))
        for i, symptom in enumerate(symptoms_list):
            if symptom in selected_symptoms:
                input_data[i] = 1

        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_data], columns=symptoms_list)

        # Make prediction
        prediction_encoded = model.predict(input_df)[0]
        predicted_disease = label_encoder.inverse_transform([prediction_encoded])[0]

        st.success(f"**Predicted Disease:** {predicted_disease.title()}")
        st.balloons() # A little celebration for prediction

st.sidebar.markdown("---")
st.sidebar.info("This model provides predictions based on symptoms. Always consult a healthcare professional for a definitive diagnosis.")
