
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved scaler and model
with open("scaler.sav", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("trained_scaled_model.sav", "rb") as model_file:
    model = pickle.load(model_file)

# Title of the app
st.title("Breast Cancer Diagnosis Predictor")
st.write("Enter the following features to predict the diagnosis (Benign or Malignant).")

# Input fields for user input
mean_radius = st.number_input("Mean Radius:", min_value=0.0, step=0.1)
mean_texture = st.number_input("Mean Texture:", min_value=0.0, step=0.1)
mean_perimeter = st.number_input("Mean Perimeter:", min_value=0.0, step=0.1)
mean_area = st.number_input("Mean Area:", min_value=0.0, step=0.1)
mean_smoothness = st.number_input("Mean Smoothness:", min_value=0.0, step=0.01)

# Predict button
if st.button("Predict"):
    # Combine user inputs into a DataFrame
    new_data = pd.DataFrame(
        [[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]],
        columns=["mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness"]
    )
    
    # Scale the input data
    scaled_data = scaler.transform(new_data)
    
    # Make the prediction
    prediction = model.predict(scaled_data)[0]
    
    # Display the prediction
    if prediction == 1:
        st.success("The diagnosis is Malignant (Cancer detected).")
    else:
        st.success("The diagnosis is Benign (No cancer detected).")
