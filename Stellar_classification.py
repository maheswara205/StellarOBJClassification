# Importing necessary libraries
import streamlit as st 
import pandas as pd
from joblib import load
import numpy as np  # Ensure NumPy is imported for predictions

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

# Load joblib model
try:
    Model = load('XGBoost.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please upload the 'XGBoost.joblib' file.")

# Title for the App
st.title("Stellar Object Classification App")

# UI for input details
st.header("Enter the below details:")

alpha = st.number_input("Right Ascension angle:")
delta = st.number_input("Declination angle:")
u = st.number_input("Enter Ultraviolet filter value:")
g = st.number_input("Enter green filter value:")
r = st.number_input("Enter Red filter value:")
i = st.number_input("Enter Near Infrared filter value:")
z = st.number_input("Enter Infrared filter value:")
cam_col = st.selectbox("Camera column", (1, 2, 3, 4, 5, 6))
redshift = st.number_input("Enter the redshift value:")
plate = st.number_input("Enter the Plate ID:", step=1, format="%d")
MJD = st.number_input("Enter the modified Julian Date:", step=1, format="%d")

# Prediction button
if st.button("Classify Object"):
    # Prepare input as a 2D array
    input_features = np.array([[alpha, delta, u, g, r, i, z, cam_col, redshift, plate, MJD]])

    # Perform prediction
    try:
        prediction = Model.predict(input_features)

        # Display result
        st.header("Prediction Result")
        if prediction[0] == 0:
            st.success("The object is a Galaxy")
        elif prediction[0] == 1:
            st.success("The object is a Star")
        else:
            st.success("The object is a QSO")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
