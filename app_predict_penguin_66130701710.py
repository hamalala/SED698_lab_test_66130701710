
import streamlit as st
import pickle
import numpy as np

# Load the model and encoders
with open('model_penguin_66130701710.pkl', 'rb') as file:
    model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Title for the app
st.title("Penguin Species Prediction")

# Collecting user input
st.header("Input Penguin Measurements")

# Dropdown for island
island = st.selectbox("Island", ['Torgersen', 'Biscoe', 'Dream'])

# Numeric inputs for measurements
culmen_length_mm = st.number_input("Culmen Length (mm)", min_value=0.0)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", min_value=0.0)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=0)
body_mass_g = st.number_input("Body Mass (g)", min_value=0)

# Dropdown for sex
sex = st.selectbox("Sex", ['MALE', 'FEMALE'])

# Encode categorical inputs using the loaded encoders
island_encoded = island_encoder.transform([island])[0]
sex_encoded = sex_encoder.transform([sex])[0]

# Prepare the input for prediction
input_data = np.array([
    island_encoded,
    culmen_length_mm,
    culmen_depth_mm,
    flipper_length_mm,
    body_mass_g,
    sex_encoded
]).reshape(1, -1)

# Make prediction
if st.button("Predict"):
    prediction_encoded = model.predict(input_data)
    # Decode the prediction to get the species name
    prediction = species_encoder.inverse_transform(prediction_encoded)
    st.write(f"Predicted Species: {prediction[0]}")


