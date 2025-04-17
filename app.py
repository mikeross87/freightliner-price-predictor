import streamlit as st
import pandas as pd
import joblib

# Load the model and columns
model = joblib.load("price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Custom page config
st.set_page_config(page_title="Cascadia 126 Price Predictor", layout="centered")

# 🎯 Centered title using markdown and HTML
st.markdown("<h1 style='text-align: center;'>Freightliner Cascadia 126 Price Predictor</h1>", unsafe_allow_html=True)

# --- USER INPUTS ---
year = st.number_input("Year", 1990, 2025, 2020)
mileage = st.number_input("Mileage", 0, 2_000_000, 500_000, format="%d")  # adds commas!
horsepower = st.number_input("Horsepower", 100, 800, 400)
image_count = st.number_input("Image Count", 0, 100, 6)

# --- BUILD INPUT ---
input_data = pd.DataFrame([{col: 0 for col in model_columns}])
input_data["year"] = year
input_data["mileage"] = mileage
input_data["horsepower"] = horsepower
input_data["imageCount"] = image_count

# Automatically activate known model/engine columns
for col in model_columns:
    if col == "model_CASCADIA 126":
        input_data[col] = 1
    elif col.startswith("engineManufacturer_") and "CUMMINS" in col:
        input_data[col] = 1  # or DETROIT, depending on your training data

# --- CLEANUP ---
input_data.columns = input_data.columns.astype(str)
input_data = input_data[model_columns]

# --- PREDICT ---
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: ${prediction:,.2f}")
    st.markdown(
    f"""
    <div style="margin-top: 1rem; font-size: 0.9rem; color: gray;">
        <em>Note:</em> The actual price could vary between 
        <strong>${prediction - 6000:,.2f}</strong> - 
        <strong>${prediction + 6000:,.2f}</strong> depending on factors like condition, added features, or other unseen variables.
    </div>
    """, 
    unsafe_allow_html=True
)

