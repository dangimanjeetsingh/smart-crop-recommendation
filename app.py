import streamlit as st
import joblib
import pandas as pd

# Load your trained model
model = joblib.load("model.pkl")

# Page setup
st.set_page_config(page_title="Smart Crop Recommendation", page_icon="🌾", layout="centered")

# Custom brand header
st.markdown(
    """
    <style>
        .main-title {
            font-size: 38px;
            color: #00c853;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
        }
        .result-card {
            background-color: #f5f5f5;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            text-align: center;
        }
        .param-table {
            margin: auto;
            width: 80%;
            border-collapse: collapse;
        }
        .param-table td, .param-table th {
            border: 1px solid #ddd;
            padding: 10px;
        }
        .param-table tr:nth-child(even){background-color: #f2f2f2;}
        .param-table tr:hover {background-color: #ddd;}
        .param-table th {
            padding-top: 12px;
            padding-bottom: 12px;
            text-align: center;
            background-color: #00c853;
            color: white;
        }
    </style>
    <div class="main-title">🌾 SmartCrop by Chetati</div>
    """,
    unsafe_allow_html=True,
)

# Input form for parameters
st.subheader("Enter Soil and Climate Details:")

col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200)
with col2:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
with col3:
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0)

# Predict button
if st.button("🌱 Predict Best Crop"):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(input_data)[0]

    # Show results on a new styled section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-card'><h2>✅ Recommended Crop: <span style='color:#00c853'>{prediction}</span></h2></div>", unsafe_allow_html=True)

    # Display all input parameters in a clean table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🌿 Your Entered Values:")
    st.markdown(
        f"""
        <table class='param-table'>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Nitrogen (N)</td><td>{N}</td></tr>
        <tr><td>Phosphorus (P)</td><td>{P}</td></tr>
        <tr><td>Potassium (K)</td><td>{K}</td></tr>
        <tr><td>Temperature (°C)</td><td>{temperature}</td></tr>
        <tr><td>Humidity (%)</td><td>{humidity}</td></tr>
        <tr><td>pH Value</td><td>{ph}</td></tr>
        <tr><td>Rainfall (mm)</td><td>{rainfall}</td></tr>
        </table>
        """,
        unsafe_allow_html=True
    )
