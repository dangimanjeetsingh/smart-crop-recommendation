import streamlit as st

lang = st.session_state.get("lang", "English")

if lang == "English":
    st.header("🌾 Enter Your Farm Details")
else:
    st.header("🌾 अपनी खेती की जानकारी दर्ज करें")

state_climate_map = {
    "Rajasthan": ("Arid","22-36","Western Dry Zone"),
    "Punjab": ("Temperate","16-30","Trans-Gangetic Plain"),
    "Haryana": ("Temperate","16-32","Trans-Gangetic Plain"),
    "Uttar Pradesh": ("Sub-tropical","18-34","Middle Gangetic Plain"),
    "Bihar": ("Sub-tropical","20-34","Middle Gangetic Plain"),
    "Madhya Pradesh": ("Semi-arid","20-36","Central Plateau Zone"),
    "Maharashtra": ("Tropical","22-35","Western Plateau & Hills"),
    "Karnataka": ("Tropical","20-34","Southern Plateau"),
    "Kerala": ("Tropical Humid","24-32","Coastal Zone"),
    "Tamil Nadu": ("Tropical Semi-arid","24-34","Southern Zone"),
    "Gujarat": ("Arid","24-38","Western Dry Zone"),
    "Odisha": ("Sub-tropical Humid","22-34","Eastern Coastal Plain"),
    "West Bengal": ("Humid Sub-tropical","20-32","Eastern Plateau & Hills"),
    "Andhra Pradesh": ("Tropical Humid","22-36","East Coast Plains"),
    "Telangana": ("Tropical Semi-arid","20-36","Deccan Plateau"),
    "Uttarakhand": ("Hilly","14-25","Western Himalaya"),
    "Himachal Pradesh": ("Hilly","12-22","Western Himalaya"),
}

with st.form("crop_form"):
    state = st.selectbox("State", list(state_climate_map.keys()))
    soil = st.selectbox("Soil Type", ["Loamy","Sandy loam","Clayey"])
    water = st.selectbox("Water Availability", ["Low","Medium","High"])
    season = st.selectbox("Preferred Season", ["Kharif","Rabi","Perennial"])
    capital = st.number_input("Available Capital (₹ per acre)", min_value=0.0, value=50000.0)
    cycle = st.number_input("Expected Crop Cycle (days)", min_value=30, value=120)
    irrigation = st.selectbox("Access to Irrigation", ["Yes","No"])
    fertilizer = st.selectbox("Access to Fertilizers", ["Yes","No"])
    submit = st.form_submit_button("🌿 Get AI Recommendations")

if submit:
    st.session_state["inputs"] = {
        "state": state,
        "soil": soil,
        "water": water,
        "season": season,
        "capital": capital,
        "cycle": cycle,
        "irrigation": irrigation,
        "fertilizer": fertilizer
    }
    st.success("✅ Saved! Go to the 'Results' page to see recommendations.")
