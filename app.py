import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import joblib

# ====== LOAD MODEL ======
model = joblib.load("model.pkl")

st.set_page_config(page_title="KrishiVox – Smart Crop Recommender", page_icon="🌾", layout="centered")

# ======= CUSTOM HEADER =======
st.markdown("""
    <style>
        .brand-title {
            font-size: 40px;
            font-weight: 700;
            text-align: center;
            color: #FFD700;
            letter-spacing: 2px;
            margin-bottom: 20px;
        }
        .subtext {
            text-align: center;
            font-size: 16px;
            color: #CCCCCC;
            margin-bottom: 40px;
        }
    </style>
    <div class="brand-title">🌾 KrishiVox</div>
    <div class="subtext">AI-Powered Smart Crop Recommendation</div>
""", unsafe_allow_html=True)

# ====== DATASET ======
df = pd.read_csv("final_dataset.csv")
df.columns = df.columns.str.strip().str.lower()

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

# ---------- FORM PAGE ----------
if "page" not in st.session_state or st.session_state["page"] == "form":
    state = st.selectbox("Select State", list(state_climate_map.keys()))
    soil = st.selectbox("Soil Type", ["Loamy","Sandy loam","Clayey"])
    water = st.selectbox("Water Availability", ["Low","Medium","High"])
    season = st.selectbox("Preferred Season", ["Kharif","Rabi","Perennial"])
    capital = st.number_input("Available Capital (₹ per acre)", min_value=0.0, value=50000.0)
    cycle = st.number_input("Expected Crop Cycle (days)", min_value=30, value=120)
    irrigation = st.selectbox("Access to Irrigation", ["Yes","No"])
    fertilizer = st.selectbox("Access to Fertilizers", ["Yes","No"])

    # ---------- ACTION BUTTON ----------
    if st.button("🌿 Get Recommendations"):
        climate, temp, zone = state_climate_map.get(state, ("Tropical","20-35","General Zone"))

        # ------- ML Prediction -------
        # You can modify the order as per your model training columns
        ml_input = pd.DataFrame([[capital, cycle]], columns=["capital", "cycle"])
        predicted_crop = model.predict(ml_input)[0]

        # ------- Smart Filtering Logic -------
        filtered = df.loc[
            (df["soil_type"].str.lower().str.strip() == soil.lower()) &
            (df["season"].str.lower().str.strip() == season.lower())
        ].copy()

        filtered = filtered.loc[
            (filtered["climate_requirement"].str.contains(climate, case=False, na=False)) |
            (filtered["agro_climatic_zone"].str.contains(zone.split()[0], case=False, na=False))
        ].copy()

        filtered = filtered.loc[
            (filtered["investment_required_inr"] <= capital) &
            (filtered["duration_days"].between(cycle-40, cycle+40))
        ].copy()

        if filtered.empty:
            filtered = df.loc[
                (df["soil_type"].str.lower().str.strip() == soil.lower()) &
                (df["season"].str.lower().str.strip() == season.lower())
            ].copy()

        if filtered.empty:
            filtered = df.loc[df["season"].str.lower().str.strip() == season.lower()] \
                .sort_values("expected_roi_%", ascending=False).head(5).copy()

        scaler = MinMaxScaler()
        for c in ["expected_roi_%","net_profit_per_acre_inr","subsidy_or_grant_%",
                  "price_volatility_%","environmental_impact_score"]:
            filtered[c] = pd.to_numeric(filtered[c], errors="coerce").fillna(0)

        if len(filtered) > 1:
            filtered[["ROI_s","Profit_s","Subsidy_s","Risk_s","Eco_s"]] = scaler.fit_transform(
                filtered[["expected_roi_%","net_profit_per_acre_inr","subsidy_or_grant_%",
                          "price_volatility_%","environmental_impact_score"]]
            )
        else:
            filtered[["ROI_s","Profit_s","Subsidy_s","Risk_s","Eco_s"]] = 0.5

        filtered["final_score_%"] = (
            (0.35*filtered["ROI_s"]) +
            (0.25*filtered["Profit_s"]) +
            (0.15*(1 - abs(filtered["investment_required_inr"] - capital)/(capital+1))) +
            (0.1*filtered["Subsidy_s"]) +
            (0.1*(1 - filtered["Risk_s"])) -
            (0.05*filtered["Eco_s"])
        ) * 100

        filtered["final_score_%"] = filtered["final_score_%"].round(2)
        filtered["market_stability"] = filtered["price_volatility_%"].apply(
            lambda x: "💹 Very Stable" if x <= 4 else ("⚖️ Moderate Risk" if x <= 7 else "📉 Volatile")
        )

        # Save results
        st.session_state.update({
            "page": "results",
            "results": filtered.sort_values("final_score_%", ascending=False).head(5),
            "predicted_crop": predicted_crop,
            "inputs": {
                "State": state,
                "Soil": soil,
                "Water": water,
                "Season": season,
                "Capital": capital,
                "Cycle": cycle,
                "Irrigation": irrigation,
                "Fertilizer": fertilizer,
                "Climate": climate
            }
        })

# ---------- RESULT PAGE ----------
if "page" in st.session_state and st.session_state["page"] == "results":
    filtered = st.session_state["results"]
    predicted_crop = st.session_state["predicted_crop"]
    inputs = st.session_state["inputs"]

    st.markdown("""
        <style>
            .result-box {
                background: linear-gradient(145deg, #0d0d0d, #1a1a1a);
                border: 2px solid #FFD700;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 0 25px rgba(255,215,0,0.3);
                text-align: center;
                animation: fadeIn 1s ease-in-out;
            }
            .result-title {
                font-size: 32px;
                font-weight: 700;
                color: #FFD700;
                margin-bottom: 10px;
            }
            .crop-card {
                background-color: #111;
                border: 1px solid #444;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .crop-name {
                font-size: 28px;
                font-weight: 700;
                color: #FFD700;
            }
            .crop-details {
                color: #CCCCCC;
                font-size: 15px;
            }
            .param-table {
                margin: auto;
                width: 80%;
                border-collapse: collapse;
            }
            .param-table td, .param-table th {
                border: 1px solid #666;
                padding: 10px;
                color: #DDD;
            }
            .param-table th {
                background-color: #222;
                color: #FFD700;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='result-box'><div class='result-title'>🌱 AI Predicted Crop: <span style='color:#00FF7F'>{predicted_crop}</span></div></div>", unsafe_allow_html=True)
    st.write("")

    st.markdown("### 🌿 Your Entered Parameters:")
    st.markdown("<table class='param-table'><tr><th>Parameter</th><th>Value</th></tr>" +
                "".join([f"<tr><td>{k}</td><td>{v}</td></tr>" for k,v in inputs.items()]) +
                "</table>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='result-title'>🌾 Top Recommended Crops</div>", unsafe_allow_html=True)

    for _, row in filtered.iterrows():
        st.markdown(f"""
            <div class="crop-card">
                <div class="crop-name">{row['crop_name']}</div>
                <div class="crop-details">
                    ROI: {row['expected_roi_%']}% | Profit: ₹{int(row['net_profit_per_acre_inr']):,} per acre<br>
                    Stability: {row['market_stability']} | Climate: {row['climate_requirement']}
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.download_button("⬇️ Download Full Report", data=filtered.to_csv(index=False), file_name="Recommended_Crops.csv")
    st.write("")
    if st.button("🔄 Go Back to Form"):
        st.session_state["page"] = "form"
