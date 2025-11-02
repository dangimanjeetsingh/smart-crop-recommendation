import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# ======= PAGE CONFIG =======
st.set_page_config(page_title="KrishiVox – Smart Crop Recommender", page_icon="🌱", layout="wide")

# ======= GLOBAL THEME =======
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #e8f9ee, #f4fff9);
            font-family: 'Inter', sans-serif;
        }
        .main {
            background: transparent !important;
        }
        .title {
            text-align: center;
            font-size: 48px;
            color: #00b35f;
            font-weight: 700;
            margin-top: 10px;
        }
        .subtitle {
            text-align: center;
            color: #333;
            font-size: 20px;
            margin-bottom: 40px;
        }
        .card {
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
            padding: 25px;
            transition: 0.3s;
        }
        .card:hover {
            box-shadow: 0px 6px 20px rgba(0,0,0,0.12);
        }
        .crop-card {
            background-color: #f8fff9;
            border-radius: 12px;
            padding: 20px;
            margin: 15px 0;
            border: 1px solid #d9f3df;
        }
        .crop-name {
            color: #00b35f;
            font-size: 22px;
            font-weight: 600;
        }
        .crop-details {
            color: #444;
            font-size: 15px;
        }
        button[kind="primary"] {
            background-color: #00b35f !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# ======= HEADER =======
st.markdown("<div class='title'>🌱 KrishiVox</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Smart Crop Recommendation</div>", unsafe_allow_html=True)

# ======= LOAD DATA =======
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

# ======= INPUT FORM =======
st.markdown("### 🌾 Enter Your Farm Details")

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        state = st.selectbox("State", list(state_climate_map.keys()))
        soil = st.selectbox("Soil Type", ["Loamy","Sandy loam","Clayey"])
    with col2:
        water = st.selectbox("Water Availability", ["Low","Medium","High"])
        season = st.selectbox("Preferred Season", ["Kharif","Rabi","Perennial"])
    with col3:
        capital = st.number_input("Available Capital (₹ per acre)", min_value=0.0, value=50000.0)
        cycle = st.number_input("Expected Crop Cycle (days)", min_value=30, value=120)

    irrigation = st.selectbox("Access to Irrigation", ["Yes","No"])
    fertilizer = st.selectbox("Access to Fertilizers", ["Yes","No"])

# ======= BUTTON =======
if st.button("🌿 Get AI Recommendations", use_container_width=True):
    climate, temp, zone = state_climate_map.get(state, ("Tropical","20-35","General Zone"))

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

    # Score Calculation
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

    st.markdown("---")
    st.markdown("## 🌱 Top Recommended Crops")
    st.write("")

    for _, row in filtered.sort_values("final_score_%", ascending=False).head(5).iterrows():
        st.markdown(f"""
            <div class="crop-card">
                <div class="crop-name">{row['crop_name']}</div>
                <div class="crop-details">
                    ROI: {row['expected_roi_%']}% | Profit: ₹{int(row['net_profit_per_acre_inr']):,}/acre<br>
                    Stability: {row['market_stability']} | Climate: {row['climate_requirement']}
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.download_button("⬇️ Download Report", data=filtered.to_csv(index=False),
                       file_name="KrishiVox_Recommendations.csv", use_container_width=True)

