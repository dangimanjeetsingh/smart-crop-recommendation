import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# ----------------- HEADER -----------------
st.set_page_config(page_title="KRISHIVOX", page_icon="🌾", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #0e0e0e 0%, #1c1c1c 100%);
        color: white;
    }
    h1, h2, h3, h4, h5 {
        color: #E0C46C;
        text-align: center;
    }
    .card {
        background-color: #181818;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(255, 215, 0, 0.15);
        border: 1px solid rgba(255, 215, 0, 0.25);
        margin: 10px;
        transition: 0.3s ease;
    }
    .card:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 12px rgba(255, 215, 0, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌾 KRISHIVOX – Smart Crop Recommendation System")

# ----------------- LOAD DATA -----------------
df = pd.read_csv("final_dataset.csv")
df.columns = df.columns.str.strip().str.lower()

state_climate_map = {
    "Rajasthan": ("Arid", "22-36", "Western Dry Zone"),
    "Punjab": ("Temperate", "16-30", "Trans-Gangetic Plain"),
    "Haryana": ("Temperate", "16-32", "Trans-Gangetic Plain"),
    "Uttar Pradesh": ("Sub-tropical", "18-34", "Middle Gangetic Plain"),
    "Bihar": ("Sub-tropical", "20-34", "Middle Gangetic Plain"),
    "Madhya Pradesh": ("Semi-arid", "20-36", "Central Plateau Zone"),
    "Maharashtra": ("Tropical", "22-35", "Western Plateau & Hills"),
    "Karnataka": ("Tropical", "20-34", "Southern Plateau"),
    "Kerala": ("Tropical Humid", "24-32", "Coastal Zone"),
    "Tamil Nadu": ("Tropical Semi-arid", "24-34", "Southern Zone"),
    "Gujarat": ("Arid", "24-38", "Western Dry Zone"),
    "Odisha": ("Sub-tropical Humid", "22-34", "Eastern Coastal Plain"),
    "West Bengal": ("Humid Sub-tropical", "20-32", "Eastern Plateau & Hills"),
    "Andhra Pradesh": ("Tropical Humid", "22-36", "East Coast Plains"),
    "Telangana": ("Tropical Semi-arid", "20-36", "Deccan Plateau"),
    "Uttarakhand": ("Hilly", "14-25", "Western Himalaya"),
    "Himachal Pradesh": ("Hilly", "12-22", "Western Himalaya"),
}

# ----------------- USER INPUTS -----------------
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        state = st.selectbox("Select State", list(state_climate_map.keys()))
        soil = st.selectbox("Soil Type", ["Loamy", "Sandy loam", "Clayey"])
        water = st.selectbox("Water Availability", ["Low", "Medium", "High"])
    with col2:
        season = st.selectbox("Preferred Season", ["Kharif", "Rabi", "Perennial"])
        irrigation = st.selectbox("Access to Irrigation", ["Yes", "No"])
        fertilizer = st.selectbox("Access to Fertilizers", ["Yes", "No"])
    with col3:
        capital = st.number_input("Available Capital (₹ per acre)", min_value=0.0, value=50000.0)
        cycle = st.number_input("Expected Crop Cycle (days)", min_value=30, value=120)

# ----------------- MAIN PROCESS -----------------
if st.button("🔍 Get Recommendations"):
    climate, temp, zone = state_climate_map.get(state, ("Tropical", "20-35", "General Zone"))

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
    for c in ["expected_roi_%", "net_profit_per_acre_inr", "subsidy_or_grant_%", "price_volatility_%", "environmental_impact_score"]:
        filtered[c] = pd.to_numeric(filtered[c], errors="coerce").fillna(0)

    if len(filtered) > 1:
        filtered[["ROI_s", "Profit_s", "Subsidy_s", "Risk_s", "Eco_s"]] = scaler.fit_transform(
            filtered[["expected_roi_%", "net_profit_per_acre_inr", "subsidy_or_grant_%", "price_volatility_%", "environmental_impact_score"]]
        )
    else:
        filtered[["ROI_s", "Profit_s", "Subsidy_s", "Risk_s", "Eco_s"]] = 0.5

    filtered["final_score_%"] = (
        (0.35 * filtered["ROI_s"]) +
        (0.25 * filtered["Profit_s"]) +
        (0.15 * (1 - abs(filtered["investment_required_inr"] - capital) / (capital + 1))) +
        (0.1 * filtered["Subsidy_s"]) +
        (0.1 * (1 - filtered["Risk_s"])) -
        (0.05 * filtered["Eco_s"])
    ) * 100

    filtered["final_score_%"] = filtered["final_score_%"].round(2)
    filtered["market_stability"] = filtered["price_volatility_%"].apply(
        lambda x: "💹 Very Stable" if x <= 4 else ("⚖️ Moderate Risk" if x <= 7 else "📉 Volatile")
    )

    st.success(f"✅ {len(filtered)} Recommended Crops Found for {state} ({soil}, {season})")

    # ----------------- FILTERS -----------------
    sort_by = st.selectbox("Sort crops by:", ["final_score_%", "expected_roi_%", "net_profit_per_acre_inr", "investment_required_inr"])
    filtered = filtered.sort_values(sort_by, ascending=False).head(6)

    # ----------------- GRID VIEW -----------------
    st.markdown("### 🌱 Recommended Crops for You")
    cols = st.columns(3)
    for i, (_, row) in enumerate(filtered.iterrows()):
        col = cols[i % 3]
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <h3>{row['crop_name']} ({row['scientific_name']})</h3>
                    <p><b>💧 Water Need:</b> {row['water_requirement']}</p>
                    <p><b>🌾 Soil:</b> {row['soil_type']} | <b>☀️ Climate:</b> {row['climate_requirement']}</p>
                    <p><b>💰 ROI:</b> {row['expected_roi_%']}% | <b>Profit:</b> ₹{int(row['net_profit_per_acre_inr'])}</p>
                    <p><b>📉 Volatility:</b> {row['market_stability']}</p>
                    <p><b>⏱ Cycle:</b> {row['duration_days']} days | <b>Investment:</b> ₹{int(row['investment_required_inr'])}</p>
                    <p><b>🌿 Scheme:</b> {row['applicable_government_schemes']} ({row['subsidy_or_grant_%']}% subsidy)</p>
                    <p><b>📊 Final Score:</b> <b style='color:#FFD700'>{row['final_score_%']}%</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ----------------- DOWNLOAD -----------------
    filtered.to_csv("Recommended_Crops.csv", index=False)
    st.download_button("📥 Download Full Recommendation Report", data=open("Recommended_Crops.csv", "rb"), file_name="KRISHIVOX_Crop_Recommendations.csv")
