import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

st.header("📊 Recommended Crops")

if "inputs" not in st.session_state:
    st.warning("⚠️ Please enter your farm details first from the Input page.")
    st.stop()

inputs = st.session_state["inputs"]
state, soil, season, capital, cycle = inputs["state"], inputs["soil"], inputs["season"], inputs["capital"], inputs["cycle"]

df = pd.read_csv("final_dataset.csv")
df.columns = df.columns.str.strip().str.lower()

# Your existing ML logic (unchanged)
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

climate, temp, zone = state_climate_map.get(state, ("Tropical","20-35","General Zone"))

filtered = df.loc[
    (df["soil_type"].str.lower().str.strip() == soil.lower()) &
    (df["season"].str.lower().str.strip() == season.lower())
].copy()

filtered = filtered.loc[
    (df["climate_requirement"].str.contains(climate, case=False, na=False)) |
    (df["agro_climatic_zone"].str.contains(zone.split()[0], case=False, na=False))
].copy()

filtered = filtered.loc[
    (df["investment_required_inr"] <= capital) &
    (df["duration_days"].between(cycle-40, cycle+40))
].copy()

if filtered.empty:
    filtered = df.loc[
        (df["soil_type"].str.lower().str.strip() == soil.lower()) &
        (df["season"].str.lower().str.strip() == season.lower())
    ].copy()

if filtered.empty:
    filtered = df.loc[df["season"].str.lower().str.strip() == season.lower()] \
        .sort_values("expected_roi_%", ascending=False).head(5).copy()

# --- Scaling and Scoring ---
scaler = MinMaxScaler()
for c in ["expected_roi_%","net_profit_per_acre_inr","subsidy_or_grant_%","price_volatility_%","environmental_impact_score"]:
    filtered[c] = pd.to_numeric(filtered[c], errors="coerce").fillna(0)
if len(filtered) > 1:
    filtered[["ROI_s","Profit_s","Subsidy_s","Risk_s","Eco_s"]] = scaler.fit_transform(
        filtered[["expected_roi_%","net_profit_per_acre_inr","subsidy_or_grant_%","price_volatility_%","environmental_impact_score"]])
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

# --- GRID VIEW ---
cols = st.columns(2)
for i, (_, row) in enumerate(filtered.sort_values("final_score_%", ascending=False).head(6).iterrows()):
    with cols[i % 2]:
        st.markdown(f"""
        <div style="background-color:#111;border:1px solid #FFD700;border-radius:15px;padding:20px;margin:10px 0;">
        <h4 style="color:#FFD700;">🌱 {row['crop_name']}</h4>
        <p style="color:#ccc;font-size:15px;">
        <b>ROI:</b> {row['expected_roi_%']}%<br>
        <b>Profit:</b> ₹{int(row['net_profit_per_acre_inr']):,}/acre<br>
        <b>Govt Scheme:</b> {row['applicable_government_schemes']}<br>
        <b>Benefit:</b> {row['scheme_benefit_description']}<br>
        <b>Market Stability:</b> {row['market_stability']}<br>
        <b>Why Recommended:</b> {row['why_recommended']}
        </p></div>
        """, unsafe_allow_html=True)

st.download_button("⬇️ Download Report", data=filtered.to_csv(index=False),
                   file_name="KrishiVox_Recommendations.csv", use_container_width=True)
