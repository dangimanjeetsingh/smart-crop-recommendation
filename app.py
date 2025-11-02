import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------------------- PAGE CONFIG ----------------------------
st.set_page_config(
    page_title="KrishiVox – Smart Crop Recommender",
    page_icon="🌾",
    layout="wide"
)

# ---------------------------- LANGUAGE SETUP ----------------------------
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

lang = st.session_state["lang"]

TEXT = {
    "en": {
        "brand": "KrishiVox",
        "tagline": "AI-Powered Smart Crop Recommendation System",
        "subtitle": "From Survival to Entrepreneurship",
        "start_eng": "Start in English",
        "start_hin": "Start in Hindi",
        "desc": "Let AI help you choose the most profitable crop for your next season.",
        "section1_title": "Enter your farm details",
        "section1_desc": "Share information about your land, soil, water, and resources.",
        "section2_title": "Get smart crop recommendations",
        "section2_desc": "Receive AI-powered suggestions for the top crops for your farm.",
        "section3_title": "Plan your full journey and grow with AI",
        "section3_desc": "Get step-by-step guidance, expert support, and assistance throughout the season.",
        "home": "🏠 Home",
        "input": "🧮 Input Form",
        "results": "🌾 Results",
        "language": "Select Language / भाषा चुने",
        "state": "Select State",
        "soil": "Soil Type",
        "water": "Water Availability",
        "season": "Preferred Season",
        "capital": "Available Capital (₹ per acre)",
        "cycle": "Expected Crop Cycle (days)",
        "irrigation": "Access to Irrigation",
        "fertilizer": "Access to Fertilizers",
        "submit": "🌿 Get Recommendations",
        "back": "🔙 Go Back to Form",
        "download": "⬇️ Download Full Report",
        "top_reco": "🌱 Top Crop Recommendations",
    },
    "hi": {
        "brand": "कृषिवॉक्स",
        "tagline": "एआई-संचालित स्मार्ट फसल सिफारिश प्रणाली",
        "subtitle": "जीविका से उद्यमिता तक",
        "start_eng": "अंग्रेज़ी में शुरू करें",
        "start_hin": "हिंदी में शुरू करें",
        "desc": "एआई आपकी अगली फसल के लिए सबसे लाभदायक फसल चुनने में मदद करेगा।",
        "section1_title": "अपनी खेती का विवरण दर्ज करें",
        "section1_desc": "अपनी भूमि, मिट्टी, पानी और संसाधनों की जानकारी साझा करें।",
        "section2_title": "स्मार्ट फसल सिफारिशें प्राप्त करें",
        "section2_desc": "अपनी खेती के लिए शीर्ष फसलों की एआई-संचालित सिफारिशें प्राप्त करें।",
        "section3_title": "पूरी यात्रा की योजना बनाएं और एआई के साथ बढ़ें",
        "section3_desc": "एआई से मार्गदर्शन, विशेषज्ञ सहायता और वास्तविक समय समर्थन प्राप्त करें।",
        "home": "🏠 होम",
        "input": "🧮 इनपुट फॉर्म",
        "results": "🌾 परिणाम",
        "language": "भाषा चुने",
        "state": "राज्य चुनें",
        "soil": "मिट्टी का प्रकार",
        "water": "पानी की उपलब्धता",
        "season": "मौसम चुनें",
        "capital": "उपलब्ध पूंजी (₹ प्रति एकड़)",
        "cycle": "फसल चक्र (दिनों में)",
        "irrigation": "सिंचाई की सुविधा",
        "fertilizer": "उर्वरक की उपलब्धता",
        "submit": "🌿 सिफारिशें प्राप्त करें",
        "back": "🔙 वापस जाएं",
        "download": "⬇️ पूरी रिपोर्ट डाउनलोड करें",
        "top_reco": "🌱 शीर्ष फसल सिफारिशें",
    }
}

def t(key):
    return TEXT[st.session_state["lang"]].get(key, key)

# ---------------------------- SIDEBAR ----------------------------
with st.sidebar:
    st.markdown(f"### 🌐 {t('language')}")
    lang_choice = st.radio("", ["English", "हिन्दी"])
    if lang_choice == "English":
        st.session_state["lang"] = "en"
    else:
        st.session_state["lang"] = "hi"

    st.write("---")
    page = st.radio("Navigate", [t("home"), t("input"), t("results")])
    st.session_state["page"] = page

# ---------------------------- HOME PAGE ----------------------------
def show_home():
    st.markdown("""
    <style>
    .hero {text-align:center; padding:60px 20px;}
    .brand {font-size:60px; color:#16a34a; font-weight:800;}
    .subtitle {font-size:30px; color:#333; margin-top:-15px;}
    .desc {font-size:18px; color:#555; margin:20px auto; width:70%;}
    .button-row {margin-top:30px;}
    .card {background:white; border-radius:20px; padding:25px; box-shadow:0 4px 20px rgba(0,0,0,0.1); text-align:center;}
    .card h3 {color:#064e3b;}
    .card p {color:#333;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='hero'><div class='brand'>{t('brand')}</div>"
                f"<div class='subtitle'>{t('subtitle')}</div>"
                f"<p class='desc'>{t('desc')}</p></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### ✅ {t('section1_title')}")
        st.write(t("section1_desc"))
    with col2:
        st.markdown(f"### 🌾 {t('section2_title')}")
        st.write(t("section2_desc"))

    st.markdown("### 🤝 " + t("section3_title"))
    st.write(t("section3_desc"))

# ---------------------------- INPUT PAGE ----------------------------
def show_input():
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

    st.markdown("### 🌾 " + t("input"))
    state = st.selectbox(t("state"), list(state_climate_map.keys()))
    soil = st.selectbox(t("soil"), ["Loamy", "Sandy loam", "Clayey"])
    water = st.selectbox(t("water"), ["Low", "Medium", "High"])
    season = st.selectbox(t("season"), ["Kharif", "Rabi", "Perennial"])
    capital = st.number_input(t("capital"), min_value=0.0, value=50000.0)
    cycle = st.number_input(t("cycle"), min_value=30, value=120)
    irrigation = st.selectbox(t("irrigation"), ["Yes", "No"])
    fertilizer = st.selectbox(t("fertilizer"), ["Yes", "No"])

    if st.button(t("submit")):
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
            (filtered["duration_days"].between(cycle - 40, cycle + 40))
        ].copy()

        if filtered.empty:
            filtered = df.loc[
                (df["soil_type"].str.lower().str.strip() == soil.lower()) &
                (df["season"].str.lower().str.strip() == season.lower())
            ].copy()
        if filtered.empty:
            filtered = df.loc[df["season"].str.lower().str.strip() == season.lower()]\
                .sort_values("expected_roi_%", ascending=False).head(5).copy()

        scaler = MinMaxScaler()
        for c in ["expected_roi_%", "net_profit_per_acre_inr", "subsidy_or_grant_%", "price_volatility_%", "environmental_impact_score"]:
            filtered[c] = pd.to_numeric(filtered[c], errors="coerce").fillna(0)
        if len(filtered) > 1:
            filtered[["ROI_s", "Profit_s", "Subsidy_s", "Risk_s", "Eco_s"]] = scaler.fit_transform(
                filtered[["expected_roi_%", "net_profit_per_acre_inr", "subsidy_or_grant_%",
                         "price_volatility_%", "environmental_impact_score"]])
        else:
            filtered[["ROI_s", "Profit_s", "Subsidy_s", "Risk_s", "Eco_s"]] = 0.5

        filtered["final_score_%"] = (
            (0.35 * filtered["ROI_s"]) + (0.25 * filtered["Profit_s"]) +
            (0.15 * (1 - abs(filtered["investment_required_inr"] - capital) / (capital + 1))) +
            (0.1 * filtered["Subsidy_s"]) + (0.1 * (1 - filtered["Risk_s"])) -
            (0.05 * filtered["Eco_s"])
        ) * 100
        filtered["final_score_%"] = filtered["final_score_%"].round(2)
        filtered["market_stability"] = filtered["price_volatility_%"].apply(
            lambda x: "💹 Very Stable" if x <= 4 else ("⚖️ Moderate Risk" if x <= 7 else "📉 Volatile")
        )

        st.session_state["results"] = filtered.sort_values("final_score_%", ascending=False).head(5)
        st.session_state["page"] = t("results")

# ---------------------------- RESULT PAGE ----------------------------
def show_results():
    if "results" not in st.session_state:
        st.warning("No results found. Please fill the input form first.")
        return
    filtered = st.session_state["results"]
    st.markdown(f"### {t('top_reco')}")

    for _, row in filtered.iterrows():
        st.markdown(f"""
        <div style="background:#f9fff9;border-radius:15px;padding:15px;margin:10px 0;
        box-shadow:0 0 10px rgba(0,128,0,0.2)">
        <h3 style="color:#065f46;">{row['crop_name']}</h3>
        <p>ROI: {row['expected_roi_%']}% | Profit: ₹{int(row['net_profit_per_acre_inr']):,}/acre |
        Stability: {row['market_stability']}<br>
        Climate: {row['climate_requirement']} | Soil: {row['soil_type']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.download_button(t("download"), data=filtered.to_csv(index=False),
                       file_name="Recommended_Crops.csv")

    if st.button(t("back")):
        st.session_state["page"] = t("input")

# ---------------------------- ROUTING ----------------------------
page = st.session_state.get("page", t("home"))
if page == t("home"):
    show_home()
elif page == t("input"):
    show_input()
else:
    show_results()
