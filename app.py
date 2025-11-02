# app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ---------------- Config ----------------
st.set_page_config(page_title="KrishiVox – Smart Crop Recommender", page_icon="🌾", layout="wide")

# ---------------- Translations (small) ----------------
TEXT = {
    "en": {
        "brand": "KrishiVox",
        "subtitle": "From Survival to Entrepreneurship",
        "desc": "Let AI help you choose the most profitable crop for your next season.",
        "start_en": "Start in English",
        "start_hi": "हिंदी में शुरू करें",
        "enter_details": "🌾 Enter Your Farm Details",
        "get_reco": "🌿 Get Recommendations",
        "results_title": "🌱 Top Crop Recommendations",
        "back_home": "🏠 Back to Home",
        "download": "⬇️ Download Full Report",
    },
    "hi": {
        "brand": "कृषिवॉक्स",
        "subtitle": "जीविका से उद्यमिता तक",
        "desc": "एआई आपकी अगली फसल के लिए सबसे लाभदायक फसल चुनने में मदद करेगा।",
        "start_en": "Start in English",
        "start_hi": "हिंदी में शुरू करें",
        "enter_details": "🌾 अपनी खेती की जानकारी दर्ज करें",
        "get_reco": "🌿 सिफारिशें प्राप्त करें",
        "results_title": "🌱 शीर्ष सिफारिशें",
        "back_home": "🏠 होम पर वापस जाएँ",
        "download": "⬇️ पूरी रिपोर्ट डाउनलोड करें",
    }
}
def t(key):
    lang = st.session_state.get("lang","en")
    return TEXT.get(lang, TEXT["en"]).get(key, key)

# ---------------- Theme CSS (simple, symmetric) ----------------
st.markdown("""
    <style>
    /* page background */
    .stApp { background: linear-gradient(180deg, #f3fff6 0%, #f7fff9 45%, #fffdf7 100%); }
    /* header */
    .brand { text-align:center; font-size:56px; font-weight:800; color:#14844f; margin-top:18px; }
    .subtitle { text-align:center; font-size:20px; color:#2f4f46; margin-bottom:6px; }
    .desc { text-align:center; color:#556b63; margin-bottom:22px; }
    /* start buttons */
    .start-row { text-align:center; margin-top:18px; margin-bottom:26px; }
    .btn { display:inline-block; padding:12px 26px; margin:6px; border-radius:28px; font-weight:700; color:#fff; text-decoration:none; }
    .btn-green { background:#18a663; }
    .btn-orange { background: linear-gradient(90deg,#f39c12,#ff7a2d); }
    /* feature cards */
    .features { display:flex; gap:22px; justify-content:center; margin:32px 40px; flex-wrap:wrap; }
    .feat { background:white; border-radius:14px; padding:20px; width:300px; box-shadow:0 8px 24px rgba(20,60,40,0.06); text-align:center; }
    .feat h4 { color:#114c36; margin-bottom:6px; }
    .feat p { color:#556b63; font-size:14px; }
    /* input card */
    .input-container { background:white; border-radius:12px; padding:18px; box-shadow:0 8px 20px rgba(0,0,0,0.04); margin:20px auto; max-width:1100px; }
    /* results grid */
    .results-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; margin-top:18px; }
    .crop-card { background:white; border-radius:12px; padding:16px; box-shadow:0 12px 30px rgba(12,75,32,0.06); transition: transform .18s ease; }
    .crop-card:hover { transform: translateY(-6px); box-shadow:0 20px 50px rgba(12,75,32,0.12); }
    .crop-title { color:#0d6343; font-size:18px; font-weight:700; }
    .crop-meta { color:#556b63; margin-top:6px; font-size:13px; }
    .pill { display:inline-block; padding:6px 10px; border-radius:999px; background:#eef9f1; color:#2b6a43; font-weight:700; margin-right:8px; font-size:13px; }
    .progress { height:8px; background:#e6f6ea; border-radius:999px; overflow:hidden; margin-top:8px; }
    .progress-inner { height:100%; background:linear-gradient(90deg,#1b8a57,#6ed08a); }
    </style>
""", unsafe_allow_html=True)

# ---------------- Supported States ----------------
STATES = [
    "Rajasthan","Punjab","Haryana","Uttar Pradesh","Bihar","Madhya Pradesh",
    "Maharashtra","Karnataka","Kerala","Tamil Nadu","Gujarat","Odisha",
    "West Bengal","Andhra Pradesh","Telangana","Uttarakhand","Himachal Pradesh"
]

# ---------------- State climate mapping (unchanged) ----------------
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

# ---------------- session defaults ----------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"
if "inputs" not in st.session_state:
    st.session_state["inputs"] = {}
if "results" not in st.session_state:
    st.session_state["results"] = None

# ---------------- load dataset (cached) ----------------
@st.cache_data
def load_df(path="final_dataset.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return df

try:
    df_global = load_df("final_dataset.csv")
except Exception as e:
    st.error("Could not load final_dataset.csv — make sure it exists in the project root.")
    st.stop()

# ---------------- Pages ----------------
def show_home():
    st.markdown(f"<div class='brand'>{t('brand')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>{t('subtitle')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='desc'>{t('desc')}</div>", unsafe_allow_html=True)

    # features
    st.markdown("<div class='features'>", unsafe_allow_html=True)
    st.markdown(f"<div class='feat'><h4>{t('enter_details')}</h4><p>Share information about your land, soil, water, and resources.</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='feat'><h4>{t('get_reco')}</h4><p>Receive AI-powered suggestions for the most profitable crops.</p></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='feat'><h4>Plan & Grow</h4><p>Step-by-step guidance, expert support, and assistance throughout the season.</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Start buttons centered
    st.markdown("<div class='start-row'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button(t("start_en"), key="start_en"):
            st.session_state["lang"] = "en"
            st.session_state["page"] = "input"
            st.experimental_rerun()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button(t("start_hi"), key="start_hi"):
            st.session_state["lang"] = "hi"
            st.session_state["page"] = "input"
            st.experimental_rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def show_input():
    st.markdown(f"<div class='input-container'><h3>{t('enter_details')}</h3>", unsafe_allow_html=True)
    df = df_global.copy()

    with st.form("farm_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            state = st.selectbox("Select State", STATES, index=0)
            soil = st.selectbox("Soil Type", ["Loamy","Sandy loam","Clayey"], index=0)
            water = st.selectbox("Water Availability", ["Low","Medium","High"], index=1)
        with c2:
            season = st.selectbox("Preferred Season", ["Kharif","Rabi","Perennial"], index=0)
            capital = st.number_input("Available Capital (₹ per acre)", min_value=0.0, value=50000.0, step=1000.0)
            cycle = st.number_input("Expected Crop Cycle (days)", min_value=30, value=120, step=1)
        with c3:
            irrigation = st.selectbox("Access to Irrigation", ["Yes","No"], index=1)
            fertilizer = st.selectbox("Access to Fertilizers", ["Yes","No"], index=1)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        submitted = st.form_submit_button(t("get_reco"))

    if submitted:
        # Save inputs
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

        # === EXACT ML LOGIC (kept as in your original code) ===
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

        # Store results and go to results page
        st.session_state["results"] = filtered.sort_values("final_score_%", ascending=False).reset_index(drop=True)
        st.session_state["page"] = "results"
        st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Back to home
    if st.button(t("back_home")):
        st.session_state["page"] = "home"
        st.experimental_rerun()

def show_results():
    st.markdown(f"### {t('results_title')}")
    results = st.session_state.get("results")
    if results is None or results.empty:
        st.info(t("no_results"))
        return

    # Left filters, right grid
    left, right = st.columns([1,3])
    with left:
        st.markdown("**Filters**")
        min_roi = st.slider("Minimum ROI %", 0, 100, 0)
        min_profit = st.number_input("Minimum Profit (₹/acre)", min_value=0, value=0, step=1000)
        season_f = st.selectbox("Season", ['All','Kharif','Rabi','Perennial'])
        search = st.text_input("Search crop name...")

    display = results.copy()
    if 'final_score_%' in display.columns:
        display = display[pd.to_numeric(display['final_score_%'], errors='coerce').fillna(0) >= min_roi]
    if 'net_profit_per_acre_inr' in display.columns:
        display = display[pd.to_numeric(display['net_profit_per_acre_inr'], errors='coerce').fillna(0) >= min_profit]
    if season_f != 'All' and 'season' in display.columns:
        display = display[display['season'].str.lower().str.strip() == season_f.lower()]
    if search:
        display = display[display['crop_name'].str.contains(search, case=False, na=False)]

    if display.empty:
        st.info(t("no_results"))
        display = results.head(5).copy()

    # Summary metrics
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Crops Shown", len(display))
    if 'expected_roi_%' in display.columns:
        s2.metric("Avg ROI %", f"{float(display['expected_roi_%'].astype(float).mean()):.2f}")
    if 'net_profit_per_acre_inr' in display.columns:
        s3.metric("Avg Profit (₹)", f"{float(display['net_profit_per_acre_inr'].astype(float).mean()):.0f}")
    s4.metric("Top Score", f"{float(display['final_score_%'].max()):.2f}")

    st.markdown("<div class='results-grid'>", unsafe_allow_html=True)
    for _, row in display.sort_values("final_score_%", ascending=False).head(20).iterrows():
        crop = row.get('crop_name','Unknown')
        category = row.get('category','')
        roi = row.get('expected_roi_%','—')
        profit = row.get('net_profit_per_acre_inr','—')
        stability = row.get('market_stability','—')
        schemes = row.get('applicable_government_schemes','—')
        scheme_desc = row.get('scheme_benefit_description','')
        why = row.get('why_recommended','')
        final_score = float(row.get('final_score_%',0))
        climate_req = row.get('climate_requirement','—')
        ideal_temp = row.get('ideal_temperature_range','—')

        st.markdown(f"""
            <div class='crop-card'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <div>
                        <div class='crop-title'>{crop}</div>
                        <div class='crop-meta'>{category} • {stability}</div>
                    </div>
                    <div style='text-align:right'>
                        <div class='pill'>{roi}% ROI</div>
                        <div style='font-weight:800;color:#0d6343;margin-top:6px'>₹{int(pd.to_numeric(profit, errors='coerce') or 0):,}</div>
                    </div>
                </div>
                <div style='height:10px'></div>
                <div class='progress'><div class='progress-inner' style='width:{min(100,max(0,final_score))}%;'></div></div>
                <div style='height:10px'></div>
                <div style='font-size:13px;color:#334155'><strong>Why:</strong> {why}</div>
                <div style='font-size:13px;color:#334155'><strong>Govt schemes:</strong> {schemes}</div>
                <div style='font-size:13px;color:#334155'><strong>Scheme benefit:</strong> {scheme_desc}</div>
                <div style='font-size:13px;color:#334155'><strong>Climate:</strong> {climate_req} • <strong>Temp:</strong> {ideal_temp}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.download_button(t("download"), data=display.to_csv(index=False).encode('utf-8'), file_name="KrishiVox_Recommendations.csv")
    if st.button(t("back_home")):
        st.session_state["page"] = "home"
        st.experimental_rerun()

# ---------------- Router ----------------
page = st.session_state.get("page","home")
if page == "home":
    show_home()
elif page == "input":
    show_input()
elif page == "results":
    show_results()
else:
    show_home()
