# app.py (updated - navigation & CSS fixes)
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
        "no_results": "No matching crops found — showing closest matches."
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
        "no_results": "मेल खाने वाली फसल नहीं मिली — निकटतम मिलान दिखा रहे हैं।"
    }
}

def t(key):
    lang = st.session_state.get("lang", "en")
    return TEXT.get(lang, TEXT["en"]).get(key, key)

# ---------------- Minimal safe CSS ----------------
# Keep CSS small to avoid conflicts with Streamlit theme (prevents invisible text)
st.markdown("""
    <style>
      /* header styles only */
      .kv-brand { text-align:center; font-size:48px; font-weight:800; color:#158548; margin-top:14px; }
      .kv-sub { text-align:center; color:#234f3f; margin-bottom:6px; font-size:18px; }
      .kv-desc { text-align:center; color:#415d53; margin-bottom:20px; }
      .kv-starts { text-align:center; margin-top:18px; margin-bottom:26px; }
      .kv-btn { padding:10px 22px; border-radius:26px; color:white; font-weight:700; text-decoration:none; }
      .kv-btn-green { background:#16a34a; }
      .kv-btn-orange { background:linear-gradient(90deg,#f59e0b,#fb923c); }
      .kv-feat { display:flex; gap:18px; justify-content:center; margin:26px 12px; flex-wrap:wrap; }
      .kv-card { background:white; border-radius:12px; padding:16px; width:300px; box-shadow:0 6px 18px rgba(0,0,0,0.06); text-align:center; }
      /* results grid */
      .kv-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; margin-top:18px; }
      .kv-crop { background:white; border-radius:12px; padding:14px; box-shadow:0 10px 28px rgba(0,0,0,0.06); }
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
except Exception:
    st.error("Could not load final_dataset.csv — make sure it exists in the project root.")
    st.stop()

# ---------------- navigation callbacks ----------------
def go_to_input(lang):
    st.session_state["lang"] = lang
    st.session_state["page"] = "input"

def go_home():
    st.session_state["page"] = "home"

def process_and_go_results():
    """
    This will be bound to the form submit button via on_click.
    It reads the form widgets from st.session_state where needed,
    then runs the ML/filtering logic and sets st.session_state['results'] and page.
    """
    # form values are stored in session_state by Streamlit widgets with keys we set below
    state = st.session_state.get("form_state")
    soil = st.session_state.get("form_soil")
    water = st.session_state.get("form_water")
    season = st.session_state.get("form_season")
    capital = st.session_state.get("form_capital")
    cycle = st.session_state.get("form_cycle")
    irrigation = st.session_state.get("form_irrigation")
    fertilizer = st.session_state.get("form_fertilizer")

    # save inputs
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

    # === EXACT ML LOGIC (unchanged) ===
    df = df_global.copy()
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

    st.session_state["results"] = filtered.sort_values("final_score_%", ascending=False).reset_index(drop=True)
    st.session_state["page"] = "results"

# ---------------- Pages ----------------
def show_home():
    st.markdown(f"<div class='kv-brand'>{t('brand')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kv-sub'>{t('subtitle')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kv-desc'>{t('desc')}</div>", unsafe_allow_html=True)

    st.markdown("<div class='kv-feat'>", unsafe_allow_html=True)
    st.markdown("<div class='kv-card'><h4>Enter your farm details</h4><p>Share land, soil, and resources.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='kv-card'><h4>Get smart recommendations</h4><p>AI-powered crop suggestions.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='kv-card'><h4>Plan & Grow</h4><p>Step-by-step support during season.</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='kv-starts'>", unsafe_allow_html=True)
    # Use on_click callbacks — no manual reruns required
    st.button(t("start_en"), key="btn_start_en", on_click=go_to_input, args=("en",))
    st.markdown("&nbsp;&nbsp;")
    st.button(t("start_hi"), key="btn_start_hi", on_click=go_to_input, args=("hi",))
    st.markdown("</div>", unsafe_allow_html=True)

def show_input():
    st.markmark = st.markdown  # safe alias if we accidentally reassign later
    st.markdown("<div class='kv-card' style='max-width:1100px;margin:8px auto;padding:18px'>", unsafe_allow_html=True)
    st.mark.markdown(f"### {t('enter_details')}")
    # We use widget keys so their values are available in session_state for on_click callback
    st.selectbox("Select State", STATES, index=0, key="form_state")
    st.selectbox("Soil Type", ["Loamy","Sandy loam","Clayey"], index=0, key="form_soil")
    st.selectbox("Water Availability", ["Low","Medium","High"], index=1, key="form_water")
    st.selectbox("Preferred Season", ["Kharif","Rabi","Perennial"], index=0, key="form_season")
    st.number_input("Available Capital (₹ per acre)", min_value=0.0, value=50000.0, step=1000.0, key="form_capital")
    st.number_input("Expected Crop Cycle (days)", min_value=30, value=120, step=1, key="form_cycle")
    st.selectbox("Access to Irrigation", ["Yes","No"], index=1, key="form_irrigation")
    st.selectbox("Access to Fertilizers", ["Yes","No"], index=1, key="form_fertilizer")

    # Use a button with on_click to run processing (avoids double-click issues)
    st.markdown("<div style='margin-top:12px'>", unsafe_allow_html=True)
    st.button(t("get_reco"), key="btn_get_reco", on_click=process_and_go_results)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    if st.button(t("back_home"), key="btn_home_from_input", on_click=go_home):
        pass

def show_results():
    st.markdown(f"### {t('results_title')}")
    results = st.session_state.get("results")
    if results is None or results.empty:
        st.info(t("no_results"))
        return

    left, right = st.columns([1,3])
    with left:
        st.markdown("**Filters**")
        min_roi = st.slider("Minimum ROI %", 0, 100, 0)
        min_profit = st.number_input("Minimum Profit (₹/acre)", min_value=0, value=0, step=1000)
        season_f = st.selectbox("Season", ['All','Kharif','Rabi','Perennial'], key="filter_season")
        search = st.text_input("Search crop name...", key="filter_search")

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

    # summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Crops Shown", len(display))
    if 'expected_roi_%' in display.columns:
        c2.metric("Avg ROI %", f"{float(display['expected_roi_%'].astype(float).mean()):.2f}")
    if 'net_profit_per_acre_inr' in display.columns:
        c3.metric("Avg Profit (₹)", f"{float(display['net_profit_per_acre_inr'].astype(float).mean()):.0f}")
    c4.metric("Top Score", f"{float(display['final_score_%'].max()):.2f}")

    st.markdown("<div class='kv-grid'>", unsafe_allow_html=True)
    for _, row in display.sort_values("final_score_%", ascending=False).head(20).iterrows():
        crop = row.get('crop_name','Unknown')
        roi = row.get('expected_roi_%','—')
        profit = row.get('net_profit_per_acre_inr','—')
        stability = row.get('market_stability','—')
        why = row.get('why_recommended','')
        final_score = float(row.get('final_score_%',0))
        climate_req = row.get('climate_requirement','—')
        ideal_temp = row.get('ideal_temperature_range','—')
        schemes = row.get('applicable_government_schemes','—')
        scheme_desc = row.get('scheme_benefit_description','')

        st.markdown(f"""
            <div class='kv-crop'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <div style='max-width:65%'>
                        <div style='font-weight:800;color:#0d6343;font-size:16px'>{crop}</div>
                        <div style='color:#556b63;font-size:13px'>{stability}</div>
                    </div>
                    <div style='text-align:right'>
                        <div style='display:inline-block;padding:6px 10px;border-radius:999px;background:#eef9f1;color:#2b6a43;font-weight:700'>{roi}% ROI</div>
                        <div style='font-weight:800;color:#0d6343;margin-top:6px'>₹{int(pd.to_numeric(profit, errors='coerce') or 0):,}</div>
                    </div>
                </div>
                <div style='height:10px'></div>
                <div style='height:8px;background:#e6f6ea;border-radius:999px;overflow:hidden'>
                    <div style='height:100%;background:linear-gradient(90deg,#1b8a57,#6ed08a);width:{min(100,max(0,final_score))}%'></div>
                </div>
                <div style='margin-top:8px;color:#334155;font-size:13px'><strong>Why:</strong> {why}</div>
                <div style='margin-top:6px;color:#334155;font-size:13px'><strong>Govt schemes:</strong> {schemes}</div>
                <div style='margin-top:6px;color:#334155;font-size:13px'><strong>Scheme benefit:</strong> {scheme_desc}</div>
                <div style='margin-top:6px;color:#334155;font-size:13px'><strong>Climate:</strong> {climate_req} • <strong>Temp:</strong> {ideal_temp}</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button(t("download"), data=display.to_csv(index=False).encode('utf-8'), file_name="KrishiVox_Recommendations.csv")
    st.button(t("back_home"), key="btn_back_home_results", on_click=go_home)

# ---------------- Router ----------------
page = st.session_state.get("page", "home")
if page == "home":
    show_home()
elif page == "input":
    show_input()
elif page == "results":
    show_results()
else:
    show_home()
