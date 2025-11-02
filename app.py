# app.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# ======= PAGE CONFIG =======
st.set_page_config(page_title="KrishiVox – Smart Crop Recommender", page_icon="🌱", layout="wide")

# ======= TRANSLATIONS (EN / HI) =======
TEXT = {
    "en": {
        "brand": "KrishiVox",
        "sub": "From Survival to Entrepreneurship",
        "start_en": "Start in English",
        "start_hi": "हिंदी में शुरू करें",
        "enter_details": "🌾 Enter Your Farm Details",
        "state": "Select State",
        "soil": "Soil Type",
        "water": "Water Availability",
        "season": "Preferred Season",
        "capital": "Available Capital (₹ per acre)",
        "cycle": "Expected Crop Cycle (days)",
        "irrigation": "Access to Irrigation",
        "fertilizer": "Access to Fertilizers",
        "get_reco": "🌿 Get Recommendations",
        "top_reco": "🌱 Top Recommended Crops",
        "download": "⬇️ Download Report",
        "back": "🔄 Back to Home",
        "no_results": "No matching crops found — showing closest matches.",
        "filters": "Filters",
        "min_roi": "Minimum ROI %",
        "min_profit": "Minimum Profit (₹/acre)",
        "season_filter": "Season",
        "search": "Search crop name..."
    },
    "hi": {
        "brand": "KrishiVox",
        "sub": "जीविका से उद्यमिता तक",
        "start_en": "Start in English",
        "start_hi": "हिंदी में शुरू करें",
        "enter_details": "🌾 अपनी खेत की जानकारी दर्ज करें",
        "state": "राज्य चुनें",
        "soil": "मिट्टी का प्रकार",
        "water": "जल उपलब्धता",
        "season": "पसंदीदा मौसम",
        "capital": "उपलब्ध पूँजी (₹ प्रति एकड़)",
        "cycle": "अपेक्षित फसल चक्र (दिन)",
        "irrigation": "सिंचाई की पहुंच",
        "fertilizer": "उर्वरक की पहुंच",
        "get_reco": "🌿 सिफारिशें प्राप्त करें",
        "top_reco": "🌱 शीर्ष सिफारिशें",
        "download": "⬇️ रिपोर्ट डाउनलोड करें",
        "back": "🔄 मुख पृष्ठ पर वापस जाएँ",
        "no_results": "मेल खाने वाली फसल नहीं मिली — निकटतम मिलान दिखा रहे हैं।",
        "filters": "फ़िल्टर",
        "min_roi": "न्यूनतम ROI %",
        "min_profit": "न्यूनतम लाभ (₹/एकड़)",
        "season_filter": "मौसम",
        "search": "फसल का नाम खोजें..."
    }
}

# ======= GLOBAL THEME (soft green gradient, cards) =======
st.markdown("""
    <style>
    /* page background */
    .stApp { background: linear-gradient(180deg, #f3fff6 0%, #f7fff9 45%, #fffdf7 100%); }
    /* header */
    .brand { text-align:center; font-size:48px; font-weight:800; color:#1b8a57; margin-top:8px; }
    .subtitle { text-align:center; font-size:20px; color:#223344; margin-bottom:18px;}
    /* big buttons */
    .big-btn { background:#1b8a57; color:#fff; padding:12px 26px; border-radius:28px; font-weight:700; display:inline-block; text-decoration:none; }
    .big-btn-alt { background: linear-gradient(90deg,#f39c12,#ff7a2d); color:#fff; padding:12px 26px; border-radius:28px; font-weight:700; display:inline-block; text-decoration:none; margin-left:14px;}
    /* feature cards */
    .feature-grid { display:flex; gap:28px; justify-content:center; margin-top:30px; margin-bottom:30px; }
    .feature-card { background:white; padding:28px; border-radius:14px; width:340px; box-shadow:0 10px 20px rgba(27,138,87,0.06); text-align:center; }
    .feature-title { font-weight:700; color:#223344; font-size:18px; }
    .feature-desc { color:#5b6b6b; margin-top:10px; font-size:14px; }
    /* input form layout */
    .input-card { background:white; padding:20px; border-radius:12px; box-shadow:0 8px 20px rgba(0,0,0,0.04); margin-bottom:18px; }
    /* results grid */
    .results-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:20px; margin-top:18px; }
    .crop-card { background:white; border-radius:16px; padding:18px; box-shadow:0 10px 30px rgba(12,75,32,0.06); transition: transform .18s ease, box-shadow .18s ease; }
    .crop-card:hover { transform: translateY(-6px); box-shadow:0 18px 50px rgba(12,75,32,0.12); }
    .crop-title { font-size:20px; font-weight:800; color:#145e37; }
    .crop-meta { color:#4b5560; margin-top:6px; font-size:14px; }
    .pill { display:inline-block; padding:6px 10px; border-radius:999px; background:#eef9f1; color:#2b6a43; font-weight:600; margin-right:8px; font-size:13px; }
    .progress { height:10px; background:#e6f6ea; border-radius:999px; overflow:hidden; margin-top:8px; }
    .progress-inner { height:100%; background:linear-gradient(90deg,#1b8a57,#6ed08a); }
    </style>
""", unsafe_allow_html=True)

# ======= SESSION STATE defaults =======
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"  # default english
if "results" not in st.session_state:
    st.session_state["results"] = None
if "inputs" not in st.session_state:
    st.session_state["inputs"] = {}

# ======= Load dataset once (cache) =======
@st.cache_data
def load_df(path="final_dataset.csv"):
    df_local = pd.read_csv(path)
    df_local.columns = df_local.columns.str.strip().str.lower()
    return df_local

df = load_df("final_dataset.csv")

# ======= state -> climate mapping (same as original) =======
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

# ======= helper for translation =======
def t(key):
    lang = st.session_state.get("lang", "en")
    return TEXT[lang].get(key, key)

# ======= HOME PAGE =======
def show_home():
    st.markdown(f"<div class='brand'>{t('brand')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtitle'>{t('sub')}</div>", unsafe_allow_html=True)

    # Start buttons centered
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.write("")
    with c2:
        if st.button(t('start_en'), key="start_en_btn"):
            st.session_state["lang"] = "en"
            st.session_state["page"] = "input"
            st.experimental_rerun()
        st.write("")  # spacing
        if st.button(t('start_hi'), key="start_hi_btn"):
            st.session_state["lang"] = "hi"
            st.session_state["page"] = "input"
            st.experimental_rerun()
    with c3:
        st.write("")

    # features row
    st.markdown("""
        <div class="feature-grid">
            <div class="feature-card">
                <div style="width:48px;height:48px;border-radius:50%;background:#e8fbec;margin:0 auto;display:flex;align-items:center;justify-content:center">🌱</div>
                <div class="feature-title">Enter your farm details</div>
                <div class="feature-desc">Share information about your land, soil, water, and resources.</div>
            </div>
            <div class="feature-card">
                <div style="width:48px;height:48px;border-radius:50%;background:#fff4e0;margin:0 auto;display:flex;align-items:center;justify-content:center">📈</div>
                <div class="feature-title">Get smart crop recommendations</div>
                <div class="feature-desc">Receive AI-powered suggestions for the most profitable crops.</div>
            </div>
            <div class="feature-card">
                <div style="width:48px;height:48px;border-radius:50%;background:#e8fbec;margin:0 auto;display:flex;align-items:center;justify-content:center">🤝</div>
                <div class="feature-title">Plan and grow with AI</div>
                <div class="feature-desc">Step-by-step guidance, schemes and support for your season.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ======= INPUT PAGE (keeps your ML input fields identical) =======
def show_input():
    st.markdown(f"### {t('enter_details')}")
    st.write("")  # small gap

    # form for inputs
    with st.form("farm_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            state = st.selectbox(t('state'), list(state_climate_map.keys()), index=0)
            soil = st.selectbox(t('soil'), ["Loamy","Sandy loam","Clayey"], index=0)
            water = st.selectbox(t('water'), ["Low","Medium","High"], index=1)
        with c2:
            season = st.selectbox(t('season'), ["Kharif","Rabi","Perennial"], index=0)
            capital = st.number_input(t('capital'), min_value=0.0, value=50000.0, step=1000.0)
            cycle = st.number_input(t('cycle'), min_value=30, value=120, step=1)
        with c3:
            irrigation = st.selectbox(t('irrigation'), ["Yes","No"], index=1)
            fertilizer = st.selectbox(t('fertilizer'), ["Yes","No"], index=1)
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        submitted = st.form_submit_button(t('get_reco'))

    if submitted:
        # Save inputs in session for the results page (unchanged)
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
        # Run the exact ML logic now and store results (we will reuse same logic in results page)
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

        # Score Calculation (exact same as original)
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

        # Save & switch to results page
        st.session_state["results"] = filtered.sort_values("final_score_%", ascending=False).reset_index(drop=True)
        st.session_state["page"] = "results"
        st.experimental_rerun()

    # quick navigation back to home
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button(t('back')):
        st.session_state["page"] = "home"
        st.experimental_rerun()

# ======= RESULTS PAGE (grid, dynamic, shows all requested output fields) =======
def show_results():
    st.markdown(f"### {t('top_reco')}")
    if not st.session_state.get("results") is None:
        filtered = st.session_state["results"].copy()
    else:
        st.info(t('no_results'))
        filtered = df.copy()

    # FILTERS PANEL (left) and grid area (right)
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"**{t('filters')}**")
        min_roi = st.slider(t('min_roi'), 0, 100, 0)
        min_profit = st.number_input(t('min_profit'), min_value=0, value=0, step=1000)
        season_f = st.selectbox(t('season_filter'), ['All','Kharif','Rabi','Perennial'])
        search = st.text_input(t('search'))

        # apply filters button
        if st.button("Apply Filters"):
            pass

    # apply UI filters to a display copy
    display = filtered.copy()
    if 'final_score_%' in display.columns:
        display = display[pd.to_numeric(display['final_score_%'], errors='coerce').fillna(0) >= min_roi]
    if 'net_profit_per_acre_inr' in display.columns:
        display = display[pd.to_numeric(display['net_profit_per_acre_inr'], errors='coerce').fillna(0) >= min_profit]
    if season_f != 'All' and 'season' in display.columns:
        display = display[display['season'].str.lower().str.strip() == season_f.lower()]
    if search:
        display = display[display['crop_name'].str.contains(search, case=False, na=False)]

    if display.empty:
        st.info(t('no_results'))
        display = filtered.head(10).copy()

    # Top summary metrics
    rcols = st.columns(4)
    rcols[0].metric("Crops Shown", len(display))
    if 'expected_roi_%' in display.columns:
        rcols[1].metric("Avg ROI %", f"{display['expected_roi_%'].astype(float).mean():.2f}")
    if 'net_profit_per_acre_inr' in display.columns:
        rcols[2].metric("Avg Profit (₹)", f"{display['net_profit_per_acre_inr'].astype(float).mean():.0f}")
    rcols[3].metric("Top Score", f"{display['final_score_%'].max():.2f}")

    # Grid view cards
    st.markdown("<div class='results-grid'>", unsafe_allow_html=True)
    for _, row in display.sort_values("final_score_%", ascending=False).head(20).iterrows():
        crop = row.get('crop_name', 'Unknown')
        category = row.get('category', '')
        roi = row.get('expected_roi_%', '—')
        profit = row.get('net_profit_per_acre_inr', '—')
        stability = row.get('market_stability', '—')
        schemes = row.get('applicable_government_schemes', '—')
        scheme_desc = row.get('scheme_benefit_description', '')
        why = row.get('why_recommended', '')
        final_score = float(row.get('final_score_%', 0))

        card_html = f"""
        <div class='crop-card'>
            <div style='display:flex; justify-content:space-between; align-items:center'>
                <div>
                    <div class='crop-title'>{crop}</div>
                    <div class='crop-meta'>{category} • {stability}</div>
                </div>
                <div style='text-align:right'>
                    <div class='pill'>{roi}% ROI</div>
                    <div style='font-weight:700; color:#145e37; margin-top:6px'>₹{int(pd.to_numeric(profit, errors='coerce') or 0):,}</div>
                </div>
            </div>
            <div style='height:12px'></div>
            <div class='progress'><div class='progress-inner' style='width:{min(100, max(0, final_score))}%;'></div></div>
            <div style='height:8px'></div>
            <div style='color:#334155; font-size:14px'><strong>Why recommended:</strong> {why}</div>
            <div style='height:6px'></div>
            <div style='color:#334155; font-size:13px'><strong>Government schemes:</strong> {schemes}</div>
            <div style='height:6px'></div>
            <div style='color:#334155; font-size:13px'><strong>Scheme benefit:</strong> {scheme_desc}</div>
            <div style='height:6px'></div>
            <div style='color:#334155; font-size:13px'><strong>Climate:</strong> {row.get('climate_requirement','—')} | <strong>Ideal Temp:</strong> {row.get('ideal_temperature_range','—')}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Download & navigation
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.download_button(t('download'), data=display.to_csv(index=False).encode('utf-8'), file_name="KrishiVox_Recommendations.csv")
    if st.button(t('back')):
        st.session_state["page"] = "home"
        st.experimental_rerun()

# ======= Page router =======
if st.session_state["page"] == "home":
    show_home()
elif st.session_state["page"] == "input":
    show_input()
elif st.session_state["page"] == "results":
    show_results()
else:
    show_home()
