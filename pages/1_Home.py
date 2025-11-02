import streamlit as st
from PIL import Image

st.set_page_config(page_title="KrishiVox | Smart Crop Recommendation", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #0f1116;
        color: white;
    }
    .main-title {
        text-align: center;
        font-size: 3.2rem;
        font-weight: 700;
        color: #00cc66;
        margin-top: 80px;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #d3d3d3;
    }
    .button-row {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
    }
    .btn {
        font-size: 1.1rem;
        background: linear-gradient(90deg, #00cc66, #00994d);
        color: white;
        padding: 0.8rem 1.8rem;
        border-radius: 10px;
        margin: 0 10px;
        text-decoration: none;
    }
    .btn:hover {
        background: linear-gradient(90deg, #00e673, #00b359);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-title'>KrishiVox</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>From Survival to Entrepreneurship</p>", unsafe_allow_html=True)

st.markdown("<div class='button-row'>"
            "<a class='btn' href='/2_Input' target='_self'>Start in English</a>"
            "<a class='btn' href='/2_Input' target='_self'>हिंदी में शुरू करें</a>"
            "</div>", unsafe_allow_html=True)

st.markdown("<br><br><hr>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("✅ **Enter your farm details**<br>Share land, soil, and resource info.", unsafe_allow_html=True)
with col2:
    st.markdown("🌾 **Get smart crop recommendations**<br>Receive AI-powered suggestions for top crops.", unsafe_allow_html=True)
with col3:
    st.markdown("🤝 **Plan your full journey**<br>AI-guided steps, expert support, and real-time help.", unsafe_allow_html=True)
