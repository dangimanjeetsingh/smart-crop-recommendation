import streamlit as st

st.set_page_config(page_title="KrishiVox – Smart Crop Recommender", page_icon="🌾", layout="wide")

# --- Theme ---
st.markdown("""
<style>
body {
    background: linear-gradient(180deg, #000000, #1a1a1a, #2e2e2e);
    color: #FFD700;
    font-family: 'Poppins', sans-serif;
}
.sidebar .sidebar-content {
    background: #0d0d0d;
}
h1, h2, h3 {
    color: #FFD700 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Multilingual ---
language = st.sidebar.radio("🌐 Select Language / भाषा चुनें", ["English", "हिन्दी"])
st.session_state["lang"] = language

# --- Header ---
st.markdown("<h1 style='text-align:center;'>🌾 KrishiVox</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#ccc;'>AI-Powered Smart Crop Recommendation System</p>", unsafe_allow_html=True)

st.sidebar.success("Use the menu above to navigate between Home, Input, and Results.")
