import streamlit as st

lang = st.session_state.get("lang", "English")

if lang == "English":
    st.header("Welcome to KrishiVox 🌱")
    st.write("""
    KrishiVox is an AI-based smart crop recommender that analyzes your soil, climate,
    and investment details to recommend profitable crops.  
    """)
    st.markdown("""
    **How It Works**
    1️⃣ Enter your farm details in the Input page  
    2️⃣ Get AI recommendations  
    3️⃣ Explore crop data and government schemes  
    """)
else:
    st.header("कृषिवॉक्स में आपका स्वागत है 🌱")
    st.write("""
    कृषिवॉक्स एक एआई आधारित फसल सिफारिश प्रणाली है जो आपकी मिट्टी, जलवायु और निवेश
    के आधार पर सबसे लाभदायक फसलें सुझाती है।  
    """)
    st.markdown("""
    **कैसे काम करता है**
    1️⃣ इनपुट पेज में अपनी जानकारी भरें  
    2️⃣ एआई आधारित फसल सुझाव प्राप्त करें  
    3️⃣ सरकारी योजनाओं और लाभ देखें  
    """)
