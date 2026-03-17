import streamlit as st
import librosa
import numpy as np
from streamlit_wavesurfer import st_wavesurfer
from pydub import AudioSegment
import os

# 1. KIELIVALINTA (Pidetään koodi siistinä, vain tarvittavat)
lang = st.sidebar.selectbox("Language / Kieli", ["English", "Suomi"])
is_fi = lang == "Suomi"

st.title("🎙️ Audio Mentor Pro" if not is_fi else "🎙️ Audio Mentor Pro")
st.write("---")

uploaded_file = st.file_uploader("Upload Audio" if not is_fi else "Lataa äänitiedosto", type=["wav", "mp3"])

if uploaded_file is not None:
    # TALLENNETAAN VÄLIAIKAISESTI ANALYYSIÄ VARTEN
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # 2. INTERAKTIIVINEN WAVEFORM
    # Tämä komponentti hoitaa soittimen ja klikattavan aallon
    st.write("### 🎧 Interactive Waveform" if not is_fi else "### 🎧 Interaktiivinen ääniaalto")
    st_wavesurfer("temp.wav", height=128, wave_color="#58a6ff", progress_color="#1e429f")

    # 3. ANALYYSI
    y, sr = librosa.load("temp.wav")
    rms = np.mean(librosa.feature.rms(y=y))
    db_level = 20 * np.log10(rms) if rms > 0 else -100
    peak = np.max(np.abs(y))

    # 4. AUTO-FIX PAINIKE
    st.write("---")
    st.write("### ✨ Magic Fix (Beta)")
    if st.button("Apply Automatic Fixes" if not is_fi else "Suorita automaattinen korjaus"):
        with st.spinner("Processing..."):
            # Yksinkertainen automaattinen korjaus (Normalisointi)
            audio = AudioSegment.from_file("temp.wav")
            normalized_audio = audio.normalize() # Nostaat tasot optimaaliseksi
            normalized_audio.export("fixed.wav", format="wav")
            
            st.success("Fix Applied! / Korjaus suoritettu!")
            st.audio("fixed.wav")
            st.download_button("Download Fixed Audio", open("fixed.wav", "rb"), file_name="fixed_audio.wav")

    # 5. DIAGNOSTIIKKA (Kuten aiemmin)
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Findings" if not is_fi else "Löydökset")
        if db_level < -25:
            st.error("Level too low" if not is_fi else "Taso liian alhainen")
        if peak > 0.98:
            st.warning("Clipping detected" if not is_fi else "Säröytymistä havaittu")

    with col2:
        st.subheader("Stats" if not is_fi else "Tilastot")
        st.metric("RMS", f"{db_level:.1f} dB")
        st.metric("Peak", f"{peak:.2f}")
