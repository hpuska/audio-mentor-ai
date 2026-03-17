import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. KIELIASETUKSET
# Luodaan sanakirja kaikille tekstielementeille
texts = {
    "English": {
        "title": "🎙️ Audio Mentor Pro",
        "subtitle": "Audio Quality Diagnostics & Learning Platform",
        "upload": "Upload audio file",
        "playback": "🎧 Listen to Sample",
        "visual": "📊 Signal Visualization",
        "findings": "🔍 Diagnostics & Recommendations",
        "metrics": "📈 Technical Metrics",
        "eq_title": "🎸 Frequency Balance & EQ",
        "eq_muddy": "Muddy low-end detected.",
        "eq_muddy_tip": "💡 **Tip:** Try a High-Pass Filter (HPF) around 80-100 Hz to remove unnecessary rumble.",
        "eq_muffled": "Audio sounds a bit muffled.",
        "eq_muffled_tip": "💡 **Tip:** Try a small 'Presence boost' at 3-5 kHz to improve clarity.",
        "eq_good": "Frequency balance looks natural.",
        "sibilance": "⚡ Sibilance (Harsh S-sounds)",
        "sibilance_warn": "Strong high frequencies (S-sounds) detected.",
        "sibilance_tip": "💡 **Tip:** Use a De-Esser between 5-8 kHz to soften harshness without losing clarity.",
        "reverb": "🏠 Room Reverb",
        "reverb_warn": "Considerable reflections detected.",
        "reverb_tip": "💡 **Tip:** Use a De-Reverb tool. Next time, try acoustic treatment or move the mic closer.",
        "noise": "🔇 Noise Floor",
        "noise_warn": "High noise floor detected: ",
        "noise_tip": "💡 **Tip:** Use Noise Reduction or a Noise Gate to cut background hiss.",
        "success": "✅ Technical values are at an excellent level!"
    },
    "Suomi": {
        "title": "🎙️ Audio Mentor Pro",
        "subtitle": "Audiolaadun diagnostiikka ja oppimisalusta",
        "upload": "Lataa äänitiedosto",
        "playback": "🎧 Kuuntele näyte",
        "visual": "📊 Signaalin visualisointi",
        "findings": "🔍 Diagnostiikka ja suositukset",
        "metrics": "📈 Tekniset mittarit",
        "eq_title": "🎸 Taajuustasapaino ja EQ",
        "eq_muddy": "Havaittu kuminointia (Muddy low-end).",
        "eq_muddy_tip": "💡 **Vinkki:** Kokeile High-Pass Filteriä (HPF) n. 80-100 Hz poistaaksesi turhan kuminan.",
        "eq_muffled": "Ääni on hieman tunkkainen.",
        "eq_muffled_tip": "💡 **Vinkki:** Kokeile pientä 'Presence boostia' 3-5 kHz alueella parantaaksesi selkeyttä.",
        "eq_good": "Taajuustasapaino vaikuttaa luonnolliselta.",
        "sibilance": "⚡ Sibilanssi (Terävät ässät)",
        "sibilance_warn": "Havaittu voimakkaita korkeita taajuuksia (S-äänet).",
        "sibilance_tip": "💡 **Vinkki:** Käytä De-Esseriä välillä 5-8 kHz pehmentääksesi suhinoita.",
        "reverb": "🏠 Huonekaiku (Reverb)",
        "reverb_warn": "Äänessä tuntuu olevan paljon heijastuksia.",
        "reverb_tip": "💡 **Vinkki:** Käytä De-Reverb -työkalua tai tuo mikki lähemmäs puhujaa ensi kerralla.",
        "noise": "🔇 Pohjakohina (Noise Floor)",
        "noise_warn": "Korkea kohinataso: ",
        "noise_tip": "💡 **Vinkki:** Käytä Noise Reductionia tai Noise Gatea taustasuhinan poistoon.",
        "success": "✅ Tekniset arvot ovat erinomaisella tasolla!"
    }
}

# SIVUN ASETUKSET
st.set_page_config(page_title="Audio Mentor Pro", layout="wide")

# KIELIVALINTA SIVUPALKKIIN
lang = st.sidebar.selectbox("Language / Kieli", ["English", "Suomi"])
t = texts[lang]

# 2. ULKOASUN MUOKKAUS (CSS)
st.markdown(f"""
    <style>
    /* Taustaväri ja fontit */
    .stApp {{ background-color: #0e1117; color: #e0e0e0; }}
    
    /* Korttien tyylittely */
    div[data-testid="stExpander"] {{
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }}
    
    /* Otsikoiden värit */
    h1, h2, h3 {{ color: #58a6ff !important; font-family: 'Inter', sans-serif; }}
    
    /* Sidebarin tyyli */
    [data-testid="stSidebar"] {{ background-color: #161b22; border-right: 1px solid #30363d; }}
    </style>
    """, unsafe_allow_html=True)

st.title(t["title"])
st.caption(t["subtitle"])
st.write("---")

uploaded_file = st.file_uploader(t["upload"], type=["wav", "mp3"])

if uploaded_file is not None:
    st.write(f"### {t['playback']}")
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner('Analyzing...'):
        y, sr = librosa.load(uploaded_file)
        
        # VISUALISOINTI
        st.write(f"### {t['visual']}")
        fig, ax = plt.subplots(figsize=(12, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#58a6ff', alpha=0.8)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='#8b949e')
        st.pyplot(fig)

        # ANALYYSI
        rms = np.mean(librosa.feature.rms(y=y))
        db_level = 20 * np.log10(rms) if rms > 0 else -100
        peak = np.max(np.abs(y))
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        mean_coeffs = np.mean(S, axis=1)
        low_end = np.mean(mean_coeffs[(freqs >= 20) & (freqs <= 250)])
        high_mid = np.mean(mean_coeffs[(freqs >= 2000) & (freqs <= 7000)])

        st.write("---")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write(f"### {t['findings']}")

            # EQ
            with st.expander(t["eq_title"]):
                if low_end > np.mean(mean_coeffs) * 2:
                    st.warning(t["eq_muddy"])
                    st.info(t["eq_muddy_tip"])
                elif high_mid < np.mean(mean_coeffs) * 0.5:
                    st.info(t["eq_muffled"])
                    st.info(t["eq_muffled_tip"])
                else:
                    st.success(t["eq_good"])

            # SIBILANCE
            if high_mid > np.mean(mean_coeffs) * 1.5:
                with st.expander(t["sibilance"]):
                    st.warning(t["sibilance_warn"])
                    st.info(t["sibilance_tip"])

            # NOISE
            stft_noise = np.mean(S, axis=1)
            noise_floor = 20 * np.log10(np.percentile(stft_noise, 10))
            if noise_floor > -50:
                with st.expander(t["noise"]):
                    st.error(f"{t['noise_warn']} {noise_floor:.1f} dB")
                    st.info(t["noise_tip"])

        with col2:
            st.write(f"### {t['metrics']}")
            st.metric("RMS Level", f"{db_level:.1f} dB")
            st.metric("Peak Max", f"{peak:.2f}")

else:
    st.info("Please upload an audio file to begin analysis. / Lataa äänitiedosto aloittaaksesi.")
