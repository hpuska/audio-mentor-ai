import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# SIVUN ASETUKSET
st.set_page_config(page_title="Audio Mentor Pro", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stExpander { background-color: #1a1c23 !important; border: 1px solid #30363d !important; }
    h1, h2, h3 { color: #4e8cff !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎙️ Audio Mentor Pro")
st.write("---")

uploaded_file = st.file_uploader("Lataa äänitiedosto", type=["wav", "mp3"])

if uploaded_file is not None:
    # 1. PLAYBACK - Lisätään soitin heti latauksen jälkeen
    st.write("### 🎧 Kuuntele näyte")
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner('Analysoidaan teknisiä yksityiskohtia...'):
        y, sr = librosa.load(uploaded_file)
        
        # 2. VISUALISOINTI
        st.write("### 📊 Signaalin visualisointi")
        fig, ax = plt.subplots(figsize=(12, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#4e8cff', alpha=0.8)
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        st.pyplot(fig)

        # 3. ANALYYSI-LOGIIKKA
        rms = np.mean(librosa.feature.rms(y=y))
        db_level = 20 * np.log10(rms) if rms > 0 else -100
        peak = np.max(np.abs(y))
        
        # Taajuusanalyysi (EQ-ehdotuksia varten)
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        mean_coeffs = np.mean(S, axis=1)
        
        # Etsitään kuminan (low-end) ja kirkkauden (high-end) suhdetta
        low_end = np.mean(mean_coeffs[(freqs >= 20) & (freqs <= 250)])
        high_mid = np.mean(mean_coeffs[(freqs >= 2000) & (freqs <= 7000)])
        high_air = np.mean(mean_coeffs[freqs > 10000])

        st.write("---")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("### 🔍 Diagnostiikka & Korjausehdotukset")

            # EQ-EHDOTUKSET
            with st.expander("🎸 Taajuustasapaino & EQ"):
                if low_end > np.mean(mean_coeffs) * 2:
                    st.warning("Havaittu kuminointia (Muddy low-end).")
                    st.info("💡 **Neuvo:** Kokeile High-Pass Filteriä (HPF) n. 80-100 Hz kohdalle poistaaksesi turhan kuminan.")
                elif high_mid < np.mean(mean_coeffs) * 0.5:
                    st.info("Ääni on hieman tunkkainen.")
                    st.info("💡 **Neuvo:** Kokeile pientä 'Presence boostia' 3-5 kHz alueella parantaaksesi selkeyttä.")
                else:
                    st.success("Taajuustasapaino vaikuttaa luonnolliselta.")

            # DE-ESSER (Sibilanssi)
            if high_mid > np.mean(mean_coeffs) * 1.5:
                with st.expander("⚡ Sibilanssi (Terävät ässät)"):
                    st.warning("Havaittu voimakkaita korkeita taajuuksia (S-äänet).")
                    st.info("💡 **Neuvo:** Käytä De-Esseriä välillä 5-8 kHz. Tämä pehmentää teräviä suhinoita vaikuttamatta puheen selkeyteen.")

            # KAIUNPOISTO (Reverb Detection)
            # Arvioidaan äänen hännän pituutta yksinkertaistetusti
            if np.mean(librosa.feature.spectral_flatness(y=y)) < 0.01:
                with st.expander("🏠 Huonekaiku (Reverb)"):
                    st.warning("Äänessä tuntuu olevan paljon heijastuksia.")
                    st.info("💡 **Neuvo:** Käytä De-Reverb -työkalua. Seuraavalla kerralla kokeile akustoida tilaa tai tuo mikki lähemmäs puhujaa.")

            # KOHINANPOISTO
            stft_noise = np.mean(S, axis=1)
            noise_floor = 20 * np.log10(np.percentile(stft_noise, 10))
            if noise_floor > -50:
                with st.expander("🔇 Pohjakohina (Noise Floor)"):
                    st.error(f"Korkea kohinataso: {noise_floor:.1f} dB")
                    st.info("💡 **Neuvo:** Käytä Noise Reductionia tai Noise Gatea. Gate katkaisee äänen, kun kukaan ei puhu.")

        with col2:
            st.write("### 📈 Tekniset mittarit")
            st.metric("LUFS/RMS", f"{db_level:.1f} dB")
            st.metric("Peak Max", f"{peak:.2f}")
            st.write("---")
            st.caption("Nämä analyysit perustuvat signaalinkäsittelyyn ja auttavat ymmärtämään audion teknistä laatua.")

else:
    st.write("Lataa äänitiedosto aloittaaksesi analyysin.")
