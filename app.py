import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import soundfile as sf

# 1. KIELIVALINTA
lang = st.sidebar.selectbox("Language / Kieli", ["English", "Suomi"])
is_fi = lang == "Suomi"

st.title("🎙️ Audio Mentor Pro")
st.write("---")

uploaded_file = st.file_uploader("Upload Audio" if not is_fi else "Lataa äänitiedosto", type=["wav", "mp3"])

if uploaded_file is not None:
    # Ladataan audio muistiin
    audio_bytes = uploaded_file.read()
    y, sr = librosa.load(io.BytesIO(audio_bytes))
    
    # 2. SOITIN JA VISUALISOINTI
    st.write("### 🎧 Playback & Waveform")
    st.audio(audio_bytes, format='audio/wav')
    
    fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#58a6ff')
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    st.pyplot(fig)

    # 3. TEKNINEN ANALYYSI
    rms = np.mean(librosa.feature.rms(y=y))
    db_level = 20 * np.log10(rms) if rms > 0 else -100
    peak = np.max(np.abs(y))

    # 4. MAGIC FIX (Numpy-pohjainen normalisointi)
    st.write("---")
    st.write("### ✨ Magic Fix (Beta)")
    if st.button("Apply Automatic Fixes" if not is_fi else "Suorita automaattinen korjaus"):
        with st.spinner("Processing..."):
            # Normalisointi: Etsitään suurin huippu ja skaalataan se -1dB (0.89) tasolle
            target_peak = 0.89 
            current_peak = np.max(np.abs(y))
            
            if current_peak > 0:
                # Lasketaan kerroin ja skaalataan koko array
                scaling_factor = target_peak / current_peak
                y_fixed = y * scaling_factor
                
                # Tallennetaan korjattu audio muistiin (WAV-muodossa)
                buffer = io.BytesIO()
                sf.write(buffer, y_fixed, sr, format='WAV')
                buffer.seek(0)
                
                st.success("Fix Applied! / Korjaus suoritettu!")
                st.audio(buffer, format='audio/wav')
                st.download_button("Download Fixed Audio", buffer, file_name="fixed_audio.wav")
            else:
                st.error("Audio is empty or silent.")

    # 5. DIAGNOSTIIKKA
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Diagnostics" if not is_fi else "Diagnostiikka")
        if db_level < -25:
            st.error("Level too low. Normalization recommended." if not is_fi else "Taso liian alhainen. Suositellaan normalisointia.")
        if peak > 0.98:
            st.warning("Clipping detected. Use a limiter." if not is_fi else "Säröytymistä havaittu. Käytä limitteriä.")
        if db_level >= -25 and peak <= 0.98:
            st.success("Levels are technically sound." if not is_fi else "Tekniset tasot ovat hyvät.")

    with col2:
        st.subheader("Stats" if not is_fi else "Tilastot")
        st.metric("RMS Level", f"{db_level:.1f} dB")
        st.metric("Peak Max", f"{peak:.2f}")

else:
    st.info("Please upload an audio file. / Lataa äänitiedosto aloittaaksesi.")
