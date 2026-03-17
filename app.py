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

# Sanakirja teksteille (Helpottaa ylläpitoa)
t = {
    "title": "🎙️ Audio Mentor Pro",
    "upload": "Upload Audio" if not is_fi else "Lataa äänitiedosto",
    "magic_fix": "✨ Magic Fix (Auto-Leveling)",
    "fix_btn": "Apply Fix" if not is_fi else "Suorita korjaus",
    "diag": "Diagnostics" if not is_fi else "Diagnostiikka",
    "stats": "Technical Stats" if not is_fi else "Tekniset tiedot",
    "eq": "EQ & Balance" if not is_fi else "Taajuuskorjaus (EQ)",
    "noise": "Noise & Crackle" if not is_fi else "Kohina ja rätinä",
    "reverb": "Reverb & Space" if not is_fi else "Kaiku ja tila",
    "sibilance": "Sibilance (S-sounds)" if not is_fi else "Sibilanssi (S-äänet)"
}

st.title(t["title"])
st.write("---")

uploaded_file = st.file_uploader(t["upload"], type=["wav", "mp3"])

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    y, sr = librosa.load(io.BytesIO(audio_bytes))
    
    st.write("### 🎧 Playback")
    st.audio(audio_bytes, format='audio/wav')
    
    # --- ANALYYSI-LOGIIKKA ---
    rms = np.mean(librosa.feature.rms(y=y))
    db_level = 20 * np.log10(rms) if rms > 0 else -100
    peak = np.max(np.abs(y))
    
    # Spektrianalyysi EQ:ta varten
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    mean_spec = np.mean(S, axis=1)
    
    # 2. MAGIC FIX OSIA
    st.write(f"### {t['magic_fix']}")
    if st.button(t["fix_btn"]):
        target_peak = 0.89
        y_fixed = y * (target_peak / (peak if peak > 0 else 1))
        buffer = io.BytesIO()
        sf.write(buffer, y_fixed, sr, format='WAV')
        st.success("Levels Normalized! / Tasot normalisoitu!")
        st.audio(buffer, format='audio/wav')

    st.write("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(t["diag"])
        
        # EQ - MATALAT JA KORKEAT
        low_energy = np.mean(mean_spec[(freqs > 20) & (freqs < 200)])
        mid_energy = np.mean(mean_spec[(freqs > 400) & (freqs < 2000)])
        high_energy = np.mean(mean_spec[freqs > 5000])

        with st.expander(t["eq"]):
            if low_energy > mid_energy * 1.5:
                st.warning("Muddy low-end." if not is_fi else "Havaittu kuminointia.")
                st.info("💡 Tip: Use HPF @ 80Hz." if not is_fi else "💡 Vinkki: Käytä ylipäästösuodinta (HPF) 80Hz kohdalla.")
            if high_energy < mid_energy * 0.3:
                st.info("Dull sound." if not is_fi else "Ääni on tunkkainen.")
                st.info("💡 Tip: Boost 3-5kHz." if not is_fi else "💡 Vinkki: Korosta 3-5kHz aluetta.")

            

        # NOISE & CRACKLE
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        with st.expander(t["noise"]):
            if flatness > 0.05: # Valkoinen kohina on "litteää"
                st.error("High background noise." if not is_fi else "Korkea taustakohina.")
                st.info("💡 Tip: Use Noise Gate." if not is_fi else "💡 Vinkki: Käytä Noise Gatea.")
            # De-Crackle simulaatio: etsitään äkillisiä piikkejä
            if peak > 0.95:
                st.warning("Possible crackle/clipping." if not is_fi else "Mahdollista rätinää tai klippausta.")
                st.info("💡 Tip: Use De-Cracker or Lower Input Gain." if not is_fi else "💡 Vinkki: Käytä De-Crackle-työkalua.")

        # SIBILANCE (DE-ESSER)
        sibilance_area = np.mean(mean_spec[(freqs > 5000) & (freqs < 8000)])
        with st.expander(t["sibilance"]):
            if sibilance_area > mid_energy * 1.2:
                st.warning("Harsh S-sounds." if not is_fi else "Teräviä ässiä havaittu.")
                st.info("💡 Tip: Use De-Esser @ 6kHz." if not is_fi else "💡 Vinkki: Käytä De-Esseriä 6kHz kohdalla.")

        # REVERB
        with st.expander(t["reverb"]):
            if flatness < 0.01: # Erittäin dynaaminen ääni pienessä tilassa vs kaiku
                st.warning("Room reflections detected." if not is_fi else "Huoneheijastuksia havaittu.")
                st.info("💡 Tip: Use De-Reverb plugin." if not is_fi else "💡 Vinkki: Käytä De-Reverb-liitännäistä.")

    with col2:
        st.subheader(t["stats"])
        st.metric("RMS Level", f"{db_level:.1f} dB")
        st.metric("Peak Max", f"{peak:.2f}")
        
        # Visualisointi spektrogrammista
        st.write("#### Frequency Spectrum")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
        ax2.set_ylim(0, 10000) # Keskitytään puhealueeseen
        st.pyplot(fig2)

else:
    st.info("Upload file to begin. / Lataa tiedosto aloittaaksesi.")
