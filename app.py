import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. SIVUN ASETUKSET JA TYYLIT
st.set_page_config(page_title="Audio Mentor AI", layout="wide")

# Muokataan ulkoasua hieman ammattimaisemmaksi
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    .stExpander { background-color: #1a1c23 !important; border: 1px solid #30363d !important; }
    h1, h2, h3 { color: #4e8cff !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🎙️ Audio Mentor AI")
st.write("---")
st.subheader("Audiovisuaalisen laadun diagnostiikka- ja oppimisalusta")
st.info("Tämä työkalu analysoi äänitiedostosi teknisen laadun ja opastaa sinua tekemään parempia miksauksia.")

# 2. TIEDOSTON LATAUS
uploaded_file = st.file_uploader("Lataa äänitiedosto (.wav tai .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Ladataan audio librosalla
    with st.spinner('Analysoidaan tiedostoa...'):
        y, sr = librosa.load(uploaded_file)
        
        # Lasketaan kesto
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 3. VISUALISOINTI: Ääniaalto
        st.write("### 📊 Signaalin visualisointi")
        fig, ax = plt.subplots(figsize=(12, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#4e8cff', alpha=0.8)
        ax.set_facecolor('#0e1117')
        ax.set_xlabel("Aika (s)", color='white')
        ax.set_ylabel("Amplitudi", color='white')
        ax.tick_params(colors='white')
        fig.patch.set_facecolor('#0e1117')
        st.pyplot(fig)

        # 4. TEKNINEN ANALYYSI
        # Lasketaan RMS (keskiarvovoimakkuus)
        rms = np.mean(librosa.feature.rms(y=y))
        db_level = 20 * np.log10(rms) if rms > 0 else -100
        
        # Lasketaan Peak (huippuarvo)
        peak = np.max(np.abs(y))
        
        # Arvioidaan kohinataso (hiljaisimmat 10% pätkistä)
        stft = np.abs(librosa.stft(y))
        noise_floor_val = np.mean(np.sort(np.mean(stft, axis=0))[:int(len(stft[0])*0.1)])
        noise_db = 20 * np.log10(noise_floor_val) if noise_floor_val > 0 else -100

        st.write("---")
        
        # 5. DIAGNOSTIIKKA JA OPETUS (Kaksi saraketta)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("### 🔍 Löydökset ja suositukset")
            
            # Voimakkuuden tarkistus
            if db_level < -25:
                with st.expander("❌ Ääni on liian hiljainen (Low Gain)"):
                    st.error(f"Havaittu taso: {db_level:.1f} dB")
                    st.write("**Miksi tämä on ongelma?** Hiljainen äänitys vaatii tason nostoa digitaalisesti, mikä korostaa kohinaa (noise floor).")
                    st.info("💡 **Suositus:** Nosta gainia äänitysvaiheessa tai käytä 'Normalisointia'. Tavoittele n. -18...-14 dB RMS-tasoa.")
            
            # Säröytymisen tarkistus
            if peak > 0.98:
                with st.expander("⚠️ Signaali clippaa (Peak Distortion)"):
                    st.warning(f"Huippuarvo: {peak:.2f}")
                    st.write("**Miksi tämä on ongelma?** Digitaalinen särö on peruuttamatonta. Ääniaallon huiput leikkautuvat pois, mikä kuulostaa rätinältä.")
                    st.info("💡 **Suositus:** Laske sisääntulon tasoa. Jätä vähintään 3-6 dB 'headroomia' eli varaa huipuille.")
            
            # Kohinan tarkistus
            if noise_db > -45:
                with st.expander("⚠️ Korkea pohjakohina (Noise Floor)"):
                    st.warning(f"Arvioitu kohina: {noise_db:.1f} dB")
                    st.write("**Miksi tämä on ongelma?** Taustalla kuuluva suhina tai humina heikentää puheen selkeyttä ja ammattimaisuutta.")
                    st.info("💡 **Suositus:** Käytä Noise Gatea tai kevyttä Noise Reductionia (esim. iZotope RX). Tarkista huoneen akustiikka.")

            if db_level >= -25 and peak <= 0.98 and noise_db <= -45:
                st.success("✅ Tekniset arvot ovat erinomaisella tasolla!")

        with col2:
            st.write("### 📈 Tekniset arvot")
            st.metric("Keskiarvo (RMS)", f"{db_level:.1f} dB")
            st.metric("Huippuarvo (Peak)", f"{peak:.2f}")
            st.metric("Kohinataso (Est.)", f"{noise_db:.1f} dB")
            st.write(f"Tiedoston kesto: {duration:.1f} sekuntia")
            
            st.write("---")
            if st.button("Lataa tekninen raportti (Demo)"):
                st.write("Raportin generointi on tulossa pian!")

else:
    st.write("Odotetaan tiedostoa...")
