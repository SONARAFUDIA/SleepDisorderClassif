import streamlit as st
import pandas as pd
import pickle

# --- Muat Pipeline yang Telah Dilatih ---
try:
    with open('pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    st.error("File 'pipeline.pkl' tidak ditemukan. Mohon jalankan 'create_pipeline.py' terlebih dahulu untuk membuatnya.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat pipeline: {e}")
    st.stop()


# --- Antarmuka Aplikasi Streamlit ---

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediktor Gangguan Tidur",
    page_icon="😴",
    layout="wide"
)

# Judul Aplikasi
st.title("Aplikasi Prediksi Gangguan Tidur 😴")
st.write(
    "Aplikasi ini memprediksi kemungkinan adanya gangguan tidur (Insomnia atau Sleep Apnea) "
    "berdasarkan faktor kunci kesehatan dan gaya hidup. Isi detail di sidebar untuk mendapatkan prediksi."
)

# --- Sidebar untuk Input Pengguna ---
st.sidebar.header("Masukkan Metrik Kesehatan Anda")

# Membuat kolom input di sidebar
age = st.sidebar.number_input('Usia', min_value=18, max_value=100, value=30)
sleep_duration = st.sidebar.slider('Durasi Tidur (jam)', 4.0, 10.0, 7.5, 0.1)
heart_rate = st.sidebar.slider('Detak Jantung (bpm)', 60, 100, 70)
daily_steps = st.sidebar.slider('Langkah Harian', 1000, 15000, 8000)

systolic_bp = st.sidebar.slider('Tekanan Darah Sistolik (mmHg)', 90, 180, 120)
diastolic_bp = st.sidebar.slider('Tekanan Darah Diastolik (mmHg)', 60, 120, 80)

gender = st.sidebar.selectbox('Jenis Kelamin', ('Laki-laki', 'Perempuan'))
# Mengonversi input jenis kelamin ke format yang sesuai dengan data training ('Male'/'Female')
gender_english = 'Male' if gender == 'Laki-laki' else 'Female'

bmi_category = st.sidebar.selectbox('Kategori BMI', ('Normal', 'Overweight', 'Obesitas'))
# Mengonversi input kategori BMI ke format yang sesuai dengan data training ('Normal'/'Overweight'/'Obese')
bmi_english = 'Obese' if bmi_category == 'Obesitas' else bmi_category


# --- Logika Prediksi ---

# Tombol Prediksi
if st.sidebar.button("Prediksi Gangguan Tidur", use_container_width=True):
    # Membuat DataFrame dari input pengguna
    input_data = pd.DataFrame(
        [[age, gender_english, sleep_duration, bmi_english, heart_rate, daily_steps, systolic_bp, diastolic_bp]],
        columns=['Age', 'Gender', 'Sleep Duration', 'BMI Category', 'Heart Rate', 'Daily Steps', 'Systolic BP', 'Diastolic BP']
    )

    # Melakukan prediksi
    try:
        prediction = pipeline.predict(input_data)[0]
        prediction_proba = pipeline.predict_proba(input_data)
        
        # Menampilkan hasil prediksi di area utama
        st.subheader("Hasil Prediksi")
        
        if prediction == 'None':
            st.success(f"**Prediksi: Tidak Ada Gangguan Tidur**")
            st.write("Berdasarkan data yang diberikan, Anda kemungkinan besar tidak memiliki gangguan tidur umum.")
        else:
            st.warning(f"**Prediksi: {prediction}**")
            st.write(f"Model menyarankan adanya potensi **{prediction}**. Sangat disarankan untuk berkonsultasi dengan profesional kesehatan untuk diagnosis formal.")
        
        # Menampilkan skor probabilitas
        st.subheader("Probabilitas Prediksi")
        proba_df = pd.DataFrame(
            prediction_proba,
            columns=pipeline.classes_,
            index=['Probabilitas']
        )
        st.dataframe(proba_df.style.format("{:.2%}"))

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")

