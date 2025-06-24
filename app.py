import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

st.set_page_config(page_title="Prediksi Gangguan Tidur", page_icon="😴", layout="wide")
# Fungsi untuk memuat model yang telah dilatih
# Menggunakan cache untuk efisiensi, agar model tidak dimuat ulang setiap kali ada interaksi
@st.cache_resource
def load_model():
    """Memuat pipeline model dari file joblib."""
    try:
        model = joblib.load('best_sleep_disorder_model.joblib')
        return model
    except FileNotFoundError:
        st.error("File model 'best_sleep_disorder_model.joblib' tidak ditemukan. Pastikan file tersebut berada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None

# Memuat model
model_pipeline = load_model()

# Konfigurasi halaman Streamlit

# Judul dan deskripsi aplikasi
st.title("Aplikasi Prediksi Gangguan Tidur 😴")
st.markdown("""
Aplikasi ini menggunakan model *Machine Learning* (SVM) untuk memprediksi kemungkinan seseorang mengalami gangguan tidur 
(Insomnia atau Sleep Apnea) berdasarkan beberapa parameter kesehatan dan gaya hidup.
"""
)
st.markdown("---")


# --- UI untuk Input Pengguna ---
st.sidebar.header("Masukkan Data Anda")

# Membuat form untuk input agar lebih terstruktur
with st.sidebar.form("prediction_form"):
    # Input data dari pengguna menggunakan komponen sidebar
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    age = st.number_input("Usia", min_value=1, max_value=100, value=30, step=1)
    sleep_duration = st.number_input("Durasi Tidur (jam)", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
    
    st.markdown("##### Tekanan Darah")
    systolic_bp = st.number_input("Sistolik (cth: 120)", min_value=50, max_value=250, value=120)
    diastolic_bp = st.number_input("Diastolik (cth: 80)", min_value=30, max_value=150, value=80)
    
    st.markdown("##### Metrik Lainnya")
    heart_rate = st.number_input("Detak Jantung (bpm)", min_value=40, max_value=150, value=70)
    daily_steps = st.number_input("Langkah Harian", min_value=0, max_value=20000, value=5000)
    bmi_category = st.selectbox("Kategori BMI", ["Normal", "Overweight", "Obese"])

    # Tombol untuk melakukan prediksi
    submit_button = st.form_submit_button(label="🔮 Prediksi Sekarang")


# --- Logika Prediksi dan Tampilan Hasil ---
if submit_button and model_pipeline is not None:
    # Mengumpulkan data input menjadi DataFrame
    # Urutan kolom harus sesuai dengan saat model dilatih
    input_data = {
        'Gender': [gender],
        'Age': [age],
        'Sleep Duration': [sleep_duration],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps],
        'Systolic BP': [systolic_bp],
        'Diastolic BP': [diastolic_bp],
        'BMI Category': [bmi_category]
    }
    
    # Membuat DataFrame dari input
    input_df = pd.DataFrame(input_data)
    
    # Menampilkan data input pengguna dalam format yang rapi
    st.subheader("Data yang Anda Masukkan:")
    st.dataframe(input_df)

    input_df['Age_SleepDuration_Interaction'] = input_df['Age'] * input_df['Sleep Duration']
    input_df['Daily_Steps_Log'] = np.log(input_df['Daily Steps'] + 1)


    # Melakukan prediksi menggunakan pipeline model
    with st.spinner('Menganalisis data Anda...'):
        time.sleep(2) # Simulasi proses analisis
        prediction = model_pipeline.predict(input_df)
        prediction_proba = None
        # Cek apakah model memiliki method predict_proba untuk menampilkan probabilitas
        if hasattr(model_pipeline.named_steps['classifier'], "predict_proba"):
            # Karena ada SMOTE, probabilitas perlu dihitung dengan hati-hati
            # Untuk SVM dengan kernel default 'rbf', probability=False by default.
            # Jika Anda melatih ulang dengan probability=True, Anda bisa menggunakan kode di bawah.
            # prediction_proba = model_pipeline.predict_proba(input_df)
            pass

    st.subheader("Hasil Prediksi:")
    
    # Menampilkan hasil prediksi
    if prediction[0] == 'None':
        st.success("✅ **Normal**: Berdasarkan data Anda, kemungkinan besar Anda tidak mengalami gangguan tidur.")
    elif prediction[0] == 'Sleep Apnea':
        st.warning("⚠️ **Sleep Apnea**: Ada indikasi Anda mengalami *Sleep Apnea*. Disarankan untuk berkonsultasi dengan dokter untuk diagnosis lebih lanjut.")
    elif prediction[0] == 'Insomnia':
        st.warning("⚠️ **Insomnia**: Ada indikasi Anda mengalami *Insomnia*. Menjaga kebiasaan tidur yang baik dan berkonsultasi dengan ahli bisa membantu.")
    
    # Tampilkan probabilitas jika ada
    if prediction_proba is not None:
        st.write("Probabilitas Prediksi:")
        proba_df = pd.DataFrame(prediction_proba, columns=model_pipeline.classes_)
        st.dataframe(proba_df)

elif submit_button and model_pipeline is None:
    st.error("Model tidak dapat digunakan. Silakan periksa pesan kesalahan di atas.")

