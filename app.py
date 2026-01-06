import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load('model_churn.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Aplikasi Prediksi Churn Pelanggan")
st.write("Masukkan data pelanggan di bawah ini untuk melihat potensi Churn.") # [cite: 97]

# Form Input Fitur (Contoh beberapa fitur utama) [cite: 94]
tenure = st.number_input("Tenure (Bulan)", min_value=0, max_value=100, value=1)
monthly_charges = st.number_input("Biaya Bulanan", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Biaya", min_value=0.0, value=50.0)

# Tombol Prediksi [cite: 95]
if st.button("Prediksi Sekarang"):
    # Buat DataFrame dari input (sesuaikan kolomnya dengan X_train)
    # Ini hanya contoh sederhana, pastikan jumlah kolom sama dengan saat training
    data_input = pd.DataFrame([[tenure, monthly_charges, total_charges]], columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
    
    # Scaling & Prediksi [cite: 95]
    # Catatan: Kamu perlu menyesuaikan kolom input dengan jumlah fitur di X_train_prep tadi
    prediction = model.predict(data_input) 
    
    # Tampilkan Hasil [cite: 96]
    if prediction[0] == 1:
        st.error("Hasil: Pelanggan ini berpotensi CHURN (Berhenti Berlangganan)")
    else:

        st.success("Hasil: Pelanggan ini berpotensi TETAP BERLANGGANAN")
