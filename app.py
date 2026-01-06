import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, dan daftar kolom fitur yang sudah kita simpan sebelumnya
model = joblib.load('model_churn.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl') # Simpan daftar kolom fitur saat training

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

st.title("📊 Telco Customer Churn Prediction")
st.markdown("""
Aplikasi ini memprediksi apakah seorang pelanggan akan berhenti berlangganan (churn) 
berdasarkan data demografis dan layanan yang mereka gunakan.
""")

# Membuat Layout dengan Kolom
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Demografis & Kontrak")
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
    senior = st.selectbox("Lansia?", ["No", "Yes"])
    partner = st.selectbox("Memiliki Pasangan?", ["No", "Yes"])
    dependents = st.selectbox("Memiliki Tanggungan?", ["No", "Yes"])
    tenure = st.number_input("Lama Berlangganan (Bulan)", min_value=0, max_value=100, value=1)
    contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])

with col2:
    st.subheader("Layanan & Biaya")
    internet = st.selectbox("Layanan Internet", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    monthly_charges = st.number_input("Biaya Bulanan (Monthly Charges)", value=50.0)
    total_charges = st.number_input("Total Biaya (Total Charges)", value=50.0)

# Tombol Prediksi
if st.button("Analisis Potensi Churn"):
    # 1. Buat Dictionary dari input
    input_dict = {
        'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'InternetService': internet, 'OnlineSecurity': security,
        'Contract': contract, 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
        # Tambahkan kolom lain sesuai fitur yang kamu gunakan saat training
    }
    
    # 2. Convert ke DataFrame & One-Hot Encoding
    input_df = pd.DataFrame([input_dict])
    input_df_encoded = pd.get_dummies(input_df)
    
    # 3. Menyamakan Kolom dengan Data Training (Penting!)
    # Kita harus memastikan kolom input sama persis dengan kolom saat model dilatih
    for col in model_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0
    input_df_encoded = input_df_encoded[model_columns]
    
    # 4. Scaling & Prediksi
    input_scaled = scaler.transform(input_df_encoded)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    # 5. Tampilan Hasil
    st.divider()
    if prediction[0] == 1:
        st.error(f"### ⚠️ Hasil: Pelanggan Berpotensi CHURN")
        st.write(f"Tingkat Keyakinan: {probability:.2%}")
    else:
        st.success(f"### ✅ Hasil: Pelanggan Berpotensi TETAP BERLANGGANAN")
        st.write(f"Tingkat Keyakinan: {1 - probability:.2%}")
