import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests
from prophet import Prophet
from datetime import datetime, timedelta

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="Gold Price Forecasting",
    page_icon="🟡",
    layout="centered"
)

# ============================
# HEADER
# ============================
st.title("🟡 Gold Price Forecasting")
st.caption("Forecast Trend Harga Emas Dunia dengan Prophet")

st.divider()

# ============================
# SIDEBAR INPUT
# ============================
st.sidebar.header("⚙️ Pengaturan")

# Periode historis
hist_period = st.sidebar.selectbox(
    "Periode Data Historis",
    ["6 Bulan", "1 Tahun", "3 Tahun"]
)

# Periode prediksi
pred_period = st.sidebar.selectbox(
    "Periode Prediksi",
    ["7 Hari", "30 Hari", "90 Hari"]
)

# Kurs mode
st.sidebar.subheader("💱 Kurs USD → IDR")
kurs_mode = st.sidebar.radio("Pilih Kurs:", ["API (Real-time)", "Manual"])

@st.cache_data
def get_kurs_api():
    url = "https://open.er-api.com/v6/latest/USD"
    return requests.get(url).json()["rates"]["IDR"]

if kurs_mode == "API (Real-time)":
    kurs_idr = get_kurs_api()
else:
    kurs_idr = st.sidebar.number_input(
        "Masukkan Kurs USD → IDR",
        min_value=10000,
        max_value=20000,
        value=16000,
        step=10
    )

st.sidebar.divider()

# Refresh data
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ============================
# HITUNG RANGE TANGGAL
# ============================
today = datetime.today()

if hist_period == "6 Bulan":
    start_date = today - timedelta(days=180)
elif hist_period == "1 Tahun":
    start_date = today - timedelta(days=365)
else:
    start_date = today - timedelta(days=365 * 3)

if pred_period == "7 Hari":
    forecast_days = 7
elif pred_period == "30 Hari":
    forecast_days = 30
else:
    forecast_days = 90

# ============================
# LOAD DATA EMAS DUNIA
# ============================
@st.cache_data
def load_gold_data(start):
    df = yf.download("GC=F", start=start, progress=False)
    df = df[['Close']].reset_index()
    df.columns = ['ds', 'price_usd']
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    return df.dropna()

df = load_gold_data(start_date)

# ============================
# KONVERSI KE IDR / gram
# ============================
df['price_idr'] = df['price_usd'] * kurs_idr / 31.1035

# ============================
# INFO DATA
# ============================
st.info(
    f"📅 Data historis digunakan: {len(df)} hari  |  Periode: {hist_period}"
)

# ============================
# KPI
# ============================
current_usd = df['price_usd'].iloc[-1]
current_idr = df['price_idr'].iloc[-1]

st.metric("🌍 Harga Emas Dunia (USD/oz)", f"USD {current_usd:,.2f}")
st.metric("💱 Kurs USD → IDR", f"Rp {kurs_idr:,.0f}")
st.metric("🟡 Harga Emas Terkini (Rp/gram)", f"Rp {current_idr:,.0f}")

st.divider()

# ============================
# MODEL PROPHET
# ============================
prophet_df = df[['ds', 'price_idr']].rename(columns={'price_idr': 'y'})

model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True
)
model.fit(prophet_df)

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

# ============================
# TREND ANALYSIS
# ============================
future_price = forecast['yhat'].iloc[-1]
change_pct = (future_price - current_idr) / current_idr * 100

if change_pct > 0:
    trend = "📈 Tren Naik"
else:
    trend = "📉 Tren Turun / Stabil"

# ============================
# RINGKASAN
# ============================
st.subheader("🔎 Ringkasan Prediksi")

st.write(f"""
- **Harga saat ini:** Rp {current_idr:,.0f} / gram  
- **Estimasi {pred_period}:** Rp {future_price:,.0f} / gram  
- **Perubahan estimasi:** {change_pct:.2f}%  
- **Arah tren:** {trend}
""")

st.divider()

# ============================
# GRAFIK – HISTORIS + PREDIKSI
# ============================
st.subheader("📊 Grafik Harga Historis & Prediksi")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df['ds'], df['price_idr'], label="Harga Historis (Rp/gram)", linewidth=2)
ax.plot(forecast['ds'], forecast['yhat'], "--", label="Prediksi", linewidth=1.5)

ax.set_xlabel("Tanggal")
ax.set_ylabel("Rp / gram")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# ============================
# CARA PAKAI
# ============================
st.subheader("📘 Cara Menggunakan Aplikasi")

st.write("""
1. Pilih **periode data historis** (lebih panjang → tren lebih stabil).  
2. Pilih **periode prediksi** yang diinginkan.  
3. Lihat **KPI** untuk harga terkini emas dunia, kurs USD→IDR, dan konversi ke Rp/gram.  
4. Perhatikan **arah tren** dari hasil prediksi Prophet.  
5. Gunakan grafik untuk melihat pergerakan harga dan hasil prediksi.
""")

# ============================
# DISCLAIMER
# ============================
st.subheader("⚠️ Disclaimer")

st.caption("""
Aplikasi ini dibuat untuk **tujuan edukasi dan akademik**.  
Prediksi menggunakan model statistik (Prophet) berbasis data historis dan **bukan merupakan saran investasi**.  
Fluktuasi harga emas dipengaruhi banyak faktor eksternal yang tidak sepenuhnya dapat dimodelkan.
""")
