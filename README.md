# Walmart Sales Predictor

Flask web app untuk memprediksi weekly sales Walmart menggunakan XGBoost.
Semua fitur dalam Rupiah (IDR) — input, proses, dan output tanpa konversi kurs.

## Deploy

Build: `pip install -r requirements.txt`
Start: `waitress-serve --port=$PORT app:app`

## Dataset

Model dilatih menggunakan dataset Walmart US yang dikonversi ke Rupiah.
Data per-store disimpan dalam `store_data.json`.
