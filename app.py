import json
import os

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

MODEL_PATH_JSON = "best_model_walmart.json"
MODEL_PATH_PKL = "best_model_walmart.pkl"
FEATURES_PATH_JSON = "model_features.json"
FEATURES_PATH_PKL = "model_features.pkl"

app = Flask(__name__)

if os.path.exists(MODEL_PATH_JSON):
    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.load_model(MODEL_PATH_JSON)
elif os.path.exists(MODEL_PATH_PKL):
    model = joblib.load(MODEL_PATH_PKL)
else:
    raise FileNotFoundError(
        f"Model tidak ditemukan. Pastikan '{MODEL_PATH_JSON}' atau '{MODEL_PATH_PKL}' "
        "ada di folder yang sama dengan app.py."
    )

if os.path.exists(FEATURES_PATH_JSON):
    with open(FEATURES_PATH_JSON) as f:
        features = json.load(f)
elif os.path.exists(FEATURES_PATH_PKL):
    features = joblib.load(FEATURES_PATH_PKL)
else:
    raise FileNotFoundError(
        f"Daftar fitur tidak ditemukan. Pastikan '{FEATURES_PATH_JSON}' atau "
        f"'{FEATURES_PATH_PKL}' ada di folder yang sama dengan app.py."
    )

with open("store_data.json") as f:
    store_data = json.load(f)
STORE_LIST = sorted(int(k) for k in store_data.keys())


def get_store_latest(store_id: int) -> dict:
    sd = store_data.get(str(store_id))
    if sd is None:
        raise ValueError(f"Data untuk Store {store_id} tidak ditemukan.")
    return sd


def get_store_last_4(store_id: int) -> list[float]:
    sd = store_data.get(str(store_id))
    if sd is None:
        raise ValueError(f"Data Store {store_id} tidak ditemukan.")
    return sd["last_4_sales"]


def parse_numeric(val):
    if val in (None, ""):
        return None
    s = str(val).replace("Rp", "").replace("rp", "").replace(" ", "").strip()
    if not s:
        return None
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
            s = "".join(parts)
        elif len(parts) == 2 and len(parts[1]) == 3:
            s = "".join(parts)
    elif "," in s:
        s = s.replace(",", ".")
    return float(s)


def build_features(
    store_id: int,
    sales_w1=None,
    sales_w2=None,
    sales_w3=None,
    sales_w4=None,
    holiday_flag=None,
    temperature=None,
    fuel_price=None,
    cpi=None,
    unemployment=None,
):
    sd = store_data.get(str(store_id))
    if sd is None:
        raise ValueError(f"Data untuk Store {store_id} tidak ditemukan.")

    last_4 = sd["last_4_sales"]

    w1 = parse_numeric(sales_w1) if sales_w1 not in (None, "") else last_4[-1]
    w2 = parse_numeric(sales_w2) if sales_w2 not in (None, "") else last_4[-2]
    w3 = parse_numeric(sales_w3) if sales_w3 not in (None, "") else last_4[-3]
    w4 = parse_numeric(sales_w4) if sales_w4 not in (None, "") else last_4[-4]

    old_count = sd["row_count"] - 1
    old_sum = sd["expanding_mean"] * old_count
    new_count = old_count + 4
    new_sum = old_sum + w1 + w2 + w3 + w4
    expanding_mean = new_sum / new_count

    parsed_fp = parse_numeric(fuel_price)

    row = {
        "Store": store_id,
        "Holiday_Flag": int(holiday_flag) if holiday_flag not in (None, "") else 0,
        "Temperature": parse_numeric(temperature) if temperature not in (None, "") else float(sd["temperature"]),
        "Fuel_Price": float(parsed_fp) if parsed_fp is not None else float(sd["fuel_price"]),
        "CPI": parse_numeric(cpi) if cpi not in (None, "") else float(sd["cpi"]),
        "Unemployment": parse_numeric(unemployment) if unemployment not in (None, "") else float(sd["unemployment"]),
        "Month": int(sd["month"]),
        "Week_sin": float(np.sin(2 * np.pi * ((sd["week"] % 52) + 1) / 52)),
        "Week_cos": float(np.cos(2 * np.pi * ((sd["week"] % 52) + 1) / 52)),
        "DayOfYear": int(sd["day_of_year"]) + 7,
        "Lag_1": w1,
        "Lag_2": w2,
        "Lag_4": w4,
        "Roll_Mean_4": float(np.mean([w1, w2, w3, w4])),
        "Roll_Std_4": float(np.std([w1, w2, w3, w4], ddof=1)),
        "Store_Expanding_Mean": float(expanding_mean),
    }

    X = pd.DataFrame([row])[features]
    return X, sd


def predict_next_week(store_id: int, **overrides) -> dict:
    X, sd = build_features(store_id, **overrides)
    prediction = float(model.predict(X)[0])

    return {
        "store": store_id,
        "last_known_date": sd["last_date"],
        "predicted_weekly_sales_idr": round(prediction, 0),
        "features_used": X.iloc[0].round(4).to_dict(),
    }


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    selected_store = STORE_LIST[0]
    form_data = {}

    if request.method == "POST":
        form_data = request.form
        try:
            selected_store = int(request.form["store"])
            result = predict_next_week(
                selected_store,
                sales_w1=request.form.get("sales_w1"),
                sales_w2=request.form.get("sales_w2"),
                sales_w3=request.form.get("sales_w3"),
                sales_w4=request.form.get("sales_w4"),
                holiday_flag=request.form.get("holiday_flag") == "on",
                temperature=request.form.get("temperature"),
                fuel_price=request.form.get("fuel_price"),
                cpi=request.form.get("cpi"),
                unemployment=request.form.get("unemployment"),
            )
        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        stores=STORE_LIST,
        selected_store=selected_store,
        result=result,
        error=error,
        form_data=form_data,
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json(force=True, silent=True) or {}

    try:
        store_id = int(data["store"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Field 'store' wajib diisi dan berupa angka."}), 400

    try:
        result = predict_next_week(
            store_id,
            sales_w1=data.get("sales_w1"),
            sales_w2=data.get("sales_w2"),
            sales_w3=data.get("sales_w3"),
            sales_w4=data.get("sales_w4"),
            holiday_flag=data.get("holiday_flag"),
            temperature=data.get("temperature"),
            fuel_price=data.get("fuel_price"),
            cpi=data.get("cpi"),
            unemployment=data.get("unemployment"),
        )
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Terjadi kesalahan: {e}"}), 500


@app.route("/stores", methods=["GET"])
def list_stores():
    out = []
    for store_id in STORE_LIST:
        latest = get_store_latest(store_id)
        out.append({"store": store_id, "last_known_date": latest["last_date"]})
    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)