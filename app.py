from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Загружаем модель и энкодеры
model = joblib.load("model.pkl")
le_payment = joblib.load("le_payment.pkl")
le_status = joblib.load("le_status.pkl")

# Фичи, которые ожидаются на входе
features = [
    "price_order_local", "price_tender_local", "price_start_local",
    "distance_in_meters", "duration_in_seconds",
    "driver_rating", "caryear", "payment_method"
]

@app.route("/")
def home():
    return "ML-модель доступна через API!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Кодируем payment_method
        input_df["payment_method"] = le_payment.transform(input_df["payment_method"].astype(str))

        # Убедимся, что порядок фичей правильный
        input_df = input_df[features]

        # Прогнозируем
        prediction = model.predict(input_df)[0]

        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
