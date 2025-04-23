from flask import Flask, request, jsonify
import joblib
import pandas as pd
import boto3
import os

app = Flask(__name__)

# AWS конфигурация
S3_BUCKET = "название-твоего-бакета"
MODEL_FILES = {
    "model": "model.pkl",
    "le_payment": "le_payment.pkl",
    "le_status": "le_status.pkl"
}

# Функция загрузки с S3
def download_from_s3(filename):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_REGION")
    )
    s3.download_file(S3_BUCKET, filename, filename)

# Загружаем все нужные файлы
for key, filename in MODEL_FILES.items():
    download_from_s3(filename)

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

        prediction = model.predict(input_df)[0]

        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
