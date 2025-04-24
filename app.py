from flask import Flask, request, jsonify 
import joblib
import pandas as pd
import os
import gdown

app = Flask(__name__)

# Модели и энкодеры
MODEL_FILES = {
    "model": {
        "filename": "model.pkl",
        "gdrive_id": "1TKHOQ0TrTK8v3o13Z6tz1E2pbHE-ioTS"
        
    },
    "le_payment": {
        "filename": "le_payment.pkl",
        # Для le_payment и le_status файлы уже в репозитории, так что нет необходимости в gdrive_id
    },
    "le_status": {
        "filename": "le_status.pkl",
    }
}

# Загрузка с Google Drive
def download_from_gdrive(file_id, filename):
    url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
    gdown.download(url, filename, quiet=False)

# Проверка и загрузка моделей и энкодеров
def load_model_files():
    # Проверяем, есть ли файлы в директории
    if not os.path.exists(MODEL_FILES["model"]["filename"]):
        print(f"Загружаем {MODEL_FILES['model']['filename']} с Google Drive...")
        download_from_gdrive(MODEL_FILES["model"]["gdrive_id"], MODEL_FILES["model"]["filename"])

    # Загружаем модель и энкодеры
    model = joblib.load(MODEL_FILES["model"]["filename"])

    # Загружаем le_payment и le_status из репозитория
    le_payment = joblib.load(MODEL_FILES["le_payment"]["filename"])
    le_status = joblib.load(MODEL_FILES["le_status"]["filename"])

    return model, le_payment, le_status

# Загружаем модели и энкодеры
model, le_payment, le_status = load_model_files()

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
        label = le_status.inverse_transform([prediction])[0]

        return jsonify({"prediction": str(label)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
