from flask import Flask, request, jsonify 
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Модели и энкодеры
MODEL_FILES = {
    "model": {
        "filename": "model.pkl",
        "gdrive_id": "GOOGLE_DRIVE_ID_МОДЕЛИ"
    },
    "le_payment": {
        "filename": "le_payment.pkl",
        "gdrive_id": "GOOGLE_DRIVE_ID_PAYMENT"
    },
    "le_status": {
        "filename": "le_status.pkl",
        "gdrive_id": "GOOGLE_DRIVE_ID_STATUS"
    }
}

#Загрузка с Google Drive
def download_from_gdrive(file_id, filename):
    import gdown
    url = f"https://drive.google.com/file/d/1TKHOQ0TrTK8v3o13Z6tz1E2pbHE-ioTS/view?usp=drive_link"
    gdown.download(url, filename, quiet=False)

# Проверяем наличие и загружаем при необходимости
for key, fileinfo in MODEL_FILES.items():
    if not os.path.exists(fileinfo["filename"]):
        print(f"Загружаем {fileinfo['filename']} с Google Drive...")
        download_from_gdrive(fileinfo["gdrive_id"], fileinfo["filename"])

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
        label = le_status.inverse_transform([prediction])[0]

        return jsonify({"prediction": str(label)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
