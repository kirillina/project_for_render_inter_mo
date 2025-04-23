# Используем официальный Python образ
FROM python:3.8-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы в контейнер
COPY . .

# Устанавливаем зависимости
RUN pip install -r requirements.txt

# Открываем порт для Flask приложения
EXPOSE 5000

# Запускаем Flask приложение
CMD ["python", "app.py"]
