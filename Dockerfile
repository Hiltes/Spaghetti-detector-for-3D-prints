FROM python:3.11-slim
LABEL authors="Dariusz Szymanek"

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn onnxruntime pillow python-multipart

# Kopiujemy kod i OBA pliki modelu
COPY app.py .
COPY spaghetti_model.onnx .
COPY spaghetti_model.onnx.data .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]