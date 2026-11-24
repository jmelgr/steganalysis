FROM python:3.11-slim

WORKDIR /app

# Dependencias ligeras del sistema para Pillow/NumPy/SciPy
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ ./src/
COPY models/ ./models/

ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
