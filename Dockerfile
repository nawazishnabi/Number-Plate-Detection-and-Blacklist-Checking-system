FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /tmp/uploads

EXPOSE 10000
CMD ["gunicorn", "wsgi:app", "--bind", "0.0.0.0:10000", "--timeout", "120", "--workers", "1", "--threads", "4"]