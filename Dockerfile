FROM python:3.11-slim

# Устанавливаем системные зависимости для pdf2image и PyMuPDF
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Создаем директорию для временных файлов
RUN mkdir -p /tmp/pdf_ocr_server

EXPOSE 9000

CMD ["python", "pdf_ocr_server.py"]