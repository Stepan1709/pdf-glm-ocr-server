# Основные настройки приложения
import os

# Настройки сервера
HOST = "0.0.0.0"  # Слушаем все интерфейсы
PORT = 9000

# Настройки Ollama
# Импортируем из secrets.py
try:
    from secrets import OLLAMA_URL
except ImportError:
    # Значение по умолчанию, если secrets.py не найден
    OLLAMA_URL = "http://localhost:11434"
    print(f"⚠️  Файл secrets.py не найден! Использую OLLAMA_URL по умолчанию: {OLLAMA_URL}")

MODEL_NAME = "glm-ocr:latest"

# Настройки временных файлов
TEMP_DIR = "/tmp/pdf_ocr_server"  # Директория для временных файлов

# Настройки логирования
LOG_FILE = "log.txt"

# Создаем временную директорию, если её нет
os.makedirs(TEMP_DIR, exist_ok=True)