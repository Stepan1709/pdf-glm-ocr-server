# Основные настройки приложения
import os

# Настройки сервера
HOST = "0.0.0.0"  # Слушаем все интерфейсы
PORT = 9000

# Настройки VLLM с Qwen3.5-9B Vision
try:
    from secrets import VLLM_URL, VLLM_API_KEY, MODEL_NAME
except ImportError:
    # Значения по умолчанию, если secrets.py не найден
    VLLM_URL = "http://localhost:8300/v1"
    VLLM_API_KEY = " "
    MODEL_NAME = "Qwen3.5 9B Vision"
    print(f"⚠️  Файл secrets.py не найден! Использую значения по умолчанию")

# Настройки временных файлов
TEMP_DIR = "/tmp/pdf_ocr_server"  # Директория для временных файлов

# Настройки логирования
LOG_FILE = "log.txt"

# Создаем временную директорию, если её нет
os.makedirs(TEMP_DIR, exist_ok=True)