# Основные настройки приложения
import os

# Настройки сервера
HOST = "0.0.0.0"  # Слушаем все интерфейсы
PORT = 9000

# Настройки vLLM (OpenAI-compatible API)
# Импортируем из secrets.py
try:
    from secrets import VLLM_URL, VLLM_API_KEY
except ImportError:
    # Значения по умолчанию, если secrets.py не найден
    VLLM_URL = "http://localhost:8300"
    VLLM_API_KEY = " "
    print(f"⚠️  Файл secrets.py не найден! Использую значения по умолчанию")

MODEL_NAME = "/data/models/Qwen3.5-9B"  # Полный путь к модели

# Настройки временных файлов
TEMP_DIR = "/tmp/pdf_ocr_server"  # Директория для временных файлов

# Настройки логирования
LOG_FILE = "log.txt"

# Создаем временную директорию, если её нет
os.makedirs(TEMP_DIR, exist_ok=True)