#!/usr/bin/env python3
"""
Сервер для обработки PDF через OCR модель GLM-OCR
Принимает файлы по API, разбивает на страницы, отправляет в Ollama,
возвращает текст с нумерацией страниц.
"""

import os
import sys
import io
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import asyncio
import aiohttp
import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
import PyPDF2
from pdf2image import convert_from_bytes
import fitz  # PyMuPDF - для более надежной работы с PDF
from tqdm import tqdm
import logging
from contextlib import asynccontextmanager

# Импортируем настройки
from config import HOST, PORT, OLLAMA_URL, MODEL_NAME, TEMP_DIR, LOG_FILE

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Глобальная переменная для сессии aiohttp
session = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global session
    # Запуск: создаем сессию
    session = aiohttp.ClientSession()
    logger.info(f"🚀 Сервер запущен на http://{HOST}:{PORT}")
    logger.info(f"📡 Подключен к Ollama: {OLLAMA_URL}")
    logger.info(f"🤖 Модель: {MODEL_NAME}")

    yield

    # Завершение: закрываем сессию
    if session:
        await session.close()
    logger.info("👋 Сервер остановлен")


# Создаем приложение FastAPI
app = FastAPI(
    title="PDF OCR Server",
    description="Сервер для OCR обработки PDF с помощью GLM-OCR",
    version="1.0.0",
    lifespan=lifespan
)


def log_error(filename: str, error: Exception):
    """Запись ошибки в лог-файл"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_msg = f"{timestamp} | Файл: {filename} | Ошибка: {str(error)}\n"

    # Записываем в файл
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(error_msg)

    # Также выводим в консоль
    logger.error(f"❌ Ошибка при обработке {filename}: {str(error)}")


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Получение количества страниц в PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        return len(pdf_reader.pages)
    except Exception as e:
        logger.warning(f"PyPDF2 не смог прочитать PDF, пробуем PyMuPDF: {e}")
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        return pdf_document.page_count


async def process_page_with_ollama(page_image_bytes: bytes, page_num: int) -> str:
    """
    Отправка изображения страницы в Ollama для OCR
    Возвращает распознанный текст
    """
    try:
        # Кодируем изображение в base64
        import base64
        image_base64 = base64.b64encode(page_image_bytes).decode('utf-8')

        # Формируем запрос к Ollama
        payload = {
            "model": MODEL_NAME,
            "prompt": """Извлеки весь текст с этой страницы документа. 
Верни только текст в обычном читаемом формате, используя кириллицу для русского текста.
Не используй Unicode escape последовательности (например, /uniXXXX).
Не добавляй комментарии и пояснения.
Если текст на русском языке, он должен быть в читаемой кириллице.
Если на английском - латиницей.""",
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                # Параметры для управления ресурсами
                "num_ctx": 2048,  # Размер контекста (токенов)
                "num_predict": 4096,  # Максимум токенов для генерации
                "num_thread": 8,  # Количество потоков CPU
                "num_gpu": 24,  # Количество слоев модели на GPU (24 слоя для полной загрузки)
                "main_gpu": 0,  # Основной GPU (индекс 0)
                "tensor_split": [0.5, 0.5],  # Распределение тензоров между GPU (если несколько)
                "seed": 42,  # Фиксированный seed для воспроизводимости
            }
        }

        # Отправляем запрос
        async with session.post(f"{OLLAMA_URL}/api/generate", json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"Ollama вернул ошибку {response.status}: {error_text}")

            result = await response.json()
            text = result.get("response", "").strip()

            # Добавляем строку с номером страницы
            if text:
                return f"\nСТРАНИЦА {page_num}\n{text}\n"
            else:
                return f"\nСТРАНИЦА {page_num}\n[Пустая страница]\n"

    except Exception as e:
        logger.error(f"Ошибка при обработке страницы {page_num}: {e}")
        return f"\nСТРАНИЦА {page_num}\n[Ошибка OCR: {str(e)}]\n\n"


async def convert_pdf_page_to_image(pdf_bytes: bytes, page_num: int) -> bytes:
    """
    Конвертирует конкретную страницу PDF в изображение
    Использует pdf2image для конвертации
    """
    try:
        # Используем pdf2image для конвертации страницы
        images = convert_from_bytes(
            pdf_bytes,
            first_page=page_num,
            last_page=page_num,
            dpi=300,  # Высокое разрешение для лучшего распознавания
            fmt='png'
        )

        if not images:
            raise Exception(f"Не удалось конвертировать страницу {page_num}")

        # Конвертируем изображение в байты
        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        return img_byte_arr.getvalue()

    except Exception as e:
        logger.error(f"Ошибка конвертации страницы {page_num}: {e}")
        raise


async def process_pdf(filename: str, pdf_bytes: bytes) -> str:
    """
    Основная функция обработки PDF
    Разбивает на страницы, отправляет в Ollama, собирает результат
    """
    # Получаем количество страниц
    total_pages = get_pdf_page_count(pdf_bytes)
    logger.info(f"📄 Получен файл: {filename}")
    logger.info(f"📑 Количество страниц в файле: {total_pages}")

    all_text = []

    # Создаем прогресс-бар для обработки страниц
    with tqdm(total=total_pages, desc="Обработка страниц", unit="стр") as pbar:
        for page_num in range(1, total_pages + 1):
            try:
                # Конвертируем страницу в изображение
                page_image = await convert_pdf_page_to_image(pdf_bytes, page_num)

                # Отправляем в Ollama
                page_text = await process_page_with_ollama(page_image, page_num)
                all_text.append(page_text)

                # Обновляем прогресс-бар
                pbar.update(1)
                pbar.set_postfix({"Текущая страница": page_num})

            except Exception as e:
                error_msg = f"Ошибка при обработке страницы {page_num}: {str(e)}"
                logger.error(error_msg)
                all_text.append(f"СТРАНИЦА {page_num}\n[Ошибка: {str(e)}]\n\n")
                pbar.update(1)
                continue

    # Собираем весь текст
    full_text = "".join(all_text)

    logger.info(f"✅ Файл \"{filename}\" успешно обработан. Всего страниц: {total_pages}")

    return full_text


@app.post("/ocr", response_class=PlainTextResponse)
async def ocr_pdf(file: UploadFile = File(...)) -> str:
    """
    Основной эндпоинт для обработки PDF

    Принимает PDF файл, возвращает текст с нумерацией страниц

    Пример использования:
    curl -X POST -F "file=@document.pdf" http://localhost:9000/ocr
    """
    # Проверяем тип файла
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Файл должен быть в формате PDF"
        )

    # Читаем содержимое файла
    try:
        pdf_bytes = await file.read()

        # Проверяем, что файл не пустой
        if len(pdf_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Файл пуст"
            )

        # Обрабатываем PDF
        result_text = await process_pdf(file.filename, pdf_bytes)

        return result_text

    except Exception as e:
        # Логируем ошибку
        log_error(file.filename, e)

        # Возвращаем ошибку клиенту
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке файла: {str(e)}"
        )
    finally:
        # Принудительно освобождаем память
        if 'pdf_bytes' in locals():
            del pdf_bytes


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервера"""
    # Проверяем доступность Ollama
    try:
        async with session.get(f"{OLLAMA_URL}/api/tags") as response:
            if response.status == 200:
                models = await response.json()
                model_available = any(m.get("name", "").startswith(MODEL_NAME.split(":")[0])
                                      for m in models.get("models", []))
                return {
                    "status": "healthy",
                    "ollama": "connected",
                    "model_available": model_available
                }
    except:
        return {
            "status": "degraded",
            "ollama": "disconnected"
        }

    return {"status": "healthy", "ollama": "connected"}


@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о сервере"""
    return {
        "service": "PDF OCR Server",
        "version": "1.0.0",
        "endpoints": {
            "ocr": "POST /ocr - Отправить PDF файл для OCR",
            "health": "GET /health - Проверка состояния сервера"
        },
        "model": MODEL_NAME,
        "ollama_url": OLLAMA_URL
    }


if __name__ == "__main__":
    import uvicorn

    # Запускаем сервер
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )