"""
Клиент для работы с vLLM API (OpenAI-совместимый)
"""
import base64
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class VLLMClient:
    """Клиент для взаимодействия с vLLM сервером"""

    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    async def process_image(self, session, image_bytes: bytes, prompt: str) -> str:
        """
        Отправляет изображение в vLLM для обработки

        Args:
            session: aiohttp ClientSession
            image_bytes: изображение в байтах
            prompt: текстовый промпт

        Returns:
            Ответ от модели
        """
        try:
            # Кодируем изображение в base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # Формируем сообщение в формате OpenAI
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high"  # Высокое качество для OCR
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            # Формируем запрос к vLLM
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.1,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "stream": False
            }

            # Отправляем запрос
            async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"vLLM API error {response.status}: {error_text}")
                    raise Exception(f"vLLM вернул ошибку {response.status}: {error_text}")

                result = await response.json()

                # Извлекаем текст из ответа
                if "choices" in result and len(result["choices"]) > 0:
                    text = result["choices"][0]["message"]["content"]
                    return text.strip()
                else:
                    logger.warning(f"Неожиданный формат ответа: {result}")
                    return ""

        except Exception as e:
            logger.error(f"Ошибка при запросе к vLLM: {e}")
            raise


# Создаем глобальный экземпляр клиента
vllm_client = None


def init_vllm_client(base_url: str, api_key: str, model_name: str):
    """Инициализация клиента vLLM"""
    global vllm_client
    vllm_client = VLLMClient(base_url, api_key, model_name)
    return vllm_client