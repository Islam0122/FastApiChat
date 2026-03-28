import uuid
import json
import logging
import ssl

import aiohttp
from aiohttp import BasicAuth, ClientTimeout

from core.config import settings

logger = logging.getLogger(__name__)

TIMEOUT = ClientTimeout(total=30)
_ssl_context = ssl.create_default_context()
_ssl_context.check_hostname = False
_ssl_context.verify_mode = ssl.CERT_NONE


async def _get_access_token() -> str | None:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
    }
    payload = {"scope": "GIGACHAT_API_PERS"}

    try:
        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            async with session.post(
                settings.GIGACHAT_OAUTH_URL,
                headers=headers,
                data=payload,
                auth=BasicAuth(settings.GIGACHAT_CLIENT_ID, settings.GIGACHAT_SECRET),
                ssl=_ssl_context,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                token = data.get("access_token")
                if not token:
                    logger.error("access_token missing in response")
                    return None
                return token
    except aiohttp.ClientError as e:
        logger.error("_get_access_token error: %s", e)
        return None


async def _send_prompt(message: str, access_token: str) -> str | None:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    payload = {
        "model": "GigaChat",
        "temperature": 0.7,
        "messages": [{"role": "user", "content": message}],
    }

    try:
        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            async with session.post(
                settings.GIGACHAT_API_URL,
                headers=headers,
                json=payload,
                ssl=_ssl_context,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
    except (aiohttp.ClientError, KeyError, IndexError) as e:
        logger.error("_send_prompt error: %s", e)
        return None


async def get_ai_response(message: str) -> str:
    token = await _get_access_token()
    if not token:
        logger.warning("No access token — returning fallback response")
        return "Извините, сервис временно недоступен."

    response = await _send_prompt(message, token)
    if not response:
        return "Не удалось получить ответ от GigaChat."

    return response