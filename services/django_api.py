import logging
from typing import Optional

import aiohttp
from aiohttp import ClientTimeout

from core.config import settings

logger = logging.getLogger(__name__)

TIMEOUT = ClientTimeout(total=10)


def _get_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        base_url=settings.DJANGO_API_BASE,
        timeout=TIMEOUT,
        headers={"Content-Type": "application/json"},
    )


# ─────────────────────────────────────────────
# Chats
# ─────────────────────────────────────────────

async def get_chats() -> list[dict] | None:
    try:
        async with _get_session() as session:
            async with session.get("chats/") as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error("get_chats failed: %s", resp.status)
                return None
    except aiohttp.ClientError as e:
        logger.error("get_chats error: %s", e)
        return None


async def get_chat(chat_id: int) -> dict | None:
    try:
        async with _get_session() as session:
            async with session.get(f"chats/{chat_id}/") as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error("get_chat(%s) failed: %s", chat_id, resp.status)
                return None
    except aiohttp.ClientError as e:
        logger.error("get_chat error: %s", e)
        return None


async def create_chat(payload: dict) -> dict | None:
    try:
        async with _get_session() as session:
            async with session.post("chats/", json=payload) as resp:
                if resp.status in (200, 201):
                    return await resp.json()
                logger.error("create_chat failed: %s — %s", resp.status, await resp.text())
                return None
    except aiohttp.ClientError as e:
        logger.error("create_chat error: %s", e)
        return None


async def patch_chat(chat_id: int, payload: dict) -> dict | None:
    try:
        async with _get_session() as session:
            async with session.patch(f"chats/{chat_id}/", json=payload) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error("patch_chat(%s) failed: %s", chat_id, resp.status)
                return None
    except aiohttp.ClientError as e:
        logger.error("patch_chat error: %s", e)
        return None


async def delete_chat(chat_id: int) -> bool:
    try:
        async with _get_session() as session:
            async with session.delete(f"chats/{chat_id}/") as resp:
                if resp.status == 204:
                    return True
                logger.error("delete_chat(%s) failed: %s", chat_id, resp.status)
                return False
    except aiohttp.ClientError as e:
        logger.error("delete_chat error: %s", e)
        return False


async def get_messages(chat_id: int | None = None) -> list[dict] | None:
    try:
        params = {"chat_id": chat_id} if chat_id else {}
        async with _get_session() as session:
            async with session.get("messages/", params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error("get_messages failed: %s", resp.status)
                return None
    except aiohttp.ClientError as e:
        logger.error("get_messages error: %s", e)
        return None


async def save_message(chat_id: int, role: str, text: str) -> dict | None:
    payload = {"chat": chat_id, "role": role, "text": text}
    try:
        async with _get_session() as session:
            async with session.post("messages/", json=payload) as resp:
                if resp.status in (200, 201):
                    return await resp.json()
                logger.error("save_message failed: %s — %s", resp.status, await resp.text())
                return None
    except aiohttp.ClientError as e:
        logger.error("save_message error: %s", e)
        return None


# ─────────────────────────────────────────────
# Flats
# ─────────────────────────────────────────────

async def get_flats(
    page: int = 1,
    search: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    min_area: Optional[float] = None,
    max_area: Optional[float] = None,
    min_rooms: Optional[int] = None,
    max_rooms: Optional[int] = None,
    district: Optional[str] = None,
    source: Optional[str] = None,
    is_urgent: Optional[bool] = None,
    is_owner: Optional[bool] = None,
    ordering: Optional[str] = None,
) -> dict | None:
    params: dict = {"page": page}
    if search:      params["search"]    = search
    if min_price:   params["min_price"] = min_price
    if max_price:   params["max_price"] = max_price
    if min_area:    params["min_area"]  = min_area
    if max_area:    params["max_area"]  = max_area
    if min_rooms:   params["min_rooms"] = min_rooms
    if max_rooms:   params["max_rooms"] = max_rooms
    if district:    params["district"]  = district
    if source:      params["source"]    = source
    if ordering:    params["ordering"]  = ordering
    if is_urgent is not None: params["is_urgent"] = is_urgent
    if is_owner  is not None: params["is_owner"]  = is_owner

    try:
        async with _get_session() as session:
            async with session.get("flats/", params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error("get_flats failed: %s", resp.status)
                return None
    except aiohttp.ClientError as e:
        logger.error("get_flats error: %s", e)
        return None


async def get_flat(flat_id: int) -> dict | None:
    try:
        async with _get_session() as session:
            async with session.get(f"flats/{flat_id}/") as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error("get_flat(%s) failed: %s", flat_id, resp.status)
                return None
    except aiohttp.ClientError as e:
        logger.error("get_flat error: %s", e)
        return None


async def get_profitable_flats() -> dict | None:
    try:
        async with _get_session() as session:
            async with session.get("flats/profitable/") as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error("get_profitable_flats failed: %s", resp.status)
                return None
    except aiohttp.ClientError as e:
        logger.error("get_profitable_flats error: %s", e)
        return None


async def get_market_stats(
    rooms: Optional[int] = None,
    district: Optional[str] = None,
) -> list | None:
    params = {}
    if rooms:    params["rooms"]    = rooms
    if district: params["district"] = district
    try:
        async with _get_session() as session:
            async with session.get("market-stats/", params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error("get_market_stats failed: %s", resp.status)
                return None
    except aiohttp.ClientError as e:
        logger.error("get_market_stats error: %s", e)
        return None


async def get_stats_summary() -> dict | None:
    try:
        async with _get_session() as session:
            async with session.get("stats/") as resp:
                if resp.status == 200:
                    return await resp.json()
                logger.error("get_stats_summary failed: %s", resp.status)
                return None
    except aiohttp.ClientError as e:
        logger.error("get_stats_summary error: %s", e)
        return None