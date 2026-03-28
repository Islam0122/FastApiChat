import asyncio
import json
import logging
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core.redis_c import get_redis, close_redis
from services import django_api
from services.ai_service import get_ai_response
from websocket.manager import manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real-Time Chat + Flats API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Сколько последних сообщений (user+assistant пар) передавать AI
HISTORY_LIMIT = 10
# TTL ключа истории в Redis (24 часа)
HISTORY_TTL = 86_400


# ─────────────────────────────────────────────
# История диалога в Redis
# ─────────────────────────────────────────────

async def _load_history(redis, chat_id: int) -> list[dict]:
    key = f"chat:{chat_id}:history"
    raw = await redis.get(key)
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


async def _save_history(redis, chat_id: int, history: list[dict]) -> None:
    key = f"chat:{chat_id}:history"
    trimmed = history[-(HISTORY_LIMIT * 2):]
    await redis.set(key, json.dumps(trimmed, ensure_ascii=False), ex=HISTORY_TTL)


# ─────────────────────────────────────────────
# REST API — Chats (proxy to Django)
# ─────────────────────────────────────────────

@app.get("/api/chats/")
async def list_chats():
    data = await django_api.get_chats()
    if data is None:
        raise HTTPException(502, "Django API unavailable")
    return JSONResponse(data)


@app.post("/api/chats/")
async def create_chat(payload: dict):
    data = await django_api.create_chat(payload)
    if data is None:
        raise HTTPException(502, "Django API unavailable")
    return JSONResponse(data, status_code=201)


@app.get("/api/chats/{chat_id}/")
async def get_chat(chat_id: int):
    data = await django_api.get_chat(chat_id)
    if data is None:
        raise HTTPException(404, "Chat not found")
    return JSONResponse(data)


@app.patch("/api/chats/{chat_id}/")
async def patch_chat(chat_id: int, payload: dict):
    data = await django_api.patch_chat(chat_id, payload)
    if data is None:
        raise HTTPException(502, "Django API unavailable")
    return JSONResponse(data)


@app.delete("/api/chats/{chat_id}/")
async def delete_chat(chat_id: int):
    ok = await django_api.delete_chat(chat_id)
    if not ok:
        raise HTTPException(502, "Django API unavailable")
    return JSONResponse({"detail": "deleted"}, status_code=204)


@app.get("/api/messages/")
async def list_messages(chat_id: int | None = None):
    data = await django_api.get_messages(chat_id)
    if data is None:
        raise HTTPException(502, "Django API unavailable")
    return JSONResponse(data)


# ─────────────────────────────────────────────
# REST API — Flats (proxy to Django)
# ─────────────────────────────────────────────

@app.get("/api/flats/")
async def list_flats(
    page:      int             = Query(1,    description="Номер страницы"),
    search:    Optional[str]   = Query(None, description="Поиск по заголовку / адресу"),
    min_price: Optional[int]   = Query(None, description="Минимальная цена ($)"),
    max_price: Optional[int]   = Query(None, description="Максимальная цена ($)"),
    min_area:  Optional[float] = Query(None, description="Минимальная площадь (м²)"),
    max_area:  Optional[float] = Query(None, description="Максимальная площадь (м²)"),
    min_rooms: Optional[int]   = Query(None, description="Минимум комнат"),
    max_rooms: Optional[int]   = Query(None, description="Максимум комнат"),
    district:  Optional[str]   = Query(None, description="Район (частичное совпадение)"),
    source:    Optional[str]   = Query(None, description="Источник: house | lalafo"),
    is_urgent: Optional[bool]  = Query(None, description="Только срочные"),
    is_owner:  Optional[bool]  = Query(None, description="Только от собственника"),
    ordering:  Optional[str]   = Query(None, description="Сортировка"),
):
    data = await django_api.get_flats(
        page=page, search=search,
        min_price=min_price, max_price=max_price,
        min_area=min_area,   max_area=max_area,
        min_rooms=min_rooms, max_rooms=max_rooms,
        district=district,   source=source,
        is_urgent=is_urgent, is_owner=is_owner,
        ordering=ordering,
    )
    if data is None:
        raise HTTPException(502, "Django API unavailable")
    return JSONResponse(data)


@app.get("/api/flats/profitable/")
async def profitable_flats():
    data = await django_api.get_profitable_flats()
    if data is None:
        raise HTTPException(502, "Django API unavailable")
    return JSONResponse(data)


@app.get("/api/flats/{flat_id}/")
async def get_flat(flat_id: int):
    data = await django_api.get_flat(flat_id)
    if data is None:
        raise HTTPException(404, "Flat not found")
    return JSONResponse(data)


@app.get("/api/market-stats/")
async def market_stats(
    rooms:    Optional[int] = Query(None),
    district: Optional[str] = Query(None),
):
    data = await django_api.get_market_stats(rooms=rooms, district=district)
    if data is None:
        raise HTTPException(502, "Django API unavailable")
    return JSONResponse(data)


@app.get("/api/stats/")
async def stats_summary():
    data = await django_api.get_stats_summary()
    if data is None:
        raise HTTPException(502, "Django API unavailable")
    return JSONResponse(data)


# ─────────────────────────────────────────────
# WebSocket CHAT — умный агент с памятью
# ─────────────────────────────────────────────

@app.websocket("/ws/chat/{chat_id}")
async def websocket_chat(websocket: WebSocket, chat_id: int):
    redis = await get_redis()

    await manager.connect(websocket, chat_id, redis)

    listener_task = asyncio.create_task(
        manager.subscribe_and_listen(redis, chat_id)
    )

    try:
        while True:
            raw = await websocket.receive_text()

            # ── parse ──
            try:
                data         = json.loads(raw)
                user_message = (data.get("message") or "").strip()
            except Exception:
                await manager.send_personal_message(
                    json.dumps({"error": "Invalid JSON"}), websocket
                )
                continue

            if not user_message:
                await manager.send_personal_message(
                    json.dumps({"error": "Empty message"}), websocket
                )
                continue

            # ── сохранить сообщение пользователя в Django ──
            await django_api.save_message(chat_id, "user", user_message)

            # ── загрузить историю диалога из Redis ──
            history = await _load_history(redis, chat_id)

            # ── вызов умного агента ──
            try:
                ai_text = await asyncio.wait_for(
                    get_ai_response(
                        user_message=user_message,
                        history=history,
                    ),
                    timeout=45,  # агент делает 2 вызова GigaChat + запрос к Django
                )
            except asyncio.TimeoutError:
                ai_text = "⏱ Запрос занял слишком много времени. Попробуйте ещё раз."

            # ── обновить историю в Redis ──
            history.append({"role": "user",     "content": user_message})
            history.append({"role": "assistant", "content": ai_text})
            await _save_history(redis, chat_id, history)

            # ── сохранить ответ AI в Django ──
            await django_api.save_message(chat_id, "assistant", ai_text)

            payload = {
                "chat_id": chat_id,
                "role":    "assistant",
                "message": ai_text,
            }

            await manager.publish(redis, chat_id, payload)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected chat=%s", chat_id)

    except Exception as e:
        logger.error("WebSocket error chat=%s error=%s", chat_id, e)

    finally:
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass
        await manager.disconnect(websocket, chat_id, redis)