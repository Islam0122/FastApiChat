import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
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

app = FastAPI(title="Real-Time Chat API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # на dev можно так
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Lifespan (НОВЫЙ ПРАВИЛЬНЫЙ СПОСОБ)
# ─────────────────────────────────────────────





# ─────────────────────────────────────────────
# REST API (proxy to Django)
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
# WebSocket CHAT (REALTIME CORE)
# ─────────────────────────────────────────────

@app.websocket("/ws/chat/{chat_id}")
async def websocket_chat(websocket: WebSocket, chat_id: int):
    redis = await get_redis()

    await manager.connect(websocket, chat_id, redis)

    # listener (Redis pub/sub → WS broadcast)
    listener_task = asyncio.create_task(
        manager.subscribe_and_listen(redis, chat_id)
    )

    try:
        while True:
            raw = await websocket.receive_text()

            # ── parse message ──
            try:
                data = json.loads(raw)
                user_message = (data.get("message") or "").strip()
            except Exception:
                await manager.send_personal_message(
                    json.dumps({"error": "Invalid JSON"}),
                    websocket,
                )
                continue

            if not user_message:
                await manager.send_personal_message(
                    json.dumps({"error": "Empty message"}),
                    websocket,
                )
                continue

            # ── save user message ──
            await django_api.save_message(chat_id, "user", user_message)

            # ── AI response (protected timeout) ──
            try:
                ai_text = await asyncio.wait_for(
                    get_ai_response(user_message),
                    timeout=20
                )
            except asyncio.TimeoutError:
                ai_text = "AI timeout. Try again."

            # ── save AI message ──
            await django_api.save_message(chat_id, "assistant", ai_text)

            payload = {
                "chat_id": chat_id,
                "role": "assistant",
                "message": ai_text,
            }

            # ── publish to Redis (broadcast system) ──
            await manager.publish(redis, chat_id, payload)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected chat=%s", chat_id)

    except Exception as e:
        logger.error("WebSocket error chat=%s error=%s", chat_id, e)

    finally:
        # SAFE cleanup (no leaks)
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        await manager.disconnect(websocket, chat_id, redis)