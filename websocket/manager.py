import json
import logging
from collections import defaultdict

from fastapi import WebSocket
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[int, list[WebSocket]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, chat_id: int, redis: Redis) -> None:
        await websocket.accept()
        self._connections[chat_id].append(websocket)
        await redis.sadd(f"chat:{chat_id}:connections", id(websocket))
        logger.info("WS connected: chat=%s total=%s", chat_id, len(self._connections[chat_id]))

    async def disconnect(self, websocket: WebSocket, chat_id: int, redis: Redis) -> None:
        connections = self._connections.get(chat_id, [])
        if websocket in connections:
            connections.remove(websocket)
        await redis.srem(f"chat:{chat_id}:connections", id(websocket))
        if not connections:
            del self._connections[chat_id]
        logger.info("WS disconnected: chat=%s remaining=%s", chat_id, len(connections))

    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error("send_personal_message error: %s", e)

    async def broadcast(self, message: str, chat_id: int) -> None:
        connections = self._connections.get(chat_id, [])
        dead: list[WebSocket] = []
        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning("broadcast send failed: %s", e)
                dead.append(ws)
        for ws in dead:
            if ws in connections:
                connections.remove(ws)

    async def publish(self, redis: Redis, chat_id: int, payload: dict) -> None:
        await redis.publish(f"chat:{chat_id}", json.dumps(payload))

    async def subscribe_and_listen(self, redis: Redis, chat_id: int) -> None:
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"chat:{chat_id}")
        try:
            async for raw in pubsub.listen():
                if raw["type"] != "message":
                    continue
                await self.broadcast(raw["data"], chat_id)
        finally:
            await pubsub.unsubscribe(f"chat:{chat_id}")
            await pubsub.close()


manager = ConnectionManager()