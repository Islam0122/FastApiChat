"""
ai_service.py — умный агент для поиска квартир.

Логика:
1. Каждое сообщение пользователя уходит в GigaChat вместе с системным
   промптом, историей диалога и контекстом (найденные квартиры / статистика).
2. GigaChat отвечает ЛИБО вопросом на уточнение, ЛИБО JSON-командой вида:
       {"action": "search_flats", "params": {...}}
   ЛИБО JSON-командой:
       {"action": "reply", "text": "..."}
3. FastAPI перехватывает JSON-команды, выполняет запрос к Django API,
   добавляет результаты в контекст и делает второй вызов GigaChat —
   тот уже формулирует финальный ответ пользователю.
"""

import json
import logging
import re
import ssl
import uuid
from typing import Any

import aiohttp
from aiohttp import BasicAuth, ClientTimeout

from core.config import settings

logger = logging.getLogger(__name__)

TIMEOUT = ClientTimeout(total=30)
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE

# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Ты — умный ассистент агентства недвижимости в Бишкеке.
Твоя задача — помочь пользователю найти квартиру из реальной базы объявлений.

У тебя есть инструмент поиска квартир. Когда нужно найти квартиры — верни СТРОГО JSON:
{"action": "search_flats", "params": {"min_rooms": N, "max_rooms": N, "min_price": N, "max_price": N, "district": "...", "source": "house|lalafo", "is_urgent": true/false, "is_owner": true/false, "search": "...", "ordering": "price|-price|-created_at"}}

Все параметры необязательны — включай только те, что упомянул пользователь.
Цены — в долларах США ($). Район — на русском языке (Магистраль, Политех, Джал, Асанбай, Тунгуч, Свердлова и т.д.).

Когда хочешь показать выгодные квартиры (ниже рынка) — верни:
{"action": "search_profitable"}

Когда хочешь показать статистику рынка — верни:
{"action": "get_market_stats", "params": {"rooms": N}}

Правила поведения:
1. Если пользователь написал расплывчато — задай 1–2 уточняющих вопроса (район, бюджет, количество комнат). НЕ задавай сразу все вопросы.
2. Если данных достаточно — сразу ищи, не спрашивай лишнего.
3. После получения результатов поиска — выбери 3–5 лучших вариантов и объясни ПОЧЕМУ они подходят: цена, район, площадь, этаж, скидка от рынка.
4. Если квартир не нашлось — предложи расширить критерии.
5. Говори по-русски, дружелюбно и кратко.
6. Никогда не выдумывай квартиры — только реальные данные из базы.
7. Если вопрос не про квартиры — отвечай как обычный ассистент, НО всегда мягко возвращай к теме недвижимости.

Когда просто отвечаешь текстом (не команда поиска) — пиши обычный текст, БЕЗ JSON.
"""

# ─────────────────────────────────────────────
# GigaChat низкоуровневые вызовы
# ─────────────────────────────────────────────

async def _get_token() -> str | None:
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
    }
    try:
        async with aiohttp.ClientSession(timeout=TIMEOUT) as s:
            async with s.post(
                settings.GIGACHAT_OAUTH_URL,
                headers=headers,
                data={"scope": "GIGACHAT_API_PERS"},
                auth=BasicAuth(settings.GIGACHAT_CLIENT_ID, settings.GIGACHAT_SECRET),
                ssl=_ssl_ctx,
            ) as r:
                r.raise_for_status()
                data = await r.json()
                return data.get("access_token")
    except aiohttp.ClientError as e:
        logger.error("_get_token error: %s", e)
        return None


async def _chat(messages: list[dict], token: str) -> str | None:
    """Один вызов GigaChat. messages — список {role, content}."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }
    body = {
        "model": "GigaChat",
        "temperature": 0.3,   # пониже для более точных ответов
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
    }
    try:
        async with aiohttp.ClientSession(timeout=TIMEOUT) as s:
            async with s.post(
                settings.GIGACHAT_API_URL,
                headers=headers,
                json=body,
                ssl=_ssl_ctx,
            ) as r:
                r.raise_for_status()
                data = await r.json()
                return data["choices"][0]["message"]["content"]
    except (aiohttp.ClientError, KeyError, IndexError) as e:
        logger.error("_chat error: %s", e)
        return None


# ─────────────────────────────────────────────
# Парсинг JSON-команды из ответа GigaChat
# ─────────────────────────────────────────────

def _extract_action(text: str) -> dict | None:
    """Ищет JSON-команду в тексте ответа модели."""
    # Попытка 1: весь текст — валидный JSON
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict) and "action" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # Попытка 2: JSON внутри ```json ... ``` или просто { ... }
    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r"(\{[^{}]*\"action\"[^{}]*\})",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict) and "action" in obj:
                    return obj
            except json.JSONDecodeError:
                pass
    return None


# ─────────────────────────────────────────────
# Форматирование квартир для AI-контекста
# ─────────────────────────────────────────────

def _flat_to_context(flat: dict, index: int) -> str:
    price  = f"${flat.get('price', '?'):,}".replace(",", " ")
    ppm2   = f"{flat['price_per_m2']:.0f} $/м²" if flat.get("price_per_m2") else "—"
    rooms  = flat.get("rooms", "?")
    area   = f"{flat.get('area', '?')} м²"
    floor  = flat.get("floor_info") or (
        f"{flat.get('floor','?')}/{flat.get('total_floors','?')} эт."
        if flat.get("floor") else "—"
    )
    dist   = flat.get("district") or "не указан"
    addr   = flat.get("address", "")
    link   = flat.get("link", "")
    flags  = []
    if flat.get("is_urgent"): flags.append("СРОЧНО")
    if flat.get("is_owner"):  flags.append("от хозяина")
    discount = flat.get("discount_from_market")
    if discount:              flags.append(f"ниже рынка на {discount:.1f}%")
    flags_str = f" [{', '.join(flags)}]" if flags else ""

    return (
        f"{index}. {rooms}-комн., {area}, {floor} эт., район: {dist}\n"
        f"   Цена: {price} | {ppm2}{flags_str}\n"
        f"   Адрес: {addr}\n"
        f"   Ссылка: {link}"
    )


def _fmt_flats_for_user(flats: list[dict]) -> str:
    """Красивый текст квартир для финального ответа пользователю."""
    lines = []
    for i, flat in enumerate(flats, 1):
        price = f"${flat.get('price', '?'):,}".replace(",", " ")
        ppm2  = f"{flat['price_per_m2']:.0f} $/м²" if flat.get("price_per_m2") else ""
        rooms = flat.get("rooms", "?")
        area  = flat.get("area", "?")
        floor = flat.get("floor_info") or (
            f"{flat.get('floor')}/{flat.get('total_floors')} эт."
            if flat.get("floor") else "—"
        )
        dist  = flat.get("district") or "—"
        link  = flat.get("link", "")
        flags = []
        if flat.get("is_urgent"):           flags.append("🔥 Срочно")
        if flat.get("is_owner"):            flags.append("👤 Хозяин")
        d = flat.get("discount_from_market")
        if d:                               flags.append(f"🔻 -{d:.1f}% от рынка")
        flags_str = "  " + "  ".join(flags) if flags else ""

        lines.append(
            f"🏠 <b>{i}. {rooms}-комн., {area} м²</b> | {floor} | {dist}\n"
            f"💰 {price}" + (f" | {ppm2}" if ppm2 else "") + flags_str + "\n"
            f"🔗 {link}"
        )
    return "\n\n".join(lines)


# ─────────────────────────────────────────────
# Основная точка входа
# ─────────────────────────────────────────────

async def get_ai_response(
    user_message: str,
    history: list[dict] | None = None,
) -> str:
    """
    Умный агент.

    history — список {"role": "user"|"assistant", "content": "..."}
              (последние N сообщений из чата, без системного промпта).
    """
    from services import django_api   # импорт здесь, чтобы избежать циклов

    history = history or []

    token = await _get_token()
    if not token:
        return "Извините, сервис временно недоступен."

    # Строим messages для первого вызова
    messages = history + [{"role": "user", "content": user_message}]

    # ── Первый вызов: AI решает что делать ──
    raw = await _chat(messages, token)
    if not raw:
        return "Не удалось получить ответ. Попробуйте ещё раз."

    action = _extract_action(raw)

    # ── Если AI просто отвечает текстом ──
    if not action:
        return raw.strip()

    # ── Выполняем команду ──
    act  = action.get("action", "")
    params: dict[str, Any] = action.get("params", {})

    search_results_text = ""
    flats_for_user: list[dict] = []

    if act == "search_flats":
        data = await django_api.get_flats(**{k: v for k, v in params.items() if v is not None})
        if data and data.get("results"):
            results = data["results"]
            total   = data.get("count", len(results))
            top     = results[:8]   # передаём AI не больше 8
            flats_for_user = results[:5]  # покажем пользователю 5
            ctx_lines = [f"Найдено {total} квартир. Топ вариантов:"]
            for i, f in enumerate(top, 1):
                ctx_lines.append(_flat_to_context(f, i))
            search_results_text = "\n".join(ctx_lines)
        else:
            search_results_text = "Квартиры по заданным критериям не найдены."
            flats_for_user = []

    elif act == "search_profitable":
        data = await django_api.get_profitable_flats()
        if data and data.get("results"):
            results = data["results"]
            flats_for_user = results[:5]
            ctx_lines = [f"Найдено {data['count']} выгодных квартир:"]
            for i, f in enumerate(results[:8], 1):
                ctx_lines.append(_flat_to_context(f, i))
            search_results_text = "\n".join(ctx_lines)
        else:
            search_results_text = "Выгодных квартир сейчас нет."

    elif act == "get_market_stats":
        rooms = params.get("rooms")
        data  = await django_api.get_market_stats(rooms=rooms)
        rows  = (data if isinstance(data, list) else (data or {}).get("results", [])) if data else []
        if rows:
            lines = ["Статистика рынка (средняя и медианная цена за м²):"]
            cur = None
            for row in rows:
                if row["rooms"] != cur:
                    cur = row["rooms"]
                    lines.append(f"\n{cur}-комнатные:")
                lines.append(
                    f"  {row['district']}: ср. {row['avg_price_per_m2']:.0f} | "
                    f"мед. {row['median_price_per_m2']:.0f} $/м²"
                )
            search_results_text = "\n".join(lines)
        else:
            search_results_text = "Статистика рынка недоступна."

    else:
        # Неизвестная команда — возвращаем сырой текст
        return raw.strip()

    # ── Второй вызов: AI формулирует ответ на основе данных ──
    follow_up = (
        f"Результаты поиска:\n\n{search_results_text}\n\n"
        "На основе этих данных выбери 3–5 лучших вариантов и объясни пользователю "
        "почему именно они подходят (цена, район, площадь, этаж, скидка от рынка). "
        "Пиши дружелюбно и по-русски. НЕ выводи JSON. "
        "Ссылки на объявления обязательно включи в ответ."
    )

    messages2 = messages + [
        {"role": "assistant", "content": raw},
        {"role": "user",      "content": follow_up},
    ]

    final_raw = await _chat(messages2, token)
    if not final_raw:
        # Если второй вызов упал — отдаём базовый список
        if flats_for_user:
            return (
                "Вот что я нашёл:\n\n"
                + _fmt_flats_for_user(flats_for_user)
            )
        return search_results_text

    # Убираем случайный JSON из финального ответа
    final_action = _extract_action(final_raw)
    if final_action:
        if flats_for_user:
            return "Вот подходящие варианты:\n\n" + _fmt_flats_for_user(flats_for_user)
        return search_results_text

    # Если в финальном ответе нет карточек квартир — добавляем снизу
    if flats_for_user and not any(f.get("link", "") in final_raw for f in flats_for_user):
        return final_raw.strip() + "\n\n" + _fmt_flats_for_user(flats_for_user)

    return final_raw.strip()