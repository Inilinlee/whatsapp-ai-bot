from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
import openai
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VT Group WhatsApp AI")

openai.api_key = os.getenv("OPENAI_API_KEY")

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None

class RequestData(BaseModel):
    task_id: int
    client_phone: str
    client_name: Optional[str] = None
    channel: str = "whatsapp"
    city: Optional[str] = None
    dialog_history: List[Message]

class Clarification(BaseModel):
    question: str

class ResponseData(BaseModel):
    task_title: str = Field(..., max_length=150)
    task_description: str
    intent: Literal["вывеска", "led-экран", "мерч", "полиграфия", "другое"]
    client_need_summary: str
    executor: Literal["manager", "designer"]
    priority: Literal["low", "normal", "high"]
    need_clarification: bool = False
    clarification_questions: List[Clarification] = Field(default=[], max_items=5)
    tags: List[str] = Field(default=[])
    required_files: List[str] = Field(default=[])
    ai_status: Literal["готово", "уточнение"] = "готово"

SYSTEM_PROMPT = """Ты — супер-умный помощник компании VT Group (наружная реклама, LED-экраны, мерч, полиграфия).
Анализируй диалог из WhatsApp и возвращай только валидный JSON строго по схеме ниже.

Обязательно определяй:
- intent (одно из: вывеска, led-экран, мерч, полиграфия, другое)
- краткую потребность клиента (1–3 предложения)
- кто должен вести задачу: manager или designer
- приоритет (high только если срочно)
- если чего-то критически не хватает — ставь need_clarification=true и задай до 5 точных вопросов
"""

@app.post("/webhook/analyze")
async def analyze(data: RequestData):
    start = datetime.now()
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in data.dialog_history[-20:]:
        role = "user" if msg.role == "user" else "assistant"
        messages.append({"role": role, "content": msg.content})

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=1200
        )
        raw = resp.choices[0].message.content.strip()
        result = ResponseData.model_validate_json(raw)
        
        if result.need_clarification:
            result.ai_status = "уточнение"
            
        logger.info(f"Успешно за {(datetime.now()-start).total_seconds():.1f}с")
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}
