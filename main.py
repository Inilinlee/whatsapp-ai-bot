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
Анализируй диалог из WhatsApp и возвращай ТОЛЬКО валидный JSON, без любого другого текста, markdown или объяснений.

ВЫНУЖДЕННО используй ЭТУ ТОЧНУЮ СХЕМУ JSON (все поля обязательны, даже если пустые):

{
  "task_title": "короткое название задачи (1-3 слова)",
  "task_description": "развёрнутое описание задачи с деталями",
  "intent": "вывеска" (или led-экран, мерч, полиграфия, другое),
  "client_need_summary": "краткое описание потребности клиента в 1-3 предложениях",
  "executor": "manager" (если нужен расчёт/КП) или "designer" (если дизайн/макет),
  "priority": "low" (просто интересуюсь) / "normal" (стандарт) / "high" (срочно, дедлайн <3 дней),
  "need_clarification": true (если не хватает данных) или false,
  "clarification_questions": [{"question": "конкретный вопрос клиенту"}, ...] (до 5, только если need_clarification=true),
  "tags": ["тег1", "тег2"] (например: дизайн, монтаж, срочно),
  "required_files": ["файл1", "файл2"] (например: логотип, фото фасада),
  "ai_status": "готово" (если всё ок) или "уточнение" (если вопросы)
}

Определи intent по ключевым словам. Если срочно — high. Если не хватает размеров/бюджета/логотипа — true и вопросы."""

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
