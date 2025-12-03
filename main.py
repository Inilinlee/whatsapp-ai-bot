import os
import json
import time
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
import openai

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")
SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
MODEL = "gpt-4o-mini"
CACHE = {}

def load_prices():
    global CACHE
    if "cal" in CACHE and time.time() - CACHE["time"] < 600:
        return CACHE["data"]
    
    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid="
    try:
        cal = pd.read_csv(url + "0")  # если Календари — первый лист
        CACHE = {"data": cal.to_dict("records"), "time": time.time()}
        return CACHE["data"]
    except:
        return CACHE.get("data", [])

SYSTEM_PROMPT = """Ты — живой продавец-консультант VT Group. 
Говоришь естественно, с душой, чувствуешь настроение клиента. 
Используй только актуальные цены из таблицы ниже. 
Предлагай 3 тиража с выгодой, вплетай вопросы естественно.
Ответ только в JSON!"""

@app.post("/webhook/analyze")
async def analyze(req: Request):
    body = await req.json()
    prices = load_prices()
    history = "\n".join([f"{m['role']}: {m['content']}" for m in body["dialog_history"][-15:]])
    
    prompt = f"Диалог:\n{history}\n\nАктуальные цены:\n{pd.DataFrame(prices).to_string()}\n\nСоставь живой ответ + цены + вопросы."
    
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    
    return json.loads(resp.choices[0].message.content)
