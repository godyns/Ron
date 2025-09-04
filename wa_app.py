import os, requests
from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse
from llm_brain import reply_as_ron

app = FastAPI()

VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "verify-me")
WA_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")

@app.get("/")
def root():
    return {"ok": True, "service": "ron-wa"}

@app.get("/webhook")
def verify(
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "", status_code=200)
    return PlainTextResponse("forbidden", status_code=403)

@app.post("/webhook")
async def inbound(req: Request):
    body = await req.json()
    try:
        entry = body["entry"][0]["changes"][0]["value"]
        if "messages" not in entry:
            return {"status": "ignored"}
        msg = entry["messages"][0]
        if msg.get("type") != "text":
            return {"status": "ignored"}
        from_id = msg["from"]
        text = msg["text"]["body"]
    except Exception:
        return {"status": "ignored"}

    try:
        try:
            reply = reply_as_ron(text, user_id=from_id)
        except TypeError:
            reply = reply_as_ron(text)
    except Exception:
        reply = "oops, thoda glitch hua 😅 fir se bhej de?"

    url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product":"whatsapp","to":from_id,"type":"text","text":{"body":reply}}
    try:
        requests.post(url, json=payload, headers=headers, timeout=30)
    except Exception:
        pass
    return {"status":"ok"}
