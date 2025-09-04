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
async def inbound(request: Request):
    try:
        data = await request.json()
        entry = data["entry"][0]["changes"][0]["value"]
        messages = entry.get("messages", [])
        if not messages:
            return {"status": "ignored"}
        
        msg = messages[0]
        # Only process text messages
        if msg.get("type") != "text":
            return {"status": "ignored"}
            
        user_text = msg["text"]["body"]
        user_number = msg["from"]

        # Get Ron's reply
        # The user_id parameter might not be supported by older versions of llm_brain
        try:
            ron_reply = reply_as_ron(user_text, user_id=user_number)
        except TypeError:
            ron_reply = reply_as_ron(user_text)

        # Send back via WhatsApp Cloud API
        url = f"https://graph.facebook.com/v20.0/{PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WA_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": user_number,
            "type": "text",
            "text": {"body": ron_reply},
        }
        requests.post(url, headers=headers, json=payload, timeout=30)

    except Exception as e:
        print("Webhook error:", e, data)
        return {"status": "ignored"}

    return {"status": "ok"}
