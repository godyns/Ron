# llm_brain.py
from dotenv import load_dotenv
load_dotenv()
import os
if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()
import re, json, time
import requests
from typing import List, Dict, Optional
from openai import OpenAI
from persona_facts import PERSONA_DETAILS, ARCHETYPE_SUMMARIES

# ===== Provider Switch =====
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ===== Base System Prompt (keep this concise) =====
SYSTEM_PROMPT = """ROLE & IDENTITY
- You are **Ron Grover**, 22, from Bandra, Mumbai.
- **Language**: Hinglish only, Roman script. Use Gen-Z lingo.
- **Tone**: Magnetic, empathetic, chaotic-in-a-cute-way. Never formal or generic.
- **Length**: 1-3 short sentences by default.
- **Slang**: bro, yaar, low-key, high-key, ngl, legit, mast, sahi hai, scene on, chill.
- **Emoji**: 0-2 casual emojis max.
- **Style**: Avoid assistant vibes ("How can I assist?"). Validate feelings, then suggest one tiny next step.
- **Consistency**: Refer to your cat Champ, your mom/dad backstory, or your work as a stand-up comic when it fits.

CURRENT CONTEXT & KEY FACTS
- {{PERSONA_BLOB}}

RESPONSE SHAPE RULES
- Default: 10–25 words, 1–3 sentences, 0–2 emojis.
- If user explicitly asks for a plan/steps: 3–6 tight bullets (max 8 words each).
- Never use markdown headings, lists, or quotes unless asked.
- If reply is too formal or generic, rewrite it to be casual and specific."""

# Terms we DON'T want (to detect/formality)
BAD_PHRASES = [
    "assist you", "how can i assist", "i am an ai", "as an ai", "chatbot", "i am here to help",
    "let us", "let's explore", "i am functioning", "meaningful conversation", "further together"
]
# Devanagari Unicode range
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# ===== Minimal JSONL logging for now (swap to Supabase later) =====
LOG_PATH = os.path.join(os.path.dirname(__file__), "ron_messages.jsonl")

def log_jsonl(rec: dict):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ===== New function to build a dynamic persona blob =====
def build_persona_blob(history: Optional[List[Dict[str, str]]] = None) -> str:
    # A simple example: always include a few key facts
    facts = [
        f"Ron lives with a {PERSONA_DETAILS['pet']}."
    ]

    # Look for keywords in the last few turns of the conversation to add specific context
    last_user_message = history[-1]['content'].lower() if history else ""
    
    if "stand-up" in last_user_message or "comic" in last_user_message:
        facts.append(f"Ron works as a {PERSONA_DETAILS['work']}.")
    if "mood" in last_user_message or "off" in last_user_message:
        facts.append(f"Ron's a chaotic-in-a-cute-way ENFP who validates feelings.")
        facts.append(f"Ron's life motto is '{PERSONA_DETAILS['motto']}'.")
    if "dad" in last_user_message or "mom" in last_user_message or "family" in last_user_message:
        facts.append(f"Ron's family backstory: {PERSONA_DETAILS['origin']}.")
    
    return " | ".join(facts)

# ========= Providers =========
def _ollama_chat(messages: List[Dict[str, str]]) -> str:
    r = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "phi3",
            "messages": messages,
            "options": {"temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.05},
            "stream": False
        },
        timeout=90,
    )
    r.raise_for_status()
    # Some Ollama builds still newline-stream with stream:false; guard it:
    try:
        data = r.json()
    except json.JSONDecodeError:
        first_line = r.text.strip().splitlines()[0]
        data = json.loads(first_line)
    return data["message"]["content"]

def _openai_chat(messages: List[Dict[str, str]]) -> str:
    client = OpenAI()  # uses OPENAI_API_KEY from env
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        top_p=0.9,
        presence_penalty=0.1,
        frequency_penalty=0.2,
    )
    return resp.choices[0].message.content

def _call_provider(messages: List[Dict[str, str]]) -> str:
    if LLM_PROVIDER == "openai":
        return _openai_chat(messages)
    return _ollama_chat(messages)

# ===== Post-process: auto-rewrite if formal/Devanagari =====
def _needs_rewrite(text: str) -> bool:
    if DEVANAGARI_RE.search(text):
        return True
    lower = text.lower()
    if any(bad in lower for bad in BAD_PHRASES):
        return True
    if len(text) > 260 and ("." in text or "," in text):
        return True
    return False

def _rewrite_to_hinglish(original: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Rewrite this in Hinglish (Roman), Gen-Z, short (1–3 sentences), casual emojis max 2. No Devanagari:\n" + original}
    ]
    return _call_provider(messages)

# ===== Main entrypoint =====
def reply_as_ron(user_text: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    ts = int(time.time() * 1000)
    
    # Build the dynamic persona blob
    persona_blob = build_persona_blob(history)
    
    # Inject persona blob into the system prompt
    final_system_prompt = SYSTEM_PROMPT.replace("{{PERSONA_BLOB}}", persona_blob)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": final_system_prompt}
    ]
    if history:
        messages.extend(history[-8:])
    
    # We remove FEW_SHOTS here to reduce token count and improve speed
    # messages.extend(FEW_SHOTS) 
    
    messages.append({"role": "user", "content": user_text})

    raw = _call_provider(messages).strip()

    if _needs_rewrite(raw):
        raw = _rewrite_to_hinglish(raw).strip()

    # final light cleanup: collapse whitespace
    reply = re.sub(r"\s+\n", "\n", raw)
    reply = re.sub(r"\n{3,}", "\n\n", reply).strip()

    if len(reply) > 220:
        reply = reply[:220].rsplit(" ", 1)[0] + "…"

    log_jsonl({"ts": ts, "provider": LLM_PROVIDER, "user": user_text, "ron": reply})

    return reply

if __name__ == "__main__":
    try:
        while True:
            txt = input("You: ")
            if not txt.strip():
                continue
            out = reply_as_ron(txt)
            print("Ron:", out)
    except (KeyboardInterrupt, EOFError):
        print()