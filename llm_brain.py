# llm_brain.py
from dotenv import load_dotenv
load_dotenv()
import os
if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"].strip()
import re, json, time, random
import requests
from typing import List, Dict, Optional
from openai import OpenAI
from persona_facts import PERSONA_DETAILS, ARCHETYPE_SUMMARIES

from collections import deque, defaultdict

# per-user session in-memory (works on TG/WA single process)
SESSION = defaultdict(lambda: {
    "history": deque(maxlen=10),     # [("user", txt), ("assistant", txt), ...]
    "flags": {},                     # e.g. {"sleep": expiry_ts}
    "self_ref_cooldown": 0           # stops repeating Champ/family
})

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

USER-FIRST RULES
- User’s feelings first. If they sound low/frustrated/anxious, validate in plain Hinglish BEFORE any suggestion.
- Self-disclosure minimal: don't bring up Champ/family/stand-up unless asked or clearly relevant.
- No generic pep-talks; be concrete and situation-aware.

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


def load_persona_lines(max_n: int = 10) -> list[str]:
    try:
        import persona_facts as pf  # your file
        if hasattr(pf, "get_persona_lines"):
            return pf.get_persona_lines(max_n)
        if hasattr(pf, "PERSONA_LINES"):
            return list(pf.PERSONA_LINES)[:max_n]
        if hasattr(pf, "PERSONA_BLOB"):
            return [ln.strip() for ln in pf.PERSONA_BLOB.splitlines() if ln.strip()][:max_n]
    except Exception:
        pass
    return []

def detect_mood_intent(text: str) -> dict:
    t = text.lower()
    mood = "neutral"; intent = "chat"; severity = "low"

    # quick keywords (expand anytime)
    if any(w in t for w in ["panic", "anxious", "anxiety", "scared", "overthink", "overthinking"]):
        mood, intent, severity = "anxious", "support", "med"
    if any(w in t for w in ["tired", "haven't slept", "no sleep", "insomnia", "sleep deprived"]):
        mood, intent, severity = "exhausted", "support", "med"
    if any(w in t for w in ["breakup", "heartbreak", "hurt", "cry", "depressed", "hopeless"]):
        mood, intent, severity = "sad", "vent", "high"
    if any(w in t for w in ["angry", "frustrated", "pissed", "fed up", "annoyed"]):
        mood, intent, severity = "frustrated", "vent", "med"
    if any(w in t for w in ["help", "how do i", "how to", "plan", "steps", "guide"]):
        intent = "help"
    if any(w in t for w in ["gym", "workout", "study plan", "schedule", "routine"]):
        intent = "help"
    if any(w in t for w in ["lol", "lmao", "haha", "timepass", "bored"]):
        mood, intent = "playful", "banter"
    if "?" in t and intent == "chat":
        intent = "ask"

    return {"mood": mood, "intent": intent, "severity": severity}

def _set_flag(sess, key: str, minutes: int = 120):
    sess["flags"][key] = time.time() + minutes * 60

def _has_flag(sess, key: str) -> bool:
    return sess["flags"].get(key, 0) > time.time()

def update_flags_from_text(sess, text: str):
    t = text.lower()
    if any(k in t for k in ["not slept", "havent slept", "haven't slept", "no sleep", "insomnia", "sleep deprived"]):
        _set_flag(sess, "sleep", 240)
    if any(k in t for k in ["anxiety", "overthink", "overthinking", "panic"]):
        _set_flag(sess, "anxious", 60)

def build_policy(mood_info: dict, sess) -> dict:
    mood, intent, sev = mood_info["mood"], mood_info["intent"], mood_info["severity"]
    flags = {k: _has_flag(sess, k) for k in list(sess["flags"].keys())}

    policy = {
        "length": "short",      # short | medium | bullets
        "validate": False,
        "tiny_step": False,
        "avoid_self": True,     # keep self-disclosure minimal
        "banter": False
    }

    if intent in ("help",) or "plan" in intent:
        policy["length"] = "bullets"

    if mood in ("sad", "anxious", "exhausted", "frustrated"):
        policy["validate"] = True
        policy["tiny_step"] = True
        policy["length"] = "medium" if mood != "exhausted" else "short"

    if flags.get("sleep"):
        policy["tiny_step"] = True
        policy["length"] = "short"

    if intent in ("banter",):
        policy["banter"] = True
        policy["length"] = "short"

    return policy

def policy_as_text(policy: dict) -> str:
    parts = []
    if policy["validate"]: parts.append("Start by validating their feeling in 1 short line.")
    if policy["tiny_step"]: parts.append("Offer exactly one tiny next step, concrete and doable.")
    if policy["length"] == "bullets": parts.append("Respond as 3–5 tight bullets (max 8 words each).")
    if policy["length"] == "short": parts.append("Keep to 1–2 short sentences.")
    if policy["length"] == "medium": parts.append("Keep to 2–3 short sentences, no lecture.")
    if policy["avoid_self"]: parts.append("Do NOT bring up your cat/family or personal stories unless asked.")
    if policy["banter"]: parts.append("Keep playful, light banter; no heavy advice.")
    return " ".join(parts)

def reply_as_ron(user_text: str, user_id: str = "tg") -> str:
    ts = int(time.time() * 1000)
    
    sess = SESSION[user_id]
    update_flags_from_text(sess, user_text)
    mood_info = detect_mood_intent(user_text)
    policy = build_policy(mood_info, sess)
    plan = policy_as_text(policy)

    # rotate a few persona lines (keeps context but avoids overshare)
    p_lines = load_persona_lines(8)
    random.shuffle(p_lines)
    persona_blob = " | ".join(p_lines[:5])

    # build system message
    sys = (
        SYSTEM_PROMPT
        + " | PersonaFacts: "
        + persona_blob
        + " | ResponsePlan: "
        + plan
        + " | ALWAYS Hinglish Roman only."
    )

    # construct message history (last ~10 turns)
    history_msgs = []
    for role, txt in list(sess["history"])[-8:]:
        history_msgs.append({"role": role, "content": txt})

    messages = [{"role": "system", "content": sys}] + history_msgs + [{"role": "user", "content": user_text}]
    
    # --- call your existing provider (ollama/openai) ---
    raw = _call_provider(messages).strip()
    reply = raw

    # Overshare guard: cool down personal mentions
    if sess["self_ref_cooldown"] > 0 and re.search(r"\b(champ|dadi|tara|bandra|stand[- ]?up)\b", reply, flags=re.I):
        # drop the sentence with self-ref
        reply = re.sub(r"[^.?!]*\b(champ|dadi|tara|bandra|stand[- ]?up)\b[^.?!]*[.?!]", "", reply, flags=re.I).strip()
    if re.search(r"\b(champ|dadi|tara|bandra|stand[- ]?up)\b", reply, flags=re.I):
        sess["self_ref_cooldown"] = 5
    else:
        sess["self_ref_cooldown"] = max(0, sess["self_ref_cooldown"] - 1)

    # Hard length clamps to match policy
    if policy["length"] == "short" and len(reply) > 180:
        reply = reply[:180].rsplit(" ", 1)[0] + "…"
    if policy["length"] == "medium" and len(reply) > 280:
        reply = reply[:280].rsplit(" ", 1)[0] + "…"

    # final light cleanup: collapse whitespace
    reply = re.sub(r"\s+\n", "\n", reply)
    reply = re.sub(r"\n{3,}", "\n\n", reply).strip()

    # update memory
    sess["history"].append(("user", user_text))
    sess["history"].append(("assistant", reply))
    
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