# tg_bot.py (quick ship fix)
import os, logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from llm_brain import reply_as_ron

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
log = logging.getLogger("ron-tg")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Ron yaha hai. Hinglish mein baat karte hain 😅")

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = str(update.effective_chat.id)
    text = update.message.text or ""
    log.info("MSG from %s: %s", uid, text)
    try:
        # Try new signature first (with user_id). If your llm_brain doesn't have it, fall back.
        try:
            out = reply_as_ron(text, user_id=uid)  # if this raises TypeError, we fall back
        except TypeError:
            out = reply_as_ron(text)
        await update.message.reply_text(out)
    except Exception as e:
        log.exception("Error handling message: %s", e)
        await update.message.reply_text("oops, thoda glitch hua 😅 fir se bhej de?")

def main():
    if not BOT_TOKEN or BOT_TOKEN.startswith("PASTE_"):
        raise SystemExit("ERROR: Set TELEGRAM_BOT_TOKEN to your real BotFather token.")
    log.info("Starting Ron TG bot…")
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    log.info("Polling… send /start to your bot in Telegram.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
