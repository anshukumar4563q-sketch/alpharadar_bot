"""
================================================================================
AlphaRadar AI - Indian Stock Market Real-Time News Filter & Telegram Alert Bot
(Fixed build: fail-fast startup diagnostics, forced print() logging, real loop)
================================================================================
Run this in a single Google Colab cell. To stop it, interrupt the cell
(Runtime > Interrupt execution) — it's designed to run forever otherwise.

COLAB SETUP (run once, in the cell above this one):
    !pip install -q feedparser requests google-genai
"""

import os
import re
import sys
import time
import hashlib
import sqlite3
import traceback
from datetime import datetime, timezone

try:
    import feedparser
except ImportError:
    os.system("pip install -q feedparser")
    import feedparser

try:
    import requests
except ImportError:
    os.system("pip install -q requests")
    import requests

try:
    from google import genai
except ImportError:
    os.system("pip install -q google-genai")
    from google import genai


def log(msg: str) -> None:
    """Forced, immediately-flushed print so Colab shows live output even
    inside tight loops or right before a crash."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# ============================================================================
# CONFIGURATION — pulled from environment variables (Render dashboard),
# never hardcoded. Each accepts either naming convention so it matches
# whatever you've actually set in Render's Environment tab:
#   TELEGRAM_BOT_TOKEN / BOT_TOKEN
#   TELEGRAM_CHAT_ID   / CHAT_ID
#   GEMINI_API_KEY
# ============================================================================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

DB_PATH = "news_history.db"

# Google retires/blocks specific Gemini model IDs with little notice
# (e.g. gemini-2.5-flash was blocked for new API keys ahead of its official
# retirement). Rather than hardcoding one version, we try a short candidate
# list at startup and lock onto whichever one actually works.
# "gemini-flash-latest" is Google's alias that auto-points at the current
# Flash model, so this list should keep working across future model bumps.
GEMINI_MODEL_CANDIDATES = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-2.5-flash",
]
MODEL_NAME = None  # resolved at startup by test_gemini_connection()

POLL_INTERVAL_SECONDS = 60
REQUEST_TIMEOUT = 20

# Retry settings for transient Gemini errors (503 UNAVAILABLE, 429 rate
# limit, timeouts). These are temporary server-side conditions, not real
# failures, so we retry with exponential backoff before giving up.
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_BASE_DELAY = 2  # seconds — doubles each attempt: 2s, 4s, 8s

RSS_FEEDS = {
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/2146842.cms",
    "Moneycontrol Top News": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "Livemint Markets": "https://www.livemint.com/rss/markets",
}


# ============================================================================
# STARTUP DIAGNOSTICS — catches placeholder tokens & bad keys BEFORE the loop
# so you get a clear message instead of a buried 404
# ============================================================================
def validate_config() -> bool:
    ok = True

    if not BOT_TOKEN or BOT_TOKEN.startswith("YOUR_") or ":" not in BOT_TOKEN:
        log("❌ BOT_TOKEN is empty or invalid. Check that TELEGRAM_BOT_TOKEN (or BOT_TOKEN) is set "
            "in Render's Environment tab and looks like '123456789:AAE...'.")
        ok = False

    if not CHAT_ID or CHAT_ID.startswith("YOUR_"):
        log("❌ CHAT_ID is empty or invalid. Check that TELEGRAM_CHAT_ID (or CHAT_ID) is set "
            "in Render's Environment tab. Use @userinfobot or getUpdates to find your chat id.")
        ok = False

    if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("YOUR_"):
        log("❌ GEMINI_API_KEY is empty or invalid. Check that it's set in Render's Environment tab.")
        ok = False

    return ok


def test_telegram_connection() -> bool:
    """Calls getMe — the cheapest possible Telegram call — so a bad token
    surfaces immediately as a readable message instead of a 404 later."""
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200 and r.json().get("ok"):
            bot_name = r.json()["result"].get("username", "unknown")
            log(f"✅ Telegram bot connected: @{bot_name}")
            return True
        log(f"❌ Telegram connection failed (HTTP {r.status_code}): {r.text[:300]}")
        log("   This is the classic 404 cause: BOT_TOKEN is wrong or was never set.")
        return False
    except requests.exceptions.RequestException as e:
        log(f"❌ Telegram connection error: {e}")
        return False


def send_startup_notification() -> bool:
    """Fires a plain confirmation message to Telegram once, at startup,
    so you know end-to-end delivery works before any real alerts depend on it."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "🚀 AlphaRadar AI Connected Successfully! Monitoring Started...",
        "parse_mode": "Markdown",
    }
    try:
        r = requests.post(url, data=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            log("✅ Startup notification sent to Telegram.")
            return True
        log(f"⚠️ Startup notification failed (HTTP {r.status_code}): {r.text[:300]}")
        return False
    except requests.exceptions.RequestException as e:
        log(f"⚠️ Startup notification error: {e}")
        return False


def test_gemini_connection() -> bool:
    """Tries each candidate model in order and locks onto the first one
    that actually responds. This is what makes the bot resilient to Google
    blocking/retiring a specific model ID without notice — instead of a
    hardcoded model dying with a 404, we just fall through to the next one."""
    global MODEL_NAME
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        log(f"❌ Could not create Gemini client: {e}")
        return False

    for candidate in GEMINI_MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents="Reply with the single word: OK",
            )
            text = (getattr(response, "text", "") or "").strip()
            MODEL_NAME = candidate
            log(f"✅ Gemini connected using model '{candidate}', test reply: '{text}'")
            return True
        except Exception as e:
            log(f"⚠️ Model '{candidate}' unavailable: {e}")
            continue

    log("❌ None of the candidate Gemini models responded.")
    log("   Common causes: invalid GEMINI_API_KEY, outdated google-genai package, or all candidates deprecated.")
    log("   Try: !pip install -q -U google-genai   then check https://ai.google.dev/gemini-api/docs/models for current model IDs.")
    return False


# ============================================================================
# DATABASE LAYER (SQLite deduplication)
# ============================================================================
def init_db() -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sent_news (
                news_id      TEXT PRIMARY KEY,
                title        TEXT,
                source       TEXT,
                status       TEXT,
                processed_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()
        log("✅ Database ready.")
    except sqlite3.DatabaseError as e:
        log(f"⚠️ Database file corrupted ({e}). Recreating it.")
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()


def is_news_processed(news_id: str) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM sent_news WHERE news_id = ?", (news_id,))
        res = cur.fetchone()
        conn.close()
        return res is not None
    except sqlite3.Error as e:
        log(f"DB read error: {e}")
        return False


def mark_news_processed(news_id: str, title: str, source: str, status: str) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO sent_news (news_id, title, source, status, processed_at) VALUES (?, ?, ?, ?, ?)",
            (news_id, title, source, status, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        log(f"DB write error: {e}")


def strip_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    return re.sub(r"\s+", " ", text).strip()


def make_news_id(entry) -> str:
    news_id = entry.get("id") or entry.get("link")
    if not news_id:
        news_id = hashlib.md5(entry.get("title", "").encode("utf-8")).hexdigest()
    return news_id


# ============================================================================
# 1. RSS NEWS INGESTION
# ============================================================================
def fetch_all_feeds() -> list:
    all_entries = []
    for name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            if getattr(feed, "bozo", 0) and not feed.entries:
                log(f"⚠️ Feed '{name}' failed to parse: {getattr(feed, 'bozo_exception', 'unknown XML error')}")
                continue

            count_before = len(all_entries)
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                link = entry.get("link", "").strip()
                if not title or not link:
                    continue
                all_entries.append(
                    {
                        "id": make_news_id(entry),
                        "title": title,
                        "summary": strip_html(entry.get("summary", "") or entry.get("description", "")),
                        "link": link,
                        "source": name,
                    }
                )
            log(f"Fetched {len(all_entries) - count_before} usable entries from '{name}'.")
        except Exception as e:
            log(f"❌ Feed error on '{name}': {e}")
    return all_entries


# ============================================================================
# 2. AI IMPACT ANALYSIS (Google GenAI SDK — Gemini)
# ============================================================================
ANALYSIS_PROMPT_TEMPLATE = """You are a Principal Equity Research Analyst specializing in Indian stock markets (NSE/BSE).

Analyze the following news item strictly for its TRADEABLE market impact.

RULES:
- IGNORE and discard routine market wrap-ups, generic "markets close higher/lower" summaries, repetitive commentary, opinion pieces, listicles, or anything with no clear, specific, actionable impact on a stock, sector, Nifty, or BankNifty.
- ONLY flag news that has genuine HIGH or MEDIUM impact: earnings surprises, M&A, regulatory action, management changes, large orders/contracts, guidance changes, macro/policy shocks, rating actions, block deals, litigation, capacity expansion, etc.
- If the news is low impact or routine noise, respond with EXACTLY the single word: IGNORE
- Do not explain your reasoning. Do not add any text outside the specified format.

If the news IS high or medium impact, respond in EXACTLY this Telegram Markdown format (no extra commentary, no code fences):

🚨 **MARKET IMPACT ALERT** 🚨

📌 **Stock / Entity:** [Exact Company Name / Nifty / BankNifty]
💥 **Impact Level:** [🔴 HIGH IMPACT / 🟠 MEDIUM IMPACT]

📝 **Summary:**
• [First concise, actionable bullet point]
• [Second concise, actionable bullet point]

📊 **Market Bias:** [📈 BULLISH / 📉 BEARISH / ⚖️ NEUTRAL] - [5-word reasoning]

---
NEWS TITLE: {title}
NEWS SUMMARY: {summary}
SOURCE: {source}
---
"""

_genai_client = None


def get_genai_client():
    global _genai_client
    if _genai_client is None:
        _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    return _genai_client


def _is_transient_gemini_error(e: Exception) -> bool:
    """503 UNAVAILABLE, 429 rate limits, and timeouts are temporary —
    worth retrying. 400/401/403/404 mean something is actually wrong
    (bad key, bad model, bad request) and retrying won't help."""
    text = str(e).upper()
    return any(marker in text for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "TIMEOUT", "DEADLINE"))


def analyze_news(entry: dict):
    """Returns:
      - the formatted alert text, if high/medium impact
      - the string "IGNORE" if the model judged it low-impact noise
        (safe to mark processed — we never want to re-analyze it)
      - None if the API call failed even after retries
        (must NOT be marked processed — retry it next cycle)
    """
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        title=entry["title"],
        summary=entry["summary"] or "N/A",
        source=entry["source"],
    )
    client = get_genai_client()
    last_error = None

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            text = (getattr(response, "text", "") or "").strip()

            if not text:
                # Empty response with no exception — treat as a soft failure,
                # worth a retry rather than silently marking it "ignored".
                last_error = "empty response from model"
                raise ValueError(last_error)

            if text.upper() == "IGNORE" or text.upper().startswith("IGNORE"):
                return "IGNORE"

            return text

        except Exception as e:
            last_error = e
            if _is_transient_gemini_error(e) and attempt < GEMINI_MAX_RETRIES:
                delay = GEMINI_RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 2s, 4s, 8s
                log(f"⚠️ Gemini transient error on attempt {attempt}/{GEMINI_MAX_RETRIES} for "
                    f"'{entry['title'][:50]}': {e} — retrying in {delay}s")
                time.sleep(delay)
                continue
            else:
                # Non-transient error, or retries exhausted — stop trying this item.
                break

    log(f"❌ Gemini analysis failed for '{entry['title'][:60]}' after {GEMINI_MAX_RETRIES} attempt(s): {last_error}")
    return None


# ============================================================================
# 3. TELEGRAM ALERT DISPATCH
# ============================================================================
def send_telegram_alert(message: str, link: str) -> bool:
    full_message = f"{message}\n\n🔗 [Read Full Article]({link})"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": full_message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": False,
    }
    try:
        r = requests.post(url, data=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return True
        log(f"❌ Telegram HTTP {r.status_code}: {r.text[:300]}")
        return False
    except requests.exceptions.Timeout:
        log("❌ Telegram request timed out.")
        return False
    except requests.exceptions.RequestException as e:
        log(f"❌ Telegram request error: {e}")
        return False


# ============================================================================
# MAIN CYCLE + CONTINUOUS LOOP
# ============================================================================
def run_cycle() -> None:
    log("--- Starting scan cycle ---")
    entries = fetch_all_feeds()
    log(f"Total entries fetched: {len(entries)}")

    new_alerts = 0
    for entry in entries:
        if is_news_processed(entry["id"]):
            continue

        analysis = analyze_news(entry)

        if analysis is None:
            # Gemini call failed even after retries — do NOT mark as processed,
            # so it gets picked up and retried on the next scan cycle.
            log(f"⏭️  Leaving unprocessed for retry next cycle: {entry['title'][:70]}")
            continue

        if analysis == "IGNORE":
            # Genuinely low-impact — safe to mark so we never re-spend an
            # API call analyzing it again.
            mark_news_processed(entry["id"], entry["title"], entry["source"], "ignored")
            continue

        if send_telegram_alert(analysis, entry["link"]):
            mark_news_processed(entry["id"], entry["title"], entry["source"], "sent")
            new_alerts += 1
            log(f"✅ Alert sent: {entry['title'][:70]}")
        else:
            log(f"⚠️ Delivery failed, will retry next cycle: {entry['title'][:70]}")

        time.sleep(1.5)

    log(f"--- Cycle complete: {new_alerts} new alert(s) sent ---")


def main() -> None:
    log("=" * 60)
    log(" AlphaRadar AI — Indian Market News Filter starting up ")
    log("=" * 60)

    if not validate_config():
        log("🛑 Fix the config values above and re-run the cell. Stopping now.")
        return

    init_db()

    telegram_ok = test_telegram_connection()
    gemini_ok = test_gemini_connection()

    if not (telegram_ok and gemini_ok):
        log("🛑 Startup checks failed — see the errors above. Stopping before entering the loop.")
        return

    send_startup_notification()

    log("✅ All startup checks passed. Entering continuous monitoring loop.")
    log("   (Runtime > Interrupt execution to stop.)")

    while True:
        try:
            run_cycle()
        except Exception:
            log("❌ Unhandled error in monitoring cycle:")
            traceback.print_exc()

        log(f"Sleeping {POLL_INTERVAL_SECONDS}s until next scan...")
        try:
            time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            log("Stopped by user.")
            break


if __name__ == "__main__":
    main()
