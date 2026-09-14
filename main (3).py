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
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
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

# Render's Web Service tier requires binding to a port within its scan
# timeout, or the deploy is killed even though the bot itself works fine.
# This doesn't serve real traffic — it exists purely to satisfy that check.
PORT = int(os.getenv("PORT", "10000"))

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
MODEL_NAME = None  # resolved at startup by resolve_gemini_model()

# How often the bot scans RSS feeds. This alone does NOT cap Gemini usage —
# a single busy cycle can still contain many fresh headlines. It's paired
# with GEMINI_DAILY_CALL_BUDGET below, which is the actual quota guardrail.
CHECK_INTERVAL_SECONDS = 3600  # 1 hour. Try 2700 (45 min) for faster coverage.
REQUEST_TIMEOUT = 20

# Retry settings for transient Gemini errors (503 UNAVAILABLE, 429 rate
# limit, timeouts). These are temporary server-side conditions, not real
# failures, so we retry with exponential backoff before giving up.
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_BASE_DELAY = 2  # seconds — doubles each attempt: 2s, 4s, 8s

# ----------------------------------------------------------------------
# QUOTA GUARDRAIL — the actual fix for hitting daily rate limits.
# Every real Gemini call (analysis attempts AND model-probe calls) counts
# against this budget, tracked per calendar day (UTC) in SQLite so it
# survives restarts. Set this comfortably below your real daily quota —
# free tier is commonly ~20/day per model; with up to 4 candidate models
# in the list above, 40 is a conservative shared budget. Tune to match
# whatever ai.google.dev/gemini-api/docs/rate-limits shows for your key.
# ----------------------------------------------------------------------
GEMINI_DAILY_CALL_BUDGET = 40

# How often we're willing to re-probe candidate models when Gemini is down.
# Prevents burning quota by retrying on every restart or every scan cycle —
# each probe call costs against the budget just like a real analysis call.
GEMINI_RESOLUTION_COOLDOWN_SECONDS = 300
_last_model_resolution_attempt = 0.0

RSS_FEEDS = {
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/2146842.cms",
    "Moneycontrol Top News": "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "Livemint Markets": "https://www.livemint.com/rss/markets",
}


# ============================================================================
# HEALTH-CHECK HTTP SERVER (for Render Web Service port binding)
# Serves a trivial 200 OK on 0.0.0.0:PORT so Render's deploy port-scan
# passes. Runs in a background daemon thread — never blocks or interferes
# with the bot's own polling loop, which keeps running on the main thread.
# ============================================================================
class _HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"AlphaRadar AI - Bot is running.")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Suppress http.server's default per-request logging — Render's
        # own port-scan and any uptime monitor would otherwise spam the logs.
        pass


def start_health_check_server() -> None:
    """Starts the health-check server on a daemon thread. Called first,
    before config validation or any network calls, so the port opens as
    fast as possible and isn't delayed by (or dependent on) Telegram/Gemini
    connectivity checks."""
    def _serve():
        try:
            server = HTTPServer(("0.0.0.0", PORT), _HealthCheckHandler)
            log(f"✅ Health-check server listening on 0.0.0.0:{PORT}")
            server.serve_forever()
        except Exception as e:
            log(f"❌ Health-check server failed to start on port {PORT}: {e}")

    thread = threading.Thread(target=_serve, daemon=True, name="health-check-server")
    thread.start()


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


def resolve_gemini_model(force: bool = False) -> bool:
    """Tries each candidate model in order and locks onto the first one
    that actually responds. Resilient to two separate failure modes:
      - a model being deprecated/retired (404) -> just skip to the next one
      - a model's quota being exhausted (RESOURCE_EXHAUSTED) -> also skip to
        the next one, since free-tier quotas are typically tracked per
        (project, model), so a different candidate may still have budget.

    Cooldown-limited: repeated calls within GEMINI_RESOLUTION_COOLDOWN_SECONDS
    are skipped (no API calls made, returns whatever the last known state
    was) so a Gemini outage doesn't turn into a quota-burning probe loop —
    whether from Render restarts or from every single poll cycle retrying.
    Pass force=True to bypass the cooldown (used once at startup).
    """
    global MODEL_NAME, _last_model_resolution_attempt

    if MODEL_NAME is not None:
        return True

    now = time.time()
    if not force and (now - _last_model_resolution_attempt) < GEMINI_RESOLUTION_COOLDOWN_SECONDS:
        return False
    _last_model_resolution_attempt = now

    remaining = gemini_budget_remaining()
    if remaining <= 0:
        log(f"🛑 Daily Gemini call budget ({GEMINI_DAILY_CALL_BUDGET}) already used today. "
            f"Skipping model resolution — resumes automatically after midnight UTC.")
        return False

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        log(f"❌ Could not create Gemini client: {e}")
        return False

    for candidate in GEMINI_MODEL_CANDIDATES:
        if gemini_budget_remaining() <= 0:
            log(f"🛑 Daily Gemini call budget ({GEMINI_DAILY_CALL_BUDGET}) exhausted mid-probe. "
                f"Stopping resolution attempts for today.")
            break
        try:
            record_gemini_call()
            response = client.models.generate_content(
                model=candidate,
                contents="Reply with the single word: OK",
            )
            text = (getattr(response, "text", "") or "").strip()
            MODEL_NAME = candidate
            log(f"✅ Gemini connected using model '{candidate}', test reply: '{text}' "
                f"(budget used today: {gemini_calls_used_today()}/{GEMINI_DAILY_CALL_BUDGET})")
            return True
        except Exception as e:
            log(f"⚠️ Model '{candidate}' unavailable: {e}")
            continue

    log("❌ None of the candidate Gemini models responded right now.")
    log("   Common causes: invalid GEMINI_API_KEY, outdated google-genai package, all candidates "
        "deprecated, or free-tier daily quota exhausted across every candidate.")
    log(f"   Will automatically retry in ~{GEMINI_RESOLUTION_COOLDOWN_SECONDS}s. "
        f"Check https://ai.google.dev/gemini-api/docs/models for current model IDs and "
        f"https://ai.google.dev/gemini-api/docs/rate-limits for your quota.")
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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS gemini_usage (
                usage_date  TEXT PRIMARY KEY,
                calls_used  INTEGER NOT NULL DEFAULT 0
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


def _today_utc_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def gemini_calls_used_today() -> int:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT calls_used FROM gemini_usage WHERE usage_date = ?", (_today_utc_str(),))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else 0
    except sqlite3.Error as e:
        log(f"DB read error (gemini_usage): {e}")
        return 0


def record_gemini_call() -> None:
    """Increments today's Gemini call counter. Call this once for every
    real API request made — probe calls during model resolution included,
    since those cost against the quota exactly like an analysis call does."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        today = _today_utc_str()
        cur.execute(
            """
            INSERT INTO gemini_usage (usage_date, calls_used) VALUES (?, 1)
            ON CONFLICT(usage_date) DO UPDATE SET calls_used = calls_used + 1
            """,
            (today,),
        )
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        log(f"DB write error (gemini_usage): {e}")


def gemini_budget_remaining() -> int:
    return max(0, GEMINI_DAILY_CALL_BUDGET - gemini_calls_used_today())


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
    worth retrying within seconds. 400/401/403/404 mean something is
    actually wrong (bad key, bad model, bad request) and retrying won't help."""
    text = str(e).upper()
    return any(marker in text for marker in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "TIMEOUT", "DEADLINE"))


def _is_daily_quota_exhausted(e: Exception) -> bool:
    """A RESOURCE_EXHAUSTED tied to a *per-day* quota won't recover within
    a short backoff window — retrying 2s/4s/8s later is pointless. Detected
    from Google's quotaId, e.g. 'GenerateRequestsPerDayPerProjectPerModel'."""
    text = str(e).upper().replace("_", "")
    return "RESOURCEEXHAUSTED" in text and "PERDAY" in text


def analyze_news(entry: dict):
    """Returns:
      - the formatted alert text, if high/medium impact
      - the string "IGNORE" if the model judged it low-impact noise
        (safe to mark processed — we never want to re-analyze it)
      - None if the API call failed even after retries, or if no Gemini
        model is currently reachable at all
        (must NOT be marked processed — retry it next cycle)
    """
    global MODEL_NAME

    if not MODEL_NAME:
        # Gemini isn't connected right now (outage/quota exhaustion). Don't
        # call the API at all — just leave this item for a later cycle.
        return None

    if gemini_budget_remaining() <= 0:
        # Our own daily budget (not just Google's) is used up. Skip without
        # calling the API — leave unprocessed for a later cycle/day.
        return None

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        title=entry["title"],
        summary=entry["summary"] or "N/A",
        source=entry["source"],
    )
    client = get_genai_client()
    last_error = None

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        if gemini_budget_remaining() <= 0:
            log(f"🛑 Daily Gemini call budget exhausted mid-retry for '{entry['title'][:50]}'. Stopping.")
            return None

        try:
            record_gemini_call()
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

            if _is_daily_quota_exhausted(e):
                # No point retrying within this cycle or even this hour.
                # Drop the resolved model so the next cycle's
                # resolve_gemini_model() call tries a different candidate
                # (daily quotas are typically tracked per model) instead of
                # hammering the same exhausted one.
                log(f"⚠️ Daily quota exhausted for model '{MODEL_NAME}': {e}")
                log("   Switching to a different candidate model next cycle instead of retrying this one.")
                MODEL_NAME = None
                break

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
def _per_cycle_gemini_cap() -> int:
    """Spreads the daily budget evenly across the day's scan cycles, so one
    busy hour (e.g. market open) can't burn the whole day's quota and leave
    nothing for the rest of the day."""
    cycles_per_day = max(1, 86400 // CHECK_INTERVAL_SECONDS)
    return max(1, GEMINI_DAILY_CALL_BUDGET // cycles_per_day)


def run_cycle() -> None:
    log("--- Starting scan cycle ---")

    if not MODEL_NAME:
        if not resolve_gemini_model():
            log("⏳ Gemini still unavailable this cycle (cooldown, outage, or budget exhausted). "
                "Skipping this scan — Telegram bot stays up and will keep retrying automatically.")
            return

    remaining_today = gemini_budget_remaining()
    if remaining_today <= 0:
        log(f"🛑 Daily Gemini call budget ({GEMINI_DAILY_CALL_BUDGET}) already used today. "
            f"Skipping this scan — resumes automatically after midnight UTC.")
        return

    cycle_cap = min(_per_cycle_gemini_cap(), remaining_today)
    log(f"Gemini budget: {remaining_today}/{GEMINI_DAILY_CALL_BUDGET} left today, "
        f"up to {cycle_cap} analysis call(s) this cycle.")

    entries = fetch_all_feeds()
    log(f"Total entries fetched: {len(entries)}")

    new_alerts = 0
    calls_this_cycle = 0
    for entry in entries:
        if is_news_processed(entry["id"]):
            continue

        if calls_this_cycle >= cycle_cap:
            log(f"⏭️  Per-cycle Gemini cap ({cycle_cap}) reached — remaining fresh items "
                f"will be picked up next cycle.")
            break

        analysis = analyze_news(entry)
        calls_this_cycle += 1

        if analysis is None:
            # Gemini call failed even after retries, or budget ran out mid-cycle —
            # do NOT mark as processed, so it gets retried on a later scan.
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

    log(f"--- Cycle complete: {new_alerts} new alert(s) sent, "
        f"{gemini_calls_used_today()}/{GEMINI_DAILY_CALL_BUDGET} Gemini calls used today ---")


def main() -> None:
    log("=" * 60)
    log(" AlphaRadar AI — Indian Market News Filter starting up ")
    log("=" * 60)

    # Bind the health-check port immediately, before any config validation
    # or network calls — Render's port scan has its own timeout independent
    # of whether Telegram/Gemini are configured correctly, so this must not
    # wait on (or fail because of) those checks.
    start_health_check_server()

    if not validate_config():
        log("🛑 Fix the config values above and re-run the cell. Stopping now.")
        return

    init_db()

    telegram_ok = test_telegram_connection()
    if not telegram_ok:
        log("🛑 Telegram connection failed — nothing works without this. Stopping before entering the loop.")
        return

    # Gemini connectivity is checked but NOT fatal: a rate limit or a
    # temporary outage on Google's side should never take the whole bot
    # down. If it's unavailable now, the loop below starts anyway and
    # retries resolve_gemini_model() each cycle (cooldown-limited so it
    # doesn't hammer an exhausted quota).
    gemini_ok = resolve_gemini_model(force=True)
    if not gemini_ok:
        log("⚠️ Gemini is not reachable right now (see errors above). Starting the loop anyway — "
            "it will keep retrying in the background and resume alerts once a model is available.")

    send_startup_notification()

    log("✅ Startup checks complete. Entering continuous monitoring loop.")
    log("   (Runtime > Interrupt execution to stop.)")

    while True:
        try:
            run_cycle()
        except Exception:
            log("❌ Unhandled error in monitoring cycle:")
            traceback.print_exc()

        log(f"Sleeping {CHECK_INTERVAL_SECONDS}s until next scan...")
        try:
            time.sleep(CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            log("Stopped by user.")
            break


if __name__ == "__main__":
    main()
