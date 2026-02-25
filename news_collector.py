"""
news_collector.py — VIT Automated Collection Pipeline
======================================================
Fetches US–Iran conflict articles from two external APIs:
  • NewsdataHub   (X_API_KEY)
  • TheNewsAPI    (NEWS_API_KEY)

Saves each relevant, de-duplicated article into the Notion database
used by analyzer_v2.py, following the VIT_COLLECTION_PROTOCOL format.

Usage:
  python news_collector.py               # Live run — writes to Notion
  python news_collector.py --dry-run     # Print only, no Notion writes
  python news_collector.py --dry-run --limit 5   # Limit articles per query
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from datetime import date, datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

NOTION_TOKEN: str       = os.getenv("NOTION_API_KEY", "")
X_API_KEY: str          = os.getenv("X_API_KEY", "")
NEWS_API_KEY: str       = os.getenv("NEWS_API_KEY", "")
NOTION_DB_ID: str       = "a9327cf7-8083-433e-aa7e-bca30160ffb6"

# Fixed start date as requested: collect from 2026-01-01 forward
COLLECT_FROM_DATE: str  = "2026-01-01"
TODAY: str              = date.today().isoformat()          # e.g. "2026-02-25"

MAX_RETRIES: int        = 3
RETRY_BACKOFF: float    = 2.0
REQUEST_TIMEOUT: int    = 30
INTER_CALL_SLEEP: float = 1.2   # seconds between outbound API calls

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("collector.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("collector")

# ---------------------------------------------------------------------------
# Query Bank (VIT Collection Scope)
# ---------------------------------------------------------------------------
# Each string is a standalone query sent to every API.
# Grouped by theme for readability.

QUERIES: list[str] = [
    # ── Core conflict
    "US Iran conflict",
    "Iran nuclear deal 2026",
    "Iran IRGC military",
    "Iran sanctions 2026",
    "Iran missile test",
    "Iran nuclear enrichment",
    "Iran proxy attack",
    # ── US military posture
    "US military buildup Middle East",
    "US Navy Middle East deployment",
    "US carrier strike group Gulf",
    "US Navy carrier Iran",
    "Pentagon Iran",
    "CENTCOM Iran",
    "US Air Force refueling aircraft Middle East",
    "US B-52 Middle East",
    # ── Trump / US administration
    "Trump Iran statement",
    "Trump Iran threat",
    "US administration Iran policy",
    "Trump Iran nuclear deal",
    # ── European / international
    "EU Europe Iran statement",
    "Europe Iran sanctions",
    "UN Secretary General Iran",
    "UN Iran conflict",
    # ── Proxy actors
    "Hezbollah military",
    "Houthi Red Sea attack",
    "Iraqi militia Iran",
    "Houthi missile ship",
    # ── Regional incidents
    "Iran airstrikes Syria Iraq",
    "Gulf security threat",
    "Israel Iran military",
    "Israel Iran statement",
    "IDF Iran",
    "Israel Iran nuclear",
    # ── Travel / Embassy / Evacuation
    "travel ban Middle East warning",
    "embassy evacuation Lebanon Iraq",
    "travel advisory Iran Iraq Lebanon",
    # ── Arms / hardware
    "Iran arms Russia China",
    "Iran weapons transfer",
    "Iran ballistic missile",
    # ── Analytical / intelligence signals
    "Pentagon pizza index war signal",   # humint signal
    "de-escalation Iran restraint",
    "Iran diplomacy talks 2026",
    "US Iran emergency meeting",
]

# ---------------------------------------------------------------------------
# Relevance filter — article must mention at least one of these
# ---------------------------------------------------------------------------

RELEVANCE_KEYWORDS: list[str] = [
    "iran", "irgc", "hezbollah", "houthi", "houthis",
    "centcom", "persian gulf", "red sea", "khamenei",
    "rouhani", "tehran", "strait of hormuz", "hormuz",
    "us navy", "carrier", "trump iran", "pentagon iran",
    "gulf state", "nuclear", "enrichment", "proxy",
    "iraq militia", "kataib", "lebanese", "yemen",
]

# ---------------------------------------------------------------------------
# Event-type inference (matches VIT_COLLECTION_PROTOCOL categories)
# ---------------------------------------------------------------------------

_EVENT_TYPE_MAP: list[tuple[list[str], str]] = [
    (["strike", "airstrike", "bomb", "attack", "killed", "casualties", "rockets fired",
       "shelling", "explosion"], "Strike"),
    (["missile test", "missile launch", "drill", "exercise", "military maneuver",
       "war games"], "Drill"),
    (["sanctions", "embargo", "asset freeze", "blacklist", "export control",
       "economic pressure", "financial"], "Sanctions"),
    (["travel ban", "travel advisory", "security alert", "evacuation", "departure warning",
       "heightened alert", "embassy closed"], "Security Alert"),
    (["navy", "carrier", "armada", "deployment", "troops", "redeployment", "buildup",
       "refueling", "b-52", "aircraft", "warship", "destroyer", "frigate"], "Military"),
    (["talks", "negotiation", "diplomacy", "agreement", "ceasefire", "deal", "meeting",
       "un secretary", "de-escalation", "restraint", "dialogue"], "Diplomatic"),
    (["condemn", "statement", "rhetoric", "threat", "warned", "declared", "announced",
       "trump said", "minister said"], "Political Statement"),
]

DEFAULT_EVENT_TYPE = "General/Analytical"


def infer_event_type(text: str) -> str:
    text_lower = text.lower()
    for keywords, event_type in _EVENT_TYPE_MAP:
        if any(kw in text_lower for kw in keywords):
            return event_type
    return DEFAULT_EVENT_TYPE


# ---------------------------------------------------------------------------
# Region inference
# ---------------------------------------------------------------------------

_REGION_MAP: list[tuple[list[str], str]] = [
    (["lebanon", "beirut", "hezbollah"], "Lebanon"),
    (["iraq", "baghdad", "kataib", "hashd"], "Iraq"),
    (["syria", "damascus"], "Syria"),
    (["israel", "tel aviv", "idf", "netanyahu"], "Israel"),
    (["iran", "tehran", "irgc", "khamenei", "isfahan"], "Iran"),
    (["saudi", "riyadh", "uae", "dubai", "abu dhabi", "gulf", "bahrain", "qatar", "kuwait"], "Gulf States"),
    (["red sea", "aden", "bab el-mandeb", "hodeidah", "houthi", "yemen"], "Red Sea / Yemen"),
    (["hormuz", "persian gulf"], "Persian Gulf"),
    (["pentagon", "white house", "washington", "centcom"], "United States"),
    (["europe", "eu ", "european", "brussels", "nato"], "Europe"),
    (["un ", "united nations", "new york", "secretary-general"], "United Nations"),
]

DEFAULT_REGION = "Middle East"


def infer_region(text: str) -> str:
    text_lower = text.lower()
    for keywords, region in _REGION_MAP:
        if any(kw in text_lower for kw in keywords):
            return region
    return DEFAULT_REGION


# ---------------------------------------------------------------------------
# Primary actors inference
# ---------------------------------------------------------------------------

_ACTOR_PATTERNS: list[tuple[list[str], str]] = [
    (["pentagon", "centcom", "u.s. military", "us military", "us navy", "us air force",
      "trump", "biden", "state department", "white house"], "United States"),
    (["iran", "irgc", "khamenei", "iranian", "tehran", "rouhani", "zarif"], "Iran"),
    (["hezbollah", "nasrallah"], "Hezbollah"),
    (["houthi", "ansar allah", "yemen rebel"], "Houthis"),
    (["iraqi militia", "kataib", "hashd", "pmu", "popular mobilization"], "Iraqi Militias"),
    (["israel", "idf", "netanyahu", "mossad", "gantz"], "Israel"),
    (["saudi", "riyadh", "mbs"], "Saudi Arabia"),
    (["russia", "kremlin", "putin"], "Russia"),
    (["china", "beijing", "xi jinping"], "China"),
    (["un ", "united nations", "guterres", "secretary-general"], "United Nations"),
    (["eu ", "europe", "nato", "macron", "scholz", "ursula"], "Europe/NATO"),
]


def infer_actors(text: str) -> str:
    text_lower = text.lower()
    found = [actor for keywords, actor in _ACTOR_PATTERNS
             if any(kw in text_lower for kw in keywords)]
    return ", ".join(dict.fromkeys(found)) if found else "Unknown"  # deduplicated, ordered


# ---------------------------------------------------------------------------
# Notion helpers
# ---------------------------------------------------------------------------

def _notion_headers() -> dict:
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }


def fetch_existing_titles() -> set[str]:
    """Return a lowercased set of article titles already in the Notion DB."""
    url = f"https://api.notion.com/v1/databases/{NOTION_DB_ID}/query"
    existing: set[str] = set()
    payload: dict = {"page_size": 100}

    while True:
        try:
            resp = requests.post(url, headers=_notion_headers(), json=payload,
                                 timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                log.warning("Could not query Notion for deduplication: %s", resp.text[:200])
                break
            data = resp.json()
            for page in data.get("results", []):
                props = page.get("properties", {})
                title_list = props.get("Title", {}).get("title", [])
                if title_list:
                    existing.add(title_list[0].get("plain_text", "").lower().strip())
            if not data.get("has_more"):
                break
            payload["start_cursor"] = data["next_cursor"]
        except Exception as exc:
            log.error("Error fetching existing titles: %s", exc)
            break

    log.info("Loaded %d existing titles from Notion for deduplication.", len(existing))
    return existing


def build_full_text_block(article: dict) -> str:
    """Construct the VIT-protocol formatted text block for Notion 'Full text' property."""
    title   = article.get("title", "N/A")
    source  = article.get("source_name", article.get("source", "N/A"))
    date_   = article.get("date", "N/A")
    region  = article.get("region", "N/A")
    actors  = article.get("primary_actors", "N/A")
    content = article.get("content", article.get("description", "N/A")) or "N/A"
    event   = article.get("event_type", "N/A")
    short   = article.get("short_description", content[:400] if content != "N/A" else "N/A")
    url     = article.get("url", "")

    block = (
        f"TITLE: {title}\n"
        f"SOURCE: {source}\n"
        f"DATE: {date_}\n"
        f"REGION: {region}\n"
        f"PRIMARY_ACTORS: {actors}\n"
        f"URL: {url}\n"
        f"EVENT_TYPE: {event}\n"
        f"SHORT_DESCRIPTION: {short}\n\n"
        f"FULL ARTICLE:\n{content}"
    )
    # Notion rich_text max is 2000 chars per chunk; truncate safely
    return block[:1999]


def save_to_notion(article: dict, dry_run: bool) -> bool:
    """Create a new page in the Notion database. Returns True on success."""
    title   = article.get("title", "Untitled")
    date_   = article.get("date", TODAY)
    url     = article.get("url", "")
    full_text = build_full_text_block(article)

    # Validate date format — Notion requires ISO 8601 (YYYY-MM-DD)
    try:
        datetime.strptime(date_, "%Y-%m-%d")
    except ValueError:
        date_ = TODAY

    if dry_run:
        log.info("[DRY-RUN] Would save: %s | %s | %s", date_, article.get("source_name", "?"), title[:80])
        log.info("          Event: %s | Region: %s | Actors: %s",
                 article.get("event_type"), article.get("region"), article.get("primary_actors"))
        return True

    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "Title": {
                "title": [{"type": "text", "text": {"content": title[:2000]}}]
            },
            "Date": {
                "date": {"start": date_}
            },
            "Sources": {
                "url": url if url else None
            },
            "Full text": {
                "rich_text": [{"type": "text", "text": {"content": full_text}}]
            },
        }
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                "https://api.notion.com/v1/pages",
                headers=_notion_headers(),
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code in (200, 201):
                log.info("✅ Saved → %s | %s", date_, title[:80])
                return True
            elif resp.status_code == 409:
                log.warning("Duplicate detected by Notion API — skipping: %s", title[:60])
                return False
            else:
                log.warning("Notion write failed (attempt %d/%d): %d — %s",
                            attempt, MAX_RETRIES, resp.status_code, resp.text[:200])
        except requests.RequestException as exc:
            log.warning("Request error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF ** attempt
            log.info("Retrying in %.1fs...", wait)
            time.sleep(wait)

    log.error("❌ Failed to save after %d attempts: %s", MAX_RETRIES, title[:80])
    return False


# ---------------------------------------------------------------------------
# API: NewsdataHub  — https://api.newsdatahub.com/v1/news
# ---------------------------------------------------------------------------

def fetch_newsdatahub(query: str, limit: int) -> list[dict]:
    """
    Query NewsdataHub API.
    Docs:  https://www.newsdatahub.com/documentation
    Auth:  X-Api-Key header
    Dates: pub_after / pub_before   (YYYY-MM-DD)
    """
    if not X_API_KEY:
        log.warning("X_API_KEY not set — skipping NewsdataHub.")
        return []

    base_url = "https://api.newsdatahub.com/v1/news"
    headers  = {"X-Api-Key": X_API_KEY}
    articles: list[dict] = []
    params = {
        "q":          query,
        "language":   "en",
        "pub_after":  COLLECT_FROM_DATE,
        "pub_before": TODAY,
        "per_page":   min(limit, 10),
    }

    cursor = None
    pages_fetched = 0
    max_pages = max(1, limit // 10)

    while pages_fetched < max_pages:
        if cursor:
            params["next_page"] = cursor

        try:
            resp = _get_with_retry(base_url, headers=headers, params=params)
            if resp is None:
                break

            data = resp.json()
            items = data.get("data", [])
            if not items:
                break

            for item in items:
                pub = item.get("pub_date", item.get("published_at", TODAY))
                # Normalize date to YYYY-MM-DD
                pub_date = _parse_date(pub)
                content = (item.get("content") or item.get("description") or "").strip()
                articles.append({
                    "title":           item.get("title", "").strip(),
                    "url":             item.get("url", item.get("link", "")),
                    "date":            pub_date,
                    "source_name":     item.get("source_id", item.get("source_name", query)),
                    "description":     item.get("description", ""),
                    "content":         content,
                })

            cursor = data.get("next_page")
            pages_fetched += 1
            if not cursor:
                break

        except Exception as exc:
            log.error("NewsdataHub parse error for query '%s': %s", query, exc)
            break

        time.sleep(INTER_CALL_SLEEP)

    return articles


# ---------------------------------------------------------------------------
# API: TheNewsAPI  — https://api.thenewsapi.com/v1/news/all
# ---------------------------------------------------------------------------

def fetch_thenewsapi(query: str, limit: int) -> list[dict]:
    """
    Query TheNewsAPI.
    Docs:  https://www.thenewsapi.com/documentation
    Auth:  api_token query param
    Dates: published_after / published_before  (ISO 8601)
    Free tier: 3 articles per request, 100 req/day.
    """
    if not NEWS_API_KEY:
        log.warning("NEWS_API_KEY not set — skipping TheNewsAPI.")
        return []

    base_url = "https://api.thenewsapi.com/v1/news/all"
    articles: list[dict] = []
    page = 1
    # Free plan: max 3 per call; cap pages to stay within daily quota
    per_page = 3
    max_pages = min(3, max(1, limit // per_page))

    while page <= max_pages:
        params = {
            "api_token":       NEWS_API_KEY,
            "search":          query,
            "language":        "en",
            "published_after": f"{COLLECT_FROM_DATE}T00:00:00",
            "limit":           per_page,
            "page":            page,
        }

        try:
            resp = _get_with_retry(base_url, params=params)
            if resp is None:
                break

            data = resp.json()
            items = data.get("data", [])
            if not items:
                break

            for item in items:
                pub = item.get("published_at", TODAY)
                pub_date = _parse_date(pub)
                content = (item.get("description") or item.get("snippet") or "").strip()
                articles.append({
                    "title":       item.get("title", "").strip(),
                    "url":         item.get("url", ""),
                    "date":        pub_date,
                    "source_name": item.get("source", {}).get("name", "") if isinstance(item.get("source"), dict) else str(item.get("source", query)),
                    "description": content,
                    "content":     content,
                })

        except Exception as exc:
            log.error("TheNewsAPI parse error for query '%s': %s", query, exc)
            break

        page += 1
        time.sleep(INTER_CALL_SLEEP)

    return articles


# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

def _get_with_retry(url: str, headers: dict = None, params: dict = None) -> Optional[requests.Response]:
    """GET with exponential-backoff retry. Returns response or None."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers or {}, params=params,
                                timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                wait = RETRY_BACKOFF ** attempt * 5   # longer wait on rate-limit
                log.warning("Rate limited (attempt %d/%d). Waiting %.0fs…", attempt, MAX_RETRIES, wait)
                time.sleep(wait)
            elif resp.status_code in (401, 403):
                log.error("Auth error %d for %s — check your API key.", resp.status_code, url)
                return None
            else:
                log.warning("HTTP %d for %s (attempt %d/%d)", resp.status_code, url, attempt, MAX_RETRIES)
        except requests.Timeout:
            log.warning("Timeout (attempt %d/%d) for %s", attempt, MAX_RETRIES, url)
        except requests.ConnectionError:
            log.warning("Connection error (attempt %d/%d) for %s", attempt, MAX_RETRIES, url)
        except requests.RequestException as exc:
            log.error("Unexpected error: %s", exc)
            return None

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF ** attempt
            time.sleep(wait)

    return None


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    r"(\d{4}-\d{2}-\d{2})",                              # 2026-02-25
    r"(\d{4}-\d{2}-\d{2})T",                             # 2026-02-25T…
    r"(\w+ \d{1,2},? \d{4})",                            # February 25, 2026
]

def _parse_date(raw: str) -> str:
    """Extract YYYY-MM-DD from various date string formats."""
    if not raw:
        return TODAY
    # ISO datetime — just take the date part
    if "T" in raw:
        return raw.split("T")[0]
    # Already YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw
    # Try known formats
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    # Fallback: extract first YYYY-MM-DD-like pattern
    m = re.search(r"\d{4}-\d{2}-\d{2}", raw)
    return m.group(0) if m else TODAY


# ---------------------------------------------------------------------------
# Relevance check
# ---------------------------------------------------------------------------

def is_relevant(article: dict) -> bool:
    """Return True if the article passes the VIT relevance gate."""
    combined = (
        (article.get("title") or "") + " " +
        (article.get("description") or "") + " " +
        (article.get("content") or "")
    ).lower()

    if not combined.strip():
        return False

    # Must mention at least one relevance keyword
    return any(kw in combined for kw in RELEVANCE_KEYWORDS)


# ---------------------------------------------------------------------------
# Enrichment (add inferred fields)
# ---------------------------------------------------------------------------

def enrich(article: dict) -> dict:
    combined = (
        (article.get("title") or "") + " " +
        (article.get("description") or "") + " " +
        (article.get("content") or "")
    )
    article["region"]         = infer_region(combined)
    article["primary_actors"] = infer_actors(combined)
    article["event_type"]     = infer_event_type(combined)
    article["short_description"] = (
        (article.get("description") or article.get("content") or "")[:400]
    )
    return article


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def run_collection(dry_run: bool, limit_per_query: int) -> None:
    log.info("=" * 60)
    log.info("VIT News Collector — %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
    log.info("Date range: %s → %s", COLLECT_FROM_DATE, TODAY)
    log.info("Mode: %s", "DRY-RUN" if dry_run else "LIVE (writing to Notion)")
    log.info("Queries: %d | Limit per query: %d", len(QUERIES), limit_per_query)
    log.info("=" * 60)

    # Validate API keys
    missing = []
    if not NOTION_TOKEN:  missing.append("NOTION_API_KEY")
    if not X_API_KEY:     missing.append("X_API_KEY")
    if not NEWS_API_KEY:  missing.append("NEWS_API_KEY")
    if missing:
        log.warning("Missing keys in .env: %s — some sources will be skipped.", ", ".join(missing))
    if not NOTION_TOKEN and not dry_run:
        log.critical("NOTION_API_KEY required for live run. Exiting.")
        sys.exit(1)

    # Pre-load existing titles to avoid duplicates
    existing_titles: set[str] = set()
    if not dry_run and NOTION_TOKEN:
        existing_titles = fetch_existing_titles()

    # Session-level dedup (titles seen in this run)
    session_titles: set[str] = set()

    stats = {"fetched": 0, "relevant": 0, "saved": 0, "skipped_dup": 0, "skipped_irrelevant": 0}

    for query in QUERIES:
        log.info("─── Query: \"%s\" ───", query)

        # Fetch from both APIs
        raw_articles: list[dict] = []

        if X_API_KEY:
            ndh = fetch_newsdatahub(query, limit=limit_per_query)
            log.info("  NewsdataHub → %d articles", len(ndh))
            raw_articles.extend(ndh)

        if NEWS_API_KEY:
            tna = fetch_thenewsapi(query, limit=limit_per_query)
            log.info("  TheNewsAPI  → %d articles", len(tna))
            raw_articles.extend(tna)

        stats["fetched"] += len(raw_articles)

        for article in raw_articles:
            title = article.get("title", "").strip()
            if not title:
                continue

            title_key = title.lower()

            # Deduplication
            if title_key in existing_titles or title_key in session_titles:
                stats["skipped_dup"] += 1
                log.debug("  SKIP (dup): %s", title[:70])
                continue

            # Relevance gate
            if not is_relevant(article):
                stats["skipped_irrelevant"] += 1
                log.debug("  SKIP (irrelevant): %s", title[:70])
                continue

            stats["relevant"] += 1

            # Enrich with inferred metadata
            article = enrich(article)

            # Write to Notion (or dry-run print)
            if save_to_notion(article, dry_run):
                stats["saved"] += 1
                session_titles.add(title_key)   # prevent intra-session dupes
                existing_titles.add(title_key)  # prevent later-query dupes

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("COLLECTION COMPLETE")
    log.info("  Total fetched          : %d", stats["fetched"])
    log.info("  Relevant (passed gate) : %d", stats["relevant"])
    log.info("  Saved to Notion        : %d", stats["saved"])
    log.info("  Skipped (duplicate)    : %d", stats["skipped_dup"])
    log.info("  Skipped (irrelevant)   : %d", stats["skipped_irrelevant"])
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VIT News Collector — fetches US–Iran conflict articles into Notion."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print articles without writing to Notion.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max articles to fetch per query per API (default: 10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_collection(dry_run=args.dry_run, limit_per_query=args.limit)
