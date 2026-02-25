"""Political Pattern Analyzer v1.0 — Strategic intelligence pipeline.

Ingests articles from Notion, applies NLP sentiment analysis and weighted
risk scoring, and generates a JSON report for dashboard consumption.
"""

import os
import sys
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv
from textblob import TextBlob

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# --- Configuration -----------------------------------------------------------

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("analyzer")

NOTION_TOKEN: str = os.getenv("NOTION_API_KEY", "")
DATA_SOURCE_ID: str = "a9327cf7-8083-433e-aa7e-bca30160ffb6"

MAX_RETRIES: int = 3
RETRY_BACKOFF: float = 2.0
REQUEST_TIMEOUT: int = 30

if not NOTION_TOKEN:
    log.critical("NOTION_API_KEY is missing. Add it to your .env file.")
    sys.exit(1)

# --- Strategic Dictionary (Risk Scoring) ------------------------------------

MILITARY_TERMS: dict[str, list[str]] = {
    "critical": [
        "decapitation", "regime change", "invasion", "existential",
        "pre-emptive", "nuclear breakout", "massive retaliation",
    ],
    "high": [
        "strike", "carrier", "armada", "missile", "buildup",
        "offensive doctrine", "punitive", "escalation",
    ],
    "medium": [
        "deployment", "assets", "patrol", "exercise", "maneuver", "posture",
    ],
}

DIPLOMATIC_TERMS: dict[str, list[str]] = {
    "high": ["treaty", "agreement", "breakthrough", "rapprochement", "normalization"],
    "medium": ["talks", "negotiation", "dialogue", "concession", "relief"],
    "low": ["meeting", "statement", "visit", "consultation"],
}

MILITARY_WEIGHTS: dict[str, int] = {"critical": 15, "high": 8, "medium": 3}
DIPLOMATIC_WEIGHTS: dict[str, int] = {"high": -10, "medium": -5, "low": -2}

# --- Data Models -------------------------------------------------------------

@dataclass
class ArticleAnalysis:
    title: str
    sentiment_polarity: float
    sentiment_method: str
    risk_score: float
    risk_level: str
    key_phrases: list[str]
    military_hits: int = 0
    diplomatic_hits: int = 0

@dataclass
class ReportMeta:
    generated_at: str
    article_count: int
    avg_risk_score: float
    max_risk_score: float
    analyzer_version: str = "1.0"


# --- Notion API --------------------------------------------------------------

def fetch_notion_data() -> list[dict]:
    """Fetch articles from Notion database with retry logic."""
    url = f"https://api.notion.com/v1/data_sources/{DATA_SOURCE_ID}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2025-09-03",
        "Content-Type": "application/json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                return response.json().get("results", [])

            log.error("Notion API error (attempt %d/%d): %d",
                      attempt, MAX_RETRIES, response.status_code)

        except requests.Timeout:
            log.warning("Request timed out (attempt %d/%d)", attempt, MAX_RETRIES)
        except requests.ConnectionError:
            log.warning("Connection failed (attempt %d/%d)", attempt, MAX_RETRIES)
        except requests.RequestException as exc:
            log.error("Unexpected request error: %s", exc)
            break

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF ** attempt
            log.info("Retrying in %.1fs...", wait)
            time.sleep(wait)

    log.error("Failed to fetch data after %d attempts.", MAX_RETRIES)
    return []


# --- NLP Engine ---------------------------------------------------------------

def get_sentiment(text: str) -> tuple[float, str]:
    """Return (polarity, method) using VADER if available, else TextBlob."""
    if VADER_AVAILABLE:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return round(scores["compound"], 3), "vader"

    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 3), "textblob"


def calculate_risk_score(text: str, sentiment: float) -> tuple[float, int, int]:
    """Calculate 0-100 risk score with frequency-weighted term matching.

    Returns (risk_score, military_hits, diplomatic_hits).
    """
    base_score = (1 - sentiment) * 20
    term_score = 0
    text_lower = text.lower()
    military_hits = 0
    diplomatic_hits = 0

    for tier, terms in MILITARY_TERMS.items():
        weight = MILITARY_WEIGHTS[tier]
        for term in terms:
            count = text_lower.count(term)
            if count > 0:
                term_score += weight * min(count, 3)  # cap at 3x per term
                military_hits += count

    for tier, terms in DIPLOMATIC_TERMS.items():
        weight = DIPLOMATIC_WEIGHTS[tier]
        for term in terms:
            count = text_lower.count(term)
            if count > 0:
                term_score += weight * min(count, 3)
                diplomatic_hits += count

    total_risk = base_score + term_score
    return round(max(0.0, min(100.0, total_risk)), 1), military_hits, diplomatic_hits


def classify_risk(score: float) -> str:
    """Map numeric risk score to a human-readable level."""
    if score > 75:
        return "CRITICAL"
    if score > 50:
        return "HIGH"
    if score > 25:
        return "MEDIUM"
    return "LOW"


# --- Article Processing -------------------------------------------------------

def extract_text(props: dict, field: str) -> str:
    """Safely extract ALL rich_text blocks from a Notion property."""
    blocks = props.get(field, {}).get("rich_text", [])
    return " ".join(block.get("plain_text", "") for block in blocks).strip()


def extract_title(props: dict) -> str:
    """Safely extract title from Notion properties."""
    title_list = props.get("Title", {}).get("title", [])
    return title_list[0].get("plain_text", "Untitled") if title_list else "Untitled"


def analyze_article(article: dict) -> Optional[ArticleAnalysis]:
    """Analyze a single Notion article for sentiment and risk."""
    props = article.get("properties", {})
    title = extract_title(props)
    full_text = extract_text(props, "Full text")

    if not full_text:
        log.debug("Skipping '%s' — no full text", title)
        return None

    sentiment, method = get_sentiment(full_text)
    risk_score, mil_hits, dip_hits = calculate_risk_score(full_text, sentiment)
    risk_level = classify_risk(risk_score)

    try:
        blob = TextBlob(full_text)
        key_phrases = list(blob.noun_phrases[:5])
    except Exception:
        key_phrases = []
        log.debug("TextBlob corpus missing — skipping noun phrases for '%s'", title)

    return ArticleAnalysis(
        title=title,
        sentiment_polarity=sentiment,
        sentiment_method=method,
        risk_score=risk_score,
        risk_level=risk_level,
        key_phrases=key_phrases,
        military_hits=mil_hits,
        diplomatic_hits=dip_hits,
    )


# --- Report Generation --------------------------------------------------------

def generate_report(results: list[ArticleAnalysis]) -> dict:
    """Build the full analysis report with metadata."""
    risk_scores = [r.risk_score for r in results]
    meta = ReportMeta(
        generated_at=datetime.now().isoformat(),
        article_count=len(results),
        avg_risk_score=round(sum(risk_scores) / len(risk_scores), 1) if risk_scores else 0,
        max_risk_score=max(risk_scores) if risk_scores else 0,
    )
    return {
        "meta": asdict(meta),
        "articles": [asdict(r) for r in results],
    }


# --- Dashboard Update ---------------------------------------------------------

DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")
DATA_MARKER_START = "// __REPORT_DATA_START__"
DATA_MARKER_END = "// __REPORT_DATA_END__"


def update_dashboard(report: dict) -> None:
    """Inject report data directly into dashboard.html for offline use."""
    if not os.path.exists(DASHBOARD_PATH):
        log.warning("dashboard.html not found — skipping dashboard update")
        return

    with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    data_block = f"{DATA_MARKER_START}\n        const EMBEDDED_REPORT = {json.dumps(report, ensure_ascii=False)};\n        {DATA_MARKER_END}"

    if DATA_MARKER_START in html:
        import re
        pattern = re.escape(DATA_MARKER_START) + r".*?" + re.escape(DATA_MARKER_END)
        html = re.sub(pattern, data_block, html, flags=re.DOTALL)
    else:
        html = html.replace("// --- Boot ---", f"{data_block}\n\n        // --- Boot ---")

    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    log.info("📊 Dashboard updated: %s", DASHBOARD_PATH)


# --- Main Entry ---------------------------------------------------------------

def main() -> None:
    log.info("--- Political Pattern Analyzer v1.0 ---")

    if VADER_AVAILABLE:
        log.info("NLP Engine: VADER + TextBlob (noun phrases)")
    else:
        log.warning("VADER not installed — falling back to TextBlob only. "
                     "Install with: pip install vaderSentiment")

    articles = fetch_notion_data()
    log.info("Fetched %d articles from Notion.", len(articles))

    results: list[ArticleAnalysis] = []
    for article in articles:
        analysis = analyze_article(article)
        if analysis:
            results.append(analysis)
            log.info("[%s] %s — Risk: %.1f | Sentiment: %.2f (%s)",
                     analysis.risk_level, analysis.title,
                     analysis.risk_score, analysis.sentiment_polarity,
                     analysis.sentiment_method)

    if not results:
        log.warning("No articles were analyzed. Check your Notion data source.")
        return

    report = generate_report(results)
    output_path = "analysis_report_v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    log.info("✅ Report: %s (%d articles, avg risk: %.1f)",
             output_path, report["meta"]["article_count"],
             report["meta"]["avg_risk_score"])

    update_dashboard(report)


if __name__ == "__main__":
    main()