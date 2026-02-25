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
import google.generativeai as genai
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
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
DATA_SOURCE_ID: str = "a9327cf7-8083-433e-aa7e-bca30160ffb6"

MAX_RETRIES: int = 3
RETRY_BACKOFF: float = 2.0
REQUEST_TIMEOUT: int = 30

if not NOTION_TOKEN:
    log.critical("NOTION_API_KEY is missing. Add it to your .env file.")
    sys.exit(1)

# --- Strategic Dictionary (Risk Scoring) ------------------------------------

# v2.0 - Expanded Dictionary including Military, Defiance, Diplomatic, Gray, Coercive, and Hybrid terms.

MILITARY_TERMS: dict[str, list[str]] = {
    "critical": [
        "decapitation", "regime change", "invasion", "existential",
        "pre-emptive", "nuclear breakout", "massive retaliation",
        "obliterate", "total war", "first strike", "annihilation",
        "nuclear option", "mutually assured destruction", "carpet bombing",
        "scorched earth", "ethnic cleansing", "genocide", "weapons of mass destruction",
        "chemical weapons", "biological weapons", "casus belli"
    ],
    "high": [
        "strike", "carrier", "armada", "missile", "buildup",
        "offensive doctrine", "punitive", "escalation", "bombardment",
        "war footing", "brinkmanship", "arms race", "proliferation",
        "troop surge", "mobilization", "aerial campaign", "naval blockade",
        "siege", "shelling", "incursion", "ground offensive", "surgical strike",
        "deterrence", "compellence", "gunboat diplomacy", "show of force",
        "escalation dominance"
    ],
    "medium": [
        "deployment", "assets", "patrol", "exercise", "maneuver", "posture",
        "readiness", "fortification", "troop movement", "reinforcement",
        "military expansion", "defense buildup", "surveillance", "reconnaissance",
        "military drill", "forward deployment", "no-fly zone", "buffer zone",
        "rules of engagement"
    ],
}

DEFIANCE_TERMS: dict[str, list[str]] = {
    "critical": [
        "not bound", "not be bound", "red line", "will not tolerate", "non-negotiable",
        "freedom of action", "right to defend", "reserve the right", "will not comply",
        "refuse to comply", "no longer bound", "no longer obligated", "reject the authority",
        "reject the jurisdiction", "null and void", "not honor", "not abide by",
        "renounce", "act of war", "declaration of war", "acts of aggression"
    ],
    "high": [
        "reject", "refuse", "unacceptable", "defy", "no concessions",
        "regret", "miscalculation", "consequences", "axis of resistance",
        "not a party to", "will not accept", "cannot accept", "withdraw from",
        "pull out of", "does not recognize", "will not recognize", "suspend cooperation",
        "halt cooperation", "suspend participation", "ultimatum", "final warning",
        "last chance", "illegitimate", "illegal occupation", "sovereign right",
        "not respect", "does not comply", "demarche", "expel diplomats",
        "recall ambassador", "sever relations", "downgrade relations",
        "persona non grata", "cross-border incursion", "provocation",
        "hostile act", "belligerent"
    ],
    "medium": [
        "sovereign decision", "will act alone", "independent action", "will not be dictated",
        "on our own terms", "firm resolve", "unwavering position", "no retreat",
        "irrevocable", "protest note", "note verbale", "formal objection",
        "strongly condemn", "categorical denial", "unilateral action"
    ]
}

GRAY_TERMS: dict[str, list[str]] = {
    "critical": [
        "will not comply with", "refuse to comply with", "not bound by", "no longer recognize",
        "withdraw from the", "exit the agreement", "not abide by the", "not honor the",
        "declare null and void", "reject the framework", "reject the resolution",
        "defy the ruling", "renounce the treaty", "revoke participation",
        "terminate the agreement", "abrogate the treaty", "repudiate the accord",
        "void the agreement", "defy international law", "violate the resolution"
    ],
    "high": [
        "suspend all cooperation", "respond with force", "hold responsible",
        "bear the consequences", "will not stand idly", "forced to respond",
        "on the table", "all options", "strategic patience has limits",
        "proportional response", "severe consequences", "decisive action",
        "right to retaliate", "will pay a price", "cross a threshold",
        "point of no return", "escalatory measures", "coercive measures",
        "impose costs", "diplomatic fallout", "spiral of distrust"
    ],
    "medium": [
        "reassessing our position", "reviewing our commitments", "reconsider our participation",
        "deeply concerned", "gravely concerned", "cannot remain silent",
        "calls into question", "undermines", "destabilizing", "provocative",
        "reckless behavior", "irresponsible", "dangerous precedent",
        "eroding trust", "calculated ambiguity", "unilateral measures"
    ]
}

COERCIVE_TERMS: dict[str, list[str]] = {
    "critical": [
        "total embargo", "economic warfare", "complete blockade", "weaponize trade",
        "weaponize energy", "financial strangulation", "economic strangulation"
    ],
    "high": [
        "sanctions", "embargo", "blockade", "asset freeze", "trade restriction",
        "arms embargo", "economic coercion", "economic pressure", "punitive measures",
        "punitive sanctions", "secondary sanctions", "snap-back sanctions",
        "energy cutoff", "trade war", "financial sanctions", "blacklist", "export controls"
    ],
    "medium": [
        "travel ban", "diplomatic isolation", "economic leverage", "conditionality",
        "compliance mechanism", "enforcement measure", "restrictive measures",
        "denial of access", "supply disruption"
    ]
}

HYBRID_TERMS: dict[str, list[str]] = {
    "critical": [
        "proxy war", "hybrid warfare", "asymmetric attack", "state-sponsored terrorism",
        "cyber warfare", "cyber attack", "information warfare", "weaponization"
    ],
    "high": [
        "gray zone operations", "gray zone", "disinformation campaign", "propaganda",
        "election interference", "foreign interference", "subversion", "covert operations",
        "insurgency", "non-state actors", "paramilitary", "militia", "sabotage",
        "destabilization", "irregular warfare", "false flag", "plausible deniability",
        "fifth column"
    ],
    "medium": [
        "influence operations", "narrative warfare", "cognitive warfare",
        "strategic communication", "lawfare", "economic espionage",
        "critical infrastructure", "supply chain attack", "dual-use technology",
        "regime proxy"
    ]
}

DIPLOMATIC_TERMS: dict[str, list[str]] = {
    "high": [
        "treaty", "agreement", "breakthrough", "rapprochement", "normalization",
        "ceasefire", "peace pact", "disarmament", "arms control", "peace accord",
        "peace process", "reconciliation", "détente", "non-aggression pact", "armistice"
    ],
    "medium": [
        "talks", "negotiation", "dialogue", "concession", "relief", "mediation",
        "de-escalation", "diplomatic solution", "peaceful resolution",
        "confidence-building", "good faith", "constructive engagement",
        "back-channel", "humanitarian corridor", "truce", "peacekeeping"
    ],
    "low": [
        "meeting", "statement", "visit", "consultation", "cooperation", "summit",
        "envoy", "goodwill gesture", "communiqué", "memorandum of understanding",
        "bilateral", "multilateral"
    ],
}

MILITARY_WEIGHTS: dict[str, int] = {"critical": 15, "high": 8, "medium": 3}
DEFIANCE_WEIGHTS: dict[str, int] = {"critical": 12, "high": 7, "medium": 4}
GRAY_WEIGHTS: dict[str, int] = {"critical": 10, "high": 7, "medium": 4}
COERCIVE_WEIGHTS: dict[str, int] = {"critical": 12, "high": 7, "medium": 4}
HYBRID_WEIGHTS: dict[str, int] = {"critical": 12, "high": 7, "medium": 4}
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
    source_name: str = "Unknown"
    date: str = "2026-01-01"

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


def calculate_risk_score(text: str, sentiment: float, recent_history: list = None) -> tuple[float, int, int]:
    """Calculate 0-100 risk score with frequency-weighted term matching
    and Temporal Correlation (v3.0).

    Returns (risk_score, military_hits, diplomatic_hits).
    """
    base_score = (1 - sentiment) * 20
    term_score = 0
    text_lower = text.lower()
    military_hits = 0
    diplomatic_hits = 0
    categories_hit = set()

    # Categorize and weigh terms
    term_categories = [
        ("MILITARY", MILITARY_TERMS, MILITARY_WEIGHTS),
        ("DEFIANCE", DEFIANCE_TERMS, DEFIANCE_WEIGHTS),
        ("GRAY", GRAY_TERMS, GRAY_WEIGHTS),
        ("COERCIVE", COERCIVE_TERMS, COERCIVE_WEIGHTS),
        ("HYBRID", HYBRID_TERMS, HYBRID_WEIGHTS)
    ]

    for cat_name, terms_dict, weights_dict in term_categories:
        for tier, terms in terms_dict.items():
            weight = weights_dict[tier]
            for term in terms:
                count = text_lower.count(term)
                if count > 0:
                    term_score += weight * min(count, 3)  # cap at 3x per term
                    military_hits += count
                    categories_hit.add(cat_name)

    for tier, terms in DIPLOMATIC_TERMS.items():
        weight = DIPLOMATIC_WEIGHTS[tier]
        for term in terms:
            count = text_lower.count(term)
            if count > 0:
                term_score += weight * min(count, 3)
                diplomatic_hits += count

    total_risk = base_score + term_score

    # --- TEMPORAL CORRELATION LOGIC ---
    multiplier = 1.0
    if recent_history:
        # Check for military escalation clusters in the last 10 analyzed items
        recent_mil = any(a.risk_score > 40 and a.military_hits > 5 for a in recent_history[-10:])
        
        # Scenario A: Security alert / Evacuation FOLLOWING military moves
        if ("evacuation" in text_lower or "departure" in text_lower) and recent_mil:
            multiplier += 0.40
            log.warning("!!! STRATEGIC CLUSTER: Security evacuation following military build-up detected.")
        
        # Scenario B: Multiple high-risk categories in one burst
        if len(categories_hit) >= 3 and recent_mil:
            multiplier += 0.20
            log.info("Sustained multi-vector pressure detected (+20% risk).")

    total_risk *= multiplier

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

def extract_title(props: dict) -> str:
    """Safely extract title from Notion properties."""
    title_list = props.get("Title", {}).get("title", [])
    return title_list[0].get("plain_text", "Untitled") if title_list else "Untitled"


def extract_date(props: dict) -> str:
    """Safely extract date from Notion properties."""
    date_obj = props.get("Date", {}).get("date")
    return date_obj.get("start", "2026-01-01") if date_obj else "2026-01-01"


def extract_text(props: dict, page_id: str = None) -> str:
    """Extract text from property OR fetch all blocks from the page if property is short."""
    # 1. Try property first
    blocks = props.get("Full text", {}).get("rich_text", [])
    prop_text = " ".join(block.get("plain_text", "") for block in blocks).strip()
    
    # 2. If property is short or missing, and we have a page_id, fetch blocks
    if (len(prop_text) < 100 or "..." in prop_text) and page_id:
        log.info(f"Fetching full page content for: {page_id}")
        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        headers = {
            "Authorization": f"Bearer {NOTION_TOKEN}",
            "Notion-Version": "2025-09-03",
        }
        try:
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                results = r.json().get("results", [])
                page_text = ""
                for block in results:
                    block_type = block.get("type")
                    if block_type == "paragraph":
                        rich_texts = block.get("paragraph", {}).get("rich_text", [])
                        page_text += " ".join(t.get("plain_text", "") for t in rich_texts) + "\n"
                if len(page_text) > len(prop_text):
                    return page_text.strip()
        except Exception as e:
            log.error(f"Failed to fetch page blocks: {e}")
            
    return prop_text

def analyze_article(article: dict, recent_results: list = None) -> Optional[ArticleAnalysis]:
    """Analyze a single Notion article for sentiment and risk."""
    props = article.get("properties", {})
    page_id = article.get("id")
    title = extract_title(props)
    full_text = extract_text(props, page_id)
    date = extract_date(props)
    
    # Extract source name for intellectual property/branding
    source_url = props.get("Sources", {}).get("url", "")
    source_name = "Unknown Source"
    if source_url:
        from urllib.parse import urlparse
        domain = urlparse(source_url).netloc
        source_name = domain.replace("www.", "").split(".")[0].upper()

    if not full_text:
        log.debug("Skipping '%s' — no full text", title)
        return None

    sentiment, method = get_sentiment(full_text)
    risk_score, mil_hits, dip_hits = calculate_risk_score(full_text, sentiment, recent_results)
    risk_level = classify_risk(risk_score)

    try:
        blob = TextBlob(full_text)
        # Clean phrases to prevent JSON injection/syntax errors
        key_phrases = []
        for p in blob.noun_phrases[:5]:
            clean_p = p.replace("\"", "").replace("\\", "").strip()
            if clean_p:
                key_phrases.append(clean_p)
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
        source_name=source_name,
        date=date
    )


def get_strategic_outlook(results: list) -> str:
    """Call the Gemini API to generate a dynamic strategic outlook based on live results."""
    if not results:
        return "No data available for strategic assessment."

    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY is not set — skipping AI outlook generation.")
        return "Strategic outlook unavailable: GEMINI_API_KEY not configured in .env."

    log.info("Generating Strategic Outlook via Gemini API...")

    top_events = sorted(results, key=lambda x: x.risk_score, reverse=True)[:10]
    summary_text = "\n".join(
        [f"- {a.title} (Risk: {a.risk_score}, Sentiment: {a.sentiment_polarity})" for a in top_events]
    )

    prompt = f"""You are a senior geopolitical intelligence analyst specializing in the US-Iran 2026 crisis.
Based on the following recent events and risk scores, provide a concise (3-4 sentences) strategic outlook.
Identify the primary driver of risk and the likely trajectory for the next 48-72 hours.
Tone: Professional, urgent, objective.

RECENT EVENTS:
{summary_text}

OUTPUT: A single paragraph of 3-4 sentences. Do not include any preamble or labels."""

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-3-pro-preview")
        response = model.generate_content(prompt)
        outlook = response.text.strip()
        log.info("Strategic outlook generated successfully.")
        return outlook

    except Exception as e:
        log.error(f"Failed to generate strategic outlook via Gemini API: {e}")
        return "Strategic assessment currently unavailable due to a Gemini API error."


# --- Report Generation --------------------------------------------------------

def generate_report(results: list[ArticleAnalysis]) -> dict:
    """Build the full analysis report with metadata and trend analysis."""
    risk_scores = [r.risk_score for r in results]
    
    # Sort articles by date to find the latest data point
    sorted_articles = sorted(results, key=lambda x: x.date, reverse=True)
    if not sorted_articles:
        return {"meta": {}, "articles": []}
        
    latest_date = sorted_articles[0].date
    
    # 1. Current Risk = Average of the LATEST single day
    latest_day_scores = [r.risk_score for r in results if r.date == latest_date]
    current_risk = round(sum(latest_day_scores) / len(latest_day_scores), 1)
    
    # 2. Previous Context = Average of the 3 days BEFORE the latest day
    latest_dt = datetime.strptime(latest_date, "%Y-%m-%d")
    context_window = [
        r.risk_score for r in results 
        if 1 <= (latest_dt - datetime.strptime(r.date, "%Y-%m-%d")).days <= 3
    ]
    previous_risk = round(sum(context_window) / len(context_window), 1) if context_window else current_risk
    
    trend = "STABLE"
    if current_risk > previous_risk + 1: trend = "UP"
    elif current_risk < previous_risk - 1: trend = "DOWN"

    # New: Strategic Outlook via Gemini 3 Pro
    strategic_outlook = get_strategic_outlook(results)

    report_data = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "article_count": len(results),
            "avg_risk_score": current_risk, # Now matches the latest graph point
            "max_risk_score": max(risk_scores) if risk_scores else 0,
            "strategic_outlook": strategic_outlook,
            "model_id": "google/gemini-3-pro-preview"
        },
        "trend": {
            "status": trend,
            "current": current_risk,
            "previous": previous_risk,
            "global_avg": round(sum(risk_scores) / len(risk_scores), 1) if risk_scores else 0
        },
        "articles": [asdict(r) for r in results],
    }
    return report_data


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


def push_to_github():
    """Sync the updated dashboard and report to GitHub Pages."""
    log.info("🚀 Syncing results to GitHub Pages...")
    try:
        import subprocess
        # Copy dashboard to index.html for GitHub Pages
        subprocess.run(["cp", "dashboard.html", "index.html"], check=True)
        
        # Git operations
        subprocess.run(["git", "add", "index.html", "analysis_report_v2.json", "dashboard.html"], check=True)
        commit_msg = f"auto-sync: analysis update {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        log.info("✅ Successfully pushed to GitHub Pages.")
    except Exception as e:
        log.error(f"❌ Failed to push to GitHub: {e}")

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
        analysis = analyze_article(article, results)
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
    
    # Auto-push to GitHub
    push_to_github()

    # New: Update Strategic Log
    update_strategic_log(report)


def update_strategic_log(report: dict):
    """Append current session stats to STRATEGIC_LOG.md for long-term archiving."""
    log_path = "STRATEGIC_LOG.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M %p")
    meta = report.get("meta", {})
    trend = report.get("trend", {})
    
    log_entry = f"""
## [{timestamp}] Analysis Cycle
- **Risk Score:** {trend.get('current', 0)} ({trend.get('status', 'STABLE')})
- **Article Count:** {meta.get('article_count', 0)}
- **AI Outlook:** {meta.get('strategic_outlook', 'N/A')}
---
"""
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
        log.info("📜 Strategic Log updated.")
    except Exception as e:
        log.error(f"Failed to update Strategic Log: {e}")


if __name__ == "__main__":
    main()