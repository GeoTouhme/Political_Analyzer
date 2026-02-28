"""Political Pattern Analyzer v2.0 — Strategic intelligence pipeline (Logic v6.0).

Ingests articles from Notion, applies NLP sentiment analysis and weighted
risk scoring, and generates a JSON report for dashboard consumption.
"""

import os
import re
import sys
import json
import math
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from textblob import TextBlob
from gdelt_collector import fetch_gdelt_articles, merge_sources

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
DATA_SOURCE_ID: str = "87aeafb55c8a4d0998efb3b186d33403"
NOTION_VERSION: str = "2022-06-28"

MAX_RETRIES: int = 3
RETRY_BACKOFF: float = 2.0
REQUEST_TIMEOUT: int = 30

if not NOTION_TOKEN:
    log.critical("NOTION_API_KEY is missing. Add it to your .env file.")
    sys.exit(1)

# --- Strategic Dictionary (Risk Scoring) ------------------------------------

MILITARY_TERMS: dict[str, list[str]] = {
    "critical": ["decapitation", "regime change", "invasion", "existential", "pre-emptive", "nuclear breakout", "massive retaliation", "obliterate", "total war", "first strike", "annihilation", "nuclear option", "mutually assured destruction", "carpet bombing", "scorched earth", "weapons of mass destruction", "chemical weapons", "biological weapons", "casus belli"],
    "high": ["strike", "carrier", "armada", "missile", "buildup", "offensive doctrine", "punitive", "escalation", "bombardment", "war footing", "brinkmanship", "arms race", "proliferation", "troop surge", "mobilization", "aerial campaign", "naval blockade", "siege", "shelling", "incursion", "ground offensive", "surgical strike", "deterrence", "compellence", "gunboat diplomacy", "show of force", "escalation dominance"],
    "medium": ["deployment", "assets", "patrol", "exercise", "maneuver", "posture", "readiness", "fortification", "troop movement", "reinforcement", "military expansion", "defense buildup", "surveillance", "reconnaissance", "military drill", "forward deployment", "no-fly zone", "buffer zone", "rules of engagement"]
}

DEFIANCE_TERMS: dict[str, list[str]] = {
    "critical": ["not bound", "not be bound", "red line", "will not tolerate", "non-negotiable", "freedom of action", "right to defend", "reserve the right", "will not comply", "refuse to comply", "no longer bound", "null and void", "not honor", "not abide by", "renounce", "act of war", "declaration of war", "acts of aggression"],
    "high": ["reject", "refuse", "unacceptable", "defy", "no concessions", "regret", "miscalculation", "consequences", "axis of resistance", "will not accept", "cannot accept", "withdraw from", "does not recognize", "will not recognize", "suspend cooperation", "ultimatum", "final warning", "last chance", "illegitimate", "expel diplomats", "recall ambassador", "sever relations", "persona non grata", "provocation", "hostile act", "belligerent"],
    "medium": ["sovereign decision", "will act alone", "independent action", "firm resolve", "unwavering position", "no retreat", "irrevocable", "formal objection", "strongly condemn", "categorical denial", "unilateral action"]
}

GRAY_TERMS: dict[str, list[str]] = {
    "critical": ["will not comply with", "refuse to comply with", "not bound by", "no longer recognize", "withdraw from the", "exit the agreement", "not abide by the", "not honor the", "declare null and void", "reject the framework", "reject the resolution", "renounce the treaty", "terminate the agreement", "abrogate the treaty", "defy international law", "violate the resolution"],
    "high": ["suspend all cooperation", "respond with force", "hold responsible", "bear the consequences", "will not stand idly", "forced to respond", "on the table", "all options", "strategic patience has limits", "proportional response", "severe consequences", "decisive action", "right to retaliate", "will pay a price", "point of no return", "escalatory measures", "coercive measures", "impose costs", "spiral of distrust"],
    "medium": ["reassessing our position", "reviewing our commitments", "deeply concerned", "gravely concerned", "cannot remain silent", "calls into question", "undermines", "destabilizing", "provocative", "reckless behavior", "irresponsible", "dangerous precedent", "eroding trust", "calculated ambiguity", "unilateral measures"]
}

COERCIVE_TERMS: dict[str, list[str]] = {
    "critical": ["total embargo", "economic warfare", "complete blockade", "weaponize trade", "weaponize energy", "financial strangulation", "economic strangulation"],
    "high": ["sanctions", "embargo", "blockade", "asset freeze", "trade restriction", "arms embargo", "economic coercion", "economic pressure", "punitive measures", "punitive sanctions", "secondary sanctions", "snap-back sanctions", "energy cutoff", "trade war", "financial sanctions", "blacklist", "export controls"],
    "medium": ["travel ban", "diplomatic isolation", "economic leverage", "conditionality", "compliance mechanism", "enforcement measure", "restrictive measures", "denial of access", "supply disruption"]
}

HYBRID_TERMS: dict[str, list[str]] = {
    "critical": ["proxy war", "hybrid warfare", "asymmetric attack", "state-sponsored terrorism", "cyber warfare", "cyber attack", "information warfare", "weaponization"],
    "high": ["gray zone operations", "gray zone", "disinformation campaign", "propaganda", "election interference", "foreign interference", "subversion", "covert operations", "insurgency", "non-state actors", "paramilitary", "militia", "sabotage", "destabilization", "irregular warfare", "false flag", "plausible deniability", "fifth column"],
    "medium": ["influence operations", "narrative warfare", "cognitive warfare", "strategic communication", "lawfare", "economic espionage", "critical infrastructure", "supply chain attack", "dual-use technology", "regime proxy"]
}

DIPLOMATIC_TERMS: dict[str, list[str]] = {
    "high": ["treaty", "agreement", "breakthrough", "rapprochement", "normalization", "ceasefire", "peace pact", "disarmament", "arms control", "peace accord", "peace process", "reconciliation", "détente", "non-aggression pact", "armistice"],
    "medium": ["talks", "negotiation", "dialogue", "concession", "relief", "mediation", "de-escalation", "diplomatic solution", "peaceful resolution", "confidence-building", "good faith", "constructive engagement", "back-channel", "humanitarian corridor", "truce", "peacekeeping"],
    "low": ["meeting", "statement", "visit", "consultation", "cooperation", "summit", "envoy", "goodwill gesture", "communiqué", "memorandum of understanding", "bilateral", "multilateral"]
}

MILITARY_WEIGHTS: dict[str, int] = {"critical": 22, "high": 12, "medium": 5}
DEFIANCE_WEIGHTS: dict[str, int] = {"critical": 18, "high": 10, "medium": 5}
GRAY_WEIGHTS: dict[str, int] = {"critical": 15, "high": 10, "medium": 5}
COERCIVE_WEIGHTS: dict[str, int] = {"critical": 16, "high": 9, "medium": 4}
HYBRID_WEIGHTS: dict[str, int] = {"critical": 16, "high": 9, "medium": 4}
DIPLOMATIC_WEIGHTS: dict[str, int] = {"high": -5, "medium": -2, "low": -1}

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
    credibility_weight: float = 1.0
    raw_score: float = 0.0
    data_source: str = "notion"

# --- Notion API --------------------------------------------------------------

def fetch_notion_data() -> list[dict]:
    url = f"https://api.notion.com/v1/databases/{DATA_SOURCE_ID}/query"
    headers = {"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"}
    all_results = []
    has_more, next_cursor = True, None
    while has_more:
        payload = {"start_cursor": next_cursor} if next_cursor else {}
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    data = response.json()
                    all_results.extend(data.get("results", []))
                    has_more, next_cursor = data.get("has_more", False), data.get("next_cursor")
                    log.info(f"Fetched {len(all_results)} articles so far...")
                    break
                log.error(f"Notion API error: {response.status_code} — {response.text}")
            except Exception as e:
                log.error(f"Request error: {e}")
            time.sleep(RETRY_BACKOFF ** attempt)
        else: break
    return all_results

# --- NLP Engine ---------------------------------------------------------------

def get_sentiment(text: str) -> tuple[float, str]:
    if VADER_AVAILABLE:
        return round(SentimentIntensityAnalyzer().polarity_scores(text)["compound"], 3), "vader"
    return round(TextBlob(text).sentiment.polarity, 3), "textblob"

NEGATION_WINDOW = 4
NEGATORS: set[str] = {"not", "no", "never", "ruled", "rules", "denied", "denies", "rejected", "without", "halt", "halted", "cease", "ceasefire", "won't", "cannot", "can't", "doesn't", "wouldn't", "shouldn't"}

def is_negated(tokens: list[str], term_idx: int) -> bool:
    window_start = max(0, term_idx - NEGATION_WINDOW)
    return bool(set(tokens[window_start:term_idx]) & NEGATORS)

REFERENCE_LENGTH = 500
_TITLE_THREAT_PATTERNS: list[str] = [r"\bmilitary\s+strike\b", r"\bair\s+strike\b", r"\bnaval\s+strike\b", r"\bwar\s+(imminent|warning|declaration|footing)\b", r"\b(nuclear|missile|ballistic)\s+(attack|threat|launch|test)\b", r"\bfull[- ]?scale\s+(war|conflict|attack|offensive)\b", r"\b(preemptive|surgical)\s+(strike|attack|action)\b", r"\bkinetic\s+(action|response|option)\b", r"\b(evacuation|departure)\s+(order|warning|alert)\b", r"\b(maximum|high|full)\s+(alert|readiness)\b"]

def calculate_risk_score(text: str, sentiment: float, recent_history: list = None, title: str = "") -> tuple[float, int, int]:
    base_score = max(0, -sentiment) * 20
    term_score, text_lower = 0.0, text.lower()
    tokens = text_lower.split()
    category_hits = {"MILITARY": 0, "DEFIANCE": 0, "GRAY": 0, "COERCIVE": 0, "HYBRID": 0}
    diplomatic_hits = 0

    for cat_name, terms_dict, weights_dict in [("MILITARY", MILITARY_TERMS, MILITARY_WEIGHTS), ("DEFIANCE", DEFIANCE_TERMS, DEFIANCE_WEIGHTS), ("GRAY", GRAY_TERMS, GRAY_WEIGHTS), ("COERCIVE", COERCIVE_TERMS, COERCIVE_WEIGHTS), ("HYBRID", HYBRID_TERMS, HYBRID_WEIGHTS)]:
        for tier, terms in terms_dict.items():
            weight = weights_dict[tier]
            for term in terms:
                if term in text_lower:
                    term_tokens = term.split()
                    count = 0
                    for i in range(len(tokens) - len(term_tokens) + 1):
                        if tokens[i:i+len(term_tokens)] == term_tokens:
                            if not is_negated(tokens, i): count += 1
                    if count > 0:
                        term_score += weight * (1 + math.log(count))
                        category_hits[cat_name] += count

    for tier, terms in DIPLOMATIC_TERMS.items():
        weight = DIPLOMATIC_WEIGHTS[tier]
        for term in terms:
            count = text_lower.count(term)
            if count > 0:
                term_score += weight * (1 + math.log(count))
                diplomatic_hits += count

    term_score *= math.sqrt(REFERENCE_LENGTH / max(len(tokens), 1))
    total_risk = base_score + term_score
    if any(t in text_lower for t in ["nuclear", "enrichment", "uranium", "warhead", "ballistic missile", "carrier", "strike group", "kinetic", "preemptive", "full alert"]):
        total_risk = max(total_risk, 35.0)
    if any(re.search(p, title.lower()) for p in _TITLE_THREAT_PATTERNS):
        total_risk = min(total_risk * 1.25, 85.0)

    # Simple cluster logic
    if any(a.risk_score > 40 for a in (recent_history[-10:] if recent_history else [])):
        if ("evacuation" in text_lower or "departure" in text_lower): total_risk *= 1.4
        if len([c for c in category_hits.values() if c > 0]) >= 3: total_risk *= 1.2
        if category_hits["HYBRID"] > 0 and any(a in text_lower for a in ["hezbollah", "houthi", "militia"]): total_risk *= 1.3

    return round(max(0.0, min(100.0, total_risk)), 1), category_hits["MILITARY"], diplomatic_hits

def classify_risk(score: float) -> str:
    if score > 75: return "CRITICAL"
    if score > 50: return "HIGH"
    if score > 25: return "MEDIUM"
    return "LOW"

# --- Main Pipeline ------------------------------------------------------------

def run_analysis():
    log.info("--- Political Pattern Analyzer v2.0 (Logic v6.0) ---")
    notion_raw = fetch_notion_data()
    results: list[ArticleAnalysis] = []
    
    for i, page in enumerate(notion_raw):
        props = page.get("properties", {})
        title_list = props.get("Title", {}).get("title", [])
        title = title_list[0].get("plain_text", "Untitled") if title_list else "Untitled"
        date_str = props.get("Date", {}).get("date", {}).get("start", "2026-01-01") if props.get("Date", {}).get("date") else "2026-01-01"
        source = props.get("Source", {}).get("select", {}).get("name", "Unknown") if props.get("Source", {}).get("select") else "Unknown"
        
        # Build text: Use Content if it's one of the latest 150 articles
        text = title
        if i >= len(notion_raw) - 150:
             # Try to pull content from Notion blocks if possible (simplified here)
             text = title # In a full version, we'd fetch blocks here
             
        sentiment, method = get_sentiment(text)
        score, mil, dip = calculate_risk_score(text, sentiment, recent_history=results, title=title)
        
        results.append(ArticleAnalysis(
            title=title, sentiment_polarity=sentiment, sentiment_method=method,
            risk_score=score, risk_level=classify_risk(score), key_phrases=[],
            military_hits=mil, diplomatic_hits=dip, source_name=source, date=date_str
        ))

    # Calculate Stats
    if results:
        scores = [r.risk_score for r in results]
        overall_avg = round(sum(scores) / len(scores), 1)
        peak_score = round(max(scores), 1)

        # NEW: Calculate Current Risk (Latest 150 articles)
        now = datetime.now()
        recent_scores = [r.risk_score for r in results[-150:]]
        current_risk = round(sum(recent_scores) / len(recent_scores), 1) if recent_scores else overall_avg

        summary = {
            "last_updated": now.strftime("%Y-%m-%d %H:%M"),
            "article_count": len(results),
            "overall_avg": overall_avg,
            "current_risk": current_risk,
            "peak_score": peak_score,
            "risk_distribution": {
                "CRITICAL": len([r for r in results if r.risk_level == "CRITICAL"]),
                "HIGH": len([r for r in results if r.risk_level == "HIGH"]),
                "MEDIUM": len([r for r in results if r.risk_level == "MEDIUM"]),
                "LOW": len([r for r in results if r.risk_level == "LOW"]),
            }
        }
        
        with open("analysis_report_v2.json", "w") as f:
            json.dump({"summary": summary, "articles": [asdict(r) for r in results]}, f, indent=4)
        log.info(f"✅ Report: analysis_report_v2.json (Overall: {overall_avg} | Current: {current_risk})")

if __name__ == "__main__":
    run_analysis()
