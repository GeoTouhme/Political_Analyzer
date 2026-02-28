"""Political Pattern Analyzer v2.2 — Strategic intelligence pipeline (Logic v6.1).

Changes from v2.1:
  - Daily score now uses Max-Anchored formula instead of simple average.
    Formula: (max_score × 0.40) + (active_avg × 0.35) + (top3_avg × 0.25)
    where active_avg ignores zero-score articles (prevents Diplomatic Dilution).
  - DailyRisk dataclass gains: active_article_count, active_avg_risk,
    top3_avg_risk, simple_avg_risk fields for full transparency.
  - day_over_day_delta now tracks change in anchored score (not simple avg).

Why Max-Anchored?
  When diplomatic coverage floods the dataset (e.g. Geneva talks days),
  many articles score 0 and drag the simple average down even when a handful
  of articles carry extreme threat signals. The anchored formula ensures that
  peak signals are never buried by volume.

Expected SQLite schema (table: articles):
  CREATE TABLE articles (
      notion_id   TEXT PRIMARY KEY,
      title       TEXT NOT NULL,
      date        TEXT NOT NULL,   -- format: YYYY-MM-DD
      full_text   TEXT,
      url         TEXT
  );
"""

import os
import re
import sys
import json
import math
import logging
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Optional

import pandas as pd
from google import genai
from google.genai import types
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

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# Path to local SQLite database (same directory as this script by default)
DB_PATH: str = os.getenv("SQLITE_DB_PATH", os.path.join(os.path.dirname(__file__), "notion_backup.sqlite"))
DB_TABLE: str = "articles"   # change if your table name differs

# --- Strategic Dictionary (Risk Scoring) ------------------------------------

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


@dataclass
class DailyRisk:
    """Aggregated risk metrics for a single calendar day (Logic v6.1).

    Score hierarchy (all stored for transparency):
      anchored_risk  — PRIMARY score, Max-Anchored formula (used everywhere)
      active_avg     — mean of non-zero articles only
      top3_avg       — mean of top-3 highest-risk articles
      simple_avg     — plain mean kept for comparison
      max_risk       — single highest article score of the day
    """
    date: str
    article_count: int
    active_article_count: int         # articles with risk_score > 0
    anchored_risk: float              # PRIMARY: (max*0.40)+(active_avg*0.35)+(top3_avg*0.25)
    simple_avg: float                 # legacy plain mean (comparison only)
    active_avg: float                 # mean of non-zero-score articles
    top3_avg: float                   # mean of top-3 scoring articles
    max_risk: float
    min_risk: float
    risk_level: str                   # classified from anchored_risk
    dominant_articles: list[str]      # titles of top-3 highest-risk articles
    avg_military_hits: float
    avg_diplomatic_hits: float
    day_over_day_delta: float         # Δ final_risk vs previous day
    contradiction_index: float        # CI = sqrt(mil×dip) × contra_ratio per day
    contradiction_boost: float        # points added to anchored (capped at 25)
    final_risk: float                 # PRIMARY output: anchored_risk + contradiction_boost
    contradiction_ratio: float        # share of articles with both mil+dip hits


# --- SQLite Data Source -------------------------------------------------------

def fetch_local_data() -> list[dict]:
    """Load articles from the local SQLite database.

    Returns a list of plain dicts with keys:
        id, title, date, full_text, source_url
    so the rest of the pipeline needs no Notion-specific parsing.
    """
    if not os.path.exists(DB_PATH):
        log.critical("SQLite database not found at: %s", DB_PATH)
        log.critical("Set SQLITE_DB_PATH in your .env or place notion_backup.sqlite next to this script.")
        sys.exit(1)

    try:
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row          # enables column access by name
        cur = con.cursor()

        # Auto-detect available columns so we're robust to minor schema differences
        cur.execute(f"PRAGMA table_info({DB_TABLE})")
        columns = {row["name"] for row in cur.fetchall()}
        log.info("SQLite columns detected: %s", sorted(columns))

        required = {"title", "date", "full_text"}
        missing = required - columns
        if missing:
            log.critical("Table '%s' is missing required columns: %s", DB_TABLE, missing)
            sys.exit(1)

        select_cols = ", ".join(
            col for col in ["id", "title", "date", "full_text", "source_url"]
            if col in columns
        )
        cur.execute(f"SELECT {select_cols} FROM {DB_TABLE} ORDER BY date ASC")
        rows = [dict(row) for row in cur.fetchall()]
        con.close()

        log.info("Loaded %d articles from SQLite (%s).", len(rows), DB_PATH)
        return rows

    except sqlite3.Error as exc:
        log.critical("SQLite error: %s", exc)
        sys.exit(1)


# --- NLP Engine ---------------------------------------------------------------

def get_sentiment(text: str) -> tuple[float, str]:
    """Return (polarity, method) using VADER if available, else TextBlob."""
    if VADER_AVAILABLE:
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return round(scores["compound"], 3), "vader"

    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 3), "textblob"


# --- Negation Detection ------------------------------------------------------

NEGATION_WINDOW = 4
NEGATORS: set[str] = {
    "not", "no", "never", "ruled", "rules", "denied", "denies",
    "rejected", "without", "halt", "halted", "cease", "ceasefire",
    "won't", "cannot", "can't", "doesn't", "wouldn't", "shouldn't"
}


def is_negated(tokens: list[str], term_idx: int) -> bool:
    window_start = max(0, term_idx - NEGATION_WINDOW)
    window = set(tokens[window_start:term_idx])
    return bool(window & NEGATORS)


# --- Length Normalization ----------------------------------------------------

REFERENCE_LENGTH = 500


def normalize_by_length(term_score: float, word_count: int) -> float:
    length_factor = math.sqrt(REFERENCE_LENGTH / max(word_count, 1))
    return term_score * length_factor


# --- Title Multiplier --------------------------------------------------------

_TITLE_THREAT_PATTERNS: list[str] = [
    r"\bmilitary\s+strike\b",
    r"\bair\s+strike\b",
    r"\bnaval\s+strike\b",
    r"\bwar\s+(imminent|warning|declaration|footing)\b",
    r"\b(nuclear|missile|ballistic)\s+(attack|threat|launch|test)\b",
    r"\bfull[- ]?scale\s+(war|conflict|attack|offensive)\b",
    r"\b(preemptive|surgical)\s+(strike|attack|action)\b",
    r"\bkinetic\s+(action|response|option)\b",
    r"\b(evacuation|departure)\s+(order|warning|alert)\b",
    r"\b(maximum|high|full)\s+(alert|readiness)\b",
]


def apply_title_multiplier(total_risk: float, title: str) -> float:
    title_lower = title.lower()
    matched = any(re.search(p, title_lower) for p in _TITLE_THREAT_PATTERNS)
    if matched:
        log.debug("Title-threat multiplier applied for: %s", title[:60])
        return min(total_risk * 1.25, 85.0)
    return total_risk


# --- Temporal Correlation Clusters -------------------------------------------

@dataclass
class TemporalCluster:
    name: str
    condition: Callable[[str, bool, set], bool]
    multiplier: float
    log_message: str


TEMPORAL_CLUSTERS: list[TemporalCluster] = [
    TemporalCluster(
        name="EVAC_AFTER_MILITARY",
        condition=lambda text, recent_mil, cats: (
            ("evacuation" in text or "departure" in text) and recent_mil
        ),
        multiplier=0.40,
        log_message="Security evacuation following military buildup",
    ),
    TemporalCluster(
        name="MULTI_VECTOR_PRESSURE",
        condition=lambda text, recent_mil, cats: (
            len(cats) >= 3 and recent_mil
        ),
        multiplier=0.20,
        log_message="Multi-vector pressure across 3+ categories",
    ),
    TemporalCluster(
        name="SANCTIONS_PLUS_MILITARY",
        condition=lambda text, recent_mil, cats: (
            "COERCIVE" in cats and "MILITARY" in cats and recent_mil
        ),
        multiplier=0.25,
        log_message="Economic coercion + military threat cluster",
    ),
    TemporalCluster(
        name="PROXY_ESCALATION",
        condition=lambda text, recent_mil, cats: (
            "HYBRID" in cats and recent_mil and
            any(actor in text for actor in [
                "hezbollah", "houthi", "kataib", "militia",
                "proxy", "iraqi factions", "islamic resistance",
            ])
        ),
        multiplier=0.30,
        log_message="Proxy actor activation following direct military signaling",
    ),
    TemporalCluster(
        name="DIPLOMATIC_COLLAPSE",
        condition=lambda text, recent_mil, cats: (
            "DEFIANCE" in cats and
            any(w in text for w in [
                "withdraw", "suspend talks", "walk out",
                "no basis for negotiation", "preconditions rejected",
            ])
        ),
        multiplier=0.20,
        log_message="Diplomatic track collapse signal detected",
    ),
]


def apply_temporal_correlation(
    total_risk: float,
    text_lower: str,
    categories_hit: set,
    recent_history: list,
) -> float:
    if not recent_history:
        return total_risk

    recent_mil = any(
        a.risk_score > 40 and a.military_hits > 3
        for a in recent_history[-10:]
    )

    cumulative_multiplier = 1.0
    triggered: list[str] = []

    for cluster in TEMPORAL_CLUSTERS:
        if cluster.condition(text_lower, recent_mil, categories_hit):
            cumulative_multiplier += cluster.multiplier
            triggered.append(cluster.name)
            log.warning("[CLUSTER] %s: %s", cluster.name, cluster.log_message)

    cumulative_multiplier = min(cumulative_multiplier, 2.0)

    if triggered:
        log.info("Active clusters: %s | Total multiplier: %.2fx", triggered, cumulative_multiplier)

    return total_risk * cumulative_multiplier


def calculate_risk_score(text: str, sentiment: float, recent_history: list = None, title: str = "") -> tuple[float, int, int]:
    """Calculate 0-100 risk score. Returns (risk_score, military_hits, diplomatic_hits)."""
    base_score = max(0, -sentiment) * 20

    term_score: float = 0.0
    text_lower = text.lower()
    tokens = text_lower.split()
    diplomatic_hits = 0
    categories_hit: set[str] = set()
    category_hits: dict[str, int] = {
        "MILITARY": 0, "DEFIANCE": 0, "GRAY": 0, "COERCIVE": 0, "HYBRID": 0
    }

    term_categories = [
        ("MILITARY", MILITARY_TERMS, MILITARY_WEIGHTS),
        ("DEFIANCE", DEFIANCE_TERMS, DEFIANCE_WEIGHTS),
        ("GRAY", GRAY_TERMS, GRAY_WEIGHTS),
        ("COERCIVE", COERCIVE_TERMS, COERCIVE_WEIGHTS),
        ("HYBRID", HYBRID_TERMS, HYBRID_WEIGHTS),
    ]

    for cat_name, terms_dict, weights_dict in term_categories:
        for tier, terms in terms_dict.items():
            weight = weights_dict[tier]
            for term in terms:
                if term not in text_lower:
                    continue
                term_tokens = term.split()
                count = 0
                for i, token in enumerate(tokens):
                    if token == term_tokens[0]:
                        phrase_end = i + len(term_tokens)
                        if tokens[i:phrase_end] == term_tokens:
                            if not is_negated(tokens, i):
                                count += 1
                if count > 0:
                    term_score += weight * (1 + math.log(count))
                    category_hits[cat_name] += count
                    categories_hit.add(cat_name)

    for tier, terms in DIPLOMATIC_TERMS.items():
        weight = DIPLOMATIC_WEIGHTS[tier]
        for term in terms:
            count = text_lower.count(term)
            if count > 0:
                term_score += weight * (1 + math.log(count))
                diplomatic_hits += count

    word_count = len(tokens)
    term_score = normalize_by_length(term_score, word_count)
    military_hits = category_hits["MILITARY"]
    total_risk = base_score + term_score

    critical_threats = [
        "nuclear", "enrichment", "uranium", "warhead", "ballistic missile",
        "carrier", "carriers", "carrier strike group", "csg",
        "combat-ready", "combat ready", "strike group", "kinetic",
        "waves of strikes", "operational plans", "preemptive",
        "full alert", "high alert", "maximum readiness", "war scare",
        "all-out war", "military action",
    ]
    if any(threat in text_lower for threat in critical_threats):
        total_risk = max(total_risk, 35.0)

    total_risk = apply_title_multiplier(total_risk, title)
    total_risk = apply_temporal_correlation(total_risk, text_lower, categories_hit, recent_history or [])

    return round(max(0.0, min(100.0, total_risk)), 1), military_hits, diplomatic_hits


def classify_risk(score: float) -> str:
    if score > 75:
        return "CRITICAL"
    if score > 50:
        return "HIGH"
    if score > 25:
        return "MEDIUM"
    return "LOW"


# --- Article Analysis --------------------------------------------------------

def analyze_article(row: dict, recent_results: list = None) -> Optional[ArticleAnalysis]:
    """Analyze a single article row fetched from SQLite."""
    title     = (row.get("title") or "Untitled").strip()
    full_text = (row.get("full_text") or "").strip()
    date      = (row.get("date") or "2026-01-01")[:10]   # keep YYYY-MM-DD only
    source_url = row.get("source_url") or ""

    # Derive a short source label from the URL
    source_name = "Unknown"
    if source_url:
        from urllib.parse import urlparse
        domain = urlparse(source_url).netloc
        source_name = domain.replace("www.", "").split(".")[0].upper()

    if not full_text:
        log.debug("Skipping '%s' — no full text", title)
        return None

    sentiment, method = get_sentiment(full_text)
    risk_score, mil_hits, dip_hits = calculate_risk_score(
        full_text, sentiment, recent_results, title=title
    )
    risk_level = classify_risk(risk_score)

    try:
        blob = TextBlob(full_text)
        key_phrases = []
        for p in blob.noun_phrases[:5]:
            clean_p = p.replace('"', "").replace("\\", "").strip()
            if clean_p:
                key_phrases.append(clean_p)
    except Exception:
        key_phrases = []

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
        date=date,
    )


# --- Daily Breakdown ----------------------------------------------------------

def _contradiction_index(articles: list[ArticleAnalysis]) -> tuple[float, float, float]:
    """Compute Contradiction Index (CI) for a group of articles.

    Detects days where military and diplomatic signals co-exist strongly —
    a pattern that precedes sudden escalations when talks collapse.

    When diplomatic coverage floods the dataset (e.g. Geneva talks), many
    articles contain both military context AND diplomatic language. Their
    co-presence is itself a danger signal: talks are happening *because*
    the military threat is real, not because it has receded.

    Formula:
        CI = sqrt(total_mil_hits × total_dip_hits) × (contra_articles / total)

    Boost applied to anchored score:
        boost = min(CI × 0.15, 25.0)   — capped at 25 points

    Returns:
        (ci_value, boost, contra_ratio)
    """
    mil_total = sum(a.military_hits for a in articles)
    dip_total = sum(a.diplomatic_hits for a in articles)
    n = len(articles)

    # Contradictory articles: those with BOTH military and diplomatic hits
    contra_n = sum(
        1 for a in articles if a.military_hits > 0 and a.diplomatic_hits > 0
    )
    contra_ratio = round(contra_n / n, 3) if n else 0.0

    if mil_total > 0 and dip_total > 0 and contra_n > 0:
        import math as _math
        ci = round(_math.sqrt(mil_total * dip_total) * contra_ratio, 1)
    else:
        ci = 0.0

    boost = round(min(ci * 0.15, 25.0), 1)
    return ci, boost, contra_ratio


def _max_anchored_score(scores: list[float]) -> tuple[float, float, float, float]:
    """Compute Max-Anchored daily risk score to prevent Diplomatic Dilution.

    When a high volume of diplomatic/neutral articles (score=0) floods a day,
    simple averaging buries genuine threat signals from a few high-scoring
    articles. This formula keeps the peak signal dominant.

    Formula:
        anchored = (max_score × 0.40) + (active_avg × 0.35) + (top3_avg × 0.25)

    Returns:
        (anchored_score, active_avg, top3_avg, max_score)
    """
    if not scores:
        return 0.0, 0.0, 0.0, 0.0

    max_score = max(scores)

    # Active avg: exclude zero-score articles (they add no threat signal)
    active = [s for s in scores if s > 0]
    active_avg = round(sum(active) / len(active), 1) if active else 0.0

    # Top-3 avg: mean of the three highest-scoring articles
    top3 = sorted(scores, reverse=True)[:3]
    top3_avg = round(sum(top3) / len(top3), 1)

    anchored = round(
        (max_score * 0.40) + (active_avg * 0.35) + (top3_avg * 0.25), 1
    )
    return min(anchored, 100.0), active_avg, top3_avg, round(max_score, 1)


def build_daily_breakdown(results: list[ArticleAnalysis]) -> list[DailyRisk]:
    """Group articles by date and compute per-day risk metrics (Logic v6.1).

    Uses Max-Anchored scoring to prevent high article volume from diluting
    genuine threat signals on days with mixed diplomatic/military coverage.

    Returns a list of DailyRisk objects sorted chronologically.
    day_over_day_delta tracks change in anchored_risk vs previous day.
    """
    if not results:
        return []

    df = pd.DataFrame([asdict(r) for r in results])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    daily: list[DailyRisk] = []
    prev_anchored: Optional[float] = None  # tracks final_risk of previous day

    for date_str, group in sorted(df.groupby("date")):
        scores = group["risk_score"].tolist()

        # --- Max-Anchored formula (v6.1) ---
        anchored, active_avg, top3_avg, max_risk = _max_anchored_score(scores)
        simple_avg  = round(group["risk_score"].mean(), 1)
        min_risk    = round(group["risk_score"].min(), 1)
        active_n    = int((group["risk_score"] > 0).sum())

        # Contradiction Index — detects dangerous mil+dip co-presence
        day_articles = [r for r in results if r.date == date_str]
        ci_val, ci_boost, contra_ratio = _contradiction_index(day_articles)
        final_risk  = round(min(anchored + ci_boost, 100.0), 1)

        risk_level  = classify_risk(final_risk)   # level from final score

        # Top-3 article titles (highest individual scores)
        top3_titles = group.nlargest(3, "risk_score")["title"].tolist()

        avg_mil = round(group["military_hits"].mean(), 2)
        avg_dip = round(group["diplomatic_hits"].mean(), 2)
        delta   = round(final_risk - prev_anchored, 1) if prev_anchored is not None else 0.0

        daily.append(DailyRisk(
            date=date_str,
            article_count=len(group),
            active_article_count=active_n,
            anchored_risk=anchored,
            simple_avg=simple_avg,
            active_avg=active_avg,
            top3_avg=top3_avg,
            max_risk=max_risk,
            min_risk=min_risk,
            risk_level=risk_level,
            dominant_articles=top3_titles,
            avg_military_hits=avg_mil,
            avg_diplomatic_hits=avg_dip,
            day_over_day_delta=delta,
            contradiction_index=ci_val,
            contradiction_boost=ci_boost,
            final_risk=final_risk,
            contradiction_ratio=contra_ratio,
        ))
        prev_anchored = final_risk

    return daily


# --- Gemini Strategic Outlook ------------------------------------------------

def get_strategic_outlook(
    results: list,
    daily_breakdown: list[DailyRisk],
    trend: str = "STABLE",
    current_risk: float = 0.0,
    prev_risk: float = 0.0,
) -> str:
    if not results:
        return "No data available for strategic assessment."

    if not GEMINI_API_KEY:
        log.warning("GEMINI_API_KEY not set — skipping AI outlook.")
        return "Strategic outlook unavailable: GEMINI_API_KEY not configured in .env."

    log.info("Generating Strategic Outlook via Gemini API...")

    threat_articles  = sorted(results, key=lambda x: x.risk_score, reverse=True)[:8]
    diplomatic_articles = sorted(results, key=lambda x: x.diplomatic_hits, reverse=True)[:4]

    def fmt(a: ArticleAnalysis) -> str:
        return (
            f"  [{a.risk_level}] \"{a.title}\" | Source: {a.source_name} | Date: {a.date} "
            f"| Risk: {a.risk_score} | MilHits: {a.military_hits} | DipHits: {a.diplomatic_hits}"
        )

    # Summarise the daily trend for the model
    daily_summary = "\n".join(
        f"  {d.date}: anchored={d.anchored_risk} simple={d.simple_avg} ({d.risk_level}) Δ{d.day_over_day_delta:+.1f} | n={d.article_count} active={d.active_article_count}"
        for d in daily_breakdown[-7:]   # last 7 days
    )

    trend_desc = {
        "UP":     f"ESCALATING (latest day avg {current_risk} vs previous {prev_risk})",
        "DOWN":   f"DE-ESCALATING (latest day avg {current_risk} vs previous {prev_risk})",
        "STABLE": f"STABLE (latest day avg {current_risk})",
    }.get(trend, "UNKNOWN")

    prompt = f"""You are a senior intelligence analyst specializing in the 2026 US-Iran Persian Gulf crisis.

SYSTEM CONTEXT:
- Overall trend: {trend_desc}
- Dataset: {len(results)} articles across {len(daily_breakdown)} days

DAILY RISK EVOLUTION (last 7 days):
{daily_summary}

THREAT SIGNALS (highest-risk articles):
{chr(10).join(fmt(a) for a in threat_articles)}

DIPLOMATIC SIGNALS:
{chr(10).join(fmt(a) for a in diplomatic_articles)}

TASK: Write a 3-sentence strategic assessment for a high-level policy brief.
Sentence 1: Identify the PRIMARY DRIVER of current risk (cite specific event/actor).
Sentence 2: Assess diplomatic off-ramps — viable or collapsing?
Sentence 3: Most likely trajectory over the next 48-72 hours.

RULES: Direct, specific, data-grounded. No preamble, no markdown. Plain text only."""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=200),
        )
        return response.text.strip()
    except Exception as exc:
        log.error("Gemini API error: %s", exc)
        return "Strategic assessment unavailable due to Gemini API error."


# --- Report Generation -------------------------------------------------------

def generate_report(results: list[ArticleAnalysis]) -> dict:
    """Build the full analysis report with per-day breakdown (v2.1)."""
    if not results:
        return {"meta": {}, "trend": {}, "daily_breakdown": [], "articles": []}

    # --- Daily breakdown (new) ---
    daily_breakdown = build_daily_breakdown(results)

    # --- Trend: compare latest day vs day before it ---
    if len(daily_breakdown) >= 2:
        current_risk = daily_breakdown[-1].final_risk
        prev_risk    = daily_breakdown[-2].final_risk
    elif daily_breakdown:
        current_risk = daily_breakdown[-1].final_risk
        prev_risk    = current_risk  # single day dataset
    else:
        current_risk = prev_risk = 0.0

    trend = "STABLE"
    if current_risk > prev_risk + 1:
        trend = "UP"
    elif current_risk < prev_risk - 1:
        trend = "DOWN"

    df = pd.DataFrame([asdict(r) for r in results])
    global_avg = round(df["risk_score"].mean(), 1)
    max_risk   = round(df["risk_score"].max(), 1)

    strategic_outlook = get_strategic_outlook(
        results, daily_breakdown, trend=trend,
        current_risk=current_risk, prev_risk=prev_risk
    )

    return {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "article_count": len(results),
            "day_count": len(daily_breakdown),
            "global_avg_risk": float(global_avg),
            "max_risk_score": float(max_risk),
            "strategic_outlook": strategic_outlook,
            "model_id": "google/gemini-3-pro-preview",
        },
        "trend": {
            "status": trend,
            "latest_day_final": float(current_risk),
            "previous_day_simple": float(prev_risk),
            "global_avg": float(global_avg),
        },
        # *** NEW: one entry per day, sorted chronologically ***
        "daily_breakdown": [asdict(d) for d in daily_breakdown],
        "articles": [asdict(r) for r in results],
    }


# --- Dashboard Update --------------------------------------------------------

DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "dashboard.html")
DATA_MARKER_START = "// __REPORT_DATA_START__"
DATA_MARKER_END   = "// __REPORT_DATA_END__"


def update_dashboard(report: dict) -> None:
    if not os.path.exists(DASHBOARD_PATH):
        log.warning("dashboard.html not found — skipping dashboard update")
        return

    with open(DASHBOARD_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    data_block = (
        f"{DATA_MARKER_START}\n"
        f"        const EMBEDDED_REPORT = {json.dumps(report, ensure_ascii=False)};\n"
        f"        {DATA_MARKER_END}"
    )

    if DATA_MARKER_START in html:
        pattern = re.escape(DATA_MARKER_START) + r".*?" + re.escape(DATA_MARKER_END)
        html = re.sub(pattern, data_block, html, flags=re.DOTALL)
    else:
        html = html.replace("// --- Boot ---", f"{data_block}\n\n        // --- Boot ---")

    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    log.info("📊 Dashboard updated: %s", DASHBOARD_PATH)


def push_to_github() -> None:
    import shutil, subprocess
    log.info("🚀 Syncing results to GitHub/Vercel...")
    try:
        # Check current directory
        cwd = os.getcwd()
        if not cwd.endswith("political_analyzer"):
             os.chdir(os.path.join(cwd, "political_analyzer"))

        # 1. Update Legacy Dashboard (GitHub Pages)
        shutil.copy("dashboard.html", "index.html")
        
        # 2. Update React Dashboard (Vercel)
        # Copy the latest report to the React public folder
        react_public_report = os.path.join("dashboard_v3", "public", "analysis_report_v2.json")
        shutil.copy("analysis_report_v2.json", react_public_report)

        # 3. Git Operations
        # Add all relevant files for both dashboards
        subprocess.run(["git", "add", "index.html", "analysis_report_v2.json", "dashboard.html", "dashboard_v3/"], check=True)
        commit_msg = f"auto-sync: v3 analysis update {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        log.info("✅ Successfully pushed to GitHub. Vercel deployment starting...")
    except Exception as exc:
        log.error("❌ Failed to push to GitHub: %s", exc)


def update_strategic_log(report: dict) -> None:
    log_path = "STRATEGIC_LOG.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M %p")
    meta  = report.get("meta", {})
    trend = report.get("trend", {})

    # Append a one-liner per day to the log
    daily_lines = "\n".join(
        f"  - {d['date']}: anchored={d['anchored_risk']} simple={d['simple_avg']} ({d['risk_level']}) Δ{d['day_over_day_delta']:+.1f} active={d['active_article_count']}/{d['article_count']}"
        for d in report.get("daily_breakdown", [])
    )

    log_entry = (
        f"\n## [{timestamp}] Analysis Cycle\n"
        f"- **Trend:** {trend.get('status', 'STABLE')} "
        f"(latest day: {trend.get('latest_day_avg', 0)})\n"
        f"- **Article Count:** {meta.get('article_count', 0)} "
        f"across {meta.get('day_count', 0)} days\n"
        f"- **Daily Risk:**\n{daily_lines}\n"
        f"- **AI Outlook:** {meta.get('strategic_outlook', 'N/A')}\n"
        f"---\n"
    )
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)
        log.info("📜 Strategic Log updated.")
    except Exception as exc:
        log.error("Failed to update Strategic Log: %s", exc)


# --- Main Entry --------------------------------------------------------------

def main() -> None:
    log.info("--- Political Pattern Analyzer v2.1 (Logic v6.0 | SQLite source) ---")

    if VADER_AVAILABLE:
        log.info("NLP Engine: VADER + TextBlob")
    else:
        log.warning("VADER not installed — falling back to TextBlob only.")

    # *** CHANGED: read from local SQLite instead of Notion API ***
    rows = fetch_local_data()
    log.info("Loaded %d rows from %s.", len(rows), DB_PATH)

    results: list[ArticleAnalysis] = []
    for row in rows:
        analysis = analyze_article(row, results)
        if analysis:
            results.append(analysis)
            log.info(
                "[%s] %s — Risk: %.1f | Sentiment: %.2f (%s)",
                analysis.risk_level, analysis.title,
                analysis.risk_score, analysis.sentiment_polarity,
                analysis.sentiment_method,
            )

    if not results:
        log.warning("No articles were analyzed. Check the SQLite database content.")
        return

    report = generate_report(results)

    # Print daily summary to console
    log.info("─" * 60)
    log.info("DAILY RISK BREAKDOWN:")
    for d in report["daily_breakdown"]:
        log.info(
            "  %s | final=%-5.1f anchored=%-5.1f CI_boost=%-4.1f %-10s | Δ%+.1f | n=%d",
            d["date"], d["final_risk"], d["anchored_risk"], d["contradiction_boost"],
            f"({d['risk_level']})", d["day_over_day_delta"], d["article_count"],
        )
    log.info("─" * 60)

    output_path = "analysis_report_v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    log.info(
        "✅ Report: %s (%d articles | %d days | latest avg risk: %.1f)",
        output_path,
        report["meta"]["article_count"],
        report["meta"]["day_count"],
        report["trend"]["latest_day_final"],
    )

    update_dashboard(report)
    push_to_github()
    update_strategic_log(report)


if __name__ == "__main__":
    main()

