"""
PGCM-2026 — GDELT Collector v1.0
==================================
Fetches real-time geopolitical articles from GDELT and converts them
into the same ArticleAnalysis format used by the Notion pipeline.

Designed to run alongside analyzer_v2.py as a parallel data source.

Usage (standalone):
    python gdelt_collector.py

Usage (integrated — call from analyzer_v2.py):
    from gdelt_collector import fetch_gdelt_articles
    gdelt_results = fetch_gdelt_articles(days_back=1)
"""

import logging
import math
import re
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse

import requests
import pandas as pd

try:
    from gdeltdoc import GdeltDoc, Filters
    GDELT_AVAILABLE = True
except ImportError:
    GDELT_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gdelt_collector")


# ---------------------------------------------------------------------------
# Source Credibility Tiers
# Addresses the known issue of partisan sources inflating scores
# ---------------------------------------------------------------------------

SOURCE_TIERS: dict[str, float] = {
    # Tier 1 — High credibility (weight: 1.0)
    "reuters":      1.0,
    "apnews":       1.0,
    "bbc":          1.0,
    "ft":           1.0,
    "economist":    1.0,
    "wsj":          1.0,
    "nytimes":      1.0,

    # Tier 2 — Reliable with editorial lean (weight: 0.85)
    "theguardian":  0.85,
    "aljazeera":    0.85,
    "haaretz":      0.85,
    "axios":        0.85,
    "politico":     0.85,
    "thehill":      0.85,
    "bloomberg":    0.85,
    "foreignpolicy":0.85,
    "foreignaffairs":0.85,
    "defensenews":  0.85,
    "jpost":        0.85,

    # Tier 3 — Specialized / regional (weight: 0.75)
    "middleeasteye":0.75,
    "irna":         0.75,
    "tasnimnews":   0.75,
    "ynetnews":     0.75,
    "timesofisrael":0.75,
    "mei":          0.75,   # Middle East Institute

    # Tier 4 — Hyperpartisan / lower trust (weight: 0.55)
    "oann":         0.55,
    "breitbart":    0.55,
    "presstv":      0.55,   # Iranian state media
    "rt":           0.55,   # Russian state media
    "foxnews":      0.55,
}

DEFAULT_CREDIBILITY = 0.70   # for unknown domains


def get_credibility_weight(domain: str) -> float:
    """Return the credibility weight for a given domain."""
    domain_clean = domain.replace("www.", "").split(".")[0].lower()
    return SOURCE_TIERS.get(domain_clean, DEFAULT_CREDIBILITY)


def get_source_name(url: str) -> str:
    """Extract a clean source name from a URL."""
    if not url:
        return "GDELT"
    try:
        domain = urlparse(url).netloc.replace("www.", "")
        return domain.split(".")[0].upper()
    except Exception:
        return "GDELT"


# ---------------------------------------------------------------------------
# GDELT Search Queries — Crisis-Specific
# ---------------------------------------------------------------------------

GDELT_QUERIES: list[dict] = [
    # Military & kinetic signals
    {
        "keyword": "Iran military strike",
        "label": "MILITARY",
        "domains": [],
    },
    {
        "keyword": "Iran aircraft carrier",
        "label": "MILITARY_NAVAL",
        "domains": [],
    },
    # Nuclear track
    {
        "keyword": "Iran nuclear uranium",
        "label": "NUCLEAR",
        "domains": [],
    },
    # Diplomatic track
    {
        "keyword": "Iran negotiations talks",
        "label": "DIPLOMATIC",
        "domains": [],
    },
    # Proxy & hybrid signals
    {
        "keyword": "Iran Houthi Hezbollah",
        "label": "PROXY",
        "domains": [],
    },
    # Sanctions & economic coercion
    {
        "keyword": "Iran sanctions embargo",
        "label": "COERCIVE",
        "domains": [],
    },
]


# ---------------------------------------------------------------------------
# ArticleAnalysis dataclass (mirrors analyzer_v2.py)
# ---------------------------------------------------------------------------

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
    source_name: str = "GDELT"
    date: str = "2026-01-01"
    credibility_weight: float = 1.0   # NEW — not in Notion pipeline
    raw_score: float = 0.0            # score before credibility adjustment
    data_source: str = "gdelt"        # tag to distinguish from notion articles


# ---------------------------------------------------------------------------
# Logic v6.0 — Scoring Engine (same as analyzer_v2.py)
# ---------------------------------------------------------------------------

NEGATION_WINDOW = 4
NEGATORS: set[str] = {
    "not", "no", "never", "ruled", "rules", "denied", "denies",
    "rejected", "without", "halt", "halted", "cease", "ceasefire",
    "won't", "cannot", "can't", "doesn't", "wouldn't", "shouldn't"
}

REFERENCE_LENGTH = 500

_TITLE_THREAT_PATTERNS: list[str] = [
    r"\bmilitary\s+strike\b", r"\bair\s+strike\b", r"\bnaval\s+strike\b",
    r"\bwar\s+(imminent|warning|declaration|footing)\b",
    r"\b(nuclear|missile|ballistic)\s+(attack|threat|launch|test)\b",
    r"\bfull[- ]?scale\s+(war|conflict|attack|offensive)\b",
    r"\b(preemptive|surgical)\s+(strike|attack|action)\b",
    r"\bkinetic\s+(action|response|option)\b",
    r"\b(evacuation|departure)\s+(order|warning|alert)\b",
    r"\b(maximum|high|full)\s+(alert|readiness)\b",
]

MILITARY_TERMS  = {"critical": ["decapitation","regime change","invasion","existential","pre-emptive","nuclear breakout","massive retaliation","obliterate","total war","first strike","annihilation","nuclear option","mutually assured destruction","carpet bombing","scorched earth","weapons of mass destruction","chemical weapons","biological weapons","casus belli"],"high": ["strike","carrier","armada","missile","buildup","offensive doctrine","punitive","escalation","bombardment","war footing","brinkmanship","arms race","proliferation","troop surge","mobilization","aerial campaign","naval blockade","siege","shelling","incursion","ground offensive","surgical strike","deterrence","compellence","gunboat diplomacy","show of force","escalation dominance"],"medium": ["deployment","assets","patrol","exercise","maneuver","posture","readiness","fortification","troop movement","reinforcement","military expansion","defense buildup","surveillance","reconnaissance","military drill","forward deployment","no-fly zone","buffer zone","rules of engagement"]}
DEFIANCE_TERMS  = {"critical": ["not bound","not be bound","red line","will not tolerate","non-negotiable","freedom of action","right to defend","reserve the right","will not comply","refuse to comply","no longer bound","null and void","not honor","not abide by","renounce","act of war","declaration of war","acts of aggression"],"high": ["reject","refuse","unacceptable","defy","no concessions","miscalculation","consequences","axis of resistance","will not accept","cannot accept","withdraw from","does not recognize","will not recognize","suspend cooperation","ultimatum","final warning","last chance","illegitimate","expel diplomats","recall ambassador","sever relations","persona non grata","provocation","hostile act","belligerent"],"medium": ["sovereign decision","will act alone","independent action","firm resolve","unwavering position","no retreat","irrevocable","formal objection","strongly condemn","categorical denial","unilateral action"]}
GRAY_TERMS      = {"critical": ["will not comply with","refuse to comply with","not bound by","no longer recognize","withdraw from the","exit the agreement","not abide by the","not honor the","declare null and void","reject the framework","reject the resolution","renounce the treaty","terminate the agreement","abrogate the treaty","defy international law","violate the resolution"],"high": ["suspend all cooperation","respond with force","hold responsible","bear the consequences","will not stand idly","forced to respond","on the table","all options","strategic patience has limits","proportional response","severe consequences","decisive action","right to retaliate","will pay a price","point of no return","escalatory measures","coercive measures","impose costs","spiral of distrust"],"medium": ["reassessing our position","reviewing our commitments","deeply concerned","gravely concerned","cannot remain silent","calls into question","undermines","destabilizing","provocative","reckless behavior","irresponsible","dangerous precedent","eroding trust","calculated ambiguity","unilateral measures"]}
COERCIVE_TERMS  = {"critical": ["total embargo","economic warfare","complete blockade","weaponize trade","weaponize energy","financial strangulation","economic strangulation"],"high": ["sanctions","embargo","blockade","asset freeze","trade restriction","arms embargo","economic coercion","economic pressure","punitive measures","punitive sanctions","secondary sanctions","snap-back sanctions","energy cutoff","trade war","financial sanctions","blacklist","export controls"],"medium": ["travel ban","diplomatic isolation","economic leverage","conditionality","compliance mechanism","enforcement measure","restrictive measures","denial of access","supply disruption"]}
HYBRID_TERMS    = {"critical": ["proxy war","hybrid warfare","asymmetric attack","state-sponsored terrorism","cyber warfare","cyber attack","information warfare","weaponization"],"high": ["gray zone operations","gray zone","disinformation campaign","propaganda","election interference","foreign interference","subversion","covert operations","insurgency","non-state actors","paramilitary","militia","sabotage","destabilization","irregular warfare","false flag","plausible deniability","fifth column"],"medium": ["influence operations","narrative warfare","cognitive warfare","strategic communication","lawfare","economic espionage","critical infrastructure","supply chain attack","dual-use technology","regime proxy"]}
DIPLOMATIC_TERMS= {"high": ["treaty","agreement","breakthrough","rapprochement","normalization","ceasefire","peace pact","disarmament","arms control","peace accord","peace process","reconciliation","détente","non-aggression pact","armistice"],"medium": ["talks","negotiation","dialogue","concession","relief","mediation","de-escalation","diplomatic solution","peaceful resolution","confidence-building","good faith","constructive engagement","back-channel","humanitarian corridor","truce","peacekeeping"],"low": ["meeting","statement","visit","consultation","cooperation","summit","envoy","goodwill gesture","communiqué","memorandum of understanding","bilateral","multilateral"]}

MILITARY_WEIGHTS   = {"critical": 22, "high": 12, "medium": 5}
DEFIANCE_WEIGHTS   = {"critical": 18, "high": 10, "medium": 5}
GRAY_WEIGHTS       = {"critical": 15, "high": 10, "medium": 5}
COERCIVE_WEIGHTS   = {"critical": 16, "high":  9, "medium": 4}
HYBRID_WEIGHTS     = {"critical": 16, "high":  9, "medium": 4}
DIPLOMATIC_WEIGHTS = {"high": -5, "medium": -2, "low": -1}


def is_negated(tokens: list[str], term_idx: int) -> bool:
    window_start = max(0, term_idx - NEGATION_WINDOW)
    window = set(tokens[window_start:term_idx])
    return bool(window & NEGATORS)


def get_sentiment(text: str) -> tuple[float, str]:
    if VADER_AVAILABLE:
        return round(SentimentIntensityAnalyzer().polarity_scores(text)["compound"], 3), "vader"
    if TEXTBLOB_AVAILABLE:
        return round(TextBlob(text).sentiment.polarity, 3), "textblob"
    return 0.0, "none"


def calculate_raw_score(text: str, sentiment: float, title: str = "") -> tuple[float, int, int]:
    """Logic v6.0 scoring — returns (raw_score, military_hits, diplomatic_hits)."""
    base_score = max(0, -sentiment) * 20
    term_score: float = 0.0
    text_lower = text.lower()
    tokens = text_lower.split()
    diplomatic_hits = 0
    categories_hit: set[str] = set()
    category_hits: dict[str, int] = {
        "MILITARY": 0, "DEFIANCE": 0, "GRAY": 0, "COERCIVE": 0, "HYBRID": 0
    }

    for cat_name, terms_dict, weights_dict in [
        ("MILITARY",  MILITARY_TERMS,  MILITARY_WEIGHTS),
        ("DEFIANCE",  DEFIANCE_TERMS,  DEFIANCE_WEIGHTS),
        ("GRAY",      GRAY_TERMS,      GRAY_WEIGHTS),
        ("COERCIVE",  COERCIVE_TERMS,  COERCIVE_WEIGHTS),
        ("HYBRID",    HYBRID_TERMS,    HYBRID_WEIGHTS),
    ]:
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

    # Length normalization
    word_count = max(len(tokens), 1)
    term_score *= math.sqrt(REFERENCE_LENGTH / word_count)

    total_risk = base_score + term_score

    # Strategic floor
    if any(t in text_lower for t in [
        "nuclear","enrichment","uranium","warhead","ballistic missile",
        "carrier","carrier strike group","combat-ready","kinetic",
        "preemptive","full alert","high alert","maximum readiness",
        "war scare","all-out war","military action",
    ]):
        total_risk = max(total_risk, 35.0)

    # Title multiplier
    if any(re.search(p, title.lower()) for p in _TITLE_THREAT_PATTERNS):
        total_risk = min(total_risk * 1.25, 85.0)

    military_hits = category_hits["MILITARY"]
    return round(max(0.0, min(100.0, total_risk)), 1), military_hits, diplomatic_hits


def classify_risk(score: float) -> str:
    if score > 75: return "CRITICAL"
    if score > 50: return "HIGH"
    if score > 25: return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Text Fetcher — Optional full-body scraping
# ---------------------------------------------------------------------------

def fetch_full_text(url: str, timeout: int = 10) -> str:
    """
    Attempt to fetch full article text from URL.
    Falls back gracefully — GDELT title/excerpt is used if this fails.
    
    NOTE: Many news sites block scrapers. This works best for
    open-access sources like AP, Reuters wire articles.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; PGCM-Research-Bot/1.0; "
                "+https://geotouhme.github.io/Political_Analyzer/)"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return ""

        # Simple text extraction — strips HTML tags
        text = re.sub(r"<[^>]+>", " ", resp.text)
        text = re.sub(r"\s+", " ", text).strip()

        # Sanity check — must be long enough to be useful
        return text if len(text) > 200 else ""

    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Core Collector
# ---------------------------------------------------------------------------

def fetch_gdelt_articles(
    days_back: int = 1,
    max_articles_per_query: int = 25,
    attempt_full_text: bool = False,
) -> list[ArticleAnalysis]:
    """
    Main entry point — fetch and score GDELT articles for the last N days.

    Args:
        days_back:              How many days of articles to retrieve (1 = today)
        max_articles_per_query: Max articles per GDELT keyword query
        attempt_full_text:      If True, try to scrape full article body
                                (slower, may be blocked by some sites)

    Returns:
        List of ArticleAnalysis objects in the same format as analyzer_v2.py
    """
    if not GDELT_AVAILABLE:
        log.error("gdeltdoc not installed. Run: pip install gdeltdoc")
        return []

    gd = GdeltDoc()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    log.info("GDELT Collector — fetching %d day(s): %s → %s", days_back, start_str, end_str)

    all_rows: list[dict] = []
    seen_urls: set[str] = set()

    for query in GDELT_QUERIES:
        try:
            f = Filters(
                keyword=query["keyword"],
                start_date=start_str,
                end_date=end_str,
                num_records=max_articles_per_query,
                language="English",
            )
            result = gd.article_search(f)

            if result is None or result.empty:
                log.info("  [%s] No results for: %s", query["label"], query["keyword"])
                continue

            count_new = 0
            for _, row in result.iterrows():
                url = str(row.get("url", ""))
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    row_dict = row.to_dict()
                    row_dict["_query_label"] = query["label"]
                    all_rows.append(row_dict)
                    count_new += 1

            log.info("  [%s] %d new articles", query["label"], count_new)
            time.sleep(1.5)   # GDELT rate limit — be polite

        except Exception as e:
            log.warning("  [%s] Query failed: %s", query["label"], e)

    log.info("Total unique articles: %d", len(all_rows))

    if not all_rows:
        log.warning("No GDELT articles retrieved. Check network or GDELT availability.")
        return []

    # ---------------------------------------------------------------------------
    # Score each article
    # ---------------------------------------------------------------------------

    results: list[ArticleAnalysis] = []

    for row in all_rows:
        title = str(row.get("title", "")).strip()
        url   = str(row.get("url", ""))

        if not title:
            continue

        # Build the best text available
        text = title
        for col in ["excerpt", "summary", "body"]:
            candidate = str(row.get(col, "")).strip()
            if len(candidate) > len(text):
                text = candidate

        # Optional: attempt full-text scrape
        if attempt_full_text and url:
            full = fetch_full_text(url)
            if len(full) > len(text):
                text = full
                log.debug("Full text fetched for: %s", title[:60])

        # Date parsing
        date_str = "2026-01-01"
        raw_date = str(row.get("seendate", row.get("date", "")))
        try:
            if len(raw_date) >= 8:
                date_str = datetime.strptime(raw_date[:8], "%Y%m%d").strftime("%Y-%m-%d")
            else:
                date_str = pd.to_datetime(raw_date).strftime("%Y-%m-%d")
        except Exception:
            pass

        # Source credibility
        domain = urlparse(url).netloc.replace("www.", "") if url else ""
        credibility = get_credibility_weight(domain)
        source_name = get_source_name(url)

        # Score
        sentiment, method = get_sentiment(text)
        raw_score, mil_hits, dip_hits = calculate_raw_score(text, sentiment, title=title)

        # Apply credibility weighting to final score
        # Low-credibility sources get a dampened score
        adjusted_score = round(raw_score * credibility, 1)
        risk_level = classify_risk(adjusted_score)

        # Key phrases (best-effort)
        key_phrases: list[str] = []
        if TEXTBLOB_AVAILABLE:
            try:
                phrases = TextBlob(text).noun_phrases[:5]
                key_phrases = [
                    p.replace('"', '').replace('\\', '').strip()
                    for p in phrases if p.strip()
                ]
            except Exception:
                pass

        results.append(ArticleAnalysis(
            title=title,
            sentiment_polarity=sentiment,
            sentiment_method=method,
            risk_score=adjusted_score,
            risk_level=risk_level,
            key_phrases=key_phrases,
            military_hits=mil_hits,
            diplomatic_hits=dip_hits,
            source_name=source_name,
            date=date_str,
            credibility_weight=credibility,
            raw_score=raw_score,
            data_source="gdelt",
        ))

        log.info("[%s][%s] %s — Raw: %.1f | Adjusted: %.1f | Cred: %.2f",
                 risk_level, source_name, title[:55], raw_score, adjusted_score, credibility)

    log.info("GDELT scoring complete — %d articles processed", len(results))
    return results


# ---------------------------------------------------------------------------
# Stats Summary
# ---------------------------------------------------------------------------

def gdelt_summary(results: list[ArticleAnalysis]) -> dict:
    """Generate a summary dict for the GDELT collection cycle."""
    if not results:
        return {}

    scores = [r.risk_score for r in results]
    avg = round(sum(scores) / len(scores), 1)
    peak = round(max(scores), 1)

    level_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in results:
        level_counts[r.risk_level] = level_counts.get(r.risk_level, 0) + 1

    top_sources = {}
    for r in results:
        top_sources[r.source_name] = top_sources.get(r.source_name, 0) + 1

    return {
        "collected_at": datetime.now().isoformat(),
        "article_count": len(results),
        "avg_risk_score": avg,
        "peak_risk_score": peak,
        "risk_distribution": level_counts,
        "top_sources": dict(sorted(top_sources.items(), key=lambda x: -x[1])[:10]),
        "data_source": "gdelt",
    }


# ---------------------------------------------------------------------------
# Merge with Notion results
# ---------------------------------------------------------------------------

def merge_sources(
    notion_results: list,
    gdelt_results: list[ArticleAnalysis],
    gdelt_weight: float = 0.6,
) -> list:
    """
    Merge Notion and GDELT article lists.

    gdelt_weight: Overall dampening applied to GDELT scores before merging.
                  GDELT uses titles/excerpts (not full text) so its scores
                  are naturally lower — this prevents unfair comparison.
                  Default 0.6 = GDELT scores count as 60% of Notion scores.

    De-duplication: Articles with identical titles are deduplicated,
    keeping the Notion version (higher text quality) when both exist.
    """
    merged: list = []

    # 1. Add ALL Notion articles without deduplication (trust Notion dataset)
    for article in notion_results:
        merged.append(article)

    # 2. Add GDELT articles ONLY if they are not already in Notion
    notion_titles = {a.title.lower().strip()[:80] for a in notion_results}
    gdelt_added = 0
    
    for article in gdelt_results:
        title_key = article.title.lower().strip()[:80]
        if title_key not in notion_titles:
            # Apply overall GDELT weight on top of credibility adjustment
            article.risk_score = round(
                min(article.risk_score * gdelt_weight, 100.0), 1
            )
            article.risk_level = classify_risk(article.risk_score)
            merged.append(article)
            gdelt_added += 1

    log.info(
        "Merged: %d Notion + %d GDELT = %d total articles",
        len(notion_results), gdelt_added, len(merged)
    )
    return merged


# ---------------------------------------------------------------------------
# Standalone Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 65)
    print("  PGCM-2026 — GDELT Collector v1.0")
    print(f"  Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65 + "\n")

    if not GDELT_AVAILABLE:
        print("ERROR: gdeltdoc is not installed.")
        print("Run: pip install gdeltdoc")
        return

    # Fetch last 24 hours
    results = fetch_gdelt_articles(
        days_back=1,
        max_articles_per_query=25,
        attempt_full_text=False,    # set True if you want to try scraping
    )

    if not results:
        print("\nNo articles collected. Try increasing days_back or checking GDELT.")
        return

    # Print results table
    print(f"\n{'─' * 65}")
    print(f"  {'RISK':<10} {'SOURCE':<12} {'SCORE':>6}  TITLE")
    print(f"{'─' * 65}")

    for r in sorted(results, key=lambda x: -x.risk_score):
        print(f"  [{r.risk_level:<8}] {r.source_name:<12} {r.risk_score:>5.1f}  {r.title[:45]}")

    # Summary
    summary = gdelt_summary(results)
    print(f"\n{'─' * 65}")
    print(f"  Articles collected : {summary['article_count']}")
    print(f"  Average risk score : {summary['avg_risk_score']}")
    print(f"  Peak risk score    : {summary['peak_risk_score']}")
    print(f"  Risk distribution  : {summary['risk_distribution']}")
    print(f"  Top sources        : {list(summary['top_sources'].keys())[:5]}")

    # Save output
    output = {
        "summary": summary,
        "articles": [asdict(r) for r in results],
    }
    with open("gdelt_collection.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"\n  Saved: gdelt_collection.json")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
