# Project: Political Pattern Analyzer (George & vit)

## 1. Vision
To develop a data-driven political analysis engine that helps George refine his political insights and develop strategic social media posts/analyses. The goal is to move beyond surface-level reading to deep pattern recognition in geopolitical trends.

## 2. Core Strategy
- **Input:** Raw, full-text political articles (avoiding truncated summaries to preserve nuances).
- **Storage:** A structured Notion database (`Political Analyses`) acting as the central data lake.
- **Processing:** A Python-based analysis suite using:
    - **Sentiment Analysis:** To gauge the "hawkish" or "dovish" tone of institutions.
    - **Entity NER (spaCy):** Mapping actors, locations, and military assets.
    - **Pattern Recognition:** Identifying contradictions and shifts in diplomatic/military rhetoric.
- **Output:** Strategic summaries and "Pattern Reports" to guide George's writing.

## 3. Workflow Protocol
1. **Source Discovery:** Use `agent-browser` to bypass protections and fetch full text.
2. **Notion Ingestion:** Push both a human-readable summary and the raw full text to Notion.
3. **Execution:** Run the Python analyzer on the gathered data to extract strategic insights.

## 4. Current Status (Feb 25, 2026)
- **Data Lake:** 11 strategic articles successfully scraped and synced to Notion database `Political Analyses`.
- **Analyzer v0.2 Deployed:** Python script `analyzer_v2.py` implemented with:
    - **Risk Scoring Engine:** Weighted logic using military vs. diplomatic terminology.
    - **Sentiment Analysis:** Integrated `TextBlob` to measure discourse polarity.
    - **JSON Reporting:** Automated export to `analysis_report_v2.json`.
- **Visualization:** Strategic HTML Dashboard developed for visual pattern recognition.
- **Project Infrastructure:** Dedicated project folder created at `~/political_analyzer/` with `.env` management.

## 5. Key Findings from Nucleus Data
- **The Standoff Pattern:** General risk level is stabilized at "MEDIUM" (~32.3/100).
- **The Decoupling:** Military buildup (CSIS) vs. Diplomatic hope (Reuters/AP) shows a sharp strategic divergence.
- **The Economic Brake:** High negative sentiment in NYT/Atlantic indicates that oil price volatility is the primary deterrent for a full-scale US strike.

## 6. Next Steps
1. **Refine NLP Logic:** Move from `TextBlob` to `spaCy` for actor-action mapping (Entity Relationship Extraction).
2. **Post Generation:** Draft George's first political post based on the "Standoff" pattern.
3. **Automation:** Schedule periodic checks of Bing/Google News for new data points.
