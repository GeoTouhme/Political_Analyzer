# Political Pattern Analyzer (v0.2) 👾

Strategic intelligence pipeline designed to ingest, analyze, and visualize geopolitical trends between the U.S. and Iran (2026 Conflict Context). This tool helps bridge the gap between raw news signals and actionable political commentary.

## 🚀 Overview
The analyzer processes full-text articles from a Notion database, applies NLP for sentiment analysis and weighted risk scoring, and generates a visual dashboard for pattern recognition.

### Key Features
- **Notion Integration:** Automated ingestion of strategic articles from multiple global sources (NYT, Reuters, CSIS, MEI, etc.).
- **Risk Scoring Engine:** Calculates a 0-100 risk score based on military vs. diplomatic terminology density.
- **Sentiment Analysis:** Uses `TextBlob` to measure discourse polarity (Hawkish vs. Dovish).
- **Visual Dashboard:** Self-contained HTML/Chart.js visualization of the "Strategic Triangle" (Intentions, Capabilities, and Context).

## 🛠 Project Structure
- `analyzer_v2.py`: The core NLP engine (Sentiment + Risk Logic).
- `dashboard.html`: Interactive visualization of current intelligence.
- `POLITICAL_ANALYSIS_PROJECT.md`: Project charter and methodology.
- `analysis_report_v2.json`: Machine-readable output of the latest run.

## 📊 Methodology (The Strategic Triangle)
The system analyzes three core pillars:
1. **Intentions:** Iranian offensive doctrine and internal political rigidity.
2. **Capabilities:** U.S. military force composition and naval buildup metrics.
3. **Context:** Economic deterrents (oil prices) and diplomatic framework status in Geneva.

## 🔒 Security
- `.env` management for Notion and GitHub API keys (secrets are ignored by Git).
- Local-first processing to minimize API overhead.

---
*Created by George & vit (OpenClaw Assistant)*
