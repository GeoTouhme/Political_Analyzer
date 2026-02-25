# Political Pattern Analyzer (v1.0) 👾

Strategic intelligence pipeline designed to ingest, analyze, and visualize geopolitical trends between the U.S. and Iran (2026 Conflict Context). This tool bridges the gap between raw news signals and actionable political commentary.

## 🚀 Overview
The analyzer processes full-text articles from a Notion database, applies dual NLP engines for sentiment analysis and frequency-weighted risk scoring, and generates a visual dashboard for pattern recognition.

### Key Features
- **Notion Integration:** Automated ingestion of strategic articles from multiple global sources (NYT, Reuters, CSIS, MEI, etc.)
- **Dual NLP Sentiment:** VADER (news-optimized) + TextBlob fallback with automatic detection
- **Risk Scoring Engine:** Frequency-weighted 0-100 risk score based on military vs. diplomatic terminology density
- **Visual Dashboard:** Self-contained HTML/Chart.js dashboard with dynamic JSON loading
- **Structured Reports:** JSON output with metadata (timestamp, averages, peak risk)
- **Robust Error Handling:** Retry logic with exponential backoff for API calls

## 🛠 Setup

```bash
pip install -r requirements.txt
python -m textblob.download_corpora  # Optional: improves noun phrase extraction
```

Create a `.env` file:
```
NOTION_API_KEY=your_notion_token_here
```

## 📊 Usage

```bash
python analyzer_v2.py          # Run analysis
python -m pytest test_analyzer.py -v  # Run tests
```

Open `dashboard.html` in your browser to view results.

## 📁 Project Structure
- `analyzer_v2.py` — Core NLP engine (Sentiment + Risk + Report generation)
- `dashboard.html` — Interactive visualization dashboard
- `test_analyzer.py` — Unit test suite (24 tests)
- `analysis_report_v2.json` — Machine-readable output of the latest run
- `POLITICAL_ANALYSIS_PROJECT.md` — Project charter and methodology

## 📊 Methodology (The Strategic Triangle)
1. **Intentions:** Iranian offensive doctrine and internal political rigidity
2. **Capabilities:** U.S. military force composition and naval buildup metrics
3. **Context:** Economic deterrents (oil prices) and diplomatic framework status

## 🔒 Security
- `.env` management for Notion API keys (gitignored)
- Fail-fast validation — missing tokens abort immediately
- No raw API responses leaked in error output

---
*Created by George & vit (OpenClaw Assistant)*
