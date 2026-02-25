# Persian Gulf Conflict Monitor (PGCM-2026)

An autonomous intelligence monitoring platform designed to track, analyze, and visualize the 2026 US-Iran geopolitical crisis.

## 🚀 Overview
PGCM-2026 is an end-to-end OSINT (Open Source Intelligence) pipeline. It functions as a dual-role agent system: a **Collector (VIT)** that harvests raw signals from the global information environment, and an **Analyzer** that applies strategic weighted scoring and NLP to quantify escalation risks.

---

## 🛠 Project Architecture

### 1. Data Collection (VIT Protocol)
The system follows a strict **Geopolitical Collection Protocol** (`VIT_COLLECTION_PROTOCOL.md`) to ensure high-signal ingestion:
*   **Targeted Scraping:** Monitors official statements (State Dept, IRGC, UN, E3), military movements (CENTCOM, Russian/Chinese deployments), and regional security alerts.
*   **Deep Harvesting:** Unlike standard scrapers, PGCM-2026 performs "Full Raw Scraping," entering URLs to extract the complete article body, bypassing snippet limitations.
*   **Notion Integration:** Raw data is stored in a structured Notion database with metadata including source attribution, regional tags, and precise timestamps.

### 2. NLP & Risk Analysis Engine
The core logic resides in `analyzer_v2.py`, utilizing a hybrid analytical approach:
*   **Sentiment Analysis:** Uses **VADER** (specialized for news/social sentiment) and **TextBlob** to detect hawk/dove rhetoric.
*   **Strategic Weighting (v3.0):** Matches text against a 6-layer dictionary:
    *   🔴 **Military:** Troop movements, carrier strikes, nuclear readiness.
    *   ⚫ **Defiance:** Direct threats, treaty exits, "Red Line" rhetoric.
    *   ⚪ **Gray Zone:** Diplomatic aggression, sanctions, cyber ops.
    *   🟠 **Coercive:** Economic pressure, blockades.
    *   🟣 **Hybrid:** Asymmetric warfare, proxy activation.
    *   🟢 **Diplomatic:** De-escalation signals (negative weights).
*   **Temporal Correlation:** A unique "Short-term Memory" logic that identifies clusters of events. For example, a "Security Evacuation" following a "Military Build-up" within 48 hours triggers a **+40% risk multiplier**.

### 3. AI Strategic Outlook
At the end of each cycle, the system triggers **Google Gemini 3 Pro** to perform a high-level strategic review. It analyzes the top 10 most dangerous events and generates a 3-4 sentence "Command Summary" identifying the primary risk driver and the likely trajectory for the next 72 hours.

---

## 📊 Dashboard & Visualization
The analysis results are automatically injected into an interactive, mobile-responsive dashboard (`dashboard.html`):
*   **Risk Timeline:** A daily average of the conflict's "heat."
*   **Current Risk & Trend:** Real-time comparison of the latest 24 hours vs. the previous 3 days.
*   **Source Ledger:** A transparent list of all analyzed articles with their individual risk scores and extracted noun phrases.
*   **Auto-Deployment:** Every update is automatically committed and pushed to **GitHub Pages** for global access.

---

## 📂 Key Files
*   `analyzer_v2.py`: The main engine (Scraping, NLP, Analysis, GitHub Sync).
*   `VIT_COLLECTION_PROTOCOL.md`: The operational directive for the collector.
*   `strategic_dictionary.txt`: The custom weighted lexicon for geopolitical scoring.
*   `dashboard.html`: The interactive front-end.
*   `collection_db.json`: Local cache of raw event data.

---

## 🌐 Live Access
View the live monitor here: [https://geotouhme.github.io/Political_Analyzer/](https://geotouhme.github.io/Political_Analyzer/)

**System Status:** `ACTIVE`
**Last Analysis Sync:** Automated via OpenClaw Runtime.
