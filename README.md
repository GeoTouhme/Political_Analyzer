# Persian Gulf Conflict Monitor (PGCM-2026)

An autonomous intelligence monitoring platform designed to track, analyze, and visualize the 2026 US-Iran geopolitical crisis.

## 🚀 Overview
PGCM-2026 is an end-to-end OSINT (Open Source Intelligence) pipeline. It functions as a dual-role agent system: a **Collector (VIT)** that harvests raw signals from the global information environment, and an **Analyzer** that applies strategic weighted scoring, temporal correlation, and AI-driven strategic forecasting.

---

## 🛠 Project Architecture

### 1. Data Collection (VIT Protocol)
The system follows a strict **Geopolitical Collection Protocol** (`VIT_COLLECTION_PROTOCOL.md`) to ensure high-signal ingestion:
*   **Deep Harvesting:** PGCM-2026 performs "Full Raw Scraping," entering URLs to extract the complete article body and injecting it into Notion Page Blocks to preserve context for deep NLP analysis.
*   **Targeted Scoping:** Monitors military movements (Armadas, S-500 transit), embassy security postures (Evacuations, Travel Bans), and international rhetoric (E3, UN, Russia/China/Afghanistan/Pakistan links).

### 2. NLP & Risk Analysis Engine (Logic v3.0)
The core engine (`analyzer_v2.py`) uses a multi-layered analytical approach:
*   **Weighted Risk Scoring:** Matches text against a 6-category strategic dictionary (Military, Defiance, Gray, Coercive, Hybrid, and Diplomatic).
*   **Temporal Correlation:** Identifies "Escalation Clusters." If a high-risk military event is followed by a security evacuation within a 48-hour window, the system triggers a **+40% risk multiplier**.
*   **Hybrid Sentiment:** Combines **VADER** and **TextBlob** to assess the hawkish or dovish nature of political rhetoric.

### 3. AI Strategic Outlook (Gemini 3 Pro)
At the end of each analysis cycle, the system utilizes **Google Gemini 3 Pro** to perform a high-level strategic review. It identifies the primary risk driver and forecasts the likely trajectory for the next 48-72 hours. This analysis is displayed prominently on the dashboard with full model attribution.

---

## 📊 Dashboard & UX Features
The results are visualized in a mobile-first, interactive dashboard:
*   **Smart Navigation:** Features a scrollable article ledger, a real-time search bar, and risk filters (All / High Risk / Latest).
*   **Risk Metrics:** Displays Current Risk (latest 24h average), Peak Risk, Trend Indicators (▲/▼), and a historical Risk Timeline.
*   **Auto-Deployment:** Every analysis cycle automatically pushes updates to **GitHub Pages** for real-time monitoring.

---

## 📂 Key Files
*   `analyzer_v2.py`: The integrated engine (Scraping, NLP, Analysis, AI Forecasting, GitHub Sync).
*   `VIT_COLLECTION_PROTOCOL.md`: Operational directive for the collection agent.
*   `strategic_dictionary.txt`: Custom weighted lexicon for geopolitical scoring.
*   `dashboard.html`: The interactive front-end (mapped to `index.html` for deployment).
*   `analysis_report_v2.json`: The raw data output for the dashboard.

---

## 🌐 Live Access
View the live monitor here: [https://geotouhme.github.io/Political_Analyzer/](https://geotouhme.github.io/Political_Analyzer/)

**System Status:** `ACTIVE`
**Model Attribution:** Strategic Analysis powered by `google/gemini-3-pro-preview`.
