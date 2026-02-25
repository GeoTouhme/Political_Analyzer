# PGCM-2026: Operational Progress & System Evolution Matrix

This document serves as the long-term memory for the development of the **Persian Gulf Conflict Monitor (PGCM-2026)**.

---

## 🏗 System Architecture & Technology Stack

### 1. Data Layer (The Collector)
- **Agent Role:** VIT (Geopolitical Monitoring Agent).
- **Protocol:** `VIT_COLLECTION_PROTOCOL.md` (Strict rules for US/Iran/Israel/Proxy signals).
- **Ingestion:** Full Raw Scraping via `agent-browser` (Scraping article body, not just snippets).
- **Storage:** Notion API (Structured database) + Notion Blocks (For storing full-text articles exceeding property limits).

### 2. Analytical Layer (The Engine)
- **Logic Version:** v3.0 (Temporal Correlation).
- **NLP Suite:** 
  - **VADER:** Primary sentiment engine for news-tone detection.
  - **TextBlob:** Noun phrase extraction and secondary sentiment validation.
- **Weighted Dictionary:** 6-category lexicon (Military, Defiance, Gray, Coercive, Hybrid, Diplomatic).
- **Dynamic Multipliers:** +40% risk multiplier for "Escalation Clusters" (e.g., Military deployment followed by Security Evacuation within 48h).

### 3. Intelligence Layer (The Strategist)
- **AI Model:** `google/gemini-3-pro-preview`.
- **Function:** Automated synthesis of the top 10 high-risk events into a 3-4 sentence "Strategic Outlook" every analysis cycle.

### 4. Presentation Layer (The Dashboard)
- **Frontend:** Mobile-responsive HTML5/Tailwind-style CSS.
- **Visuals:** Chart.js for Risk Timelines, Sentiment Scatter Plots, and Risk Distribution.
- **UX Features:** Internal scroll containers, real-time search filtering, and risk-level sorting.
- **Deployment:** Automated CI/CD pipeline pushing to **GitHub Pages**.

---

## 📈 Milestones & Evolution

### Phase 1: Foundation (Feb 20-22, 2026)
- Connected Notion & GitHub pipelines.
- Built `analyzer_v1.py` for basic keyword matching.
- Established the weighted dictionary v1.0.

### Phase 2: Deep Analysis (Feb 23-24, 2026)
- Upgraded to `analyzer_v2.py`.
- Integrated VADER NLP.
- Implemented **Deep Harvesting** (Full text scraping from URLs).
- Created the interactive HTML Dashboard.

### Phase 3: Intelligence Upgrade (Feb 25, 2026)
- **Logic v3.0 Deployment:** Implemented Temporal Correlation to detect patterns over time.
- **AI Integration:** Linked Gemini 3 Pro for Strategic Forecasting.
- **UX Refinement:** Solved "Infinite Scroll" issue with fixed containers and search bars.
- **Branding:** Transitioned from "Political Analyzer" to "PGCM-2026".

---

## 📊 Current System State (Last Sync: Feb 25, 07:49 AM)
- **Article Volume:** 62 verified strategic events.
- **Current Risk Level:** 28.7 (Trend: DOWN ▼ from previous peak).
- **Highest Threat Detected:** 74.0 (Tehran response to strikes).
- **Reliability:** 100% automated sync from Notion to Live Dashboard.

---

## 🛠 Lessons Learned & Engineering Principles
1. **Context is King:** Summaries miss small tactical details (S-500 models, specific ship names); always scrape the full body.
2. **JSON Safety:** Sanitize AI outputs meticulously to prevent JSON injection in web frontends (especially technical symbols like 🧿).
3. **Mobile First:** A conflict monitor is useless if it's unreadable on a phone during a crisis.
4. **Temporal Memory:** Geopolitics is not a series of isolated events; risk is cumulative.

---
**Status:** `READY FOR DEPLOYMENT / MONITORING MODE`
**Lead Engineer:** George
**Agent Interface:** VIT Assistant
