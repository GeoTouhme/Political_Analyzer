# Roadmap: Collector V2 Upgrade & Key Rotation

## 0. Database Migration: Notion to Local SQLite
- **Why:** Faster processing, zero API latency, no rate limits, and full data ownership.
- **Action:** Design a SQLite schema to replace the Notion database. 
- **Fields:** `id`, `title`, `date`, `source`, `url`, `event_type`, `region`, `actors`, `full_text`, `ai_risk_score`, `math_pulse_score`, `historical_pattern`.
- **Sync Logic:** The collector will now write directly to `political_analysis.sqlite`. Notion may be used optionally as a read-only mirror or deprecated entirely.

## 1. Multi-Key Architecture (Key Rotation)
- Modify `news_collector.py` to support `X_API_KEYS` and `NEWS_API_KEYS` as comma-separated lists in `.env`.
- Implement a `KeyManager` class or logic to cycle through keys when `429 (Rate Limit)` or `403 (Quota Exceeded)` is encountered.
- Add logging to track which key is active and which ones are exhausted.

## 2. Failover & Cooldown Logic
- If all keys in a pool are exhausted, implement a smart wait (e.g., 1 hour) before retrying or exiting gracefully.
- Prevent infinite loops if the network is down.

## 3. Gemini Strategic Integration (The "Intuition" Layer)
- Update the collection pipeline to send new articles to Gemini 3 Pro.
- **Task for Gemini:**
    - Generate a **Strategic Risk Score** (1-100).
    - Identify **Historical Patterns** (e.g., "Similar to 2019 tanker attacks").
    - Store these values in Notion alongside the article metadata.

## 4. Dashboard Expansion
- Update `dashboard.html` / `spectrum_dashboard.html` to display the dual-track analysis:
    - **Math Score:** Based on volume and keyword density (Mathematical Pulse).
    - **AI Score:** Based on Gemini's strategic assessment (Strategic Intuition).

## 5. Execution Steps (Target: Tomorrow)
1. Update `.env` with multiple keys.
2. Refactor `news_collector.py` for Key Rotation.
3. Integrate Gemini API call for article enrichment.
4. Test run and verify Notion updates.
