#!/bin/bash
cd /home/ubuntu/.openclaw/workspace/political_analyzer
echo "[$(date)] Pulse started: Collecting RSS..."
/usr/bin/python3 rss_collector.py
echo "[$(date)] Collecting Historical RSS (Feb 27-28 window)..."
/usr/bin/python3 rss_historical_collector.py
echo "[$(date)] Running AI-Only Deep Analyzer v5.1..."
/usr/bin/python3 analyzer_v5_ai_only.py
echo "[$(date)] Pulse complete. Dashboard updated."
