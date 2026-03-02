import os
import re
import sys
import json
import math
import logging
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable, Optional
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- Configuration -----------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("analyzer_v4")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DB_PATH = os.path.join(BASE_DIR, "notion_backup.sqlite")
MODEL_PATH = os.path.join(BASE_DIR, "transformer_v2/war_model")
DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard.html")
REPORT_PATH = os.path.join(BASE_DIR, "analysis_report_v2.json")

# --- AI Transformer Engine ---------------------------------------------------
def get_ai_scores(articles_df):
    if not os.path.exists(MODEL_PATH):
        log.warning("Transformer model not found at %s. Skipping AI scores.", MODEL_PATH)
        return articles_df

    log.info("Loading War-Transformer for deep analysis...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scores = []
    with torch.no_grad():
        for _, row in articles_df.iterrows():
            text = row['translated_title'] if row['translated_title'] else row['title']
            if not text:
                scores.append(0.0)
                continue
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            scores.append(round(probs[0][1].item() * 100, 1))
    
    articles_df['ai_risk_score'] = scores
    return articles_df

# --- Logic from V3 (Simplified for speed) ------------------------------------
MILITARY_TERMS = ["strike", "carrier", "missile", "buildup", "invasion", "nuclear", "war", "kinetic", "bomb"]
def simple_keyword_score(text):
    text = text.lower()
    score = 0
    for term in MILITARY_TERMS:
        if term in text: score += 15
    return min(score, 100.0)

# --- Main Logic --------------------------------------------------------------
def main():
    log.info("--- Political Pattern Analyzer v4.0 (Transformer + Keyword Hybrid) ---")
    
    if not os.path.exists(DB_PATH):
        log.error("Database not found at %s", DB_PATH)
        return

    conn = sqlite3.connect(DB_PATH)
    cols = [c[1] for c in conn.execute("PRAGMA table_info(articles)").fetchall()]
    title_col = "translated_title" if "translated_title" in cols else "title"
    
    query = f"SELECT notion_id, title, {title_col} as translated_title, date, url FROM articles ORDER BY date DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()

    log.info("Analyzing %d articles...", len(df))
    
    # Run Transformer Deep Analysis on the latest 100 articles
    latest_df = df.head(100).copy()
    latest_df = get_ai_scores(latest_df)
    
    # Top AI Insights
    ai_insights = latest_df.sort_values(by='ai_risk_score', ascending=False).head(15).to_dict(orient='records')

    # Update existing report
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            report = json.load(f)
    else:
        report = {"meta": {}, "trend": {}, "daily_breakdown": [], "articles": []}

    report["ai_deep_analysis"] = ai_insights
    report["meta"]["generated_at"] = datetime.now().isoformat()
    
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    # Update Dashboard HTML
    update_dashboard_html(report)
    log.info("Dashboard updated with AI Deep Analysis.")
    
    # Sync to GitHub
    push_to_github()

def update_dashboard_html(report):
    if not os.path.exists(DASHBOARD_PATH):
        log.warning("Dashboard file not found at %s", DASHBOARD_PATH)
        return

    with open(DASHBOARD_PATH, "r") as f: html = f.read()
    
    data_marker_start = "// __REPORT_DATA_START__"
    data_marker_end = "// __REPORT_DATA_END__"
    data_block = f"{data_marker_start}\n        const EMBEDDED_REPORT = {json.dumps(report, ensure_ascii=False)};\n        {data_marker_end}"
    html = re.sub(re.escape(data_marker_start) + r".*?" + re.escape(data_marker_end), data_block, html, flags=re.DOTALL)
    
    if 'id="ai-analysis-section"' not in html:
        ai_section_html = """
        <div id="ai-analysis-section" class="card" style="margin-top: 30px; border: 1px solid #ff5858;">
            <h3 style="color: #ff5858;">🤖 AI Deep Analysis (Transformer V4)</h3>
            <p style="font-size: 0.85rem; color: #8b949e;">Top signals detected by the custom-trained neural network (BERT-Multilingual).</p>
            <table>
                <thead>
                    <tr><th>Confidence</th><th>Article Intelligence</th><th>Date</th></tr>
                </thead>
                <tbody id="ai-table-body"></tbody>
            </table>
        </div>
        """
        html = html.replace('<div class="card">', ai_section_html + '\n        <div class="card">', 1)
    
    js_render_logic = """
            // Render AI Analysis
            const aiBody = document.getElementById('ai-table-body');
            if (aiBody && EMBEDDED_REPORT.ai_deep_analysis) {
                aiBody.innerHTML = EMBEDDED_REPORT.ai_deep_analysis.map(a => `
                    <tr>
                        <td><span class="badge" style="background: rgba(255,88,88,${a.ai_risk_score/100}); color: white;">${a.ai_risk_score}%</span></td>
                        <td><a href="${a.url}" target="_blank" style="color: var(--accent-color); text-decoration: none;">${a.translated_title || a.title}</a></td>
                        <td>${a.date}</td>
                    </tr>
                `).join('');
            }
    """
    if "Render AI Analysis" not in html:
        html = html.replace("// --- Render ---", "// --- Render ---\n" + js_render_logic)
        
    with open(DASHBOARD_PATH, "w") as f: f.write(html)

def push_to_github():
    import subprocess
    log.info("🚀 Syncing results to GitHub...")
    try:
        os.chdir(BASE_DIR)
        subprocess.run(["git", "add", "dashboard.html", "analysis_report_v2.json"], check=True)
        subprocess.run(["git", "commit", "-m", "update: v4 transformer analysis"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        log.info("✅ Pushed to GitHub.")
    except Exception as e:
        log.error("Failed to push to GitHub: %s", e)

if __name__ == "__main__":
    main()
