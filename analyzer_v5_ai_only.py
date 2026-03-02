import os
import sys
import json
import sqlite3
import logging
import torch
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("analyzer_v5_ai")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "notion_backup.sqlite")
MODEL_PATH = os.path.join(BASE_DIR, "transformer_v2/war_model")
REPORT_PATH = os.path.join(BASE_DIR, "analysis_report_v2.json")

def classify_risk(score):
    if score > 85: return "CRITICAL"
    if score > 65: return "HIGH"
    if score > 35: return "MEDIUM"
    return "LOW"

def main():
    log.info("--- Political Pattern Analyzer v5.1 (AI-ONLY RESTORED) ---")
    
    if not os.path.exists(MODEL_PATH):
        log.error("Model not found at %s. Please ensure training completed.", MODEL_PATH)
        return

    # 1. Load AI Engine
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2. Fetch Data from SQLite
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT notion_id, title, date, url FROM articles", conn)
    conn.close()

    log.info("Analyzing %d articles...", len(df))

    # 3. AI Inference
    ai_scores = []
    with torch.no_grad():
        for i, text in enumerate(df['title']):
            inputs = tokenizer(str(text), return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            score = round(probs[0][1].item() * 100, 1)
            ai_scores.append(score)

    df['risk_score'] = ai_scores
    df['risk_level'] = df['risk_score'].apply(classify_risk)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # 4. Aggregation
    daily_results = []
    unique_dates = sorted(df['date'].unique())
    prev_final = None

    for d_str in unique_dates:
        day_df = df[df['date'] == d_str]
        max_score = day_df['risk_score'].max()
        top_avg = day_df.nlargest(3, 'risk_score')['risk_score'].mean()
        anchored_risk = round((max_score * 0.6) + (top_avg * 0.4), 1)
        delta = round(anchored_risk - prev_final, 1) if prev_final is not None else 0.0
        prev_final = anchored_risk

        daily_results.append({
            "date": d_str, "anchored_risk": anchored_risk, "final_risk": anchored_risk,
            "risk_level": classify_risk(anchored_risk), "article_count": len(day_df),
            "max_risk": max_score, "day_over_day_delta": delta,
            "dominant_articles": day_df.nlargest(3, 'risk_score')['title'].tolist()
        })

    # 5. Build Final Report
    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "article_count": len(df),
            "day_count": len(daily_results),
            "model_id": "Custom-Transformer-V4-BERT"
        },
        "daily_breakdown": daily_results,
        "ai_deep_analysis": df.nlargest(20, 'risk_score')[['title', 'url', 'risk_score', 'date']].rename(columns={'risk_score': 'ai_risk_score'}).to_dict(orient='records'),
        "articles": df.sort_values(by='date', ascending=False)[['title', 'url', 'risk_score', 'date', 'risk_level']].to_dict(orient='records')
    }

    with open(REPORT_PATH, "w") as f: json.dump(report, f, indent=4, ensure_ascii=False)
    log.info("✅ SUCCESS: AI-Only Analysis completed and saved.")

if __name__ == "__main__":
    main()
