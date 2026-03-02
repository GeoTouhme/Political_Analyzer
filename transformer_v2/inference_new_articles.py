import sqlite3
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Config
DB_PATH = 'political_analyzer/notion_backup.sqlite'
MODEL_PATH = 'political_analyzer/transformer_v2/war_model'
MAX_LEN = 64

def run_inference():
    print("--- LOADING CUSTOM WAR-TRANSFORMER ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 1. Fetch latest articles (last 24 hours approximately)
    conn = sqlite3.connect(DB_PATH)
    # Using translated_title for best accuracy
    df = pd.read_sql_query("SELECT notion_id, title, translated_title, date, url FROM articles ORDER BY backed_up_at DESC LIMIT 500", conn)
    conn.close()

    print(f"Analyzing {len(df)} latest articles...")

    results = []
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            text = row['translated_title'] if row['translated_title'] else row['title']
            if not text: continue
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            war_score = probs[0][1].item() * 100 # Class 1 is "War State"
            
            results.append({
                'title': text,
                'date': row['date'],
                'score': war_score,
                'url': row['url']
            })

    # Sort by risk score
    results_df = pd.DataFrame(results).sort_values(by='score', ascending=False)
    
    print("\n" + "="*60)
    print("⚠️ HIGH RISK INTEL DETECTED (TOP 10) ⚠️")
    print("="*60)
    for i, res in results_df.head(10).iterrows():
        print(f"[{res['score']:.1f}%] - {res['title'][:80]}...")
    
    # Save to a temporary analysis file for George
    results_df.to_csv('political_analyzer/transformer_v2/latest_ai_analysis.csv', index=False)
    print("\nFull analysis saved to latest_ai_analysis.csv")

if __name__ == "__main__":
    run_inference()
