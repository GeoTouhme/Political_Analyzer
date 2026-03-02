import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# 1. Load the Model Data (2026 War vs Lead-up)
def train_2026_model():
    conn_backtest = sqlite3.connect('political_analyzer/war_backtest_2025.sqlite')
    df_backtest = pd.read_sql_query("SELECT translated_title as title, date FROM articles", conn_backtest)
    conn_backtest.close()
    
    conn_main = sqlite3.connect('political_analyzer/notion_backup.sqlite')
    df_main = pd.read_sql_query("SELECT translated_title as title, date FROM articles", conn_main)
    conn_main.close()

    df_train = pd.concat([df_backtest, df_main], ignore_index=True).dropna()
    
    # Labeling: 1 = War state (Feb 18, 2026 onwards)
    df_train['date'] = pd.to_datetime(df_train['date'].str.replace('-', ''), format='%Y%m%d', errors='coerce')
    df_train['target'] = df_train['date'].apply(lambda x: 1 if x >= pd.Timestamp('2026-02-18') else 0)
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df_train['title'])
    y = df_train['target']
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf, vectorizer

# 2. Test on 2020 Data
def run_soleimani_blind_test(rf, vectorizer):
    conn_2020 = sqlite3.connect('political_analyzer/soleimani_backtest.sqlite')
    df_2020 = pd.read_sql_query("SELECT title, date FROM articles", conn_2020)
    conn_2020.close()
    
    print(f"Loaded {len(df_2020)} articles from 2020 period.")
    
    # Feature extraction using the 2026 vocabulary
    X_2020 = vectorizer.transform(df_2020['title'])
    
    # Predict Probability of "War State"
    probs = rf.predict_proba(X_2020)[:, 1]
    df_2020['war_risk_score'] = probs * 100
    
    # Group by date to see the trend
    trend = df_2020.groupby('date')['war_risk_score'].mean().reset_index()
    print("\n--- SOLEIMANI 2020 BLIND TEST RESULTS ---")
    print(trend.sort_values(by='date'))
    
    # Check if Jan 3rd had a spike
    jan3_risk = trend[trend['date'] == '2020-01-03']['war_risk_score'].values
    if len(jan3_risk) > 0:
        print(f"\n[ALERT] Risk Score on Jan 3, 2020 (Assassination Day): {jan3_risk[0]:.2f}%")

if __name__ == "__main__":
    print("Training 2026 Logic...")
    rf, vec = train_2026_model()
    print("Running 2020 Test...")
    run_soleimani_blind_test(rf, vec)
