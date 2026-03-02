import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re

def load_and_fix_data():
    # Backtest DB
    conn_backtest = sqlite3.connect('political_analyzer/war_backtest_2025.sqlite')
    df_backtest = pd.read_sql_query("SELECT title, date FROM articles", conn_backtest)
    conn_backtest.close()
    
    # Main DB
    conn_main = sqlite3.connect('political_analyzer/notion_backup.sqlite')
    df_main = pd.read_sql_query("SELECT title, date FROM articles", conn_main)
    conn_main.close()

    df = pd.concat([df_backtest, df_main], ignore_index=True)
    
    # Fix the dates: handle both YYYYMMDD and YYYY-MM-DD
    def fix_date(d):
        if not d: return None
        d = str(d).replace('-', '')
        if len(d) >= 8:
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        return None

    df['date_fixed'] = df['date'].apply(fix_date)
    df['date_dt'] = pd.to_datetime(df['date_fixed'], errors='coerce')
    return df

def train_backtest_model():
    print("--- RANDOM FOREST BACKTEST ENGINE (FIXED DATES) ---")
    df = load_and_fix_data()
    df = df.dropna(subset=['title', 'date_dt'])
    
    # War window: Feb 18, 2026 to Mar 1, 2026
    df['target'] = df['date_dt'].apply(lambda x: 1 if (x >= pd.Timestamp('2026-02-18')) else 0)

    print(f"Total articles: {len(df)}")
    print(f"War-period articles (Class 1): {df['target'].sum()}")
    print(f"Lead-up articles (Class 0): {len(df) - df['target'].sum()}")

    if df['target'].sum() == 0:
        print("Error: No articles found in the war period. Check date filtering.")
        return

    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 3))
    X = vectorizer.fit_transform(df['title'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nSignals Identified (Top Features):")
    importances = rf.feature_importances_
    features = vectorizer.get_feature_names_out()
    for i in np.argsort(importances)[::-1][:15]:
        print(f"- {features[i]}: {importances[i]:.4f}")

if __name__ == "__main__":
    train_backtest_model()
