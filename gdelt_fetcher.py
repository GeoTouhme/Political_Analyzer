import requests
import json
import sqlite3
from datetime import datetime
import os

# Configuration
DB_PATH = "political_analyzer/notion_backup.sqlite"
# Logic: Look for Iran, Persian Gulf, Hormuz, or Military Strike in English news from the last 24 hours
GDELT_QUERY = '(Iran OR "Persian Gulf" OR "Strait of Hormuz" OR "military strike") sourcelang:eng'
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def fetch_gdelt_articles():
    print(f"📡 Querying GDELT DOC API for: {GDELT_QUERY}")
    
    params = {
        "query": GDELT_QUERY,
        "mode": "artlist",
        "format": "json",
        "maxresults": "50", # Top 50 recent articles
        "timespan": "24h"   # Last 24 hours only
    }
    
    try:
        response = requests.get(GDELT_URL, params=params, timeout=20)
        if response.status_code != 200:
            print(f"❌ GDELT Error: Status {response.status_code}")
            return []
        
        data = response.json()
        articles = data.get("articles", [])
        print(f"✅ Found {len(articles)} potential articles from GDELT.")
        return articles
    except Exception as e:
        print(f"❌ Error fetching from GDELT: {e}")
        return []

def save_to_sqlite(articles):
    if not articles:
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check table schema to match your analyzer_v3 expectations
    # Expected: notion_id, title, date, full_text, url
    
    count = 0
    for art in articles:
        url = art.get("url")
        title = art.get("title")
        date_str = art.get("seendate", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))[:10] # YYYY-MM-DD
        
        # GDELT API doesn't give full text directly, we use the title as placeholder for text 
        # or you can run your scraper on the URL later. For now, let's inject metadata.
        notion_id = f"gdelt_{hash(url)}"
        
        try:
            # Using INSERT OR IGNORE to prevent duplicates
            cursor.execute("""
                INSERT OR IGNORE INTO articles (notion_id, title, date, full_text, url)
                VALUES (?, ?, ?, ?, ?)
            """, (notion_id, title, date_str, title, url)) # Injected title as text to allow immediate scoring
            if cursor.rowcount > 0:
                count += 1
        except Exception as e:
            continue
            
    conn.commit()
    conn.close()
    print(f"💾 Injected {count} NEW articles into {DB_PATH}")

if __name__ == "__main__":
    new_articles = fetch_gdelt_articles()
    if new_articles:
        save_to_sqlite(new_articles)
        print("🚀 GDELT injection complete. Run your analyzer_v3.py now to update the dashboard.")
    else:
        print("📭 No new articles found in the last 24h.")
