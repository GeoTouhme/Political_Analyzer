import os
import requests
import sqlite3
import json
from datetime import datetime
from dotenv import load_dotenv

# Load keys from political_analyzer/.env
load_dotenv("political_analyzer/.env")

# Use X2_API_KEY as a backup (ndh_q7F4fVt8ryrHY-3NhcC98TJLVNsW2UtB_WOc6YxuM3M)
API_KEY = os.getenv("X2_API_KEY")
DB_PATH = "political_analyzer/notion_backup.sqlite"
BASE_URL = "https://newsdata.io/api/1/news"

def fetch_newsdata_backup():
    if not API_KEY:
        print("❌ Error: X2_API_KEY not found.")
        return []

    # Expanded query to include regional escalation vectors
    query = '("Persian Gulf" OR "Strait of Hormuz" OR "Iran military" OR "Hezbollah" OR "Houthi Red Sea" OR "Israel strike Iran" OR "Iran nuclear program")'
    
    params = {
        'apikey': API_KEY,
        'q': query,
        'language': 'en'
    }

    print(f"📡 Fetching from NewsData.io using X2_API_KEY...")
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if data.get("status") == "success":
            results = data.get("results", [])
            print(f"✅ Success: Found {len(results)} articles.")
            return results
        else:
            print(f"❌ API Error: {data.get('results', {}).get('message', 'Unknown error')}")
            return []

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        return []

def save_to_sqlite(articles):
    if not articles:
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    count = 0
    for art in articles:
        url = art.get("link")
        title = art.get("title")
        full_text = art.get("description") or art.get("content") or title
        date_str = art.get("pubDate", datetime.now().strftime("%Y-%m-%d"))[:10]
        
        # Unique ID based on URL
        notion_id = f"ndh_{hash(url)}"
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO articles (notion_id, title, date, full_text, url)
                VALUES (?, ?, ?, ?, ?)
            """, (notion_id, title, date_str, full_text, url))
            if cursor.rowcount > 0:
                count += 1
        except Exception:
            continue
            
    conn.commit()
    conn.close()
    print(f"💾 Injected {count} NEW articles from NewsData into {DB_PATH}")

if __name__ == "__main__":
    new_arts = fetch_newsdata_backup()
    if new_arts:
        save_to_sqlite(new_arts)
        print("🚀 NewsData injection complete.")
    else:
        print("📭 No new articles collected.")
