import requests
import sqlite3
import hashlib
import time
import feedparser
import urllib.parse
from datetime import datetime

# GDELT Doc API is 3-month limited for recent news. 
# For 2020, we use Google News RSS Archive search with specialized date parameters.
QUERIES = [
    "Iran US military escalation December 2019",
    "Soleimani Baghdad airport strike",
    "Qasem Soleimani killed Iran response",
    "US Iran war threats January 2020",
    "Baghdad US embassy attack December 2019"
]

DB_PATH = 'political_analyzer/soleimani_backtest.sqlite'

def get_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''CREATE TABLE IF NOT EXISTS articles 
                    (id TEXT PRIMARY KEY, title TEXT, date TEXT, url TEXT)''')
    conn.close()

def fetch_soleimani_data():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    total_added = 0
    
    print("--- FETCHING SOLEIMANI 2020 BACKTEST DATA ---")

    for q in QUERIES:
        print(f"Searching: {q}...")
        # Precise daterange in Google News RSS: after:2019-12-01 before:2020-01-20
        encoded_q = urllib.parse.quote(f'{q} after:2019-12-01 before:2020-01-20')
        url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title = entry.title
                link = entry.link
                
                # Parse date from RSS metadata
                p_date = entry.get('published_parsed')
                if p_date:
                    date_str = time.strftime('%Y-%m-%d', p_date)
                else:
                    date_str = "2020-01-03" # Fallback to event day
                
                try:
                    cursor.execute("INSERT OR IGNORE INTO articles (id, title, date, url) VALUES (?, ?, ?, ?)",
                                 (get_hash(link), title, date_str, link))
                    total_added += 1
                except: continue
        except Exception as e:
            print(f"  [ERROR] {e}")
        
        time.sleep(2) # Avoid rate limits

    conn.commit()
    conn.close()
    print(f"\n--- SUCCESS ---")
    print(f"Total articles added for 2020 Backtest: {total_added}")

if __name__ == "__main__":
    fetch_soleimani_data()
